"""ACE Stage 2 inference: retarget v3 benchmark queries with autoregressive single-shot G.

Per ACE_DESIGN_V3 §3.5:
  1. Encode source through Stage A
  2. Window into 8-token chunks with stride=4 (50% overlap)
  3. Per chunk: z_pred = G(z_src, prev_z_tgt, src/tgt skel/graph)
     - chunk_0 uses target-conditioned START as prev_z_tgt
     - chunk_n≥1 uses chunk_(n-1)'s output as prev_z_tgt
  4. NO Heun integration (G is deterministic), NO CFG ensemble (ACE has no CFG)
  5. Latent overlap-add → decode continuous (paper-faithful, no quantize) → unnormalize → save
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from model.ace.generator import ACEGenerator, ACEStartTokens
from model.moreflow.skel_graph import SkelGraphEncoder, build_skel_features, pad_to_max_joints
from model.moreflow.stage_a_registry import StageARegistry

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
COND_PATH = DATA_ROOT / 'cond.npy'
MOTION_DIR = DATA_ROOT / 'motions'
WINDOW = 32
STRIDE = 4
TOKENS_PER_WINDOW = 8


@torch.no_grad()
def retarget_one_query(query, G, skel_enc, starts, registry, skel_features, max_J,
                        skel_to_id, device):
    src_skel = query['skel_a']
    tgt_skel = query['skel_b']
    src_fname = query['src_fname']

    UNK_ID = len(skel_to_id)
    src_id = skel_to_id.get(src_skel, UNK_ID)
    tgt_id = skel_to_id.get(tgt_skel, UNK_ID)
    if src_id == UNK_ID or tgt_id == UNK_ID:
        # Both src and tgt held-out → fall back to skel id 0 with graph features only
        if src_id == UNK_ID: src_id = 0
        if tgt_id == UNK_ID: tgt_id = 0

    # Skel graph features
    src_pj_pad, src_pj_mask = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)
    tgt_pj_pad, tgt_pj_mask = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)
    src_pj_pad = src_pj_pad.unsqueeze(0).to(device)
    src_pj_mask = src_pj_mask.unsqueeze(0).to(device)
    src_a = skel_features[src_skel]['agg'].unsqueeze(0).to(device)
    tgt_pj_pad = tgt_pj_pad.unsqueeze(0).to(device)
    tgt_pj_mask = tgt_pj_mask.unsqueeze(0).to(device)
    tgt_a = skel_features[tgt_skel]['agg'].unsqueeze(0).to(device)
    src_graph = skel_enc(src_pj_pad, src_pj_mask, src_a)
    tgt_graph = skel_enc(tgt_pj_pad, tgt_pj_mask, tgt_a)

    # Load + encode source motion
    motion_phys = np.load(MOTION_DIR / src_fname).astype(np.float32)
    T = motion_phys.shape[0]
    T_crop = (T // STRIDE) * STRIDE
    motion_phys = motion_phys[:T_crop]
    motion_t = torch.from_numpy(motion_phys).to(device)

    motion_norm = registry.normalize(src_skel, motion_t.unsqueeze(0))
    z_src_full, _ = registry.encode_window(src_skel, motion_norm.squeeze(0))
    z_src_full = z_src_full.squeeze(0)                                    # [T_token_full, codebook_dim]
    T_token_full = z_src_full.shape[0]

    # Chunk into 8-token windows with stride=4 (50% overlap)
    chunk_token_stride = TOKENS_PER_WINDOW // 2
    if T_token_full <= TOKENS_PER_WINDOW:
        chunk_starts = [0]
    else:
        chunk_starts = list(range(0, T_token_full - TOKENS_PER_WINDOW + 1, chunk_token_stride))
        last_start = T_token_full - TOKENS_PER_WINDOW
        if chunk_starts[-1] != last_start:
            chunk_starts.append(last_start)

    codebook_dim = registry.get(tgt_skel)['model'].codebook_dim
    z_tgt_acc = torch.zeros(T_token_full, codebook_dim, device=device)
    z_tgt_count = torch.zeros(T_token_full, device=device)

    src_id_t = torch.tensor([src_id], dtype=torch.long, device=device)
    tgt_id_t = torch.tensor([tgt_id], dtype=torch.long, device=device)

    # Initial prev_z_tgt = START token for chunk 0
    prev_z_pred = starts(tgt_id_t)                                        # [1, 8, codebook_dim]

    for n, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + TOKENS_PER_WINDOW, T_token_full)
        chunk_z = z_src_full[chunk_start:chunk_end]
        if chunk_end - chunk_start < TOKENS_PER_WINDOW:
            pad = TOKENS_PER_WINDOW - chunk_z.shape[0]
            chunk_z = torch.cat([chunk_z, torch.zeros(pad, chunk_z.shape[1], device=device)])
        chunk_z = chunk_z.unsqueeze(0)                                     # [1, 8, codebook_dim]

        z_pred = G(chunk_z, prev_z_pred, src_id_t, tgt_id_t, src_graph, tgt_graph)
        # Update prev for next chunk = current's predicted output. Detach for safety —
        # if this function is ever called from a training context (not no_grad), would
        # otherwise chain graphs across chunks → memory blowup.
        prev_z_pred = z_pred.detach()                                      # [1, 8, codebook_dim]

        chunk_len_actual = chunk_end - chunk_start
        z_tgt_acc[chunk_start:chunk_end] += z_pred.squeeze(0)[:chunk_len_actual]
        z_tgt_count[chunk_start:chunk_end] += 1.0

    if (z_tgt_count == 0).any():
        n_missed = int((z_tgt_count == 0).sum())
        print(f"  WARN: {n_missed}/{T_token_full} tokens missed by overlap-add")
    z_tgt_avg = z_tgt_acc / z_tgt_count.clamp(min=1.0).unsqueeze(-1)

    # Paper-faithful: continuous z directly through Stage A decoder (no quantization).
    # ACE paper has continuous embedding z throughout; π(z, x_{t-1}) decodes z directly.
    # Stage A decoder is robust to continuous (non-codebook) z (verified on src motion path).
    decoded = registry.decode_tokens(tgt_skel, z_tgt_avg.unsqueeze(0))
    decoded_phys = registry.unnormalize(tgt_skel, decoded).squeeze(0)
    return decoded_phys.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--bench_version', choices=['v3', 'v5'], default='v3')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Custom manifest path (overrides bench_version+fold lookup)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = ckpt['skels']
    skel_to_id = ckpt['skel_to_id']
    print(f"Trained on {len(skels)} skels")

    G = ACEGenerator(n_skels=len(skels)).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(ckpt['skel_enc'])
    skel_enc.eval()

    # Build registry FIRST so we have z_means for START tokens
    print("Loading Stage A registry for ALL 70 skels...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)

    # Reconstruct z_means from cache for START tokens (train scope)
    # Easier: just load ckpt's saved start state and starts module's z_means buffer
    # (it was saved as state_dict)
    # Compute z_means from cache_all.pt
    cache_path = PROJECT_ROOT / 'save/moreflow_flow/cache_all.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    z_means = torch.zeros(len(skels), 8, 256)
    for s in skels:
        z_means[skel_to_id[s]] = cache[s]['z_continuous'].mean(dim=0)
    starts = ACEStartTokens(n_skels=len(skels), n_train_skels=len(skels),
                             codebook_dim=256, n_tokens=8, z_means=z_means).to(device)
    starts.load_state_dict(ckpt['starts'])
    starts.eval()

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        queries_dir = 'queries_v5' if args.bench_version == 'v5' else 'queries'
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/{queries_dir}/fold_{args.fold}/manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries']
    if args.limit:
        queries = queries[:args.limit]
    print(f"Processing {len(queries)} queries from fold {args.fold}...")

    t0 = time.time()
    n_done = n_skipped = 0
    for i, q in enumerate(queries):
        out_path = out_dir / f"query_{q['query_id']:04d}.npy"
        if out_path.exists():
            n_done += 1
            continue
        try:
            result = retarget_one_query(q, G, skel_enc, starts, registry, skel_features,
                                         max_J, skel_to_id, device)
            np.save(out_path, result.astype(np.float32))
            n_done += 1
        except Exception as e:
            print(f"  query {q['query_id']} ({q['skel_a']}→{q['skel_b']}): FAILED — {e}")
            import traceback; traceback.print_exc()
            n_skipped += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] done={n_done} skipped={n_skipped} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")

    print(f"\nFinished: {n_done}/{len(queries)} written, {n_skipped} skipped.")


if __name__ == '__main__':
    main()
