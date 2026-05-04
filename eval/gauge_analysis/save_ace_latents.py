"""Extract z_tgt_avg latents from a trained ACE checkpoint, for gauge alignment study.

Mirrors sample/ace_retarget.py except we save the per-query latent z_tgt_avg
(shape [T_token_full, codebook_dim]) instead of the decoded motion.

Usage:
    python /tmp/save_ace_latents.py --ckpt save/ace/ace_ablation_no_ladv/ckpt_final.pt \
        --manifest eval/benchmark_v3/queries_sif_intersection/manifest.json \
        --out_dir /tmp/ace_latents/no_ladv_seed42
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from model.ace.generator import ACEGenerator, ACEStartTokens
from model.moreflow.skel_graph import SkelGraphEncoder, build_skel_features, pad_to_max_joints
from model.moreflow.stage_a_registry import StageARegistry

MOTION_DIR = Path(DATASET_DIR) / 'motions'
TOKENS_PER_WINDOW = 8
STRIDE = 4 * 8


def extract_latent_one_query(query, G, skel_enc, starts, registry, skel_features, max_J,
                              skel_to_id, device):
    """Returns z_tgt_avg [T_token_full, codebook_dim] (numpy)."""
    src_skel = query['skel_a']
    tgt_skel = query['skel_b']
    src_fname = query['src_fname']

    UNK_ID = len(skel_to_id)
    src_id = skel_to_id.get(src_skel, UNK_ID)
    tgt_id = skel_to_id.get(tgt_skel, UNK_ID)
    if src_id == UNK_ID or tgt_id == UNK_ID:
        if src_id == UNK_ID: src_id = 0
        if tgt_id == UNK_ID: tgt_id = 0

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

    motion_phys = np.load(MOTION_DIR / src_fname).astype(np.float32)
    T = motion_phys.shape[0]
    T_crop = (T // STRIDE) * STRIDE
    motion_phys = motion_phys[:T_crop]
    motion_t = torch.from_numpy(motion_phys).to(device)

    motion_norm = registry.normalize(src_skel, motion_t.unsqueeze(0))
    z_src_full, _ = registry.encode_window(src_skel, motion_norm.squeeze(0))
    z_src_full = z_src_full.squeeze(0)
    T_token_full = z_src_full.shape[0]

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

    prev_z_pred = starts(tgt_id_t)

    for n, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + TOKENS_PER_WINDOW, T_token_full)
        chunk_z = z_src_full[chunk_start:chunk_end]
        if chunk_end - chunk_start < TOKENS_PER_WINDOW:
            pad = TOKENS_PER_WINDOW - chunk_z.shape[0]
            chunk_z = torch.cat([chunk_z, torch.zeros(pad, chunk_z.shape[1], device=device)])
        chunk_z = chunk_z.unsqueeze(0)

        z_pred = G(chunk_z, prev_z_pred, src_id_t, tgt_id_t, src_graph, tgt_graph)
        prev_z_pred = z_pred.detach()

        chunk_len_actual = chunk_end - chunk_start
        z_tgt_acc[chunk_start:chunk_end] += z_pred.squeeze(0)[:chunk_len_actual]
        z_tgt_count[chunk_start:chunk_end] += 1.0

    z_tgt_avg = z_tgt_acc / z_tgt_count.clamp(min=1.0).unsqueeze(-1)
    return z_tgt_avg.cpu().numpy(), tgt_skel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[latents] Device: {device}")
    print(f"[latents] Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = ckpt['skels']
    skel_to_id = ckpt['skel_to_id']
    print(f"[latents] Trained on {len(skels)} skels")

    G = ACEGenerator(n_skels=len(skels)).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(ckpt['skel_enc'])
    skel_enc.eval()

    print("[latents] Loading Stage A registry for ALL 70 skels...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)

    cache_path = PROJECT_ROOT / 'save/moreflow_flow/cache_all.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    z_means = torch.zeros(len(skels), 8, 256)
    for s in skels:
        if s in cache:
            z_means[skel_to_id[s]] = cache[s]['z_continuous'].mean(dim=0)
    starts = ACEStartTokens(n_skels=len(skels), n_train_skels=len(skels),
                             codebook_dim=256, n_tokens=8, z_means=z_means).to(device)
    starts.load_state_dict(ckpt['starts'])
    starts.eval()

    cond_dict = np.load(Path(DATASET_DIR) / 'cond.npy', allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)

    print(f"[latents] Loading manifest: {args.manifest}")
    manifest = json.loads(Path(args.manifest).read_text())
    queries = manifest.get('queries', manifest.get('triples', []))
    if args.limit:
        queries = queries[:args.limit]
    print(f"[latents] Processing {len(queries)} queries...")

    triple_meta = []
    t0 = time.time()
    with torch.no_grad():
        for i, q in enumerate(queries):
            try:
                z_lat, tgt_skel = extract_latent_one_query(
                    q, G, skel_enc, starts, registry, skel_features, max_J, skel_to_id, device)
                np.save(out_dir / f'z_query_{i:04d}.npy', z_lat)
                triple_meta.append({
                    'query_id': i,
                    'src_skel': q.get('skel_a'), 'tgt_skel': tgt_skel,
                    'src_fname': q.get('src_fname'), 'T_tokens': z_lat.shape[0],
                    'codebook_dim': z_lat.shape[1],
                })
            except Exception as e:
                print(f"  [{i+1}/{len(queries)}] FAILED ({q.get('skel_a','?')}->{q.get('skel_b','?')}): {e}")
            if (i+1) % 50 == 0:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(queries)}] done ({dt:.0f}s)")

    (out_dir / 'meta.json').write_text(json.dumps(triple_meta, indent=2))
    print(f"[latents] saved {len(triple_meta)} queries to {out_dir} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
