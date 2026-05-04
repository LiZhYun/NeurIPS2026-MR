"""VQ-ActionBridge v2 inference: source-conditioned generation.

Pipeline:
  1. Encode src motion → src VQ z (via Stage A encoder), then Encoder E → behavior_tokens
  2. Sample target z conditioned on (behavior_tokens, skel_b features) via Euler ODE
  3. Decode via frozen MoReFlow Stage A per-skel VQ-VAE
  4. Save physical motion .npy

Differences from v1 inference:
  - Uses Encoder E to extract behavior_tokens from source z (no separate classifier)
  - drop_skel=True for held-out skels (uses skel_graph features only)

Usage:
  python -m sample.actionbridge_v2_retarget --ckpt save/actionbridge/actionbridge_v2/ckpt_final.pt --fold 42
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
from model.actionbridge.encoder import ActionBridgeEncoder
from model.actionbridge.generator_v2 import ActionBridgeGeneratorV2
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from model.moreflow.stage_a_registry import StageARegistry

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
COND_PATH = DATA_ROOT / 'cond.npy'
MOTION_DIR = DATA_ROOT / 'motions'
WINDOW = 32  # MoReFlow Stage A window


@torch.no_grad()
def sample_z(G, behavior_tokens, tgt_skel_id, tgt_graph, device, drop_tgt_skel=False,
             n_steps=50, cfg_w=2.0):
    """Euler ODE sampling: noise → target z [B, 8, 256]."""
    B = behavior_tokens.shape[0]
    x = torch.randn(B, 8, 256, device=device)
    null_bt = torch.zeros_like(behavior_tokens)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v_cond = G(x, t, tgt_skel_id, tgt_graph, behavior_tokens, drop_skel=drop_tgt_skel)
        if cfg_w > 0:
            v_uncond = G(x, t, tgt_skel_id, tgt_graph, null_bt,
                         drop_skel=drop_tgt_skel, drop_behavior=True)
            v = (1 + cfg_w) * v_cond - cfg_w * v_uncond
        else:
            v = v_cond
        x = x + v * dt
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--cfg_w', type=float, default=2.0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--manifest_root', default='eval/benchmark_v3/queries',
                        help='queries (v3) or queries_v5 (v5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading ckpt: {args.ckpt}")
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = sd['skels']
    skel_to_id = sd['skel_to_id']
    n_clusters = sd.get('n_clusters', 11)

    # Load training args
    train_args_path = Path(args.ckpt).parent / 'args.json'
    train_args = json.load(open(train_args_path)) if train_args_path.exists() else {}

    # Build models
    E = ActionBridgeEncoder(n_skels=len(skels), n_clusters=n_clusters,
                            d_model=train_args.get('d_model_enc', 384),
                            n_layers=train_args.get('n_layers_enc', 4)).to(device)
    G = ActionBridgeGeneratorV2(n_skels=len(skels),
                                d_model=train_args.get('d_model_gen', 384),
                                n_layers=train_args.get('n_layers_gen', 6)).to(device)
    skel_enc_g = SkelGraphEncoder().to(device)
    E.load_state_dict(sd['E'])
    G.load_state_dict(sd['G'])
    skel_enc_g.load_state_dict(sd['skel_enc_g'])
    E.eval(); G.eval(); skel_enc_g.eval()

    # Stage A registry for decoding (need ALL 70 for inference) — fp32 for numeric compat
    print("Loading Stage A registry for ALL 70 skels (fp32)...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)

    # Skel graph features (need for ALL skels at inference)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in all_skels)
    pj_padded, pj_mask, pj_agg = {}, {}, {}
    for s in all_skels:
        p, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = p.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Manifest
    manifest = json.load(open(PROJECT_ROOT / args.manifest_root / f'fold_{args.fold}/manifest.json'))
    queries = manifest['queries']
    if args.limit: queries = queries[:args.limit]

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / f'save/actionbridge_inference/v2/fold_{args.fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(queries)} queries from fold {args.fold}...")
    t0 = time.time()
    n_done = n_skipped = 0
    n_unk_src = n_unk_tgt = 0
    for i, q in enumerate(queries):
        qid = q['query_id']
        out_path = out_dir / f'query_{qid:04d}.npy'
        if out_path.exists():
            n_done += 1; continue
        try:
            skel_a = q['skel_a']; skel_b = q['skel_b']
            src_fname = q['src_fname']

            # Encode src motion → z via Stage A
            motion_phys = np.load(MOTION_DIR / src_fname).astype(np.float32)
            T = motion_phys.shape[0]
            T_crop = (T // 4) * 4
            motion_phys = motion_phys[:T_crop]
            motion_t = torch.from_numpy(motion_phys).to(device)
            motion_norm = registry.normalize(skel_a, motion_t.unsqueeze(0))
            z_src_full, _ = registry.encode_window(skel_a, motion_norm.squeeze(0))
            z_src_full = z_src_full.squeeze(0)  # [T_token, 256]

            # Take first 8 tokens (matching Stage B window)
            if z_src_full.shape[0] >= 8:
                z_src = z_src_full[:8].unsqueeze(0)  # [1, 8, 256]
            else:
                pad = torch.zeros(8 - z_src_full.shape[0], 256, device=device)
                z_src = torch.cat([z_src_full, pad], dim=0).unsqueeze(0)

            # Source skel/graph
            drop_src = (skel_a not in skel_to_id)
            sid_t = torch.tensor([skel_to_id.get(skel_a, 0)], dtype=torch.long, device=device)
            sg = skel_enc_g(pj_padded[skel_a].unsqueeze(0), pj_mask[skel_a].unsqueeze(0),
                            pj_agg[skel_a].unsqueeze(0))
            if drop_src: n_unk_src += 1

            # Encode → behavior tokens
            behavior_tokens, _ = E(z_src, sid_t, sg, drop_skel=drop_src)

            # Target skel/graph
            drop_tgt = (skel_b not in skel_to_id)
            tid_t = torch.tensor([skel_to_id.get(skel_b, 0)], dtype=torch.long, device=device)
            tg = skel_enc_g(pj_padded[skel_b].unsqueeze(0), pj_mask[skel_b].unsqueeze(0),
                            pj_agg[skel_b].unsqueeze(0))
            if drop_tgt: n_unk_tgt += 1

            # Sample target z
            z_pred = sample_z(G, behavior_tokens, tid_t, tg, device,
                              drop_tgt_skel=drop_tgt, n_steps=args.n_steps, cfg_w=args.cfg_w)

            # Decode via per-skel VQ
            x_norm = registry.decode_tokens(skel_b, z_pred)
            x_phys = registry.unnormalize(skel_b, x_norm).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(out_path, x_phys)
            n_done += 1
        except Exception as e:
            print(f"  q{qid}: FAIL {e}")
            n_skipped += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] done={n_done} skipped={n_skipped} "
                  f"unk_src={n_unk_src} unk_tgt={n_unk_tgt} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")
    print(f"\nFinished: {n_done} written, {n_skipped} skipped, "
          f"unk_src={n_unk_src} unk_tgt={n_unk_tgt}")


if __name__ == '__main__':
    main()
