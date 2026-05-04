"""DPG-SB-v2 inference on V5.

Pipeline per query (src_motion, tgt_skel, action):
  1. Look up source z from cache_all (pre-computed by MoReFlow Stage A)
  2. Normalize src z by src_skel's mean/std
  3. Pick retrieval-init z_b from same-action clip on tgt_skel (oracle-style)
  4. Normalize init z by tgt_skel's mean/std
  5. Add noise to init: z_start = z_init + noise
  6. SB integration: integrate flow from z_start to predicted z_b via Euler steps
  7. Denormalize: z_b * std(tgt_skel) + mean(tgt_skel)
  8. Decode via MoReFlow Stage A: z_b → normalized motion → unnormalize → physical motion [T, J, 13]
  9. Save query_NNNN.npy

Usage:
  python -m eval.baselines.run_dpg_sb_v2 --folds 42 43 \
      --ckpt save/dpg_sb/dpg_sb_v2_full/final.pt --out_tag dpg_sb_v2 --n_steps 20
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
from eval.benchmark_v3.action_taxonomy import (
    parse_action_from_filename, action_to_cluster, ACTION_CLUSTERS,
)
from model.dpg_sb.dpg_sb_model import BridgeGenerator
from model.moreflow.stage_a_registry import StageARegistry

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
CACHE_ALL = PROJECT_ROOT / 'save/moreflow_flow/cache_all.pt'
SAVE_ROOT = PROJECT_ROOT / 'save/dpg_sb'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())


def load_model(ckpt_path: str, device):
    ckpt_dir = Path(ckpt_path).parent
    args_dict = json.load(open(ckpt_dir / 'args.json'))
    G = BridgeGenerator(
        codebook_dim=256, n_tokens=8,
        d_model=args_dict['d_model'],
        n_layers=args_dict['n_layers'],
        n_heads=args_dict['n_heads'],
        n_skels=args_dict['n_skels'],
        n_exact_actions=args_dict['n_exact_actions'],
        src_layers=args_dict['src_layers'],
        dropout=args_dict['dropout'],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(state['G'])
    G.eval()
    return G, args_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_tag', type=str, default='dpg_sb_v2')
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--noise_scale', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--manifest', type=str, default=None,
                        help='Custom manifest path (overrides V5 fold-based lookup)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading model: {args.ckpt}")
    G, args_dict = load_model(args.ckpt, device)
    skels_train = args_dict['skels']
    skel_to_id = {s: i for i, s in enumerate(skels_train)}
    exact_actions = args_dict['exact_actions']
    exact_to_idx = {a: i for i, a in enumerate(exact_actions)}
    print(f"Trained skels: {len(skels_train)}, exact actions: {len(exact_actions)}")

    print(f"Loading cache_all...")
    cache = torch.load(CACHE_ALL, map_location='cpu', weights_only=False)
    all_skels = sorted(s for s in cache.keys() if not s.startswith('_'))
    print(f"  {len(all_skels)} skels in cache_all")

    # Build per-skel z normalization (fresh from cache_all so test skels have stats)
    z_stats = {}
    z_per_skel = {}
    for s in all_skels:
        z_raw = cache[s]['z_continuous'].float()
        mu = z_raw.mean(dim=0, keepdim=True)
        sigma = z_raw.std(dim=0, keepdim=True).clamp_min(1e-3)
        z_per_skel[s] = ((z_raw - mu) / sigma).to(device)
        z_stats[s] = (mu.to(device), sigma.to(device))

    # Build (skel, fname) → row_idx and (skel, action) → [(fname, ri)]
    fname_to_ri = {}  # (skel, fname) → ri
    skel_action_clips = defaultdict(list)  # (skel, action) → [(fname, ri)]
    for s in all_skels:
        meta = cache[s]['meta']
        for ri, (fname, _) in enumerate(meta):
            fname_to_ri[(s, fname)] = ri
            action = parse_action_from_filename(fname)
            if action_to_cluster(action) is not None:
                skel_action_clips[(s, action)].append((fname, ri))

    # Stage A registry for decoding (need test skels too)
    print(f"Loading StageARegistry for all 70 skels (one-time)...")
    test_skels = ['Anaconda', 'Buzzard', 'Cat', 'Crab', 'Elephant', 'Raptor', 'Rat',
                   'Spider', 'Trex', 'Alligator']
    decode_skels = sorted(set(skels_train + test_skels))
    reg = StageARegistry(decode_skels, device=str(device))
    n_skels = reg.n_skels if isinstance(reg.n_skels, int) else reg.n_skels()
    print(f"  Loaded {n_skels} per-skel decoders")

    # For source skels not in trained set (test skels): use NULL skel_a_id (0) — best we can do
    # For target skels not in trained set: same, use NULL skel_b_id (0) — model doesn't know test skels

    rng = np.random.RandomState(42)

    for fold in args.folds:
        if args.manifest:
            manifest = json.load(open(args.manifest))
        else:
            manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold}/manifest.json'))
        out_dir = SAVE_ROOT / args.out_tag / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)

        n_done = n_failed = 0
        per_query = []
        t0 = time.time()
        for i, q in enumerate(manifest['queries'][:args.max_queries]):
            qid = q['query_id']
            src_skel = q['skel_a']; tgt_skel = q['skel_b']
            src_fname = q['src_fname']
            src_action = q['src_action']

            try:
                # 1. Get source z (normalized) — encode from disk if not in cache
                if (src_skel, src_fname) in fname_to_ri:
                    src_ri = fname_to_ri[(src_skel, src_fname)]
                    z_a = z_per_skel[src_skel][src_ri].unsqueeze(0).to(device)  # [1, 8, 256]
                else:
                    # Encode source motion on-the-fly via Stage A
                    src_motion = np.load(MOTION_DIR / src_fname).astype(np.float32)
                    # Pad / window to 32 frames (Stage A window size)
                    if src_motion.shape[0] >= 32:
                        win = src_motion[:32]
                    else:
                        pad = np.zeros((32 - src_motion.shape[0], src_motion.shape[1], 13), dtype=np.float32)
                        win = np.concatenate([src_motion, pad], axis=0)
                    win_t = torch.from_numpy(win).to(device).unsqueeze(0)  # [1, T, J, 13]
                    win_norm = reg.normalize(src_skel, win_t)
                    z_raw, _ = reg.encode_window(src_skel, win_norm)         # [1, 8, 256] raw
                    # Normalize using src_skel stats
                    mu_a, sigma_a = z_stats[src_skel]
                    z_a = ((z_raw - mu_a) / sigma_a).to(device)

                # 2. Retrieval init: pick a same-action clip on tgt_skel
                # EXCLUDE all positives_cluster + positives_exact + adversarials to avoid leakage
                forbidden = set()
                for key in ('positives_cluster', 'positives_exact',
                            'adversarials_easy', 'adversarials_hard',
                            'distractors_same_target_skel'):
                    for x in q.get(key, []):
                        forbidden.add(x['fname'])

                init_candidates = skel_action_clips.get((tgt_skel, src_action), [])
                init_candidates = [(f, r) for f, r in init_candidates if f not in forbidden]
                used_noise_init = False
                if not init_candidates:
                    # Fall back to any clip on tgt_skel (out-of-action)
                    all_tgt = []
                    for k in skel_action_clips:
                        if k[0] == tgt_skel:
                            for f, r in skel_action_clips[k]:
                                if f not in forbidden:
                                    all_tgt.append((f, r))
                    if all_tgt:
                        init_candidates = all_tgt
                    else:
                        # Last resort: use Gaussian noise as init
                        used_noise_init = True

                if used_noise_init:
                    init_fname = '__noise__'
                    z_init = torch.randn(1, 8, 256, device=device)
                else:
                    init_fname, init_ri = init_candidates[rng.randint(len(init_candidates))]
                    z_init = z_per_skel[tgt_skel][init_ri].unsqueeze(0).to(device)  # [1, 8, 256]

                # 3. Add noise
                noise = torch.randn_like(z_init) * args.noise_scale
                z_start = z_init + noise

                # 4. Set up conditioning IDs (use 0 if held-out)
                sa_id = torch.tensor([skel_to_id.get(src_skel, 0)], device=device, dtype=torch.long)
                sb_id = torch.tensor([skel_to_id.get(tgt_skel, 0)], device=device, dtype=torch.long)
                aid = torch.tensor([exact_to_idx.get(src_action, 0)], device=device, dtype=torch.long)

                # 5. SB Euler integration: t in [0, 1]
                with torch.no_grad():
                    src_tokens = G.encode_source(z_a)
                    z_t = z_start.clone()
                    dt = 1.0 / args.n_steps
                    for k in range(args.n_steps):
                        t_now = (k + 0.5) / args.n_steps
                        t_b = torch.tensor([t_now], device=device)
                        v = G(z_t, t_b, src_tokens, aid, sa_id, sb_id)
                        z_t = z_t + dt * v

                # 6. Denormalize
                mu_b, sigma_b = z_stats[tgt_skel]
                z_b_pred = z_t * sigma_b + mu_b  # [1, 8, 256]

                # 7. Decode via Stage A
                with torch.no_grad():
                    motion_norm = reg.decode_tokens(tgt_skel, z_b_pred)  # [1, T, J, 13] normalized
                    motion = reg.unnormalize(tgt_skel, motion_norm)        # physical

                motion_np = motion[0].cpu().numpy().astype(np.float32)
                np.save(out_dir / f'query_{qid:04d}.npy', motion_np)
                per_query.append({
                    'query_id': qid, 'status': 'ok',
                    'src_skel': src_skel, 'tgt_skel': tgt_skel,
                    'src_action': src_action,
                    'init_fname': init_fname,
                    'output_T': int(motion_np.shape[0]),
                    'output_J': int(motion_np.shape[1]),
                })
                n_done += 1
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=2)
                print(f"  q{qid} FAILED: {e}\n{tb}")
                per_query.append({'query_id': qid, 'status': 'failed', 'error': str(e)})
                n_failed += 1

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(manifest['queries']) - i - 1)
                print(f"  fold {fold} [{i+1}/{len(manifest['queries'])}] "
                      f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s, ok={n_done}")

        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump({'method': args.out_tag, 'fold': fold, 'ckpt': args.ckpt,
                       'n_steps': args.n_steps, 'noise_scale': args.noise_scale,
                       'n_done': n_done, 'n_failed': n_failed,
                       'per_query': per_query}, f, indent=2)
        print(f"\nFold {fold}: {n_done} ok, {n_failed} failed.")


if __name__ == '__main__':
    main()
