"""D — Per-skel diffusion (AnyTop SDEdit) refinement of M3 Phase A candidates.

Per FINAL_PROPOSAL V4: per-skel latent diffusion priors as fallback. Here we
reuse the existing AnyTop A3 baseline (single multi-skel diffusion model,
skel_id-conditioned) instead of training 70 separate per-skel models —
practically equivalent since AnyTop already learned each skel's prior via
self-recon.

Pipeline per query:
  1. Get top-1 candidate from M3 Phase A v1 (skel_b motion in 13-dim)
  2. SDEdit refine via AnyTop:
     a. Diffuse candidate to t_init * T_max (q_sample)
     b. Reverse n_steps DDIM steps with skel_b conditioning
  3. Output refined motion, save as query_NNNN.npy

Codex's make-or-break flag: init from RETRIEVED CANDIDATE's noisy version,
NOT pure noise. We do that by construction (start from candidate, add noise).

Usage:
  python -m eval.baselines.run_D_anytop_sdedit \
      --folds 42 --t_init 0.4 --n_steps 20 \
      --phase_a_dir save/m3/m3_rerank_v1 --out_tag D_sdedit_t40
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

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR

DATA_ROOT = Path(DATASET_DIR)
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
SAVE_ROOT = PROJECT_ROOT / 'save/D_sdedit'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42])
    parser.add_argument('--phase_a_dir', type=str, default='save/m3/m3_rerank_v1')
    parser.add_argument('--out_tag', type=str, default='D_sdedit_t40')
    parser.add_argument('--t_init', type=float, default=0.4)
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--lambda_com', type=float, default=0.0,
                        help='COM guidance weight (0 = disable)')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"D = AnyTop SDEdit on M3 Phase A candidates")
    print(f"  t_init={args.t_init}, n_steps={args.n_steps}, lambda_com={args.lambda_com}")

    # Lazy import — anytop_project loads heavy models on first call
    from eval.anytop_projection import anytop_project

    phase_a_root = Path(args.phase_a_dir)
    if not phase_a_root.is_absolute():
        phase_a_root = PROJECT_ROOT / phase_a_root

    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()

    for fold in args.folds:
        manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold}/manifest.json'))
        out_dir = SAVE_ROOT / args.out_tag / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)
        phase_a_fold = phase_a_root / f'fold_{fold}'
        if not phase_a_fold.exists():
            print(f"WARN: Phase A fold dir not found: {phase_a_fold}"); continue

        n_done = n_failed = 0
        per_query = []
        t0 = time.time()

        for i, q in enumerate(manifest['queries'][:args.max_queries]):
            qid = q['query_id']
            skel_b = q['skel_b']
            cand_path = phase_a_fold / f'query_{qid:04d}.npy'
            if not cand_path.exists():
                n_failed += 1; continue

            try:
                cand = np.load(cand_path).astype(np.float32)
                # Crop joints to skel_b's actual count
                n_joints_b = len(cond_dict[skel_b]['parents'])
                if cand.shape[1] > n_joints_b:
                    cand = cand[:, :n_joints_b]

                refined_dict = anytop_project(
                    x_init=cand,
                    target_skel=skel_b,
                    hard_constraints=None,
                    t_init=args.t_init,
                    n_steps=args.n_steps,
                    lambda_com=args.lambda_com,
                    device=args.device,
                )
                x_refined = refined_dict['x_refined']
                np.save(out_dir / f'query_{qid:04d}.npy', x_refined.astype(np.float32))
                per_query.append({
                    'query_id': qid, 'status': 'ok',
                    'skel_b': skel_b, 'n_joints': int(n_joints_b),
                    'runtime': float(refined_dict.get('runtime_seconds', 0)),
                })
                n_done += 1
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=2)
                print(f"  q{qid} FAILED: {e}\n{tb}")
                per_query.append({'query_id': qid, 'status': 'failed', 'error': str(e)})
                n_failed += 1

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(manifest['queries']) - i - 1)
                print(f"  fold {fold} [{i+1}/{len(manifest['queries'])}] "
                      f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s, ok={n_done}")

        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump({
                'method': args.out_tag, 'fold': fold,
                't_init': args.t_init, 'n_steps': args.n_steps,
                'n_done': n_done, 'n_failed': n_failed,
                'per_query': per_query,
            }, f, indent=2, default=str)
        print(f"\nFold {fold}: {n_done} ok, {n_failed} failed.")


if __name__ == '__main__':
    main()
