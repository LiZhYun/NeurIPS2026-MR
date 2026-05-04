"""Run MoReFlow Stage B on v3 benchmark (inference + eval glue).

Wraps sample/moreflow_retarget.py + eval/benchmark_v3/eval_v3.py for one Stage B run
across both v3 folds (42 + 43).

Usage:
  python -m eval.baselines.run_moreflow_v3 --run_name primary_70
  python -m eval.baselines.run_moreflow_v3 --run_name inductive_60
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='Stage B run name (matches save/moreflow_flow/<run_name>/ckpt_final.pt)')
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference; just run eval on existing outputs')
    parser.add_argument('--skip_eval', action='store_true', help='Skip eval; just run inference')
    parser.add_argument('--limit', type=int, default=None, help='Limit queries (debug)')
    args = parser.parse_args()

    ckpt = PROJECT_ROOT / f'save/moreflow_flow/{args.run_name}/ckpt_final.pt'
    if not ckpt.exists():
        print(f"FATAL: {ckpt} not found.", file=sys.stderr)
        sys.exit(1)

    out_root = PROJECT_ROOT / f'save/moreflow/v3/{args.run_name}'

    for fold in args.folds:
        out_dir = out_root / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_inference:
            cmd = [
                'python', '-u', '-m', 'sample.moreflow_retarget',
                '--ckpt', str(ckpt),
                '--fold', str(fold),
                '--out_dir', str(out_dir),
            ]
            if args.limit:
                cmd += ['--limit', str(args.limit)]
            print(f"\n=== Inference fold {fold} ===")
            print(' '.join(cmd))
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if r.returncode != 0:
                print(f"  Inference fold {fold} FAILED")
                continue

        if not args.skip_eval:
            print(f"\n=== Eval fold {fold} ===")
            # Run eval once per distance metric — matches how the other baselines were evaluated.
            # eval_v3 expects --method_dir (not --predictions_dir). --restrict_split isn't
            # supported; post-hoc slicing happens during result aggregation instead.
            for distance in ('procrustes', 'zscore_dtw', 'q_component'):
                cmd = [
                    'python', '-u', '-m', 'eval.benchmark_v3.eval_v3',
                    '--method_dir', str(out_dir),
                    '--fold', str(fold),
                    '--method_name', f'moreflow_{args.run_name}',
                    '--distance', distance,
                ]
                print(' '.join(cmd))
                r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
                if r.returncode != 0:
                    print(f"  Eval fold {fold} distance={distance} FAILED")


if __name__ == '__main__':
    main()
