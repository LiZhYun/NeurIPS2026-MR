"""Run ACE Stage 2 on v3 benchmark (inference + eval glue).

Mirrors run_moreflow_v3.py — invokes sample/ace_retarget.py + eval/benchmark_v3/eval_v3.py.

Usage:
  python -m eval.baselines.run_ace_v3 --run_name ace_primary_70
  python -m eval.baselines.run_ace_v3 --run_name ace_inductive_60
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--skip_inference', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    ckpt = PROJECT_ROOT / f'save/ace/{args.run_name}/ckpt_final.pt'
    if not ckpt.exists():
        print(f"FATAL: {ckpt} not found.", file=sys.stderr)
        sys.exit(1)

    out_root = PROJECT_ROOT / f'save/ace_inference/v3/{args.run_name}'

    for fold in args.folds:
        out_dir = out_root / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_inference:
            cmd = [
                sys.executable, '-u', '-m', 'sample.ace_retarget',
                '--ckpt', str(ckpt),
                '--fold', str(fold),
                '--out_dir', str(out_dir),
            ]
            if args.limit:
                cmd += ['--limit', str(args.limit)]
            print(f"\n=== ACE Inference fold {fold} ===")
            print(' '.join(cmd))
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if r.returncode != 0:
                print(f"  Inference fold {fold} FAILED")
                continue

        if not args.skip_eval:
            print(f"\n=== ACE Eval fold {fold} ===")
            for distance in ('procrustes', 'zscore_dtw', 'q_component'):
                cmd = [
                    sys.executable, '-u', '-m', 'eval.benchmark_v3.eval_v3',
                    '--method_dir', str(out_dir),
                    '--fold', str(fold),
                    '--method_name', f'ace_{args.run_name}',
                    '--distance', distance,
                ]
                print(' '.join(cmd))
                r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
                if r.returncode != 0:
                    print(f"  Eval fold {fold} distance={distance} FAILED")


if __name__ == '__main__':
    main()
