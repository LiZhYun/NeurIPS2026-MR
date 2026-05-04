"""Re-eval all methods on v3 with fixed q_component metric (per-query z-scored composite).

Per Codex Round 1 (2026-04-23): the original q_component used raw unnormalized averaging
which let one component dominate. Fixed to per-query within-pool z-scoring.

This script re-runs eval_v3 with --distance q_component for all known method dirs.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# All v3 method dirs (fold-level)
METHOD_DIRS = [
    # legacy baselines
    ('eval/results/baselines/anytop_train_v3/v3_fold_{F}', 'anytop_train_v3'),
    ('eval/results/baselines/k_frame_nn/v3_fold_{F}', 'k_frame_nn'),
    ('eval/results/baselines/k_retrieve/v3_fold_{F}', 'k_retrieve'),
    ('eval/results/baselines/m2m_lite/v3_fold_{F}', 'm2m_lite'),
    # moreflow
    ('save/moreflow/v3/primary_70/fold_{F}', 'moreflow_primary_70'),
    ('save/moreflow/v3/inductive_60/fold_{F}', 'moreflow_inductive_60'),
    # ace
    ('save/ace_inference/v3/ace_primary_70/fold_{F}', 'ace_primary_70'),
    ('save/ace_inference/v3/ace_inductive_60/fold_{F}', 'ace_inductive_60'),
    # oracles
    ('save/oracles/v3/action_oracle/fold_{F}', 'oracle_action_oracle'),
    ('save/oracles/v3/self_positive/fold_{F}', 'oracle_self_positive'),
    ('save/oracles/v3/random_skel_b/fold_{F}', 'oracle_random_skel_b'),
]

FOLDS = [42, 43]


def main():
    n_done = 0
    n_failed = 0
    for dir_template, name in METHOD_DIRS:
        for fold in FOLDS:
            method_dir = PROJECT_ROOT / dir_template.format(F=fold)
            if not method_dir.exists():
                print(f"SKIP {name} fold {fold}: dir missing ({method_dir})")
                continue
            cmd = [
                sys.executable, '-u', '-m', 'eval.benchmark_v3.eval_v3',
                '--method_dir', str(method_dir),
                '--fold', str(fold),
                '--method_name', name,
                '--distance', 'q_component',
            ]
            print(f"=== {name} fold {fold} q_component ===")
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if r.returncode == 0:
                n_done += 1
            else:
                n_failed += 1
                print(f"  FAILED: {name} fold {fold}")
    print(f"\nDone: {n_done} succeeded, {n_failed} failed")


if __name__ == '__main__':
    main()
