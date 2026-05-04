"""Evaluate all K_retrieve ablation variants against ground truth target invariant reps.

Outputs: per-variant DTW, F1, Phase with bootstrap CI; paired comparison against 'full' baseline.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_paired.metrics import (
    end_effector_dtw, contact_timing_f1, phase_consistency,
)
from model.skel_blind.encoder import encode_motion_to_invariant
from model.skel_blind.invariant_dataset import TEST_SKELETONS

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
PAIRS_DIR = PROJECT_ROOT / 'eval/benchmark_paired/pairs'

VARIANTS = ['full', 'com_only', 'contact_only', 'cadence_only',
            'limb_only', 'no_com', 'no_contact', 'oracle_action']


def make_skel_cond(cond_dict, name):
    return {
        'joints_names': cond_dict[name]['joints_names'],
        'parents': cond_dict[name]['parents'],
        'object_type': name,
    }


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = np.array([rng.choice(arr, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(lo), float(arr.mean()), float(hi))


def win_rate_ci(a_vals, b_vals, lower_is_better=True, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    wins = (a < b).astype(np.float64) if lower_is_better else (a > b).astype(np.float64)
    rates = np.array([rng.choice(wins, size=len(wins), replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(rates, 2.5)
    hi = np.percentile(rates, 97.5)
    return (float(lo), float(wins.mean()), float(hi))


def main():
    print("Loading cond_dict...")
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()

    with open(PAIRS_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    pairs = manifest['pairs']
    print(f"Loaded {len(pairs)} pairs")

    # For each variant: per-pair metrics
    per_variant = {v: [] for v in VARIANTS}

    # Also include the existing K_retrieve_only (should match 'full')
    per_variant['retrieve_only_ref'] = []

    # Use saved retrieved outputs
    variant_dirs = {v: PROJECT_ROOT / f'eval/results/k_compare/K_retrieve_{v}_200pair'
                    for v in VARIANTS}
    # Add the reference implementation output
    variant_dirs['retrieve_only_ref'] = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_only_200pair'
    all_variants = VARIANTS + ['retrieve_only_ref']

    for p in pairs:
        pid = p['pair_id']
        skel_b = p['skel_b']
        T_b = None

        pair_data = np.load(PAIRS_DIR / p['pair_file'])
        inv_b = pair_data['inv_b']
        T_b = inv_b.shape[0]
        skel_cond = make_skel_cond(cond_dict, skel_b)

        for v in all_variants:
            vpath = variant_dirs[v] / f'pair_{pid:04d}.npy'
            if not vpath.exists():
                continue
            try:
                motion = np.load(vpath)
                motion_trunc = motion[:T_b]
                if motion_trunc.shape[0] < 2:
                    continue
                inv_v = encode_motion_to_invariant(motion_trunc, skel_cond)
                dtw = float(end_effector_dtw(inv_v, inv_b))
                f1 = float(contact_timing_f1(inv_v, inv_b))
                ph = float(phase_consistency(inv_v, inv_b))
                per_variant[v].append({
                    'pair_id': pid, 'dtw': dtw, 'f1': f1, 'phase': ph,
                })
            except Exception as e:
                print(f"  pair {pid} variant {v}: {e}")

    # Aggregate per variant
    print(f"\n{'='*80}")
    print(f"{'Variant':20s} {'N':>4s} {'DTW↓':>16s} {'F1↑':>16s} {'Phase↑':>16s}")
    print('-' * 80)
    summary = {}
    for v in all_variants:
        if not per_variant[v]:
            continue
        dtws = [r['dtw'] for r in per_variant[v]]
        f1s = [r['f1'] for r in per_variant[v]]
        phs = [r['phase'] for r in per_variant[v]]
        dtw_ci = bootstrap_ci(dtws)
        f1_ci = bootstrap_ci(f1s)
        ph_ci = bootstrap_ci(phs)
        summary[v] = {
            'n': len(per_variant[v]),
            'dtw': dtw_ci, 'f1': f1_ci, 'phase': ph_ci,
            'per_pair': per_variant[v],
        }
        print(f"{v:20s} {len(dtws):>4d} "
              f"{dtw_ci[1]:.3f} [{dtw_ci[0]:.3f},{dtw_ci[2]:.3f}] "
              f"{f1_ci[1]:.3f} [{f1_ci[0]:.3f},{f1_ci[2]:.3f}] "
              f"{ph_ci[1]:.3f} [{ph_ci[0]:.3f},{ph_ci[2]:.3f}]")

    # Paired vs 'full'
    print(f"\n--- PAIRED vs 'full' (win rates for each variant)---")
    if 'full' in summary:
        full_by_pid = {r['pair_id']: r for r in per_variant['full']}
        for v in all_variants:
            if v == 'full' or v not in summary:
                continue
            vby = {r['pair_id']: r for r in per_variant[v]}
            common = sorted(set(full_by_pid.keys()) & set(vby.keys()))
            if not common:
                continue
            v_dtw = [vby[pid]['dtw'] for pid in common]
            f_dtw = [full_by_pid[pid]['dtw'] for pid in common]
            v_f1 = [vby[pid]['f1'] for pid in common]
            f_f1 = [full_by_pid[pid]['f1'] for pid in common]
            wr_dtw = win_rate_ci(v_dtw, f_dtw, lower_is_better=True)
            wr_f1 = win_rate_ci(v_f1, f_f1, lower_is_better=False)
            print(f"  {v:18s} (n={len(common)}): "
                  f"DTW_win={wr_dtw[1]*100:.1f}% [{wr_dtw[0]*100:.1f},{wr_dtw[2]*100:.1f}] "
                  f"F1_win={wr_f1[1]*100:.1f}% [{wr_f1[0]*100:.1f},{wr_f1[2]*100:.1f}]")

    out_path = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_ablations_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
