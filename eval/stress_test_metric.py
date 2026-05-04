"""Stress-test the invariant-rep DTW metric against reviewer concerns:

1. Run all methods on BOTH raw DTW and body-scale-normalized DTW
2. Add an ADVERSARIAL baseline: retrieve a target-skel clip with OBVIOUSLY
   WRONG action (deliberately pick a non-matching action from a disjoint category)
3. If K_retrieve wins and adversarial-retrieval LOSES, metric is OK.
   If adversarial wins or ties, metric is broken.

Output: eval/results/k_compare/metric_stress_test.json
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_paired.metrics.end_effector_dtw import end_effector_dtw
from eval.benchmark_paired.metrics.end_effector_dtw_normalized import (
    end_effector_dtw_normalized,
)
from eval.benchmark_paired.metrics.contact_timing_f1 import contact_timing_f1
from model.skel_blind.encoder import encode_motion_to_invariant

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
PAIRS_DIR = PROJECT_ROOT / 'eval/benchmark_paired/pairs'
K_RETRIEVE_ONLY_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_only_200pair'
MOTION_DIR = DATA_ROOT / 'motions'

# Adversarial: for GT action "walk", pick clip with "attack" or "die" (wrong action)
DISJOINT_ACTIONS = {
    'walk': ['attack', 'die', 'bite', 'eat', 'sniff'],
    'run': ['attack', 'die', 'bite', 'eat', 'sniff'],
    'idle': ['attack', 'run', 'jump', 'die'],
    'attack': ['idle', 'walk', 'sniff', 'eat'],
    'fly': ['walk', 'idle', 'sit', 'eat'],
    'default': ['attack', 'die', 'bite'],  # generic disjoint
}


def make_skel_cond(cond_dict, name):
    return {
        'joints_names': cond_dict[name]['joints_names'],
        'parents': cond_dict[name]['parents'],
        'object_type': name,
    }


def bootstrap_ci(values, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return (float(lo), float(arr.mean()), float(hi))


def win_rate_ci(a, b, lower_is_better=True, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    wins = (a < b) if lower_is_better else (a > b)
    wins = wins.astype(np.float64)
    rates = np.array([rng.choice(wins, size=len(wins), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(rates, 2.5)), float(wins.mean()), float(np.percentile(rates, 97.5)))


def parse_action_coarse(fname):
    import re
    parts = fname.split('___')
    if len(parts) >= 2:
        raw = parts[1].rsplit('_', 1)[0].lower()
        stripped = re.sub(r'\d+$', '', raw)
        return stripped if stripped else raw
    return fname.lower()


def list_tgt_clips(skel, motion_dir):
    return sorted([f for f in os.listdir(motion_dir) if f.startswith(skel + '___') and f.endswith('.npy')])


def main():
    print("Loading cond...")
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()

    with open(PAIRS_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    pairs = manifest['pairs']
    print(f"Evaluating {len(pairs)} pairs")

    # Cache per-target-skel clips
    tgt_clips = {}
    rng = np.random.RandomState(42)

    results = []
    for i, p in enumerate(pairs):
        pid = p['pair_id']
        skel_b = p['skel_b']
        action = p['action']
        gt_file = p['file_b']

        pair_data = np.load(PAIRS_DIR / p['pair_file'])
        inv_a = pair_data['inv_a']
        inv_b = pair_data['inv_b']
        T_b = inv_b.shape[0]
        skel_cond = make_skel_cond(cond_dict, skel_b)

        # Random target (excluding GT)
        if skel_b not in tgt_clips:
            tgt_clips[skel_b] = list_tgt_clips(skel_b, MOTION_DIR)
        pool = [f for f in tgt_clips[skel_b] if f != gt_file]
        if not pool:
            continue

        # K_retrieve (already computed)
        kr_path = K_RETRIEVE_ONLY_DIR / f'pair_{pid:04d}.npy'
        if not kr_path.exists():
            continue

        # Adversarial: pick a target-skel clip whose coarse action is DISJOINT from action
        disjoint = DISJOINT_ACTIONS.get(action, DISJOINT_ACTIONS['default'])
        adv_candidates = [f for f in pool if parse_action_coarse(f) in disjoint]
        if adv_candidates:
            adv_fname = adv_candidates[rng.randint(0, len(adv_candidates))]
        else:
            adv_fname = None  # no disjoint clip available

        # Random
        rand_fname = pool[rng.randint(0, len(pool))]

        # Encode all
        try:
            kr_motion = np.load(kr_path)[:T_b]
            inv_kr = encode_motion_to_invariant(kr_motion, skel_cond)

            rand_motion = np.load(MOTION_DIR / rand_fname)[:T_b]
            inv_rand = encode_motion_to_invariant(rand_motion, skel_cond)

            inv_adv = None
            adv_action = None
            if adv_fname:
                adv_motion = np.load(MOTION_DIR / adv_fname)[:T_b]
                inv_adv = encode_motion_to_invariant(adv_motion, skel_cond)
                adv_action = parse_action_coarse(adv_fname)

            # Metrics against GT
            def both_dtw(inv_x):
                return {
                    'dtw_raw': float(end_effector_dtw(inv_x, inv_b)),
                    'dtw_zscore': float(end_effector_dtw_normalized(inv_x, inv_b, 'zscore')),
                    'dtw_scale': float(end_effector_dtw_normalized(inv_x, inv_b, 'scale')),
                    'f1': float(contact_timing_f1(inv_x, inv_b)),
                }

            rec = {
                'pair_id': pid, 'action': action, 'target_skel': skel_b,
                'adv_action': adv_action,
                'k_retrieve': both_dtw(inv_kr),
                'random': both_dtw(inv_rand),
                'zero_training': both_dtw(inv_a[:T_b]),
            }
            if inv_adv is not None:
                rec['adversarial'] = both_dtw(inv_adv)
            results.append(rec)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(pairs)}] K_dtw_raw={rec['k_retrieve']['dtw_raw']:.3f} "
                      f"zscore={rec['k_retrieve']['dtw_zscore']:.3f}")
        except Exception as e:
            print(f"  FAILED pair {pid}: {e}")

    print(f"\n{'='*80}")
    print(f"STRESS TEST: Idea K_retrieve vs Random vs ADVERSARIAL (wrong action)")
    print(f"{'='*80}")
    print(f"n = {len(results)}")
    n_adv = sum(1 for r in results if 'adversarial' in r)
    print(f"n with adversarial = {n_adv}")

    def agg_vs(method, metric):
        vals = [r[method][metric] for r in results if method in r]
        if not vals:
            return None
        return bootstrap_ci(vals)

    print("\n--- DTW (raw) ---")
    for method in ['k_retrieve', 'random', 'zero_training', 'adversarial']:
        ci = agg_vs(method, 'dtw_raw')
        if ci:
            print(f"  {method:15s}: {ci[1]:.4f} [{ci[0]:.4f}, {ci[2]:.4f}]")

    print("\n--- DTW (z-score normalized — scale invariant) ---")
    for method in ['k_retrieve', 'random', 'zero_training', 'adversarial']:
        ci = agg_vs(method, 'dtw_zscore')
        if ci:
            print(f"  {method:15s}: {ci[1]:.4f} [{ci[0]:.4f}, {ci[2]:.4f}]")

    print("\n--- DTW (scale normalized) ---")
    for method in ['k_retrieve', 'random', 'zero_training', 'adversarial']:
        ci = agg_vs(method, 'dtw_scale')
        if ci:
            print(f"  {method:15s}: {ci[1]:.4f} [{ci[0]:.4f}, {ci[2]:.4f}]")

    print("\n--- F1 ---")
    for method in ['k_retrieve', 'random', 'zero_training', 'adversarial']:
        ci = agg_vs(method, 'f1')
        if ci:
            print(f"  {method:15s}: {ci[1]:.4f} [{ci[0]:.4f}, {ci[2]:.4f}]")

    print("\n--- PAIRED: K_retrieve vs each (all metrics) ---")
    for baseline in ['random', 'adversarial']:
        group = [r for r in results if baseline in r]
        if not group:
            continue
        for metric in ['dtw_raw', 'dtw_zscore', 'dtw_scale', 'f1']:
            a = [r['k_retrieve'][metric] for r in group]
            b = [r[baseline][metric] for r in group]
            wr = win_rate_ci(a, b, lower_is_better='dtw' in metric)
            direction = '↓' if 'dtw' in metric else '↑'
            print(f"  K vs {baseline:13s} on {metric:10s} ({direction}, n={len(group)}): "
                  f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f}, {wr[2]*100:.1f}]")

    out_path = PROJECT_ROOT / 'eval/results/k_compare/metric_stress_test.json'
    with open(out_path, 'w') as f:
        json.dump({'n': len(results), 'n_adv': n_adv, 'per_pair': results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
