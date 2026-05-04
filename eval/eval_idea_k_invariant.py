"""Apples-to-apples: Idea K vs Supervised CFM vs Random vs ZT on invariant-rep metrics.

Per reviewer feedback (2026-04-18):
  - Truncate to common length (T_b) before encoding to fix DTW-divisor and phase asymmetry
  - Add fair baselines:
      * Supervised CFM (same coord system as Idea K — primary upper bound)
      * Random target motion (sampled from target skeleton's own clip set)
      * Zero-training source-rep (kept for reference, but is a strawman due to coord asymmetry)
  - Report PAIRED bootstrap CIs on K-baseline deltas
  - Report win-rate CIs

Output: eval/results/k_compare/K_200pair/invariant_eval_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_paired.metrics import (
    end_effector_dtw, contact_timing_f1, phase_consistency,
)
from eval.build_contact_groups import classify_morphology
from model.skel_blind.cfm_model import SLOT_COUNT, CHANNEL_COUNT
from model.skel_blind.encoder import encode_motion_to_invariant
from model.skel_blind.invariant_dataset import (
    INVARIANT_DIR, TEST_SKELETONS,
)
from sample.generate_cfm import load_model, sample_euler

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
PAIRS_DIR = PROJECT_ROOT / 'eval/benchmark_paired/pairs'
K_OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_200pair'
K_RETRIEVE_OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_200pair'
K_RETRIEVE_ONLY_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_only_200pair'
SUPERVISED_CKPT = PROJECT_ROOT / 'save/cfm_supervised/ckpt_final.pt'


def make_skel_cond(cond_dict, name):
    return {
        "joints_names": cond_dict[name]["joints_names"],
        "parents": cond_dict[name]["parents"],
        "object_type": name,
    }


def get_morphology(cond_dict, skel_name):
    entry = cond_dict[skel_name]
    return classify_morphology(
        list(entry["joints_names"]),
        list(entry["parents"]),
        [list(c) for c in entry["kinematic_chains"]],
    )


def split_category(skel_a, skel_b):
    a_test = skel_a in TEST_SKELETONS
    b_test = skel_b in TEST_SKELETONS
    if a_test and b_test:
        return "test_test"
    elif a_test or b_test:
        return "mixed"
    return "train_train"


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


def paired_delta_ci(a_vals, b_vals, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap CI on the paired delta (b - a). Positive = a wins for "lower is better"."""
    rng = np.random.RandomState(seed)
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    deltas = b - a
    n = len(deltas)
    means = np.array([rng.choice(deltas, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(lo), float(deltas.mean()), float(hi))


def win_rate_ci(a_vals, b_vals, lower_is_better=True, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap CI on the paired win rate (a beats b)."""
    rng = np.random.RandomState(seed)
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    if lower_is_better:
        wins = (a < b).astype(np.float64)
    else:
        wins = (a > b).astype(np.float64)
    n = len(wins)
    rates = np.array([rng.choice(wins, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(rates, (1 - ci) / 2 * 100)
    hi = np.percentile(rates, (1 + ci) / 2 * 100)
    return (float(lo), float(wins.mean()), float(hi))


def load_target_skeleton_clip_pool():
    """Build a pool of pre-encoded clips per target skeleton with names for leakage filtering."""
    with open(INVARIANT_DIR + '/manifest.json') as f:
        manifest = json.load(f)
    pool = {}
    for skel_name in manifest['skeletons']:
        npz_path = os.path.join(INVARIANT_DIR, f'{skel_name}.npz')
        data = np.load(npz_path, allow_pickle=True)
        clips = [(name, data[name]) for name in manifest['skeletons'][skel_name]['clips']]
        if clips:
            pool[skel_name] = clips
    return pool


def evaluate(n_steps=30, cfg_weight=2.0, max_pairs=200, seed=42):
    print("Loading conditioning dict...")
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()

    print("Loading clip pool for random target baseline...")
    clip_pool = load_target_skeleton_clip_pool()
    print(f"Pool: {len(clip_pool)} skeletons")

    print(f"Loading supervised CFM from {SUPERVISED_CKPT}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sup_model, sup_args = load_model(str(SUPERVISED_CKPT), device)
    window = sup_args.get('window', 40)
    print(f"Supervised CFM window={window}, device={device}")

    with open(PAIRS_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    pairs = manifest['pairs'][:max_pairs]

    rng = np.random.RandomState(seed)
    results = []
    n_failed = 0

    for i, p in enumerate(pairs):
        pid = p['pair_id']
        skel_a = p['skel_a']
        skel_b = p['skel_b']
        action = p['action']

        k_path = K_OUT_DIR / f'pair_{pid:04d}.npy'
        kr_path = K_RETRIEVE_OUT_DIR / f'pair_{pid:04d}.npy'
        if not k_path.exists():
            n_failed += 1
            continue

        try:
            k_motion = np.load(k_path)
            pair_data = np.load(PAIRS_DIR / p['pair_file'])
            inv_a = pair_data['inv_a']  # source rep (different coord frame)
            inv_b = pair_data['inv_b']  # GROUND TRUTH target rep (target coord frame)
            T_b = inv_b.shape[0]

            # ------ Truncate K's output to T_b BEFORE encoding (fixes DTW-divisor/phase asymmetry)
            k_motion_trunc = k_motion[:T_b]
            skel_cond = make_skel_cond(cond_dict, skel_b)
            inv_k = encode_motion_to_invariant(k_motion_trunc, skel_cond)

            # ------ Idea K_retrieve (if available)
            inv_k_retrieve = None
            if kr_path.exists():
                kr_motion = np.load(kr_path)
                kr_motion_trunc = kr_motion[:T_b]
                if kr_motion_trunc.shape[0] > 0:
                    inv_k_retrieve = encode_motion_to_invariant(kr_motion_trunc, skel_cond)

            # ------ K_retrieve_only (Q-retrieved clip without refinement, for ablation)
            inv_k_retrieve_only = None
            kro_path = K_RETRIEVE_ONLY_DIR / f'pair_{pid:04d}.npy'
            if kro_path.exists():
                kro_motion = np.load(kro_path)
                kro_motion_trunc = kro_motion[:T_b]
                if kro_motion_trunc.shape[0] > 0:
                    inv_k_retrieve_only = encode_motion_to_invariant(kro_motion_trunc, skel_cond)

            # Truncate ZT inv_a to T_b for fair temporal comparison
            inv_a_trunc = inv_a[:T_b] if inv_a.shape[0] >= T_b else inv_a

            # ------ Random target baseline: pick random clip from target skeleton's pool, EXCLUDE GT (no leakage)
            gt_clip_stem = os.path.splitext(p['file_b'])[0]
            inv_rand_target = None
            if skel_b in clip_pool:
                candidates = [c for n, c in clip_pool[skel_b] if n != gt_clip_stem]
                if candidates:
                    inv_rand_target = candidates[rng.randint(0, len(candidates))]
                    # Truncate to T_b (no zero-padding to keep symmetry with inv_a_trunc)
                    if inv_rand_target.shape[0] >= T_b:
                        inv_rand_target = inv_rand_target[:T_b]

            # ------ Supervised CFM: only valid for pairs where T_a ≤ window AND T_b ≤ window
            # (model uses fixed window; longer clips would force unfair truncation/padding)
            if inv_a.shape[0] <= window and T_b <= window:
                inv_a_padded = np.zeros((window, SLOT_COUNT, CHANNEL_COUNT), dtype=np.float32)
                inv_a_padded[:inv_a.shape[0]] = inv_a
                z_src = torch.from_numpy(inv_a_padded).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    inv_sup_window = sample_euler(sup_model, z_src, n_steps=n_steps, cfg_weight=cfg_weight)
                inv_sup_full = inv_sup_window[0].cpu().numpy()
                inv_sup = inv_sup_full[:T_b]
            else:
                inv_sup = None

            # ------ Compute metrics: each method's output vs GROUND TRUTH inv_b
            def metrics_against_b(inv_x):
                return {
                    'dtw': float(end_effector_dtw(inv_x, inv_b)),
                    'f1': float(contact_timing_f1(inv_x, inv_b)),
                    'phase': float(phase_consistency(inv_x, inv_b)),
                }

            morph_a = get_morphology(cond_dict, skel_a)
            morph_b = get_morphology(cond_dict, skel_b)

            result = {
                'pair_id': pid,
                'action': action,
                'skel_a': skel_a, 'skel_b': skel_b,
                'morph_a': morph_a, 'morph_b': morph_b,
                'morph_pair': f"{morph_a}→{morph_b}",
                'split_cat': split_category(skel_a, skel_b),
                'T_b': int(T_b),
                'idea_k': metrics_against_b(inv_k),
                'idea_k_retrieve': metrics_against_b(inv_k_retrieve) if inv_k_retrieve is not None else None,
                'idea_k_retrieve_only': metrics_against_b(inv_k_retrieve_only) if inv_k_retrieve_only is not None else None,
                'supervised_cfm': metrics_against_b(inv_sup) if inv_sup is not None else None,
                'zero_training': metrics_against_b(inv_a_trunc),
                'random_target': metrics_against_b(inv_rand_target) if inv_rand_target is not None else None,
            }
            results.append(result)

            if (i + 1) % 20 == 0 or i == 0:
                sup_str = f"SUP_dtw={result['supervised_cfm']['dtw']:.3f}" if result['supervised_cfm'] else "SUP_dtw=N/A"
                print(f"  [{i+1}/{len(pairs)}] {action}: {skel_a}→{skel_b} "
                      f"K_dtw={result['idea_k']['dtw']:.3f} {sup_str} "
                      f"ZT_dtw={result['zero_training']['dtw']:.3f}")

        except Exception as e:
            n_failed += 1
            print(f"  FAILED pair {pid}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"APPLES-TO-APPLES INVARIANT-REP EVALUATION (n={len(results)}, failed={n_failed})")
    print(f"{'='*70}")

    methods = ['idea_k', 'idea_k_retrieve', 'idea_k_retrieve_only', 'supervised_cfm', 'zero_training', 'random_target']
    metric_names = [('dtw', True), ('f1', False), ('phase', False)]

    # Marginal
    print("\n--- MARGINAL METRICS (mean [95% CI]) ---")
    marginal = {}
    for method in methods:
        vals = {}
        for metric, _ in metric_names:
            valid = [r[method][metric] for r in results if r[method] is not None]
            if valid:
                ci = bootstrap_ci(valid)
                vals[metric] = ci
                print(f"  {method:18s} {metric:6s}: {ci[1]:.4f} [{ci[0]:.4f}, {ci[2]:.4f}]")
        marginal[method] = vals

    # Paired comparisons: Idea K vs each baseline (also K_retrieve vs baselines)
    print("\n--- PAIRED COMPARISONS: IDEA K vs each baseline ---")
    paired = {}
    for baseline in ['supervised_cfm', 'zero_training', 'random_target', 'idea_k_retrieve']:
        paired[baseline] = {}
        valid_pairs = [(r['idea_k'], r[baseline]) for r in results
                       if r[baseline] is not None]
        if not valid_pairs:
            continue
        k_vals = {m: [p[0][m] for p in valid_pairs] for m, _ in metric_names}
        b_vals = {m: [p[1][m] for p in valid_pairs] for m, _ in metric_names}

        print(f"\n  Idea K vs {baseline} (n={len(valid_pairs)}):")
        for metric, lower_is_better in metric_names:
            wr = win_rate_ci(k_vals[metric], b_vals[metric], lower_is_better=lower_is_better)
            delta_raw = paired_delta_ci(k_vals[metric], b_vals[metric])  # b - a
            sign = 1.0 if lower_is_better else -1.0  # flip so positive Δ = K wins
            delta = (sign * delta_raw[0], sign * delta_raw[1], sign * delta_raw[2])
            paired[baseline][metric] = {
                'n': len(valid_pairs),
                'win_rate': wr, 'k_minus_baseline_signed': delta,
                'k_mean': float(np.mean(k_vals[metric])),
                'b_mean': float(np.mean(b_vals[metric])),
            }
            direction = "↓" if lower_is_better else "↑"
            print(f"    {metric:6s} ({direction}): K={paired[baseline][metric]['k_mean']:.4f} "
                  f"vs {baseline[:8]}={paired[baseline][metric]['b_mean']:.4f}, "
                  f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f}, {wr[2]*100:.1f}], "
                  f"Δ_signed={delta[1]:+.4f} [{delta[0]:+.4f}, {delta[2]:+.4f}] (positive=K wins)")

    # ALSO: K_retrieve_only vs baselines (PRIMARY winner!)
    print("\n--- PAIRED COMPARISONS: IDEA K_retrieve_only vs each baseline ---")
    paired_retrieve_only = {}
    for baseline in ['supervised_cfm', 'zero_training', 'random_target', 'idea_k', 'idea_k_retrieve']:
        paired_retrieve_only[baseline] = {}
        valid_pairs = [(r['idea_k_retrieve_only'], r[baseline]) for r in results
                       if r.get('idea_k_retrieve_only') is not None and r[baseline] is not None]
        if not valid_pairs:
            continue
        kro_vals = {m: [p[0][m] for p in valid_pairs] for m, _ in metric_names}
        b_vals = {m: [p[1][m] for p in valid_pairs] for m, _ in metric_names}
        print(f"\n  Idea K_retrieve_only vs {baseline} (n={len(valid_pairs)}):")
        for metric, lower_is_better in metric_names:
            wr = win_rate_ci(kro_vals[metric], b_vals[metric], lower_is_better=lower_is_better)
            delta_raw = paired_delta_ci(kro_vals[metric], b_vals[metric])
            sign = 1.0 if lower_is_better else -1.0
            delta = (sign * delta_raw[0], sign * delta_raw[1], sign * delta_raw[2])
            paired_retrieve_only[baseline][metric] = {
                'n': len(valid_pairs), 'win_rate': wr,
                'k_minus_baseline_signed': delta,
                'k_mean': float(np.mean(kro_vals[metric])),
                'b_mean': float(np.mean(b_vals[metric])),
            }
            direction = "↓" if lower_is_better else "↑"
            print(f"    {metric:6s} ({direction}): KRO={paired_retrieve_only[baseline][metric]['k_mean']:.4f} "
                  f"vs {baseline[:8]}={paired_retrieve_only[baseline][metric]['b_mean']:.4f}, "
                  f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f}, {wr[2]*100:.1f}], "
                  f"Δ_signed={delta[1]:+.4f} [{delta[0]:+.4f}, {delta[2]:+.4f}]")

    # K_retrieve vs baselines
    print("\n--- PAIRED COMPARISONS: IDEA K_retrieve (refined) vs each baseline ---")
    paired_retrieve = {}
    for baseline in ['supervised_cfm', 'zero_training', 'random_target', 'idea_k']:
        paired_retrieve[baseline] = {}
        valid_pairs = [(r['idea_k_retrieve'], r[baseline]) for r in results
                       if r.get('idea_k_retrieve') is not None and r[baseline] is not None]
        if not valid_pairs:
            continue
        kr_vals = {m: [p[0][m] for p in valid_pairs] for m, _ in metric_names}
        b_vals = {m: [p[1][m] for p in valid_pairs] for m, _ in metric_names}
        print(f"\n  Idea K_retrieve vs {baseline} (n={len(valid_pairs)}):")
        for metric, lower_is_better in metric_names:
            wr = win_rate_ci(kr_vals[metric], b_vals[metric], lower_is_better=lower_is_better)
            delta_raw = paired_delta_ci(kr_vals[metric], b_vals[metric])
            sign = 1.0 if lower_is_better else -1.0
            delta = (sign * delta_raw[0], sign * delta_raw[1], sign * delta_raw[2])
            paired_retrieve[baseline][metric] = {
                'n': len(valid_pairs),
                'win_rate': wr,
                'k_minus_baseline_signed': delta,
                'k_mean': float(np.mean(kr_vals[metric])),
                'b_mean': float(np.mean(b_vals[metric])),
            }
            direction = "↓" if lower_is_better else "↑"
            print(f"    {metric:6s} ({direction}): KR={paired_retrieve[baseline][metric]['k_mean']:.4f} "
                  f"vs {baseline[:8]}={paired_retrieve[baseline][metric]['b_mean']:.4f}, "
                  f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f}, {wr[2]*100:.1f}], "
                  f"Δ_signed={delta[1]:+.4f} [{delta[0]:+.4f}, {delta[2]:+.4f}]")

    # By split — primary headline cut
    print("\n--- BY SPLIT (DTW only) ---")
    by_split = {}
    for cat in ["train_train", "mixed", "test_test"]:
        group = [r for r in results if r['split_cat'] == cat]
        if not group:
            continue
        by_split[cat] = {'n': len(group)}
        for method in methods:
            valid = [r[method]['dtw'] for r in group if r[method] is not None]
            if valid:
                by_split[cat][method] = bootstrap_ci(valid) + (len(valid),)

        # Paired win rates per baseline within split
        for baseline in ['supervised_cfm', 'zero_training', 'random_target']:
            valid_pairs = [(r['idea_k']['dtw'], r[baseline]['dtw']) for r in group
                           if r[baseline] is not None]
            if valid_pairs:
                k_v = [p[0] for p in valid_pairs]
                b_v = [p[1] for p in valid_pairs]
                wr = win_rate_ci(k_v, b_v, lower_is_better=True)
                by_split[cat][f'k_vs_{baseline}_win'] = wr + (len(valid_pairs),)

        msg = f"  {cat} (n={len(group)}): K_dtw={by_split[cat]['idea_k'][1]:.4f}"
        for baseline in ['supervised_cfm', 'zero_training', 'random_target']:
            wr_key = f'k_vs_{baseline}_win'
            if wr_key in by_split[cat]:
                wr = by_split[cat][wr_key]
                msg += f" K_vs_{baseline[:3]}={wr[1]*100:.0f}% (n={wr[3]})"
        print(msg)

    summary = {
        'method': 'Idea_K',
        'n_pairs': len(results),
        'n_failed': n_failed,
        'marginal': marginal,
        'paired_idea_k': paired,
        'paired_idea_k_retrieve': paired_retrieve,
        'paired_idea_k_retrieve_only': paired_retrieve_only,
        'by_split': by_split,
        'per_pair': results,
    }

    out_path = K_OUT_DIR / 'invariant_eval_v2.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Print headline
    print("\n" + "=" * 70)
    print("HEADLINE (positive Δ_signed = K wins)")
    print("=" * 70)
    print("\n[Original Idea K]")
    for baseline in ['random_target', 'zero_training', 'supervised_cfm']:
        if baseline not in paired or 'dtw' not in paired[baseline]:
            continue
        n = paired[baseline]['dtw']['n']
        wr = paired[baseline]['dtw']['win_rate']
        d = paired[baseline]['dtw']['k_minus_baseline_signed']
        print(f"  K vs {baseline:18s} (n={n}) DTW: "
              f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f},{wr[2]*100:.1f}], "
              f"Δ_signed={d[1]:+.4f}")
    print("\n[Idea K_retrieve_only — Q-retrieval, no refinement] *** PRIMARY ***")
    for baseline in ['random_target', 'zero_training', 'supervised_cfm', 'idea_k', 'idea_k_retrieve']:
        if baseline not in paired_retrieve_only or 'dtw' not in paired_retrieve_only[baseline]:
            continue
        n = paired_retrieve_only[baseline]['dtw']['n']
        wr = paired_retrieve_only[baseline]['dtw']['win_rate']
        d = paired_retrieve_only[baseline]['dtw']['k_minus_baseline_signed']
        print(f"  KRO vs {baseline:18s} (n={n}) DTW: "
              f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f},{wr[2]*100:.1f}], "
              f"Δ_signed={d[1]:+.4f}")
    print("\n[Idea K_retrieve — Q-retrieval + AnyTop refinement]")
    for baseline in ['random_target', 'zero_training', 'supervised_cfm', 'idea_k']:
        if baseline not in paired_retrieve or 'dtw' not in paired_retrieve[baseline]:
            continue
        n = paired_retrieve[baseline]['dtw']['n']
        wr = paired_retrieve[baseline]['dtw']['win_rate']
        d = paired_retrieve[baseline]['dtw']['k_minus_baseline_signed']
        print(f"  KR vs {baseline:18s} (n={n}) DTW: "
              f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f},{wr[2]*100:.1f}], "
              f"Δ_signed={d[1]:+.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pairs', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--cfg_weight', type=float, default=2.0)
    args = parser.parse_args()
    evaluate(n_steps=args.n_steps, cfg_weight=args.cfg_weight, max_pairs=args.max_pairs)
