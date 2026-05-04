"""Principled cross-skeleton metrics — Q-component comparison.

Per user feedback: cross-skeleton metrics need careful handling.
This evaluates ALL methods on Q-component metrics (morphology-invariant by design):
  1. COM path shape (after body-scale norm)
  2. Heading velocity (body-scale norm, in body frame)
  3. Contact schedule (per-group F1, properly aligned)
  4. Cadence (Hz, body-scale invariant)
  5. Limb usage distribution (energy fraction per kinematic chain)

These are the same morphology-equivariant features used in Q extraction.
Comparing Q(method_output) vs Q(GT) tests semantic preservation independently
of the invariant rep space (no circularity).

Output: eval/results/k_compare/principled_metrics.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAIRS_DIR = PROJECT_ROOT / 'eval/benchmark_paired/pairs'
METHODS = {
    'k_retrieve': 'eval/results/k_compare/K_retrieve_only_200pair',
    'k_frame_nn': 'eval/results/k_compare/K_frame_nn_200pair',
    'k_dtw_align': 'eval/results/k_compare/K_dtw_align_200pair',
}


def extract_q_from_array(motion, cond, contact_groups_for_skel, fps=30):
    """Extract Q components from motion array [T, J, 13]. Same as extract_quotient
    but takes array instead of filename."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.effect_program import canonical_body_frame
    from eval.quotient_extractor import (
        compute_subtree_masses, body_scale, compute_heading_velocity,
        compute_contact_schedule_grouped, compute_cadence, compute_limb_usage,
        compute_contact_schedule_aggregate,
    )

    parents = cond['parents']
    offsets = cond['offsets']
    chains = cond['kinematic_chains']
    J = offsets.shape[0]

    if motion.shape[1] > J:
        motion = motion[:, :J]
    contacts = (motion[..., 12] > 0.5).astype(np.int8)
    positions = recover_from_bvh_ric_np(motion.astype(np.float32))
    T = positions.shape[0]

    subtree = compute_subtree_masses(parents, offsets)
    centroids, R = canonical_body_frame(positions, subtree)
    scale = body_scale(offsets)
    com_path = (centroids - centroids[0:1]) / scale
    heading_vel = compute_heading_velocity(centroids, R, fps=fps) / scale

    if contact_groups_for_skel is not None:
        sched, names = compute_contact_schedule_grouped(contacts, contact_groups_for_skel)
    else:
        sched = compute_contact_schedule_aggregate(contacts)
        names = None

    cadence = compute_cadence(sched, fps=fps)
    limb_usage = compute_limb_usage(positions, chains, fps=fps)

    return {
        'com_path': com_path.astype(np.float32),
        'heading_vel': heading_vel.astype(np.float32),
        'contact_sched': sched.astype(np.float32),
        'contact_group_names': names,
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
        'body_scale': float(scale),
    }


def compare_q(q_pred, q_gt, contact_groups_pred, contact_groups_gt):
    """Compare two Q dicts component-wise, returning per-component metrics."""
    # COM path: relative L2 (lower = better)
    com_pred = np.asarray(q_pred['com_path'])
    com_gt = np.asarray(q_gt['com_path'])
    T = min(com_pred.shape[0], com_gt.shape[0])
    com_diff = float(np.linalg.norm(com_pred[:T] - com_gt[:T]))
    com_norm = float(np.linalg.norm(com_gt[:T]) + 1e-9)
    com_rel = com_diff / com_norm

    # Heading vel: relative L2
    hv_pred = np.asarray(q_pred['heading_vel'])
    hv_gt = np.asarray(q_gt['heading_vel'])
    T = min(hv_pred.shape[0], hv_gt.shape[0])
    hv_diff = float(np.linalg.norm(hv_pred[:T] - hv_gt[:T]))
    hv_norm = float(np.linalg.norm(hv_gt[:T]) + 1e-9)
    hv_rel = hv_diff / hv_norm

    # Cadence: absolute difference (Hz)
    cad_diff = abs(float(q_pred['cadence']) - float(q_gt['cadence']))

    # Limb usage: L2 on aligned distribution
    # Sort both descending, pad to common length
    lu_pred = -np.sort(-np.asarray(q_pred['limb_usage']))
    lu_gt = -np.sort(-np.asarray(q_gt['limb_usage']))
    K = max(len(lu_pred), len(lu_gt))
    lu_pred = np.pad(lu_pred, (0, K - len(lu_pred)))
    lu_gt = np.pad(lu_gt, (0, K - len(lu_gt)))
    lu_l2 = float(np.linalg.norm(lu_pred - lu_gt))

    # Contact schedule: aggregate to total contact count per frame, F1
    cs_pred = np.asarray(q_pred['contact_sched'])
    cs_gt = np.asarray(q_gt['contact_sched'])
    if cs_pred.ndim > 1:
        cs_pred_agg = (cs_pred.sum(axis=1) > 0).astype(np.float32)
    else:
        cs_pred_agg = (cs_pred > 0).astype(np.float32)
    if cs_gt.ndim > 1:
        cs_gt_agg = (cs_gt.sum(axis=1) > 0).astype(np.float32)
    else:
        cs_gt_agg = (cs_gt > 0).astype(np.float32)
    T = min(len(cs_pred_agg), len(cs_gt_agg))
    pred_b = cs_pred_agg[:T] > 0.5
    gt_b = cs_gt_agg[:T] > 0.5
    tp = float(((pred_b == 1) & (gt_b == 1)).sum())
    fp = float(((pred_b == 1) & (gt_b == 0)).sum())
    fn = float(((pred_b == 0) & (gt_b == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    cs_f1 = 2 * p * r / (p + r + 1e-8)

    return {
        'com_rel_l2': com_rel,
        'hv_rel_l2': hv_rel,
        'cadence_abs': cad_diff,
        'limb_usage_l2': lu_l2,
        'contact_sched_f1': cs_f1,
    }


def bootstrap_ci(values, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(means, 2.5)), float(arr.mean()), float(np.percentile(means, 97.5)))


def win_rate_ci(a, b, lower_is_better=True, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    a, b = np.asarray(a), np.asarray(b)
    wins = (a < b) if lower_is_better else (a > b)
    wins = wins.astype(np.float64)
    rates = np.array([rng.choice(wins, size=len(wins), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(rates, 2.5)), float(wins.mean()), float(np.percentile(rates, 97.5)))


def main():
    print("Loading cond...")
    DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()
    with open(PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        contact_groups = json.load(f)

    motion_dir = DATA_ROOT / 'motions'

    # Load adversarial info from stress_test_v2
    with open(PROJECT_ROOT / 'eval/results/k_compare/metric_stress_test_v2.json') as f:
        stress = json.load(f)
    pid_to_adv_action = {r['pair_id']: r.get('adv_cluster') for r in stress['per_pair']}
    pid_to_adv_meta = {r['pair_id']: r for r in stress['per_pair']}

    with open(PAIRS_DIR / 'manifest.json') as f:
        manifest = json.load(f)

    # Build per-method per-pair Q comparisons
    results_by_method = defaultdict(list)

    rng_adv = np.random.RandomState(42)
    tgt_clip_cache = {}

    print(f"\nProcessing {len(manifest['pairs'])} pairs...")
    t0 = time.time()
    for i, p in enumerate(manifest['pairs']):
        pid = p['pair_id']
        skel_b = p['skel_b']
        gt_file = p['file_b']

        if skel_b not in contact_groups:
            continue

        # Load GT motion (target) and compute Q(GT)
        try:
            gt_motion = np.load(motion_dir / gt_file).astype(np.float32)
            cg_b = contact_groups.get(skel_b)
            q_gt = extract_q_from_array(gt_motion, cond_dict[skel_b], cg_b)
        except Exception as e:
            print(f"  pair {pid} GT failed: {e}")
            continue

        # For each method's output
        for method_name, method_dir in METHODS.items():
            method_out = PROJECT_ROOT / method_dir / f'pair_{pid:04d}.npy'
            if not method_out.exists():
                continue
            try:
                m_motion = np.load(method_out).astype(np.float32)
                q_pred = extract_q_from_array(m_motion, cond_dict[skel_b], cg_b)
                cmp = compare_q(q_pred, q_gt, cg_b, cg_b)
                cmp['pair_id'] = pid
                results_by_method[method_name].append(cmp)
            except Exception as e:
                print(f"  pair {pid} method {method_name} failed: {e}")

        # Adversarial baseline
        try:
            if skel_b not in tgt_clip_cache:
                tgt_clip_cache[skel_b] = sorted([f for f in os.listdir(motion_dir)
                                                  if f.startswith(skel_b + '___') and f.endswith('.npy')
                                                  and f != gt_file])
            adv_meta = pid_to_adv_meta.get(pid, {})
            adv_cluster = adv_meta.get('adv_cluster')
            if adv_cluster:
                from eval.stress_test_v2 import action_to_cluster, parse_action_coarse
                adv_pool = [f for f in tgt_clip_cache[skel_b]
                            if action_to_cluster(parse_action_coarse(f)) == adv_cluster]
                if adv_pool:
                    adv_fname = adv_pool[rng_adv.randint(0, len(adv_pool))]
                    adv_motion = np.load(motion_dir / adv_fname).astype(np.float32)
                    q_adv = extract_q_from_array(adv_motion, cond_dict[skel_b], cg_b)
                    cmp_adv = compare_q(q_adv, q_gt, cg_b, cg_b)
                    cmp_adv['pair_id'] = pid
                    results_by_method['adversarial'].append(cmp_adv)
        except Exception as e:
            pass

        # Random baseline
        try:
            rand_pool = tgt_clip_cache.get(skel_b, [])
            if rand_pool:
                rand_fname = rand_pool[rng_adv.randint(0, len(rand_pool))]
                rand_motion = np.load(motion_dir / rand_fname).astype(np.float32)
                q_rand = extract_q_from_array(rand_motion, cond_dict[skel_b], cg_b)
                cmp_rand = compare_q(q_rand, q_gt, cg_b, cg_b)
                cmp_rand['pair_id'] = pid
                results_by_method['random'].append(cmp_rand)
        except Exception as e:
            pass

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(manifest['pairs']) - i - 1)
            print(f"  [{i+1}/{len(manifest['pairs'])}] elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    print(f"\n{'='*70}")
    print(f"PRINCIPLED CROSS-SKELETON METRICS (Q-component comparison)")
    print(f"{'='*70}")
    print(f"\n{'Method':18s} {'COM↓':>14s} {'HV↓':>14s} {'Cad↓':>10s} {'Limb↓':>12s} {'CSF1↑':>14s}")
    summary = {}
    for method in ['k_retrieve', 'k_frame_nn', 'k_dtw_align', 'adversarial', 'random']:
        if method not in results_by_method:
            continue
        rs = results_by_method[method]
        com = bootstrap_ci([r['com_rel_l2'] for r in rs])
        hv = bootstrap_ci([r['hv_rel_l2'] for r in rs])
        cad = bootstrap_ci([r['cadence_abs'] for r in rs])
        lu = bootstrap_ci([r['limb_usage_l2'] for r in rs])
        cf1 = bootstrap_ci([r['contact_sched_f1'] for r in rs])
        summary[method] = {'n': len(rs), 'com': com, 'hv': hv, 'cad': cad, 'lu': lu, 'cf1': cf1}
        print(f"{method:18s} {com[1]:>13.3f} {hv[1]:>13.3f} {cad[1]:>9.3f} {lu[1]:>11.3f} {cf1[1]:>13.3f}")

    # Paired wins: K_retrieve vs adversarial (key test)
    print(f"\n--- PAIRED: K_retrieve vs adversarial (KEY semantic discrimination test) ---")
    pid_to_adv = {r['pair_id']: r for r in results_by_method.get('adversarial', [])}
    for method in ['k_retrieve', 'k_frame_nn', 'k_dtw_align']:
        if method not in results_by_method:
            continue
        method_pairs = []
        for r in results_by_method[method]:
            pid = r['pair_id']
            if pid in pid_to_adv:
                method_pairs.append((r, pid_to_adv[pid]))
        if not method_pairs:
            continue
        print(f"\n  [{method}] (n={len(method_pairs)}):")
        for metric_name, lower in [('com_rel_l2', True), ('hv_rel_l2', True), ('cadence_abs', True),
                                    ('limb_usage_l2', True), ('contact_sched_f1', False)]:
            a_vals = [pair[0][metric_name] for pair in method_pairs]
            b_vals = [pair[1][metric_name] for pair in method_pairs]
            wr = win_rate_ci(a_vals, b_vals, lower_is_better=lower)
            d = '↓' if lower else '↑'
            print(f"    {metric_name:18s} ({d}): win={wr[1]*100:.1f}% [{wr[0]*100:.1f}, {wr[2]*100:.1f}]")

    out = PROJECT_ROOT / 'eval/results/k_compare/principled_metrics.json'
    with open(out, 'w') as f:
        json.dump({'summary': summary, 'per_pair': dict(results_by_method)}, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
