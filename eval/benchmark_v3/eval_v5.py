"""Evaluator v5 — tier-separated, per-cluster, block-bootstrap.

Per refine-logs/BENCHMARK_V5_DESIGN.md (v5.1).

Computes, per query:
  - cluster-tier AUC: ranks positives_cluster vs adversarials_easy
  - exact-tier AUC:   ranks positives_exact vs adversarials_hard (only if exact_tier_eligible)
  - distractor metrics: same-target-skel open retrieval

Reports:
  - per-tier × per-cluster × per-stratum AUC + block bootstrap CI (block by skel_pair)
  - Overall macro-AUC over support-qualified clusters (≥20 test_test queries + ≥3 target skels)
  - Overall micro-AUC over all support-qualified queries (fallback)

Usage:
  python -m eval.benchmark_v3.eval_v5 --method_dir <dir> --fold 42 --method_name X --distance procrustes
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score
from data_loaders.truebones.truebones_utils.param_utils import (
    OBJECT_SUBSETS_DICT,
)
from eval.benchmark_v3.metrics_v3 import (
    dist_zscore_dtw_inv, dist_procrustes_trajectory, q_component_distances,
)
from model.skel_blind.encoder import encode_motion_to_invariant
from eval.benchmark_v3.eval_v3 import (
    make_skel_cond, _recover_positions, _q_from_motion, _zscore_q_pool,
    Q_CACHE_PATH,
)
from eval.moreflow_phi import load_contact_groups
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
QUERIES_ROOT = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'


def auc_safe(pos_dists, neg_dists):
    if not pos_dists or not neg_dists:
        return None
    scores = -np.concatenate([pos_dists, neg_dists])
    labels = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    try: return float(roc_auc_score(labels, scores))
    except Exception: return None


def block_bootstrap_ci(values_with_blocks, n_boot=500, ci=0.95, seed=42):
    """values_with_blocks: list of (value, block_id). Bootstrap by block, mean of values."""
    if not values_with_blocks: return (0.0, 0.0, 0.0)
    rng = np.random.RandomState(seed)
    block_to_values = defaultdict(list)
    for v, b in values_with_blocks:
        if v is None: continue
        block_to_values[b].append(v)
    blocks = list(block_to_values.keys())
    if not blocks: return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sampled_blocks = rng.choice(len(blocks), len(blocks), replace=True)
        all_vals = []
        for bi in sampled_blocks:
            all_vals.extend(block_to_values[blocks[bi]])
        means.append(np.mean(all_vals) if all_vals else 0.0)
    means = np.array(means)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    raw_mean = float(np.mean([v for v, _ in values_with_blocks if v is not None]))
    return (lo, raw_mean, hi)


def _compute_distance(distance_metric, pos_pred, inv_pred, q_pred,
                      ref_motion, skel_b, cond_dict, skel_cond,
                      contact_groups, get_q, ref_fname):
    if distance_metric == 'procrustes':
        pos_ref = _recover_positions(ref_motion, skel_b, cond_dict)
        return dist_procrustes_trajectory(pos_pred, pos_ref)
    elif distance_metric == 'zscore_dtw':
        inv_ref = encode_motion_to_invariant(ref_motion, skel_cond)
        return dist_zscore_dtw_inv(inv_pred, inv_ref)
    elif distance_metric == 'q_component':
        q_ref = get_q(ref_fname) or _q_from_motion(ref_motion, skel_b, cond_dict, contact_groups)
        return q_component_distances(q_pred, q_ref)
    raise ValueError(distance_metric)


def load_q_cache():
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    def get_q(fname):
        i = fname_to_idx.get(fname)
        if i is None: return None
        return {
            'com_path': qc['com_path'][i], 'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]), 'limb_usage': qc['limb_usage'][i],
        }
    return get_q


def evaluate_method(method_dir: Path, fold_seed: int, method_name: str,
                    distance: str, max_queries: int = 10000):
    manifest = json.load(open(QUERIES_ROOT / f'fold_{fold_seed}/manifest.json'))
    queries = manifest['queries']
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()
    contact_groups = load_contact_groups()
    get_q = load_q_cache()

    per_query = []
    n_skipped = 0
    for q in queries[:max_queries]:
        qid = q['query_id']
        skel_b = q['skel_b']
        pred_path = method_dir / f'query_{qid:04d}.npy'
        if not pred_path.exists():
            n_skipped += 1; continue
        try:
            pred = np.load(pred_path)
            skel_cond = make_skel_cond(cond_dict, skel_b)
            is_pos_only = pred.ndim == 3 and pred.shape[-1] == 3
            if is_pos_only and distance in ('zscore_dtw', 'q_component'):
                n_skipped += 1; continue
            if not is_pos_only:
                inv_pred = encode_motion_to_invariant(pred.astype(np.float32), skel_cond)
                q_pred = (_q_from_motion(pred, skel_b, cond_dict, contact_groups)
                         if distance == 'q_component' else None)
            else:
                inv_pred = None; q_pred = None
            pos_pred = _recover_positions(pred, skel_b, cond_dict)

            args_for_dist = dict(distance_metric=distance, pos_pred=pos_pred,
                                 inv_pred=inv_pred, q_pred=q_pred, skel_b=skel_b,
                                 cond_dict=cond_dict, skel_cond=skel_cond,
                                 contact_groups=contact_groups, get_q=get_q)

            # Cluster-tier: positives_cluster vs adversarials_easy
            pos_c_dists, neg_c_dists = [], []
            for p in q['positives_cluster']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_c_dists.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'],
                                                     **args_for_dist))
            for a in q['adversarials_easy']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                neg_c_dists.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'],
                                                     **args_for_dist))

            # Exact-tier: positives_exact vs adversarials_hard
            pos_e_dists, neg_e_dists = [], []
            for p in q['positives_exact']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_e_dists.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'],
                                                     **args_for_dist))
            for a in q['adversarials_hard']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                neg_e_dists.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'],
                                                     **args_for_dist))

            # Q-component z-scoring (per-query within candidate pool)
            if distance == 'q_component':
                pool_c = pos_c_dists + neg_c_dists
                if pool_c:
                    z_c = _zscore_q_pool(pool_c)
                    pos_c_dists = z_c[:len(pos_c_dists)]
                    neg_c_dists = z_c[len(pos_c_dists):]
                pool_e = pos_e_dists + neg_e_dists
                if pool_e:
                    z_e = _zscore_q_pool(pool_e)
                    pos_e_dists = z_e[:len(pos_e_dists)]
                    neg_e_dists = z_e[len(pos_e_dists):]

            cluster_auc = auc_safe(pos_c_dists, neg_c_dists) if q['cluster_tier_eligible'] else None
            exact_auc = auc_safe(pos_e_dists, neg_e_dists) if q['exact_tier_eligible'] else None

            per_query.append({
                'query_id': qid,
                'split': q['split'],
                'cluster': q['cluster'],
                'skel_a': q['skel_a'], 'skel_b': skel_b,
                'support_reason': q['support_reason'],
                'n_pos_cluster': q['n_pos_cluster'], 'n_pos_exact': q['n_pos_exact'],
                'n_easy': q['n_easy'], 'n_hard': q['n_hard'],
                'cluster_tier_eligible': q['cluster_tier_eligible'],
                'exact_tier_eligible': q['exact_tier_eligible'],
                'cluster_auc': cluster_auc,
                'exact_auc': exact_auc,
                'block_id': f"{q['skel_a']}__{q['skel_b']}",
            })
        except Exception as e:
            print(f"  q{qid}: FAIL {e}")
            n_skipped += 1

    # Aggregate
    out = {
        'method': method_name, 'distance': distance, 'fold': fold_seed,
        'n_queries': len(per_query), 'n_skipped': n_skipped,
        'per_query': per_query,
    }
    # Cluster-tier overall
    cluster_blocks = [(r['cluster_auc'], r['block_id']) for r in per_query if r['cluster_auc'] is not None]
    cluster_blocks_tt = [(r['cluster_auc'], r['block_id']) for r in per_query
                         if r['cluster_auc'] is not None and r['split'] == 'test_test']
    out['cluster_tier_overall_auc_ci'] = block_bootstrap_ci(cluster_blocks)
    out['cluster_tier_test_test_auc_ci'] = block_bootstrap_ci(cluster_blocks_tt)
    out['cluster_tier_n'] = len(cluster_blocks)
    out['cluster_tier_test_test_n'] = len(cluster_blocks_tt)

    # Exact-tier overall
    exact_blocks = [(r['exact_auc'], r['block_id']) for r in per_query if r['exact_auc'] is not None]
    exact_blocks_tt = [(r['exact_auc'], r['block_id']) for r in per_query
                       if r['exact_auc'] is not None and r['split'] == 'test_test']
    out['exact_tier_overall_auc_ci'] = block_bootstrap_ci(exact_blocks)
    out['exact_tier_test_test_auc_ci'] = block_bootstrap_ci(exact_blocks_tt)
    out['exact_tier_n'] = len(exact_blocks)
    out['exact_tier_test_test_n'] = len(exact_blocks_tt)

    # Per-cluster (cluster-tier, test_test)
    by_cluster_tt = defaultdict(list)
    for r in per_query:
        if r['cluster_auc'] is not None and r['split'] == 'test_test':
            by_cluster_tt[r['cluster']].append((r['cluster_auc'], r['block_id']))
    out['per_cluster_test_test_cluster_tier'] = {
        c: {'auc_ci': block_bootstrap_ci(v), 'n': len(v)}
        for c, v in by_cluster_tt.items()
    }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_dir', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--method_name', required=True)
    parser.add_argument('--distance', choices=['procrustes', 'zscore_dtw', 'q_component'],
                        default='procrustes')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    out_dir = Path(args.out_dir) if args.out_dir else method_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_method(method_dir, args.fold, args.method_name, args.distance)
    out_path = out_dir / f'v5_eval_{args.distance}.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Quick print
    cl = summary['cluster_tier_overall_auc_ci']
    cl_tt = summary['cluster_tier_test_test_auc_ci']
    ex = summary['exact_tier_overall_auc_ci']
    ex_tt = summary['exact_tier_test_test_auc_ci']
    print(f"\n=== {args.method_name} fold {args.fold} {args.distance} ===")
    print(f"  cluster-tier overall:   {cl[1]:.3f} [{cl[0]:.3f}, {cl[2]:.3f}] (n={summary['cluster_tier_n']})")
    print(f"  cluster-tier test_test: {cl_tt[1]:.3f} [{cl_tt[0]:.3f}, {cl_tt[2]:.3f}] (n={summary['cluster_tier_test_test_n']})")
    print(f"  exact-tier   overall:   {ex[1]:.3f} [{ex[0]:.3f}, {ex[2]:.3f}] (n={summary['exact_tier_n']})")
    print(f"  exact-tier   test_test: {ex_tt[1]:.3f} [{ex_tt[0]:.3f}, {ex_tt[2]:.3f}] (n={summary['exact_tier_test_test_n']})")
    print(f"  per-cluster (cluster-tier, test_test):")
    for c, info in sorted(summary['per_cluster_test_test_cluster_tier'].items(),
                          key=lambda x: -x[1]['n']):
        a = info['auc_ci']
        print(f"    {c}: {a[1]:.3f} [{a[0]:.3f}, {a[2]:.3f}] (n={info['n']})")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
