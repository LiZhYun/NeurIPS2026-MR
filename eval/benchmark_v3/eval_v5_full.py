"""V5-full evaluator: AUC over the 30k V5-full manifest queries.

Reuses eval_v5's distance + bootstrap helpers. Predictions are named
query_NNNNN.npy (5-digit zero-padded) under method_dir.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
from eval.benchmark_v3.eval_v5 import (
    auc_safe, block_bootstrap_ci, _compute_distance, _zscore_q_pool,
    _recover_positions, _q_from_motion, encode_motion_to_invariant,
    make_skel_cond, load_q_cache,
)
from eval.moreflow_phi import load_contact_groups

MANIFEST_PATH = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5_full/manifest.json'
DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_dir', required=True)
    parser.add_argument('--method_name', required=True)
    parser.add_argument('--distance', choices=['procrustes', 'zscore_dtw', 'q_component'],
                        required=True)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--max_queries', type=int, default=0)
    parser.add_argument('--splits', nargs='+', default=['test_test', 'mixed', 'train_train'])
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    out_dir = Path(args.out_dir) if args.out_dir else method_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading manifest: {MANIFEST_PATH}')
    manifest = json.load(open(MANIFEST_PATH))
    queries = manifest['queries']
    queries = [q for q in queries if q['split'] in args.splits]
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    print(f'  {len(queries)} queries (splits: {args.splits})')

    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()
    contact_groups = load_contact_groups()
    get_q = load_q_cache()

    per_query = []
    n_skipped = 0
    t0 = time.time()
    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_b = q['skel_b']
        pred_path = method_dir / f'query_{qid:05d}.npy'
        if not pred_path.exists():
            n_skipped += 1
            continue
        try:
            pred = np.load(pred_path)
            skel_cond = make_skel_cond(cond_dict, skel_b)
            is_pos_only = pred.ndim == 3 and pred.shape[-1] == 3
            if is_pos_only and args.distance in ('zscore_dtw', 'q_component'):
                n_skipped += 1; continue
            if not is_pos_only:
                inv_pred = encode_motion_to_invariant(pred.astype(np.float32), skel_cond)
                q_pred = (_q_from_motion(pred, skel_b, cond_dict, contact_groups)
                          if args.distance == 'q_component' else None)
            else:
                inv_pred = None; q_pred = None
            pos_pred = _recover_positions(pred, skel_b, cond_dict)

            args_for_dist = dict(distance_metric=args.distance, pos_pred=pos_pred,
                                 inv_pred=inv_pred, q_pred=q_pred, skel_b=skel_b,
                                 cond_dict=cond_dict, skel_cond=skel_cond,
                                 contact_groups=contact_groups, get_q=get_q)

            pos_c, neg_c = [], []
            for p in q['positives_cluster']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_c.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'], **args_for_dist))
            for a in q['adversarials_easy']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                neg_c.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'], **args_for_dist))

            pos_e, neg_e = [], []
            for p in q['positives_exact']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_e.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'], **args_for_dist))
            for a in q['adversarials_hard']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                neg_e.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'], **args_for_dist))

            if args.distance == 'q_component':
                pool_c = pos_c + neg_c
                if pool_c:
                    z_c = _zscore_q_pool(pool_c)
                    pos_c = z_c[:len(pos_c)]; neg_c = z_c[len(pos_c):]
                pool_e = pos_e + neg_e
                if pool_e:
                    z_e = _zscore_q_pool(pool_e)
                    pos_e = z_e[:len(pos_e)]; neg_e = z_e[len(pos_e):]

            cluster_auc = auc_safe(pos_c, neg_c) if q['cluster_tier_eligible'] else None
            exact_auc = auc_safe(pos_e, neg_e) if q['exact_tier_eligible'] else None

            per_query.append({
                'query_id': qid, 'split': q['split'], 'cluster': q['cluster'],
                'skel_a': q['skel_a'], 'skel_b': skel_b,
                'cluster_tier_eligible': q['cluster_tier_eligible'],
                'exact_tier_eligible': q['exact_tier_eligible'],
                'cluster_auc': cluster_auc, 'exact_auc': exact_auc,
                'block_id': f"{q['skel_a']}__{q['skel_b']}",
            })
        except Exception as e:
            n_skipped += 1
            if n_skipped < 5:
                print(f'  q{qid} fail: {type(e).__name__}: {e}')
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, i + 1) * (len(queries) - i - 1)
            print(f'  [{i+1}/{len(queries)}] processed={len(per_query)} skipped={n_skipped} elapsed={elapsed:.0f}s eta={eta:.0f}s')

    print(f'Eval done: {len(per_query)} OK, {n_skipped} skipped, {time.time()-t0:.0f}s')

    summary = {
        'method': args.method_name, 'distance': args.distance,
        'n_queries': len(queries), 'n_processed': len(per_query), 'n_skipped': n_skipped,
        'wall_clock_s': time.time() - t0,
    }

    for tier in ['cluster', 'exact']:
        for split in ['all', 'train_train', 'mixed', 'test_test']:
            if split == 'all':
                rows = [r for r in per_query if r.get(f'{tier}_auc') is not None]
            else:
                rows = [r for r in per_query if r['split'] == split and r.get(f'{tier}_auc') is not None]
            if not rows:
                continue
            ci = block_bootstrap_ci(
                [(r[f'{tier}_auc'], r['block_id']) for r in rows], n_boot=400, seed=42)
            summary[f'{tier}_tier_{split}_auc_ci'] = ci
            summary[f'{tier}_tier_{split}_n'] = len(rows)

    by_cluster = defaultdict(list)
    for r in per_query:
        if r['split'] == 'test_test' and r.get('cluster_auc') is not None:
            by_cluster[r['cluster']].append((r['cluster_auc'], r['block_id']))
    summary['per_cluster_test_test_cluster_tier'] = {}
    for c, vals in by_cluster.items():
        ci = block_bootstrap_ci(vals, n_boot=400, seed=42)
        summary['per_cluster_test_test_cluster_tier'][c] = {'auc_ci': ci, 'n': len(vals)}

    print(f'\n=== {args.method_name} V5-full {args.distance} ===')
    for tier in ['cluster', 'exact']:
        for split in ['all', 'train_train', 'mixed', 'test_test']:
            key = f'{tier}_tier_{split}_auc_ci'
            if key in summary:
                lo, mid, hi = summary[key][:3]
                n = summary[f'{tier}_tier_{split}_n']
                print(f'  {tier}-tier {split:<12} AUC: {mid:.4f} [{lo:.4f}, {hi:.4f}] (n={n})')

    out_path = out_dir / f'v5full_eval_{args.distance}.json'
    summary['per_query'] = per_query
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
