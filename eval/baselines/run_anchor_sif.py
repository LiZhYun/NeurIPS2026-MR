"""Run ANCHOR (m3_rerank_v1) on the SIF manifest.

Forks process_fold from run_m3_physics_optim.py to use a custom manifest.
Saves retrieved clip motion as query_NNNN.npy for SIF metric computation.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from eval.baselines.run_m3_physics_optim import (
    rerank_one_query, build_q_sig_table, Q_CACHE_PATH, CLIP_INDEX_PATH,
)
from eval.baselines.run_i5_action_classifier_v3 import train_classifier

MOTION_DIR = Path(DATASET_DIR) / 'motions'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='eval/benchmark_v3/queries_sif/manifest.json')
    parser.add_argument('--out_dir', type=str,
                        default='eval/results/baselines/m3_rerank_v1_sif')
    parser.add_argument('--w_cluster', type=float, default=1.0)
    parser.add_argument('--w_q', type=float, default=2.0)
    parser.add_argument('--w_action', type=float, default=3.0)
    parser.add_argument('--topk_q', type=int, default=10)
    args = parser.parse_args()

    weights = {'cluster': args.w_cluster, 'q': args.w_q, 'action': args.w_action}
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Q cache + classifier + clip index...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    qsig_table = build_q_sig_table(qc, None)
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels)
    clip_index = json.load(open(CLIP_INDEX_PATH))

    manifest = json.load(open(PROJECT_ROOT / args.manifest))
    queries = manifest['queries']
    print(f"Manifest: {len(queries)} queries")

    per_query = []
    t0 = time.time()
    n_ok = n_fail = 0
    for i, q in enumerate(queries):
        qid = q['query_id']
        rec = {'query_id': qid, 'skel_a': q['skel_a'], 'skel_b': q['skel_b'],
               'src_action': q['src_action'], 'src_fname': q['src_fname'],
               'status': 'pending'}
        try:
            r = rerank_one_query(q, clf, clip_index, qsig_table, qc,
                                 fname_to_idx, weights, topk_q=args.topk_q)
            picked = r['picked_fname']
            motion = np.load(MOTION_DIR / picked).astype(np.float32)
            np.save(out_dir / f'query_{qid:04d}.npy', motion)
            rec.update({'status': 'ok', 'picked_fname': picked,
                        'picked_action': r['picked_action'],
                        'picked_cluster': r['picked_cluster']})
            n_ok += 1
        except Exception as e:
            rec['status'] = f'error: {e}'
            n_fail += 1
        per_query.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(queries)}] ok={n_ok} fail={n_fail} ({time.time()-t0:.0f}s)")

    summary = {'method': 'm3_rerank_v1_sif', 'manifest': args.manifest,
               'n_queries': len(queries), 'n_ok': n_ok, 'n_failed': n_fail,
               'wall_clock_s': time.time() - t0, 'per_query': per_query}
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_dir}/metrics.json")
    print(f"  ok={n_ok}, fail={n_fail}, wall_clock={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
