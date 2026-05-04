"""Run M3_CqA / M3_Cq / M3_CqPred / k_retrieve / i5 / action_oracle on V5-full.

V5-full has 30k+ queries vs the V5 sampler's 300/fold. This script reuses the
existing rerank_one_query / rerank_one_query_cqpred logic but iterates the V5-full
manifest and writes outputs to a flat directory.

Usage:
    conda run -n anytop python -m eval.baselines.run_m3_v5full \
        --variant M3_CqA --out_tag m3_cqa_v5full
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

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR, OBJECT_SUBSETS_DICT
from eval.baselines.run_m3_physics_optim import (
    rerank_one_query, build_q_sig_table, Q_CACHE_PATH, CLIP_INDEX_PATH,
)
from eval.baselines.run_i5_action_classifier_v3 import train_classifier, CLUSTERS
from eval.baselines.run_m3_cqpred import rerank_one_query_cqpred, featurize_q_30d

MOTION_DIR = Path(DATASET_DIR) / 'motions'
MANIFEST = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5_full/manifest.json'
SAVE_ROOT = PROJECT_ROOT / 'save'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True,
                        choices=['M3_CqA', 'M3_Cq', 'M3_CqPred'])
    parser.add_argument('--out_tag', type=str, required=True)
    parser.add_argument('--max_queries', type=int, default=0,
                        help='0 = all queries')
    args = parser.parse_args()

    weights_by_variant = {
        'M3_CqA':    {'cluster': 1.0, 'q': 2.0, 'action': 3.0},
        'M3_Cq':     {'cluster': 1.0, 'q': 1.0, 'action': 0.0},
        'M3_CqPred': {'cluster': 1.0, 'q': 1.0, 'action': 0.0},
    }
    weights = weights_by_variant[args.variant]
    print(f'Variant: {args.variant}, weights: {weights}')

    print('Loading Q cache...')
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}

    print('Building Q sig table...')
    qsig_table = build_q_sig_table(qc, None)

    print('Training I-5 classifier...')
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels)

    print('Loading clip index...')
    clip_index = json.load(open(CLIP_INDEX_PATH))

    # For M3_CqPred, pre-compute per-target-clip predicted clusters
    pred_cluster_target = None
    if args.variant == 'M3_CqPred':
        print('Pre-computing predicted clusters for all clips...')
        pred_cluster_target = {}
        for fname, idx in fname_to_idx.items():
            feat = featurize_q_30d({
                'com_path': qc['com_path'][idx],
                'heading_vel': qc['heading_vel'][idx],
                'contact_sched': qc['contact_sched'][idx],
                'cadence': float(qc['cadence'][idx]),
                'limb_usage': qc['limb_usage'][idx],
            })
            pc_idx = int(clf.predict(feat[None, :])[0])
            pred_cluster_target[fname] = CLUSTERS[pc_idx]
        print(f'  predicted clusters cached for {len(pred_cluster_target)} clips')

    print(f'Loading manifest: {MANIFEST}')
    manifest = json.load(open(MANIFEST))
    queries = manifest['queries']
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    print(f'  {len(queries)} queries to process')

    out_dir = SAVE_ROOT / 'm3_v5full' / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query = []
    t0 = time.time()
    n_done = n_failed = 0
    for i, q in enumerate(queries):
        qid = q['query_id']
        rec = {'query_id': qid, 'split': q['split'], 'cluster': q['cluster'],
               'skel_a': q['skel_a'], 'skel_b': q['skel_b'],
               'src_action': q['src_action'], 'status': 'pending'}
        try:
            if args.variant == 'M3_CqPred':
                r = rerank_one_query_cqpred(q, clf, clip_index, qsig_table, qc,
                                            fname_to_idx, pred_cluster_target,
                                            weights, topk_q=10)
            else:
                r = rerank_one_query(q, clf, clip_index, qsig_table, qc,
                                     fname_to_idx, weights, topk_q=10)
            picked_fname = r['picked_fname']
            motion = np.load(MOTION_DIR / picked_fname).astype(np.float32)
            np.save(out_dir / f'query_{qid:05d}.npy', motion)
            rec.update({'status': 'ok', **{k: v for k, v in r.items() if k != 'picked_fname'},
                        'picked_fname': picked_fname})
            n_done += 1
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = f'{type(e).__name__}: {e}'
            n_failed += 1
        per_query.append(rec)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f'  [{i+1}/{len(queries)}] ok={n_done} failed={n_failed} elapsed={elapsed:.0f}s eta={eta:.0f}s')

    total = time.time() - t0
    print(f'Done: {n_done} ok, {n_failed} failed, {total:.0f}s total')
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'method': args.variant, 'n_queries': len(per_query),
                   'n_ok': n_done, 'n_failed': n_failed, 'wall_clock_s': total,
                   'per_query': per_query}, f, indent=1)


if __name__ == '__main__':
    main()
