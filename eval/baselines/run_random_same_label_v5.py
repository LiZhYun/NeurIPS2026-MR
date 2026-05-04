"""Random-same-label retrieval baselines.

Two trivial label-aware baselines that ablate ANCHOR's descriptor contribution:
- random_same_cluster: pick a uniformly random clip on skel_b whose filename cluster
  matches the predicted source cluster (from I-5 RandomForest).
- random_same_exact_action: pick a uniformly random clip on skel_b whose filename
  exact-action matches the source's exact-action label.

If these baselines achieve V5 cluster-tier AUC near ANCHOR's, the win is metadata
alone. If they're significantly lower, the Q-descriptor and rerank carry real signal.

Usage:
  python -m eval.baselines.run_random_same_label_v5 --baseline random_same_cluster --folds 42 43
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
    Q_CACHE_PATH, CLIP_INDEX_PATH,
)
from eval.baselines.run_i5_action_classifier_v3 import (
    train_classifier, CLUSTER_TO_IDX as I5_CLUSTER_TO_IDX, featurize_q,
)

QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
MOTION_DIR = Path(DATASET_DIR) / 'motions'
SAVE_ROOT = PROJECT_ROOT / 'eval/results/baselines'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=['random_same_cluster', 'random_same_exact_action'],
                        required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--manifest', type=str, default=None,
                        help='Custom manifest (else V5 fold_X)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_tag', type=str, default=None,
                        help='Output tag (default = baseline name)')
    args = parser.parse_args()

    out_tag = args.out_tag or args.baseline
    rng = np.random.RandomState(args.seed)

    # Load Q cache + train classifier (needed for cluster-pred baseline)
    print("Loading Q cache + classifier + clip index...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_qc_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels)
    I5_IDX_TO_CLUSTER = {v: k for k, v in I5_CLUSTER_TO_IDX.items()}
    clip_index = json.load(open(CLIP_INDEX_PATH))
    clip_idx = clip_index['index']
    print(f"  {len(clip_idx)} skeletons indexed")

    folds_to_process = args.folds
    if args.manifest:
        # Single fold = 999, single manifest
        folds_to_process = [999]

    for fold in folds_to_process:
        if args.manifest:
            manifest_path = Path(args.manifest)
        else:
            manifest_path = QUERIES_V5_DIR / f'fold_{fold}/manifest.json'
        manifest = json.load(open(manifest_path))
        out_dir = SAVE_ROOT / out_tag / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)

        per_query = []
        n_ok = n_fail = 0
        t0 = time.time()
        for q in manifest['queries']:
            qid = q['query_id']
            skel_b = q['skel_b']
            src_action = q.get('src_action', '')
            src_fname = q.get('src_fname', '')
            rec = {'query_id': qid, 'skel_a': q.get('skel_a'), 'skel_b': skel_b,
                   'src_action': src_action, 'cluster': q.get('cluster'),
                   'split': q.get('split'), 'status': 'pending'}

            try:
                if args.baseline == 'random_same_cluster':
                    if src_fname not in fname_to_qc_idx:
                        rec['status'] = 'skipped_no_qc'
                        per_query.append(rec)
                        continue
                    qi = fname_to_qc_idx[src_fname]
                    feat = featurize_q(qc['com_path'][qi], qc['heading_vel'][qi],
                                       qc['contact_sched'][qi], qc['cadence'][qi],
                                       qc['limb_usage'][qi])
                    pred_cluster_idx = int(clf.predict(feat.reshape(1, -1))[0])
                    pred_cluster = I5_IDX_TO_CLUSTER.get(pred_cluster_idx, '')
                    candidates = clip_idx.get(skel_b, {}).get(pred_cluster, [])
                elif args.baseline == 'random_same_exact_action':
                    # Iterate all clusters' clips on skel_b, filter by action
                    candidates = []
                    for cluster_key, clips in clip_idx.get(skel_b, {}).items():
                        for c in clips:
                            if c.get('action') == src_action:
                                candidates.append(c)
                else:
                    raise ValueError(args.baseline)

                if not candidates:
                    rec['status'] = 'skipped_no_candidates'
                    per_query.append(rec)
                    continue

                pick = candidates[rng.choice(len(candidates))]
                motion = np.load(MOTION_DIR / pick['fname']).astype(np.float32)
                # Crop to median target T from positives_cluster (V5 protocol)
                pos_T = [p['T'] for p in q.get('positives_cluster', [])]
                T_tgt = int(np.median(pos_T)) if pos_T else q.get('src_T', motion.shape[0])
                T_out = min(T_tgt, motion.shape[0])
                motion = motion[:T_out]

                np.save(out_dir / f'query_{qid:04d}.npy', motion)
                rec.update({'status': 'ok', 'picked_fname': pick['fname'],
                            'picked_action': pick.get('action'),
                            'picked_cluster': pick.get('cluster')})
                n_ok += 1
            except Exception as e:
                rec['status'] = f'error: {e}'
                n_fail += 1
            per_query.append(rec)

        summary = {
            'method': out_tag, 'fold': fold, 'manifest': str(manifest_path),
            'n_queries': len(manifest['queries']),
            'n_ok': n_ok, 'n_failed': n_fail,
            'wall_clock_s': time.time() - t0,
            'per_query': per_query,
        }
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Fold {fold}: ok={n_ok}, fail={n_fail}, wall_clock={time.time()-t0:.0f}s")
        print(f"  saved: {out_dir}/metrics.json")


if __name__ == '__main__':
    main()
