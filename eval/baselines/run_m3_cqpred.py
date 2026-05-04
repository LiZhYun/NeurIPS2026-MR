"""M3_CqPred — predicted-target-cluster variant (Round 7 ablation).

Codex Round 7 requested this ablation: "If target clip clusters are inferred
from the Q-based classifier rather than read from filenames, does M3 stay
close to M3_Cq?"

Difference from run_m3_physics_optim.py:
- M3_Cq / M3_CqA use `clip_index['index'][skel_b][cluster]` — filename-derived
  cluster for each target candidate.
- M3_CqPred uses the I-5 classifier to predict each target clip's cluster
  from its Q-features. NO target-side filename cluster labels used.
- Classifier is STILL trained from filename-derived labels (this is a common
  setup — label-free at inference over target library, label-supervised at
  training time).
- Q-signatures still come from the cache for rerank (Q-features are motion
  geometry, no labels).

Supervision level: **Tier 1.5** — source classifier trained on filename labels
(same as M3_Cq), but target-side cluster lookup replaced with classifier
inference. Target library needs no metadata.

Usage:
  python -m eval.baselines.run_m3_cqpred --fold 42 --out_tag m3_cqpred
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
from eval.baselines.run_m3_physics_optim import (
    build_q_sig_table, featurize_q_30d, Q_CACHE_PATH, CLIP_INDEX_PATH,
    QUERIES_V5_DIR, SAVE_ROOT,
)
from eval.baselines.run_i5_action_classifier_v3 import (
    train_classifier, CLUSTERS,
)
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT

MOTION_DIR = Path(DATASET_DIR) / 'motions'


def rerank_one_query_cqpred(q_manifest, clf, clip_index, qsig_table, qc,
                            fname_to_idx, pred_cluster_target,
                            weights, topk_q=10):
    """Same as rerank_one_query but cluster filter uses classifier-predicted
    target cluster rather than clip_index filename cluster.

    pred_cluster_target: {fname: cluster_string} — pre-computed via classifier.
    """
    skel_b = q_manifest['skel_b']
    src_fname = q_manifest['src_fname']
    src_action_raw = q_manifest['src_action']

    src_idx = fname_to_idx.get(src_fname)
    if src_idx is None:
        raise RuntimeError(f'source {src_fname} not in Q cache')
    src_feat_30d = featurize_q_30d({
        'com_path': qc['com_path'][src_idx],
        'heading_vel': qc['heading_vel'][src_idx],
        'contact_sched': qc['contact_sched'][src_idx],
        'cadence': float(qc['cadence'][src_idx]),
        'limb_usage': qc['limb_usage'][src_idx],
    })
    pred_cluster_idx = int(clf.predict(src_feat_30d[None, :])[0])
    pred_cluster_src = CLUSTERS[pred_cluster_idx]

    src_sig = qsig_table[src_fname]
    src_sig_norm = src_sig / (np.linalg.norm(src_sig) + 1e-9)

    all_skel_b_clips = []
    for cluster_filename, clips in clip_index['index'][skel_b].items():
        for c in clips:
            fname = c['fname']
            cand = dict(c)
            # OVERRIDE: use classifier-predicted cluster for target clip
            pc = pred_cluster_target.get(fname, None)
            if pc is None:
                cand['cluster'] = cluster_filename  # fallback if not in cache
                cand['cluster_source'] = 'filename_fallback'
            else:
                cand['cluster'] = pc
                cand['cluster_source'] = 'predicted'
            cand['cluster_filename'] = cluster_filename
            all_skel_b_clips.append(cand)

    pool_by_fname = {}
    for c in all_skel_b_clips:
        if c['cluster'] == pred_cluster_src:
            pool_by_fname[c['fname']] = c

    q_ranked = []
    for c in all_skel_b_clips:
        if c['fname'] not in qsig_table:
            continue
        cand_sig = qsig_table[c['fname']]
        cand_norm = cand_sig / (np.linalg.norm(cand_sig) + 1e-9)
        q_sim = float(src_sig_norm @ cand_norm)
        q_ranked.append((q_sim, c))
    q_ranked.sort(key=lambda x: -x[0])
    for q_sim, c in q_ranked[:topk_q]:
        pool_by_fname[c['fname']] = c

    if not pool_by_fname:
        for c in all_skel_b_clips:
            pool_by_fname[c['fname']] = c

    cand_records = []
    for fname, cand in pool_by_fname.items():
        if fname not in qsig_table:
            continue
        cand_sig = qsig_table[fname]
        cand_norm = cand_sig / (np.linalg.norm(cand_sig) + 1e-9)
        q_sim = float(src_sig_norm @ cand_norm)
        action_match = 1.0 if cand['action'] == src_action_raw else 0.0
        cluster_match = 1.0 if cand['cluster'] == pred_cluster_src else 0.0
        composite = (weights['cluster'] * cluster_match
                     + weights['q'] * q_sim
                     + weights['action'] * action_match)
        cand_records.append({
            'fname': fname,
            'action': cand['action'],
            'cluster_pred': cand['cluster'],
            'cluster_filename': cand.get('cluster_filename'),
            'cluster_source': cand.get('cluster_source'),
            'q_sim': q_sim,
            'action_match': action_match,
            'cluster_match': cluster_match,
            'composite': composite,
        })
    if not cand_records:
        raise RuntimeError(f'No scored candidates for {src_fname} -> {skel_b}')

    cand_records.sort(key=lambda r: -r['composite'])
    best = cand_records[0]
    return {
        'picked_fname': best['fname'],
        'picked_composite': best['composite'],
        'picked_q_sim': best['q_sim'],
        'picked_action': best['action'],
        'picked_cluster_pred': best['cluster_pred'],
        'picked_cluster_filename': best['cluster_filename'],
        'cluster_agreement': best['cluster_pred'] == best.get('cluster_filename'),
        'picked_cluster_match': best['cluster_match'],
        'pred_cluster_src': pred_cluster_src,
        'n_candidates': len(cand_records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='Override --fold with multiple folds')
    parser.add_argument('--out_tag', type=str, default='m3_cqpred')
    parser.add_argument('--w_cluster', type=float, default=1.0)
    parser.add_argument('--w_q', type=float, default=1.0)
    parser.add_argument('--w_action', type=float, default=0.0,
                        help='0 for M3_CqPred (tier-1.5 no action); set to 3.0 for M3_CqApred')
    parser.add_argument('--topk_q', type=int, default=None)
    args = parser.parse_args()

    folds = args.folds if args.folds else [args.fold]
    weights = {'cluster': args.w_cluster, 'q': args.w_q, 'action': args.w_action}

    print(f"Loading Q cache: {Q_CACHE_PATH}")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}

    print("Training I-5 cluster classifier (filename-derived labels; same as M3_Cq)...")
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels)

    print(f"Loading clip index: {CLIP_INDEX_PATH}")
    clip_index = json.load(open(CLIP_INDEX_PATH))

    print("Building Q signature table...")
    qsig_table = build_q_sig_table(qc, None)

    # PRE-COMPUTE predicted cluster for ALL clips in Q cache (this is the key change)
    print("Pre-computing classifier-predicted clusters for all clips...")
    pred_cluster_target = {}
    mismatch_count = 0
    total_count = 0
    # Build a reverse map: fname -> filename cluster
    fname_to_fcluster = {}
    for skel, by_cluster in clip_index['index'].items():
        for fcluster, clips in by_cluster.items():
            for c in clips:
                fname_to_fcluster[c['fname']] = fcluster
    for fname, idx in fname_to_idx.items():
        feat = featurize_q_30d({
            'com_path': qc['com_path'][idx],
            'heading_vel': qc['heading_vel'][idx],
            'contact_sched': qc['contact_sched'][idx],
            'cadence': float(qc['cadence'][idx]),
            'limb_usage': qc['limb_usage'][idx],
        })
        pc_idx = int(clf.predict(feat[None, :])[0])
        pc = CLUSTERS[pc_idx]
        pred_cluster_target[fname] = pc
        fc = fname_to_fcluster.get(fname)
        if fc is not None:
            total_count += 1
            if pc != fc:
                mismatch_count += 1
    print(f"  Classifier predicts cluster on {len(pred_cluster_target)} clips; "
          f"pred vs filename agreement: {1-mismatch_count/max(total_count,1):.3f} "
          f"({total_count - mismatch_count}/{total_count})")

    for fold_seed in folds:
        manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold_seed}/manifest.json'))
        out_dir = SAVE_ROOT / args.out_tag / f'fold_{fold_seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== fold {fold_seed} → {out_dir} ===")

        n_done = n_failed = 0
        per_query = []
        t0 = time.time()
        for i, q in enumerate(manifest['queries']):
            qid = q['query_id']
            rec = {'query_id': qid, 'cluster': q['cluster'], 'split': q['split'],
                   'skel_a': q['skel_a'], 'skel_b': q['skel_b'],
                   'src_action': q['src_action'], 'status': 'pending'}
            try:
                r = rerank_one_query_cqpred(q, clf, clip_index, qsig_table, qc,
                                            fname_to_idx, pred_cluster_target,
                                            weights, topk_q=args.topk_q or 10)
                picked_fname = r['picked_fname']
                motion = np.load(MOTION_DIR / picked_fname).astype(np.float32)
                np.save(out_dir / f'query_{qid:04d}.npy', motion)
                rec.update({'status': 'ok', **{k: v for k, v in r.items() if k != 'picked_fname'},
                            'picked_fname': picked_fname})
                n_done += 1
            except Exception as e:
                rec['status'] = 'failed'
                rec['error'] = f'{type(e).__name__}: {e}'
                n_failed += 1
            per_query.append(rec)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(manifest['queries'])}] ok={n_done} failed={n_failed} "
                      f"({time.time()-t0:.0f}s)")
        print(f"fold {fold_seed}: ok={n_done}, failed={n_failed}, "
              f"total_time={time.time()-t0:.0f}s")
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump({'method': 'M3_CqPred', 'fold': fold_seed, 'weights': weights,
                       'n_queries': len(per_query), 'n_ok': n_done, 'n_failed': n_failed,
                       'classifier_target_pred_agreement': 1 - mismatch_count/max(total_count,1),
                       'per_query': per_query}, f, indent=2)


if __name__ == '__main__':
    main()
