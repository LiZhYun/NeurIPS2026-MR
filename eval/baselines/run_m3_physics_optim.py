"""M3 reranker (Phase A) — cluster-pred + Q-similarity hybrid retrieval.

Per Codex 2026-04-23 radical-pivot verdict: this is the cheap reranker phase
that combines (a) I-5 cluster-classifier filtering and (b) k_retrieve
Q-similarity. The full physics-grounded test-time optimization is implemented
separately in `eval/baselines/run_m3_phaseb_motion_optim.py`.

Pipeline (Phase A — cluster-supervised retrieval over target-skel library):
  *NOT training-free*: Trains an I-5 RandomForest cluster classifier from
  filename-derived cluster labels (see `train_classifier` in
  `run_i5_action_classifier_v3.py`). Inference is then retrieval + rerank over
  the target-skel clip library, with both classifier-prediction and filename
  cluster labels involved.

  1. Train I-5 cluster classifier on train_v3 clips (filename-derived labels)
  2. Classify source → cluster_pred (I-5 RandomForest on Q-features)
  3. Build candidate pool = (clips in skel_b with filename_cluster == cluster_pred)
                          ∪ (top-topk_q clips in skel_b by Q-similarity)
     Note: filename_cluster is read from clip_index — this uses filename-derived
     target labels. For the classifier-predicted-target-cluster ablation, see
     `run_m3_cqpred.py`.
  4. For each candidate, composite_score =
        w_cluster · I(cand.cluster == cluster_pred)
      + w_q      · cosine_sim(Q(source), Q(cand))
      + w_action · I(cand.action == src_action)
     (w_action uses the source action label from the filename. Set w_action=0
     for M3_Cq — cluster-supervised without exact-action tie-breaker.)
  5. Pick top-1 by composite → save existing target-skel motion clip

Usage:
  python -m eval.baselines.run_m3_physics_optim --folds 42 43 \
      --w_action 3.0 --w_q 2.0 --w_cluster 1.0 --out_tag m3A_rerank_full

  python -m eval.baselines.run_m3_physics_optim --folds 42 43 \
      --w_action 0.0 --w_q 1.0 --w_cluster 1.0 --out_tag m3A_rerank_noaction
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from eval.baselines.run_i5_action_classifier_v3 import (
    featurize_q, train_classifier, build_target_pool, CLUSTERS, CLUSTER_TO_IDX,
)
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
CLIP_INDEX_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
SAVE_ROOT = PROJECT_ROOT / 'save/m3'


def build_q_sig_table(qc, feat_table):
    """Return {fname: (q_signature, q_feat_30d)} for all cached clips."""
    from eval.pilot_Q_experiments import q_signature
    qsig_table = {}
    for i, m in enumerate(qc['meta']):
        q = {
            'com_path': qc['com_path'][i],
            'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        sig = q_signature(q)
        qsig_table[m['fname']] = sig
    return qsig_table


def cosine(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(a_norm @ b_norm)


def featurize_q_30d(qc_entry):
    return featurize_q(qc_entry['com_path'], qc_entry['heading_vel'],
                       qc_entry['contact_sched'], qc_entry['cadence'],
                       qc_entry['limb_usage'])


def build_q_star(q_src, skel_a, skel_b, contact_groups, cond):
    """Approximate Q*(target) from source Q via body-scale normalization.

    The source Q's contact_sched already respects contact groups; we keep it
    as-is (numeric schedule is scale-invariant). COM path + heading_vel are
    already body-scale-normalized in extract_quotient. cadence is freq-Hz.
    limb_usage is a chain-indexed distribution that doesn't directly translate
    to target skel chains.

    For M3's purposes we use q_src directly as the q_star approximation —
    matches Idea K's build_q_star behavior. Could be refined with body-part
    mapping but that's outside gate-cycle scope.
    """
    return q_src


def rerank_one_query(q_manifest, clf, clip_index, qsig_table, qc,
                     fname_to_idx, weights, topk_q=10):
    """Pick best candidate by composite (cluster + Q + action) score.

    Pool = union of (predicted-cluster clips on skel_b)
         + (top-topk_q clips on skel_b by Q-similarity to source).

    The Q-pool ensures w_cluster carries information: a candidate from the
    Q-pool that is NOT in pred_cluster gets w_cluster=0, so the score correctly
    penalizes cross-cluster picks when the classifier is confident.

    Returns: dict with 'picked_fname', composite scores, candidate records.
    """
    skel_b = q_manifest['skel_b']
    src_fname = q_manifest['src_fname']
    src_action_raw = q_manifest['src_action']

    # Featurize source & classify cluster
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
    pred_cluster = CLUSTERS[pred_cluster_idx]

    # Source Q signature
    src_sig = qsig_table[src_fname]
    src_sig_norm = src_sig / (np.linalg.norm(src_sig) + 1e-9)

    # Full skel_b library
    all_skel_b_clips = []
    for cluster, clips in clip_index['index'][skel_b].items():
        for c in clips:
            all_skel_b_clips.append({**c, 'cluster': cluster})

    # Pool = (cluster-pred filter) ∪ (top-K Q-similarity across full library)
    pool_by_fname = {}

    # 1. Cluster pool
    for c in all_skel_b_clips:
        if c['cluster'] == pred_cluster:
            pool_by_fname[c['fname']] = c

    # 2. Q pool: rank ALL skel_b clips by Q-cosine, take top topk_q
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
        # Last-resort fallback to entire library
        for c in all_skel_b_clips:
            pool_by_fname[c['fname']] = c

    # Score every candidate in the union pool
    cand_records = []
    for fname, cand in pool_by_fname.items():
        if fname not in qsig_table:
            continue
        cand_sig = qsig_table[fname]
        cand_norm = cand_sig / (np.linalg.norm(cand_sig) + 1e-9)
        q_sim = float(src_sig_norm @ cand_norm)
        action_match = 1.0 if cand['action'] == src_action_raw else 0.0
        cluster_match = 1.0 if cand['cluster'] == pred_cluster else 0.0
        composite = (weights['cluster'] * cluster_match
                     + weights['q'] * q_sim
                     + weights['action'] * action_match)
        cand_records.append({
            'fname': fname,
            'action': cand['action'],
            'cluster': cand['cluster'],
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
        'picked_cluster': best['cluster'],
        'picked_cluster_match': best['cluster_match'],
        'pred_cluster': pred_cluster,
        'n_candidates': len(cand_records),
        'pool_size_cluster': sum(1 for c in pool_by_fname.values()
                                 if c['cluster'] == pred_cluster),
        'pool_size_q': topk_q,
    }


def process_fold(fold_seed, clf, clip_index, qsig_table, qc, fname_to_idx,
                 weights, topk_q=10, bench_version='v5', out_tag='m3_rerank'):
    manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold_seed}/manifest.json'))
    out_dir = SAVE_ROOT / out_tag / f'fold_{fold_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_failed = 0
    per_query = []
    t0 = time.time()
    for i, q in enumerate(manifest['queries']):
        qid = q['query_id']
        rec = {'query_id': qid, 'cluster': q['cluster'], 'split': q['split'],
               'skel_a': q['skel_a'], 'skel_b': q['skel_b'], 'status': 'pending',
               'src_action': q['src_action']}
        try:
            r = rerank_one_query(q, clf, clip_index, qsig_table, qc,
                                 fname_to_idx, weights, topk_q=topk_q)
            picked_fname = r['picked_fname']
            motion = np.load(MOTION_DIR / picked_fname).astype(np.float32)

            np.save(out_dir / f'query_{qid:04d}.npy', motion)
            rec.update({
                'status': 'ok',
                'picked_fname': picked_fname,
                'picked_action': r['picked_action'],
                'picked_cluster': r['picked_cluster'],
                'pred_cluster': r['pred_cluster'],
                'picked_q_sim': r['picked_q_sim'],
                'picked_composite': r['picked_composite'],
                'picked_cluster_match': r['picked_cluster_match'],
                'action_matches_source': r['picked_action'] == q['src_action'],
                'cluster_matches_source': r['picked_cluster'] == q['cluster'],
            })
            n_done += 1
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = f'{type(e).__name__}: {e}'
            print(f'  q{qid} FAILED: {e}')
            n_failed += 1
        per_query.append(rec)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(manifest['queries']) - i - 1)
            print(f"  [{i+1}/{len(manifest['queries'])}] "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    summary = {
        'method': out_tag, 'fold': fold_seed, 'bench_version': bench_version,
        'weights': weights, 'topk_q': topk_q,
        'n_done': n_done, 'n_failed': n_failed,
        'total_time_sec': time.time() - t0,
        'per_query': per_query,
    }
    meta_path = out_dir / 'metrics.json'
    with open(meta_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Diagnostic counts
    action_match_rate = (
        sum(1 for r in per_query if r.get('action_matches_source'))
        / max(n_done, 1)
    )
    cluster_match_rate = (
        sum(1 for r in per_query if r.get('cluster_matches_source'))
        / max(n_done, 1)
    )
    pred_cluster_match_rate = (
        sum(1 for r in per_query if r.get('picked_cluster_match'))
        / max(n_done, 1)
    )
    print(f"\nFold {fold_seed}: {n_done}/{len(manifest['queries'])} ok, "
          f"{n_failed} failed. action_match={action_match_rate:.3f}, "
          f"cluster_match={cluster_match_rate:.3f}, "
          f"pred_cluster_picked={pred_cluster_match_rate:.3f}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--w_cluster', type=float, default=1.0)
    parser.add_argument('--w_q', type=float, default=2.0)
    parser.add_argument('--w_action', type=float, default=3.0)
    parser.add_argument('--topk_q', type=int, default=10,
                        help='Top-K Q-similar clips added to the candidate pool')
    parser.add_argument('--out_tag', type=str, default='m3A_rerank')
    parser.add_argument('--skip_eval', action='store_true')
    args = parser.parse_args()

    weights = {'cluster': args.w_cluster, 'q': args.w_q, 'action': args.w_action}
    print(f"Weights: {weights}, topk_q: {args.topk_q}")

    # Load Q cache
    print("Loading Q cache...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    print(f"  {len(qc['meta'])} clips")

    # Build Q signatures
    print("Building Q signature table...")
    qsig_table = build_q_sig_table(qc, None)
    print(f"  {len(qsig_table)} sigs cached")

    # Train I-5 classifier
    print("Training I-5 cluster classifier...")
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels)

    # Load clip index for candidate pools
    clip_index = json.load(open(CLIP_INDEX_PATH))
    print(f"Clip index: {len(clip_index['index'])} skeletons")

    out_dirs = []
    for fold in args.folds:
        print(f"\n=== Processing fold {fold} ===")
        out_dir = process_fold(fold, clf, clip_index, qsig_table, qc,
                                fname_to_idx, weights, topk_q=args.topk_q,
                                bench_version='v5', out_tag=args.out_tag)
        out_dirs.append((fold, out_dir))

    if not args.skip_eval:
        import subprocess
        for fold, out_dir in out_dirs:
            for distance in ('procrustes', 'zscore_dtw', 'q_component'):
                print(f"\n=== Eval: {args.out_tag} fold {fold} {distance} ===")
                cmd = ['python', '-u', '-m', 'eval.benchmark_v3.eval_v5',
                       '--method_dir', str(out_dir),
                       '--fold', str(fold),
                       '--method_name', args.out_tag,
                       '--distance', distance]
                subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == '__main__':
    main()
