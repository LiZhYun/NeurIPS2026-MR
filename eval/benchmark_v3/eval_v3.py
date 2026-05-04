"""v3 benchmark evaluation — unified protocol for ALL methods.

Loads:
  - v3 query manifest (source, positive set, K adversarials)
  - A method's predicted motions (pair_<qid>.npy per query)
Computes primary metrics (contrastive AUC, retrieval top-K) + secondary.

Usage:
  python -m eval.benchmark_v3.eval_v3 \
    --method_dir eval/results/baselines/anytop_train/v3_fold_42 \
    --fold 42 --method_name anytop_train
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

from eval.benchmark_v3.metrics_v3 import (
    dist_zscore_dtw_inv, dist_procrustes_trajectory, q_component_distances,
    contrastive_auc, contrastive_accuracy, best_match_distance,
    rank_of_best_positive, retrieval_topk, bootstrap_ci,
)
from model.skel_blind.encoder import encode_motion_to_invariant

N_DISTRACTORS_PER_QUERY = 20  # open-world retrieval distractors (other skels/actions)

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'


def make_skel_cond(cond_dict, name):
    return {
        'joints_names': cond_dict[name]['joints_names'],
        'parents': cond_dict[name]['parents'],
        'object_type': name,
    }


def _recover_positions(motion, skel_name, cond_dict):
    """Recover [T, J, 3] joint positions from a saved motion.

    Detects format:
    - shape [T, J, 3]: positions ALREADY recovered (e.g., M2M-official outputs FK positions)
    - shape [T, J, 13]: standard 13-dim format → call recover_from_bvh_ric_np
    """
    n_joints = len(cond_dict[skel_name]['parents'])
    if motion.shape[1] > n_joints:
        motion = motion[:, :n_joints]
    if motion.shape[-1] == 3:
        # Already positions, no recovery needed
        return motion.astype(np.float32)
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    return recover_from_bvh_ric_np(motion.astype(np.float32))


def _q_from_motion(motion_13, skel_name, cond_dict, contact_groups):
    """Compute Q dict from a motion array (no file I/O)."""
    from eval.principled_cross_skel_metrics import extract_q_from_array
    cg = contact_groups.get(skel_name)
    return extract_q_from_array(motion_13, cond_dict[skel_name], cg)


def build_distractor_pool(manifest, motion_dir, n_per_query=N_DISTRACTORS_PER_QUERY,
                          seed=42):
    """Build a fixed pool of distractor motions (other skel + other action)
    sampled DETERMINISTICALLY for reproducible open-world retrieval."""
    rng = np.random.RandomState(seed)
    # All clip filenames available, indexed by skeleton + cluster
    all_clips = []
    for q in manifest['queries']:
        for p in q['positives']:
            all_clips.append({'fname': p['fname'], 'skel': q['skel_b'],
                              'cluster': q['cluster']})
        for a in q['adversarials']:
            all_clips.append({'fname': a['fname'], 'skel': q['skel_b'],
                              'cluster': a['cluster']})
    # Dedupe
    seen = set()
    unique = []
    for c in all_clips:
        if c['fname'] not in seen:
            seen.add(c['fname'])
            unique.append(c)

    # For each query, pick distractors with DIFFERENT skel AND different cluster
    distractor_map = {}
    for q in manifest['queries']:
        cands = [c for c in unique
                 if c['skel'] != q['skel_b'] and c['cluster'] != q['cluster']]
        if len(cands) < n_per_query:
            distractor_map[q['query_id']] = cands
        else:
            indices = rng.choice(len(cands), size=n_per_query, replace=False)
            distractor_map[q['query_id']] = [cands[i] for i in indices]
    return distractor_map


def load_q_cache():
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}

    def get_q(fname):
        idx = fname_to_idx.get(fname)
        if idx is None:
            return None
        return {
            'com_path': qc['com_path'][idx],
            'heading_vel': qc['heading_vel'][idx],
            'contact_sched': qc['contact_sched'][idx],
            'cadence': float(qc['cadence'][idx]),
            'limb_usage': qc['limb_usage'][idx],
        }
    return get_q


def _compute_distance(distance_metric, pos_pred, inv_pred, q_pred,
                      ref_motion, skel_b, cond_dict, skel_cond,
                      contact_groups, get_q, ref_fname):
    """Compute one (pred, ref) distance under chosen metric.

    For q_component returns the raw 4-component dict so caller can z-score
    within the per-query candidate pool (Codex audit fix 2026-04-23: avoid
    test-tuning weights, avoid raw-magnitude domination of composite).
    """
    if distance_metric == 'procrustes':
        pos_ref = _recover_positions(ref_motion, skel_b, cond_dict)
        return dist_procrustes_trajectory(pos_pred, pos_ref)
    elif distance_metric == 'zscore_dtw':
        inv_ref = encode_motion_to_invariant(ref_motion, skel_cond)
        return dist_zscore_dtw_inv(inv_pred, inv_ref)
    elif distance_metric == 'q_component':
        q_ref = get_q(ref_fname) or _q_from_motion(ref_motion, skel_b, cond_dict, contact_groups)
        # Return raw components — z-scoring happens at per-query aggregation step.
        return q_component_distances(q_pred, q_ref)
    raise ValueError(f'Unknown dist: {distance_metric}')


def _zscore_q_pool(comp_dicts):
    """Z-score 4 Q components within a per-query candidate pool, sum to scalar.

    comp_dicts: list of dicts {com_rel_l2, cs_one_minus_f1, cadence_abs, limb_l2}.
    Returns: list of float distances (one per candidate), each = sum of z-scored components.
    Z-scoring is symmetric across positives + adversarials + distractors → no class bias.
    """
    if not comp_dicts:
        return []
    keys = sorted(comp_dicts[0].keys())
    arrs = {k: np.asarray([c[k] for c in comp_dicts], dtype=np.float64) for k in keys}
    out = np.zeros(len(comp_dicts), dtype=np.float64)
    for k in keys:
        v = arrs[k]
        s = float(v.std()) if v.std() > 1e-12 else 1.0
        m = float(v.mean())
        out += (v - m) / s
    return out.tolist()


def eval_method(method_dir: Path, manifest_path: Path, method_name: str,
                cond_dict, contact_groups, get_q, distractor_map,
                distance_metric: str = 'procrustes',
                max_queries: int = 10000):
    """Evaluate one method's outputs against v3 manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:max_queries]

    per_query = []
    t_skipped = 0
    for q in queries:
        qid = q['query_id']
        skel_b = q['skel_b']
        pred_path = method_dir / f'query_{qid:04d}.npy'
        if not pred_path.exists():
            t_skipped += 1
            continue

        try:
            pred = np.load(pred_path)
            skel_cond = make_skel_cond(cond_dict, skel_b)
            is_positions_only = pred.ndim == 3 and pred.shape[-1] == 3
            if is_positions_only and distance_metric in ('zscore_dtw', 'q_component'):
                # M2M-official outputs positions only; can't compute these metrics
                t_skipped += 1
                continue
            if not is_positions_only:
                inv_pred = encode_motion_to_invariant(pred.astype(np.float32), skel_cond)
                q_pred = (_q_from_motion(pred, skel_b, cond_dict, contact_groups)
                          if distance_metric == 'q_component' else None)
            else:
                inv_pred = None
                q_pred = None
            pos_pred = _recover_positions(pred, skel_b, cond_dict)

            args_for_dist = dict(distance_metric=distance_metric, pos_pred=pos_pred,
                                 inv_pred=inv_pred, q_pred=q_pred, skel_b=skel_b,
                                 cond_dict=cond_dict, skel_cond=skel_cond,
                                 contact_groups=contact_groups, get_q=get_q)

            # Positives
            pos_dists = []
            for p in q['positives']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_dists.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'],
                                                    **args_for_dist))
            # Adversarials
            neg_dists = []
            for a in q['adversarials']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                # Skel-b cond used (adversarials are on same skeleton)
                neg_dists.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'],
                                                    **args_for_dist))
            # Open-world distractors (different skel + different cluster)
            # IMPORTANT: per Codex review, cross-skel Procrustes is invalid (joint
            # truncation to Jmin is not semantically aligned). So we ONLY compute
            # open-world distractor distances when the chosen metric is morphology-
            # invariant: zscore_dtw or q_component.
            distractor_dists = []
            if distance_metric in ('zscore_dtw', 'q_component'):
                for d in distractor_map.get(qid, []):
                    ref = np.load(MOTION_DIR / d['fname']).astype(np.float32)
                    d_skel_cond = make_skel_cond(cond_dict, d['skel'])
                    if distance_metric == 'q_component':
                        d_q_ref = get_q(d['fname']) or _q_from_motion(ref, d['skel'], cond_dict, contact_groups)
                        # Return raw component dict (consistent with _compute_distance)
                        dd = q_component_distances(q_pred, d_q_ref)
                    else:  # zscore_dtw
                        d_inv_ref = encode_motion_to_invariant(ref, d_skel_cond)
                        dd = dist_zscore_dtw_inv(inv_pred, d_inv_ref)
                    distractor_dists.append(dd)

            # For q_component: per-query within-pool z-scoring of 4 components.
            # Avoids raw-magnitude domination of composite (Codex audit fix 2026-04-23).
            if distance_metric == 'q_component':
                pool = pos_dists + neg_dists + distractor_dists
                z_pool = _zscore_q_pool(pool)
                n_pos = len(pos_dists)
                n_neg = len(neg_dists)
                pos_dists = z_pool[:n_pos]
                neg_dists = z_pool[n_pos:n_pos + n_neg]
                distractor_dists = z_pool[n_pos + n_neg:]

            rec = {
                'query_id': qid,
                'split': q['split'],
                'cluster': q['cluster'],
                'skel_a': q['skel_a'],
                'skel_b': skel_b,
                'n_positives': len(pos_dists),
                'n_negatives': len(neg_dists),
                'n_distractors': len(distractor_dists),
                'auc': contrastive_auc(pos_dists, neg_dists),
                'contrastive_acc': contrastive_accuracy(pos_dists, neg_dists),
                'best_match_dist': best_match_distance(pos_dists),
                'rank_best_pos': rank_of_best_positive(pos_dists, neg_dists),
                'mean_pos_dist': float(np.mean(pos_dists)) if pos_dists else 0.0,
                'mean_neg_dist': float(np.mean(neg_dists)) if neg_dists else 0.0,
            }

            # Closed-pool retrieval (positives + adversarials only)
            all_dists_closed = pos_dists + neg_dists
            labels_closed = ['positive'] * len(pos_dists) + ['negative_adv'] * len(neg_dists)
            rec['retrieval_closed'] = retrieval_topk(all_dists_closed, labels_closed,
                                                     k_vals=(1, 3, 5))

            # Open-world retrieval (positives + adversarials + distractors)
            # Only valid for morphology-invariant metrics (per Codex)
            if distractor_dists:
                all_dists_open = pos_dists + neg_dists + distractor_dists
                labels_open = (['positive'] * len(pos_dists)
                               + ['negative_adv'] * len(neg_dists)
                               + ['negative_distractor'] * len(distractor_dists))
                rec['retrieval_open'] = retrieval_topk(all_dists_open, labels_open,
                                                        k_vals=(1, 5, 10))
            else:
                rec['retrieval_open'] = None  # not valid for procrustes

            per_query.append(rec)
        except Exception as e:
            print(f"  query {qid} FAILED: {e}")
            t_skipped += 1

    # Aggregate
    print(f"\n=== {method_name} on {manifest_path.parent.name} "
          f"({distance_metric} distance) ===")
    print(f"  n_queries={len(per_query)}, skipped={t_skipped}")

    def agg(field):
        return bootstrap_ci([r[field] for r in per_query])

    summary_overall = {
        'n_queries': len(per_query),
        'n_skipped': t_skipped,
        'auc': agg('auc'),
        'contrastive_acc': agg('contrastive_acc'),
        'best_match_dist': agg('best_match_dist'),
        'rank_best_pos': agg('rank_best_pos'),
        'closed_top_1': bootstrap_ci([r['retrieval_closed']['top_1'] for r in per_query]),
        'closed_top_3': bootstrap_ci([r['retrieval_closed']['top_3'] for r in per_query]),
    }
    has_open = any(r['retrieval_open'] is not None for r in per_query)
    if has_open:
        summary_overall['open_top_1'] = bootstrap_ci(
            [r['retrieval_open']['top_1'] for r in per_query if r['retrieval_open']])
        summary_overall['open_top_5'] = bootstrap_ci(
            [r['retrieval_open']['top_5'] for r in per_query if r['retrieval_open']])

    print(f"  AUC (contrastive) = {summary_overall['auc'][1]:.3f} "
          f"[{summary_overall['auc'][0]:.3f}, {summary_overall['auc'][2]:.3f}]")
    print(f"  Contrastive acc   = {summary_overall['contrastive_acc'][1]:.3f}")
    print(f"  Closed Top-1      = {summary_overall['closed_top_1'][1]:.3f}")
    if has_open:
        print(f"  Open Top-1        = {summary_overall['open_top_1'][1]:.3f}")
        print(f"  Open Top-5        = {summary_overall['open_top_5'][1]:.3f}")
    else:
        print(f"  Open retrieval    = N/A (procrustes invalid cross-skel)")
    print(f"  Best-match dist   = {summary_overall['best_match_dist'][1]:.4f}")

    # Per-split breakdown
    by_split = {}
    for split in ['train', 'dev', 'mixed', 'test_test']:
        group = [r for r in per_query if r['split'] == split]
        if not group:
            continue
        def agg_g(field):
            return bootstrap_ci([r[field] for r in group])
        by_split[split] = {
            'n': len(group),
            'auc': agg_g('auc'),
            'contrastive_acc': agg_g('contrastive_acc'),
            'closed_top_1': bootstrap_ci([r['retrieval_closed']['top_1'] for r in group]),
        }
        open_vals = [r['retrieval_open']['top_1'] for r in group if r['retrieval_open']]
        if open_vals:
            by_split[split]['open_top_1'] = bootstrap_ci(open_vals)
        print(f"  [{split} n={len(group)}]: AUC={by_split[split]['auc'][1]:.3f}, "
              f"CA={by_split[split]['contrastive_acc'][1]:.3f}, "
              f"ClosedTop1={by_split[split]['closed_top_1'][1]:.3f}")

    return {
        'method': method_name,
        'distance_metric': distance_metric,
        'manifest': str(manifest_path),
        'overall': summary_overall,
        'by_split': by_split,
        'per_query': per_query,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_dir', type=str, required=True,
                        help='Dir containing query_NNNN.npy files')
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--method_name', type=str, required=True)
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--distance', type=str, default='procrustes',
                        choices=['procrustes', 'zscore_dtw', 'q_component'])
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/queries/fold_{args.fold}/manifest.json'
    out_dir = Path(args.out_dir) if args.out_dir else method_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cond, contact_groups, Q cache...")
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()
    with open(PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        contact_groups = json.load(f)
    get_q = load_q_cache()

    print(f"Building distractor pool...")
    with open(manifest_path) as f:
        manifest = json.load(f)
    distractor_map = build_distractor_pool(manifest, MOTION_DIR)
    print(f"  Avg {N_DISTRACTORS_PER_QUERY} distractors per query")

    summary = eval_method(method_dir, manifest_path, args.method_name,
                          cond_dict, contact_groups, get_q, distractor_map,
                          distance_metric=args.distance,
                          max_queries=args.max_queries)

    out_name = f'v3_eval_{args.distance}.json'
    with open(out_dir / out_name, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / out_name}")


if __name__ == '__main__':
    main()
