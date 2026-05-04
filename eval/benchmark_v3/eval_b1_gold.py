"""B1 gold evaluator — fine-semantic verification on a curated subset.

Reads the B1 gold manifest (eval/benchmark_v3/queries_b1_gold/combined/manifest.json),
loads each method's predictions from <method_root>/fold_{fold_seed}/query_{orig_id:04d}.npy,
and computes per-query gold metrics:

  - exact_tier_auc:  ranks positives_exact vs adversarials_hard (from v5 manifest)
  - hit_at_1:        binary, 1 if best-ranked candidate is a positive_exact
  - hit_at_all_pos:  binary, 1 if all positives_exact rank above all adversarials_hard
  - mrr:             mean reciprocal rank of first positive_exact in
                     (positives_exact + adversarials_hard) sorted by predicted
                     distance ascending

Aggregates with block bootstrap by skel_pair.

Usage:
  python -m eval.benchmark_v3.eval_b1_gold \
      --method_root save/oracles/v3/i5_action_classifier_v5 \
      --method_name i5_action_classifier --distance procrustes
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score
from eval.benchmark_v3.eval_v5 import (
    block_bootstrap_ci, _compute_distance, load_q_cache, MOTION_DIR,
)
from eval.benchmark_v3.eval_v3 import (
    make_skel_cond, _recover_positions, _q_from_motion, _zscore_q_pool,
)
from model.skel_blind.encoder import encode_motion_to_invariant
from eval.moreflow_phi import load_contact_groups

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
# Default to strict manifest if available (built with min_positives_exact >= 2);
# otherwise fall back to lenient manifest.
GOLD_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_b1_gold'
GOLD_MANIFEST_STRICT = GOLD_DIR / 'combined_strict_min2/manifest.json'
GOLD_MANIFEST_LENIENT = GOLD_DIR / 'combined/manifest.json'
GOLD_MANIFEST = GOLD_MANIFEST_STRICT if GOLD_MANIFEST_STRICT.exists() else GOLD_MANIFEST_LENIENT


def _auc_safe(pos, neg):
    if not pos or not neg:
        return None
    scores = -np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    try: return float(roc_auc_score(labels, scores))
    except Exception: return None


def _hit_at_1(pos, neg):
    """1 if min(pos) < min(neg). Returns None if missing data."""
    if not pos or not neg: return None
    return 1.0 if min(pos) < min(neg) else 0.0


def _hit_at_all_pos(pos, neg):
    """1 if max(pos) < min(neg). i.e., all positives ranked above all hard negatives."""
    if not pos or not neg: return None
    return 1.0 if max(pos) < min(neg) else 0.0


def _mrr(pos, neg):
    """Mean reciprocal rank of first positive in pos+neg sorted ascending."""
    if not pos or not neg: return None
    all_dists = list(pos) + list(neg)
    labels = [1] * len(pos) + [0] * len(neg)
    order = np.argsort(all_dists)
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def evaluate_b1(method_root: Path, method_name: str, distance: str,
                gold_manifest_path: Path = None):
    if gold_manifest_path is None:
        gold_manifest_path = GOLD_MANIFEST
    gold = json.load(open(gold_manifest_path))
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()
    contact_groups = load_contact_groups()
    get_q = load_q_cache()

    per_query = []
    n_skipped_missing = 0
    for q in gold['queries']:
        gold_qid = q['query_id']
        orig_qid = q['original_query_id']
        fold_seed = q['fold_seed']
        skel_b = q['skel_b']
        pred_path = method_root / f'fold_{fold_seed}' / f'query_{orig_qid:04d}.npy'
        if not pred_path.exists():
            n_skipped_missing += 1
            continue

        try:
            pred = np.load(pred_path).astype(np.float32)
            skel_cond = make_skel_cond(cond_dict, skel_b)
            is_pos_only = pred.ndim == 3 and pred.shape[-1] == 3
            if is_pos_only and distance in ('zscore_dtw', 'q_component'):
                n_skipped_missing += 1; continue
            if not is_pos_only:
                inv_pred = encode_motion_to_invariant(pred, skel_cond)
                q_pred = (_q_from_motion(pred, skel_b, cond_dict, contact_groups)
                         if distance == 'q_component' else None)
            else:
                inv_pred = None; q_pred = None
            pos_pred = _recover_positions(pred, skel_b, cond_dict)

            args_for_dist = dict(distance_metric=distance, pos_pred=pos_pred,
                                 inv_pred=inv_pred, q_pred=q_pred, skel_b=skel_b,
                                 cond_dict=cond_dict, skel_cond=skel_cond,
                                 contact_groups=contact_groups, get_q=get_q)

            # Exact-tier ONLY: positives_exact vs adversarials_hard
            pos_dists, neg_dists = [], []
            for p in q['positives_exact']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_dists.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'],
                                                    **args_for_dist))
            for a in q['adversarials_hard']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                neg_dists.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'],
                                                    **args_for_dist))

            # Q-component pool z-scoring (per-query)
            if distance == 'q_component':
                pool = pos_dists + neg_dists
                if pool:
                    z = _zscore_q_pool(pool)
                    pos_dists = z[:len(pos_dists)]
                    neg_dists = z[len(pos_dists):]

            rec = {
                'gold_qid': gold_qid,
                'original_qid': orig_qid,
                'fold_seed': fold_seed,
                'split': q['split'],
                'cluster': q['cluster'],
                'src_action': q['src_action'],
                'skel_a': q['skel_a'], 'skel_b': skel_b,
                'n_pos_exact': len(pos_dists),
                'n_adv_hard': len(neg_dists),
                'auc': _auc_safe(pos_dists, neg_dists),
                'hit_at_1': _hit_at_1(pos_dists, neg_dists),
                'hit_at_all_pos': _hit_at_all_pos(pos_dists, neg_dists),
                'mrr': _mrr(pos_dists, neg_dists),
                'block_id': f"{q['skel_a']}__{q['skel_b']}",
            }
            per_query.append(rec)
        except Exception as e:
            print(f"  q{gold_qid} ({orig_qid}, fold {fold_seed}): FAIL {e}")
            n_skipped_missing += 1

    out = {
        'method': method_name, 'distance': distance,
        'gold_manifest': str(gold_manifest_path),
        'n_queries': len(per_query), 'n_skipped': n_skipped_missing,
        'per_query': per_query,
    }
    auc_blocks = [(r['auc'], r['block_id']) for r in per_query if r['auc'] is not None]
    hit1_blocks = [(r['hit_at_1'], r['block_id']) for r in per_query if r['hit_at_1'] is not None]
    hitall_blocks = [(r['hit_at_all_pos'], r['block_id']) for r in per_query if r['hit_at_all_pos'] is not None]
    mrr_blocks = [(r['mrr'], r['block_id']) for r in per_query if r['mrr'] is not None]

    out['auc_ci'] = block_bootstrap_ci(auc_blocks)
    out['hit_at_1_ci'] = block_bootstrap_ci(hit1_blocks)
    out['hit_at_all_pos_ci'] = block_bootstrap_ci(hitall_blocks)
    out['mrr_ci'] = block_bootstrap_ci(mrr_blocks)

    by_cluster = defaultdict(list)
    for r in per_query:
        if r['hit_at_1'] is not None:
            by_cluster[r['cluster']].append((r['hit_at_1'], r['block_id']))
    out['per_cluster_hit_at_1'] = {
        c: {'ci': block_bootstrap_ci(v), 'n': len(v)} for c, v in by_cluster.items()
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_root', required=True,
                        help='Root dir containing fold_42/ and fold_43/ subdirs')
    parser.add_argument('--method_name', required=True)
    parser.add_argument('--distance', choices=['procrustes', 'zscore_dtw', 'q_component'],
                        default='procrustes')
    parser.add_argument('--gold_manifest', default=None)
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    method_root = Path(args.method_root)
    out_dir = Path(args.out_dir) if args.out_dir else method_root
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_b1(method_root, args.method_name, args.distance,
                          Path(args.gold_manifest) if args.gold_manifest else None)
    out_path = out_dir / f'b1_gold_eval_{args.distance}.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    auc = summary['auc_ci']
    h1 = summary['hit_at_1_ci']
    ha = summary['hit_at_all_pos_ci']
    mrr = summary['mrr_ci']
    print(f"\n=== B1 GOLD: {args.method_name} {args.distance} (n={summary['n_queries']}) ===")
    print(f"  exact-tier AUC : {auc[1]:.3f} [{auc[0]:.3f}, {auc[2]:.3f}]")
    print(f"  hit@1          : {h1[1]:.3f} [{h1[0]:.3f}, {h1[2]:.3f}]")
    print(f"  hit@all_pos    : {ha[1]:.3f} [{ha[0]:.3f}, {ha[2]:.3f}]")
    print(f"  MRR            : {mrr[1]:.3f} [{mrr[0]:.3f}, {mrr[2]:.3f}]")
    print(f"  per-cluster hit@1:")
    for c, info in sorted(summary['per_cluster_hit_at_1'].items(), key=lambda x: -x[1]['n']):
        ci = info['ci']
        print(f"    {c:12s}: {ci[1]:.3f} [{ci[0]:.3f}, {ci[2]:.3f}] (n={info['n']})")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
