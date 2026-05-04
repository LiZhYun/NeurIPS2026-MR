"""B1 — Hand-curated paired GOLD subset for fine-semantic verification.

Per Codex 2026-04-23 radical-pivot verdict: B1 is the critical oral-blocking
benchmark that "validates fine semantics and tells you whether mined anchors
are real." Without human annotators, we build a high-confidence subset using
objective filters that approximate gold-quality labels.

Gold criteria (intersection):
  - Query is exact_tier_eligible (has >=1 positive_exact AND >=1 adversarial_hard)
  - src_action in WELL_DEFINED_ACTIONS = {idle, walk, run, jump, attack, die,
    fly, swim, slowforward, fastforward, slowfly, fastfly}
  - Source clip's Q-signature is within 1.5 IQR of its (skel_a, action) centroid
    in 30-d Q-feature space → rejects outlier/noisy source motions
  - Target skeleton has >=2 positives_exact (well-supported)
  - skel_b in test_skeletons (held-out target morphology — primary inductive claim)
  - skel_a may be train OR test (we just need a clean source clip; source
    skeleton being held-out is not what makes the answer harder for transfer)

Per-query GOLD metrics (in addition to v5 AUC):
  - hit@1: predicted distance ranks a positive_exact above all adversarials_hard
  - hit@all_pos: predicted distance ranks ALL positives_exact above all hards
  - mrr: mean reciprocal rank of first positive_exact in (positives_exact +
    adversarials_hard) sorted by predicted distance ascending

Usage:
  python -m eval.benchmark_v3.build_b1_gold_pairs --fold 42
  python -m eval.benchmark_v3.build_b1_gold_pairs --fold 42 --target_n 60
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

from eval.benchmark_v3.action_taxonomy import ACTION_CLUSTERS, action_to_cluster
from eval.baselines.run_i5_action_classifier_v3 import featurize_q

# Well-defined actions: those with clear motion templates. Excludes ambiguous
# labels like "agitated", "reaction" where source style varies wildly.
WELL_DEFINED_ACTIONS = {
    # locomotion
    'walk', 'run', 'fastrun', 'slowforward', 'fastforward', 'land',
    # idle / breathing
    'idle', 'idle1', 'idle2', 'idle3',
    # jump
    'jump',
    # combat
    'attack', 'attack2', 'attack3', 'bite', 'bite1', 'kick',
    # death
    'die',
    # flight
    'fly', 'slowfly', 'fastfly',
    # swim
    'swim',
}

Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
GOLD_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_b1_gold'


def build_q_feature_table(qc):
    """Return {fname: 30-d feature vector}."""
    table = {}
    for i, m in enumerate(qc['meta']):
        feat = featurize_q(qc['com_path'][i], qc['heading_vel'][i],
                           qc['contact_sched'][i], qc['cadence'][i],
                           qc['limb_usage'][i])
        table[m['fname']] = feat
    return table


def build_centroids(qc, feat_table):
    """Build (skel, action) → centroid + IQR in Q-feature space."""
    by_key = defaultdict(list)
    for i, m in enumerate(qc['meta']):
        action = m['fname'].split('___')[1].split('_')[0].lower() if '___' in m['fname'] else None
        if action is None:
            continue
        key = (m['skeleton'], action)
        by_key[key].append(feat_table[m['fname']])

    centroids = {}
    for key, feats in by_key.items():
        if len(feats) < 2:
            continue
        F = np.stack(feats)  # [N, 30]
        med = np.median(F, axis=0)
        # Use overall L2 distance from median; threshold = 1.5 * IQR of L2 dists
        dists = np.linalg.norm(F - med[None, :], axis=1)
        q1, q3 = np.percentile(dists, [25, 75])
        iqr = max(q3 - q1, 1e-3)
        threshold = q3 + 1.5 * iqr
        centroids[key] = {'med': med, 'threshold': float(threshold), 'n': len(feats)}
    return centroids


def is_canonical(fname, skel_a, action, feat_table, centroids):
    """True iff fname's Q-features are within 1.5 IQR of its (skel, action) centroid."""
    key = (skel_a, action)
    if key not in centroids:
        return True  # if no centroid (singleton class), accept by default
    feat = feat_table.get(fname)
    if feat is None:
        return False
    med = centroids[key]['med']
    th = centroids[key]['threshold']
    return float(np.linalg.norm(feat - med)) <= th


def build_gold(folds, target_n: int = 80, seed: int = 42, min_positives_exact: int = 1,
               min_total_pool: int = 0):
    """min_total_pool: minimum (positives_exact + adversarials_hard) pool size.
    Setting min_total_pool=4 enforces non-trivial discrimination."""
    print(f"Loading Q cache...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)

    print(f"Building Q feature table ({len(qc['meta'])} clips)...")
    feat_table = build_q_feature_table(qc)

    print(f"Building (skel, action) centroids...")
    centroids = build_centroids(qc, feat_table)
    print(f"  {len(centroids)} (skel, action) centroids")

    # Combine queries across folds
    all_queries = []
    test_skels_union = set()
    for fold_seed in folds:
        manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold_seed}/manifest.json'))
        for q in manifest['queries']:
            qq = dict(q)
            qq['fold_seed'] = fold_seed
            qq['original_query_id'] = q['query_id']
            all_queries.append(qq)
        test_skels_union |= set(manifest['test_skeletons'])
    queries = all_queries
    test_skels = test_skels_union
    print(f"Combined folds {folds}: {len(queries)} queries, {len(test_skels)} test skels (union)")

    # Filter pipeline
    candidates = []
    reasons = defaultdict(int)
    for q in queries:
        # 1. Exact-tier eligible
        if not q.get('exact_tier_eligible', False):
            reasons['not_exact_tier_eligible'] += 1; continue
        # 2. Source action well-defined
        src_action = q['src_action']
        if src_action not in WELL_DEFINED_ACTIONS:
            reasons['not_well_defined_action'] += 1; continue
        # 3. Target skel held-out — what matters for inductive cross-skel claim
        if q['skel_b'] not in test_skels:
            reasons['target_not_held_out'] += 1; continue
        # 4. Target has >=min positives_exact
        if len(q.get('positives_exact', [])) < min_positives_exact:
            reasons[f'lt_{min_positives_exact}_positives_exact'] += 1; continue
        # 4b. Total pool (positives_exact + adversarials_hard) is non-trivial
        n_total = len(q.get('positives_exact', [])) + len(q.get('adversarials_hard', []))
        if n_total < min_total_pool:
            reasons[f'lt_{min_total_pool}_total_pool'] += 1; continue
        # 5. Source canonical
        if not is_canonical(q['src_fname'], q['skel_a'], src_action,
                            feat_table, centroids):
            reasons['source_not_canonical'] += 1; continue
        candidates.append(q)

    print(f"\nFilter funnel:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
    print(f"  candidates after all filters: {len(candidates)}")

    # If too many, downsample preserving cluster + skel diversity
    rng = np.random.RandomState(seed)
    if len(candidates) > target_n:
        # Group by cluster, sample proportionally
        by_cluster = defaultdict(list)
        for q in candidates:
            by_cluster[q['cluster']].append(q)
        target_per_cluster = max(1, target_n // len(by_cluster))
        chosen = []
        for cluster, qs in by_cluster.items():
            n_take = min(target_per_cluster, len(qs))
            idxs = rng.choice(len(qs), n_take, replace=False)
            chosen.extend([qs[i] for i in idxs])
        # Top up if under budget by uniform sampling
        if len(chosen) < target_n:
            remaining = [q for q in candidates if q not in chosen]
            n_extra = min(target_n - len(chosen), len(remaining))
            extra_idxs = rng.choice(len(remaining), n_extra, replace=False)
            chosen.extend([remaining[i] for i in extra_idxs])
        candidates = chosen

    # Reassign query_ids 0..N-1 for the gold subset (keep original_query_id)
    gold_queries = []
    for new_id, q in enumerate(sorted(candidates, key=lambda x: x['query_id'])):
        gq = dict(q)
        gq['original_query_id'] = q['query_id']
        gq['query_id'] = new_id
        # Compute gold-spec metadata
        gq['gold_n_pos_exact'] = len(q['positives_exact'])
        gq['gold_n_adv_hard'] = len(q['adversarials_hard'])
        gq['gold_n_pool'] = (len(q['positives_exact']) + len(q['adversarials_hard']))
        gold_queries.append(gq)

    # Cluster + action distribution check
    by_cluster = defaultdict(int)
    by_action = defaultdict(int)
    by_skel_pair = defaultdict(int)
    for q in gold_queries:
        by_cluster[q['cluster']] += 1
        by_action[q['src_action']] += 1
        by_skel_pair[f"{q['skel_a']}__{q['skel_b']}"] += 1

    print(f"\nGold subset: {len(gold_queries)} queries")
    print(f"  Clusters: {dict(sorted(by_cluster.items(), key=lambda x: -x[1]))}")
    print(f"  Actions: {dict(sorted(by_action.items(), key=lambda x: -x[1]))}")
    print(f"  Skel pairs: {len(by_skel_pair)} unique")

    # Save
    gold_out = {
        'version': 'b1_gold_v1',
        'source_folds': list(folds),
        'target_n': target_n,
        'gold_filters': {
            'exact_tier_eligible': True,
            'well_defined_actions': sorted(WELL_DEFINED_ACTIONS),
            'target_held_out_only': True,
            'min_positives_exact': min_positives_exact,
            'min_total_pool': min_total_pool,
            'source_canonical_iqr': 1.5,
        },
        'cluster_distribution': dict(by_cluster),
        'action_distribution': dict(by_action),
        'n_skel_pairs': len(by_skel_pair),
        'queries': gold_queries,
    }
    if min_total_pool >= 4:
        subdir = f'combined_strict_pool{min_total_pool}'
    elif min_positives_exact >= 2:
        subdir = f'combined_strict_min{min_positives_exact}'
    else:
        subdir = 'combined'
    out_dir = GOLD_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'manifest.json'
    with open(out_path, 'w') as f:
        json.dump(gold_out, f, indent=2)
    print(f"\nSaved: {out_path}")
    return out_path, gold_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--target_n', type=int, default=200,
                        help='Target gold subset size (200 = effectively all that pass filters)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_positives_exact', type=int, default=2,
                        help='Min number of positives_exact required '
                             '(2 = strict gold per Codex; 1 = any exact eligible)')
    parser.add_argument('--min_total_pool', type=int, default=0,
                        help='Min size of (positives_exact + adversarials_hard) pool. '
                             '4 = useful pool of >=4 candidates per query.')
    args = parser.parse_args()
    build_gold(args.folds, args.target_n, args.seed,
               args.min_positives_exact, args.min_total_pool)


if __name__ == '__main__':
    main()
