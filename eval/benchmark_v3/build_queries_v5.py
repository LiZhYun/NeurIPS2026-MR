"""Generate v5 benchmark queries — tier-separated, all-10-cluster, oral-defensible.

Per refine-logs/BENCHMARK_V5_DESIGN.md (Codex 7/10 with v5.1 patches).

KEY DIFFERENCES from v3:
- Use ALL 10 ACTION_CLUSTERS (was: 4)
- Tier-separated labels:
    cluster tier: positives = same-cluster clips; easy negatives = diff-cluster
    exact tier: positives = exact-action clips; hard negatives = same-cluster diff-action
- Variable hard/easy negatives per query (cap K=5 each)
- Per-tier eligibility flags + feasibility preflight
- Same-target-skel distractors (fixes v3 open-world bug)
- Eligibility: exact-tier requires n_hard ≥ 1; cluster-tier requires n_easy ≥ 1
- Manifest fields per query: tier flags, n_hard, n_easy, support_reason

Output: eval/benchmark_v3/queries_v5/fold_{seed}/manifest.json
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)

CLIP_INDEX_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'
OUT_ROOT = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'

# Held-out skeletons (10 for test_test split)
TEST_SKELETONS = {
    'Cat', 'Elephant', 'Spider', 'Crab', 'Trex',
    'Raptor', 'Buzzard', 'Alligator', 'Anaconda', 'Rat',
}

# All 10 clusters (was: only 4 in v3)
ALL_CLUSTERS = sorted(ACTION_CLUSTERS.keys())

# Target sizes per split
TARGETS = {'train': 50, 'dev': 50, 'mixed': 100, 'test_test': 100}

# Caps
K_HARD_MAX = 5            # max same-cluster diff-action negatives
K_EASY_MAX = 5            # max diff-cluster negatives
K_DISTRACTORS = 20        # same-target-skel distractors
MAX_POSITIVES_CLUSTER = 8
MAX_POSITIVES_EXACT = 4
LENGTH_TOL = 0.30


def load_clip_index_full():
    """Build {(skel, action): [clips]}, {(skel, cluster): [clips]}, all_clips_per_skel."""
    cidx = json.load(open(CLIP_INDEX_PATH))
    by_skel_action = defaultdict(list)
    by_skel_cluster = defaultdict(list)
    by_skel = defaultdict(list)
    by_skel_action_other = defaultdict(list)  # same skel, same cluster, DIFFERENT action
    for skel, clusters in cidx['index'].items():
        for cluster, clips in clusters.items():
            for clip in clips:
                rec = {'fname': clip['fname'], 'T': clip['T'],
                       'action': clip['action'], 'cluster': cluster}
                by_skel_action[(skel, clip['action'])].append(rec)
                by_skel_cluster[(skel, cluster)].append(rec)
                by_skel[skel].append(rec)
    return by_skel_action, by_skel_cluster, by_skel


def length_match(clips, target_T, tol=LENGTH_TOL):
    return [c for c in clips if abs(c['T'] - target_T) / max(target_T, 1) <= tol]


def build_query(src_clip, src_skel, tgt_skel, by_skel_action, by_skel_cluster, by_skel,
                rng, query_id, split):
    """Build one query with full v5 tier separation + eligibility."""
    src_action = src_clip['action']
    src_cluster = src_clip['cluster']
    src_T = src_clip['T']

    # Cluster-tier positives: all target_skel clips in same cluster
    pos_cluster_all = by_skel_cluster.get((tgt_skel, src_cluster), [])
    # Exact-tier positives: all target_skel clips with EXACT action
    pos_exact_all = by_skel_action.get((tgt_skel, src_action), [])

    # Cluster-tier easy negatives: target_skel clips in different cluster, length-matched
    neg_easy_pool = [c for c in by_skel.get(tgt_skel, []) if c['cluster'] != src_cluster]
    neg_easy_pool = length_match(neg_easy_pool, src_T) or neg_easy_pool[:K_EASY_MAX*2]

    # Exact-tier hard negatives: target_skel clips in SAME cluster but DIFFERENT action
    neg_hard_pool = [c for c in by_skel_cluster.get((tgt_skel, src_cluster), [])
                     if c['action'] != src_action]
    # Don't length-filter hard negatives — too few candidates already

    # Cap + dedupe positives by filename
    pos_cluster = sorted({c['fname']: c for c in pos_cluster_all}.values(),
                        key=lambda c: c['fname'])[:MAX_POSITIVES_CLUSTER]
    pos_exact = sorted({c['fname']: c for c in pos_exact_all}.values(),
                       key=lambda c: c['fname'])[:MAX_POSITIVES_EXACT]

    # Sample negatives
    if neg_easy_pool:
        idx_easy = rng.choice(len(neg_easy_pool), min(K_EASY_MAX, len(neg_easy_pool)), replace=False)
        neg_easy = [neg_easy_pool[i] for i in idx_easy]
    else:
        neg_easy = []
    if neg_hard_pool:
        idx_hard = rng.choice(len(neg_hard_pool), min(K_HARD_MAX, len(neg_hard_pool)), replace=False)
        neg_hard = [neg_hard_pool[i] for i in idx_hard]
    else:
        neg_hard = []

    # Same-target-skel distractors: any clip on target_skel, exclude positives + exact
    excluded = {c['fname'] for c in pos_cluster} | {c['fname'] for c in pos_exact}
    distractor_pool = [c for c in by_skel.get(tgt_skel, []) if c['fname'] not in excluded]
    if distractor_pool:
        n_take = min(K_DISTRACTORS, len(distractor_pool))
        idx_d = rng.choice(len(distractor_pool), n_take, replace=False)
        distractors = [distractor_pool[i] for i in idx_d]
    else:
        distractors = []

    n_easy = len(neg_easy)
    n_hard = len(neg_hard)
    n_pos_cluster = len(pos_cluster)
    n_pos_exact = len(pos_exact)

    # Eligibility
    cluster_tier_eligible = (n_pos_cluster >= 1) and (n_easy >= 1)
    exact_tier_eligible = (n_pos_exact >= 1) and (n_hard >= 1)

    # Support reason
    if n_pos_exact >= 1:
        support_reason = 'exact_action'
    elif n_pos_cluster >= 1:
        support_reason = 'cluster_only'
    else:
        support_reason = 'absent'

    return {
        'query_id': query_id,
        'split': split,
        'skel_a': src_skel,
        'skel_b': tgt_skel,
        'cluster': src_cluster,
        'src_fname': src_clip['fname'],
        'src_T': src_T,
        'src_action': src_action,
        'positives_cluster': pos_cluster,
        'positives_exact': pos_exact,
        'adversarials_hard': neg_hard,
        'adversarials_easy': neg_easy,
        'distractors_same_target_skel': distractors,
        'n_pos_cluster': n_pos_cluster,
        'n_pos_exact': n_pos_exact,
        'n_hard': n_hard,
        'n_easy': n_easy,
        'cluster_tier_eligible': cluster_tier_eligible,
        'exact_tier_eligible': exact_tier_eligible,
        'support_reason': support_reason,
    }


def build_fold(seed, by_skel_action, by_skel_cluster, by_skel, all_skels):
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    queries = []
    qid = 0

    train_skels = sorted(set(all_skels) - TEST_SKELETONS)
    test_skels_l = sorted(TEST_SKELETONS)

    # Build pool of all (skel, src_clip) candidates per cluster
    candidates_by_cluster_and_split = defaultdict(lambda: {'train_train': [], 'mixed': [], 'test_test': []})
    for skel, clips in by_skel.items():
        for c in clips:
            if c['cluster'] not in ALL_CLUSTERS: continue
            split_class = 'test_test' if skel in TEST_SKELETONS else 'train_train'
            candidates_by_cluster_and_split[c['cluster']][split_class].append((skel, c))

    def sample_skel_pair_for(split_name, allowed_clusters):
        """Sample (src_skel, src_clip, tgt_skel) for given split + cluster."""
        cluster = py_rng.choice(allowed_clusters)
        if split_name in ('train', 'dev'):
            cands = candidates_by_cluster_and_split[cluster]['train_train']
            if not cands: return None
            src_skel, src_clip = py_rng.choice(cands)
            tgt_skel = py_rng.choice([s for s in train_skels if s != src_skel])
        elif split_name == 'mixed':
            # one held-out, one train
            if py_rng.random() < 0.5:
                cands = candidates_by_cluster_and_split[cluster]['train_train']
                if not cands: return None
                src_skel, src_clip = py_rng.choice(cands)
                tgt_skel = py_rng.choice(test_skels_l)
            else:
                cands = candidates_by_cluster_and_split[cluster]['test_test']
                if not cands: return None
                src_skel, src_clip = py_rng.choice(cands)
                tgt_skel = py_rng.choice(train_skels)
        else:  # test_test
            cands = candidates_by_cluster_and_split[cluster]['test_test']
            if not cands: return None
            src_skel, src_clip = py_rng.choice(cands)
            tgt_skel = py_rng.choice([s for s in test_skels_l if s != src_skel])
        return src_skel, src_clip, tgt_skel

    for split, target_n in TARGETS.items():
        # Cluster availability: skip clusters with no candidates for this split
        if split in ('train', 'dev'):
            available = [c for c in ALL_CLUSTERS if candidates_by_cluster_and_split[c]['train_train']]
        elif split == 'mixed':
            available = [c for c in ALL_CLUSTERS if (
                candidates_by_cluster_and_split[c]['train_train'] or
                candidates_by_cluster_and_split[c]['test_test'])]
        else:
            available = [c for c in ALL_CLUSTERS if candidates_by_cluster_and_split[c]['test_test']]
        if not available:
            print(f"  WARN: split {split} has no available clusters")
            continue
        n_built = 0
        max_tries = target_n * 50  # avoid infinite loop
        tries = 0
        while n_built < target_n and tries < max_tries:
            tries += 1
            res = sample_skel_pair_for(split, available)
            if res is None: continue
            src_skel, src_clip, tgt_skel = res
            q = build_query(src_clip, src_skel, tgt_skel,
                           by_skel_action, by_skel_cluster, by_skel,
                           rng, qid, split)
            # Require at least cluster-tier OR exact-tier eligible (so query has SOMETHING)
            if not (q['cluster_tier_eligible'] or q['exact_tier_eligible']):
                continue
            queries.append(q)
            qid += 1
            n_built += 1
        print(f"  fold {seed} split {split}: built {n_built}/{target_n} ({tries} tries)")

    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    args = parser.parse_args()

    print("Loading clip index...")
    by_skel_action, by_skel_cluster, by_skel = load_clip_index_full()
    all_skels = sorted(by_skel.keys())
    print(f"  {len(all_skels)} skels, {sum(len(c) for c in by_skel.values())} total clips")

    OUT_ROOT.mkdir(exist_ok=True)
    for seed in args.folds:
        print(f"\n=== Building fold {seed} ===")
        queries = build_fold(seed, by_skel_action, by_skel_cluster, by_skel, all_skels)

        # Stats
        from collections import Counter
        cluster_dist = Counter(q['cluster'] for q in queries)
        split_dist = Counter(q['split'] for q in queries)
        ct_eligible = sum(q['cluster_tier_eligible'] for q in queries)
        et_eligible = sum(q['exact_tier_eligible'] for q in queries)
        n_hard_dist = Counter(q['n_hard'] for q in queries)
        print(f"  Total queries: {len(queries)}")
        print(f"  Splits: {dict(split_dist)}")
        print(f"  Clusters: {dict(cluster_dist)}")
        print(f"  Cluster-tier eligible: {ct_eligible}/{len(queries)}")
        print(f"  Exact-tier eligible: {et_eligible}/{len(queries)}")
        print(f"  n_hard distribution: {dict(n_hard_dist)}")

        out_dir = OUT_ROOT / f'fold_{seed}'
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / 'manifest.json', 'w') as f:
            json.dump({
                'version': 'v5.1',
                'seed': seed,
                'all_clusters': ALL_CLUSTERS,
                'test_skeletons': sorted(TEST_SKELETONS),
                'k_hard_max': K_HARD_MAX, 'k_easy_max': K_EASY_MAX,
                'k_distractors': K_DISTRACTORS,
                'n_queries': len(queries),
                'queries': queries,
            }, f, indent=2)
        print(f"  Saved: {out_dir / 'manifest.json'}")


if __name__ == '__main__':
    main()
