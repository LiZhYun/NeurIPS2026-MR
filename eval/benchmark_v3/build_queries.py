"""Generate v3 benchmark queries with stratified sampling.

Per reviewer feedback (2026-04-18):
  - K=3 length-matched adversarials per query
  - Multiple positives (all target_skel clips with matching cluster)
  - Deduplicate near-clones in positive set
  - 2 folds (seed 42 + 43) for robustness
  - Splits: train(50) / dev(50) / mixed(100) / test_test(100) per fold

Output: eval/benchmark_v3/queries/fold_{seed}/manifest.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_v3.action_taxonomy import ACTION_CLUSTERS, disjoint_cluster_pool

CLIP_INDEX_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'
OUT_ROOT = PROJECT_ROOT / 'eval/benchmark_v3/queries'

# Held-out skeletons (10 for test splits)
TEST_SKELETONS = {
    'Cat', 'Elephant', 'Spider', 'Crab', 'Trex',
    'Raptor', 'Buzzard', 'Alligator', 'Anaconda', 'Rat',
}

# Core clusters — broad coverage across skeletons
CORE_CLUSTERS = ['locomotion', 'combat', 'idle', 'death']

# Target sizes per split (per fold)
TARGETS = {'train': 50, 'dev': 50, 'mixed': 100, 'test_test': 100}

# Min/max clips per positive set (after dedup cap)
MIN_POSITIVES = 1  # allow single-positive queries (rare, but valid)
MAX_POSITIVES = 8

# Number of adversarials per query
K_ADV = 3

# Length matching tolerance for adversarials (±PERCENT of positive median length)
LENGTH_TOL_STRICT = 0.20
LENGTH_TOL_RELAXED = 0.50


def deduplicate_positives(clips, q_sig_lookup=None, threshold=0.95):
    """Greedy near-clone removal via Q-signature cosine similarity, then cap.

    For each clip, keep only if its max cosine sim to already-kept clips < threshold.
    Falls back to filename-sort-and-truncate if Q sigs unavailable.
    """
    if not clips:
        return clips
    if q_sig_lookup is None:
        return sorted(clips, key=lambda c: c['fname'])[:MAX_POSITIVES]

    # Get Q sigs for each clip; fall back to filename order if missing
    sigs = []
    for c in clips:
        sig = q_sig_lookup(c['fname'])
        sigs.append(sig)

    # Greedy dedup
    kept = []
    kept_sigs = []
    for c, sig in zip(clips, sigs):
        if sig is None:
            kept.append(c)
            kept_sigs.append(None)
            continue
        sig_norm = sig / (np.linalg.norm(sig) + 1e-9)
        is_clone = False
        for ks in kept_sigs:
            if ks is None:
                continue
            ks_norm = ks / (np.linalg.norm(ks) + 1e-9)
            if float(sig_norm @ ks_norm) > threshold:
                is_clone = True
                break
        if not is_clone:
            kept.append(c)
            kept_sigs.append(sig)
        if len(kept) >= MAX_POSITIVES:
            break
    return kept


def sample_adversarials(tgt_skel, tgt_cluster, clip_index, pos_median_T, rng,
                        k=K_ADV):
    """Sample K adversarials from target_skel's clips in DIFFERENT clusters."""
    tgt_info = clip_index.get(tgt_skel, {})
    disjoint = disjoint_cluster_pool(tgt_cluster)

    # Pool: all target_skel clips in disjoint clusters
    adv_pool = []
    for c in disjoint:
        adv_pool.extend(tgt_info.get(c, []))

    if not adv_pool:
        return None  # no adversarials possible

    # Length-match: strict first (±20%), fall back to relaxed (±50%)
    for tol in [LENGTH_TOL_STRICT, LENGTH_TOL_RELAXED]:
        length_matched = [
            c for c in adv_pool
            if abs(c['T'] - pos_median_T) <= tol * pos_median_T
        ]
        if len(length_matched) >= k:
            return rng.sample(length_matched, k), tol

    # No length match possible — REJECT this query (per Codex: don't silently relax)
    return None, None


def _build_q_sig_lookup():
    """Returns a function fname → q_signature (or None if not in cache)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from eval.pilot_Q_experiments import q_signature
    qc = np.load(PROJECT_ROOT / 'idea-stage/quotient_cache.npz', allow_pickle=True)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}

    def lookup(fname):
        idx = fname_to_idx.get(fname)
        if idx is None:
            return None
        q = {
            'com_path': qc['com_path'][idx],
            'heading_vel': qc['heading_vel'][idx],
            'contact_sched': qc['contact_sched'][idx],
            'cadence': float(qc['cadence'][idx]),
            'limb_usage': qc['limb_usage'][idx],
        }
        return q_signature(q)
    return lookup


def generate_fold(seed, clip_index, skel_stats, q_sig_lookup=None):
    """Generate one fold of queries with stratified sampling."""
    rng = random.Random(seed)

    all_skels = sorted(clip_index.keys())
    train_skels = [s for s in all_skels if s not in TEST_SKELETONS]
    test_skels = [s for s in all_skels if s in TEST_SKELETONS]

    print(f"\n=== FOLD seed={seed} ===")
    print(f"  Train skels (core): {len(train_skels)}")
    print(f"  Test skels (held-out): {len(test_skels)}")
    print(f"  Core clusters: {CORE_CLUSTERS}")

    # For each split, generate queries
    queries_by_split = defaultdict(list)

    # Build source-target candidate pairs per split
    def eligible_sources(skel_pool, cluster):
        """Skels in pool that have ≥1 clip in this cluster."""
        return [s for s in skel_pool if cluster in clip_index.get(s, {})]

    def sample_for_split(split_name, skel_a_pool, skel_b_pool, n_target):
        """Sample n_target queries with cross-skeleton stratification."""
        candidates = []
        # Enumerate all possible (skel_a, skel_b, cluster) combos with ≥1 clip each
        for cluster in CORE_CLUSTERS:
            eligible_a = eligible_sources(skel_a_pool, cluster)
            eligible_b = eligible_sources(skel_b_pool, cluster)
            for sa in eligible_a:
                for sb in eligible_b:
                    if sa == sb:
                        continue
                    # Need adversarial possible: sb must have clips in a disjoint cluster
                    disjoint = disjoint_cluster_pool(cluster)
                    has_adv = any(c in clip_index.get(sb, {}) and clip_index[sb][c]
                                  for c in disjoint)
                    if not has_adv:
                        continue
                    candidates.append((sa, sb, cluster))

        if not candidates:
            print(f"    {split_name}: NO candidates!")
            return []

        rng.shuffle(candidates)
        # Balance across clusters: round-robin
        by_cluster = defaultdict(list)
        for c in candidates:
            by_cluster[c[2]].append(c)
        for c in by_cluster:
            rng.shuffle(by_cluster[c])

        sampled = []
        cluster_keys = list(by_cluster.keys())
        idx_per_cluster = {c: 0 for c in cluster_keys}
        while len(sampled) < n_target:
            made_progress = False
            for c in cluster_keys:
                if idx_per_cluster[c] < len(by_cluster[c]):
                    sampled.append(by_cluster[c][idx_per_cluster[c]])
                    idx_per_cluster[c] += 1
                    made_progress = True
                    if len(sampled) >= n_target:
                        break
            if not made_progress:
                break
        return sampled[:n_target]

    # Generate sources per split with HARD ISOLATION on (skel_a, skel_b, cluster) triples.
    # Per Codex: don't add to used_triples until query is actually feasible
    # (length-matched adversarials, valid src/tgt clips). Otherwise we waste triples
    # and severely underfill splits even though feasibility upper bound is high.
    used_triples = set()

    def sample_for_split_excl(split_name, skel_a_pool, skel_b_pool):
        """Get all candidate triples for split, EXCLUDING already-used."""
        cands = sample_for_split(split_name, skel_a_pool, skel_b_pool,
                                  10000)  # large to get full enumeration
        return [c for c in cands if c not in used_triples]

    # Order matters: test_test FIRST (most precious data), then mixed, dev, train
    splits_config = [
        ('test_test', test_skels, test_skels),
        ('mixed_A_test', test_skels, train_skels),
        ('mixed_B_test', train_skels, test_skels),
        ('dev', train_skels, train_skels),
        ('train', train_skels, train_skels),
    ]
    splits_targets = {
        'test_test': TARGETS['test_test'],
        'mixed_A_test': TARGETS['mixed'] // 2,
        'mixed_B_test': TARGETS['mixed'] // 2,
        'dev': TARGETS['dev'],
        'train': TARGETS['train'],
    }

    # Build candidate pool for each split (intersection logic deferred to feasibility check)
    split_candidates = {name: sample_for_split_excl(name, sa, sb)
                        for name, sa, sb in splits_config}

    # Construct queries with backfilling. Iterate splits in priority order.
    all_queries = []
    qid = 0

    def try_build_query(sa, sb, cluster, split_label):
        """Try to construct a feasible query. Returns query dict or None."""
        nonlocal qid
        src_clips = clip_index[sa].get(cluster, [])
        tgt_clips = clip_index[sb].get(cluster, [])
        if not src_clips or not tgt_clips:
            return None
        src = rng.choice(src_clips)
        positives = deduplicate_positives(tgt_clips, q_sig_lookup=q_sig_lookup)
        if not positives:
            return None
        pos_median_T = int(np.median([c['T'] for c in positives]))
        advs_result = sample_adversarials(sb, cluster, clip_index, pos_median_T, rng)
        if advs_result[0] is None or len(advs_result[0]) < K_ADV:
            return None
        advs, adv_length_tol = advs_result
        q = {
            'query_id': qid,
            'split': 'mixed' if split_label.startswith('mixed_') else split_label,
            'skel_a': sa, 'skel_b': sb, 'cluster': cluster,
            'src_fname': src['fname'], 'src_T': src['T'], 'src_action': src['action'],
            'positives': [{'fname': p['fname'], 'T': p['T'], 'action': p['action']}
                          for p in positives],
            'n_positives': len(positives),
            'pos_median_T': pos_median_T,
            'adversarials': [{'fname': a['fname'], 'T': a['T'], 'action': a['action'],
                              'cluster': a['cluster']} for a in advs],
            'n_adversarials': len(advs),
            'adv_length_tol': adv_length_tol,
        }
        qid += 1
        return q

    for split_name, _, _ in splits_config:
        target_n = splits_targets[split_name]
        cands = split_candidates[split_name]
        rng.shuffle(cands)
        n_built = 0
        for (sa, sb, cluster) in cands:
            if (sa, sb, cluster) in used_triples:
                continue
            q = try_build_query(sa, sb, cluster, split_name)
            if q is None:
                continue  # try next candidate (don't reserve triple)
            used_triples.add((sa, sb, cluster))
            all_queries.append(q)
            n_built += 1
            if n_built >= target_n:
                break
        print(f"  [{split_name}]: built {n_built}/{target_n} queries")

    # (Query construction now happens in the loop above with backfilling)

    # Report
    print(f"  Generated {len(all_queries)} queries")
    by_split = defaultdict(int)
    for q in all_queries:
        by_split[q['split']] += 1
    for s, n in by_split.items():
        print(f"    {s}: {n}")

    # Save
    out_dir = OUT_ROOT / f'fold_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'seed': seed,
        'k_adversarials': K_ADV,
        'max_positives': MAX_POSITIVES,
        'core_clusters': CORE_CLUSTERS,
        'targets': TARGETS,
        'length_tol': {'strict': LENGTH_TOL_STRICT, 'relaxed': LENGTH_TOL_RELAXED},
        'min_frames': 30,
        'test_skeletons': sorted(TEST_SKELETONS),
        'n_queries': len(all_queries),
        'queries': all_queries,
    }
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {out_dir}/manifest.json")

    return all_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    args = parser.parse_args()

    with open(CLIP_INDEX_PATH) as f:
        data = json.load(f)
    clip_index = data['index']
    skel_stats = data['skel_stats']

    print("Loading Q-signature cache for positive deduplication...")
    q_sig_lookup = _build_q_sig_lookup()

    for seed in args.seeds:
        generate_fold(seed, clip_index, skel_stats, q_sig_lookup=q_sig_lookup)


if __name__ == '__main__':
    main()
