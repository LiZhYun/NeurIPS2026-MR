"""V5-full benchmark: enumerate ALL valid (skel_a, src_clip, skel_b) queries.

Where the existing build_queries_v5 samples 300 queries per fold for cluster
balance, V5-full enumerates EVERY (src_clip, tgt_skel) pair with:
- skel_a != skel_b
- at least one same-cluster positive in tgt_skel library

This produces ~30k cluster-tier queries spanning every cross-skeleton pair on
Truebones, suitable for testing whether retrieval methods generalize at scale.

Splits are derived from the existing test_test held-out skel set.

Usage:
    conda run -n anytop python -m eval.benchmark_v3.build_queries_v5_full \
        --output_dir eval/benchmark_v3/queries_v5_full
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.benchmark_v3.build_queries_v5 import (
    load_clip_index_full, build_query, TEST_SKELETONS,
)
TEST_SKELS_V3 = TEST_SKELETONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'eval/benchmark_v3/queries_v5_full'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_per_skel_a', type=int, default=0,
                        help='Optional cap on src clips per skel_a (0 = no cap)')
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Load clip index, group by various keys
    by_skel_action, by_skel_cluster, by_skel = load_clip_index_full()
    all_skels = sorted(by_skel.keys())
    test_skels = set(TEST_SKELS_V3)

    queries = []
    qid = 0
    n_skels = len(all_skels)
    n_pairs_attempted = 0
    n_pairs_with_positives = 0

    for sa_idx, skel_a in enumerate(sorted(all_skels)):
        src_clips = by_skel.get(skel_a, [])
        if args.max_per_skel_a > 0 and len(src_clips) > args.max_per_skel_a:
            idx = rng.choice(len(src_clips), args.max_per_skel_a, replace=False)
            src_clips = [src_clips[i] for i in idx]
        for src_clip in src_clips:
            for skel_b in sorted(all_skels):
                if skel_b == skel_a: continue
                n_pairs_attempted += 1
                # Determine split: test_test if both held-out, mixed if exactly one,
                # train_train otherwise
                a_held = skel_a in test_skels
                b_held = skel_b in test_skels
                if a_held and b_held:
                    split = 'test_test'
                elif a_held or b_held:
                    split = 'mixed'
                else:
                    split = 'train_train'
                # Quick filter: must have at least one same-cluster positive on tgt
                src_cluster = src_clip['cluster']
                if not by_skel_cluster.get((skel_b, src_cluster), []):
                    continue
                n_pairs_with_positives += 1
                q = build_query(src_clip, skel_a, skel_b,
                                by_skel_action, by_skel_cluster, by_skel,
                                rng, qid, split)
                queries.append(q)
                qid += 1
        if (sa_idx + 1) % 10 == 0:
            print(f'  skel_a {sa_idx + 1}/{n_skels} processed, {len(queries)} queries built')

    # Stats
    by_split = defaultdict(int); by_cluster_split = defaultdict(int)
    n_cluster_eligible = 0; n_exact_eligible = 0
    for q in queries:
        by_split[q['split']] += 1
        by_cluster_split[(q['split'], q['cluster'])] += 1
        if q['cluster_tier_eligible']: n_cluster_eligible += 1
        if q['exact_tier_eligible']: n_exact_eligible += 1

    print(f'\nTotal queries: {len(queries)}')
    print(f'  pairs_attempted: {n_pairs_attempted}, pairs_with_positives: {n_pairs_with_positives}')
    print(f'  cluster_tier_eligible: {n_cluster_eligible}')
    print(f'  exact_tier_eligible: {n_exact_eligible}')
    print(f'  by split: {dict(by_split)}')

    manifest = {
        'version': 'v5_full',
        'description': 'Comprehensive enumeration of all (src_clip, tgt_skel) cross-skel queries with at least one same-cluster positive',
        'seed': args.seed,
        'n_queries': len(queries),
        'cluster_eligible': n_cluster_eligible,
        'exact_eligible': n_exact_eligible,
        'by_split': dict(by_split),
        'test_skeletons': list(test_skels),
        'queries': queries,
    }
    out_path = out / 'manifest.json'
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
