"""Build SIF (Source-Instance Fidelity) manifest.

Selects triples (skel_a, skel_b, action) where skel_a has >= 3 distinct source
clips with that action. For each triple, lists all eligible source clips.
Output format compatible with V5 inference drivers.
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

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='eval/benchmark_v3/queries_sif/manifest.json')
    parser.add_argument('--min_src_clips', type=int, default=3)
    parser.add_argument('--max_src_clips', type=int, default=6)
    parser.add_argument('--max_triples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Index all clips by (skel, exact_action)
    by_skel_action = defaultdict(list)
    for skel_dir in sorted(MOTION_DIR.iterdir()):
        if not skel_dir.is_dir():
            # All motions are flat — treat MOTION_DIR as having flat .npy files
            continue
    # Truebones motion structure: flat .npy files named <Skel>___<Action>_<id>.npy
    for fp in MOTION_DIR.glob('*.npy'):
        fname = fp.name
        parts = fname.split('___')
        if len(parts) < 2:
            continue
        skel = parts[0]
        rest = parts[1]
        action = parse_action_from_filename(fname)
        cluster = action_to_cluster(action)
        if cluster is None:
            continue
        by_skel_action[(skel, action)].append(fname)

    print(f"Total (skel, action) cells: {len(by_skel_action)}")
    eligible_cells = {k: v for k, v in by_skel_action.items() if len(v) >= args.min_src_clips}
    print(f"Eligible cells (>= {args.min_src_clips} clips): {len(eligible_cells)}")

    # Pick target skels (any skel with the same action that's not skel_a)
    by_action = defaultdict(set)
    for (skel, action), clips in by_skel_action.items():
        by_action[action].add(skel)

    # Build triples: for each eligible cell (skel_a, action), pick one or more skel_b
    # different from skel_a but with the same action
    triples = []
    sorted_cells = sorted(eligible_cells.items(), key=lambda x: -len(x[1]))
    for (skel_a, action), clips in sorted_cells:
        candidate_bs = sorted(by_action[action] - {skel_a})
        if not candidate_bs:
            continue
        # Pick 1 random skel_b per cell to bound the manifest size
        skel_b = candidate_bs[rng.choice(len(candidate_bs))]
        # Take up to max_src_clips source clips
        sel_clips = clips[:args.max_src_clips]
        triples.append({
            'skel_a': skel_a,
            'skel_b': skel_b,
            'action': action,
            'cluster': action_to_cluster(action),
            'sources': sel_clips,
        })
        if len(triples) >= args.max_triples:
            break

    print(f"Built {len(triples)} triples (mean n_src = {np.mean([len(t['sources']) for t in triples]):.2f})")

    # Flatten into V5-style query list (one query per source clip, but with shared triple_id)
    queries = []
    qid = 0
    for triple_idx, t in enumerate(triples):
        for src_fname in t['sources']:
            sp = MOTION_DIR / src_fname
            try:
                src_T = int(np.load(sp).shape[0])
            except Exception:
                src_T = 32
            # Provide a synthetic 'positives_cluster' with the source T to keep
            # V5 driver crop logic consistent (median over a single value = src_T).
            queries.append({
                'query_id': qid,
                'triple_id': triple_idx,
                'skel_a': t['skel_a'],
                'skel_b': t['skel_b'],
                'cluster': t['cluster'],
                'src_fname': src_fname,
                'src_action': t['action'],
                'src_T': src_T,
                'split': 'sif',
                'positives_cluster': [{'fname': src_fname, 'T': src_T,
                                        'action': t['action'], 'cluster': t['cluster']}],
            })
            qid += 1

    out_dir = (PROJECT_ROOT / args.out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(PROJECT_ROOT / args.out_path, 'w') as f:
        json.dump({
            'version': 'sif_v1',
            'description': 'Source-Instance Fidelity benchmark: multi-source triples',
            'seed': args.seed,
            'min_src_clips': args.min_src_clips,
            'max_src_clips': args.max_src_clips,
            'n_triples': len(triples),
            'n_queries': len(queries),
            'triples': triples,
            'queries': queries,
        }, f, indent=2)
    print(f"Saved: {PROJECT_ROOT / args.out_path}")
    print(f"  {len(triples)} triples, {len(queries)} queries")


if __name__ == '__main__':
    main()
