"""Generate oracle/reference baseline predictions on v3 benchmark.

Per Codex audit (2026-04-22): v3 benchmark validity needs upper/lower-bound controls
to verify metrics can detect a clear win when one exists. Adds 3 cheap oracles:

  1. action_oracle    — uses ground-truth src_action to retrieve EXACT-action clip on skel_b
                        (upper bound: should hit AUC ≈ 1.0 if metric works)
  2. self_positive    — submits one of the query's positives as prediction
                        (sanity: must hit AUC = 1.0 by construction)
  3. random_skel_b    — random clip on skel_b (any action) — null lower bound

Output: save/oracles/v3/{name}/fold_{F}/query_NNNN.npy

Then runs eval_v3 with all 3 distance metrics for each.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
CLIP_INDEX_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'
SAVE_ROOT = PROJECT_ROOT / 'save/oracles/v3'
# Override via --benchmark-version v5 to use queries_v5/
QUERIES_DIR_BY_VERSION = {
    'v3': PROJECT_ROOT / 'eval/benchmark_v3/queries',
    'v5': PROJECT_ROOT / 'eval/benchmark_v3/queries_v5',
}


def build_action_lookup():
    """Build {(skel, action) -> [fname, ...]} and {(skel, cluster) -> [fname, ...]}."""
    cidx = json.load(open(CLIP_INDEX_PATH))
    by_action = defaultdict(list)
    by_cluster = defaultdict(list)
    by_skel = defaultdict(list)
    for skel, clusters in cidx['index'].items():
        for cluster, clips in clusters.items():
            for clip in clips:
                by_action[(skel, clip['action'])].append(clip['fname'])
                by_cluster[(skel, cluster)].append(clip['fname'])
                by_skel[skel].append(clip['fname'])
    return by_action, by_cluster, by_skel


def load_motion(fname):
    return np.load(MOTION_DIR / fname).astype(np.float32)


def gen_oracles_for_fold(fold_seed, by_action, by_cluster, by_skel, bench_version='v3'):
    """Generate all oracle predictions for one fold."""
    manifest_path = QUERIES_DIR_BY_VERSION[bench_version] / f'fold_{fold_seed}/manifest.json'
    manifest = json.load(open(manifest_path))
    rng = random.Random(fold_seed)

    suffix = '' if bench_version == 'v3' else f'_{bench_version}'
    out_dirs = {
        f'action_oracle{suffix}': SAVE_ROOT / f'action_oracle{suffix}' / f'fold_{fold_seed}',
        f'self_positive{suffix}': SAVE_ROOT / f'self_positive{suffix}' / f'fold_{fold_seed}',
        f'random_skel_b{suffix}': SAVE_ROOT / f'random_skel_b{suffix}' / f'fold_{fold_seed}',
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_action_exact = n_action_cluster_fb = n_action_random_fb = 0
    # Detect manifest version for positive field naming
    pos_field = 'positives_cluster' if bench_version == 'v5' else 'positives'
    for q in manifest['queries']:
        qid = q['query_id']
        skel_b = q['skel_b']
        src_action = q['src_action']
        cluster = q['cluster']

        # ---- action_oracle: exact-action on skel_b, fall back to cluster, then random ----
        cands = by_action.get((skel_b, src_action), [])
        if cands:
            n_action_exact += 1
        else:
            cands = by_cluster.get((skel_b, cluster), [])
            if cands:
                n_action_cluster_fb += 1
            else:
                cands = by_skel.get(skel_b, [])
                n_action_random_fb += 1
        chosen = rng.choice(cands)
        np.save(out_dirs[f'action_oracle{suffix}'] / f'query_{qid:04d}.npy',
                load_motion(chosen))

        # ---- self_positive: take the FIRST positive's motion ----
        positives = q.get(pos_field, q.get('positives', []))
        if not positives:
            continue
        pos = positives[0]
        np.save(out_dirs[f'self_positive{suffix}'] / f'query_{qid:04d}.npy',
                load_motion(pos['fname']))

        # ---- random_skel_b: any clip on skel_b ----
        cands_all = by_skel.get(skel_b, [])
        if not cands_all:
            print(f"  WARN q{qid}: no clips for skel {skel_b} — skipping random")
            continue
        chosen_r = rng.choice(cands_all)
        np.save(out_dirs[f'random_skel_b{suffix}'] / f'query_{qid:04d}.npy',
                load_motion(chosen_r))

    print(f"fold {fold_seed}:")
    print(f"  action_oracle: {n_action_exact} exact, {n_action_cluster_fb} cluster-fb, {n_action_random_fb} random-fb")
    print(f"  self_positive: 300 positives used")
    print(f"  random_skel_b: 300 random clips")
    return out_dirs


def eval_oracles(out_dirs, fold_seed):
    """Run eval_v3 on each oracle dir for all 3 distance metrics."""
    for name, out_dir in out_dirs.items():
        for distance in ('procrustes', 'zscore_dtw', 'q_component'):
            cmd = [
                sys.executable, '-u', '-m', 'eval.benchmark_v3.eval_v3',
                '--method_dir', str(out_dir),
                '--fold', str(fold_seed),
                '--method_name', f'oracle_{name}',
                '--distance', distance,
            ]
            print(f"  {name} fold={fold_seed} dist={distance}")
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--bench_version', choices=['v3', 'v5'], default='v3')
    args = parser.parse_args()

    by_action, by_cluster, by_skel = build_action_lookup()
    print(f"Action keys: {len(by_action)}, cluster keys: {len(by_cluster)}, skel keys: {len(by_skel)}")

    for fold in args.folds:
        print(f"\n=== Generating oracles for fold {fold} (bench={args.bench_version}) ===")
        out_dirs = gen_oracles_for_fold(fold, by_action, by_cluster, by_skel, args.bench_version)
        if not args.skip_eval:
            print(f"\n=== Eval oracles for fold {fold} ===")
            eval_oracles(out_dirs, fold)


if __name__ == '__main__':
    main()
