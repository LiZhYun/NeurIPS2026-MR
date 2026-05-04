"""Build per-skeleton clip index with cleaned action labels.

Output: eval/benchmark_v3/clip_index.json
Format:
  {
    "Horse": {
      "locomotion": [{"fname": "Horse___Walk_1.npy", "T": 67, "action": "walk"}, ...],
      "combat": [...],
    },
    ...
  }

Plus statistics: skeletons excluded for low resource, clips dropped as 'other', etc.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_v3.action_taxonomy import (
    parse_action_from_filename, action_to_cluster, is_other_label,
)

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
CLIP_META_PATH = PROJECT_ROOT / 'eval/results/effect_cache/clip_metadata.json'
OUT_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'

MIN_FRAMES = 30  # minimum clip length
MIN_CLIPS_PER_ACTION = 2  # skeleton must have ≥2 clips per action to qualify for that action


def main():
    print("Loading data...")
    cond = np.load(COND_PATH, allow_pickle=True).item()
    skel_names = sorted(cond.keys())
    print(f"  {len(skel_names)} skeletons in cond_dict")

    # Use clip_metadata.json if available (already has coarse_label) — but we re-parse from filename for consistency
    clip_files = sorted([f for f in os.listdir(MOTION_DIR) if f.endswith('.npy')])
    print(f"  {len(clip_files)} motion files")

    # Build per-skeleton index
    index = defaultdict(lambda: defaultdict(list))
    n_dropped_other = 0
    n_dropped_short = 0
    n_dropped_no_skel = 0
    n_kept = 0

    for f in clip_files:
        # Identify skeleton: prefix match against cond_dict keys
        # Truebones convention: SkelName___ActionName_NN.npy or SkelName_ActionName_NN.npy
        skel = None
        # Try ___ separator (most reliable)
        if '___' in f:
            candidate = f.split('___')[0]
            if candidate in cond:
                skel = candidate
        # Fallback: longest matching prefix
        if skel is None:
            for s in sorted(skel_names, key=len, reverse=True):
                if f.startswith(s + '___') or f.startswith(s + '_'):
                    skel = s
                    break
        if skel is None:
            n_dropped_no_skel += 1
            continue

        # Length check
        try:
            arr = np.load(MOTION_DIR / f, mmap_mode='r')
            T = arr.shape[0]
        except Exception:
            n_dropped_no_skel += 1
            continue
        if T < MIN_FRAMES:
            n_dropped_short += 1
            continue

        # Action label
        action = parse_action_from_filename(f)
        if is_other_label(action):
            n_dropped_other += 1
            continue
        cluster = action_to_cluster(action)

        index[skel][cluster].append({
            'fname': f,
            'T': int(T),
            'action': action,
            'cluster': cluster,
        })
        n_kept += 1

    print(f"\n=== CLIP INDEX SUMMARY ===")
    print(f"  Kept: {n_kept}")
    print(f"  Dropped (no skel): {n_dropped_no_skel}")
    print(f"  Dropped (T<{MIN_FRAMES}): {n_dropped_short}")
    print(f"  Dropped ('other' label): {n_dropped_other}")

    # Skeleton stats
    skel_stats = {}
    skels_with_no_clips = []
    for skel in skel_names:
        n_clips = sum(len(v) for v in index.get(skel, {}).values())
        n_clusters = len(index.get(skel, {}))
        skel_stats[skel] = {'n_clips': n_clips, 'n_clusters': n_clusters,
                             'clusters': list(index.get(skel, {}).keys())}
        if n_clips == 0:
            skels_with_no_clips.append(skel)

    print(f"\n  Skeletons with ZERO clean clips: {len(skels_with_no_clips)}")
    for s in skels_with_no_clips:
        print(f"    - {s}")

    # Clusters per skeleton
    print(f"\n  Skeletons by cluster count:")
    by_count = defaultdict(list)
    for s, st in skel_stats.items():
        by_count[st['n_clusters']].append(s)
    for c in sorted(by_count.keys(), reverse=True):
        print(f"    {c} clusters: {len(by_count[c])} skels")

    # Cluster coverage across all skeletons
    cluster_coverage = defaultdict(int)
    for s, clusters in index.items():
        for c, clips in clusters.items():
            cluster_coverage[c] += 1
    print(f"\n  Cluster coverage (# skels with ≥1 clip in cluster):")
    for c, n in sorted(cluster_coverage.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n} skels")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'index': {k: dict(v) for k, v in index.items()},
        'skel_stats': skel_stats,
        'cluster_coverage': dict(cluster_coverage),
        'meta': {
            'min_frames': MIN_FRAMES,
            'n_kept': n_kept,
            'n_dropped_other': n_dropped_other,
            'n_dropped_short': n_dropped_short,
            'n_dropped_no_skel': n_dropped_no_skel,
        },
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
