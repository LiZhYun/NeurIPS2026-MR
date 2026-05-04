"""Extend MoReFlow cache files with ACE's prev_row_idx + is_clip_start fields.

Cache extension spec (per ACE_DESIGN_V3 §3.6):
  For each window at position t_start in clip C:
    - prev_chunk = clip[t_start - 16 : t_start + 16]  (32 frames = 8 tokens)
    - prev_row_idx[i] = i - 4 within the same clip's contiguous block (since stride=4)
    - is_clip_start[i] = True if t_start < 16 (no full prior chunk fits)

Index-based (NOT a separate z_prev tensor) — saves disk + prevents cache drift.

The original cache stores meta = list of (clip_fname, t_start) per row in extraction order
(clip-by-clip, ordered by t_start within clip). So within a clip's contiguous rows:
  i corresponds to t_start = some constant per clip (depends on first row's t_start)
  i+1 corresponds to t_start + 4 (stride)
  i-4 corresponds to t_start - 16  ← this is what we want as prev_chunk

Sanity-check at construction time that prev_row_idx[i] indeed corresponds to
prev_meta == (same clip, t_start - 16).

Usage:
  python -m scripts.moreflow_extend_cache_for_ace --scope all
  python -m scripts.moreflow_extend_cache_for_ace --scope train_v3
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_ROOT = PROJECT_ROOT / 'save/moreflow_flow'
STRIDE = 4   # in tokens; matches WINDOW=32 / token_dim=4 (1 token = 4 frames)


def extend_skel(skel_data):
    """Add prev_row_idx + is_clip_start to one skel's cache dict."""
    meta = skel_data['meta']                                              # list of (clip_fname, t_start)
    N = len(meta)
    prev_row_idx = -torch.ones(N, dtype=torch.long)
    is_clip_start = torch.ones(N, dtype=torch.bool)                       # default True

    # Group rows by clip
    clip_to_rows = {}
    for i, (fname, t_start) in enumerate(meta):
        clip_to_rows.setdefault(fname, []).append((i, t_start))

    for fname, rows in clip_to_rows.items():
        # Sort by t_start
        rows_sorted = sorted(rows, key=lambda x: x[1])
        # For each row, find prev_row_idx = row 4 stride positions earlier within this clip
        # In token units, prev = current - 4 tokens = current - 16 frames
        # prev_t_start = current_t_start - 16
        t_to_idx = {t: i for i, t in rows_sorted}
        for i, t_start in rows_sorted:
            prev_t = t_start - 16
            if prev_t < 0 or prev_t not in t_to_idx:
                # No full prior chunk → is_clip_start=True, prev_row_idx=-1
                continue
            prev_row_idx[i] = t_to_idx[prev_t]
            is_clip_start[i] = False

    skel_data['prev_row_idx'] = prev_row_idx
    skel_data['is_clip_start'] = is_clip_start
    return int(is_clip_start.sum().item()), N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['train_v3', 'all'], default='all')
    args = parser.parse_args()

    cache_path = CACHE_ROOT / f'cache_{args.scope}.pt'
    if not cache_path.exists():
        print(f"FATAL: {cache_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {cache_path} ({cache_path.stat().st_size / (1024**2):.0f} MB)...")
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    skels = [k for k in cache.keys() if k != '_meta']
    print(f"Skels in cache: {len(skels)}")

    total_starts = 0
    total_rows = 0
    for skel in skels:
        n_start, n = extend_skel(cache[skel])
        total_starts += n_start
        total_rows += n
        print(f"  {skel}: {n} rows, {n_start} clip-start ({100*n_start/n:.1f}%)")

    print(f"\nTotal: {total_starts}/{total_rows} clip-start rows ({100*total_starts/total_rows:.2f}%)")

    # Save (atomic)
    tmp_path = cache_path.with_suffix('.pt.tmp')
    print(f"Saving extended cache to {cache_path} (atomic via .tmp)...")
    torch.save(cache, tmp_path)
    os.replace(tmp_path, cache_path)
    new_size_mb = cache_path.stat().st_size / (1024**2)
    print(f"New cache size: {new_size_mb:.0f} MB")


if __name__ == '__main__':
    main()
