"""Pre-extract per-skel windowed z-tokens + Φ_c features for MoReFlow Stage B.

For each skel in scope, walks all training clips (respecting Stage A train/val split),
extracts every 32-frame window with stride=4, encodes through frozen Stage A → caches:
  - z_continuous[skel] : Tensor[N_windows, 8, codebook_dim]   STE-quantized z (continuous embedding)
  - z_indices[skel]    : LongTensor[N_windows, 8]              codebook indices
  - phi[skel][c_type]  : Tensor[N_windows, D_COND_PADDED=24]   numpy Φ_c values (precomputed for coupling)
  - meta[skel]         : list of (clip_fname, t_start) for debug + leak checks

Output: save/moreflow_flow/cache_<scope>.pt (single file per scope).

Usage:
  python -m scripts.moreflow_extract_windows --scope train_v3
  python -m scripts.moreflow_extract_windows --scope all
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import (
    OBJECT_SUBSETS_DICT, V3_TEST_SKELETONS,
)
from model.moreflow.stage_a_registry import StageARegistry
from eval.moreflow_phi import (
    CONDITIONS, D_COND_PADDED,
    phi as phi_np, precompute_skel_descriptors, load_contact_groups,
)

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
SAVE_ROOT = PROJECT_ROOT / 'save/moreflow_flow'
WINDOW = 32
STRIDE = 4


def list_skel_motions(skel_name):
    return sorted([f for f in MOTION_DIR.iterdir()
                   if (f.name.startswith(skel_name + '___') or
                       f.name.startswith(skel_name + '_'))
                   and f.suffix == '.npy'])


def get_train_clip_split(skel_name, n_all, val_frac, seed=42):
    """Re-derive Stage A's per-skel train/val split (clip-level).

    Mirrors the logic in train/train_moreflow_vqvae.py:train_one_skel.
    Returns set of train clip indices.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_all)
    if n_all >= 3 and val_frac > 0:
        n_val = max(1, int(round(n_all * val_frac)))
        n_val = min(n_val, n_all - 2)
        val_idx = set(perm[:n_val].tolist())
        return [i for i in range(n_all) if i not in val_idx]
    else:
        return list(range(n_all))


def extract_one_skel(skel_name, registry, skel_descs, cond_dict, args):
    """Extract all train windows for one skel.

    Returns dict: {z_continuous, z_indices, phi_<c>, meta}
    """
    # Stage A args.json carries the val_frac actually used
    args_json_path = PROJECT_ROOT / f'save/moreflow_vqvae/{skel_name}/args.json'
    with open(args_json_path) as f:
        a = json.load(f)
    val_frac = float(a.get('val_frac', 0.10))
    seed = int(a.get('seed', 42))

    # Find all clips
    all_clips = list_skel_motions(skel_name)
    if not all_clips:
        return None
    # Filter to clips ≥ WINDOW frames
    valid_clips = []
    for c in all_clips:
        m = np.load(c)
        if m.shape[0] >= WINDOW:
            valid_clips.append(c)
    if not valid_clips:
        return None
    n_all = len(valid_clips)

    # Train clips only (respect Stage A split)
    train_indices = get_train_clip_split(skel_name, n_all, val_frac, seed=seed)
    train_clips = [valid_clips[i] for i in train_indices]

    # Normalize via Stage A's mean/std
    mean = registry.get(skel_name)['mean']
    std = registry.get(skel_name)['std']
    desc = skel_descs[skel_name]

    all_z = []
    all_idx = []
    all_phi = {c: [] for c in CONDITIONS}
    all_meta = []

    for clip_path in train_clips:
        motion = np.load(clip_path).astype(np.float32)
        T = motion.shape[0]
        n_windows = (T - WINDOW) // STRIDE + 1
        for i in range(n_windows):
            t_start = i * STRIDE
            window = motion[t_start:t_start + WINDOW]                          # [WINDOW, J, 13]
            # Encode (with normalization) through Stage A
            window_t = torch.from_numpy(window).to(registry.device).float()
            window_norm = (window_t - mean) / std
            window_norm = torch.nan_to_num(window_norm)
            with torch.no_grad():
                z, idx = registry.encode_window(skel_name, window_norm)
            all_z.append(z.squeeze(0).cpu())                                    # [8, codebook_dim]
            all_idx.append(idx.squeeze(0).cpu())                                # [8]
            # Φ_c values (numpy, on physical-units window)
            for c in CONDITIONS:
                all_phi[c].append(torch.from_numpy(phi_np(window, c, desc)))    # [24]
            all_meta.append((clip_path.name, int(t_start)))

    if not all_z:
        return None
    out = {
        'z_continuous': torch.stack(all_z),                                     # [N, 8, codebook_dim]
        'z_indices': torch.stack(all_idx),                                      # [N, 8]
        'meta': all_meta,
    }
    for c in CONDITIONS:
        out[f'phi_{c}'] = torch.stack(all_phi[c])                                # [N, 24]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', type=str, default='train_v3',
                        choices=['train_v3', 'test_v3', 'all'],
                        help='train_v3=60 inductive train; test_v3=10 held-out test; all=70')
    parser.add_argument('--out', type=str, default=None,
                        help='Override output filename (default: cache_<scope>.pt)')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_ROOT / (args.out or f'cache_{args.scope}.pt')
    if args.skip_existing and out_path.exists():
        print(f"Output {out_path} exists; skipping (--skip_existing).")
        return

    # Pick skels
    if args.scope == 'train_v3':
        skels = list(OBJECT_SUBSETS_DICT['train_v3'])
    elif args.scope == 'test_v3':
        skels = list(OBJECT_SUBSETS_DICT['test_v3'])
    elif args.scope == 'all':
        skels = list(OBJECT_SUBSETS_DICT['train_v3']) + list(OBJECT_SUBSETS_DICT['test_v3'])
    else:
        raise ValueError(args.scope)
    print(f"Scope: {args.scope} ({len(skels)} skels)")

    print("Loading cond + contact_groups + Stage A registry...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond_dict, contact_groups)
    registry = StageARegistry(skels, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Extract per-skel
    cache = {}
    t0 = time.time()
    for i, skel in enumerate(skels):
        if skel not in registry.tokenizers:
            print(f"[{i+1}/{len(skels)}] {skel}: SKIP (no tokenizer)")
            continue
        if skel_descs[skel]['n_ee'] == 0:
            print(f"[{i+1}/{len(skels)}] {skel}: WARN no EEs (cond will be partially zero)")
        try:
            data = extract_one_skel(skel, registry, skel_descs, cond_dict, args)
            if data is None:
                print(f"[{i+1}/{len(skels)}] {skel}: SKIP (no train windows)")
                continue
            cache[skel] = data
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(skels) - i - 1)
            print(f"[{i+1}/{len(skels)}] {skel}: {data['z_continuous'].shape[0]} windows "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
        except Exception as e:
            print(f"[{i+1}/{len(skels)}] {skel}: FAILED ({type(e).__name__}: {e})")
            import traceback
            traceback.print_exc()

    # Save
    cache['_meta'] = {
        'scope': args.scope,
        'window': WINDOW,
        'stride': STRIDE,
        'n_skels': len(cache) - 0,  # subtract _meta added below
        'skel_list': list(cache.keys()),
    }
    print(f"\nSaving cache → {out_path}")
    tmp_path = out_path.with_suffix('.pt.tmp')
    torch.save(cache, tmp_path)
    import os
    os.replace(tmp_path, out_path)
    total_windows = sum(v['z_continuous'].shape[0] for k, v in cache.items() if k != '_meta')
    print(f"Total windows across {cache['_meta']['n_skels']} skels: {total_windows}")
    print(f"Cache file size: {out_path.stat().st_size / (1024**3):.2f} GB")


if __name__ == '__main__':
    main()
