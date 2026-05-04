"""Per-frame nearest-neighbor retargeting via invariant representation.

Evolution of K_retrieve: instead of picking ONE target clip (whose Q matches source Q),
we retrieve target-skeleton frames INDIVIDUALLY — for each source frame t, find the
target-skeleton frame (from any clip) whose invariant rep is most similar.

This gives:
- Per-frame semantic alignment (no clip-level quantization)
- Target skeleton's natural poses (no IK artifacts)
- Output in target skeleton's coordinate system

Output: eval/results/k_compare/K_frame_nn_200pair/pair_NNNN.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.skel_blind.invariant_dataset import INVARIANT_DIR, TEST_SKELETONS

MANIFEST = PROJECT_ROOT / 'eval/benchmark_paired/pairs/manifest.json'
MOTION_DIR = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'


def load_target_frame_db(tgt_skel, inv_manifest, exclude_fname=None):
    """Build a database of (source_clip_name, frame_idx, inv_frame[32*8], motion_frame[J,13])
    for all frames of all clips of target skeleton.

    Returns: dict with 'inv_flat': [N, 256], 'motion_frames': [N, J, 13], 'sources': [N]
    """
    skel_info = inv_manifest['skeletons'].get(tgt_skel)
    if skel_info is None:
        return None
    inv_data = np.load(os.path.join(INVARIANT_DIR, f'{tgt_skel}.npz'), allow_pickle=True)

    all_inv = []
    all_motion = []
    all_sources = []
    for clip_name in skel_info['clips']:
        if clip_name == exclude_fname:
            continue
        inv_clip = inv_data[clip_name]  # [T, 32, 8]
        # Load motion to get joint positions
        motion_path = MOTION_DIR / f'{clip_name}.npy'
        if not motion_path.exists():
            continue
        motion = np.load(motion_path).astype(np.float32)
        T_inv = inv_clip.shape[0]
        T_motion = motion.shape[0]
        T = min(T_inv, T_motion)
        # Flatten invariant rep to [T, 256]
        inv_flat = inv_clip[:T].reshape(T, -1)
        all_inv.append(inv_flat)
        all_motion.append(motion[:T])
        all_sources.extend([(clip_name, t) for t in range(T)])

    if not all_inv:
        return None
    inv_flat = np.concatenate(all_inv, axis=0)  # [N, 256]
    motion_frames = np.concatenate(all_motion, axis=0)  # [N, J, 13]
    return {
        'inv_flat': inv_flat,
        'motion_frames': motion_frames,
        'sources': all_sources,
    }


def frame_nn_retarget(source_inv_flat, target_db):
    """For each source frame, find the target frame with closest invariant rep.

    Args:
      source_inv_flat: [T_src, 256]
      target_db: dict from load_target_frame_db

    Returns:
      target_motion: [T_src, J, 13]
      nn_sources: [T_src] list of (clip_name, frame_idx)
    """
    tgt_inv = target_db['inv_flat']  # [N, 256]
    tgt_motion = target_db['motion_frames']  # [N, J, 13]

    # Compute pairwise squared L2 distance: [T_src, N]
    # Use chunked computation for memory efficiency
    T_src = source_inv_flat.shape[0]
    N = tgt_inv.shape[0]
    nn_idx = np.zeros(T_src, dtype=np.int64)
    CHUNK = 100  # process 100 source frames at a time
    for start in range(0, T_src, CHUNK):
        end = min(start + CHUNK, T_src)
        src_chunk = source_inv_flat[start:end]  # [chunk, 256]
        # d[i,j] = ||src[i] - tgt[j]||^2 = src[i]^2 + tgt[j]^2 - 2 src[i].tgt[j]
        src_sq = (src_chunk ** 2).sum(axis=1, keepdims=True)  # [chunk, 1]
        tgt_sq = (tgt_inv ** 2).sum(axis=1, keepdims=True).T  # [1, N]
        cross = src_chunk @ tgt_inv.T  # [chunk, N]
        dist = src_sq + tgt_sq - 2 * cross
        nn_idx[start:end] = dist.argmin(axis=1)

    output = tgt_motion[nn_idx]  # [T_src, J, 13]
    nn_sources = [target_db['sources'][i] for i in nn_idx]
    return output, nn_sources


def split_category(skel_a, skel_b):
    a_test = skel_a in TEST_SKELETONS
    b_test = skel_b in TEST_SKELETONS
    if a_test and b_test:
        return "test_test"
    elif a_test or b_test:
        return "mixed"
    return "train_train"


def run(max_pairs=200):
    out_dir = PROJECT_ROOT / 'eval/results/k_compare/K_frame_nn_200pair'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load invariant rep manifest
    with open(os.path.join(INVARIANT_DIR, 'manifest.json')) as f:
        inv_manifest = json.load(f)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    pairs = manifest['pairs'][:max_pairs]
    print(f"Loaded {len(pairs)} pairs")

    # Cache target-skel frame DBs
    db_cache = {}
    per_pair = []
    t_total_0 = time.time()

    for i, p in enumerate(pairs):
        pid = p['pair_id']
        skel_a = p['skel_a']
        skel_b = p['skel_b']
        tgt_fname = p['file_b']
        action = p['action']

        rec = {
            'pair_id': pid, 'action': action,
            'source_skel': skel_a, 'target_skel': skel_b,
            'split': split_category(skel_a, skel_b),
            'status': 'pending',
        }

        try:
            # Build/fetch target frame DB (exclude GT)
            gt_clip_stem = os.path.splitext(tgt_fname)[0]
            cache_key = (skel_b, gt_clip_stem)
            if cache_key not in db_cache:
                db = load_target_frame_db(skel_b, inv_manifest, exclude_fname=gt_clip_stem)
                db_cache[cache_key] = db
            db = db_cache[cache_key]
            if db is None:
                rec['status'] = 'skipped_no_db'
                per_pair.append(rec)
                continue

            # Load source invariant rep from pair file (already encoded)
            pair_data = np.load(PROJECT_ROOT / 'eval/benchmark_paired/pairs' / p['pair_file'])
            inv_a = pair_data['inv_a']  # [T_src, 32, 8]
            T_src = inv_a.shape[0]
            source_inv_flat = inv_a.reshape(T_src, -1)

            # Retarget
            t0 = time.time()
            target_motion, nn_sources = frame_nn_retarget(source_inv_flat, db)
            runtime = time.time() - t0
            rec['runtime_sec'] = runtime
            rec['db_size'] = db['inv_flat'].shape[0]

            # Count clip diversity in output
            unique_clips = len(set(s[0] for s in nn_sources))
            rec['n_unique_source_clips'] = unique_clips

            out_path = out_dir / f'pair_{pid:04d}.npy'
            np.save(out_path, target_motion)
            rec['status'] = 'ok'

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t_total_0
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                print(f"  [{i+1}/{len(pairs)}] {action}: {skel_a}→{skel_b} "
                      f"db={db['inv_flat'].shape[0]} unique_clips={unique_clips}/{T_src} "
                      f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            import traceback
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            print(f"  FAILED pair {pid}: {e}")

        per_pair.append(rec)

    total_time = time.time() - t_total_0
    print(f"\nTotal: {total_time:.0f}s, {sum(1 for r in per_pair if r['status']=='ok')}/{len(per_pair)} OK")

    summary = {
        'method': 'K_frame_nn',
        'n_pairs': len(per_pair),
        'n_ok': sum(1 for r in per_pair if r['status'] == 'ok'),
        'total_time_sec': total_time,
        'per_pair': per_pair,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pairs', type=int, default=200)
    args = parser.parse_args()
    run(max_pairs=args.max_pairs)
