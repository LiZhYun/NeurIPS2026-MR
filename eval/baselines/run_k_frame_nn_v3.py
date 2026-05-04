"""K_frame_nn baseline on v3 benchmark.

Per-frame nearest-neighbor in invariant rep space.
For each source frame, find target_skel frame with closest invariant rep.
Output target_skel joint positions for that frame.

POOL POLICY (per Codex review): retrieval pool = full target_skel library.
Picking a positive frame = method working correctly (semantic match).
Eval handles fairness via adversarials.

Usage:
  python -m eval.baselines.run_k_frame_nn_v3 --fold 42
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.skel_blind.invariant_dataset import INVARIANT_DIR

PAIRS_DIR_V3 = PROJECT_ROOT / 'eval/benchmark_v3/queries'
MOTION_DIR = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'


def load_target_frame_db(tgt_skel, inv_manifest, exclude_fnames):
    """DB of (clip_name, frame_idx, inv_flat[256], joint_motion[J,13])."""
    skel_info = inv_manifest['skeletons'].get(tgt_skel)
    if skel_info is None:
        return None
    inv_data = np.load(os.path.join(INVARIANT_DIR, f'{tgt_skel}.npz'), allow_pickle=True)
    exclude_stems = {f.rsplit('.', 1)[0] for f in exclude_fnames}

    all_inv, all_motion, all_sources = [], [], []
    for clip_name in skel_info['clips']:
        if clip_name in exclude_stems:
            continue
        inv_clip = inv_data[clip_name]
        motion_path = MOTION_DIR / f'{clip_name}.npy'
        if not motion_path.exists():
            continue
        motion = np.load(motion_path).astype(np.float32)
        T = min(inv_clip.shape[0], motion.shape[0])
        all_inv.append(inv_clip[:T].reshape(T, -1))
        all_motion.append(motion[:T])
        all_sources.extend([(clip_name, t) for t in range(T)])

    if not all_inv:
        return None
    return {
        'inv_flat': np.concatenate(all_inv, axis=0),
        'motion_frames': np.concatenate(all_motion, axis=0),
        'sources': all_sources,
    }


def frame_nn_retarget(source_inv_flat, target_db):
    tgt_inv = target_db['inv_flat']
    tgt_motion = target_db['motion_frames']
    T_src = source_inv_flat.shape[0]
    nn_idx = np.zeros(T_src, dtype=np.int64)
    CHUNK = 100
    for start in range(0, T_src, CHUNK):
        end = min(start + CHUNK, T_src)
        src_chunk = source_inv_flat[start:end]
        src_sq = (src_chunk ** 2).sum(axis=1, keepdims=True)
        tgt_sq = (tgt_inv ** 2).sum(axis=1, keepdims=True).T
        cross = src_chunk @ tgt_inv.T
        dist = src_sq + tgt_sq - 2 * cross
        nn_idx[start:end] = dist.argmin(axis=1)
    return tgt_motion[nn_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_queries', type=int, default=10000)
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = PAIRS_DIR_V3 / f'fold_{args.fold}/manifest.json'
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / f'eval/results/baselines/k_frame_nn/v3_fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")

    print("Loading invariant manifest...")
    with open(os.path.join(INVARIANT_DIR, 'manifest.json')) as f:
        inv_manifest = json.load(f)

    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"Running {len(queries)} queries")

    # Need source's invariant rep for each query
    inv_data_cache = {}

    def get_source_inv(skel_name, src_fname):
        if skel_name not in inv_data_cache:
            inv_data_cache[skel_name] = np.load(
                os.path.join(INVARIANT_DIR, f'{skel_name}.npz'), allow_pickle=True)
        stem = src_fname.rsplit('.', 1)[0]
        if stem not in inv_data_cache[skel_name]:
            return None
        return inv_data_cache[skel_name][stem]

    db_cache = {}
    per_query = []
    t0 = time.time()

    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_a = q['skel_a']
        skel_b = q['skel_b']
        src_fname = q['src_fname']
        cluster = q['cluster']
        split = q['split']

        rec = {'query_id': qid, 'cluster': cluster, 'split': split,
               'skel_a': skel_a, 'skel_b': skel_b, 'status': 'pending'}

        try:
            # Per Codex review: do NOT exclude positives — same-cluster clips are
            # the right answer for retrieval. Per-frame NN picking from positives
            # = method working correctly. Eval handles fairness via adversarials.
            cache_key = skel_b
            if cache_key not in db_cache:
                db = load_target_frame_db(skel_b, inv_manifest, exclude_fnames=set())
                db_cache[cache_key] = db
            db = db_cache[cache_key]
            if db is None:
                rec['status'] = 'skipped_no_db'
                per_query.append(rec)
                continue

            inv_a = get_source_inv(skel_a, src_fname)
            if inv_a is None:
                rec['status'] = 'skipped_no_src_inv'
                per_query.append(rec)
                continue

            T_src = inv_a.shape[0]
            source_inv_flat = inv_a.reshape(T_src, -1)
            target_motion = frame_nn_retarget(source_inv_flat, db)

            # Telemetry: which positives/adversarials did we pick frames from?
            positive_stems = {p['fname'].rsplit('.', 1)[0] for p in q['positives']}
            adv_stems = {a['fname'].rsplit('.', 1)[0] for a in q['adversarials']}

            np.save(out_dir / f'query_{qid:04d}.npy', target_motion.astype(np.float32))
            rec['status'] = 'ok'
            rec['db_size'] = int(db['inv_flat'].shape[0])
            rec['T_out'] = int(target_motion.shape[0])
            # NN frame source clips per output frame
            n_frames_in_pos = sum(1 for s in db['sources'][:T_src]
                                   if s[0] in positive_stems)
            n_frames_in_adv = sum(1 for s in db['sources'][:T_src]
                                   if s[0] in adv_stems)
            rec['picked_in_positives_frames'] = n_frames_in_pos
            rec['picked_in_adversarials_frames'] = n_frames_in_adv

            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(queries) - i - 1)
                print(f"  [{i+1}/{len(queries)}] {cluster}/{split} {skel_a}→{skel_b} "
                      f"db={rec['db_size']} (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            print(f"  FAILED query {qid}: {e}")

        per_query.append(rec)

    total_time = time.time() - t0
    n_ok = sum(1 for r in per_query if r['status'] == 'ok')
    print(f"\nTotal: {total_time:.0f}s, {n_ok}/{len(per_query)} OK")

    summary = {
        'method': 'K_frame_nn_v3',
        'manifest': str(manifest_path),
        'n_queries': len(per_query),
        'n_ok': n_ok,
        'total_time_sec': total_time,
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {out_dir}/metrics.json")


if __name__ == '__main__':
    main()
