"""K_dtw_align: K_retrieve + DTW alignment to source.

Final evolution. Combines:
  1. Clip-level Q-similarity retrieval (target clip with best Q match)
  2. Frame-level DTW alignment of retrieved clip to source's invariant rep
  3. Output is the retrieved clip's joint positions, re-timed to match source

This:
  - Preserves natural target motion (single clip, no jitter)
  - Aligns timing to source's invariant rep
  - Should beat both K_retrieve (better timing) and K_frame_nn (less jitter)

Output: eval/results/k_compare/K_dtw_align_200pair/pair_NNNN.npy
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.skel_blind.invariant_dataset import INVARIANT_DIR

PAIRS_DIR = PROJECT_ROOT / 'eval/benchmark_paired/pairs'
MOTION_DIR = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
K_RETRIEVE_METRICS = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_200pair/metrics.json'


def dtw_align_path(src, tgt):
    """Compute DTW alignment between src [T_s, D] and tgt [T_t, D].

    Returns: alignment list [(s_idx, t_idx)] giving the warping path.
    """
    T_s, T_t = src.shape[0], tgt.shape[0]
    D = np.full((T_s + 1, T_t + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    # Cost matrix
    for i in range(1, T_s + 1):
        for j in range(1, T_t + 1):
            c = float(np.linalg.norm(src[i-1] - tgt[j-1]))
            D[i, j] = c + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
    # Backtrack
    path = []
    i, j = T_s, T_t
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        choices = [(D[i-1, j-1], i-1, j-1), (D[i-1, j], i-1, j), (D[i, j-1], i, j-1)]
        choices.sort()
        _, i, j = choices[0]
    path.reverse()
    return path


def warp_motion_to_source(tgt_motion, alignment, T_src):
    """Re-time target motion to T_src frames using DTW alignment.
    For each source frame s, take the median of all target frames aligned to s."""
    out = np.zeros((T_src,) + tgt_motion.shape[1:], dtype=tgt_motion.dtype)
    counts = np.zeros(T_src, dtype=np.int64)
    accum = np.zeros((T_src,) + tgt_motion.shape[1:], dtype=np.float32)
    for s, t in alignment:
        accum[s] += tgt_motion[t]
        counts[s] += 1
    for s in range(T_src):
        if counts[s] > 0:
            out[s] = accum[s] / counts[s]
        else:
            # Fallback: use closest source frame's value
            nearby = [s2 for s2 in range(T_src) if counts[s2] > 0]
            if nearby:
                closest = min(nearby, key=lambda x: abs(x - s))
                out[s] = out[closest] if counts[closest] > 0 else accum[closest] / counts[closest]
    return out


def run(max_pairs=200):
    out_dir = PROJECT_ROOT / 'eval/results/k_compare/K_dtw_align_200pair'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load K_retrieve metrics to get retrieved file per pair
    with open(K_RETRIEVE_METRICS) as f:
        kr_metrics = json.load(f)
    pid_to_retrieved = {r['pair_id']: r.get('retrieved_fname')
                        for r in kr_metrics['per_pair']
                        if r.get('retrieved_fname')}

    with open(PAIRS_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    pairs = manifest['pairs'][:max_pairs]

    # Load invariant manifest for retrieved clip's invariant rep
    with open(os.path.join(INVARIANT_DIR, 'manifest.json')) as f:
        inv_manifest = json.load(f)

    # Build per-skel inv-rep cache lazily
    inv_data_cache = {}

    per_pair = []
    t_total_0 = time.time()

    for i, p in enumerate(pairs):
        pid = p['pair_id']
        skel_b = p['skel_b']
        action = p['action']

        rec = {'pair_id': pid, 'action': action, 'target_skel': skel_b, 'status': 'pending'}

        if pid not in pid_to_retrieved:
            rec['status'] = 'no_retrieved'
            per_pair.append(rec)
            continue

        retrieved_fname = pid_to_retrieved[pid]
        retrieved_stem = os.path.splitext(retrieved_fname)[0]

        try:
            # Load source inv rep from pair file
            pair_data = np.load(PAIRS_DIR / p['pair_file'])
            inv_a = pair_data['inv_a']  # [T_src, 32, 8]
            T_src = inv_a.shape[0]

            # Load retrieved motion
            retrieved_motion = np.load(MOTION_DIR / retrieved_fname).astype(np.float32)
            T_tgt = retrieved_motion.shape[0]

            # Load retrieved invariant rep
            if skel_b not in inv_data_cache:
                inv_data_cache[skel_b] = np.load(
                    os.path.join(INVARIANT_DIR, f'{skel_b}.npz'), allow_pickle=True)
            inv_data = inv_data_cache[skel_b]
            if retrieved_stem not in inv_data:
                rec['status'] = 'no_inv_for_retrieved'
                per_pair.append(rec)
                continue
            retrieved_inv = inv_data[retrieved_stem]  # [T_tgt, 32, 8]
            T_tgt_inv = retrieved_inv.shape[0]
            # Align to motion length (in case mismatch)
            T_align = min(T_tgt, T_tgt_inv)
            retrieved_motion = retrieved_motion[:T_align]
            retrieved_inv = retrieved_inv[:T_align]

            # DTW align source inv → retrieved inv
            src_flat = inv_a.reshape(T_src, -1)
            tgt_flat = retrieved_inv.reshape(T_align, -1)
            alignment = dtw_align_path(src_flat, tgt_flat)

            # Warp retrieved motion to source's timing
            output = warp_motion_to_source(retrieved_motion, alignment, T_src)

            out_path = out_dir / f'pair_{pid:04d}.npy'
            np.save(out_path, output)
            rec['status'] = 'ok'
            rec['retrieved_fname'] = retrieved_fname
            rec['T_src'] = T_src
            rec['T_tgt'] = T_align

            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - t_total_0
                print(f"  [{i+1}/{len(pairs)}] {action} {skel_b} T_src={T_src} T_tgt={T_align}")

        except Exception as e:
            import traceback
            rec['status'] = 'failed'
            rec['error'] = str(e)
            print(f"  FAILED pair {pid}: {e}")

        per_pair.append(rec)

    total_time = time.time() - t_total_0
    print(f"\nTotal: {total_time:.0f}s, {sum(1 for r in per_pair if r['status']=='ok')}/{len(per_pair)} OK")

    summary = {
        'method': 'K_dtw_align',
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
