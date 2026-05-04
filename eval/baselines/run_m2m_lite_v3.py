"""M2M-lite baseline on v3 benchmark.

Wraps `eval/motion2motion_run.py` (our reimplementation of Motion2Motion's
core sparse-correspondence + patch-matching algorithm) for v3 manifests.

For each query:
  1. Load source motion (skel_a, [T_s, J_s, 13])
  2. Pick example/scaffold motion (skel_b, [T_e, J_t, 13]) — full library
     (per Codex: don't exclude positives; pool = all target_skel clips)
  3. Auto-generate sparse correspondences via contact_groups heuristic
  4. Run M2M-lite patch matching → output [T_s, J_t, 13]

Output: eval/results/baselines/m2m_lite/v3_fold_{42,43}/query_NNNN.npy
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

MOTION_DIR = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'


def list_target_skel_motions(skel_name, motion_dir):
    return sorted([f for f in os.listdir(motion_dir)
                   if (f.startswith(skel_name + '___') or f.startswith(skel_name + '_'))
                   and f.endswith('.npy')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu',
                        help='M2M-lite is lightweight CPU-friendly')
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/queries/fold_{args.fold}/manifest.json'
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / f'eval/results/baselines/m2m_lite/v3_fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")

    # Imports
    from eval.motion2motion_run import (
        author_sparse_mapping, run_m2m_lite,
    )
    from eval.run_k_pipeline_200pairs import load_assets

    cond, contact_groups, _ = load_assets()

    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"Running {len(queries)} queries")

    # Cache per-target-skel example pool
    tgt_clip_cache = {}
    # Per-Codex: per-query deterministic seed derived from (fold, query_id, skel_b)
    # to avoid manifest-order-dependent scaffold choice
    import hashlib

    def per_query_seed(fold, qid, skel_b):
        s = f"{fold}_{qid}_{skel_b}".encode()
        h = hashlib.md5(s).hexdigest()
        return int(h[:8], 16)

    per_query = []
    t_total = time.time()

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
            # 1. Load source motion
            src_motion = np.load(MOTION_DIR / src_fname).astype(np.float32)

            # 2. Pick scaffold (example) motion from target_skel library
            # Per Codex: EXCLUDE positives + adversarials from scaffold pool
            # (otherwise M2M deforms a positive → unfair leakage).
            # Scaffold is a "side-channel" for M2M's patch matching, not the prediction target.
            if skel_b not in tgt_clip_cache:
                tgt_clip_cache[skel_b] = list_target_skel_motions(skel_b, MOTION_DIR)
                for f in tgt_clip_cache[skel_b]:
                    leading = f.split('___')[0] if '___' in f else f.split('_')[0]
                    assert leading == skel_b, f"Pool pollution: {f} not from {skel_b}"
            full_pool = tgt_clip_cache[skel_b]
            forbidden = ({p['fname'] for p in q['positives']}
                         | {a['fname'] for a in q['adversarials']})
            pool = [f for f in full_pool if f not in forbidden]
            if not pool:
                # Fallback: very small skel library — use full pool (disclose in record)
                pool = full_pool
                rec['scaffold_fallback'] = 'positive_excluded_empty'
            # Per-query deterministic seed (not dependent on iteration order)
            qseed = per_query_seed(args.fold, qid, skel_b)
            qrng = np.random.RandomState(qseed)
            ex_fname = pool[qrng.randint(0, len(pool))]
            ex_motion = np.load(MOTION_DIR / ex_fname).astype(np.float32)
            # Defensive min-length check (M2M PATCH_SIZE+2 ≈ 13)
            if ex_motion.shape[0] < 13:
                pool_long = [f for f in pool
                             if np.load(MOTION_DIR / f, mmap_mode='r').shape[0] >= 13]
                if not pool_long:
                    raise RuntimeError(f'No target clips ≥13 frames for {skel_b}')
                ex_fname = pool_long[qrng.randint(0, len(pool_long))]
                ex_motion = np.load(MOTION_DIR / ex_fname).astype(np.float32)

            # 3. Auto-generate sparse correspondences via contact_groups
            if skel_a not in contact_groups or skel_b not in contact_groups:
                rec['status'] = 'skipped_no_cg'
                per_query.append(rec)
                continue

            pairs_ij, mapping_desc = author_sparse_mapping(
                skel_a, skel_b, cond, contact_groups, max_pairs=6)
            if not pairs_ij:
                rec['status'] = 'skipped_no_mapping'
                per_query.append(rec)
                continue
            src_j_idxs = [p[0] for p in pairs_ij]
            tgt_j_idxs = [p[1] for p in pairs_ij]

            # 4. Run M2M-lite patch matching (outputs T_s, J_t, 13)
            t0 = time.time()
            output = run_m2m_lite(
                src_motion, ex_motion, src_j_idxs, tgt_j_idxs,
                tgt_n_joints=ex_motion.shape[1],
                device=args.device, seed=qseed)  # per-query seed for determinism
            runtime = time.time() - t0

            # Per Codex: match output length to query's pos_median_T (target task length).
            # AnyTop uses pos_median_T as output length. M2M natively returns T_s, so we
            # interpolate to pos_median_T for fair task semantics.
            target_len = q['pos_median_T']
            if output.shape[0] != target_len:
                from scipy.interpolate import interp1d
                T_in = output.shape[0]
                # Linear interp along time axis for each (J, C) channel
                xs_in = np.linspace(0, 1, T_in)
                xs_out = np.linspace(0, 1, target_len)
                output_resamp = np.zeros((target_len,) + output.shape[1:], dtype=np.float32)
                for j in range(output.shape[1]):
                    for c in range(output.shape[2]):
                        f_interp = interp1d(xs_in, output[:, j, c], kind='linear',
                                            assume_sorted=True)
                        output_resamp[:, j, c] = f_interp(xs_out)
                output = output_resamp

            np.save(out_dir / f'query_{qid:04d}.npy', output.astype(np.float32))
            rec['status'] = 'ok'
            rec['scaffold_fname'] = ex_fname
            rec['n_pairs'] = len(pairs_ij)
            rec['mapping_desc'] = mapping_desc
            rec['runtime_sec'] = runtime
            rec['T_src'] = int(src_motion.shape[0])
            rec['T_target'] = int(target_len)
            rec['T_out'] = int(output.shape[0])

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - t_total
                eta = elapsed / (i + 1) * (len(queries) - i - 1)
                print(f"  [{i+1}/{len(queries)}] {cluster}/{split} {skel_a}→{skel_b} "
                      f"pairs={len(pairs_ij)} T={rec['T_out']} "
                      f"({runtime:.1f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            print(f"  FAILED query {qid}: {e}")

        per_query.append(rec)

    total_time = time.time() - t_total
    n_ok = sum(1 for r in per_query if r['status'] == 'ok')
    print(f"\nTotal: {total_time:.0f}s, {n_ok}/{len(per_query)} OK")

    summary = {
        'method': 'M2M_lite_v3',
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
