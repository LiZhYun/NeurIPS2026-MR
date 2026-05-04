"""M2M-lite on v5 benchmark — adapted from run_m2m_lite_v3.py.

V5 manifest schema differs from V3:
  V3: q['positives'], q['adversarials']
  V5: q['positives_cluster'], q['positives_exact'],
      q['adversarials_easy'], q['adversarials_hard'],
      q['distractors_same_target_skel']

For scaffold pool exclusion (avoid leakage), forbidden = union of all
candidate clips used by the v5 evaluator.

Output length: V5 has no `pos_median_T`. Use median of positives_cluster's T
field instead (falls back to src_T).

Usage:
  python -m eval.baselines.run_m2m_lite_v5 --fold 42
"""
from __future__ import annotations
import argparse
import hashlib
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


def per_query_seed(fold, qid, skel_b):
    s = f"{fold}_{qid}_{skel_b}".encode()
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--out_tag', type=str, default='m2m_lite_v5')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/queries_v5/fold_{args.fold}/manifest.json'
    out_dir = PROJECT_ROOT / f'eval/results/baselines/{args.out_tag}/fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")

    from eval.motion2motion_run import author_sparse_mapping, run_m2m_lite
    from eval.run_k_pipeline_200pairs import load_assets

    cond, contact_groups, _ = load_assets()

    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"Running {len(queries)} queries")

    tgt_clip_cache = {}
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
            src_motion = np.load(MOTION_DIR / src_fname).astype(np.float32)

            if skel_b not in tgt_clip_cache:
                tgt_clip_cache[skel_b] = list_target_skel_motions(skel_b, MOTION_DIR)
            full_pool = tgt_clip_cache[skel_b]

            # V5 forbidden set: everything in the eval candidate pool
            forbidden = set()
            for key in ('positives_cluster', 'positives_exact',
                        'adversarials_easy', 'adversarials_hard',
                        'distractors_same_target_skel'):
                for x in q.get(key, []):
                    forbidden.add(x['fname'])
            pool = [f for f in full_pool if f not in forbidden]
            if not pool:
                pool = full_pool
                rec['scaffold_fallback'] = 'positive_excluded_empty'

            qseed = per_query_seed(args.fold, qid, skel_b)
            qrng = np.random.RandomState(qseed)
            ex_fname = pool[qrng.randint(0, len(pool))]
            ex_motion = np.load(MOTION_DIR / ex_fname).astype(np.float32)
            if ex_motion.shape[0] < 13:
                pool_long = [f for f in pool
                             if np.load(MOTION_DIR / f, mmap_mode='r').shape[0] >= 13]
                if not pool_long:
                    raise RuntimeError(f'No target clips ≥13 frames for {skel_b}')
                ex_fname = pool_long[qrng.randint(0, len(pool_long))]
                ex_motion = np.load(MOTION_DIR / ex_fname).astype(np.float32)

            if skel_a not in contact_groups or skel_b not in contact_groups:
                rec['status'] = 'skipped_no_cg'
                per_query.append(rec); continue

            pairs_ij, mapping_desc = author_sparse_mapping(
                skel_a, skel_b, cond, contact_groups, max_pairs=6)
            if not pairs_ij:
                rec['status'] = 'skipped_no_mapping'
                per_query.append(rec); continue
            src_j_idxs = [p[0] for p in pairs_ij]
            tgt_j_idxs = [p[1] for p in pairs_ij]

            t0 = time.time()
            output = run_m2m_lite(
                src_motion, ex_motion, src_j_idxs, tgt_j_idxs,
                tgt_n_joints=ex_motion.shape[1],
                device=args.device, seed=qseed)
            runtime = time.time() - t0

            # V5 output length: median of positives_cluster T
            pos_T = [p['T'] for p in q.get('positives_cluster', [])]
            if not pos_T:
                target_len = q.get('src_T', output.shape[0])
            else:
                target_len = int(np.median(pos_T))
            if output.shape[0] != target_len:
                from scipy.interpolate import interp1d
                T_in = output.shape[0]
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
            rec.update({
                'status': 'ok', 'scaffold_fname': ex_fname,
                'n_pairs': len(pairs_ij), 'mapping_desc': mapping_desc,
                'runtime_s': runtime, 'output_T': int(output.shape[0]),
            })

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = f'{type(e).__name__}: {e}'
            print(f'  q{qid} FAILED: {e}')

        per_query.append(rec)

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_total
            n_ok = sum(1 for r in per_query if r['status'] == 'ok')
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  fold {args.fold} [{i+1}/{len(queries)}] elapsed {elapsed:.0f}s, "
                  f"ETA {eta:.0f}s, ok={n_ok}")

    n_ok = sum(1 for r in per_query if r['status'] == 'ok')
    summary = {
        'method': args.out_tag, 'fold': args.fold,
        'manifest': str(manifest_path),
        'n_queries': len(per_query), 'n_ok': n_ok,
        'total_time_sec': time.time() - t_total,
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFold {args.fold}: {n_ok}/{len(per_query)} ok. Saved to {out_dir}")


if __name__ == '__main__':
    main()
