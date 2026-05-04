"""K_retrieve baseline on v3 benchmark.

Q-similarity retrieval: pick top-1 target_skel clip whose Q-signature is closest
to the source's Q*. POOL POLICY (per Codex review): retrieval pool = full
target_skel library. Picking a positive = method working correctly.

Inference-time only — no learning, no contamination.

Usage:
  python -m eval.baselines.run_k_retrieve_v3 --fold 42
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

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR

MOTION_DIR = Path(DATASET_DIR) / 'motions'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'


def list_target_skel_motions(skel_name, motion_dir):
    """All .npy files for given target skeleton."""
    return sorted([f for f in os.listdir(motion_dir)
                   if (f.startswith(skel_name + '___') or f.startswith(skel_name + '_'))
                   and f.endswith('.npy')])


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
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/queries/fold_{args.fold}/manifest.json'
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / f'eval/results/baselines/k_retrieve/v3_fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")

    from eval.run_k_pipeline_200pairs import load_assets
    from eval.quotient_extractor import extract_quotient
    from eval.pilot_Q_experiments import q_signature
    from eval.run_k_pipeline_30pairs import build_q_star

    print("Loading assets...")
    cond, contact_groups, motion_dir = load_assets()

    # Load Q cache for fast signature lookup
    print(f"Loading Q cache from {Q_CACHE_PATH}...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    qfname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    print(f"  Q cache: {len(qfname_to_idx)} clips")

    sig_cache = {}

    def get_q_sig(fname, skel_name):
        if fname in sig_cache:
            return sig_cache[fname]
        idx = qfname_to_idx.get(fname)
        if idx is not None:
            q = {
                'com_path': qc['com_path'][idx],
                'heading_vel': qc['heading_vel'][idx],
                'contact_sched': qc['contact_sched'][idx],
                'cadence': float(qc['cadence'][idx]),
                'limb_usage': qc['limb_usage'][idx],
            }
            sig = q_signature(q)
            sig_cache[fname] = sig
            return sig
        # Fallback: compute on the fly
        try:
            q = extract_quotient(fname, cond[skel_name],
                                 contact_groups=contact_groups,
                                 motion_dir=motion_dir)
            sig = q_signature(q)
            sig_cache[fname] = sig
            return sig
        except Exception:
            return None

    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"Running {len(queries)} queries")

    tgt_clip_cache = {}
    per_query = []
    t_total_0 = time.time()

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
            if skel_a not in contact_groups or skel_b not in contact_groups:
                rec['status'] = 'skipped_no_cg'
                per_query.append(rec)
                continue

            # 1. Compute Q*
            q_src = extract_quotient(src_fname, cond[skel_a],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            q_star = build_q_star(q_src, skel_a, skel_b, contact_groups, cond)
            q_star_sig = q_signature(q_star)
            q_star_norm = q_star_sig / (np.linalg.norm(q_star_sig) + 1e-9)

            # 2. Build retrieval pool: target's full library
            # Per Codex review: do NOT exclude positives — that forces retrieval
            # to pick wrong-cluster (since positives = entire same-cluster set).
            # Retrieval picking a positive = method working correctly (semantic match).
            # Eval will measure prediction vs positives + adversarials — fair.
            if skel_b not in tgt_clip_cache:
                tgt_clip_cache[skel_b] = list_target_skel_motions(skel_b, motion_dir)
            pool = tgt_clip_cache[skel_b]
            if not pool:
                raise RuntimeError(f'Empty retrieval pool for {skel_b}')

            # 3. Retrieve top-1 by Q-signature cosine sim
            best_sim = -np.inf
            best_fname = None
            for cand in pool:
                cand_sig = get_q_sig(cand, skel_b)
                if cand_sig is None:
                    continue
                cand_norm = cand_sig / (np.linalg.norm(cand_sig) + 1e-9)
                sim = float(q_star_norm @ cand_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_fname = cand
            if best_fname is None:
                raise RuntimeError('No valid candidate Q sigs')

            # Retrieval may pick a positive — that's the method WORKING (semantic match)
            # Track for transparency. Support v3 + v5 manifest schemas.
            pos_list = q.get('positives', q.get('positives_cluster', []))
            adv_list = q.get('adversarials', q.get('adversarials_easy', []) + q.get('adversarials_hard', []))
            forbidden_check = ({p['fname'] for p in pos_list} | {a['fname'] for a in adv_list})
            rec['picked_in_positives'] = best_fname in {p['fname'] for p in pos_list}
            rec['picked_in_adversarials'] = best_fname in {a['fname'] for a in adv_list}

            # 4. Save retrieved motion as the prediction
            motion = np.load(MOTION_DIR / best_fname).astype(np.float32)
            np.save(out_dir / f'query_{qid:04d}.npy', motion)
            rec['status'] = 'ok'
            rec['retrieved_fname'] = best_fname
            rec['retrieval_cosine'] = best_sim
            rec['pool_size'] = len(pool)

            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - t_total_0
                eta = elapsed / (i + 1) * (len(queries) - i - 1)
                print(f"  [{i+1}/{len(queries)}] {cluster}/{split} {skel_a}→{skel_b} "
                      f"sim={best_sim:.3f} pool={len(pool)} (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            print(f"  FAILED query {qid}: {e}")

        per_query.append(rec)

    total_time = time.time() - t_total_0
    n_ok = sum(1 for r in per_query if r['status'] == 'ok')
    print(f"\nTotal: {total_time:.0f}s, {n_ok}/{len(per_query)} OK")

    summary = {
        'method': 'K_retrieve_v3',
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
