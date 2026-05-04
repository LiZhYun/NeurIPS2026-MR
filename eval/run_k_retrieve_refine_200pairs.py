"""Idea K_v2: Retrieve-then-Refine.

Replaces Idea K's Stage 2 IK output (which produces unnatural artifacts) with
a RETRIEVED real motion clip from the target skeleton's library that maximizes
Q-similarity to the source. Then refines via AnyTop projection with K's hard
constraints (contact_positions, com_path).

Per evaluation 2026-04-18:
  - Random target clip beats Idea K (DTW 0.19 vs 0.46)
  - Idea K's IK output is far from target's natural distribution
  - Solution: bootstrap with retrieval (in-distribution) + AnyTop refinement (constraint-satisfying)

Pipeline:
  1. Compute K's Q* (target-skel quotient from source)
  2. Retrieve top-1 target-skel clip by cosine(q_signature(Q_clip), q_signature(Q*))
  3. Build hard constraints from K's Q* (contact mask, COM path)
  4. AnyTop project from retrieved clip with hard constraints

Outputs:
  eval/results/k_compare/K_retrieve_200pair/pair_NNNN.npy
  eval/results/k_compare/K_retrieve_200pair/metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.skel_blind.invariant_dataset import TEST_SKELETONS

OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_200pair'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = PROJECT_ROOT / 'eval/benchmark_paired/pairs/manifest.json'
MOTIONS_DIR = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'

T_INIT = 0.3
N_STEPS = 20


def load_assets():
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'),
                   allow_pickle=True).item()
    with open(PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        cg = json.load(f)
    return cond, cg, os.path.join(DATASET_DIR, 'motions')


def split_category(skel_a, skel_b):
    a_test = skel_a in TEST_SKELETONS
    b_test = skel_b in TEST_SKELETONS
    if a_test and b_test:
        return "test_test"
    elif a_test or b_test:
        return "mixed"
    return "train_train"


def list_target_skel_motions(skel_name, motion_dir):
    """Find all motion files for a given target skeleton."""
    files = []
    for f in os.listdir(motion_dir):
        if not f.endswith('.npy'):
            continue
        if f.startswith(skel_name + '___'):
            # Make sure no LONGER skeleton name matches
            files.append(f)
    return sorted(files)


def run(max_pairs=200):
    from eval.quotient_extractor import extract_quotient
    from eval.pilot_Q_experiments import q_signature
    from eval.anytop_projection import anytop_project
    from eval.run_k_pipeline_30pairs import build_q_star

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading assets...")
    cond, contact_groups, motion_dir = load_assets()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    pairs = manifest['pairs'][:max_pairs]
    print(f"Loaded {len(pairs)} pairs")

    # Cache target-skel motion lists per skeleton
    tgt_clip_cache = {}

    # Load precomputed Q cache (1070 clips, ~5s extract each — saves ~3h)
    Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
    print(f"Loading precomputed Q cache from {Q_CACHE_PATH}...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    qmeta = list(qc['meta'])
    qfname_to_idx = {m['fname']: i for i, m in enumerate(qmeta)}
    print(f"  Q cache: {len(qmeta)} clips")

    sig_cache = {}  # fname -> (q_sig, q_dict)

    def get_clip_q_sig(fname, skel_name):
        if fname in sig_cache:
            return sig_cache[fname]
        # Try precomputed cache first
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
            sig_cache[fname] = (sig, q)
            return sig, q
        # Fallback: extract on demand
        try:
            q = extract_quotient(fname, cond[skel_name],
                                 contact_groups=contact_groups,
                                 motion_dir=motion_dir)
            sig = q_signature(q)
            sig_cache[fname] = (sig, q)
            return sig, q
        except Exception as e:
            print(f"  [warn] Q-extract failed for {fname}: {type(e).__name__}: {e}")
            return None, None

    per_pair = []
    t_total_0 = time.time()

    for i, p in enumerate(pairs):
        pid = p['pair_id']
        src_skel = p['skel_a']
        tgt_skel = p['skel_b']
        src_fname = p['file_a']
        tgt_fname = p['file_b']  # to exclude from retrieval pool
        action = p['action']

        rec = {
            'pair_id': pid, 'action': action,
            'source_skel': src_skel, 'target_skel': tgt_skel,
            'source_fname': src_fname,
            'split': split_category(src_skel, tgt_skel),
            'status': 'pending',
        }

        if src_skel not in contact_groups or tgt_skel not in contact_groups:
            rec['status'] = 'skipped_no_cg'
            per_pair.append(rec)
            continue

        t0 = time.time()
        try:
            # 1. Compute K's Q* on target
            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            q_star = build_q_star(q_src, src_skel, tgt_skel,
                                  contact_groups, cond)
            q_star_sig = q_signature(q_star)

            # 2. Retrieve top-1 target-skel clip by Q similarity, EXCLUDE GT
            if tgt_skel not in tgt_clip_cache:
                tgt_clip_cache[tgt_skel] = list_target_skel_motions(tgt_skel, motion_dir)
            full_pool = tgt_clip_cache[tgt_skel]
            if not full_pool:
                # Fragile prefix matching may miss skeletons like Cat/Chicken/Fox/Puppy
                raise RuntimeError(f'list_target_skel_motions returned empty for {tgt_skel} '
                                   f'— filename prefix bug')
            if tgt_fname not in full_pool:
                raise RuntimeError(f'GT {tgt_fname} not found in pool for {tgt_skel} '
                                   f'— refusing to proceed (silent leak risk)')
            tgt_files = [f for f in full_pool if f != tgt_fname]
            if not tgt_files:
                raise RuntimeError(f'Only GT clip in pool for {tgt_skel}')

            best_sim = -np.inf
            best_fname = None
            best_q = None
            q_star_sig_norm = q_star_sig / (np.linalg.norm(q_star_sig) + 1e-9)
            for cand_fname in tgt_files:
                cand_sig, cand_q = get_clip_q_sig(cand_fname, tgt_skel)
                if cand_sig is None:
                    continue
                cand_sig_norm = cand_sig / (np.linalg.norm(cand_sig) + 1e-9)
                sim = float(q_star_sig_norm @ cand_sig_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_fname = cand_fname
                    best_q = cand_q
            if best_fname is None:
                raise RuntimeError('No valid target-skel clip found')
            assert best_fname != tgt_fname, f'GT leak: retrieved {best_fname} == GT {tgt_fname}'

            rec['retrieved_fname'] = best_fname
            rec['retrieval_cosine'] = best_sim

            # 3. Load retrieved motion as init
            retrieved_motion = np.load(os.path.join(motion_dir, best_fname)).astype(np.float32)

            # 4. Build hard constraints from K's Q*
            # contact_positions per joint: aggregate Q* contact_sched broadcast to target joint mask
            T_init = retrieved_motion.shape[0]
            n_joints = retrieved_motion.shape[1]
            contact_mask_TJ = (retrieved_motion[..., 12] > 0.5).astype(np.float32)
            # Use Q*'s body_scale to scale COM path
            com_path_T3 = q_star['com_path'] * q_star['body_scale']
            # Crop COM to retrieved clip length
            if com_path_T3.shape[0] > T_init:
                com_path_T3 = com_path_T3[:T_init]
            elif com_path_T3.shape[0] < T_init:
                # Pad com path by repeating last frame
                pad = np.tile(com_path_T3[-1:], (T_init - com_path_T3.shape[0], 1))
                com_path_T3 = np.concatenate([com_path_T3, pad], axis=0)

            # 5. AnyTop project from retrieved clip
            try:
                proj = anytop_project(
                    retrieved_motion, tgt_skel,
                    hard_constraints={'contact_positions': contact_mask_TJ,
                                      'com_path': com_path_T3},
                    t_init=T_INIT, n_steps=N_STEPS, device=device)
                out_motion = proj['x_refined']
                rec['anytop_runtime'] = float(proj['runtime_seconds'])
                rec['status'] = 'ok'
            except Exception as e:
                rec['status'] = 'stage3_failed'
                rec['error'] = str(e)
                out_motion = retrieved_motion

            out_path = OUT_DIR / f'pair_{pid:04d}.npy'
            np.save(out_path, out_motion)
            rec['pair_runtime'] = float(time.time() - t0)

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t_total_0
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                print(f"  [{i+1}/{len(pairs)}] {action}: {src_skel}→{tgt_skel} "
                      f"retrieved={best_fname[:30]}... sim={best_sim:.3f} "
                      f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f"  FAILED pair {pid}: {e}")

        per_pair.append(rec)

    total_time = time.time() - t_total_0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/len(per_pair):.1f}s/pair)")

    summary = {
        'method': 'Idea_K_retrieve',
        'n_pairs': len(per_pair),
        'n_ok': sum(1 for r in per_pair if r['status'] == 'ok'),
        'n_stage3_failed': sum(1 for r in per_pair if r['status'] == 'stage3_failed'),
        'n_skipped': sum(1 for r in per_pair if 'skipped' in r['status']),
        'n_failed': sum(1 for r in per_pair if r['status'] == 'failed'),
        'total_time_sec': total_time,
        'hparams': {'t_init': T_INIT, 'n_steps': N_STEPS},
        'per_pair': per_pair,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {OUT_DIR / 'metrics.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pairs', type=int, default=200)
    args = parser.parse_args()
    run(max_pairs=args.max_pairs)
