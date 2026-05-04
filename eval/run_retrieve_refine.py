"""Idea H: retrieve-then-refine hybrid for cross-skeleton retargeting.

Protocol (per pair):
    1. Load source Q from the Q cache (idea-stage/quotient_cache.npz).
    2. Rank all target-skeleton clips in the Q cache by cosine similarity of
       their 19-dim q_signature (eval/pilot_Q_experiments.q_signature) to the
       source's signature. Take the top-1 target-skel clip.
    3. Load that clip's raw [T, J, 13] motion tensor from the dataset.
    4. Build hard_constraints from the retrieved motion itself:
          contact_positions : [T, J] = retrieved's foot-contact channel
          com_path          : [T, 3]  = (0, mean joint-Y per frame, 0)
                              (same proxy anytop_projection._main_test uses)
    5. Call anytop_project(x_init=retrieved, target_skel=tgt,
                           hard_constraints=hc, t_init=0.3, n_steps=20).
    6. Save refined motion + metrics (Q errors vs source Q; contact F1 vs
       retrieved's own schedule; wall time).

We also record the retrieved clip's pre-refinement Q so we can compare Q
preservation before/after the AnyTop prior projection. Same stratified-summary
schema as run_baselines.py so k_action_accuracy.py works without changes.

This is intentionally a NEW file (no modifications to AnyTop base modules).

Usage:
    conda run -n anytop python -m eval.run_retrieve_refine
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'

OUT_DIR = ROOT / 'eval/results/k_compare/retrieve_refine'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Refinement hyper-parameters (match run_k_pipeline_30pairs for t_init, spec n_steps=20)
T_INIT = 0.3
N_STEPS = 20
LAMBDA_COM = 1.0
FOOT_CH_IDX = 12
POS_Y_IDX = 1


# -------------------- helpers --------------------

def l2_norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def cosine_sim(a, b):
    return l2_norm(a) @ l2_norm(b).T


def q_component_l2(q_src: dict, q_tgt: dict) -> dict:
    """Per-component L2 distance between two Q dicts (matches run_baselines)."""
    def _l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            if a.ndim == b.ndim and a.ndim >= 1:
                T = min(a.shape[0], b.shape[0])
                a = a[:T]; b = b[:T]
            else:
                return None
        return float(np.linalg.norm((a - b).reshape(-1)))
    out = {
        'com_path': _l2(q_src['com_path'], q_tgt['com_path']),
        'heading_vel': _l2(q_src['heading_vel'], q_tgt['heading_vel']),
        'cadence': float(abs(float(q_src['cadence']) - float(q_tgt['cadence']))),
    }
    cs_src = np.asarray(q_src['contact_sched']).reshape(
        q_src['contact_sched'].shape[0], -1).sum(axis=-1)
    cs_tgt = np.asarray(q_tgt['contact_sched']).reshape(
        q_tgt['contact_sched'].shape[0], -1).sum(axis=-1)
    T = min(len(cs_src), len(cs_tgt))
    out['contact_sched_aggregate'] = float(np.linalg.norm(cs_src[:T] - cs_tgt[:T]))
    lu_src = -np.sort(-np.asarray(q_src['limb_usage']))[:5]
    lu_tgt = -np.sort(-np.asarray(q_tgt['limb_usage']))[:5]
    K = min(len(lu_src), len(lu_tgt))
    lu_src = np.pad(lu_src[:K], (0, max(0, 5 - K)))
    lu_tgt = np.pad(lu_tgt[:K], (0, max(0, 5 - K)))
    out['limb_usage_top5'] = float(np.linalg.norm(lu_src - lu_tgt))
    return out


def contact_f1(sched_rec: np.ndarray, sched_tgt: np.ndarray,
               thresh: float = 0.5) -> float:
    pred = (np.asarray(sched_rec) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(sched_tgt) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


def build_q_sig_array(qc):
    """[N, 19] q_signature matrix aligned with qc['meta'] order."""
    from eval.pilot_Q_experiments import q_signature
    N = len(qc['meta'])
    sigs = []
    for i in range(N):
        q = {
            'com_path': qc['com_path'][i],
            'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        sigs.append(q_signature(q))
    return np.stack(sigs)


def stratified_summary(all_entries):
    """Group metrics by family_gap bucket and support_absent flag.

    Reports: near_present, absent, moderate, extreme, all.
    Matches run_baselines.py schema so downstream scripts (k_action_accuracy.py,
    unified comparison) interpret entries identically.
    """
    buckets = defaultdict(list)
    for e in all_entries:
        fam = e['family_gap']
        if fam == 'near_present' or fam == 'near':
            buckets['near_present'].append(e)
        if fam == 'moderate':
            buckets['moderate'].append(e)
        if fam == 'extreme':
            buckets['extreme'].append(e)
        if e['support_same_label'] == 0:
            buckets['absent'].append(e)
    buckets['all'] = list(all_entries)

    numeric_keys = [
        'q_com_path_l2', 'q_heading_vel_l2', 'q_contact_sched_l2',
        'q_cadence_abs_diff', 'q_limb_usage_top5_l2',
        'q_com_path_l2_pre', 'q_heading_vel_l2_pre', 'q_contact_sched_l2_pre',
        'q_cadence_abs_diff_pre', 'q_limb_usage_top5_pre',
        'q_com_path_delta', 'q_heading_vel_delta', 'q_contact_sched_delta',
        'q_cadence_delta', 'q_limb_usage_delta',
        'contact_f1_self', 'contact_f1_vs_retrieved',
        'refine_runtime_s', 'retrieval_time_s', 'wall_time_s',
    ]
    summary = {}
    for stratum, entries in buckets.items():
        summary[stratum] = {'n': len(entries)}
        for k in numeric_keys:
            vals = [e[k] for e in entries if e.get(k) is not None]
            summary[stratum][k] = float(np.mean(vals)) if vals else None
    return summary


# -------------------- main --------------------

def run():
    t_start_all = time.time()

    from eval.quotient_extractor import extract_quotient
    from eval.pilot_Q_experiments import q_signature
    from eval.anytop_projection import anytop_project

    print('Loading caches...')
    with open(EVAL_PAIRS) as f:
        eval_data = json.load(f)
    pairs = eval_data['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    fname_to_meta_idx = {m['fname']: i for i, m in enumerate(meta)}
    q_meta = list(qc['meta'])
    fname_to_q_idx = {m['fname']: i for i, m in enumerate(q_meta)}
    print(f"  Q cache: {len(q_meta)}  meta: {len(meta)}  pairs: {len(pairs)}")

    skel_to_meta_idx = defaultdict(list)
    for i, m in enumerate(meta):
        skel_to_meta_idx[m['skeleton']].append(i)

    print('Building Q signatures...')
    q_sigs = build_q_sig_array(qc)
    print(f"  q_sig dim: {q_sigs.shape[1]}  ({q_sigs.shape[0]} clips)")

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  device: {device}")

    per_pair = []
    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']
        src_fname = p['source_fname']
        tgt_skel = p['target_skel']
        src_label = p['source_label']
        family_gap = p['family_gap']
        support = int(p['support_same_label'])
        strat = {'near': 'near_present'}.get(family_gap, family_gap)

        print(f"\n=== pair {pid:02d}  {src_skel}({src_fname}) -> {tgt_skel}  "
              f"gap={family_gap}  supp={support} ===")

        rec = {
            'pair_id': pid,
            'src_fname': src_fname,
            'src_skel': src_skel,
            'src_label': src_label,
            'tgt_skel': tgt_skel,
            'family_gap': strat,
            'support_same_label': support,
            'status': 'pending',
            'error': None,
        }
        pair_t0 = time.time()

        try:
            if src_fname not in fname_to_q_idx:
                raise RuntimeError(f'source missing from Q cache: {src_fname}')
            if tgt_skel not in cond_dict:
                raise RuntimeError(f'target skel missing from cond: {tgt_skel}')

            src_q_idx = fname_to_q_idx[src_fname]
            src_q_sig = q_sigs[src_q_idx]

            # Target pool restricted to Q cache (defensive: drop same fname).
            tgt_pool = [i for i in skel_to_meta_idx[tgt_skel]
                        if meta[i]['fname'] != src_fname
                        and meta[i]['fname'] in fname_to_q_idx]
            if not tgt_pool:
                raise RuntimeError(f'empty target Q pool for {tgt_skel}')

            # --- retrieve top-1 by cosine(src, cand) on 19-dim q_signature ---
            t_retr0 = time.time()
            cand_q_idx = np.array(
                [fname_to_q_idx[meta[i]['fname']] for i in tgt_pool])
            sims = cosine_sim(src_q_sig[None], q_sigs[cand_q_idx])[0]
            best = int(np.argmax(sims))
            retr_meta_idx = tgt_pool[best]
            retr_fname = meta[retr_meta_idx]['fname']
            retr_coarse = meta[retr_meta_idx]['coarse_label']
            retrieval_time = time.time() - t_retr0
            rec['retrieved_fname'] = retr_fname
            rec['retrieved_coarse_label'] = retr_coarse
            rec['retrieval_cosine'] = float(sims[best])
            rec['retrieval_time_s'] = retrieval_time

            # --- source Q (for metric target) & retrieved-clip Q (pre-refine) ---
            src_q = extract_quotient(src_fname, cond_dict[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTIONS_DIR))
            retr_q_pre = extract_quotient(retr_fname, cond_dict[tgt_skel],
                                          contact_groups=contact_groups,
                                          motion_dir=str(MOTIONS_DIR))
            q_pre = q_component_l2(src_q, retr_q_pre)
            rec['q_com_path_l2_pre'] = q_pre.get('com_path')
            rec['q_heading_vel_l2_pre'] = q_pre.get('heading_vel')
            rec['q_contact_sched_l2_pre'] = q_pre.get('contact_sched_aggregate')
            rec['q_cadence_abs_diff_pre'] = q_pre.get('cadence')
            rec['q_limb_usage_top5_pre'] = q_pre.get('limb_usage_top5')

            # --- load the retrieved motion for projection ---
            retrieved = np.load(MOTIONS_DIR / retr_fname).astype(np.float32)
            T_retr, J_retr, F_retr = retrieved.shape

            # --- build hard_constraints from the retrieved motion ---
            contact_mask = (retrieved[..., FOOT_CH_IDX] > 0.5).astype(np.float32)  # [T, J]
            mean_y_path = retrieved[..., POS_Y_IDX].mean(axis=1)                   # [T]
            com_path_T3 = np.stack([np.zeros(T_retr), mean_y_path,
                                    np.zeros(T_retr)], axis=-1).astype(np.float32)
            hard_con = {'contact_positions': contact_mask,
                        'com_path': com_path_T3}

            # --- refine via frozen-AnyTop prior projection ---
            t_ref0 = time.time()
            proj = anytop_project(retrieved, tgt_skel,
                                  hard_constraints=hard_con,
                                  t_init=T_INIT, n_steps=N_STEPS,
                                  lambda_com=LAMBDA_COM, device=device)
            refine_runtime = time.time() - t_ref0
            rec['refine_runtime_s'] = float(proj['runtime_seconds'])
            x_refined = proj['x_refined']  # [T', J_retr, 13] denormalised

            # --- save refined motion ---
            out_fname = f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            out_path = OUT_DIR / out_fname
            np.save(out_path, x_refined.astype(np.float32))
            rec['output_file'] = out_fname

            # --- post-refinement Q on target skeleton ---
            # Save the refined motion, then let extract_quotient load it back so
            # we use the exact same extraction path as the rest of the pipeline.
            tmp_name = f'__retrieve_refine_tmp_pair_{pid:02d}.npy'
            tmp_path = MOTIONS_DIR / tmp_name
            try:
                np.save(tmp_path, x_refined.astype(np.float32))
                refined_q = extract_quotient(tmp_name, cond_dict[tgt_skel],
                                             contact_groups=contact_groups,
                                             motion_dir=str(MOTIONS_DIR))
            finally:
                if tmp_path.exists():
                    try: tmp_path.unlink()
                    except Exception: pass

            q_post = q_component_l2(src_q, refined_q)
            rec['q_com_path_l2'] = q_post.get('com_path')
            rec['q_heading_vel_l2'] = q_post.get('heading_vel')
            rec['q_contact_sched_l2'] = q_post.get('contact_sched_aggregate')
            rec['q_cadence_abs_diff'] = q_post.get('cadence')
            rec['q_limb_usage_top5_l2'] = q_post.get('limb_usage_top5')

            # Deltas = post - pre (negative ⇒ refinement pulled closer to source Q)
            def _delta(post, pre):
                if post is None or pre is None:
                    return None
                return float(post - pre)
            rec['q_com_path_delta']     = _delta(rec['q_com_path_l2'],     rec['q_com_path_l2_pre'])
            rec['q_heading_vel_delta']  = _delta(rec['q_heading_vel_l2'],  rec['q_heading_vel_l2_pre'])
            rec['q_contact_sched_delta']= _delta(rec['q_contact_sched_l2'],rec['q_contact_sched_l2_pre'])
            rec['q_cadence_delta']      = _delta(rec['q_cadence_abs_diff'],rec['q_cadence_abs_diff_pre'])
            rec['q_limb_usage_delta']   = _delta(rec['q_limb_usage_top5_l2'], rec['q_limb_usage_top5_pre'])

            # Contact F1 of refined schedule versus the retrieved schedule
            # (we used the retrieved contacts as the hard constraint, so this
            # quantifies how faithfully the refinement respected them).
            rec['contact_f1_vs_retrieved'] = contact_f1(
                refined_q['contact_sched'], retr_q_pre['contact_sched'])
            # Self-consistency: refined schedule vs its own foot-contact channel.
            rec['contact_f1_self'] = contact_f1(
                refined_q['contact_sched'], retr_q_pre['contact_sched'])

            rec['wall_time_s'] = float(time.time() - pair_t0)
            rec['status'] = 'ok'
            print(f"  ok  retr={retr_fname} ({retr_coarse})  "
                  f"q_com_pre={rec['q_com_path_l2_pre']:.3f}->post={rec['q_com_path_l2']:.3f}  "
                  f"refine={refine_runtime:.2f}s")
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['wall_time_s'] = float(time.time() - pair_t0)
            print(f"  FAILED: {e}")
        per_pair.append(rec)

    total_time = time.time() - t_start_all

    # Write metrics.json (schema consistent with run_baselines.py).
    ok = [r for r in per_pair if r['status'] == 'ok']
    failed = [r for r in per_pair if r['status'] != 'ok']
    stratified = stratified_summary(ok)
    out = {
        'method': 'retrieve_refine',
        'hparams': {'t_init': T_INIT, 'n_steps': N_STEPS,
                    'lambda_com': LAMBDA_COM, 'signature_dim': int(q_sigs.shape[1])},
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok': len(ok),
        'n_failed': len(failed),
        'per_pair': per_pair,
        'stratified': stratified,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n=== DONE: total {total_time:.1f}s  n_ok={len(ok)}/{len(pairs)} ===")
    print(f"metrics saved: {OUT_DIR / 'metrics.json'}")

    # Also print a compact Q-preservation table.
    print('\nStratified Q-preservation (com_path_l2  pre -> post  delta):')
    for s in ['near_present', 'absent', 'moderate', 'extreme', 'all']:
        b = stratified.get(s, {})
        pre = b.get('q_com_path_l2_pre'); post = b.get('q_com_path_l2')
        delta = b.get('q_com_path_delta')
        n = b.get('n', 0)
        if pre is None:
            print(f"  {s:14s} n={n} —")
        else:
            print(f"  {s:14s} n={n}  pre={pre:.3f}  post={post:.3f}  delta={delta:+.3f}")

    return out


if __name__ == '__main__':
    run()
