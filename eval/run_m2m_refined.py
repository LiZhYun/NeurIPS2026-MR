"""Idea R: M2M-lite + H_v4 hybrid (m2m_refined).

Combines strengths of two prior methods:
  * M2M-lite: pyramidal patch retrieval from target's training data (best for
    overall Q preservation; contact_f1_vs_source 0.682 overall but 0.362 on absent).
  * H_v4: AnyTop SDEdit-style refinement with source-anchored hard constraints
    (best among learned methods on the 'absent' stratum with 0.459).

Pipeline for each of 30 pairs:
  1. Run M2M-lite (reusing `run_m2m_lite` from eval/motion2motion_run.py) to get
     an initial retargeted motion [T, J_tgt, 13] on the target skeleton.
  2. Build source-anchored hard constraints via v2's
     `build_source_anchored_contact_mask` + rescaled source COM path.
  3. Pass M2M's output as `x_init` to `anytop_project` with t_init=0.3,
     n_steps=20, lambda_com=1.0.
  4. Save refined motion to eval/results/k_compare/m2m_refined/.
  5. Compute pre/post Q-component L2 distances + contact_f1_vs_source.
  6. Run classifier-v2 scoring (via k_action_accuracy_v2_full.py pattern).

Expected budget: ~15 min (M2M 0.05s/pair, refinement 0.9s/pair; 30 pairs + overhead).

Output: eval/results/k_compare/m2m_refined/metrics.json.
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse v2 helpers (contact mask builder, Q comp L2, contact F1, stratified summary).
from eval.run_retrieve_refine_v2 import (
    q_component_l2, contact_f1,
    build_source_anchored_contact_mask, stratified_summary,
    EVAL_PAIRS, COND_PATH, MOTIONS_DIR, CONTACT_GROUPS_PATH,
    T_INIT, N_STEPS, LAMBDA_COM,
)

# Reuse M2M-lite core runner and sparse-mapping heuristic.
from eval.motion2motion_run import (
    run_m2m_lite, author_sparse_mapping, find_target_example, load_motion,
)

OUT_DIR = ROOT / 'eval/results/k_compare/m2m_refined'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run():
    t_start_all = time.time()
    from eval.quotient_extractor import extract_quotient
    from eval.anytop_projection import anytop_project

    print('Loading pairs + caches...')
    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  device: {device}')

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
        print(f"\n=== pair {pid:02d} {src_skel}({src_fname}) -> {tgt_skel}  "
              f"gap={family_gap}  supp={support} ===")
        rec = {
            'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
            'src_label': src_label, 'tgt_skel': tgt_skel,
            'family_gap': strat, 'support_same_label': support,
            'status': 'pending', 'error': None,
        }
        t_pair0 = time.time()
        try:
            if tgt_skel not in cond_dict:
                raise RuntimeError(f'missing cond: {tgt_skel}')
            if src_skel not in cond_dict:
                raise RuntimeError(f'missing cond: {src_skel}')

            # --- Stage A: M2M-lite ---
            t_m2m0 = time.time()
            src_motion = load_motion(src_fname, cond_dict[src_skel])
            ex_fname = find_target_example(tgt_skel, MOTIONS_DIR)
            if ex_fname is None:
                raise RuntimeError(f'no target example for {tgt_skel}')
            ex_motion = load_motion(ex_fname, cond_dict[tgt_skel])

            pairs_ij, desc = author_sparse_mapping(
                src_skel, tgt_skel, cond_dict, contact_groups)
            src_j_idxs = [p_[0] for p_ in pairs_ij]
            tgt_j_idxs = [p_[1] for p_ in pairs_ij]
            J_tgt = len(cond_dict[tgt_skel]['joints_names'])
            rec['m2m_target_example'] = ex_fname
            rec['m2m_sparse_mapping'] = desc
            rec['m2m_n_sparse_pairs'] = len(pairs_ij)

            m2m_out = run_m2m_lite(
                src_motion, ex_motion, src_j_idxs, tgt_j_idxs,
                tgt_n_joints=J_tgt, device='cpu', seed=42,
            )  # [T_src, J_tgt, 13]
            rec['m2m_runtime_s'] = float(time.time() - t_m2m0)
            T_m2m = m2m_out.shape[0]
            assert m2m_out.shape[1] == J_tgt, 'M2M output joint count mismatch'

            # --- Source Q + M2M pre-refine Q (pre-refine baseline) ---
            src_q = extract_quotient(
                src_fname, cond_dict[src_skel],
                contact_groups=contact_groups,
                motion_dir=str(MOTIONS_DIR),
            )
            T_src = int(src_q['n_frames'])

            tmp_m2m = f'__m2m_refined_m2m_pair_{pid:02d}.npy'
            tmp_m2m_path = MOTIONS_DIR / tmp_m2m
            try:
                np.save(tmp_m2m_path, m2m_out.astype(np.float32))
                m2m_q = extract_quotient(
                    tmp_m2m, cond_dict[tgt_skel],
                    contact_groups=contact_groups,
                    motion_dir=str(MOTIONS_DIR),
                )
            finally:
                if tmp_m2m_path.exists():
                    try: tmp_m2m_path.unlink()
                    except Exception: pass

            q_pre = q_component_l2(src_q, m2m_q)
            rec['q_com_path_l2_pre'] = q_pre.get('com_path')
            rec['q_heading_vel_l2_pre'] = q_pre.get('heading_vel')
            rec['q_contact_sched_l2_pre'] = q_pre.get('contact_sched_aggregate')
            rec['q_cadence_abs_diff_pre'] = q_pre.get('cadence')
            rec['q_limb_usage_top5_pre'] = q_pre.get('limb_usage_top5')

            # contact_f1_vs_source pre-refine (M2M-lite only baseline)
            rs_pre = np.asarray(m2m_q['contact_sched'])
            ss = np.asarray(src_q['contact_sched'])
            T_pre = rs_pre.shape[0]
            idx = np.clip(np.linspace(0, ss.shape[0] - 1, T_pre).astype(int), 0, ss.shape[0] - 1)
            ss_pre = ss[idx]
            rs_pre_agg = rs_pre.sum(axis=1) if rs_pre.ndim == 2 else rs_pre
            ss_pre_agg = ss_pre.sum(axis=1) if ss_pre.ndim == 2 else ss_pre
            rec['contact_f1_vs_source_pre'] = contact_f1(rs_pre_agg, ss_pre_agg)

            # --- Stage B: source-anchored hard constraints ---
            tgt_groups_clean = {
                k: v for k, v in contact_groups.get(tgt_skel, {}).items()
                if not str(k).startswith('_')
            }
            source_contact_mask = build_source_anchored_contact_mask(
                src_q, tgt_skel, tgt_groups_clean,
                contact_groups.get(src_skel, {}),
                n_frames_target=T_m2m, n_joints_target=J_tgt,
            )
            src_com_path = np.asarray(src_q['com_path']).astype(np.float32)
            src_bs = float(src_q['body_scale'])
            tgt_offsets = cond_dict[tgt_skel]['offsets']
            tgt_bs = float(np.linalg.norm(tgt_offsets, axis=1).sum() + 1e-6)
            scale_ratio = tgt_bs / max(src_bs, 1e-6)
            idx_com = np.clip(
                np.linspace(0, src_com_path.shape[0] - 1, T_m2m).astype(int),
                0, src_com_path.shape[0] - 1,
            )
            src_com_resampled = src_com_path[idx_com] * scale_ratio

            hard_con = {
                'contact_positions': source_contact_mask,
                'com_path': src_com_resampled,
            }

            # --- Stage C: AnyTop SDEdit refinement on M2M output ---
            t_ref0 = time.time()
            proj = anytop_project(
                m2m_out.astype(np.float32), tgt_skel,
                hard_constraints=hard_con,
                t_init=T_INIT, n_steps=N_STEPS,
                lambda_com=LAMBDA_COM, device=device,
            )
            rec['refine_runtime_s'] = float(proj['runtime_seconds'])
            x_refined = proj['x_refined']

            # --- save output ---
            out_fname = f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(OUT_DIR / out_fname, x_refined.astype(np.float32))
            rec['output_file'] = out_fname

            # --- post-refinement Q ---
            tmp_post = f'__m2m_refined_post_pair_{pid:02d}.npy'
            tmp_post_path = MOTIONS_DIR / tmp_post
            try:
                np.save(tmp_post_path, x_refined.astype(np.float32))
                refined_q = extract_quotient(
                    tmp_post, cond_dict[tgt_skel],
                    contact_groups=contact_groups,
                    motion_dir=str(MOTIONS_DIR),
                )
            finally:
                if tmp_post_path.exists():
                    try: tmp_post_path.unlink()
                    except Exception: pass

            q_post = q_component_l2(src_q, refined_q)
            rec['q_com_path_l2'] = q_post.get('com_path')
            rec['q_heading_vel_l2'] = q_post.get('heading_vel')
            rec['q_contact_sched_l2'] = q_post.get('contact_sched_aggregate')
            rec['q_cadence_abs_diff'] = q_post.get('cadence')
            rec['q_limb_usage_top5_l2'] = q_post.get('limb_usage_top5')

            def _delta(post, pre):
                if post is None or pre is None:
                    return None
                return float(post - pre)
            rec['q_com_path_delta'] = _delta(rec['q_com_path_l2'], rec['q_com_path_l2_pre'])
            rec['q_heading_vel_delta'] = _delta(rec['q_heading_vel_l2'], rec['q_heading_vel_l2_pre'])
            rec['q_contact_sched_delta'] = _delta(rec['q_contact_sched_l2'], rec['q_contact_sched_l2_pre'])
            rec['q_cadence_delta'] = _delta(rec['q_cadence_abs_diff'], rec['q_cadence_abs_diff_pre'])
            rec['q_limb_usage_delta'] = _delta(rec['q_limb_usage_top5_l2'], rec['q_limb_usage_top5_pre'])

            # Contact F1: refined vs source
            rs = np.asarray(refined_q['contact_sched'])
            T_ref = rs.shape[0]
            idx_p = np.clip(np.linspace(0, ss.shape[0] - 1, T_ref).astype(int), 0, ss.shape[0] - 1)
            ss_aligned = ss[idx_p]
            rs_agg = rs.sum(axis=1) if rs.ndim == 2 else rs
            ss_agg = ss_aligned.sum(axis=1) if ss_aligned.ndim == 2 else ss_aligned
            rec['contact_f1_vs_source'] = contact_f1(rs_agg, ss_agg)
            rec['contact_f1_self'] = 1.0

            rec['wall_time_s'] = float(time.time() - t_pair0)
            rec['status'] = 'ok'
            cd = rec.get('q_com_path_delta', 0) or 0
            print(f"  ok  m2m_ex={ex_fname.split('___')[0]}  "
                  f"q_com pre={rec['q_com_path_l2_pre']:.3f}->post={rec['q_com_path_l2']:.3f} "
                  f"({cd:+.3f})  "
                  f"c_f1_src pre={rec['contact_f1_vs_source_pre']:.3f}->post={rec['contact_f1_vs_source']:.3f}  "
                  f"m2m={rec['m2m_runtime_s']:.2f}s refine={rec['refine_runtime_s']:.2f}s")
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['wall_time_s'] = float(time.time() - t_pair0)
            print(f"  FAILED: {e}")
        per_pair.append(rec)

    total_time = time.time() - t_start_all
    ok = [r for r in per_pair if r['status'] == 'ok']
    stratified = stratified_summary(ok)

    # Also aggregate pre (M2M-only) contact_f1 by stratum for direct comparison
    pre_contact_by = defaultdict(list)
    for r in ok:
        fam = r['family_gap']
        if r['support_same_label'] == 0:
            pre_contact_by['absent'].append(r['contact_f1_vs_source_pre'])
        else:
            pre_contact_by[fam].append(r['contact_f1_vs_source_pre'])
        pre_contact_by['all'].append(r['contact_f1_vs_source_pre'])
    pre_contact_by_stratum = {
        k: float(np.mean(v)) if v else None for k, v in pre_contact_by.items()
    }

    out = {
        'method': 'm2m_refined',
        'variant': 'm2m_lite_plus_anytop_source_anchored',
        'hparams': {
            't_init': T_INIT,
            'n_steps': N_STEPS,
            'lambda_com': LAMBDA_COM,
        },
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok': len(ok),
        'n_failed': len(per_pair) - len(ok),
        'per_pair': per_pair,
        'stratified': stratified,
        'pre_refine_contact_f1_by_stratum': pre_contact_by_stratum,
    }
    (OUT_DIR / 'metrics.json').write_text(json.dumps(out, indent=2, default=str))
    print(f"\n=== DONE: total {total_time:.1f}s  n_ok={len(ok)}/{len(pairs)} ===")
    print(f"metrics saved: {OUT_DIR / 'metrics.json'}")

    print('\nStratified contact_f1_vs_source (post-refine) vs M2M-only (pre-refine):')
    for s in ['near_present', 'absent', 'moderate', 'extreme', 'all']:
        b = stratified.get(s, {})
        n = b.get('n', 0)
        cf_post = b.get('contact_f1_vs_source')
        cf_pre = pre_contact_by_stratum.get(s)
        cd = b.get('q_com_path_delta')
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else '—'
        print(f"  {s:14s} n={n}  c_f1_pre(m2m)={fmt(cf_pre)}  "
              f"c_f1_post(hybrid)={fmt(cf_post)}  q_com_delta={fmt(cd)}")
    return out


if __name__ == '__main__':
    run()
