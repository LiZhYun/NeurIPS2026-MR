"""Idea H v4: retrieve-top-k + weighted blend + source-anchored refinement.

Building on v2 (best-performing variant so far):
  * v2 retrieves top-1 target clip, uses source-anchored constraints, gets
    overall behavior_preserved 0.233 (up from v1's 0.167) but absent label
    stays 0.
  * v3 tried stronger refinement (t_init=0.5, n_steps=40) -> worse on every metric.

v4 changes:
  1. Retrieve TOP-3 target clips by Q-signature cosine.
  2. Time-resample each to a common length T* = source's T.
  3. WEIGHTED BLEND by softmax(cosine similarity / tau=0.05). This gives a
     richer init that averages multiple near-neighbor target behaviors.
  4. Pass blended motion as init to AnyTop projection with source-anchored
     constraints (v2's scheme: contact mask from source-remapped schedule,
     COM from source rescaled by body_scale).
  5. Keep t_init=0.3 and n_steps=20 (the v2 sweet spot).

Hypothesis: blending >1 candidate smooths out wrong-action artifacts from any
single retrieved clip, while the source-anchored constraints pull toward
source behavior. Net: higher behavior-preservation on absent regime.
"""
from __future__ import annotations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse v2's helpers by importing it
from eval.run_retrieve_refine_v2 import (
    q_component_l2, contact_f1, build_q_sig_array,
    build_source_anchored_contact_mask, stratified_summary,
    cosine_sim, l2_norm,
    EVAL_PAIRS, META_PATH, Q_CACHE_PATH, COND_PATH, MOTIONS_DIR,
    CONTACT_GROUPS_PATH, FOOT_CH_IDX, POS_Y_IDX,
    T_INIT, N_STEPS, LAMBDA_COM,
)

OUT_DIR = ROOT / 'eval/results/k_compare/retrieve_refine_v4'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK = 3
BLEND_TAU = 0.05  # softmax temperature for cosine-sim weights


def blend_topk_motions(motions_and_weights, target_T):
    """Resample each [T_i, J, 13] motion to target_T via nearest-frame index,
    then compute weighted average. All motions must share J and F=13.
    """
    # All motions already share the same target skeleton (same J, F).
    resampled = []
    for m in motions_and_weights:
        m_mot = m['motion']
        t_i = m_mot.shape[0]
        idx = np.clip(np.linspace(0, t_i - 1, target_T).astype(int), 0, t_i - 1)
        resampled.append(m_mot[idx])
    w = np.array([m['weight'] for m in motions_and_weights], dtype=np.float32)
    w = w / (w.sum() + 1e-9)
    stacked = np.stack(resampled, axis=0)  # [K, T, J, 13]
    blended = (stacked * w[:, None, None, None]).sum(axis=0)
    return blended.astype(np.float32)


def run():
    t_start_all = time.time()
    from eval.quotient_extractor import extract_quotient
    from eval.anytop_projection import anytop_project

    print('Loading caches...')
    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    q_meta = list(qc['meta'])
    fname_to_q_idx = {m['fname']: i for i, m in enumerate(q_meta)}
    skel_to_meta_idx = defaultdict(list)
    for i, m in enumerate(meta):
        skel_to_meta_idx[m['skeleton']].append(i)

    print('Building Q signatures...')
    q_sigs = build_q_sig_array(qc)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    per_pair = []
    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']; src_fname = p['source_fname']
        tgt_skel = p['target_skel']; src_label = p['source_label']
        family_gap = p['family_gap']; support = int(p['support_same_label'])
        strat = {'near': 'near_present'}.get(family_gap, family_gap)
        print(f"\n=== pair {pid:02d} {src_skel}({src_fname}) -> {tgt_skel}  "
              f"gap={family_gap}  supp={support} ===")
        rec = {'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
               'src_label': src_label, 'tgt_skel': tgt_skel,
               'family_gap': strat, 'support_same_label': support,
               'status': 'pending', 'error': None}
        t_pair0 = time.time()
        try:
            if src_fname not in fname_to_q_idx:
                raise RuntimeError(f'source missing Q cache: {src_fname}')
            if tgt_skel not in cond_dict:
                raise RuntimeError(f'missing cond: {tgt_skel}')

            src_q_idx = fname_to_q_idx[src_fname]
            src_q_sig = q_sigs[src_q_idx]

            tgt_pool = [i for i in skel_to_meta_idx[tgt_skel]
                        if meta[i]['fname'] != src_fname
                        and meta[i]['fname'] in fname_to_q_idx]
            if not tgt_pool:
                raise RuntimeError(f'empty tgt pool {tgt_skel}')

            # --- Retrieve top-K ---
            t_retr0 = time.time()
            cand_q_idx = np.array([fname_to_q_idx[meta[i]['fname']] for i in tgt_pool])
            sims = cosine_sim(src_q_sig[None], q_sigs[cand_q_idx])[0]
            k_avail = min(TOPK, len(sims))
            order = np.argsort(-sims)[:k_avail]
            topk = [{'meta_idx': tgt_pool[int(i)],
                     'q_idx': int(cand_q_idx[int(i)]),
                     'sim': float(sims[int(i)]),
                     'fname': meta[tgt_pool[int(i)]]['fname'],
                     'coarse_label': meta[tgt_pool[int(i)]]['coarse_label']}
                    for i in order]
            # Softmax weights on sims with temperature
            w = np.array([c['sim'] for c in topk])
            w = np.exp(w / BLEND_TAU); w = w / w.sum()
            for c, wt in zip(topk, w):
                c['weight'] = float(wt)
            retrieval_time = time.time() - t_retr0
            rec['topk_fnames'] = [c['fname'] for c in topk]
            rec['topk_weights'] = [c['weight'] for c in topk]
            rec['topk_coarse_labels'] = [c['coarse_label'] for c in topk]
            rec['retrieval_time_s'] = retrieval_time

            # --- Source Q + each retrieved clip's Q (pre-refine reference) ---
            src_q = extract_quotient(src_fname, cond_dict[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTIONS_DIR))
            T_src = int(src_q['n_frames'])

            # --- Load + blend top-K motions ---
            motions_and_weights = []
            for c in topk:
                m = np.load(MOTIONS_DIR / c['fname']).astype(np.float32)
                motions_and_weights.append({'motion': m, 'weight': c['weight']})
            # Use max-T of top-k as blend target to retain dynamics
            T_blend = max([m['motion'].shape[0] for m in motions_and_weights])
            blended = blend_topk_motions(motions_and_weights, T_blend)
            J_tgt = blended.shape[1]

            # Compute pre-refine Q on the BLENDED motion (write tmp, extract)
            tmp_pre = f'__rr_v4_pre_pair_{pid:02d}.npy'
            tmp_pre_path = MOTIONS_DIR / tmp_pre
            try:
                np.save(tmp_pre_path, blended)
                blended_q = extract_quotient(tmp_pre, cond_dict[tgt_skel],
                                             contact_groups=contact_groups,
                                             motion_dir=str(MOTIONS_DIR))
            finally:
                if tmp_pre_path.exists():
                    try: tmp_pre_path.unlink()
                    except Exception: pass
            q_pre = q_component_l2(src_q, blended_q)
            rec['q_com_path_l2_pre'] = q_pre.get('com_path')
            rec['q_heading_vel_l2_pre'] = q_pre.get('heading_vel')
            rec['q_contact_sched_l2_pre'] = q_pre.get('contact_sched_aggregate')
            rec['q_cadence_abs_diff_pre'] = q_pre.get('cadence')
            rec['q_limb_usage_top5_pre'] = q_pre.get('limb_usage_top5')

            # --- source-anchored hard constraints ---
            tgt_groups_clean = {k: v for k, v in contact_groups.get(tgt_skel, {}).items()
                                if not str(k).startswith('_')}
            source_contact_mask = build_source_anchored_contact_mask(
                src_q, tgt_skel, tgt_groups_clean,
                contact_groups.get(src_skel, {}),
                n_frames_target=T_blend, n_joints_target=J_tgt,
            )
            src_com_path = np.asarray(src_q['com_path']).astype(np.float32)
            src_bs = float(src_q['body_scale'])
            tgt_offsets = cond_dict[tgt_skel]['offsets']
            tgt_bs = float(np.linalg.norm(tgt_offsets, axis=1).sum() + 1e-6)
            scale_ratio = tgt_bs / max(src_bs, 1e-6)
            idx = np.clip(np.linspace(0, src_com_path.shape[0] - 1, T_blend).astype(int),
                          0, src_com_path.shape[0] - 1)
            src_com_resampled = src_com_path[idx] * scale_ratio

            hard_con = {'contact_positions': source_contact_mask,
                        'com_path': src_com_resampled}

            # --- refine ---
            t_ref0 = time.time()
            proj = anytop_project(blended, tgt_skel,
                                  hard_constraints=hard_con,
                                  t_init=T_INIT, n_steps=N_STEPS,
                                  lambda_com=LAMBDA_COM, device=device)
            rec['refine_runtime_s'] = float(proj['runtime_seconds'])
            x_refined = proj['x_refined']

            out_fname = f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(OUT_DIR / out_fname, x_refined.astype(np.float32))
            rec['output_file'] = out_fname

            tmp_post = f'__rr_v4_post_pair_{pid:02d}.npy'
            tmp_post_path = MOTIONS_DIR / tmp_post
            try:
                np.save(tmp_post_path, x_refined.astype(np.float32))
                refined_q = extract_quotient(tmp_post, cond_dict[tgt_skel],
                                             contact_groups=contact_groups,
                                             motion_dir=str(MOTIONS_DIR))
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
                if post is None or pre is None: return None
                return float(post - pre)
            rec['q_com_path_delta'] = _delta(rec['q_com_path_l2'], rec['q_com_path_l2_pre'])
            rec['q_heading_vel_delta'] = _delta(rec['q_heading_vel_l2'], rec['q_heading_vel_l2_pre'])
            rec['q_contact_sched_delta'] = _delta(rec['q_contact_sched_l2'], rec['q_contact_sched_l2_pre'])
            rec['q_cadence_delta'] = _delta(rec['q_cadence_abs_diff'], rec['q_cadence_abs_diff_pre'])
            rec['q_limb_usage_delta'] = _delta(rec['q_limb_usage_top5_l2'], rec['q_limb_usage_top5_pre'])

            # Contact F1 vs source
            rs = np.asarray(refined_q['contact_sched'])
            ss = np.asarray(src_q['contact_sched'])
            T_ref = rs.shape[0]
            idx = np.clip(np.linspace(0, ss.shape[0] - 1, T_ref).astype(int), 0, ss.shape[0] - 1)
            ss_aligned = ss[idx]
            rs_agg = rs.sum(axis=1) if rs.ndim == 2 else rs
            ss_agg = ss_aligned.sum(axis=1) if ss_aligned.ndim == 2 else ss_aligned
            rec['contact_f1_vs_source'] = contact_f1(rs_agg, ss_agg)
            rec['contact_f1_self'] = 1.0

            rec['wall_time_s'] = float(time.time() - t_pair0)
            rec['status'] = 'ok'
            cd = rec.get('q_com_path_delta', 0) or 0
            print(f"  ok  topk={[c['fname'].split('___')[0] for c in topk]}  "
                  f"q_com pre={rec['q_com_path_l2_pre']:.3f}->post={rec['q_com_path_l2']:.3f} "
                  f"({cd:+.3f})  c_f1_src={rec['contact_f1_vs_source']:.3f}  "
                  f"refine={rec['refine_runtime_s']:.2f}s")
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['wall_time_s'] = float(time.time() - t_pair0)
            print(f"  FAILED: {e}")
        per_pair.append(rec)

    total_time = time.time() - t_start_all
    ok = [r for r in per_pair if r['status'] == 'ok']
    stratified = stratified_summary(ok)
    out = {
        'method': 'retrieve_refine_v4',
        'variant': 'topk_blend_source_anchored',
        'hparams': {'t_init': T_INIT, 'n_steps': N_STEPS,
                    'lambda_com': LAMBDA_COM, 'topk': TOPK,
                    'blend_tau': BLEND_TAU,
                    'signature_dim': int(q_sigs.shape[1])},
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok': len(ok),
        'n_failed': len(per_pair) - len(ok),
        'per_pair': per_pair,
        'stratified': stratified,
    }
    (OUT_DIR / 'metrics.json').write_text(json.dumps(out, indent=2, default=str))
    print(f"\n=== DONE: total {total_time:.1f}s  n_ok={len(ok)}/{len(pairs)} ===")
    print('Stratified contact_f1_vs_source + q_com_path delta (post-pre):')
    for s in ['near_present', 'absent', 'moderate', 'extreme', 'all']:
        b = stratified.get(s, {})
        n = b.get('n', 0)
        cf = b.get('contact_f1_vs_source')
        cd = b.get('q_com_path_delta')
        cf_s = f"{cf:.3f}" if cf is not None else "—"
        cd_s = f"{cd:+.3f}" if cd is not None else "—"
        print(f"  {s:14s} n={n}  contact_f1_vs_source={cf_s}  q_com_delta={cd_s}")
    return out


if __name__ == '__main__':
    run()
