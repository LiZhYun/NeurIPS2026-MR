"""Idea H v2: retrieve-then-refine with SOURCE-anchored hard constraints.

Iteration over v1:
  * v1 used the RETRIEVED clip's own foot-contact channel + COM as hard
    constraints -> refinement just smoothed the retrieved clip, never migrated
    toward source behavior. Empirically v1 ~= q_retrieval on classifier labels
    (both 0.167 overall; 0.000 on support-absent with v1 classifier).

H_v2 changes:
  1. Extract source Q, remap contact schedule to target's contact groups.
  2. Build contact_positions[T, J_tgt] BINARY mask from source's remapped
     schedule: for each frame t, for each group g active in source_sched[t,g]
     >= 0.5, mark every joint in target's contact_groups[tgt_skel][g] as
     contact=1. Remaining joints = 0.
  3. Use SOURCE's COM path (scaled by target body_scale / source body_scale)
     as com_path soft guidance, not retrieved's.
  4. Keep retrieval + t_init + n_steps identical to v1 for A/B comparison.

Wall time expected similar to v1 (~75s total on 30 pairs).
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

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'

OUT_DIR = ROOT / 'eval/results/k_compare/retrieve_refine_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_INIT = 0.3
N_STEPS = 20
LAMBDA_COM = 1.0
FOOT_CH_IDX = 12
POS_Y_IDX = 1


def l2_norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def cosine_sim(a, b):
    return l2_norm(a) @ l2_norm(b).T


def q_component_l2(q_src: dict, q_tgt: dict) -> dict:
    def _l2(a, b):
        a = np.asarray(a); b = np.asarray(b)
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


def contact_f1(sched_rec: np.ndarray, sched_tgt: np.ndarray, thresh: float = 0.5) -> float:
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
    from eval.pilot_Q_experiments import q_signature
    N = len(qc['meta'])
    sigs = []
    for i in range(N):
        q = {
            'com_path': qc['com_path'][i], 'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i], 'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        sigs.append(q_signature(q))
    return np.stack(sigs)


def build_source_anchored_contact_mask(src_q, tgt_skel, target_contact_groups,
                                       src_contact_groups, n_frames_target, n_joints_target):
    """Turn source's [T_src, C_src] contact schedule into a [T_tgt, J_tgt] binary
    mask anchored to TARGET's joint indices, via contact-group name overlap.

    If source and target share named groups (LF/RF/LH/RH/L/R/LW/RW/etc),
    copy src's per-frame group activations to those joints. Otherwise fall back
    to uniform: use src's aggregate per-frame contact fraction as a global
    probability that ALL target contact-group joints are in contact.
    """
    src_sched = np.asarray(src_q['contact_sched'])  # [T_src] or [T_src, C_src]
    src_names = src_q.get('contact_group_names') or []
    tgt_names = sorted(target_contact_groups.keys()) if target_contact_groups else []
    # All joints listed in ANY target group.
    all_tgt_contact_joints = set()
    for g in tgt_names:
        all_tgt_contact_joints.update(int(j) for j in target_contact_groups[g])

    T_tgt = n_frames_target
    mask = np.zeros((T_tgt, n_joints_target), dtype=np.float32)

    # Time-resample src schedule onto T_tgt frames.
    if src_sched.ndim == 1:
        # aggregate -> broadcast per-frame fraction to all contact joints
        src_T = len(src_sched)
        idx = np.clip(np.linspace(0, src_T - 1, T_tgt).astype(int), 0, src_T - 1)
        agg = src_sched[idx]
        for t in range(T_tgt):
            if agg[t] >= 0.5:
                for j in all_tgt_contact_joints:
                    if 0 <= j < n_joints_target:
                        mask[t, j] = 1.0
        return mask

    # Grouped schedule: src_sched is [T_src, C_src]
    src_T, C_src = src_sched.shape
    idx = np.clip(np.linspace(0, src_T - 1, T_tgt).astype(int), 0, src_T - 1)
    src_resampled = src_sched[idx]  # [T_tgt, C_src]

    # Name-overlap mapping
    overlap = [n for n in tgt_names if n in src_names]
    if overlap:
        for i, tn in enumerate(tgt_names):
            if tn not in src_names:
                continue
            si = src_names.index(tn)
            activations = src_resampled[:, si]  # [T_tgt]
            joints = [int(j) for j in target_contact_groups[tn] if 0 <= int(j) < n_joints_target]
            for t in range(T_tgt):
                if activations[t] >= 0.5:
                    for j in joints:
                        mask[t, j] = 1.0
        # Non-overlapping target groups stay inactive — best we can do without
        # a specific source group. They will NOT be forced to contact.
        return mask

    # No overlap: fall back to aggregate fraction -> all target contact joints
    agg = src_resampled.mean(axis=1)
    for t in range(T_tgt):
        if agg[t] >= 0.5:
            for j in all_tgt_contact_joints:
                if 0 <= j < n_joints_target:
                    mask[t, j] = 1.0
    return mask


def stratified_summary(all_entries):
    buckets = defaultdict(list)
    for e in all_entries:
        fam = e['family_gap']
        if fam in ('near_present', 'near'):
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
        'contact_f1_vs_source', 'contact_f1_self',
        'refine_runtime_s', 'retrieval_time_s', 'wall_time_s',
    ]
    summary = {}
    for stratum, entries in buckets.items():
        summary[stratum] = {'n': len(entries)}
        for k in numeric_keys:
            vals = [e[k] for e in entries if e.get(k) is not None]
            summary[stratum][k] = float(np.mean(vals)) if vals else None
    return summary


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
    print(f"  q_sig dim: {q_sigs.shape[1]}  ({q_sigs.shape[0]} clips)")

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  device: {device}")

    per_pair = []
    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']; src_fname = p['source_fname']
        tgt_skel = p['target_skel']; src_label = p['source_label']
        family_gap = p['family_gap']
        support = int(p['support_same_label'])
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

            t_retr0 = time.time()
            cand_q_idx = np.array([fname_to_q_idx[meta[i]['fname']] for i in tgt_pool])
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

            # --- Source Q (anchored target) and retrieved pre-refine Q ---
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

            # --- Load retrieved motion for init ---
            retrieved = np.load(MOTIONS_DIR / retr_fname).astype(np.float32)
            T_retr, J_retr, F_retr = retrieved.shape

            # --- v2 CHANGE: SOURCE-anchored hard constraints ---
            tgt_groups = contact_groups.get(tgt_skel, {})
            tgt_groups_clean = {k: v for k, v in tgt_groups.items()
                                if not str(k).startswith('_')}
            source_contact_mask = build_source_anchored_contact_mask(
                src_q, tgt_skel, tgt_groups_clean,
                contact_groups.get(src_skel, {}),
                n_frames_target=T_retr, n_joints_target=J_retr
            )

            # Source COM path rescaled to target body_scale, then broadcast Y only
            # (we use y-channel only in anytop_projection).
            src_com_path = np.asarray(src_q['com_path']).astype(np.float32)  # [T_src, 3]
            src_bs = float(src_q['body_scale'])
            # Compute target body_scale from cond (simple sum-of-bone-lengths proxy)
            tgt_offsets = cond_dict[tgt_skel]['offsets']
            tgt_bs = float(np.linalg.norm(tgt_offsets, axis=1).sum() + 1e-6)
            scale_ratio = tgt_bs / max(src_bs, 1e-6)
            src_com_resampled = np.zeros((T_retr, 3), dtype=np.float32)
            t_src = src_com_path.shape[0]
            idx = np.clip(np.linspace(0, t_src - 1, T_retr).astype(int), 0, t_src - 1)
            src_com_resampled = src_com_path[idx] * scale_ratio

            hard_con = {'contact_positions': source_contact_mask,
                        'com_path': src_com_resampled}

            # --- refine ---
            t_ref0 = time.time()
            proj = anytop_project(retrieved, tgt_skel,
                                  hard_constraints=hard_con,
                                  t_init=T_INIT, n_steps=N_STEPS,
                                  lambda_com=LAMBDA_COM, device=device)
            rec['refine_runtime_s'] = float(proj['runtime_seconds'])
            x_refined = proj['x_refined']

            # --- save output ---
            out_fname = f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            out_path = OUT_DIR / out_fname
            np.save(out_path, x_refined.astype(np.float32))
            rec['output_file'] = out_fname

            # --- post-refinement Q ---
            tmp_name = f'__retrieve_refine_v2_tmp_pair_{pid:02d}.npy'
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

            def _delta(post, pre):
                if post is None or pre is None: return None
                return float(post - pre)
            rec['q_com_path_delta']     = _delta(rec['q_com_path_l2'], rec['q_com_path_l2_pre'])
            rec['q_heading_vel_delta']  = _delta(rec['q_heading_vel_l2'], rec['q_heading_vel_l2_pre'])
            rec['q_contact_sched_delta']= _delta(rec['q_contact_sched_l2'], rec['q_contact_sched_l2_pre'])
            rec['q_cadence_delta']      = _delta(rec['q_cadence_abs_diff'], rec['q_cadence_abs_diff_pre'])
            rec['q_limb_usage_delta']   = _delta(rec['q_limb_usage_top5_l2'], rec['q_limb_usage_top5_pre'])

            # Contact F1: refined schedule vs SOURCE schedule (the new alignment target)
            # Match shapes temporally.
            rs = np.asarray(refined_q['contact_sched'])
            ss = np.asarray(src_q['contact_sched'])
            # Resample source contact to refined T
            T_ref = rs.shape[0]
            idx = np.clip(np.linspace(0, ss.shape[0] - 1, T_ref).astype(int), 0, ss.shape[0] - 1)
            ss_aligned = ss[idx]
            # Aggregate to [T] sum
            rs_agg = rs.sum(axis=1) if rs.ndim == 2 else rs
            ss_agg = ss_aligned.sum(axis=1) if ss_aligned.ndim == 2 else ss_aligned
            rec['contact_f1_vs_source'] = contact_f1(rs_agg, ss_agg)
            rec['contact_f1_self'] = contact_f1(rs, rs)  # sanity 1.0

            rec['wall_time_s'] = float(time.time() - t_pair0)
            rec['status'] = 'ok'
            com_delta = rec.get('q_com_path_delta', 0) or 0
            print(f"  ok  retr={retr_fname}  "
                  f"q_com pre={rec['q_com_path_l2_pre']:.3f} -> post={rec['q_com_path_l2']:.3f} "
                  f"({com_delta:+.3f})  "
                  f"c_f1_vs_src={rec['contact_f1_vs_source']:.3f}  "
                  f"refine={rec['refine_runtime_s']:.2f}s")
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['wall_time_s'] = float(time.time() - t_pair0)
            print(f"  FAILED: {e}")
        per_pair.append(rec)

    total_time = time.time() - t_start_all
    ok = [r for r in per_pair if r['status'] == 'ok']
    failed = [r for r in per_pair if r['status'] != 'ok']
    stratified = stratified_summary(ok)
    out = {
        'method': 'retrieve_refine_v2',
        'variant': 'source_anchored_constraints',
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

    # Quick summary table
    print('\nStratified contact_f1_vs_source + q_com_path delta (post-pre):')
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
