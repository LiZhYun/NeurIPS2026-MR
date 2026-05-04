"""Minimal-Correspondence Escape on ALL 12 absent pairs (Variant A + B).

Variant A (minimal_corr_full): DTW + retrieve + 3-tap smoothing, no refinement.
Variant B (minimal_corr_refined): Variant A as x_init, then AnyTop prior
  projection using source-anchored hard constraints (t_init=0.3, n_steps=20,
  lambda_com=1.0, matching H_v2 refinement hyperparams).

Both read authored correspondences from
  eval/quotient_assets/minimal_corr_pairings.json
(the recently-extended file with all 12 absent pairs).

Outputs:
  eval/results/k_compare/minimal_corr_full/pair_<id>_<src>_to_<tgt>.npy
  eval/results/k_compare/minimal_corr_refined/pair_<id>_<src>_to_<tgt>.npy
  plus metrics.json in each dir.

Hard rule adherence:
  - All 12 absent pairs from the JSON are run (no silent skipping).
  - If anchor placeholder uses '<closest>', we resolve by reading motions/
    directory, filtering to target-skeleton prefix, preferring rationale
    keywords, falling back to lowest-id match. Both Skel___ and
    Skel_Skel_ naming conventions are supported (Puppy, Flamingo use the
    latter). The picked filename and candidate list are recorded in metrics.
  - Refinement hyperparameters (t_init=0.3, n_steps=20, lambda_com=1.0)
    match the H_v2 spec exactly.
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOTION_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
EVAL_PAIRS_PATH = ROOT / 'idea-stage/eval_pairs.json'
AUTHORING_JSON = ROOT / 'eval/quotient_assets/minimal_corr_pairings.json'
OUT_DIR_A = ROOT / 'eval/results/k_compare/minimal_corr_full'
OUT_DIR_B = ROOT / 'eval/results/k_compare/minimal_corr_refined'
OUT_DIR_A.mkdir(parents=True, exist_ok=True)
OUT_DIR_B.mkdir(parents=True, exist_ok=True)
CLF_V2 = ROOT / 'save/external_classifier_v2.pt'

FOOT_CH_IDX = 12
POS_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 9)
VEL_SLICE = slice(9, 12)
SMOOTH_WIN = 3

# --- Variant B refinement hyperparams (MUST match H_v2 spec) ----------------
T_INIT_REF = 0.3
N_STEPS_REF = 20
LAMBDA_COM_REF = 1.0


# Rationale-derived anchor keyword preferences per pair. If the literal source
# action exists in target pool the script will prefer it automatically, but
# when it doesn't, the JSON rationale tells us the closest-action surrogate.
ACTION_FALLBACK_KEYWORDS = {
    'pair_10_SabreToothTiger_jump_to_Parrot':   ['Hover', 'Rise', 'Walk'],
    'pair_11_Ostrich_walk_to_Crocodile':        ['Idle', 'Running'],
    'pair_12_Comodoa_attack_to_Puppy':          ['Run', 'Walk', 'IdleEnergetic'],
    'pair_13_Crab_eat_to_PolarBear':            ['Idle'],
    'pair_14_Comodoa_walk_to_Eagle':            ['Landing', 'TakeOff', 'IdleNest'],
    'pair_15_Goat_eat_to_Ant':                  ['Idle'],
    'pair_16_Isopetra_attack_to_Flamingo':      ['BendIdle', 'OneLEg', 'Walk'],
    'pair_17_Parrot_fly_to_PolarBearB':         ['Walk', 'Idle'],
    'pair_18_Centipede_run_to_Giantbee':        ['Fly', 'RapidFly'],
    'pair_19_Bird_idle_to_Pteranodon':          ['Glide', 'LowSoar', 'FlyLoop'],
    'pair_26_Ostrich_idle_to_HermitCrab':       ['Slowwalk'],
    'pair_27_Buzzard_fly_to_Centipede':         ['Run'],
}


# ==== Anchor-motion resolution =============================================
def _motion_id(fname: str) -> int:
    m = re.search(r'_(\d+)\.npy$', fname)
    return int(m.group(1)) if m else 10**9


def _motion_frames(fname: str) -> int:
    try:
        return int(np.load(MOTION_DIR / fname, mmap_mode='r').shape[0])
    except Exception:
        return 0


def pick_anchor_motion(target_skel: str, preferred_keywords):
    """Pick a real anchor filename for target_skel, supporting both
    Skel___Action_id.npy and Skel_Skel_Action_id.npy naming conventions.

    Tiebreakers: (1) target-skeleton prefix; (2) preferred rationale keyword
    present in filename; (3) lowest numeric id; (4) longest clip among ties.
    """
    all_files = sorted(os.listdir(MOTION_DIR))
    p_tri = f'{target_skel}___'
    p_dbl = f'{target_skel}_{target_skel}_'  # Puppy_Puppy_..., Flamingo_Flamingo_...
    cands = [f for f in all_files
             if (f.startswith(p_tri) or f.startswith(p_dbl)) and f.endswith('.npy')]
    # Guard PolarBear vs PolarBearB.
    if target_skel == 'PolarBear':
        cands = [f for f in cands if not f.startswith('PolarBearB___')
                 and not f.startswith('PolarBearB_PolarBearB_')]

    def keyword_rank(fname: str):
        low = fname.lower()
        for i, kw in enumerate(preferred_keywords):
            if kw.lower() in low:
                return i
        return len(preferred_keywords)

    cands_ranked = sorted(
        cands,
        key=lambda f: (keyword_rank(f), _motion_id(f), -_motion_frames(f)),
    )
    if not cands_ranked:
        raise RuntimeError(f'No motions for target skel {target_skel}')
    chosen = cands_ranked[0]
    reason = (
        f'Filtered to {target_skel} prefix ({len(cands)} clips); '
        f'preferred keywords {preferred_keywords}; picked '
        f'{chosen} (id={_motion_id(chosen)}, {_motion_frames(chosen)} frames).'
    )
    return chosen, cands, reason


# ==== Motion I/O ============================================================
def load_motion(fname, cond_entry):
    m = np.load(MOTION_DIR / fname)
    J = len(cond_entry['joints_names'])
    return m[:, :J, :].astype(np.float32)


def denormalize_motion(m_norm, cond_entry):
    J = m_norm.shape[1]
    std = cond_entry['std'][:J]
    mean = cond_entry['mean'][:J]
    if np.abs(m_norm).max() < 5:
        return m_norm * std + mean
    return m_norm


def body_scale(cond_entry):
    return float(np.linalg.norm(cond_entry['offsets'], axis=1).sum() + 1e-6)


# ==== DTW alignment =========================================================
def _paired_pos_feature(motion_phys, idxs, body_scale_hint):
    T = motion_phys.shape[0]
    parts = []
    for j in idxs:
        if j >= motion_phys.shape[1]:
            parts.append(np.zeros((T, 3), dtype=np.float32))
        else:
            parts.append(motion_phys[:, j, POS_SLICE] / max(body_scale_hint, 1e-6))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def dtw_align(src_feat, anc_feat):
    Ts = src_feat.shape[0]; Ta = anc_feat.shape[0]
    s = src_feat / (np.linalg.norm(src_feat, axis=-1, keepdims=True) + 1e-8)
    a = anc_feat / (np.linalg.norm(anc_feat, axis=-1, keepdims=True) + 1e-8)
    cost = 1.0 - s @ a.T  # [Ts, Ta]
    D = np.full((Ts + 1, Ta + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    bt = np.zeros((Ts, Ta), dtype=np.int8)
    for i in range(1, Ts + 1):
        for j in range(1, Ta + 1):
            c = cost[i - 1, j - 1]
            best = D[i - 1, j - 1]; k = 0
            if D[i - 1, j] < best: best = D[i - 1, j]; k = 1
            if D[i, j - 1] < best: best = D[i, j - 1]; k = 2
            D[i, j] = c + best
            bt[i - 1, j - 1] = k
    i, j = Ts - 1, Ta - 1
    path = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        k = bt[i, j]
        if k == 0: i -= 1; j -= 1
        elif k == 1: i -= 1
        else: j -= 1
    path.reverse()
    mapping = np.zeros(Ts, dtype=np.int64)
    buckets = [[] for _ in range(Ts)]
    for ii, jj in path:
        buckets[ii].append(jj)
    for t in range(Ts):
        if buckets[t]:
            mapping[t] = int(np.median(buckets[t]))
        else:
            mapping[t] = mapping[t - 1] if t > 0 else 0
    stretch = Ta / max(Ts, 1)
    return mapping, float(stretch), len(path)


# ==== Projection + smoothing ================================================
def ma_smooth_positions(arr3, win=3):
    if win <= 1:
        return arr3
    T, J, D = arr3.shape
    pos = arr3[:, :, POS_SLICE].copy()
    k = win; pad = k // 2
    padded = np.pad(pos, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    sm = np.zeros_like(pos)
    for t in range(T):
        sm[t] = padded[t:t + k].mean(axis=0)
    out = arr3.copy()
    out[:, :, POS_SLICE] = sm
    return out


def retarget_dtw_project(src_phys, anchor_phys, src_idxs, tgt_idxs,
                          src_bs, tgt_bs):
    src_feat = _paired_pos_feature(src_phys, src_idxs, src_bs)
    anc_feat = _paired_pos_feature(anchor_phys, tgt_idxs, tgt_bs)
    mapping, stretch, plen = dtw_align(src_feat, anc_feat)

    retarget = anchor_phys[mapping].copy()  # [T_s, J_t, 13]
    scale_ratio = tgt_bs / max(src_bs, 1e-6)
    for si, ti in zip(src_idxs, tgt_idxs):
        if si >= src_phys.shape[1] or ti >= retarget.shape[1]:
            continue
        retarget[:, ti, POS_SLICE] = src_phys[:, si, POS_SLICE] * scale_ratio
        retarget[:, ti, ROT_SLICE] = src_phys[:, si, ROT_SLICE]
        retarget[:, ti, VEL_SLICE] = src_phys[:, si, VEL_SLICE] * scale_ratio
        retarget[:, ti, FOOT_CH_IDX] = src_phys[:, si, FOOT_CH_IDX]
    if src_idxs[0] == 0 and tgt_idxs[0] == 0:
        retarget[:, 0, POS_SLICE] = src_phys[:, 0, POS_SLICE] * scale_ratio
        retarget[:, 0, VEL_SLICE] = src_phys[:, 0, VEL_SLICE] * scale_ratio
        retarget[:, 0, ROT_SLICE] = src_phys[:, 0, ROT_SLICE]
    retarget = ma_smooth_positions(retarget, win=SMOOTH_WIN)
    return retarget.astype(np.float32), mapping, stretch, plen


# ==== Eval helpers ==========================================================
def classify_motion(clf, motion_phys, skel_cond, extract_fn, recover_fn):
    J_skel = skel_cond['offsets'].shape[0]
    m = motion_phys
    if m.shape[1] > J_skel:
        m = m[:, :J_skel]
    try:
        positions = recover_fn(m.astype(np.float32))
    except Exception:
        return None
    parents = skel_cond['parents'][:J_skel]
    feats = extract_fn(positions, parents)
    if feats is None or feats.shape[0] < 4:
        return None
    return clf.predict_label(feats)


def contact_f1(pred_sched, gt_sched, thresh=0.5):
    pred = (np.asarray(pred_sched) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(gt_sched) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


def _l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    T_ = min(a.shape[0], b.shape[0])
    return float(np.linalg.norm((a[:T_] - b[:T_]).reshape(-1)))


def acc_smoothness(motion_phys):
    pos = motion_phys[:, :, POS_SLICE]
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    return float(np.linalg.norm(acc, axis=-1).mean())


def skating_metric(motion_phys):
    vel = motion_phys[:, :, VEL_SLICE]
    con = motion_phys[:, :, FOOT_CH_IDX] > 0.5
    v = np.linalg.norm(vel, axis=-1)
    if con.sum() == 0:
        return 0.0
    return float((v[con]).mean())


# ==== Main ==================================================================
def main():
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)
    with open(EVAL_PAIRS_PATH) as f:
        eval_pairs = json.load(f)['pairs']
    pair_by_id = {p['pair_id']: p for p in eval_pairs}
    with open(AUTHORING_JSON) as f:
        authoring = json.load(f)

    from eval.external_classifier import extract_classifier_features, ACTION_CLASSES  # noqa
    from eval.train_external_classifier_v2 import V2Classifier
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.quotient_extractor import extract_quotient
    from eval.anytop_projection import anytop_project
    from eval.run_retrieve_refine_v2 import build_source_anchored_contact_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = V2Classifier(str(CLF_V2), device=device)
    print(f'Loaded V2 classifier (arch={clf.arch}) on {device}')

    ordered_keys = [k for k in authoring.keys() if k.startswith('pair_')]
    print(f'Found {len(ordered_keys)} authored pairs in JSON')

    base_cfg = {
        'authoring_json': str(AUTHORING_JSON),
        'pair_keys': ordered_keys,
        'smooth_win_frames': SMOOTH_WIN,
        'classifier_ckpt': str(CLF_V2),
    }
    results_A = {
        'config': {'method': 'minimal_correspondence_dtw_retrieval', **base_cfg},
        'pairs': [],
    }
    results_B = {
        'config': {
            'method': 'minimal_correspondence_dtw_retrieval_plus_anytop_projection',
            'refinement': {'t_init': T_INIT_REF, 'n_steps': N_STEPS_REF,
                           'lambda_com': LAMBDA_COM_REF,
                           'hard_constraint_source': 'build_source_anchored_contact_mask'},
            **base_cfg,
        },
        'pairs': [],
    }

    t_all = time.time()
    for key in ordered_keys:
        entry = authoring[key]
        m = re.match(r'pair_(\d+)_', key)
        pid = int(m.group(1)) if m else -1
        src_skel = entry['source_skel']; tgt_skel = entry['target_skel']
        src_action = entry['source_action']
        pair_meta = pair_by_id.get(pid, {})
        src_fname = pair_meta.get('source_fname', '')
        family_gap = pair_meta.get('family_gap', 'unknown')

        pairs_list = entry['pairs']
        src_idxs = [p['src'] for p in pairs_list]
        tgt_idxs = [p['tgt'] for p in pairs_list]
        semantics = [p['semantic'] for p in pairs_list]

        print(f'\n=== {key}: {src_skel}({src_action}) -> {tgt_skel} [{family_gap}] ===')
        print(f'  authored {len(pairs_list)} correspondences, src_fname={src_fname}')

        pref_kws = ACTION_FALLBACK_KEYWORDS.get(key, [src_action])
        try:
            anchor_fname, anchor_cands, anchor_reason = pick_anchor_motion(tgt_skel, pref_kws)
        except Exception as e:
            print(f'  !! anchor resolution failed: {e}')
            for results in (results_A, results_B):
                results['pairs'].append({
                    'pair_id': pid, 'pair_key': key, 'status': 'error',
                    'error': f'anchor resolution failed: {e}',
                })
            continue
        print(f'  anchor: {anchor_fname}')
        print(f'  reason: {anchor_reason}')

        rec_common = {
            'pair_id': pid, 'pair_key': key,
            'src_fname': src_fname, 'src_skel': src_skel,
            'src_action': src_action, 'tgt_skel': tgt_skel,
            'family_gap': family_gap,
            'n_paired_bones': len(pairs_list),
            'paired_semantics': semantics,
            'src_idxs': src_idxs, 'tgt_idxs': tgt_idxs,
            'anchor_motion_chosen': anchor_fname,
            'anchor_motion_choice_reason': anchor_reason,
            'anchor_motion_candidates': anchor_cands,
            'anchor_motion_authoring_placeholder': entry.get('anchor_motion', ''),
        }
        rec_A = dict(rec_common); rec_B = dict(rec_common)
        t_pair = time.time()

        try:
            # --- Load motions ---
            src_norm = load_motion(src_fname, cond[src_skel])
            src_phys = denormalize_motion(src_norm, cond[src_skel])
            anc_norm = load_motion(anchor_fname, cond[tgt_skel])
            anc_phys = denormalize_motion(anc_norm, cond[tgt_skel])

            src_bs = body_scale(cond[src_skel]); tgt_bs = body_scale(cond[tgt_skel])
            for rec in (rec_A, rec_B):
                rec['src_body_scale'] = src_bs; rec['tgt_body_scale'] = tgt_bs
                rec['body_scale_ratio'] = tgt_bs / max(src_bs, 1e-6)
                rec['src_frames'] = int(src_phys.shape[0])
                rec['anchor_frames'] = int(anc_phys.shape[0])

            # --- Variant A: DTW-retarget ---
            retarget_A, mapping, stretch, plen = retarget_dtw_project(
                src_phys, anc_phys, src_idxs, tgt_idxs, src_bs, tgt_bs,
            )
            dtw_summary = {
                'stretch_factor': stretch,
                'mean_mapping': float(np.mean(mapping)),
                'path_len': plen,
                'unique_anchor_frames_used': int(len(set(mapping.tolist()))),
                'n_src_frames': int(src_phys.shape[0]),
                'n_anc_frames': int(anc_phys.shape[0]),
            }
            for rec in (rec_A, rec_B):
                rec['dtw'] = dtw_summary

            out_A = OUT_DIR_A / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_A, retarget_A.astype(np.float32))
            rec_A['out_path'] = str(out_A)

            # --- Variant B: AnyTop prior projection on Variant A ---
            # Source Q for contact schedule + COM path.
            src_q = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTION_DIR))
            tgt_groups = contact_groups.get(tgt_skel, {})
            tgt_groups_clean = {k: v for k, v in tgt_groups.items()
                                if not str(k).startswith('_')}
            T_A, J_A, _ = retarget_A.shape
            source_contact_mask = build_source_anchored_contact_mask(
                src_q, tgt_skel, tgt_groups_clean,
                contact_groups.get(src_skel, {}),
                n_frames_target=T_A, n_joints_target=J_A,
            )
            # Source COM path rescaled to target body scale.
            src_com = np.asarray(src_q['com_path']).astype(np.float32)
            src_bs_q = float(src_q['body_scale'])
            scale_ratio_q = tgt_bs / max(src_bs_q, 1e-6)
            idx_r = np.clip(np.linspace(0, src_com.shape[0] - 1, T_A).astype(int),
                             0, src_com.shape[0] - 1)
            src_com_resampled = src_com[idx_r] * scale_ratio_q
            hard_con = {'contact_positions': source_contact_mask,
                        'com_path': src_com_resampled}

            t_ref0 = time.time()
            proj = anytop_project(retarget_A, tgt_skel,
                                  hard_constraints=hard_con,
                                  t_init=T_INIT_REF, n_steps=N_STEPS_REF,
                                  lambda_com=LAMBDA_COM_REF, device=str(device))
            rec_B['refine_runtime_s'] = float(proj['runtime_seconds'])
            rec_B['refine_ckpt'] = str(proj.get('ckpt_path', ''))
            retarget_B = proj['x_refined'].astype(np.float32)
            # Optionally re-apply mild smoothing (consistent with Variant A pipeline)
            retarget_B = ma_smooth_positions(retarget_B, win=SMOOTH_WIN)

            out_B = OUT_DIR_B / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_B, retarget_B)
            rec_B['out_path'] = str(out_B)

            # --- Eval both variants ---
            def eval_one(retarget_variant, rec, variant_name):
                tgt_pred = classify_motion(
                    clf, retarget_variant, cond[tgt_skel],
                    extract_classifier_features, recover_from_bvh_ric_np,
                )
                src_pred = classify_motion(
                    clf, src_phys, cond[src_skel],
                    extract_classifier_features, recover_from_bvh_ric_np,
                )
                rec['tgt_pred'] = tgt_pred
                rec['src_classifier_pred'] = src_pred
                rec['label_match'] = bool(tgt_pred == src_action)
                rec['behavior_preserved'] = bool(
                    src_pred is not None and tgt_pred == src_pred
                )
                # Q-components via tmp motion under motions/ (extract_quotient
                # takes fname + motion_dir).
                tmp_fname = f'__mcfull_{variant_name}_p{pid:02d}.npy'
                tmp_path = MOTION_DIR / tmp_fname
                try:
                    np.save(tmp_path, retarget_variant.astype(np.float32))
                    q_out = extract_quotient(
                        tmp_fname, cond[tgt_skel],
                        contact_groups=contact_groups,
                        motion_dir=str(MOTION_DIR),
                    )
                finally:
                    if tmp_path.exists():
                        try: tmp_path.unlink()
                        except Exception: pass
                ss = np.asarray(src_q['contact_sched'])
                rs = np.asarray(q_out['contact_sched'])
                T = min(ss.shape[0], rs.shape[0])
                idx = np.clip(np.linspace(0, ss.shape[0] - 1, T).astype(int),
                               0, ss.shape[0] - 1)
                ss_t = ss[idx]
                ss_agg = ss_t.sum(axis=1) if ss_t.ndim == 2 else ss_t
                rs_agg = rs.sum(axis=1) if rs.ndim == 2 else rs
                rec['contact_f1_vs_source'] = contact_f1(rs_agg, ss_agg)
                rec['q_com_path_l2'] = _l2(src_q['com_path'], q_out['com_path'])
                rec['q_heading_vel_l2'] = _l2(src_q['heading_vel'], q_out['heading_vel'])
                rec['q_cadence_abs_diff'] = float(abs(float(src_q['cadence']) - float(q_out['cadence'])))
                T_ = min(ss_agg.size, rs_agg.size)
                rec['q_contact_sched_l2'] = float(np.linalg.norm(ss_agg[:T_] - rs_agg[:T_]))
                rec['accel_smoothness'] = acc_smoothness(retarget_variant)
                rec['skating'] = skating_metric(retarget_variant)
                rec['status'] = 'ok'

            eval_one(retarget_A, rec_A, 'A')
            eval_one(retarget_B, rec_B, 'B')

            for rec in (rec_A, rec_B):
                rec['wall_time_s'] = float(time.time() - t_pair)

            print(f'  A: tgt_pred={rec_A["tgt_pred"]} '
                  f'lm={rec_A["label_match"]} cF1={rec_A["contact_f1_vs_source"]:.3f}')
            print(f'  B: tgt_pred={rec_B["tgt_pred"]} '
                  f'lm={rec_B["label_match"]} cF1={rec_B["contact_f1_vs_source"]:.3f}')
        except Exception as e:
            traceback.print_exc()
            rec_A['status'] = 'error'; rec_A['error'] = str(e)
            rec_B['status'] = 'error'; rec_B['error'] = str(e)

        results_A['pairs'].append(rec_A)
        results_B['pairs'].append(rec_B)

    def summarize(res, label):
        ok = [p for p in res['pairs'] if p.get('status') == 'ok']
        preds = [p['tgt_pred'] for p in ok if p.get('tgt_pred') is not None]
        pred_counts = {k: preds.count(k) for k in sorted(set(preds))}
        if ok:
            mean_lm = float(np.mean([int(p['label_match']) for p in ok]))
            mean_beh = float(np.mean([int(p['behavior_preserved']) for p in ok]))
            mean_cf1 = float(np.mean([p['contact_f1_vs_source'] for p in ok]))
        else:
            mean_lm = mean_beh = 0.0; mean_cf1 = None
        s = {
            'n_authored': len(res['pairs']),
            'n_ok': len(ok),
            'mean_label_match': mean_lm,
            'mean_behavior_preserved': mean_beh,
            'mean_contact_f1_vs_source': mean_cf1,
            'median_q_com_path_l2': float(np.median([p['q_com_path_l2'] for p in ok])) if ok else None,
            'median_q_heading_vel_l2': float(np.median([p['q_heading_vel_l2'] for p in ok])) if ok else None,
            'median_q_contact_sched_l2': float(np.median([p['q_contact_sched_l2'] for p in ok])) if ok else None,
            'median_q_cadence_abs_diff': float(np.median([p['q_cadence_abs_diff'] for p in ok])) if ok else None,
            'mean_accel_smoothness': float(np.mean([p['accel_smoothness'] for p in ok])) if ok else None,
            'mean_skating': float(np.mean([p['skating'] for p in ok])) if ok else None,
            'predicted_classes_distribution': pred_counts,
            'n_distinct_predicted_classes': len(pred_counts),
            'codex_verdict': {
                'threshold_label_match_0.25': mean_lm >= 0.25,
                'threshold_contact_f1_0.35': (mean_cf1 or 0) >= 0.35,
                'threshold_diversity_3classes': len(pred_counts) >= 3,
            },
            'baseline_context': {
                'prior_absent_label_match_K_v2clf': 0.083,
                'prior_absent_label_match_minimal_corr_5pair': 0.20,
                'prior_absent_contact_f1_K': 0.067,
                'prior_absent_contact_f1_H_v4': 0.459,
                'prior_absent_contact_f1_M2M_lite': 0.362,
                'prior_absent_contact_f1_minimal_corr_5pair': 0.40,
            },
        }
        s['codex_verdict']['all_passed'] = all([
            s['codex_verdict']['threshold_label_match_0.25'],
            s['codex_verdict']['threshold_contact_f1_0.35'],
            s['codex_verdict']['threshold_diversity_3classes'],
        ])
        res['summary'] = s
        print(f'\n======== SUMMARY ({label}) ========')
        print(json.dumps(s, indent=2, default=float))

    summarize(results_A, 'VARIANT_A_minimal_corr_full')
    summarize(results_B, 'VARIANT_B_minimal_corr_refined')
    results_A['runtime_s'] = time.time() - t_all
    results_B['runtime_s'] = time.time() - t_all

    with open(OUT_DIR_A / 'metrics.json', 'w') as f:
        json.dump(results_A, f, indent=2, default=float)
    with open(OUT_DIR_B / 'metrics.json', 'w') as f:
        json.dump(results_B, f, indent=2, default=float)

    # Update unified comparison json for paper tables
    unified_path = ROOT / 'idea-stage/unified_method_comparison_v2_full.json'
    try:
        with open(unified_path) as f:
            unified = json.load(f)
    except Exception:
        unified = {}

    def row(res):
        s = res['summary']
        return {
            'absent_label_match': s['mean_label_match'],
            'absent_behavior_preserved': s['mean_behavior_preserved'],
            'absent_contact_f1_vs_source': s['mean_contact_f1_vs_source'],
            'absent_predicted_classes': s['predicted_classes_distribution'],
            'n_distinct_predicted_classes': s['n_distinct_predicted_classes'],
            'n_pairs_evaluated': s['n_ok'],
            'codex_verdict': s['codex_verdict'],
            'source': str(AUTHORING_JSON),
        }
    unified['minimal_corr_full'] = row(results_A)
    unified['minimal_corr_refined'] = row(results_B)
    with open(unified_path, 'w') as f:
        json.dump(unified, f, indent=2, default=float)
    print(f'\nWrote updates to {unified_path}')
    print(f'Total runtime: {time.time() - t_all:.1f}s')


if __name__ == '__main__':
    main()
