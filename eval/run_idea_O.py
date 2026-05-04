"""Idea O: retrieve top-1 -> DTW time-align to source contact timing ->
warm-start constrained IK (source-Q anchored) -> bridge to 13-dim. SKIP Stage 3.

Hybrid of H (retrieval init) + K (source-Q anchored IK) without Stage 3's
frozen-AnyTop projection. Rationale:
  * Stage 3 tends to mint contacts (skating) because it refines in a diffusion
    prior; skipping it should reduce foot-skating.
  * Retrieval warm-start gives IK a behaviourally meaningful starting pose
    instead of flat zeros (which collapse to neutral pose).
  * DTW time-aligning the retrieved clip's contact schedule to the source's
    contact schedule ensures the warm-start theta lines up with when source
    expects feet up/down, so IK has less time-warping work to do.

Wall time: ~5-8 min on CPU (500 iter IK x 30 pairs).

Outputs:
  * per-pair motion -> eval/results/k_compare/idea_O_dtw_morph/pair_<id>_<src>_to_<tgt>.npy
  * metrics.json    -> eval/results/k_compare/idea_O_dtw_morph/metrics.json
  * classifier_eval.json -> eval/results/k_compare/idea_O_dtw_morph/classifier_eval.json
"""
from __future__ import annotations
import os
import sys
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/idea_O_dtw_morph'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PAIRS = PROJECT_ROOT / 'idea-stage/eval_pairs.json'
META_PATH = PROJECT_ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
COND_PATH = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json'
CLF_CKPT = PROJECT_ROOT / 'save/external_classifier.pt'

N_IK_ITERS = 500
FOOT_CH_IDX = 12


# ============================ DTW (hand-rolled) ============================

def dtw_path(a: np.ndarray, b: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """Symmetric DP DTW on 1D sequences a, b (L1 cost).

    Returns (cost, path) where path = [(i, j), ...] with i indexing a and j
    indexing b. Time/space O(len(a) * len(b)); fine for T<=400.
    """
    n, m = len(a), len(b)
    # Cost matrix (L1 on binary signals == Hamming-like).
    C = np.abs(a[:, None] - b[None, :]).astype(np.float64)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j],
                                             D[i, j - 1],
                                             D[i - 1, j - 1])
    # Backtrace
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return float(D[n, m]), list(reversed(path))


def build_time_warp_from_dtw(retrieved_contact: np.ndarray,
                             source_contact: np.ndarray) -> Tuple[np.ndarray, float]:
    """Align RETRIEVED contact signal to SOURCE contact signal via DTW.

    Returns (warp, stretch) where:
      warp : int array of length T_src, warp[t_src] = frame index in the
             retrieved clip that best aligns to source frame t_src.
      stretch : mean absolute log(retrieved-frame-spacing) (how much
                time-warping occurred; 0 = perfect 1:1 alignment).
    """
    r = (retrieved_contact >= 0.5).astype(np.float32)
    s = (source_contact >= 0.5).astype(np.float32)
    _, path = dtw_path(r, s)
    T_src = len(s)
    # For each source frame, pick the last retrieved frame aligned to it
    # (average would blur; this gives a deterministic pick).
    mapping = [None] * T_src
    for ri, si in path:
        mapping[si] = ri
    # Fill any gaps via forward-fill / backward-fill.
    last = None
    for t in range(T_src):
        if mapping[t] is None and last is not None:
            mapping[t] = last
        elif mapping[t] is not None:
            last = mapping[t]
    # Backfill from the other direction if start had None.
    for t in range(T_src - 1, -1, -1):
        if mapping[t] is None:
            # fallback: 0
            mapping[t] = 0
    warp = np.asarray(mapping, dtype=np.int64)
    # Stretch: ratio of retrieved spacing to source spacing (1.0 = identity).
    if T_src >= 2:
        step_r = np.diff(warp).astype(np.float32)
        # Mean of |step_r - 1| measures deviation from identity.
        stretch = float(np.mean(np.abs(step_r - 1.0)))
    else:
        stretch = 0.0
    return warp, stretch


# ============================ 6D -> axis-angle ============================

def motion_13d_to_theta(motion_13d: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Convert retrieved 13-dim motion's stored 6D rotations (channel 3:9)
    into a per-joint axis-angle theta [T, J, 3] suitable as an IK warm-start.

    Recall: motion_13d[t, j, 3:9] for j>=1 stores PARENT's local rotation;
    motion_13d[t, 0, 3:9] stores root-yaw rotation.

    To recover joint j's local rotation R_j, find any child c (parents[c]==j)
    and read motion_13d[t, c, 3:9].  Joint 0's own rotation lives in slot 0.
    Leaf joints without children get identity.
    """
    from utils.rotation_conversions import rotation_6d_to_matrix_np
    import torch
    from utils.rotation_conversions import matrix_to_axis_angle

    T, J, F = motion_13d.shape
    assert F == 13, f'expected 13 channels, got {F}'
    cont6d_stored = motion_13d[..., 3:9].astype(np.float32)   # [T, J, 6]
    # cont6d_stored[:, c] = R_parents[c] for c>=1; cont6d_stored[:, 0] = R_root.
    # We want R_j for each j.
    cont6d_local = np.zeros_like(cont6d_stored)
    cont6d_local[:] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    cont6d_local[:, 0] = cont6d_stored[:, 0]                   # root yaw
    # For non-root joints: find a child slot that has parents[c] == j.
    children_of = [[] for _ in range(J)]
    for c in range(1, J):
        p = int(parents[c])
        if 0 <= p < J and p != c:
            children_of[p].append(c)
    for j in range(1, J):
        kids = children_of[j]
        if kids:
            c0 = kids[0]
            cont6d_local[:, j] = cont6d_stored[:, c0]
        # else: stays identity (leaf joint)
    # 6D -> 3x3 (numpy-safe, normalises internally).
    R = rotation_6d_to_matrix_np(cont6d_local)                # [T, J, 3, 3]
    # 3x3 -> axis-angle via pytorch helper.
    Rt = torch.tensor(R, dtype=torch.float32)
    theta = matrix_to_axis_angle(Rt).numpy().astype(np.float32)
    return theta                                              # [T, J, 3]


# ============================ Q/contact utils (reuse) ============================

def load_assets():
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR  # noqa: F401
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        cg = json.load(f)
    return cond, cg, str(MOTIONS_DIR)


def remap_contact_sched(q_src, src_skel, tgt_skel, contact_groups):
    """Copy of k_pipeline's remap (deterministic, name-match-with-fallback)."""
    src_sched = np.asarray(q_src['contact_sched'])
    src_names = q_src.get('contact_group_names') or []
    T = src_sched.shape[0]
    if tgt_skel not in contact_groups:
        return src_sched.copy(), src_names
    tgt_groups = contact_groups[tgt_skel]
    tgt_names = sorted(tgt_groups.keys())
    C_tgt = len(tgt_names)
    if src_sched.ndim == 1:
        broadcast = np.tile(src_sched[:, None], (1, C_tgt))
        return broadcast.astype(np.float32), tgt_names
    overlap = [n for n in tgt_names if n in src_names]
    if overlap:
        sched = np.zeros((T, C_tgt), dtype=np.float32)
        for i, tn in enumerate(tgt_names):
            if tn in src_names:
                sched[:, i] = src_sched[:, src_names.index(tn)]
            else:
                sched[:, i] = src_sched.mean(axis=1)
        return sched, tgt_names
    agg = src_sched.mean(axis=1) if src_sched.ndim == 2 else src_sched
    sched = np.tile(agg[:, None], (1, C_tgt)).astype(np.float32)
    return sched, tgt_names


def build_q_star(q_src, src_skel, tgt_skel, contact_groups, cond):
    sched_tgt, tgt_group_names = remap_contact_sched(
        q_src, src_skel, tgt_skel, contact_groups)
    limb_src = np.asarray(q_src['limb_usage'], dtype=np.float32)
    K_tgt = len(cond[tgt_skel].get('kinematic_chains', []))
    if K_tgt == 0:
        limb_tgt = limb_src.copy()
    elif K_tgt == limb_src.size:
        limb_tgt = limb_src.copy()
    else:
        x_src = np.linspace(0, 1, num=limb_src.size, endpoint=True)
        x_tgt = np.linspace(0, 1, num=K_tgt, endpoint=True)
        limb_tgt = np.interp(x_tgt, x_src, limb_src).astype(np.float32)
        limb_tgt = limb_tgt / (limb_tgt.sum() + 1e-8)
    return {
        'com_path':            q_src['com_path'].astype(np.float32),
        'heading_vel':         q_src['heading_vel'].astype(np.float32),
        'contact_sched':       sched_tgt,
        'contact_group_names': tgt_group_names,
        'cadence':             float(q_src['cadence']),
        'limb_usage':          limb_tgt,
        'body_scale':          float(q_src.get('body_scale', 1.0)),
        'n_joints':            int(cond[tgt_skel]['offsets'].shape[0]),
        'n_frames':            int(q_src['com_path'].shape[0]),
    }


def contact_f1(sched_rec, sched_tgt, thresh=0.5):
    pred = (np.asarray(sched_rec) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(sched_tgt) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


def skating_proxy(motion_13, contact_groups, tgt_skel, cond,
                  fps=30, contact_thresh=0.5):
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_ric_np,
    )
    pos = recover_from_bvh_ric_np(motion_13.astype(np.float32))
    T, J, _ = pos.shape
    contact = motion_13[..., 12]
    if contact_groups is None or tgt_skel not in contact_groups:
        foot_joints = np.unique(np.where(contact > contact_thresh)[1]).tolist()
    else:
        foot_joints = []
        for grp in contact_groups[tgt_skel].values():
            foot_joints.extend([int(j) for j in grp if 0 <= int(j) < J])
        foot_joints = sorted(set(foot_joints))
    if not foot_joints:
        return 0.0
    y_vel = np.zeros((T, len(foot_joints)))
    x_vel = np.zeros_like(y_vel)
    z_vel = np.zeros_like(y_vel)
    if T >= 2:
        dp = pos[1:, foot_joints, :] - pos[:-1, foot_joints, :]
        x_vel[1:] = dp[..., 0]; y_vel[1:] = dp[..., 1]; z_vel[1:] = dp[..., 2]
    horiz_speed = np.sqrt(x_vel ** 2 + z_vel ** 2) * fps
    in_contact = (contact[:, foot_joints] > contact_thresh)
    if not in_contact.any():
        return 0.0
    return float(horiz_speed[in_contact].mean())


def _stratum_label(p):
    if p['support_same_label'] == 0:
        return 'absent'
    if p['family_gap'] == 'near':
        return 'near_present'
    if p['family_gap'] == 'moderate':
        return 'moderate'
    if p['family_gap'] == 'extreme':
        return 'extreme'
    return 'other'


# ============================ retrieval (Q cosine) ============================

def l2_norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def cosine_sim(a, b):
    return l2_norm(a) @ l2_norm(b).T


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


# ============================ aggregate contact ============================

def motion_aggregate_contact(motion_13: np.ndarray) -> np.ndarray:
    """Binary aggregate contact over time: any foot joint in contact -> 1."""
    c = motion_13[..., FOOT_CH_IDX]
    return (c.max(axis=1) > 0.5).astype(np.float32)


def q_contact_aggregate(contact_sched: np.ndarray) -> np.ndarray:
    """Aggregate multi-group contact schedule to a binary per-frame signal."""
    s = np.asarray(contact_sched)
    if s.ndim == 1:
        return (s > 0.5).astype(np.float32)
    return (s.max(axis=1) > 0.5).astype(np.float32)


# ============================ main runner ============================

def run():
    t_total_0 = time.time()
    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)

    cond, contact_groups, motion_dir = load_assets()

    q_meta = list(qc['meta'])
    fname_to_q_idx = {m['fname']: i for i, m in enumerate(q_meta)}
    skel_to_meta_idx = defaultdict(list)
    for i, m in enumerate(meta):
        skel_to_meta_idx[m['skeleton']].append(i)

    print('[O] building Q signatures...')
    q_sigs = build_q_sig_array(qc)
    print(f'  q_sig dim={q_sigs.shape[1]} N={q_sigs.shape[0]}')

    from eval.quotient_extractor import extract_quotient
    from eval.ik_solver_warmstart import solve_ik_warmstart
    from eval.ik_solver import solve_ik, _q_component_errors  # baseline for convergence
    from eval.k_pipeline_bridge import theta_to_motion_13dim, bridge_diagnostics

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  device={device}')

    per_pair = []
    stretch_log = []
    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']
        tgt_skel = p['target_skel']
        src_fname = p['source_fname']
        print(f'\n=== pair {pid:02d}  {src_skel}({src_fname}) -> {tgt_skel}  '
              f'gap={p["family_gap"]}  supp={p["support_same_label"]} ===')

        rec = {
            'pair_id': pid, 'src_skel': src_skel, 'tgt_skel': tgt_skel,
            'src_fname': src_fname, 'src_label': p['source_label'],
            'family_gap': p['family_gap'],
            'support_same_label': int(p['support_same_label']),
            'stratum': _stratum_label(p),
            'status': 'pending', 'error': None,
        }

        t0 = time.time()
        try:
            # --- 1. Retrieve top-1 target-skel clip ---
            if src_fname not in fname_to_q_idx:
                raise RuntimeError(f'source missing Q cache: {src_fname}')
            if tgt_skel not in cond:
                raise RuntimeError(f'missing cond: {tgt_skel}')
            src_q_idx = fname_to_q_idx[src_fname]
            src_q_sig = q_sigs[src_q_idx]
            tgt_pool = [i for i in skel_to_meta_idx[tgt_skel]
                        if meta[i]['fname'] != src_fname
                        and meta[i]['fname'] in fname_to_q_idx]
            if not tgt_pool:
                raise RuntimeError(f'empty tgt pool {tgt_skel}')
            cand_q_idx = np.array([fname_to_q_idx[meta[i]['fname']] for i in tgt_pool])
            t_retr0 = time.time()
            sims = cosine_sim(src_q_sig[None], q_sigs[cand_q_idx])[0]
            best = int(np.argmax(sims))
            retr_meta_idx = tgt_pool[best]
            retr_fname = meta[retr_meta_idx]['fname']
            rec['retrieved_fname'] = retr_fname
            rec['retrieved_coarse_label'] = meta[retr_meta_idx]['coarse_label']
            rec['retrieval_cosine'] = float(sims[best])
            rec['retrieval_time_s'] = float(time.time() - t_retr0)

            # --- 2. Source Q and Q* on target ---
            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            rec['n_frames_src'] = int(q_src['n_frames'])
            q_star = build_q_star(q_src, src_skel, tgt_skel, contact_groups, cond)

            # --- 3. Load retrieved motion, DTW-align to source contact timing ---
            retrieved = np.load(MOTIONS_DIR / retr_fname).astype(np.float32)
            T_retr = retrieved.shape[0]
            T_src = int(q_src['n_frames'])
            # Build aggregate binary contact signals.
            r_contact = motion_aggregate_contact(retrieved)
            s_contact = q_contact_aggregate(q_src['contact_sched'])
            warp, stretch = build_time_warp_from_dtw(r_contact, s_contact)
            rec['dtw_stretch'] = stretch
            stretch_log.append(stretch)
            # Resample retrieved along time axis using warp -> [T_src, J, 13]
            retrieved_warped = retrieved[warp]
            rec['T_src'] = T_src
            rec['T_retr'] = T_retr
            rec['T_after_warp'] = retrieved_warped.shape[0]

            # --- 4. Convert retrieved (warped) 6D rotations -> axis-angle theta ---
            tgt_parents = np.asarray(cond[tgt_skel]['parents'], dtype=np.int64)
            theta_init = motion_13d_to_theta(retrieved_warped, tgt_parents)

            # --- 5. Warm-start IK with theta_init, targeting Q* ---
            t_ik0 = time.time()
            ik_out = solve_ik_warmstart(
                q_star, cond[tgt_skel], contact_groups[tgt_skel],
                n_iters=N_IK_ITERS, verbose=False, device=device,
                theta_init=theta_init, track_convergence=True)
            rec['ik_runtime'] = float(ik_out['runtime_sec'])
            rec['ik_iters'] = N_IK_ITERS
            loss_trace = ik_out.get('loss_trace', [])
            rec['ik_final_loss'] = float(loss_trace[-1]) if loss_trace else None
            rec['ik_min_loss'] = float(min(loss_trace)) if loss_trace else None

            # Q* match errors (reconstructed vs Q*)
            q_rec = ik_out['q_reconstructed']
            errs = _q_component_errors(q_rec, q_star)
            rec['q_errors'] = errs
            rec['contact_f1_vs_qstar'] = contact_f1(
                q_rec['contact_sched'], q_star['contact_sched'])
            # Convergence: iters to reach 1.5x min loss.
            if loss_trace:
                tgt_loss = rec['ik_min_loss'] * 1.5
                for it, lv in enumerate(loss_trace):
                    if lv <= tgt_loss:
                        rec['ik_iters_to_1p5x_min'] = it
                        break
                else:
                    rec['ik_iters_to_1p5x_min'] = len(loss_trace)
            else:
                rec['ik_iters_to_1p5x_min'] = None

            # --- 6. Bridge to 13-dim (SKIP Stage 3) ---
            try:
                motion_13 = theta_to_motion_13dim(
                    ik_out['theta'], ik_out['root_pos'],
                    ik_out['positions'],
                    tgt_skel, cond,
                    contact_groups=contact_groups)
            except Exception as e:
                print(f'  bridge warning: {e}; identity-6D fallback')
                motion_13 = theta_to_motion_13dim(
                    ik_out['theta'], ik_out['root_pos'],
                    ik_out['positions'],
                    tgt_skel, cond,
                    contact_groups=contact_groups,
                    fit_rotations=False)
            diag = bridge_diagnostics(motion_13, tgt_skel, cond)
            rec['bridge_diag'] = diag

            # --- 7. Save (NO Stage 3) ---
            out_fname = f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(OUT_DIR / out_fname, motion_13)
            rec['output_file'] = out_fname
            rec['skating_proxy'] = skating_proxy(motion_13, contact_groups,
                                                 tgt_skel, cond)
            rec['status'] = 'ok'
            rec['pair_runtime_s'] = float(time.time() - t0)
            print(f'  ok  retr={retr_fname}  stretch={stretch:.3f}  '
                  f'q_com_rel={errs["com_path_rel_l2"]:.3f}  '
                  f'cF1={rec["contact_f1_vs_qstar"]:.3f}  '
                  f'skate={rec["skating_proxy"]:.3f}  '
                  f'ik={rec["ik_runtime"]:.1f}s')
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime_s'] = float(time.time() - t0)
            print(f'  FAILED: {e}')
        per_pair.append(rec)

    total_time = time.time() - t_total_0

    # ======== stratified summary ========
    strata_order = ['near_present', 'absent', 'moderate', 'extreme']
    strata = {s: [] for s in strata_order}
    for r in per_pair:
        strata[r['stratum']].append(r)

    def _stats(entries):
        ok = [e for e in entries if e['status'] == 'ok']
        if not ok:
            return {'n_total': len(entries), 'n_ok': 0}
        qs = lambda k: float(np.mean([e['q_errors'][k] for e in ok]))
        return {
            'n_total': len(entries),
            'n_ok': len(ok),
            'com_path_rel_l2':     qs('com_path_rel_l2'),
            'heading_vel_rel_l2':  qs('heading_vel_rel_l2'),
            'contact_sched_mae':   qs('contact_sched_mae'),
            'cadence_abs':         qs('cadence_abs'),
            'limb_usage_rel_l2':   qs('limb_usage_rel_l2'),
            'contact_f1':          float(np.mean([e['contact_f1_vs_qstar'] for e in ok])),
            'skating_proxy':       float(np.mean([e['skating_proxy'] for e in ok])),
            'ik_runtime':          float(np.mean([e['ik_runtime'] for e in ok])),
            'ik_iters_to_1p5x_min': float(np.mean([e['ik_iters_to_1p5x_min']
                                                    for e in ok
                                                    if e.get('ik_iters_to_1p5x_min') is not None])),
            'dtw_stretch':         float(np.mean([e['dtw_stretch'] for e in ok])),
            'pair_runtime':        float(np.mean([e['pair_runtime_s'] for e in ok])),
        }
    strata_stats = {s: _stats(strata[s]) for s in strata_order}

    out = {
        'method': 'idea_O_dtw_morph',
        'note': 'retrieve top-1 -> DTW align -> theta warm-start IK (source Q*) -> bridge; no Stage 3',
        'hparams': {'n_ik_iters': N_IK_ITERS},
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok':  sum(1 for r in per_pair if r['status'] == 'ok'),
        'n_failed': sum(1 for r in per_pair if r['status'] == 'failed'),
        'mean_dtw_stretch_all': float(np.mean(stretch_log)) if stretch_log else 0.0,
        'strata_stats': strata_stats,
        'per_pair': per_pair,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n=== DONE (Idea O): total {total_time:.1f}s  n_ok={out["n_ok"]}/{out["n_pairs"]} ===')
    print(f'mean DTW stretch: {out["mean_dtw_stretch_all"]:.3f}')
    print(f'metrics saved to {OUT_DIR / "metrics.json"}')
    return out


# ============================ classifier eval ============================

def run_classifier_eval():
    """Classify Idea-O outputs with external classifier v2; compute label
    accuracy + behavior preservation per stratum.
    """
    import torch
    from eval.external_classifier import (
        ActionClassifier, extract_classifier_features, ACTION_CLASSES,
    )
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_ric_np,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[classifier] device={device}')
    state = torch.load(str(CLF_CKPT), map_location='cpu', weights_only=False)
    clf = ActionClassifier().to(device).eval()
    clf.load_state_dict(state['model'])

    cond = np.load(COND_PATH, allow_pickle=True).item()

    with open(EVAL_PAIRS) as f:
        eval_set = json.load(f)
    pairs = eval_set['pairs']

    @torch.no_grad()
    def classify_motion(motion_13dim, skel):
        J_skel = cond[skel]['offsets'].shape[0]
        m = motion_13dim
        if m.shape[1] > J_skel:
            m = m[:, :J_skel]
        if np.abs(m).max() < 5:
            mean = cond[skel]['mean'][:J_skel]
            std = cond[skel]['std'][:J_skel]
            m = m.astype(np.float32) * std + mean
        try:
            positions = recover_from_bvh_ric_np(m.astype(np.float32))
        except Exception as e:
            return None, str(e)
        parents = cond[skel]['parents'][:J_skel]
        feats = extract_classifier_features(positions, parents)
        if feats is None:
            return None, 'feats_none'
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        logits = clf(x)
        pred_idx = int(logits.argmax(-1).item())
        return ACTION_CLASSES[pred_idx], None

    print('[classifier] caching source predictions')
    src_pred_by_pair = {}
    src_cache = {}
    for p in pairs:
        src_fname = p['source_fname']
        if src_fname in src_cache:
            src_pred_by_pair[p['pair_id']] = src_cache[src_fname]
            continue
        m = np.load(MOTIONS_DIR / src_fname)
        pred, _ = classify_motion(m, p['source_skel'])
        src_cache[src_fname] = pred
        src_pred_by_pair[p['pair_id']] = pred

    per_pair = []
    for p in pairs:
        pid = p['pair_id']
        patterns = [f for f in os.listdir(OUT_DIR)
                    if f.startswith(f'pair_{pid:02d}_') and f.endswith('.npy')]
        if not patterns:
            per_pair.append({'pair_id': pid, 'error': 'no_output'})
            continue
        motion_path = OUT_DIR / patterns[0]
        try:
            motion = np.load(motion_path)
        except Exception as e:
            per_pair.append({'pair_id': pid, 'error': f'load: {e}'})
            continue
        pred, err = classify_motion(motion, p['target_skel'])
        if pred is None:
            per_pair.append({'pair_id': pid, 'error': err or 'classify_none'})
            continue
        src_pred = src_pred_by_pair.get(pid)
        per_pair.append({
            'pair_id': pid,
            'src_skel': p['source_skel'],
            'tgt_skel': p['target_skel'],
            'src_label': p['source_label'],
            'family_gap': p['family_gap'],
            'support_same_label': p['support_same_label'],
            'tgt_pred': pred,
            'src_classifier_pred': src_pred,
            'label_match': pred == p['source_label'],
            'behavior_preserved': src_pred is not None and pred == src_pred,
        })

    valid = [r for r in per_pair if 'error' not in r]
    print(f'[classifier] idea_O: valid {len(valid)}/{len(per_pair)}')
    strata = defaultdict(list)
    for r in valid:
        if r['support_same_label'] == 0:
            strata['absent'].append(r)
        else:
            strata[r['family_gap']].append(r)
    summary = {
        'n_valid': len(valid),
        'n_total': len(per_pair),
        'overall_label_match_rate': float(np.mean([r['label_match'] for r in valid])) if valid else 0,
        'overall_behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in valid])) if valid else 0,
        'by_stratum': {},
    }
    for gap in ['near', 'absent', 'moderate', 'extreme']:
        bucket = strata.get(gap, [])
        if bucket:
            summary['by_stratum'][gap] = {
                'n': len(bucket),
                'label_match_rate': float(np.mean([r['label_match'] for r in bucket])),
                'behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in bucket])),
            }

    out = {
        'method': 'idea_O_dtw_morph',
        'eval_set': str(EVAL_PAIRS),
        'n_pairs': len(pairs),
        'per_pair': per_pair,
        'summary': summary,
    }
    with open(OUT_DIR / 'classifier_eval.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[classifier] saved {OUT_DIR / "classifier_eval.json"}')
    print('\n=== classifier summary (idea_O) ===')
    print(f"  overall label match:        {summary['overall_label_match_rate']:.3f}")
    print(f"  overall behavior preserved: {summary['overall_behavior_preserved_rate']:.3f}")
    for gap in ['near', 'absent', 'moderate', 'extreme']:
        if gap in summary['by_stratum']:
            s = summary['by_stratum'][gap]
            print(f"  {gap:8s}: n={s['n']}, lbl={s['label_match_rate']:.3f}, beh={s['behavior_preserved_rate']:.3f}")
    return out


if __name__ == '__main__':
    run()
    run_classifier_eval()
