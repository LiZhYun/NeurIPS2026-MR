"""Ablation of Idea K: skip Stage 2 (constrained IK) and diffuse from NOISE
on the target skeleton, injecting Q* as a cheap classifier-free-style guidance
signal during DDIM reverse sampling.

Goal: isolate the contribution of the constrained-IK stage in the full K
pipeline. If K > K_no_stage2 on Q preservation, Stage 2 IK is net-positive.
If they match, Stage 2 is redundant.

Pipeline:
  1. Stage 1 Q extract on source (unchanged).
  2. Build Q* on target (remap contact schedule by group names, etc.).
  3. DDIM reverse sample from Gaussian noise on frozen AnyTop A3-230k ckpt.
     At each step, compute Q on the predicted x_0 (contact-sched + COM path
     only, to stay differentiable and cheap); L2 to Q*; gradient wrt x_t;
     add with weight lambda=0.2.
  4. Denormalise and save to eval/results/k_compare/K_no_stage2/.

Same metrics as K: Q errors, contact F1, skating_proxy, wall time.
"""
from __future__ import annotations
import os
import sys
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_no_stage2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature layout per joint: [pos(3), rot6d(6), vel(3), foot_contact(1)] = 13
POS_Y_IDX = 1       # world-Y channel within a joint's 13-dim slot
POS_X_IDX = 0
POS_Z_IDX = 2
FOOT_CH_IDX = 12

DEFAULT_CKPT = 'save/A3_baseline_dataset_truebones_bs_2_latentdim_256/model000229999.pt'


# =================== helpers ===================

def load_assets():
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'),
                   allow_pickle=True).item()
    with open(PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        cg = json.load(f)
    motion_dir = os.path.join(DATASET_DIR, 'motions')
    return cond, cg, motion_dir


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


# =================== guidance core ===================

def _denorm_positions_y_com(
    x_hat_norm: torch.Tensor,        # [B, maxJ, 13, T] normalised (padded)
    mean_t: torch.Tensor,            # [maxJ, 13] on device
    std_t: torch.Tensor,             # [maxJ, 13] on device
    n_joints: int,
) -> torch.Tensor:
    """Differentiable denorm of POS_Y channel across real joints.

    Returns per-frame mean world-Y as a [B, T] tensor (COM-height proxy).
    """
    cur_norm_y = x_hat_norm[:, :n_joints, POS_Y_IDX, :]
    mean_y = mean_t[:n_joints, POS_Y_IDX].view(1, n_joints, 1)
    std_y = std_t[:n_joints, POS_Y_IDX].view(1, n_joints, 1)
    world_y = cur_norm_y * std_y + mean_y
    return world_y.mean(dim=1)  # [B, T]


def _denorm_contacts(
    x_hat_norm: torch.Tensor,        # [B, maxJ, 13, T]
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    n_joints: int,
) -> torch.Tensor:
    """Return soft contact channel (sigmoid-like via raw denorm + clamp+threshold).

    Channel 12 in the 13-dim slot is the foot-contact channel. During
    training it's trained as a binary (0/1) scalar with MSE, so the
    normalised predicted x_0 can be denormalised and then read as a soft
    score. We use a smooth surrogate (sigmoid around 0.5) so gradient flows.
    """
    cur_norm_c = x_hat_norm[:, :n_joints, FOOT_CH_IDX, :]
    mean_c = mean_t[:n_joints, FOOT_CH_IDX].view(1, n_joints, 1)
    std_c = std_t[:n_joints, FOOT_CH_IDX].view(1, n_joints, 1)
    raw = cur_norm_c * std_c + mean_c              # [B, n, T]
    return torch.sigmoid(8.0 * (raw - 0.5))          # smooth -> (0, 1)


def _contact_sched_from_soft(
    soft_contact_BnT: torch.Tensor,  # [B, n, T]
    group_joint_ids: list,           # list[list[int]] len C
) -> torch.Tensor:
    """Compute per-group contact schedule (differentiable mean over group)."""
    B, n, T = soft_contact_BnT.shape
    device = soft_contact_BnT.device
    C = len(group_joint_ids)
    out = torch.zeros(B, T, C, device=device)
    for c, idxs in enumerate(group_joint_ids):
        idxs = [int(j) for j in idxs if 0 <= int(j) < n]
        if not idxs:
            continue
        out[:, :, c] = soft_contact_BnT[:, idxs, :].mean(dim=1)
    return out  # [B, T, C]


def _center_of_mass_y(
    x_hat_norm: torch.Tensor, mean_t, std_t, n_joints
) -> torch.Tensor:
    return _denorm_positions_y_com(x_hat_norm, mean_t, std_t, n_joints)


def _q_guidance_loss(
    x_hat_norm: torch.Tensor,
    q_star: dict,
    group_joint_ids_target: list,
    n_joints: int,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    T_valid: int,
    body_scale: float,
) -> torch.Tensor:
    """L2 guidance loss on contact_sched + COM-Y path only.

    Other Q components (heading_vel, cadence, limb_usage) are either very
    hard to differentiate through (psi/limb requires FK of rotations) or
    too expensive per reverse step. Per the ablation prompt, contact +
    COM is the cheap guidance.
    """
    device = x_hat_norm.device
    loss = torch.zeros((), device=device)

    # ---- contact schedule ----
    sched_target_TC = q_star.get('contact_sched')  # numpy [T, C]
    if sched_target_TC is not None and sched_target_TC.size > 0:
        soft_contact = _denorm_contacts(x_hat_norm, mean_t, std_t, n_joints)
        sched_pred_BTC = _contact_sched_from_soft(soft_contact, group_joint_ids_target)
        # align T
        sched_target = torch.as_tensor(sched_target_TC, dtype=torch.float32, device=device)
        T_t = min(T_valid, sched_target.shape[0], sched_pred_BTC.shape[1])
        pred = sched_pred_BTC[:, :T_t, :sched_target.shape[1]]
        tgt = sched_target[:T_t, :sched_pred_BTC.shape[2]]
        loss = loss + ((pred - tgt.unsqueeze(0)) ** 2).mean()

    # ---- COM path (Y only; COM path is body-scale-normalised so multiply back) ----
    com_path = q_star.get('com_path')  # numpy [T, 3] body-scale-normalised
    if com_path is not None and com_path.size > 0:
        com_target_y_world = torch.as_tensor(
            com_path[:, 1] * body_scale, dtype=torch.float32, device=device
        )  # [T]
        com_pred_BT = _center_of_mass_y(x_hat_norm, mean_t, std_t, n_joints)
        T_c = min(T_valid, com_target_y_world.shape[0], com_pred_BT.shape[1])
        # Subtract first-frame offset so we only match the delta path
        # (COM path is anchored to 0 in frame 0 by convention).
        delta_pred = com_pred_BT[:, :T_c] - com_pred_BT[:, 0:1]
        delta_tgt = com_target_y_world[:T_c] - com_target_y_world[0:1]
        loss = loss + ((delta_pred - delta_tgt.unsqueeze(0)) ** 2).mean()

    return loss


# =================== DDIM with Q-guidance ===================

def ddim_sample_from_noise_with_qguidance(
    model, diffusion, y: dict,
    q_star: dict, group_joint_ids_target: list,
    n_joints: int, mean_t: torch.Tensor, std_t: torch.Tensor,
    n_frames: int, T_valid: int,
    max_joints: int, feature_len: int,
    body_scale: float,
    n_steps: int = 50,
    lambda_guide: float = 0.2,
    device: torch.device = None,
    seed: int = 0,
) -> np.ndarray:
    """DDIM reverse sample from x_T = N(0, I) with guidance on contact + COM.

    Returns denormalised x_refined: [T_out, n_joints, 13].
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    # Shape used by unconditional AnyTop: [1, max_joints, feature_len, n_frames]
    x_t = torch.randn((1, max_joints, feature_len, n_frames),
                      device=device, generator=gen)

    t_max = diffusion.num_timesteps
    alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32,
                              device=device)

    # Build a descending ladder of timesteps from (t_max-1) to 0.
    steps = np.linspace(t_max - 1, 0, n_steps + 1).round().astype(int).tolist()
    seen, ordered = set(), []
    for s in steps:
        s = int(s)
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    steps = ordered

    for i, t_val in enumerate(steps):
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)

        # --- guidance block (requires grad just in this scope) ---
        if lambda_guide > 0:
            with torch.enable_grad():
                x_t_req = x_t.detach().requires_grad_(True)
                out = diffusion.p_mean_variance(
                    model, x_t_req, t_tensor,
                    clip_denoised=False, model_kwargs={'y': y}
                )
                x0_hat = out['pred_xstart']
                loss = _q_guidance_loss(
                    x0_hat, q_star, group_joint_ids_target, n_joints,
                    mean_t, std_t, T_valid, body_scale
                )
                if loss.requires_grad:
                    grad = torch.autograd.grad(loss, x_t_req,
                                               retain_graph=False,
                                               allow_unused=True)[0]
                    if grad is None:
                        grad_norm = None
                    else:
                        # Normalise gradient by its own norm to stabilise
                        g_norm = grad.norm().clamp_min(1e-8)
                        grad = grad / g_norm
                        grad_norm = float(g_norm.item())
                    if grad is not None:
                        # apply guidance step BEFORE the DDIM update:
                        # shift x_t by -lambda * unit-grad
                        x_t = (x_t_req - lambda_guide * grad).detach()
                    else:
                        x_t = x_t_req.detach()
                else:
                    x_t = x_t_req.detach()
            # recompute x0_hat with updated x_t (no grad needed)
            with torch.no_grad():
                out = diffusion.p_mean_variance(
                    model, x_t, t_tensor,
                    clip_denoised=False, model_kwargs={'y': y}
                )
                x0_hat = out['pred_xstart'].detach()
        else:
            with torch.no_grad():
                out = diffusion.p_mean_variance(
                    model, x_t, t_tensor,
                    clip_denoised=False, model_kwargs={'y': y}
                )
                x0_hat = out['pred_xstart'].detach()

        # --- DDIM reverse step (eta=0) ---
        if i == len(steps) - 1 or t_val == 0:
            x_t = x0_hat
            break
        t_next = steps[i + 1]
        a_bar_t = alphas_cum[t_val]
        a_bar_next = (alphas_cum[t_next]
                      if t_next >= 0 else torch.tensor(1.0, device=device))
        eps = (x_t - torch.sqrt(a_bar_t) * x0_hat) / torch.sqrt(
            (1.0 - a_bar_t).clamp(min=1e-12))
        x_t = (torch.sqrt(a_bar_next) * x0_hat
               + torch.sqrt((1.0 - a_bar_next).clamp(min=0.0)) * eps)
        x_t = x_t.detach()

    # Denorm + crop to T_valid
    x_ref = x_t[0, :n_joints].detach().cpu().numpy()   # [n, 13, T]
    x_ref = x_ref.transpose(2, 0, 1)                    # [T, n, 13]
    T_out = min(T_valid, x_ref.shape[0])
    x_ref = x_ref[:T_out]

    mean_np = mean_t[:n_joints].detach().cpu().numpy()
    std_np = std_t[:n_joints].detach().cpu().numpy()
    x_denorm = x_ref * std_np[None] + mean_np[None]
    return x_denorm.astype(np.float32)


# =================== Q errors / contact-F1 ===================

def q_component_errors(q_rec: dict, q_tgt: dict) -> dict:
    from eval.ik_solver import _q_component_errors
    return _q_component_errors(q_rec, q_tgt)


def contact_f1(sched_rec: np.ndarray, sched_tgt: np.ndarray,
               thresh: float = 0.5) -> float:
    pred = (np.asarray(sched_rec) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(sched_tgt) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


def skating_proxy(motion_13: np.ndarray, contact_groups: dict, tgt_skel: str,
                  cond: dict, fps: int = 30, contact_thresh: float = 0.5) -> float:
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_ric_np,
    )
    pos = recover_from_bvh_ric_np(motion_13.astype(np.float32))
    T, J, _ = pos.shape
    contact = motion_13[..., FOOT_CH_IDX]
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
        x_vel[1:] = dp[..., 0]
        y_vel[1:] = dp[..., 1]
        z_vel[1:] = dp[..., 2]
    horiz_speed = np.sqrt(x_vel ** 2 + z_vel ** 2) * fps
    in_contact = (contact[:, foot_joints] > contact_thresh)
    if not in_contact.any():
        return 0.0
    return float(horiz_speed[in_contact].mean())


def reconstruct_q_from_motion13(motion_13: np.ndarray, tgt_skel: str,
                                cond: dict, contact_groups: dict) -> dict:
    """Extract Q from a generated motion_13 tensor (mirror of Stage-1 extraction
    but from a motion tensor rather than a file)."""
    from eval.effect_program import canonical_body_frame
    from eval.quotient_extractor import (
        compute_heading_velocity, compute_cadence, compute_limb_usage,
        compute_contact_schedule_grouped,
        compute_subtree_masses,
    )
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_ric_np,
    )

    info = cond[tgt_skel]
    parents = info['parents']
    offsets = info['offsets']
    chains = info.get('kinematic_chains', [])
    J = offsets.shape[0]

    m = motion_13
    if m.shape[1] > J:
        m = m[:, :J]
    positions = recover_from_bvh_ric_np(m.astype(np.float32))
    T = positions.shape[0]
    contacts = (m[..., FOOT_CH_IDX] > 0.5).astype(np.int8)

    subtree = compute_subtree_masses(parents, offsets)
    centroids, R = canonical_body_frame(positions, subtree)
    scale = float(np.linalg.norm(offsets, axis=1).sum() + 1e-6)

    com_path = (centroids - centroids[0:1]) / scale
    heading_vel = compute_heading_velocity(centroids, R, fps=30) / scale

    if tgt_skel in contact_groups:
        sched, names = compute_contact_schedule_grouped(contacts, contact_groups[tgt_skel])
    else:
        sched = contacts.sum(axis=1).astype(np.float32) / max(J, 1)
        names = None

    cadence = compute_cadence(sched, fps=30)
    limb_usage = compute_limb_usage(positions, chains, fps=30)

    return {
        'com_path': com_path.astype(np.float32),
        'heading_vel': heading_vel.astype(np.float32),
        'contact_sched': sched.astype(np.float32),
        'contact_group_names': names,
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
        'body_scale': scale,
        'n_joints': J,
        'n_frames': T,
    }


# =================== main runner ===================

def run(n_pairs_override=None, n_steps=50, lambda_guide=0.2):
    # Imports deferred so project root is on sys.path.
    from eval.quotient_extractor import extract_quotient
    from eval.run_k_pipeline_30pairs import build_q_star
    from eval.anytop_projection import (
        _load_unconditional_model, _load_t5, _build_y,
    )
    from data_loaders.truebones.truebones_utils.get_opt import get_opt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cond, contact_groups, motion_dir = load_assets()

    opt = get_opt(0)
    max_joints = opt.max_joints
    feature_len = opt.feature_len

    ckpt_path = str(PROJECT_ROOT / DEFAULT_CKPT)
    model, diffusion, m_args = _load_unconditional_model(ckpt_path, device)
    t5 = _load_t5(m_args.t5_name, device_str=str(device))

    with open(PROJECT_ROOT / 'idea-stage/eval_pairs.json') as f:
        eval_data = json.load(f)
    pairs = eval_data['pairs']
    if n_pairs_override is not None:
        pairs = pairs[:n_pairs_override]

    n_model_frames = m_args.num_frames

    per_pair = []
    bridge_diag_agg = []
    t_total_0 = time.time()

    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']
        tgt_skel = p['target_skel']
        src_fname = p['source_fname']
        print(f'\n=== pair {pid:02d}  {src_skel}({src_fname}) -> {tgt_skel}  '
              f'gap={p["family_gap"]}  supp={p["support_same_label"]} ===')

        rec = {
            'pair_id': pid,
            'source_skel': src_skel,
            'target_skel': tgt_skel,
            'source_fname': src_fname,
            'family_gap': p['family_gap'],
            'support_same_label': int(p['support_same_label']),
            'stratum': _stratum_label(p),
            'status': 'pending',
            'error': None,
        }

        t0 = time.time()
        try:
            # Stage 1.
            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            rec['n_frames'] = int(q_src['n_frames'])
            rec['cadence_src'] = float(q_src['cadence'])

            # Remap + build Q* on target.
            q_star = build_q_star(q_src, src_skel, tgt_skel, contact_groups, cond)

            # Build y conditioning for AnyTop on the target skeleton.
            T_valid = min(int(q_star['n_frames']), n_model_frames)
            y, n_joints, mean_np, std_np = _build_y(
                tgt_skel, cond, t5, n_model_frames,
                m_args.temporal_window, max_joints, device)

            mean_pad = np.zeros((max_joints, feature_len), dtype=np.float32)
            std_pad = np.ones((max_joints, feature_len), dtype=np.float32)
            mean_pad[:n_joints] = mean_np
            std_pad[:n_joints] = std_np
            mean_t = torch.from_numpy(mean_pad).to(device)
            std_t = torch.from_numpy(std_pad).to(device)

            # Target group-joint-id list in the same canonical (sorted) order
            # used by build_q_star's remap.
            if tgt_skel in contact_groups:
                tgt_group_names = sorted(contact_groups[tgt_skel].keys())
                group_joint_ids_target = [
                    list(contact_groups[tgt_skel][n_]) for n_ in tgt_group_names
                ]
            else:
                group_joint_ids_target = []

            body_scale = float(q_star.get('body_scale', 1.0))

            # Diffuse from noise with guidance.
            t_diff_0 = time.time()
            out_motion = ddim_sample_from_noise_with_qguidance(
                model, diffusion, y,
                q_star, group_joint_ids_target,
                n_joints, mean_t, std_t,
                n_model_frames, T_valid,
                max_joints, feature_len,
                body_scale=body_scale,
                n_steps=n_steps, lambda_guide=lambda_guide,
                device=device, seed=pid,
            )
            rec['diffusion_runtime'] = float(time.time() - t_diff_0)

            # Save motion.
            out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_path, out_motion)
            rec['output_path'] = str(out_path)
            rec['status'] = 'ok'

            # Metrics: Q errors + contact F1 + skating.
            try:
                q_rec = reconstruct_q_from_motion13(out_motion, tgt_skel, cond,
                                                    contact_groups)
                q_rec['body_scale'] = q_star.get('body_scale', 1.0)
                errs = q_component_errors(q_rec, q_star)
                rec['q_errors'] = errs
                rec['contact_f1'] = contact_f1(q_rec['contact_sched'],
                                               q_star['contact_sched'])
            except Exception as e:
                rec['q_errors'] = {}
                rec['contact_f1'] = float('nan')
                rec['q_error_exc'] = str(e)
            rec['skating_proxy'] = skating_proxy(out_motion, contact_groups,
                                                 tgt_skel, cond)
            rec['pair_runtime'] = float(time.time() - t0)
            if 'q_errors' in rec and rec['q_errors']:
                print(f'  ok: com_rel_l2={rec["q_errors"].get("com_path_rel_l2", 0):.3f}  '
                      f'contact_F1={rec["contact_f1"]:.3f}  '
                      f'skate={rec["skating_proxy"]:.4f}  '
                      f'pair_runtime={rec["pair_runtime"]:.1f}s')
            else:
                print(f'  ok (metrics partial): pair_runtime={rec["pair_runtime"]:.1f}s')
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f'  FAILED: {e}')
        per_pair.append(rec)

    total_time = time.time() - t_total_0

    # Stratified means.
    strata_order = ['near_present', 'absent', 'moderate', 'extreme']
    strata = {s: [] for s in strata_order}
    for r in per_pair:
        strata[r['stratum']].append(r)

    def _stratum_stats(entries):
        ok = [e for e in entries if e['status'] == 'ok' and e.get('q_errors')]
        if not ok:
            return {'n_total': len(entries), 'n_ok': 0}

        def qs(k):
            vals = [e['q_errors'].get(k, float('nan')) for e in ok]
            vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
            return float(np.mean(vals)) if vals else float('nan')

        return {
            'n_total': len(entries),
            'n_ok': len(ok),
            'com_path_rel_l2': qs('com_path_rel_l2'),
            'heading_vel_rel_l2': qs('heading_vel_rel_l2'),
            'contact_sched_mae': qs('contact_sched_mae'),
            'cadence_abs': qs('cadence_abs'),
            'limb_usage_rel_l2': qs('limb_usage_rel_l2'),
            'contact_f1': float(np.mean([e['contact_f1'] for e in ok
                                         if np.isfinite(e['contact_f1'])])) if ok else float('nan'),
            'skating_proxy': float(np.mean([e['skating_proxy'] for e in ok])),
            'diffusion_runtime': float(np.mean([e['diffusion_runtime'] for e in ok])),
            'pair_runtime': float(np.mean([e['pair_runtime'] for e in ok])),
        }

    strata_stats = {s: _stratum_stats(strata[s]) for s in strata_order}

    out = {
        'ablation': 'K_no_stage2',
        'description': 'Stage 1 + Stage 3 only; Stage 2 IK skipped; '
                       'Q* injected as guidance on contact+COM during DDIM '
                       f'sampling from noise (lambda={lambda_guide}, n_steps={n_steps}).',
        'ckpt': ckpt_path,
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok': sum(1 for r in per_pair if r['status'] == 'ok'),
        'n_failed': sum(1 for r in per_pair if r['status'] == 'failed'),
        'strata_stats': strata_stats,
        'per_pair': per_pair,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n=== DONE: total {total_time:.1f}s, n_ok={out["n_ok"]}/{out["n_pairs"]}, '
          f'failed={out["n_failed"]} ===')
    print(f'metrics saved to {OUT_DIR / "metrics.json"}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_pairs', type=int, default=None,
                    help='override: run only first N pairs (smoke test)')
    ap.add_argument('--n_steps', type=int, default=50)
    ap.add_argument('--lambda_guide', type=float, default=0.2)
    args = ap.parse_args()
    run(n_pairs_override=args.n_pairs, n_steps=args.n_steps,
        lambda_guide=args.lambda_guide)
