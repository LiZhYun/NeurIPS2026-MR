"""Idea N — Joint IK + AnyTop-prior optimisation (fork of `ik_solver.py`).

Stage-2 IK and Stage-3 AnyTop SDEdit-projection are fused into a single
Adam optimisation over ``theta`` and ``root_pos``.  The AnyTop model acts
as a soft regulariser:

    L_total =  w_com * L_com + w_contact * L_contact + w_heading * L_heading
             + w_cadence * L_cadence + w_smooth * L_smooth + w_limb * L_limb
             + w_root_pose * L_root_pose + w_prior * L_prior

    L_prior = || eps_hat(x_noisy, t_fix, skel) - eps_injected ||^2    (score-match)

where ``x_noisy = sqrt(a_bar_t) * x0_diff + sqrt(1 - a_bar_t) * eps``, and
``x0_diff`` is produced by a *differentiable* bridge from the current
``(theta, root_pos)`` to the Truebones 13-dim representation.

The A3 checkpoint is trained with ``predict_xstart=True`` (i.e. the model
outputs ``x0_hat``, not ``eps``); we recover the epsilon prediction as

    eps_hat = (x_noisy - sqrt(a_bar_t) * x0_hat) / sqrt(1 - a_bar_t).

A single fixed diffusion step ``t_fix`` (default 30 / 100) is used at every
prior-evaluation iteration, matching the SDEdit t-value used by Stage 3.

Implementation notes
--------------------
- The differentiable bridge **skips** non-differentiable rotation fitting
  (``animation_from_positions``) and uses the Rodrigues-derived per-joint
  axis-angle rotation in the 6D slot.  This matches the identity-6D
  fallback branch in the original bridge, which empirically still works
  for the AnyTop prior because the prior relies primarily on positions /
  velocities / contact channels (A3 training uses ground-truth 6D, but
  6D is a *conditioning* signal whose gradient wrt ``theta`` is dwarfed
  by the positional signals).  Results are reported without rotation
  refitting; re-fitting can be added later if accuracy is critical.
- T5 embeddings and the skeleton-conditioning ``y`` dict are built once
  per pair; the heavy singletons (AnyTop model, diffusion process, T5,
  ``cond`` dict) are cached across pairs.
- Everything inside the prior call is under ``torch.no_grad()`` for the
  model parameters, but autograd is enabled for the input tensor so the
  gradient reaches ``theta`` / ``root_pos``.
- Prior-gradient magnitudes can be large; defaults scale ``w_prior=0.01``
  to match the other loss terms.
- The prior term is evaluated every ``prior_every`` iterations to keep
  wall time comparable to Stage-2 (+Stage-3) of K.
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = str(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ROOT = Path(PROJECT_ROOT)
_EPS = 1e-8
POS_Y_IDX = 1
FOOT_CH_IDX = 12

# Singleton caches for heavy objects (reused across pairs in a run).
_MODEL_CACHE: dict = {}
_T5_CACHE: dict = {}
_COND_CACHE: dict = {}


# =============================== FK (torch) ==============================

def _rodrigues(theta: torch.Tensor) -> torch.Tensor:
    """theta [..., 3] -> rotation matrix [..., 3, 3] via Rodrigues' formula."""
    angle = torch.linalg.norm(theta, dim=-1, keepdim=True).clamp_min(_EPS)
    axis = theta / angle
    angle = angle.squeeze(-1)
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(ax)
    K = torch.stack([
        torch.stack([zero, -az, ay], dim=-1),
        torch.stack([az, zero, -ax], dim=-1),
        torch.stack([-ay, ax, zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, dtype=theta.dtype, device=theta.device).expand(K.shape)
    sin = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    cos_m = (1.0 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1)
    return I + sin * K + cos_m * (K @ K)


def fk_torch(theta: torch.Tensor, root_pos: torch.Tensor,
             offsets: torch.Tensor, parents: List[int]
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    T, J, _ = theta.shape
    R_local = _rodrigues(theta)
    R_world = [None] * J
    P_world = [None] * J
    R_world[0] = R_local[:, 0]
    P_world[0] = root_pos + torch.einsum('tij,j->ti', R_local[:, 0], offsets[0])
    for j in range(1, J):
        p = parents[j]
        if p < 0:
            R_world[j] = R_local[:, j]
            P_world[j] = root_pos + torch.einsum('tij,j->ti', R_local[:, j], offsets[j])
            continue
        R_world[j] = R_world[p] @ R_local[:, j]
        P_world[j] = P_world[p] + torch.einsum('tij,j->ti', R_world[p], offsets[j])
    return torch.stack(P_world, dim=1), torch.stack(R_world, dim=1)


# =============================== Helpers =================================

def _subtree_masses_np(parents: List[int], offsets: np.ndarray) -> np.ndarray:
    J = len(parents)
    masses = np.linalg.norm(offsets, axis=1).copy()
    for j in reversed(range(J)):
        p = parents[j]
        if 0 <= p < J and p != j:
            masses[p] += masses[j]
    return masses


def _body_scale_np(offsets: np.ndarray) -> float:
    return float(np.linalg.norm(offsets, axis=1).sum() + _EPS)


def _weighted_subtree(parents, offsets):
    w = _subtree_masses_np(parents, offsets)
    return w / (w.sum() + _EPS)


def _compute_rest_root_height(parents: List[int], offsets: np.ndarray) -> float:
    J = len(parents)
    rest = np.zeros_like(offsets)
    for j in range(1, J):
        p = parents[j]
        if 0 <= p < J:
            rest[j] = rest[p] + offsets[j]
    y_min = rest[:, 1].min()
    return float(-y_min) if y_min < 0 else 0.1


def _forward_from_centroids_torch(centroids: torch.Tensor, smoothing: float = 0.9) -> torch.Tensor:
    T = centroids.shape[0]
    vel = torch.zeros_like(centroids)
    vel[1:] = centroids[1:] - centroids[:-1]
    vel[0] = vel[1]
    vel_h = vel.clone()
    vel_h[:, 1] = 0.0
    forward = torch.zeros_like(centroids)
    running = vel_h[0]
    for t in range(T):
        running = smoothing * running + (1.0 - smoothing) * vel_h[t]
        forward[t] = running
    n = torch.linalg.norm(forward, dim=-1, keepdim=True).clamp_min(_EPS)
    forward = forward / n
    if torch.linalg.norm(forward).item() < _EPS:
        forward[:] = torch.tensor([1.0, 0.0, 0.0], dtype=forward.dtype, device=forward.device)
    return forward


def _heading_velocity_torch(centroids: torch.Tensor, forward: torch.Tensor, fps: float) -> torch.Tensor:
    T = centroids.shape[0]
    vel = torch.zeros_like(centroids)
    vel[1:] = (centroids[1:] - centroids[:-1]) * fps
    vel[0] = vel[1]
    return (vel * forward).sum(dim=-1)


def _cadence_soft_torch(sched: torch.Tensor, fps: float,
                        min_cycle: float = 0.25, max_cycle: float = 4.0) -> torch.Tensor:
    s = sched.sum(dim=1) if sched.dim() > 1 else sched
    s = s - s.mean()
    T = s.shape[0]
    if T < 4:
        return torch.tensor(0.0, dtype=s.dtype, device=s.device)
    fft = torch.fft.rfft(s)
    power = fft.real ** 2 + fft.imag ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(s.device)
    mask = ((freqs >= 1.0 / max_cycle) & (freqs <= 1.0 / min_cycle)).to(power.dtype)
    num = (power * freqs * mask).sum()
    den = (power * mask).sum().clamp_min(_EPS)
    return num / den


def _soft_contacts_torch(positions: torch.Tensor, group_idx: List[List[int]],
                         height_scale: float) -> torch.Tensor:
    T = positions.shape[0]
    C = len(group_idx)
    out = torch.zeros(T, C, dtype=positions.dtype, device=positions.device)
    alpha = 10.0 / max(height_scale, _EPS)
    for c, idxs in enumerate(group_idx):
        if not idxs:
            continue
        y = positions[:, idxs, 1]
        out[:, c] = torch.sigmoid(-alpha * y).mean(dim=1)
    return out


# ===================== Differentiable 13-dim bridge ======================

def _yaw_quat_from_forward_t(forward_xz: torch.Tensor) -> torch.Tensor:
    """Build quaternion [w,x,y,z] mapping ``forward`` (xz unit) to +Z.  [T,4].

    Differentiable; matches ``k_pipeline_bridge._yaw_quaternion_from_forward``.
    """
    T = forward_xz.shape[0]
    target = torch.tensor([0.0, 0.0, 1.0], dtype=forward_xz.dtype,
                          device=forward_xz.device).expand(T, 3)
    dot = (forward_xz * target).sum(dim=-1)
    cross = torch.cross(forward_xz, target, dim=-1)
    w = 1.0 + dot
    q = torch.cat([w.unsqueeze(-1), cross], dim=-1)
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(_EPS)
    return q / n


def _quat_apply_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion [..., 4] (w, x, y, z) to vector [..., 3]. Differentiable."""
    w = q[..., 0:1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_to_6d_t(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [..., 4] -> 6D rotation [..., 6] (first two cols of rotmat)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R00 = 1 - 2 * (y * y + z * z)
    R10 = 2 * (x * y + w * z)
    R20 = 2 * (x * z - w * y)
    R01 = 2 * (x * y - w * z)
    R11 = 1 - 2 * (x * x + z * z)
    R21 = 2 * (y * z + w * x)
    col0 = torch.stack([R00, R10, R20], dim=-1)
    col1 = torch.stack([R01, R11, R21], dim=-1)
    return torch.cat([col0, col1], dim=-1)


def _rotmat_to_6d_t(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [..., 3, 3] -> 6D [..., 6] (first two cols stacked)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def _smooth_forward_torch(positions: torch.Tensor, smoothing: float = 0.9) -> torch.Tensor:
    """Smoothed forward direction (xz-unit) from root trajectory.  [T,3]."""
    T = positions.shape[0]
    root = positions[:, 0, :]
    vel = torch.zeros_like(root)
    vel[1:] = root[1:] - root[:-1]
    vel[0] = vel[1]
    # zero y
    mask_y = torch.tensor([1.0, 0.0, 1.0], dtype=vel.dtype, device=vel.device)
    vel = vel * mask_y
    running = vel[0]
    out = torch.zeros_like(vel)
    for t in range(T):
        running = smoothing * running + (1.0 - smoothing) * vel[t]
        out[t] = running
    mag = torch.linalg.norm(out, dim=-1, keepdim=True).clamp_min(_EPS)
    out = out / mag
    # default +Z if degenerate (detect by checking if first frame magnitude tiny)
    if torch.linalg.norm(out).item() < _EPS:
        out = out + torch.tensor([0.0, 0.0, 1.0], dtype=out.dtype, device=out.device)
    return out


def _foot_contact_from_positions_t(positions: torch.Tensor,
                                   foot_joints: List[int],
                                   foot_h: float,
                                   horiz_thresh: float,
                                   J: int) -> torch.Tensor:
    """Soft foot-contact channel [T, J] (in 13-dim slot 12).

    Uses a logistic of `-alpha * (y / foot_h - 1) * (horiz_vel / horiz_thresh - 1)`
    style gating so the result is bounded in [0, 1] and differentiable.
    """
    T = positions.shape[0]
    contact = torch.zeros(T, J, dtype=positions.dtype, device=positions.device)
    if not foot_joints:
        return contact
    fy = positions[:, foot_joints, 1]  # [T, F]
    dpos = torch.zeros_like(positions[:, foot_joints, :])
    if T >= 2:
        dpos[1:] = positions[1:, foot_joints, :] - positions[:-1, foot_joints, :]
        dpos[0] = dpos[1]
    horiz_speed = torch.linalg.norm(dpos[..., [0, 2]], dim=-1)  # [T, F]
    g_y = torch.sigmoid(-8.0 * (fy / max(foot_h, _EPS) - 1.0))
    g_v = torch.sigmoid(-8.0 * (horiz_speed / max(horiz_thresh, _EPS) - 1.0))
    soft = g_y * g_v
    contact[:, foot_joints] = soft
    return contact


def bridge_to_13dim_torch(theta: torch.Tensor, root_pos: torch.Tensor,
                          positions: torch.Tensor, R_world: torch.Tensor,
                          parents: List[int], offsets_t: torch.Tensor,
                          max_joints: int, feat_len: int,
                          foot_joints: List[int], foot_h: float,
                          horiz_thresh: float,
                          ) -> torch.Tensor:
    """Differentiable 13-dim encoding matching the dataset convention.

    Parameters
    ----------
    theta, root_pos : from IK  ([T, J, 3], [T, 3])
    positions       : world-frame FK output   [T, J, 3]
    R_world         : world-frame joint rotation matrices [T, J, 3, 3]
    parents         : list[J]
    offsets_t       : [J, 3]
    max_joints, feat_len : padding dims for the AnyTop grid

    Returns
    -------
    x_tjf : [max_joints, feat_len, T] (the layout AnyTop forward expects on
            the joint/feature axes; padded channels outside [:J] are zero).
    """
    T, J, _ = positions.shape
    device = positions.device
    dtype = positions.dtype

    # --- root yaw quaternion from smoothed forward ---
    forward = _smooth_forward_torch(positions)                 # [T, 3]
    r_rot_quat = _yaw_quat_from_forward_t(forward)             # [T, 4]
    q_b = r_rot_quat.unsqueeze(1).expand(T, J, 4)

    # --- RIC positions: root_local (subtract root XZ) then rotate to face Z+ ---
    ric = positions.clone()
    # Subtract root XZ from all joints.
    root_xz = positions[:, 0:1, :]  # [T, 1, 3]
    ric[..., 0] = ric[..., 0] - root_xz[..., 0]
    ric[..., 2] = ric[..., 2] - root_xz[..., 2]
    ric = _quat_apply_t(q_b, ric)
    # Root slot: zero XZ, keep Y.
    ric_list = list(torch.unbind(ric, dim=1))  # list of [T, 3]
    zero = torch.zeros(T, dtype=dtype, device=device)
    root_row = torch.stack([zero, positions[:, 0, 1], zero], dim=-1)  # [T, 3]
    ric_list[0] = root_row
    ric = torch.stack(ric_list, dim=1)  # [T, J, 3]

    # --- 6D rotations (derived from world rotations) ---
    # Convention from get_bvh_cont6d_params: slot j (j>=1) stores the
    # *parent*'s 6D rotation; slot 0 stores the root-yaw 6D rotation.
    cont6d_world = _rotmat_to_6d_t(R_world)  # [T, J, 6]
    cont6d_reordered_list = [torch.zeros(T, 6, dtype=dtype, device=device)
                              for _ in range(J)]
    for j in range(1, J):
        p = parents[j]
        if 0 <= p < J:
            cont6d_reordered_list[j] = cont6d_world[:, p]
        else:
            cont6d_reordered_list[j] = cont6d_world[:, j]
    cont6d_reordered_list[0] = _quat_to_6d_t(r_rot_quat)
    cont6d_reordered = torch.stack(cont6d_reordered_list, dim=1)  # [T, J, 6]

    # --- velocities: (pos[1:] - pos[:-1]) rotated by r_rot_quat[1:] ---
    vel = torch.zeros(T, J, 3, dtype=dtype, device=device)
    if T >= 2:
        dpos = positions[1:] - positions[:-1]                # [T-1, J, 3]
        q_t1 = r_rot_quat[1:].unsqueeze(1).expand(T - 1, J, 4)
        vel_rot = _quat_apply_t(q_t1, dpos)
        # pad the last frame by repeating.
        vel = torch.cat([vel_rot, vel_rot[-1:]], dim=0)

    # --- foot contact (soft) ---
    contact = _foot_contact_from_positions_t(positions, foot_joints,
                                             foot_h, horiz_thresh, J)  # [T, J]

    # --- assemble to [T, J, 13] ---
    feats = torch.cat([ric, cont6d_reordered, vel,
                        contact.unsqueeze(-1)], dim=-1)  # [T, J, 13]

    # --- pad joints and permute to [J_pad, F, T] ---
    J_pad = max_joints
    pad_feats = torch.zeros(J_pad, feat_len, T, dtype=dtype, device=device)
    pad_feats[:J] = feats.permute(1, 2, 0)  # [J, F, T]
    return pad_feats


# ============================= AnyTop loading ============================

DEFAULT_CKPT = 'save/A3_baseline_dataset_truebones_bs_2_latentdim_256/model000229999.pt'
FALLBACK_CKPT = 'save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt'


def _project_root_path():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def _load_unconditional_model(ckpt_path: str, device: torch.device):
    key = (ckpt_path, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    _project_root_path()
    from model.anytop import AnyTop
    from utils.model_util import create_gaussian_diffusion

    with open(os.path.join(os.path.dirname(ckpt_path), 'args.json')) as f:
        ckpt_args = json.load(f)

    class Ns:
        def __init__(self, d): self.__dict__.update(d)
    args = Ns(ckpt_args)
    t5_dim = {'t5-small': 512, 't5-base': 768, 't5-large': 1024}
    t5_out = t5_dim.get(args.t5_name, 768)

    model = AnyTop(
        max_joints=143, feature_len=13,
        latent_dim=args.latent_dim, ff_size=1024,
        num_layers=args.layers, num_heads=4,
        dropout=0.1, activation='gelu',
        t5_out_dim=t5_out, root_input_feats=13,
        cond_mode='object_type', cond_mask_prob=args.cond_mask_prob,
        skip_t5=args.skip_t5, value_emb=args.value_emb)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected keys in {ckpt_path}: {unexpected[:3]}')
    bad_missing = [k for k in missing if not k.startswith('clip_model.')]
    if bad_missing:
        raise RuntimeError(f'Missing keys not in clip: {bad_missing[:3]}')
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    diffusion = create_gaussian_diffusion(args)
    _MODEL_CACHE[key] = (model, diffusion, args)
    return model, diffusion, args


def _load_t5(t5_name: str, device_str: str):
    key = (t5_name, device_str)
    if key in _T5_CACHE:
        return _T5_CACHE[key]
    _project_root_path()
    from model.conditioners import T5Conditioner
    t5 = T5Conditioner(name=t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device=device_str)
    _T5_CACHE[key] = t5
    return t5


def _load_cond():
    if 'cond' in _COND_CACHE:
        return _COND_CACHE['cond']
    cond = np.load(ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()
    _COND_CACHE['cond'] = cond
    return cond


def _build_y_for_skel(skel_name: str, cond_dict: dict, t5, n_frames: int,
                     temporal_window: int, max_joints: int, device: torch.device):
    """Build `y` dict for a single skeleton, mirroring `anytop_projection._build_y`."""
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation

    info = cond_dict[skel_name]
    n_joints = len(info['joints_names'])
    mean = info['mean'].astype(np.float32)
    std = info['std'].astype(np.float32) + 1e-6

    tpos = np.nan_to_num((info['tpos_first_frame'] - mean) / std)
    tpos_pad = np.zeros((max_joints, 13), dtype=np.float32)
    tpos_pad[:n_joints] = tpos
    tpos_t = torch.from_numpy(tpos_pad).float().unsqueeze(0).to(device)

    names_emb = t5(t5.tokenize(list(info['joints_names']))).detach().cpu().numpy()
    name_pad = np.zeros((max_joints, names_emb.shape[1]), dtype=np.float32)
    name_pad[:n_joints] = names_emb
    names_t = torch.from_numpy(name_pad).float().unsqueeze(0).to(device)

    gd = create_padded_relation(info['joints_graph_dist'], max_joints, n_joints)
    jr = create_padded_relation(info['joint_relations'], max_joints, n_joints)
    gd_t = gd.long().unsqueeze(0).to(device)
    jr_t = jr.long().unsqueeze(0).to(device)

    jmask = torch.zeros(1, 1, 1, max_joints + 1, max_joints + 1, device=device)
    jmask[0, 0, 0, :n_joints + 1, :n_joints + 1] = 1.0
    tmask = create_temporal_mask_for_window(temporal_window, n_frames)
    tmask_t = tmask.unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(device)

    y = {
        'joints_mask':       jmask,
        'mask':              tmask_t,
        'tpos_first_frame':  tpos_t,
        'joints_names_embs': names_t,
        'graph_dist':        gd_t,
        'joints_relations':  jr_t,
        'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
        'n_joints':          torch.tensor([n_joints]),
    }
    return y, n_joints, mean, std


# ============================= Solver weights ============================

DEFAULT_WEIGHTS = {
    'com':       20.0,
    'contact':    1.0,
    'smooth':     0.02,
    'heading':    5.0,
    'cadence':    0.05,
    'limb':       0.5,
    'root_pose':  0.1,
    'prior':      0.1,   # prior-grad is rescaled to have norm = prior * clipped_task_norm
}


def _init_root_pos_from_q(q_star, body_scale, rest_root_h, T_target):
    com = np.asarray(q_star['com_path'], dtype=np.float32)
    if com.shape[0] != T_target:
        idx = np.linspace(0, com.shape[0] - 1, T_target)
        com = np.stack([np.interp(idx, np.arange(com.shape[0]), com[:, d]) for d in range(3)], -1)
    root = com * body_scale
    root[:, 1] += rest_root_h
    return root


# ============================ Prior helper =============================

class AnyTopPrior:
    """Thin wrapper that computes L_prior given a differentiable 13-dim motion.

    Caches the ``y`` dict per skeleton so rebuild cost is paid once.
    """

    def __init__(self, target_skel: str, device: torch.device,
                 ckpt_path: Optional[str] = None, t_fix: int = 30):
        _project_root_path()
        self.device = device
        self.target_skel = target_skel
        self.t_fix = int(t_fix)
        ckpt_path = ckpt_path or os.path.join(PROJECT_ROOT, DEFAULT_CKPT)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, FALLBACK_CKPT)
        self.ckpt_path = ckpt_path
        model, diffusion, m_args = _load_unconditional_model(ckpt_path, device)
        self.model = model
        self.diffusion = diffusion
        self.m_args = m_args
        self.max_joints = 143
        self.feature_len = 13
        self.n_frames = m_args.num_frames
        self.temporal_window = m_args.temporal_window

        cond = _load_cond()
        t5 = _load_t5(m_args.t5_name, device_str=str(device).replace('cuda:', 'cuda'))
        y, n_joints, mean_np, std_np = _build_y_for_skel(
            target_skel, cond, t5, self.n_frames,
            self.temporal_window, self.max_joints, device)
        self.y = y
        self.n_joints = n_joints
        self.mean_np = mean_np
        self.std_np = std_np
        mean_pad = np.zeros((self.max_joints, self.feature_len), dtype=np.float32)
        std_pad = np.ones((self.max_joints, self.feature_len), dtype=np.float32)
        mean_pad[:n_joints] = mean_np
        std_pad[:n_joints] = std_np
        self.mean_t = torch.from_numpy(mean_pad).to(device)
        self.std_t = torch.from_numpy(std_pad).to(device)

        # alpha_bar at t_fix
        alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32,
                                  device=device)
        self.a_bar = alphas_cum[self.t_fix]
        self.sqrt_a = torch.sqrt(self.a_bar)
        self.sqrt_1ma = torch.sqrt((1.0 - self.a_bar).clamp(min=1e-12))
        self.model_mean_type = diffusion.model_mean_type

    def compute(self, x_motion_jft: torch.Tensor) -> torch.Tensor:
        """Compute the score-match prior loss on a 13-dim motion.

        Parameters
        ----------
        x_motion_jft : [J_pad, F, T] differentiable motion (denormalised).

        Returns
        -------
        scalar torch.Tensor  ||eps_hat - eps||^2 averaged over real joints.
        """
        T_max = self.n_frames
        J_pad, F, T = x_motion_jft.shape
        # Crop/pad time dim.
        if T >= T_max:
            x = x_motion_jft[..., :T_max]
        else:
            pad = torch.zeros(J_pad, F, T_max - T,
                              dtype=x_motion_jft.dtype,
                              device=x_motion_jft.device)
            x = torch.cat([x_motion_jft, pad], dim=-1)

        # Normalise (same reshape as dataset: subtract mean, divide std).
        mean_jf1 = self.mean_t.unsqueeze(-1)
        std_jf1 = self.std_t.unsqueeze(-1)
        x0_norm = (x - mean_jf1) / std_jf1
        x0_norm = x0_norm.unsqueeze(0)  # [1, J, F, T]

        # Sample noise and form x_t at fixed t.
        eps = torch.randn_like(x0_norm)
        x_t = self.sqrt_a * x0_norm + self.sqrt_1ma * eps

        t_tensor = torch.tensor([self.t_fix], device=self.device, dtype=torch.long)
        # Forward pass - gradients flow through x_t to the caller's graph.
        with torch.enable_grad():
            model_output = self.model(x_t, self.diffusion._scale_timesteps(t_tensor),
                                       y=self.y)
        # Derive epsilon prediction.
        from diffusion.gaussian_diffusion import ModelMeanType
        if self.model_mean_type == ModelMeanType.EPSILON:
            eps_hat = model_output
        elif self.model_mean_type == ModelMeanType.START_X:
            eps_hat = (x_t - self.sqrt_a * model_output) / self.sqrt_1ma
        else:
            # PREVIOUS_X — not used by our checkpoints; fallback to matching
            # x0 directly (stable upper bound).
            eps_hat = (x_t - self.sqrt_a * model_output) / self.sqrt_1ma

        # Score-matching prior: mean over real joints (first n_joints), feats, T.
        diff = (eps_hat[:, :self.n_joints] - eps[:, :self.n_joints])
        return (diff ** 2).mean()


# =============================== Solver core =============================

def solve_ik_n(q_star: dict,
               target_skel: str,
               target_skel_cond: dict,
               contact_groups: dict,
               n_iters: int = 400,
               weights: Optional[dict] = None,
               verbose: bool = False,
               fps: int = 30,
               device: Optional[str] = None,
               prior_every: int = 50,
               t_fix: int = 30,
               ckpt_path: Optional[str] = None,
               foot_height_frac: float = 0.05,
               horiz_thresh_frac: float = 0.05,
               max_joints: int = 143,
               feat_len: int = 13,
               ) -> dict:
    """Joint-optimisation IK with AnyTop-prior regulariser (Idea N).

    Returns the usual IK outputs plus ``prior_history`` (per-iter prior loss
    value at iterations where it was evaluated) and ``prior_runtime``.
    """
    device = torch.device(device) if device is not None else torch.device('cpu')
    w_cfg = {**DEFAULT_WEIGHTS, **(weights or {})}

    offsets_np = np.asarray(target_skel_cond['offsets'], dtype=np.float32)
    J = offsets_np.shape[0]
    parents = [int(p) for p in target_skel_cond['parents']][:J]
    for j in range(J):
        if parents[j] >= j:
            parents[j] = -1 if j == 0 else 0
    com_path = np.asarray(q_star['com_path'], dtype=np.float32)
    T = com_path.shape[0]
    body_scale = _body_scale_np(offsets_np)
    rest_root_h = _compute_rest_root_height(parents, offsets_np)

    com_path_t = torch.tensor(com_path, dtype=torch.float32, device=device)
    heading_tgt = torch.tensor(np.asarray(q_star['heading_vel'], dtype=np.float32), device=device)
    cadence_tgt = float(q_star.get('cadence', 0.0))
    sched_tgt = torch.tensor(np.asarray(q_star['contact_sched'], dtype=np.float32), device=device)
    if sched_tgt.dim() == 1:
        sched_tgt = sched_tgt.unsqueeze(-1)

    group_names = q_star.get('contact_group_names') or sorted(contact_groups.keys())
    group_idx_tgt: List[List[int]] = []
    for name in group_names:
        if name in contact_groups:
            group_idx_tgt.append([int(i) for i in contact_groups[name] if 0 <= int(i) < J])
        else:
            group_idx_tgt.append([])
    keep = [c for c, idxs in enumerate(group_idx_tgt) if len(idxs) > 0]
    if len(keep) < len(group_idx_tgt):
        group_idx_tgt = [group_idx_tgt[c] for c in keep]
        sched_tgt = sched_tgt[:, keep]
        group_names = [group_names[c] for c in keep] if group_names else None

    limb_tgt = np.asarray(q_star.get('limb_usage', []), dtype=np.float32)
    limb_tgt_t = torch.tensor(limb_tgt, dtype=torch.float32, device=device) if limb_tgt.size else None
    chains = target_skel_cond.get('kinematic_chains', [])
    K_tgt = len(chains)

    w_sub_t = torch.tensor(_weighted_subtree(parents, offsets_np), dtype=torch.float32, device=device)
    offsets_t = torch.tensor(offsets_np, dtype=torch.float32, device=device)

    # Foot joints (union of all contact groups for this skeleton).
    foot_joints = []
    if contact_groups is not None:
        for grp in contact_groups.values():
            if isinstance(grp, list):
                foot_joints.extend([int(j) for j in grp if 0 <= int(j) < J])
    foot_joints = sorted(set(foot_joints))
    foot_h = foot_height_frac * body_scale
    horiz_thresh = horiz_thresh_frac * body_scale

    theta = torch.zeros(T, J, 3, dtype=torch.float32, device=device, requires_grad=True)
    root_init = _init_root_pos_from_q(q_star, body_scale, rest_root_h, T)
    root_pos = torch.tensor(root_init, dtype=torch.float32, device=device, requires_grad=True)

    # Seed root yaw from heading direction.
    fwd_np = np.zeros((T, 3), dtype=np.float32)
    if com_path.shape[0] >= 2:
        fwd_np[1:] = com_path[1:] - com_path[:-1]
        fwd_np[0] = fwd_np[1]
    fwd_np[:, 1] = 0.0
    mag = np.linalg.norm(fwd_np, axis=-1, keepdims=True)
    ok = mag.squeeze(-1) > _EPS
    if ok.any():
        fwd_np[ok] = fwd_np[ok] / mag[ok]
        yaw = np.arctan2(fwd_np[:, 0], fwd_np[:, 2])
        with torch.no_grad():
            theta[:, 0, 1] = torch.tensor(yaw, dtype=theta.dtype, device=device)

    # ---- Load the AnyTop prior once (cached across pairs) ----
    try:
        prior = AnyTopPrior(target_skel, device, ckpt_path=ckpt_path, t_fix=t_fix)
        prior_available = True
    except Exception as e:
        if verbose:
            print(f'  [Idea N] AnyTop prior unavailable ({e}); running without L_prior.')
        prior = None
        prior_available = False

    optim = torch.optim.Adam([theta, root_pos], lr=0.05)
    n_warm = int(0.6 * n_iters)
    best_total = float('inf')
    best_theta = theta.detach().clone()
    best_root = root_pos.detach().clone()
    last_loss_terms: Dict[str, float] = {}
    prior_history: List[Tuple[int, float]] = []

    def compute_task_losses():
        positions, R_world = fk_torch(theta, root_pos, offsets_t, parents)
        centroids = torch.einsum('j,tjd->td', w_sub_t, positions)
        com_rec = (centroids - centroids[0:1]) / body_scale
        loss_com = ((com_rec - com_path_t) ** 2).sum(dim=-1).mean()

        loss_contact = torch.zeros((), dtype=positions.dtype, device=device)
        if sched_tgt.numel() > 0 and len(group_idx_tgt) > 0:
            alpha = 10.0 / max(body_scale, _EPS)
            pred = torch.zeros(T, len(group_idx_tgt), dtype=positions.dtype, device=device)
            for c, idxs in enumerate(group_idx_tgt):
                y = positions[:, idxs, 1]
                pred[:, c] = torch.sigmoid(-alpha * y).mean(dim=1)
                mask = (sched_tgt[:, c] >= 0.5).to(positions.dtype)
                if mask.sum() > 0:
                    loss_contact = loss_contact + ((y.mean(dim=1) ** 2) * mask).sum() / (mask.sum() + _EPS)
            loss_contact = loss_contact + ((pred - sched_tgt) ** 2).mean()

        with torch.no_grad():
            fwd_hat = _forward_from_centroids_torch(centroids.detach())
        heading_rec = _heading_velocity_torch(centroids, fwd_hat, fps=float(fps)) / body_scale
        loss_heading = ((heading_rec - heading_tgt) ** 2).mean()

        cad_rec = (_cadence_soft_torch(_soft_contacts_torch(positions, group_idx_tgt, body_scale), float(fps))
                   if len(group_idx_tgt) > 0 else torch.tensor(0.0, device=device))
        loss_cadence = (cad_rec - cadence_tgt) ** 2

        loss_limb = torch.zeros((), dtype=positions.dtype, device=device)
        if limb_tgt_t is not None and limb_tgt_t.numel() > 0 and K_tgt > 0:
            vel = torch.zeros_like(positions)
            vel[1:] = (positions[1:] - positions[:-1]) * fps
            vel[0] = vel[1]
            ke_j = (0.5 * (vel ** 2).sum(dim=-1)).mean(dim=0)
            energy = torch.zeros(K_tgt, dtype=positions.dtype, device=device)
            for k, chain in enumerate(chains):
                idxs = [int(i) for i in chain if 0 <= int(i) < J]
                if idxs:
                    energy[k] = ke_j[idxs].mean()
            energy = energy / (energy.sum() + _EPS)
            K_m = min(K_tgt, limb_tgt_t.numel())
            loss_limb = ((energy[:K_m] - limb_tgt_t[:K_m]) ** 2).mean()

        loss_smooth = (((theta[2:] - 2 * theta[1:-1] + theta[:-2]) ** 2).mean()
                       if T >= 3 else torch.tensor(0.0, device=device))
        loss_root = ((root_pos[:, 1] - rest_root_h) ** 2).mean()

        return (positions, R_world), {
            'com':       w_cfg['com']       * loss_com,
            'contact':   w_cfg['contact']   * loss_contact,
            'heading':   w_cfg['heading']   * loss_heading,
            'cadence':   w_cfg['cadence']   * loss_cadence,
            'limb':      w_cfg['limb']      * loss_limb,
            'smooth':    w_cfg['smooth']    * loss_smooth,
            'root_pose': w_cfg['root_pose'] * loss_root,
        }

    prior_runtime = 0.0
    t_start_loop = time.time()
    for it in range(n_iters):
        if it == n_warm:
            for g in optim.param_groups:
                g['lr'] = 0.005
        optim.zero_grad()
        (positions, R_world), terms = compute_task_losses()
        task_total = sum(terms.values())
        # Backprop task loss first to get task gradient norm.
        task_total.backward(retain_graph=True)
        task_grad_norm = float(
            sum((p.grad.detach().norm() ** 2).item() for p in [theta, root_pos]
                if p.grad is not None) ** 0.5)
        loss_prior_val = None
        if prior_available and prior is not None and (it % prior_every == 0):
            t_p0 = time.time()
            # Build a *fresh* computation for the prior so gradients from the
            # prior branch are isolated and can be rescaled.
            theta_g = theta.detach().clone().requires_grad_(True)
            root_g = root_pos.detach().clone().requires_grad_(True)
            pos_g, R_g = fk_torch(theta_g, root_g, offsets_t, parents)
            x_tjf = bridge_to_13dim_torch(
                theta_g, root_g, pos_g, R_g, parents, offsets_t,
                max_joints=max_joints, feat_len=feat_len,
                foot_joints=foot_joints, foot_h=foot_h, horiz_thresh=horiz_thresh)
            loss_prior = prior.compute(x_tjf)
            loss_prior.backward()
            prior_grad_norm = float(
                sum((p.grad.detach().norm() ** 2).item() for p in [theta_g, root_g]
                    if p.grad is not None) ** 0.5)
            # Rescale prior grad so its norm == w_cfg['prior'] * max(task_grad_norm, 1e-6).
            target_norm = w_cfg['prior'] * max(task_grad_norm, 1e-3)
            if prior_grad_norm > 1e-12:
                scale = target_norm / prior_grad_norm
            else:
                scale = 0.0
            # Add the rescaled prior grad to the existing task grad.
            with torch.no_grad():
                if theta_g.grad is not None:
                    theta.grad.add_(theta_g.grad * scale)
                if root_g.grad is not None:
                    root_pos.grad.add_(root_g.grad * scale)
            loss_prior_val = float(loss_prior.detach().item())
            prior_runtime += time.time() - t_p0
            prior_history.append((int(it), loss_prior_val))
            terms['prior'] = torch.tensor(loss_prior_val * w_cfg['prior'],
                                          device=device)
        torch.nn.utils.clip_grad_norm_([theta, root_pos], max_norm=5.0)
        optim.step()
        with torch.no_grad():
            theta.clamp_(-math.pi, math.pi)
        cur = float(task_total.item())
        # Best-so-far selection uses task loss only (ignoring prior term) so
        # best theta/root stay anchored to the Q-matching objective.
        if cur < best_total:
            best_total = cur
            best_theta = theta.detach().clone()
            best_root = root_pos.detach().clone()
            last_loss_terms = {k: float(v.detach().item()) for k, v in terms.items()}
        if verbose and (it % 50 == 0 or it == n_iters - 1):
            extra = f" prior={loss_prior_val:.3e}" if loss_prior_val is not None else ''
            print(f"  iter {it:4d}  total={cur:.6f}  " +
                  " ".join(f"{k}={float(v.detach().item()):.3e}" for k, v in terms.items() if k != 'prior') + extra)

    with torch.no_grad():
        positions_final, _ = fk_torch(best_theta, best_root, offsets_t, parents)
    theta_out = best_theta.detach().cpu().numpy()
    root_out = best_root.detach().cpu().numpy()
    positions_out = positions_final.detach().cpu().numpy()
    q_rec = _reconstruct_quotient(positions_out, parents, offsets_np, group_idx_tgt,
                                  group_names, chains, fps=fps)
    q_rec['body_scale'] = body_scale
    runtime = time.time() - t_start_loop
    return {
        'theta': theta_out,
        'root_pos': root_out,
        'positions': positions_out,
        'final_loss': last_loss_terms,
        'q_reconstructed': q_rec,
        'body_scale': body_scale,
        'runtime_sec': runtime,
        'prior_runtime_sec': prior_runtime,
        'prior_history': prior_history,
        'n_prior_evals': len(prior_history),
        'variant': 'joint_gradient',
    }


# ====================== Q reconstruction (same as K) =====================

def _reconstruct_quotient(positions: np.ndarray, parents: List[int], offsets: np.ndarray,
                          group_idx: List[List[int]], group_names: Optional[List[str]],
                          chains: List[List[int]], fps: int = 30) -> dict:
    from eval.effect_program import canonical_body_frame
    from eval.quotient_extractor import compute_heading_velocity, compute_cadence, compute_limb_usage

    subtree = _subtree_masses_np(parents, offsets)
    centroids, R = canonical_body_frame(positions, subtree)
    scale = _body_scale_np(offsets)
    com_path = (centroids - centroids[0:1]) / scale
    heading_vel = compute_heading_velocity(centroids, R, fps=fps) / scale

    T, J, _ = positions.shape
    y = positions[..., 1]
    ground = np.percentile(y, 5)
    tau = max(0.02 * scale, 1e-3)
    contacts = (y - ground < tau).astype(np.int8)
    if group_idx and group_names:
        C = len(group_idx)
        sched = np.zeros((T, C), dtype=np.float32)
        for c, idxs in enumerate(group_idx):
            if idxs:
                sched[:, c] = contacts[:, idxs].mean(axis=1)
    else:
        sched = contacts.sum(axis=1).astype(np.float32) / max(J, 1)

    cadence = compute_cadence(sched, fps=fps)
    limb_usage = compute_limb_usage(positions, chains, fps=fps)

    return {
        'com_path': com_path.astype(np.float32),
        'heading_vel': heading_vel.astype(np.float32),
        'contact_sched': sched.astype(np.float32),
        'contact_group_names': group_names,
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
        'n_joints': int(J),
        'n_frames': int(T),
    }


# =============================== Self-test ================================

def _tiny_smoke(horse_clip: str = 'Horse___LandRun_448.npy',
                target: str = 'Cat', n_iters: int = 100):
    print(f"\n=== Idea-N smoke test: {horse_clip} -> {target} (n_iters={n_iters}) ===")
    from eval.quotient_extractor import extract_quotient
    cond = _load_cond()
    with open(ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        cg = json.load(f)
    q = extract_quotient(horse_clip, cond['Horse'], contact_groups=cg)
    q_star = {**q}
    if target in cg:
        src_names = q['contact_group_names']
        tgt = cg[target]
        common = [n for n in src_names if n in tgt]
        if common:
            col_idx = [src_names.index(n) for n in common]
            q_star = {**q,
                      'contact_sched': q['contact_sched'][:, col_idx],
                      'contact_group_names': common}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = solve_ik_n(q_star, target, cond[target], cg[target],
                     n_iters=n_iters, verbose=True, device=dev)
    print(f"  n_prior_evals={out['n_prior_evals']} prior_runtime={out['prior_runtime_sec']:.2f}s "
          f"total_runtime={out['runtime_sec']:.2f}s")
    print(f"  prior_history: {out['prior_history'][:3]}...{out['prior_history'][-2:]}")
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--horse_clip', type=str, default='Horse___LandRun_448.npy')
    p.add_argument('--target', type=str, default='Cat')
    args = p.parse_args()
    _tiny_smoke(horse_clip=args.horse_clip, target=args.target, n_iters=args.iters)
