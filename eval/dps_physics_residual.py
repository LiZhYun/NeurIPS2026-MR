"""Idea P7 — Diffusion Posterior Sampling with Newton-Euler physics-residual
measurement.

This is a training-free, physics-grounded replacement for DPS's classical ψ
measurement. The measurement operator returns per-joint physics residuals
(torque + velocity-smoothness + floor-penetration) which are *frame-invariant*
under skeleton identity — a physically plausible motion has small residuals on
ANY morphology, so the source/target measurements are directly comparable even
when joint counts differ.

Approximations
--------------
1. **Inertia model.** Per-joint mass m_j ∝ bone_length_j * body_scale_ratio²
   (uniform cylinder of radius 0.1 × body_scale). Scalar principal-axis inertia
   I_j = m_j * bone_length_j² / 12 (thin rod).
2. **Kinematics.** We operate directly on the AnyTop 13-dim feature tensor in
   root-local coordinates. For non-root joints:
       pos_j(t) = x_denorm[t, j, 0:3]  (root-relative position)
       v_j(t)   = x_denorm[t, j, 9:12] (velocity channel as stored by the data)
       a_j(t)   = finite-difference of v_j
   Angular velocity ω_j(t) ≈ Δ/Δt of log(R_j⁻¹(t) R_j(t+1)) derived from the
   rot6d channels (x_denorm[t, j, 3:9]). (Computed in each joint's local frame;
   sufficient for residual magnitude.)
3. **Torque residual (Newton-Euler).**
       τ_j(t) = I_j · dω_j/dt + ω_j × I_j ω_j + r_j × (m_j·(a_j − g))
   where r_j is the vector from joint to its parent (≈ bone vector = −offset_j),
   and g = (0, −9.81, 0) in root-local frame (approximation: treat root-local
   as inertial; this is the same linearisation used by prior physics-residual
   work such as PhysDiff).
4. **Measurement M(x).** Per depth-bin (n_bins = 8):
       [time-mean of |τ_j|, time-std of |τ_j|,
        time-mean of |Δv_j|, time-mean of relu(−y_j_world_proxy)]
   aggregated across joints in each depth-bin. This yields a 32-d vector that
   is comparable source↔target regardless of joint count.

DPS loop
--------
Following Chung et al. 2023. At each reverse DDIM step we compute
  ∇_{x_t} ‖M(x̂_0(x_t)) − M(x_src)‖² and add −λ∇ to the mean. Source
measurement computed once at the start on the denormalised source features.

Outputs
-------
Motion files saved as `eval/results/k_compare/physics_dps/pair_<id>_<src>_to_<tgt>.npy`
with shape [T, J, 13] in denormalised target space (same convention as DPS A).
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = str(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from os.path import join as pjoin
from typing import Optional, List, Dict

from eval.anytop_projection import (
    _load_unconditional_model, _load_t5, _build_y,
    POS_Y_IDX, FOOT_CH_IDX, ROT_START, ROT_END,
)
from utils.rotation_conversions import rotation_6d_to_matrix

GRAVITY = torch.tensor([0.0, -9.81, 0.0])


# ---------------------------- helpers ------------------------------------

def _depth_bins_from_parents(parents: np.ndarray, n_joints: int, n_bins: int = 8) -> np.ndarray:
    """Map each joint j to a depth bin in [0, n_bins)."""
    depth = np.zeros(n_joints, dtype=np.int64)
    for j in range(n_joints):
        d = 0
        cur = j
        while 0 <= parents[cur] != cur and d < n_joints:
            d += 1
            cur = int(parents[cur])
        depth[j] = d
    if depth.max() < 1:
        return np.zeros(n_joints, dtype=np.int64)
    depth_norm = depth / depth.max()
    return np.clip((depth_norm * n_bins).astype(np.int64), 0, n_bins - 1)


def _bone_lengths_from_offsets(offsets: np.ndarray) -> np.ndarray:
    """offsets: [J, 3] — local-frame offset of joint j from its parent.
    Returns [J] bone-length per joint. Root's bone length is set to median
    (proxy so it has nonzero inertia)."""
    bl = np.linalg.norm(offsets, axis=-1).astype(np.float32)   # [J]
    med = float(np.median(bl[bl > 0])) if (bl > 0).any() else 1.0
    bl = np.where(bl < 1e-4, med, bl)
    return bl


def _body_scale(offsets: np.ndarray) -> float:
    """Rough body-scale: mean distance of any joint from the root in the T-pose."""
    # We don't have the absolute T-pose positions easily; use sum of bone
    # lengths as a crude proxy for total chain extent.
    return float(max(np.linalg.norm(offsets, axis=-1).sum(), 1e-3))


def _inertia_params(offsets: np.ndarray, n_joints: int, density: float = 1.0,
                    radius_ratio: float = 0.1) -> Dict[str, np.ndarray]:
    """Return per-joint (mass, inertia, bone_length) under uniform-cylinder approx.
    Units are *relative* — we only care about residual magnitude so absolute
    values are unimportant so long as source/target use the same formula."""
    offs = offsets[:n_joints]
    bl = _bone_lengths_from_offsets(offs)                     # [J]
    body_scale = _body_scale(offs)
    radius = radius_ratio * body_scale
    # Uniform cylinder: m = ρ π r² L, I_principal = m * L² / 12
    mass = density * np.pi * (radius ** 2) * bl
    mass = np.maximum(mass, 1e-4).astype(np.float32)
    inertia = (mass * (bl ** 2) / 12.0).astype(np.float32)
    inertia = np.maximum(inertia, 1e-8)
    return {
        'mass': mass,                 # [J]
        'inertia': inertia,           # [J] scalar principal-axis inertia
        'bone_length': bl.astype(np.float32),
        'offsets': offs.astype(np.float32),   # [J, 3] local-frame
        'body_scale': float(body_scale),
    }


# -------------------- kinematics / residual ------------------------------

def _omega_from_rot6d(rot6d_tj: torch.Tensor, dt: float = 1.0 / 20.0) -> torch.Tensor:
    """Estimate per-joint angular velocity from rot6d time-series.

    rot6d_tj: [B, J, 6, T]
    returns: [B, J, 3, T] — ω in local frame (zero padded on the last frame)
    Method: convert rot6d → rotation matrix R_t, compute
        Δ = R_t⁻¹ R_{t+1}, extract ω = vee(log(Δ))/dt using first-order
        approximation ω ≈ vee(Δ − I)/dt (valid for small rotations / high fps).
    The log is computed via the standard skew-symmetric trick, gradient-safe.
    """
    B, J, _, T = rot6d_tj.shape
    # Reshape to [..., 6] and apply helper.
    rot6d_btjt = rot6d_tj.permute(0, 1, 3, 2).contiguous().view(B * J * T, 6)
    R_btjt = rotation_6d_to_matrix(rot6d_btjt)                 # [B*J*T, 3, 3]
    R = R_btjt.view(B, J, T, 3, 3)
    R_t = R[:, :, :-1]                                         # [B, J, T-1, 3, 3]
    R_next = R[:, :, 1:]
    # relative = R_t^T @ R_next
    Rel = torch.matmul(R_t.transpose(-1, -2), R_next)          # [B, J, T-1, 3, 3]
    # Skew part ≈ ω × dt = (Rel − Rel^T) / 2
    skew = 0.5 * (Rel - Rel.transpose(-1, -2))                 # [B, J, T-1, 3, 3]
    omega = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
                        dim=-1)                                 # [B, J, T-1, 3]
    omega = omega / dt
    # Pad last frame with zeros so shape matches T
    pad = torch.zeros(B, J, 1, 3, device=omega.device, dtype=omega.dtype)
    omega = torch.cat([omega, pad], dim=2)                     # [B, J, T, 3]
    return omega.permute(0, 1, 3, 2).contiguous()              # [B, J, 3, T]


def _finite_diff(x: torch.Tensor, dt: float, dim: int = -1) -> torch.Tensor:
    """First-order forward finite difference, zero-pad on the last frame."""
    d = (x.roll(-1, dims=dim) - x) / dt
    # Zero-out the last entry (wrap around artefact)
    if dim == -1 or dim == x.dim() - 1:
        d[..., -1] = 0.0
    else:
        # generic slice: build index
        idx = [slice(None)] * x.dim()
        idx[dim] = -1
        d[tuple(idx)] = 0.0
    return d


def _torque_residual(pos_tj: torch.Tensor, vel_tj: torch.Tensor, rot6d_tj: torch.Tensor,
                     mass: torch.Tensor, inertia: torch.Tensor, bone_vec: torch.Tensor,
                     dt: float = 1.0 / 20.0) -> torch.Tensor:
    """Per-joint torque residual magnitude from Newton-Euler.

    pos_tj  : [B, J, 3, T] (root-relative)
    vel_tj  : [B, J, 3, T]
    rot6d_tj: [B, J, 6, T]
    mass    : [J]
    inertia : [J]
    bone_vec: [J, 3] offset-to-parent in local frame (used as r_j → parent arm)
    dt      : 1/fps (~1/20 for Truebones after downsampling)
    returns : [B, J, T] magnitudes of τ_j(t)
    """
    B, J, _, T = pos_tj.shape
    # Angular velocity
    omega = _omega_from_rot6d(rot6d_tj, dt=dt)                 # [B, J, 3, T]
    d_omega = _finite_diff(omega, dt, dim=-1)
    # Linear acceleration (from velocity channel)
    accel = _finite_diff(vel_tj, dt, dim=-1)                   # [B, J, 3, T]

    # Broadcasts
    m = mass.view(1, J, 1, 1)                                   # [1, J, 1, 1]
    I = inertia.view(1, J, 1, 1)
    g = GRAVITY.to(pos_tj.device).view(1, 1, 3, 1)              # [1, 1, 3, 1]

    # Torque from angular terms: I·dω/dt + ω × Iω  (scalar I so Iω = I*ω)
    Iw = I * omega                                              # [B, J, 3, T]
    # ω × Iω along dim=2 (xyz)
    omega_xyz = omega.permute(0, 1, 3, 2)                       # [B, J, T, 3]
    Iw_xyz = Iw.permute(0, 1, 3, 2)
    cross_wIw = torch.cross(omega_xyz, Iw_xyz, dim=-1).permute(0, 1, 3, 2)  # [B, J, 3, T]
    tau_ang = I * d_omega + cross_wIw                           # [B, J, 3, T]

    # Force component: m · (a − g)
    f_lin = m * (accel - g)                                     # [B, J, 3, T]
    # Moment about joint: r × F, with r = bone_vec (joint→parent offset, local)
    r = bone_vec.view(1, J, 3, 1).expand(B, J, 3, T)            # [B, J, 3, T]
    r_xyz = r.permute(0, 1, 3, 2)
    f_xyz = f_lin.permute(0, 1, 3, 2)
    cross_rF = torch.cross(r_xyz, f_xyz, dim=-1).permute(0, 1, 3, 2)   # [B, J, 3, T]

    tau = tau_ang + cross_rF                                    # [B, J, 3, T]
    tau_mag = torch.sqrt((tau ** 2).sum(dim=2) + 1e-8)          # [B, J, T]
    return tau_mag


def _depth_bin_time_stats(feat_tj: torch.Tensor, depth_bins: torch.Tensor,
                          n_bins: int) -> torch.Tensor:
    """feat_tj: [B, J, T]. Return [B, 2, n_bins] (mean + std across time, avg'd
    within each depth-bin of joints)."""
    B, J, T = feat_tj.shape
    # Per-joint time-mean / time-std
    mu_j = feat_tj.mean(dim=-1)                                  # [B, J]
    sd_j = feat_tj.std(dim=-1)                                   # [B, J]
    out = feat_tj.new_zeros(B, 2, n_bins)
    for b in range(n_bins):
        mask = (depth_bins == b)
        c = int(mask.sum().item())
        if c == 0:
            continue
        out[:, 0, b] = mu_j[:, mask].mean(dim=1)
        out[:, 1, b] = sd_j[:, mask].mean(dim=1)
    return out


def physics_measurement(x_norm: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor,
                        depth_bins: torch.Tensor, n_joints: int, n_bins: int,
                        mass: torch.Tensor, inertia: torch.Tensor,
                        bone_vec: torch.Tensor, dt: float,
                        floor_y_ref: float = 0.0) -> Dict[str, torch.Tensor]:
    """Compute physics-residual measurement on a normalised motion tensor.

    x_norm: [B, J_max, 13, T] normalised features (gradients flow through this).
    Returns dict with
        torque_stats  [B, 2, n_bins]
        vel_smooth    [B, 2, n_bins]
        floor_pen     [B, 2, n_bins]
    """
    B, J_max, C, T = x_norm.shape
    assert C == 13
    mean = mean_t.view(1, J_max, C, 1)
    std = std_t.view(1, J_max, C, 1)
    x_de = x_norm * std + mean                                   # [B, J_max, 13, T]

    x_de = x_de[:, :n_joints, :, :]
    pos_tj = x_de[:, :, :3, :]                                    # root-relative positions
    rot6d_tj = x_de[:, :, 3:9, :]
    vel_tj = x_de[:, :, 9:12, :]
    # Torque residual
    tau_mag = _torque_residual(pos_tj, vel_tj, rot6d_tj,
                                mass=mass, inertia=inertia,
                                bone_vec=bone_vec, dt=dt)         # [B, J, T]
    # Velocity smoothness = |Δv|
    dv = _finite_diff(vel_tj, dt, dim=-1)
    dv_mag = torch.sqrt((dv ** 2).sum(dim=2) + 1e-8)              # [B, J, T]
    # Floor penetration using pos_y channel (root-relative, coarse but
    # source-comparable).
    y = pos_tj[:, :, 1, :]                                        # [B, J, T]
    floor = F.relu(floor_y_ref - y)                               # [B, J, T]
    # Log-compress very large torques for gradient stability
    tau_log = torch.log1p(tau_mag)
    dv_log = torch.log1p(dv_mag)
    return {
        'torque_stats': _depth_bin_time_stats(tau_log, depth_bins, n_bins),
        'vel_smooth': _depth_bin_time_stats(dv_log, depth_bins, n_bins),
        'floor_pen': _depth_bin_time_stats(floor, depth_bins, n_bins),
    }


def measurement_mse(m_pred: Dict[str, torch.Tensor],
                    m_src: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Scale-normalised MSE between pred and source measurements."""
    eps = 1e-6
    loss = 0.0
    for k in ('torque_stats', 'vel_smooth', 'floor_pen'):
        a = m_pred[k]
        b = m_src[k]
        denom = (b.pow(2).mean() + eps)
        loss = loss + ((a - b) ** 2).mean() / denom
    return loss


# ------------------------- source measurement ----------------------------

def _source_physics_measurement(source_motion: np.ndarray,
                                src_mean: np.ndarray, src_std: np.ndarray,
                                src_parents: np.ndarray,
                                src_offsets: np.ndarray, src_n_joints: int,
                                n_bins: int, max_joints: int, n_frames: int,
                                dt: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """Compute measurement on the SOURCE denormalised motion.
    We renormalise it by its own stats so `physics_measurement` can denormalise
    (keeps scaling logic symmetric with the target branch)."""
    T_real = source_motion.shape[0]
    T = min(T_real, n_frames)
    src = source_motion[:T].astype(np.float32)                   # denormalised [T, J, 13]
    norm = np.nan_to_num((src - src_mean[None]) / (src_std[None] + 1e-6))
    pad = np.zeros((max_joints, 13, n_frames), dtype=np.float32)
    pad[:src_n_joints, :, :T] = norm.transpose(1, 2, 0)
    x = torch.from_numpy(pad).unsqueeze(0).to(device)             # [1, J_max, 13, T]

    mean_pad = np.zeros((max_joints, 13), dtype=np.float32)
    std_pad = np.ones((max_joints, 13), dtype=np.float32)
    mean_pad[:src_n_joints] = src_mean
    std_pad[:src_n_joints] = src_std + 1e-6
    mean_t = torch.from_numpy(mean_pad).to(device)
    std_t = torch.from_numpy(std_pad).to(device)

    depth_bins_np = _depth_bins_from_parents(src_parents, src_n_joints, n_bins)
    depth_bins = torch.from_numpy(depth_bins_np).to(device)

    inertia_params = _inertia_params(src_offsets, src_n_joints)
    mass = torch.from_numpy(inertia_params['mass']).to(device)
    inertia = torch.from_numpy(inertia_params['inertia']).to(device)
    # bone vector = −offset (joint → parent, in parent frame). Close approximation.
    bone_vec = torch.from_numpy(-inertia_params['offsets']).to(device)

    with torch.no_grad():
        m = physics_measurement(x, mean_t, std_t, depth_bins,
                                 src_n_joints, n_bins, mass, inertia, bone_vec,
                                 dt=dt)
    return m


# ----------------------------- DPS loop ----------------------------------

def physics_dps_retarget(source_motion: np.ndarray,
                         source_skel: str,
                         source_parents: np.ndarray,
                         source_offsets: np.ndarray,
                         source_mean: np.ndarray,
                         source_std: np.ndarray,
                         target_skel: str,
                         cond_dict: dict,
                         t5,
                         model,
                         diffusion,
                         m_args,
                         n_steps: int = 50,
                         lambda_m: float = 0.1,
                         n_bins: int = 8,
                         seed: int = 42,
                         fps: float = 20.0,
                         device: str = 'cuda') -> Dict:
    """DPS retarget with Newton-Euler physics residual measurement."""
    dev = torch.device(device)
    max_joints = 143
    feature_len = 13
    n_frames = m_args.num_frames
    dt = 1.0 / float(fps)

    # --- Target conditioning ---
    y, n_joints_tgt, mean_np, std_np = _build_y(
        target_skel, cond_dict, t5, n_frames,
        m_args.temporal_window, max_joints, dev)
    mean_pad = np.zeros((max_joints, feature_len), dtype=np.float32)
    std_pad = np.ones((max_joints, feature_len), dtype=np.float32)
    mean_pad[:n_joints_tgt] = mean_np
    std_pad[:n_joints_tgt] = std_np + 1e-6
    mean_t = torch.from_numpy(mean_pad).to(dev)
    std_t = torch.from_numpy(std_pad).to(dev)

    # Target physics parameters
    tgt_info = cond_dict[target_skel]
    tgt_offsets = np.array(tgt_info['offsets'][:n_joints_tgt], dtype=np.float32)
    tgt_parents = np.array(tgt_info['parents'][:n_joints_tgt], dtype=np.int64)
    tgt_inertia = _inertia_params(tgt_offsets, n_joints_tgt)
    tgt_mass = torch.from_numpy(tgt_inertia['mass']).to(dev)
    tgt_I = torch.from_numpy(tgt_inertia['inertia']).to(dev)
    tgt_bone_vec = torch.from_numpy(-tgt_inertia['offsets']).to(dev)
    depth_bins_tgt_np = _depth_bins_from_parents(tgt_parents, n_joints_tgt, n_bins)
    depth_bins_tgt = torch.from_numpy(depth_bins_tgt_np).to(dev)

    # --- Source measurement (fixed target for DPS guidance) ---
    m_src = _source_physics_measurement(
        source_motion, source_mean, source_std,
        np.array(source_parents, dtype=np.int64),
        np.array(source_offsets, dtype=np.float32),
        int(source_motion.shape[1]),
        n_bins, max_joints, n_frames, dt=dt, device=dev)
    m_src = {k: v.detach() for k, v in m_src.items()}

    # --- Initial noise ---
    torch.manual_seed(seed)
    x_t = torch.randn(1, max_joints, feature_len, n_frames, device=dev)

    t_max = diffusion.num_timesteps
    steps = list(range(t_max - 1, -1, -1))
    if n_steps < t_max:
        idx = np.linspace(0, t_max - 1, n_steps).round().astype(int)[::-1]
        steps = idx.tolist()
    alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=dev)

    tic = time.time()
    torch.cuda.reset_peak_memory_stats(dev)

    for i, t_val in enumerate(steps):
        t_tensor = torch.tensor([t_val], device=dev, dtype=torch.long)

        x_t.requires_grad_(True)
        model_out = model(x_t, diffusion._scale_timesteps(t_tensor), **{'y': y})
        # A3 predicts x_start (START_X)
        x0_hat = model_out

        m_pred = physics_measurement(x0_hat, mean_t, std_t, depth_bins_tgt,
                                      n_joints_tgt, n_bins,
                                      tgt_mass, tgt_I, tgt_bone_vec, dt=dt)
        loss = measurement_mse(m_pred, m_src)
        grad = torch.autograd.grad(loss, x_t)[0].detach()
        # Chung-style normalisation
        grad = grad / torch.sqrt(loss.detach() + 1e-8)
        g_norm = grad.norm()
        max_norm = 50.0
        if g_norm > max_norm:
            grad = grad * (max_norm / g_norm)
        x_t = x_t.detach()
        x0_hat = x0_hat.detach()

        a_bar_t = alphas_cum[t_val]
        if i == len(steps) - 1:
            x_t = x0_hat - lambda_m * grad
            break
        t_next = steps[i + 1]
        a_bar_next = alphas_cum[t_next] if t_next >= 0 else torch.tensor(
            1.0, device=dev)
        eps = (x_t - torch.sqrt(a_bar_t) * x0_hat) / torch.sqrt(
            (1.0 - a_bar_t).clamp(min=1e-12))
        x_mean = torch.sqrt(a_bar_next) * x0_hat + torch.sqrt(
            (1.0 - a_bar_next).clamp(min=0.0)) * eps
        x_t = x_mean - lambda_m * grad

    runtime = time.time() - tic
    peak = torch.cuda.max_memory_allocated(dev)

    x_final = x_t.detach()
    x_ref = x_final[0, :n_joints_tgt].cpu().numpy()              # [J, 13, T]
    x_ref = x_ref.transpose(2, 0, 1)                             # [T, J, 13]
    x_ref_denorm = x_ref * std_np[None] + mean_np[None]

    return {
        'x_norm': x_ref.astype(np.float32),
        'x_denorm': x_ref_denorm.astype(np.float32),
        'runtime_s': float(runtime),
        'gpu_peak_mb': float(peak / 1024 / 1024),
        'n_joints_tgt': int(n_joints_tgt),
        'target_mean': mean_np.astype(np.float32),
        'target_std': std_np.astype(np.float32),
    }


# ---------------------------- driver ------------------------------------

def _load_eval_pairs(subset_k: Optional[int] = None, stratified: bool = True):
    with open(pjoin(PROJECT_ROOT, 'idea-stage/eval_pairs.json')) as f:
        data = json.load(f)
    pairs = sorted(data['pairs'], key=lambda p: p['pair_id'])
    if subset_k is None:
        return pairs
    if not stratified:
        return pairs[:subset_k]
    # Stratified by (family_gap, support_absent) to match the full-30 spec.
    # The 30-pair manifest is 10 near + 10 absent + 5 moderate + 5 extreme,
    # but 'absent' is cross-cut (absent always has support_same_label==0).
    # We pick n_near, n_absent, n_moderate, n_extreme proportional to k/30.
    stratify_key = lambda p: (
        'absent' if p['support_same_label'] == 0 else
        ('near_present' if p['family_gap'] == 'near' else p['family_gap'])
    )
    buckets = {'near_present': [], 'absent': [], 'moderate': [], 'extreme': []}
    for p in pairs:
        buckets[stratify_key(p)].append(p)
    # Proportions: (10, 10, 5, 5) / 30 scaled to subset_k
    ratio = [('near_present', 10), ('absent', 10), ('moderate', 5), ('extreme', 5)]
    quotas = []
    running = 0
    for name, w in ratio:
        q = int(round(subset_k * w / 30))
        quotas.append((name, q))
        running += q
    # Adjust to hit subset_k exactly
    diff = subset_k - running
    if diff != 0:
        # tweak 'absent' which is most important
        quotas[1] = (quotas[1][0], quotas[1][1] + diff)
    picked = []
    for name, q in quotas:
        picked.extend(buckets[name][:q])
    picked.sort(key=lambda p: p['pair_id'])
    return picked


def _load_contact_groups():
    path = pjoin(PROJECT_ROOT, 'eval/quotient_assets/contact_groups.json')
    with open(path) as f:
        return json.load(f)


def _acc_smoothness(positions: np.ndarray) -> float:
    """|d²pos/dt²| mean across joints and frames."""
    if positions is None or positions.shape[0] < 4:
        return float('nan')
    a = positions[2:] - 2 * positions[1:-1] + positions[:-2]
    return float(np.linalg.norm(a, axis=-1).mean())


def _skating(positions: np.ndarray, contact: np.ndarray) -> float:
    """Mean foot slip: speed of joints while marked contact>0.5."""
    if positions is None or positions.shape[0] < 2 or contact is None:
        return float('nan')
    T = min(positions.shape[0], contact.shape[0])
    pos = positions[:T]
    ct = contact[:T] > 0.5
    if ct.sum() < 2:
        return 0.0
    vel = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)             # [T-1, J]
    ct_mid = ct[1:]
    slip = (vel * ct_mid.astype(np.float32)).sum() / max(ct_mid.sum(), 1)
    return float(slip)


def _contact_f1(pred, gt, th=0.5):
    p = (np.asarray(pred) >= th).astype(np.int8).ravel()
    g = (np.asarray(gt) >= th).astype(np.int8).ravel()
    n = min(p.size, g.size)
    p, g = p[:n], g[:n]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    return float(2 * pr * rc / (pr + rc + 1e-8))


def _grouped_contact(motion13, skel, contact_groups):
    ch = motion13[..., FOOT_CH_IDX]
    contacts = (ch > 0.5).astype(np.float32)
    if skel not in contact_groups:
        return contacts.sum(axis=1)
    groups = contact_groups[skel]
    names = sorted(groups.keys())
    T, J = contacts.shape
    sched = np.zeros((T, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        idxs = [int(j) for j in groups[name] if 0 <= int(j) < J]
        if idxs:
            sched[:, i] = contacts[:, idxs].mean(axis=1)
    return sched.sum(axis=1)


def _classify_with_v2(x13_tgt, skel, cond, clf):
    from eval.external_classifier import extract_classifier_features
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    J = cond[skel]['offsets'].shape[0]
    m = x13_tgt
    if m.shape[1] > J:
        m = m[:, :J]
    try:
        positions = recover_from_bvh_ric_np(m.astype(np.float32))
    except Exception:
        return None, None
    parents = cond[skel]['parents'][:J]
    feats = extract_classifier_features(positions, parents)
    if feats is None or feats.shape[0] < 4:
        return None, positions
    pred = clf.predict_label(feats)
    return pred, positions


def run_pilot(ckpt_path: str, device: str, n_pairs: int, n_steps: int,
              lambda_m: float, fps: float):
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from eval.train_external_classifier_v2 import V2Classifier

    dev = torch.device(device)
    model, diffusion, m_args = _load_unconditional_model(ckpt_path, dev)
    t5 = _load_t5(m_args.t5_name, device_str=device)
    clf = V2Classifier(pjoin(PROJECT_ROOT, 'save/external_classifier_v2.pt'), device=dev)

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    motion_dir = opt.motion_dir
    contact_groups = _load_contact_groups()

    pairs = _load_eval_pairs(subset_k=n_pairs)
    out_dir = pjoin(PROJECT_ROOT, 'eval/results/k_compare/physics_dps')
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = []
    for pair in pairs:
        src_fname = pair['source_fname']
        src_skel = pair['source_skel']
        tgt_skel = pair['target_skel']
        if tgt_skel not in cond_dict or src_skel not in cond_dict:
            print(f"[skip] pair {pair['pair_id']} missing cond", flush=True)
            continue
        src_info = cond_dict[src_skel]
        src_n_j = len(src_info['joints_names'])
        src_parents = np.array(src_info['parents'][:src_n_j], dtype=np.int64)
        src_offsets = np.array(src_info['offsets'][:src_n_j], dtype=np.float32)
        src_mean = src_info['mean']
        src_std = src_info['std']
        try:
            src_raw = np.load(pjoin(motion_dir, src_fname)).astype(np.float32)
        except Exception as e:
            print(f"[skip] can't load {src_fname}: {e}", flush=True)
            continue
        src_denorm = src_raw[:, :src_n_j] * (src_std + 1e-6) + src_mean

        print(f"\n[pair {pair['pair_id']:02d}] {src_skel}({pair['source_label']}) "
              f"→ {tgt_skel} gap={pair['family_gap']} sup={pair['support_same_label']}",
              flush=True)
        try:
            res = physics_dps_retarget(
                source_motion=src_denorm,
                source_skel=src_skel,
                source_parents=src_parents,
                source_offsets=src_offsets,
                source_mean=src_mean,
                source_std=src_std,
                target_skel=tgt_skel,
                cond_dict=cond_dict,
                t5=t5, model=model, diffusion=diffusion, m_args=m_args,
                n_steps=n_steps, lambda_m=lambda_m, fps=fps, device=device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}", flush=True)
            continue

        out_fn = pjoin(out_dir, f"pair_{pair['pair_id']:02d}_{src_skel}_to_{tgt_skel}.npy")
        np.save(out_fn, res['x_denorm'])

        # ---- Metrics ----
        pred_label, positions = _classify_with_v2(res['x_denorm'], tgt_skel, cond_dict, clf)
        # Classify source too for behavior_preserved
        src_pred, _ = _classify_with_v2(src_raw, src_skel, cond_dict, clf)
        behavior_preserved = (pred_label is not None and src_pred is not None and pred_label == src_pred)
        label_match = (pred_label == pair['source_label'])

        # contact f1 vs source
        pred_cf = _grouped_contact(res['x_denorm'], tgt_skel, contact_groups)
        # denormalise source for grouping
        src_cf = _grouped_contact(src_denorm, src_skel, contact_groups)
        if pred_cf.shape[0] > 0 and src_cf.shape[0] > 0:
            idx = np.clip(np.linspace(0, src_cf.shape[0] - 1, pred_cf.shape[0]).astype(int),
                          0, src_cf.shape[0] - 1)
            src_al = src_cf[idx]
            cf1 = _contact_f1(pred_cf, src_al)
        else:
            cf1 = float('nan')

        # skating + accel smoothness
        if positions is not None:
            skating = _skating(positions, res['x_denorm'][..., FOOT_CH_IDX])
            accel = _acc_smoothness(positions)
        else:
            skating = float('nan')
            accel = float('nan')

        m = {
            'pair_id': int(pair['pair_id']),
            'family_gap': pair['family_gap'],
            'support_same_label': int(pair['support_same_label']),
            'source_label': pair['source_label'],
            'source_skel': src_skel,
            'target_skel': tgt_skel,
            'tgt_pred_v2': pred_label,
            'src_pred_v2': src_pred,
            'label_match': bool(label_match),
            'behavior_preserved': bool(behavior_preserved),
            'contact_f1_vs_source': float(cf1),
            'skating': float(skating),
            'accel_smoothness': float(accel),
            'runtime_s': float(res['runtime_s']),
            'gpu_peak_mb': float(res['gpu_peak_mb']),
            'out_path': out_fn,
        }
        all_metrics.append(m)
        print(f"  pred={pred_label}  src_pred={src_pred}  lbl_match={label_match}  "
              f"beh_pres={behavior_preserved}  c_f1={cf1:.3f}  "
              f"skate={skating:.3f}  accel={accel:.3f}  "
              f"t={m['runtime_s']:.1f}s  mem={m['gpu_peak_mb']:.0f}MB",
              flush=True)

    # ---- Aggregates ----
    def _m(xs):
        xs = [x for x in xs if isinstance(x, (int, float)) and not np.isnan(x)]
        return float(np.mean(xs)) if xs else float('nan')

    from collections import Counter
    strat_key = lambda r: ('absent' if r['support_same_label'] == 0
                           else ('near_present' if r['family_gap'] == 'near' else r['family_gap']))
    strata = {'near_present': [], 'absent': [], 'moderate': [], 'extreme': []}
    for r in all_metrics:
        strata[strat_key(r)].append(r)

    by_stratum = {}
    for k, grp in strata.items():
        if not grp:
            continue
        by_stratum[k] = {
            'n': len(grp),
            'label_match_rate': _m([1.0 if r['label_match'] else 0.0 for r in grp]),
            'behavior_preserved_rate': _m([1.0 if r['behavior_preserved'] else 0.0 for r in grp]),
            'contact_f1_mean': _m([r['contact_f1_vs_source'] for r in grp]),
            'skating_mean': _m([r['skating'] for r in grp]),
            'accel_smoothness_mean': _m([r['accel_smoothness'] for r in grp]),
            'pred_class_dist': dict(Counter(r['tgt_pred_v2'] for r in grp)),
        }

    summary = {
        'n_pairs': len(all_metrics),
        'overall_label_match_rate': _m([1.0 if r['label_match'] else 0.0 for r in all_metrics]),
        'overall_behavior_preserved_rate': _m([1.0 if r['behavior_preserved'] else 0.0 for r in all_metrics]),
        'overall_contact_f1': _m([r['contact_f1_vs_source'] for r in all_metrics]),
        'overall_skating': _m([r['skating'] for r in all_metrics]),
        'overall_accel_smoothness': _m([r['accel_smoothness'] for r in all_metrics]),
        'wall_time_per_pair_s': _m([r['runtime_s'] for r in all_metrics]),
        'wall_time_total_s': float(sum(r['runtime_s'] for r in all_metrics)),
        'gpu_peak_mb_max': float(max((r['gpu_peak_mb'] for r in all_metrics), default=0.0)),
        'per_class_pred_overall': dict(Counter(r['tgt_pred_v2'] for r in all_metrics)),
        'by_stratum': by_stratum,
    }
    out_json = pjoin(out_dir, 'metrics.json')
    with open(out_json, 'w') as f:
        json.dump({'summary': summary, 'per_pair': all_metrics,
                   'config': {
                       'n_steps': n_steps, 'lambda_m': lambda_m, 'fps': fps,
                       'ckpt': ckpt_path, 'inertia_model': 'uniform_cylinder',
                       'radius_ratio': 0.1,
                   }}, f, indent=2)
    print(f"\n[physics_dps] SUMMARY:\n{json.dumps(summary, indent=2)}", flush=True)
    print(f"\n[physics_dps] wrote {out_json}", flush=True)
    return summary, all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='save/A3_baseline_dataset_truebones_bs_2_latentdim_256/model000229999.pt')
    p.add_argument('--device', default='cuda')
    p.add_argument('--n_steps', type=int, default=40)
    p.add_argument('--lambda_m', type=float, default=0.1)
    p.add_argument('--fps', type=float, default=20.0)
    p.add_argument('--n_pairs', type=int, default=10)
    args = p.parse_args()

    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = pjoin(PROJECT_ROOT, ckpt_path)
    run_pilot(ckpt_path, args.device, args.n_pairs, args.n_steps,
              args.lambda_m, args.fps)


if __name__ == '__main__':
    main()
