"""Idea J — SMILING-motion: score-matching IRL for cross-skeleton retargeting.

Reference: Ren et al., *Imitation Learning via Score-Based Diffusion Policies*
(arXiv:2410.13855). Adapted to cross-skeleton motion retargeting as follows:

- Expert = the source motion (single clip on the source skeleton).
- A3 (the unconditional AnyTop DDPM trained on all 70 Truebones skeletons)
  provides a skeleton-conditioned score function ``s_θ(x, t, skel)``.
- For each pair (source_motion_S, target_skel_T) we optimise directly in the
  13-dim AnyTop input representation ``x ∈ [1, J_pad, 13, T]``.
- Loss:

      L(x) = w_Q  · || Q(x; T)     -  Q(source; S) ||²
           + w_σ  · score_loss(x; T)
           + w_smooth · || Δ²x ||²

  with the score loss (SMILING core) being a predict-x0 variant of score
  matching: sample ``t_fix``, draw ``ε``, form ``x_t = √a * x0 + √1-a * ε``,
  predict ``x0_hat = model(x_t, t; y_T)``, and take ``|| x0_hat - x ||²``
  averaged over real joints / features.  Because A3 has ``predict_xstart``,
  this loss has the same minimiser as classical score matching but is
  numerically better conditioned.

This implementation is *distinct* from K (Stage-1 Q → Stage-2 IK → Stage-3
DDIM SDEdit) and K_ikprior (joint-angle IK with the score as a soft
regulariser every 25 iters): here **the score is the dominant signal and
optimisation happens directly in motion-representation space**, without FK
or rotation-refitting.

Outputs
-------
* per-pair motion -> eval/results/k_compare/idea_J_smiling/pair_<id>_<src>_to_<tgt>.npy
* metrics.json    -> eval/results/k_compare/idea_J_smiling/metrics.json

Follow-up scoring goes through eval/k_action_accuracy_v2_full.py which
auto-discovers this directory.
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/idea_J_smiling'
OUT_DIR.mkdir(parents=True, exist_ok=True)

_EPS = 1e-8
POS_Y_IDX = 1
ROT_START, ROT_END = 3, 9
FOOT_CH_IDX = 12


# =================== asset loading (shared with K/N) ===================

def load_assets():
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'),
                   allow_pickle=True).item()
    with open(PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        cg = json.load(f)
    motion_dir = os.path.join(DATASET_DIR, 'motions')
    return cond, cg, motion_dir


def remap_contact_sched(q_src: dict, src_skel: str, tgt_skel: str,
                        contact_groups: dict) -> tuple:
    src_sched = q_src['contact_sched']
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


def build_q_star(q_src: dict, src_skel: str, tgt_skel: str,
                 contact_groups: dict, cond: dict) -> dict:
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


# ====================== A3 loader (shared machinery) ======================

DEFAULT_CKPT = 'save/A3_baseline_dataset_truebones_bs_2_latentdim_256/model000229999.pt'
FALLBACK_CKPT = 'save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt'


_MODEL_CACHE: dict = {}
_T5_CACHE: dict = {}


def _load_anytop(ckpt_path: str, device: torch.device):
    key = (ckpt_path, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from model.anytop import AnyTop
    from utils.model_util import create_gaussian_diffusion

    with open(os.path.join(os.path.dirname(ckpt_path), 'args.json')) as f:
        ckpt_args = json.load(f)

    class Ns:
        def __init__(self, d): self.__dict__.update(d)
    m_args = Ns(ckpt_args)
    t5_dim = {'t5-small': 512, 't5-base': 768, 't5-large': 1024}
    t5_out = t5_dim.get(m_args.t5_name, 768)

    model = AnyTop(
        max_joints=143, feature_len=13,
        latent_dim=m_args.latent_dim, ff_size=1024,
        num_layers=m_args.layers, num_heads=4,
        dropout=0.1, activation='gelu',
        t5_out_dim=t5_out, root_input_feats=13,
        cond_mode='object_type', cond_mask_prob=m_args.cond_mask_prob,
        skip_t5=m_args.skip_t5, value_emb=m_args.value_emb)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected keys: {unexpected[:3]}')
    bad_missing = [k for k in missing if not k.startswith('clip_model.')]
    if bad_missing:
        raise RuntimeError(f'Missing keys not in clip: {bad_missing[:3]}')
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    diffusion = create_gaussian_diffusion(m_args)
    _MODEL_CACHE[key] = (model, diffusion, m_args)
    return model, diffusion, m_args


def _load_t5(t5_name: str, device_str: str):
    key = (t5_name, device_str)
    if key in _T5_CACHE:
        return _T5_CACHE[key]
    from model.conditioners import T5Conditioner
    t5 = T5Conditioner(name=t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device=device_str)
    _T5_CACHE[key] = t5
    return t5


def _build_y(skel_name: str, cond: dict, t5, n_frames: int,
             temporal_window: int, max_joints: int, device: torch.device):
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation

    info = cond[skel_name]
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

    return {
        'joints_mask':       jmask,
        'mask':              tmask_t,
        'tpos_first_frame':  tpos_t,
        'joints_names_embs': names_t,
        'graph_dist':        gd_t,
        'joints_relations':  jr_t,
        'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
        'n_joints':          torch.tensor([n_joints]),
    }, n_joints, mean, std


# ========== Differentiable Q from a 13-dim motion (torch) ==========

def _subtree_masses_np(parents, offsets):
    J = len(parents)
    m = np.linalg.norm(offsets, axis=1).copy()
    for j in reversed(range(J)):
        p = parents[j]
        if 0 <= p < J and p != j:
            m[p] += m[j]
    return m


def _body_scale_np(offsets):
    return float(np.linalg.norm(offsets, axis=1).sum() + _EPS)


def _recover_positions_from_13dim_t(x_tjc: torch.Tensor) -> torch.Tensor:
    """Differentiable partial recover of global positions from the 13-dim
    representation.  Mirrors ``recover_from_bvh_ric_np`` but uses a simple
    Y-yaw quaternion built from the ROOT slot's angular-velocity (channel 0).

    x_tjc : [T, J, 13]  (denormalised)
    Returns positions [T, J, 3] in world frame (best-effort differentiable).
    """
    T, J, _ = x_tjc.shape
    device, dtype = x_tjc.device, x_tjc.dtype
    # Root angular velocity -> yaw angle (cumsum) -> quaternion (w, 0, y, 0).
    ang_vel = x_tjc[:, 0, 0]
    yaw = torch.zeros(T, dtype=dtype, device=device)
    yaw[1:] = torch.cumsum(ang_vel[:-1], dim=0)
    cos_h = torch.cos(yaw * 0.5)
    sin_h = torch.sin(yaw * 0.5)
    # Quaternion (w, x, y, z) = (cos(h), 0, sin(h), 0); inverse is (cos, 0, -sin, 0).
    q_inv = torch.stack([cos_h, torch.zeros_like(cos_h), -sin_h,
                         torch.zeros_like(cos_h)], dim=-1)  # [T, 4]

    # Root XZ linear velocities.
    r_xz_vel = torch.zeros(T, 3, dtype=dtype, device=device)
    if T >= 2:
        r_xz_vel[1:, 0] = x_tjc[:-1, 0, 9]
        r_xz_vel[1:, 2] = x_tjc[:-1, 0, 11]
    # Apply q_inv to r_xz_vel (rotate vector by quaternion).
    r_pos_world = _quat_apply_t(q_inv, r_xz_vel)  # per-frame delta
    r_pos = torch.cumsum(r_pos_world, dim=0)      # [T, 3]
    r_pos[:, 1] = x_tjc[:, 0, POS_Y_IDX]           # Y from root slot channel 1

    # Non-root joints: ric positions in channels 0..2, rotate by -q into world, add root XZ.
    ric = x_tjc[:, 1:, 0:3]                       # [T, J-1, 3]
    q_inv_b = q_inv.unsqueeze(1).expand(T, J - 1, 4)
    pos_world = _quat_apply_t(q_inv_b, ric)
    pos_world[..., 0] = pos_world[..., 0] + r_pos[:, 0:1]
    pos_world[..., 2] = pos_world[..., 2] + r_pos[:, 2:3]

    positions = torch.cat([r_pos.unsqueeze(1), pos_world], dim=1)  # [T, J, 3]
    return positions


def _quat_apply_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion (w, x, y, z) to vector v (both broadcast)."""
    w = q[..., 0:1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def _canonical_R_from_positions_t(positions: torch.Tensor, w_sub: torch.Tensor,
                                  smoothing: float = 0.9
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return weighted-subtree centroids + body-frame rotation.

    R columns are [forward, up, lateral] with forward from smoothed XZ
    velocity of the centroid; up = +Y axis.
    """
    T = positions.shape[0]
    centroids = torch.einsum('j,tjd->td', w_sub, positions)  # [T, 3]
    vel = torch.zeros_like(centroids)
    vel[1:] = centroids[1:] - centroids[:-1]
    vel[0] = vel[1]
    vel_h = vel.clone()
    vel_h[:, 1] = 0.0
    fwd = torch.zeros_like(vel_h)
    running = vel_h[0]
    for t in range(T):
        running = smoothing * running + (1.0 - smoothing) * vel_h[t]
        fwd[t] = running
    mag = torch.linalg.norm(fwd, dim=-1, keepdim=True).clamp_min(_EPS)
    fwd = fwd / mag
    if torch.linalg.norm(fwd).item() < _EPS:
        dir_z = torch.tensor([1.0, 0.0, 0.0], dtype=fwd.dtype, device=fwd.device)
        fwd = dir_z.expand_as(fwd)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=fwd.dtype, device=fwd.device).expand(T, 3)
    lat = torch.cross(up, fwd, dim=-1)
    R = torch.stack([fwd, up, lat], dim=-1)  # columns: forward/up/lateral
    return centroids, R


def compute_q_from_positions_t(positions: torch.Tensor,
                               w_sub: torch.Tensor,
                               body_scale: float,
                               group_idx: List[List[int]],
                               chains: List[List[int]],
                               fps: float = 30.0) -> Dict[str, torch.Tensor]:
    """Differentiable Q from world positions (matches quotient_extractor)."""
    T = positions.shape[0]
    centroids, R = _canonical_R_from_positions_t(positions, w_sub)
    com_path = (centroids - centroids[0:1]) / body_scale

    # Heading velocity = centroid vel projected on forward column.
    vel_c = torch.zeros_like(centroids)
    vel_c[1:] = (centroids[1:] - centroids[:-1]) * fps
    vel_c[0] = vel_c[1]
    heading_vel = (vel_c * R[:, :, 0]).sum(dim=-1) / body_scale

    # Soft contact schedule — sigmoid on joint Y (same as IK solver).
    alpha = 10.0 / max(body_scale, _EPS)
    C = len(group_idx)
    if C > 0:
        sched = torch.zeros(T, C, dtype=positions.dtype, device=positions.device)
        for c, idxs in enumerate(group_idx):
            if idxs:
                y = positions[:, idxs, 1]
                sched[:, c] = torch.sigmoid(-alpha * y).mean(dim=1)
    else:
        sched = torch.zeros(T, 0, dtype=positions.dtype, device=positions.device)

    # Cadence: peak freq on summed contact schedule.
    if sched.numel() > 0 and T >= 4:
        s = sched.sum(dim=1)
        s = s - s.mean()
        fft = torch.fft.rfft(s)
        power = fft.real ** 2 + fft.imag ** 2
        freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(s.device)
        mask = ((freqs >= 0.25) & (freqs <= 4.0)).to(power.dtype)
        num = (power * freqs * mask).sum()
        den = (power * mask).sum().clamp_min(_EPS)
        cadence = num / den
    else:
        cadence = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    # Limb usage = normalised KE per chain.
    K = len(chains)
    if K > 0:
        vel_j = torch.zeros_like(positions)
        vel_j[1:] = (positions[1:] - positions[:-1]) * fps
        vel_j[0] = vel_j[1]
        ke_j = (0.5 * (vel_j ** 2).sum(dim=-1)).mean(dim=0)  # [J]
        J_total = positions.shape[1]
        energy = torch.zeros(K, dtype=positions.dtype, device=positions.device)
        for k, chain in enumerate(chains):
            idxs = [int(i) for i in chain if 0 <= int(i) < J_total]
            if idxs:
                energy[k] = ke_j[idxs].mean()
        limb_usage = energy / (energy.sum() + _EPS)
    else:
        limb_usage = torch.zeros(1, dtype=positions.dtype, device=positions.device)
    return {
        'com_path': com_path,
        'heading_vel': heading_vel,
        'contact_sched': sched,
        'cadence': cadence,
        'limb_usage': limb_usage,
    }


# ======================= Init helpers =======================

def _init_motion_from_tpose(tpos_13: np.ndarray, n_joints: int, T: int,
                            max_joints: int, feature_len: int = 13,
                            noise_scale: float = 0.02,
                            rng: Optional[np.random.RandomState] = None
                            ) -> np.ndarray:
    """Replicate target T-pose over T frames + small Gaussian noise.

    tpos_13: [J, 13] target-skeleton rest-pose frame (denormalised).
    Returns padded [J_pad, 13, T] tensor (channel-last-time layout).
    """
    rng = rng or np.random.RandomState(0)
    x = np.zeros((max_joints, feature_len, T), dtype=np.float32)
    x[:n_joints] = tpos_13[:n_joints, :, None]
    x[:n_joints] += rng.randn(n_joints, feature_len, T).astype(np.float32) * noise_scale
    return x


# ======================= Main solver (one pair) =======================

def smiling_retarget(q_src: dict, q_star: dict, target_skel: str,
                     cond: dict, contact_groups: dict,
                     device: torch.device,
                     n_iters: int = 400,
                     w_Q: float = 1.0,
                     w_score: float = 0.3,
                     w_smooth: float = 0.01,
                     t_fix: int = 30,
                     score_every: int = 1,
                     ckpt_path: Optional[str] = None,
                     verbose: bool = False,
                     fps: int = 30,
                     seed: int = 0,
                     ) -> dict:
    """Optimise a target-skeleton motion directly in 13-dim space.

    Variables: ``x_norm`` of shape [1, J_pad, 13, T] (normalised input space).
    Returns denormalised motion [T, J_t, 13] + diagnostics.
    """
    # --- Resolve checkpoint ---
    ck = ckpt_path or os.path.join(str(PROJECT_ROOT), DEFAULT_CKPT)
    if not os.path.isabs(ck):
        ck = os.path.join(str(PROJECT_ROOT), ck)
    if not os.path.exists(ck):
        ck = os.path.join(str(PROJECT_ROOT), FALLBACK_CKPT)

    model, diffusion, m_args = _load_anytop(ck, device)
    max_joints = 143
    feat_len = 13
    t5 = _load_t5(m_args.t5_name, device_str=str(device).replace('cuda:', 'cuda'))

    n_frames_model = m_args.num_frames
    # Target number of optimisation frames: min(source T, model T_max).
    T_src = int(q_src['com_path'].shape[0])
    T = min(T_src, n_frames_model)

    y, n_joints, mean_np, std_np = _build_y(
        target_skel, cond, t5, n_frames_model,
        m_args.temporal_window, max_joints, device)

    # Stats padded to max_joints for easy broadcasting.
    mean_pad = np.zeros((max_joints, feat_len), dtype=np.float32)
    std_pad = np.ones((max_joints, feat_len), dtype=np.float32)
    mean_pad[:n_joints] = mean_np
    std_pad[:n_joints] = std_np
    mean_t = torch.from_numpy(mean_pad).to(device)
    std_t = torch.from_numpy(std_pad).to(device)

    # --- Init (denormalised, then normalise) ---
    tpos_13 = cond[target_skel]['tpos_first_frame'].astype(np.float32)  # [J, 13]
    rng = np.random.RandomState(seed)
    x_denorm_init = _init_motion_from_tpose(tpos_13, n_joints, T,
                                            max_joints, feat_len,
                                            noise_scale=0.02, rng=rng)
    # Broadcast to model's n_frames.
    if T < n_frames_model:
        x_denorm = np.zeros((max_joints, feat_len, n_frames_model), dtype=np.float32)
        x_denorm[..., :T] = x_denorm_init
    else:
        x_denorm = x_denorm_init
    # Normalise.
    x_norm_np = np.zeros_like(x_denorm)
    x_norm_np[:n_joints] = (x_denorm[:n_joints] - mean_np[:, :, None]) / (std_np[:, :, None] + _EPS)
    x_norm_np = np.nan_to_num(x_norm_np).astype(np.float32)
    x_norm_t = torch.from_numpy(x_norm_np).unsqueeze(0).to(device)
    x_norm_t.requires_grad_(True)

    # --- Targets (torch) ---
    com_src = torch.tensor(q_star['com_path'][:T], dtype=torch.float32, device=device)
    heading_src = torch.tensor(q_star['heading_vel'][:T], dtype=torch.float32, device=device)
    sched_src_np = np.asarray(q_star['contact_sched'], dtype=np.float32)
    if sched_src_np.ndim == 1:
        sched_src_np = sched_src_np[:, None]
    sched_src = torch.tensor(sched_src_np[:T], dtype=torch.float32, device=device)
    cadence_src = float(q_star['cadence'])
    limb_src = torch.tensor(np.asarray(q_star['limb_usage'], dtype=np.float32),
                            device=device)

    # --- Target-skeleton geometry for Q reconstruction ---
    info = cond[target_skel]
    parents = info['parents']
    offsets = np.asarray(info['offsets'], dtype=np.float32)
    body_scale = _body_scale_np(offsets)
    subtree = _subtree_masses_np(parents, offsets)
    w_sub = subtree / (subtree.sum() + _EPS)
    w_sub_t = torch.tensor(w_sub, dtype=torch.float32, device=device)
    chains = info.get('kinematic_chains', []) or []

    group_names = q_star.get('contact_group_names') or []
    tgt_groups = contact_groups.get(target_skel, {})
    group_idx_tgt: List[List[int]] = []
    for name in group_names:
        if name in tgt_groups:
            group_idx_tgt.append([int(i) for i in tgt_groups[name]
                                  if 0 <= int(i) < n_joints])
        else:
            group_idx_tgt.append([])
    # Drop empty groups (they can't contribute to contact loss).
    keep = [c for c, idxs in enumerate(group_idx_tgt) if len(idxs) > 0]
    if len(keep) < len(group_idx_tgt):
        group_idx_tgt = [group_idx_tgt[c] for c in keep]
        sched_src = sched_src[:, keep]

    # --- A3 alpha_bar for fixed t ---
    alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32,
                              device=device)
    a_bar = alphas_cum[int(t_fix)]
    sqrt_a = torch.sqrt(a_bar)
    sqrt_1ma = torch.sqrt((1.0 - a_bar).clamp_min(1e-12))

    # --- Optimiser ---
    optim = torch.optim.Adam([x_norm_t], lr=0.02)
    n_warm = int(0.6 * n_iters)

    best_total = float('inf')
    best_x = x_norm_t.detach().clone()
    history: List[Tuple[int, float, float, float]] = []
    last_terms: Dict[str, float] = {}

    t_loop_start = time.time()
    for it in range(n_iters):
        if it == n_warm:
            for g in optim.param_groups:
                g['lr'] = 0.005
        optim.zero_grad()

        # De-normalise the active slice of joints / frames.
        x_tjc = (x_norm_t[0, :n_joints, :, :T]            # [J, F, T_real]
                  * std_t[:n_joints, :, None]
                  + mean_t[:n_joints, :, None])
        # Reorder to [T, J, 13].
        x_tjc = x_tjc.permute(2, 0, 1)

        # Q on motion via differentiable recover.
        positions = _recover_positions_from_13dim_t(x_tjc)        # [T, J, 3]
        q_rec = compute_q_from_positions_t(positions, w_sub_t, body_scale,
                                           group_idx_tgt, chains, fps=fps)

        # --- Task loss (Q) ---
        l_com = ((q_rec['com_path'] - com_src) ** 2).sum(dim=-1).mean()
        l_heading = ((q_rec['heading_vel'] - heading_src) ** 2).mean()
        if sched_src.numel() > 0 and q_rec['contact_sched'].numel() > 0:
            l_contact = ((q_rec['contact_sched'] - sched_src) ** 2).mean()
        else:
            l_contact = torch.tensor(0.0, dtype=x_tjc.dtype, device=device)
        l_cadence = (q_rec['cadence'] - cadence_src) ** 2
        Km = min(limb_src.numel(), q_rec['limb_usage'].numel())
        if Km > 0:
            l_limb = ((q_rec['limb_usage'][:Km] - limb_src[:Km]) ** 2).mean()
        else:
            l_limb = torch.tensor(0.0, dtype=x_tjc.dtype, device=device)
        l_Q = (20.0 * l_com + 5.0 * l_heading + 1.0 * l_contact
               + 0.05 * l_cadence + 0.5 * l_limb)

        # --- Score loss (SMILING core) every score_every iters ---
        if it % score_every == 0:
            eps = torch.randn_like(x_norm_t)
            t_tensor = torch.tensor([int(t_fix)], device=device, dtype=torch.long)
            x_t = sqrt_a * x_norm_t + sqrt_1ma * eps
            model_out = model(x_t, diffusion._scale_timesteps(t_tensor), y=y)
            # A3 uses predict_xstart -> model_out ≈ x0_hat.
            from diffusion.gaussian_diffusion import ModelMeanType
            if diffusion.model_mean_type == ModelMeanType.START_X:
                x0_hat = model_out
            elif diffusion.model_mean_type == ModelMeanType.EPSILON:
                x0_hat = (x_t - sqrt_1ma * model_out) / sqrt_a.clamp_min(1e-8)
            else:
                x0_hat = model_out  # fallback
            # Score-match loss: mean over real joints / feats / frames.
            diff = (x0_hat[:, :n_joints] - x_norm_t[:, :n_joints])
            l_score = (diff ** 2).mean()
        else:
            l_score = torch.tensor(0.0, dtype=x_tjc.dtype, device=device)

        # --- Temporal smoothness (L2 of 2nd difference) ---
        if T >= 3:
            d2 = x_norm_t[..., :T][:, :n_joints, :, 2:T] \
                  - 2.0 * x_norm_t[..., :T][:, :n_joints, :, 1:T - 1] \
                  + x_norm_t[..., :T][:, :n_joints, :, 0:T - 2]
            l_smooth = (d2 ** 2).mean()
        else:
            l_smooth = torch.tensor(0.0, dtype=x_tjc.dtype, device=device)

        l_total = w_Q * l_Q + w_score * l_score + w_smooth * l_smooth
        l_total.backward()
        torch.nn.utils.clip_grad_norm_([x_norm_t], max_norm=5.0)
        optim.step()

        cur = float(l_total.item())
        q_cur = float((w_Q * l_Q).item())
        s_cur = float((w_score * l_score).item())
        history.append((int(it), cur, q_cur, s_cur))
        if cur < best_total:
            best_total = cur
            best_x = x_norm_t.detach().clone()
            last_terms = {
                'l_Q': float(l_Q.item()),
                'l_com': float(l_com.item()),
                'l_heading': float(l_heading.item()),
                'l_contact': float(l_contact.item()),
                'l_cadence': float(l_cadence.item()) if torch.is_tensor(l_cadence) else float(l_cadence),
                'l_limb': float(l_limb.item()) if torch.is_tensor(l_limb) else float(l_limb),
                'l_score': float(l_score.item()) if torch.is_tensor(l_score) else float(l_score),
                'l_smooth': float(l_smooth.item()),
            }
        if verbose and (it % 50 == 0 or it == n_iters - 1):
            print(f"  iter {it:4d}  total={cur:.4f}  Q={q_cur:.4f}  "
                  f"score={s_cur:.4f}  smooth={float(l_smooth.item()):.4e}")

    runtime = time.time() - t_loop_start

    # --- Extract best motion + denormalise back to [T, J, 13] ---
    x_best_norm = best_x[0, :n_joints, :, :T].detach().cpu().numpy()
    x_best_denorm = x_best_norm * std_np[:, :, None] + mean_np[:, :, None]
    motion_tjc = x_best_denorm.transpose(2, 0, 1).astype(np.float32)  # [T, J, 13]

    # Reconstruct Q from best motion (numpy-path, matches downstream reporting).
    q_rec_np = _reconstruct_q_numpy(motion_tjc, target_skel, cond,
                                    group_idx_tgt, chains, fps=fps)

    return {
        'motion_13dim': motion_tjc,
        'q_reconstructed': q_rec_np,
        'runtime_sec': runtime,
        'history': history,
        'final_loss': last_terms,
        'n_iters': int(n_iters),
        'w_Q': float(w_Q), 'w_score': float(w_score), 'w_smooth': float(w_smooth),
        't_fix': int(t_fix),
        'score_every': int(score_every),
        'body_scale': body_scale,
        'ckpt_path': ck,
    }


# ================= Numpy-path Q reconstruction =================

def _reconstruct_q_numpy(motion_tjc: np.ndarray, target_skel: str, cond: dict,
                         group_idx: List[List[int]], chains: List[List[int]],
                         fps: int = 30) -> dict:
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_ric_np,
    )
    from eval.effect_program import canonical_body_frame
    from eval.quotient_extractor import (
        compute_heading_velocity, compute_cadence, compute_limb_usage,
    )

    info = cond[target_skel]
    offsets = np.asarray(info['offsets'], dtype=np.float32)
    parents = info['parents']
    subtree = _subtree_masses_np(parents, offsets)
    scale = _body_scale_np(offsets)

    positions = recover_from_bvh_ric_np(motion_tjc.astype(np.float32))
    centroids, R = canonical_body_frame(positions, subtree)
    com_path = (centroids - centroids[0:1]) / scale
    heading_vel = compute_heading_velocity(centroids, R, fps=fps) / scale

    T, J, _ = positions.shape
    y = positions[..., 1]
    ground = np.percentile(y, 5)
    tau = max(0.02 * scale, 1e-3)
    contacts = (y - ground < tau).astype(np.int8)
    if group_idx:
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
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
    }


def q_component_errors(q_rec: dict, q_tgt: dict) -> dict:
    from eval.ik_solver import _q_component_errors
    return _q_component_errors(q_rec, q_tgt)


# ========================== Misc utilities ==========================

def contact_f1(sched_rec, sched_tgt, thresh=0.5) -> float:
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


def skating_proxy(motion_13, contact_groups, tgt_skel, cond, fps=30,
                  contact_thresh=0.5) -> float:
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
    x_vel = np.zeros((T, len(foot_joints)))
    z_vel = np.zeros_like(x_vel)
    if T >= 2:
        dp = pos[1:, foot_joints, :] - pos[:-1, foot_joints, :]
        x_vel[1:] = dp[..., 0]
        z_vel[1:] = dp[..., 2]
    horiz_speed = np.sqrt(x_vel ** 2 + z_vel ** 2) * fps
    in_contact = (contact[:, foot_joints] > contact_thresh)
    if not in_contact.any():
        return 0.0
    return float(horiz_speed[in_contact].mean())


def contact_f1_vs_source(motion_13: np.ndarray, sched_src_mapped: np.ndarray,
                         group_idx: List[List[int]], thresh: float = 0.5) -> float:
    """Compute F1 between motion's aggregate contact signature and the
    source-derived target-mapped schedule (as used elsewhere)."""
    T = motion_13.shape[0]
    contact_ch = motion_13[..., 12]   # [T, J]
    if not group_idx:
        pred_agg = contact_ch.mean(axis=1)[:, None]
    else:
        pred_agg = np.stack([
            contact_ch[:, idxs].mean(axis=1) if idxs else np.zeros(T)
            for idxs in group_idx
        ], axis=1)
    T_c = min(T, sched_src_mapped.shape[0])
    return contact_f1(pred_agg[:T_c], sched_src_mapped[:T_c], thresh=thresh)


def _stratum_label(p) -> str:
    if p['support_same_label'] == 0:
        return 'absent'
    if p['family_gap'] == 'near':
        return 'near_present'
    if p['family_gap'] == 'moderate':
        return 'moderate'
    if p['family_gap'] == 'extreme':
        return 'extreme'
    return 'other'


# ========================== Main runner ==========================

def run(n_iters: int = 400, w_Q: float = 1.0, w_score: float = 0.3,
        w_smooth: float = 0.01, t_fix: int = 30, score_every: int = 1,
        max_pairs: Optional[int] = None, verbose: bool = False):
    with open(PROJECT_ROOT / 'idea-stage/eval_pairs.json') as f:
        eval_data = json.load(f)
    pairs = eval_data['pairs']
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    cond, contact_groups, motion_dir = load_assets()

    from eval.quotient_extractor import extract_quotient

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    per_pair = []
    t_total_0 = time.time()
    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']
        tgt_skel = p['target_skel']
        src_fname = p['source_fname']
        print(f"\n=== pair {pid:02d}  {src_skel}({src_fname}) -> {tgt_skel}  "
              f"gap={p['family_gap']}  supp={p['support_same_label']} ===",
              flush=True)

        rec = {
            'pair_id': pid,
            'source_skel': src_skel,
            'target_skel': tgt_skel,
            'source_fname': src_fname,
            'family_gap': p['family_gap'],
            'support_same_label': int(p['support_same_label']),
            'source_label': p.get('source_label'),
            'stratum': _stratum_label(p),
            'status': 'pending',
            'error': None,
        }
        t0 = time.time()
        try:
            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            rec['n_frames'] = int(q_src['n_frames'])
            rec['cadence_src'] = float(q_src['cadence'])

            q_star = build_q_star(q_src, src_skel, tgt_skel, contact_groups, cond)

            out = smiling_retarget(
                q_src, q_star, tgt_skel, cond, contact_groups,
                device=device, n_iters=n_iters,
                w_Q=w_Q, w_score=w_score, w_smooth=w_smooth,
                t_fix=t_fix, score_every=score_every, verbose=verbose,
                seed=pid)
            motion_13 = out['motion_13dim']
            q_rec = out['q_reconstructed']
            rec['runtime'] = float(out['runtime_sec'])
            rec['final_loss'] = out['final_loss']
            # Convergence summary: relative change between first and last total
            hist = out['history']
            if hist:
                t0_total = hist[0][1]; tN_total = hist[-1][1]
                rec['total_loss_first'] = float(t0_total)
                rec['total_loss_last'] = float(tN_total)
                rec['total_loss_ratio'] = float(tN_total / (t0_total + _EPS))
                q_firsts = [h[2] for h in hist if h[2] > 0]
                s_firsts = [h[3] for h in hist if h[3] > 0]
                if q_firsts:
                    rec['Q_loss_first'] = q_firsts[0]
                    rec['Q_loss_last'] = q_firsts[-1]
                if s_firsts:
                    rec['score_loss_first'] = s_firsts[0]
                    rec['score_loss_last'] = s_firsts[-1]

            errs = q_component_errors(q_rec, q_star)
            rec['q_errors'] = errs

            rec['contact_f1_vs_source'] = contact_f1(
                q_rec['contact_sched'], q_star['contact_sched'])
            rec['skating_proxy'] = skating_proxy(motion_13, contact_groups,
                                                 tgt_skel, cond)
            out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_path, motion_13)
            rec['output_path'] = str(out_path)
            rec['pair_runtime'] = float(time.time() - t0)
            rec['status'] = 'ok'
            print(f"  ok: com_rel={errs['com_path_rel_l2']:.3f}  "
                  f"contact_F1={rec['contact_f1_vs_source']:.3f}  "
                  f"skate={rec['skating_proxy']:.4f}  "
                  f"runtime={rec['pair_runtime']:.1f}s  "
                  f"Q_last={rec.get('Q_loss_last',0):.3f}  "
                  f"score_last={rec.get('score_loss_last',0):.3f}",
                  flush=True)
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f"  FAILED: {e}", flush=True)
        per_pair.append(rec)

    total_time = time.time() - t_total_0

    # --- Aggregate by stratum ---
    strata_order = ['near_present', 'absent', 'moderate', 'extreme']
    strata = {s: [] for s in strata_order}
    for r in per_pair:
        strata[r['stratum']].append(r)

    def _stats(entries):
        ok = [e for e in entries if e['status'] == 'ok']
        if not ok:
            return {'n_total': len(entries), 'n_ok': 0}
        qs = lambda k: float(np.mean([e['q_errors'][k] for e in ok]))
        def m(k, default=0.0):
            vals = [e.get(k, default) for e in ok if e.get(k) is not None]
            return float(np.mean(vals)) if vals else default
        return {
            'n_total': len(entries),
            'n_ok': len(ok),
            'com_path_rel_l2':     qs('com_path_rel_l2'),
            'heading_vel_rel_l2':  qs('heading_vel_rel_l2'),
            'contact_sched_mae':   qs('contact_sched_mae'),
            'cadence_abs':         qs('cadence_abs'),
            'limb_usage_rel_l2':   qs('limb_usage_rel_l2'),
            'contact_f1_vs_source': float(np.mean([e['contact_f1_vs_source'] for e in ok])),
            'skating_proxy':        float(np.mean([e['skating_proxy'] for e in ok])),
            'Q_loss_first':  m('Q_loss_first'),
            'Q_loss_last':   m('Q_loss_last'),
            'score_loss_first': m('score_loss_first'),
            'score_loss_last':  m('score_loss_last'),
            'total_loss_ratio': m('total_loss_ratio'),
            'pair_runtime':  float(np.mean([e['pair_runtime'] for e in ok])),
        }
    strata_stats = {s: _stats(strata[s]) for s in strata_order}

    ok_entries = [r for r in per_pair if r['status'] == 'ok']
    overall = {}
    if ok_entries:
        overall = {
            'com_path_rel_l2': float(np.mean([e['q_errors']['com_path_rel_l2']
                                              for e in ok_entries])),
            'heading_vel_rel_l2': float(np.mean([e['q_errors']['heading_vel_rel_l2']
                                                 for e in ok_entries])),
            'contact_f1_vs_source': float(np.mean([e['contact_f1_vs_source']
                                                    for e in ok_entries])),
            'skating_proxy': float(np.mean([e['skating_proxy'] for e in ok_entries])),
            'Q_loss_first_mean': float(np.mean([e.get('Q_loss_first', 0.0) for e in ok_entries])),
            'Q_loss_last_mean':  float(np.mean([e.get('Q_loss_last',  0.0) for e in ok_entries])),
            'score_loss_first_mean': float(np.mean([e.get('score_loss_first', 0.0) for e in ok_entries])),
            'score_loss_last_mean':  float(np.mean([e.get('score_loss_last',  0.0) for e in ok_entries])),
            'total_loss_ratio_mean': float(np.mean([e.get('total_loss_ratio', 1.0) for e in ok_entries])),
            'pair_runtime': float(np.mean([e['pair_runtime'] for e in ok_entries])),
        }

    out = {
        'variant': 'Idea J (SMILING-motion: score-matching IRL in 13-dim space)',
        'w_Q': float(w_Q), 'w_score': float(w_score), 'w_smooth': float(w_smooth),
        't_fix': int(t_fix), 'score_every': int(score_every),
        'n_iters': int(n_iters),
        'total_runtime_sec': float(total_time),
        'n_pairs': len(pairs),
        'n_ok': sum(1 for r in per_pair if r['status'] == 'ok'),
        'n_failed': sum(1 for r in per_pair if r['status'] == 'failed'),
        'overall': overall,
        'strata_stats': strata_stats,
        'per_pair': per_pair,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n=== DONE: total {total_time:.1f}s, n_ok={out['n_ok']}/{out['n_pairs']}, "
          f"failed={out['n_failed']} ===", flush=True)
    print(f"metrics saved to {OUT_DIR / 'metrics.json'}", flush=True)
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iters',   type=int,   default=400)
    parser.add_argument('--w_Q',       type=float, default=1.0)
    parser.add_argument('--w_score',   type=float, default=0.3)
    parser.add_argument('--w_smooth',  type=float, default=0.01)
    parser.add_argument('--t_fix',     type=int,   default=30)
    parser.add_argument('--score_every', type=int, default=1)
    parser.add_argument('--max_pairs', type=int,   default=None)
    parser.add_argument('--verbose',   action='store_true')
    args = parser.parse_args()
    run(n_iters=args.n_iters, w_Q=args.w_Q, w_score=args.w_score,
        w_smooth=args.w_smooth, t_fix=args.t_fix,
        score_every=args.score_every, max_pairs=args.max_pairs,
        verbose=args.verbose)
