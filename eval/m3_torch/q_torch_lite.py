"""Differentiable Q-features (lite) for M3 Phase B optim.

Operates on motion features [T, J, 13] in PyTorch.
Skips full quaternion recovery (numpy-only); uses joint position channels
(data[..., 1:, :3]) directly as a proxy. Q is computed in a way that is
fully differentiable so we can backprop through it during per-query optim.

Components (all skeleton-agnostic by design):
  - COM proxy   : mean across joints of position channels per frame
  - Velocity    : finite-difference of COM
  - Contact     : data[..., :, 12] (foot-contact channel)
  - Limb energy : per-joint kinetic energy (mean of |vel|^2), then top-K aggregated
  - Bone length : per-joint distance to its parent in REST pose (compare to motion)

Returns a fixed-length feature vector that can be matched via L2 / cosine.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def com_torch(motion: torch.Tensor) -> torch.Tensor:
    """COM proxy from joint position channels.

    motion: [T, J, 13]
    returns: [T, 3] (mean of joints' (xyz) position channels)
    """
    pos = motion[..., 1:, :3]  # [T, J-1, 3] — skip root (its slot is special)
    return pos.mean(dim=-2)


def velocity_torch(x: torch.Tensor) -> torch.Tensor:
    """Finite-difference velocity. x: [T, ...]; returns: [T, ...]."""
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    v = torch.zeros_like(x)
    v[1:] = x[1:] - x[:-1]
    v[0] = v[1]
    return v


def contact_signal(motion: torch.Tensor) -> torch.Tensor:
    """Aggregate contact signal per frame. motion: [T, J, 13]; returns: [T]."""
    return motion[..., :, 12].mean(dim=-1)


def cadence_spectrum_torch(contact_per_frame: torch.Tensor, fps: int = 30,
                           min_freq: float = 0.25, max_freq: float = 4.0) -> torch.Tensor:
    """Power spectrum of (zero-mean) contact signal in cadence band.

    contact_per_frame: [T]
    returns: [F] power-spectrum values in the cadence frequency band
    """
    s = contact_per_frame - contact_per_frame.mean()
    if s.abs().max() < 1e-6:
        return torch.zeros(8, device=s.device)
    fft = torch.fft.rfft(s, dim=-1)
    freqs = torch.fft.rfftfreq(s.shape[-1], d=1.0 / fps).to(s.device)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if mask.sum() == 0:
        return torch.zeros(8, device=s.device)
    band = fft[mask].abs() ** 2
    # Pad / truncate to fixed-length 8
    if band.shape[-1] >= 8:
        return band[:8]
    pad = torch.zeros(8 - band.shape[-1], device=band.device)
    return torch.cat([band, pad], dim=-1)


def per_joint_kinetic_energy_torch(motion: torch.Tensor) -> torch.Tensor:
    """Per-joint kinetic energy = mean over time of |vel|^2.

    motion: [T, J, 13]; returns: [J] (uses position channels)
    """
    pos = motion[..., :, :3]  # [T, J, 3]
    vel = velocity_torch(pos)  # [T, J, 3]
    ke = 0.5 * (vel ** 2).sum(dim=-1)  # [T, J]
    return ke.mean(dim=0)  # [J]


def q_torch_lite(motion: torch.Tensor, fps: int = 30) -> torch.Tensor:
    """Compute a 24-d Q-feature vector from a motion clip.

    motion: [T, J, 13]
    returns: [24]  (all torch-differentiable wrt motion)

    Components:
      [0:3]   COM mean
      [3:6]   COM std
      [6:9]   COM end-start (net displacement)
      [9]     contact density (mean)
      [10]    contact change rate (mean abs diff)
      [11:19] cadence spectrum (8 bins in cadence band)
      [19:23] limb-energy top-4 (after softmax-normalized)
      [23]    limb-energy entropy
    """
    T = motion.shape[0]
    com = com_torch(motion)                     # [T, 3]
    feats = []

    # COM stats
    feats.append(com.mean(dim=0))               # [3]
    feats.append(com.std(dim=0))                # [3]
    feats.append(com[-1] - com[0])              # [3]

    # Contact
    contact = contact_signal(motion)            # [T]
    feats.append(contact.mean(dim=0, keepdim=True))           # [1]
    if T >= 2:
        contact_change = (contact[1:] - contact[:-1]).abs().mean(dim=0, keepdim=True)
    else:
        contact_change = torch.zeros(1, device=motion.device)
    feats.append(contact_change)                              # [1]

    # Cadence spectrum
    feats.append(cadence_spectrum_torch(contact, fps=fps))    # [8]

    # Limb energy
    ke = per_joint_kinetic_energy_torch(motion)               # [J]
    ke_norm = F.softmax(ke / (ke.std() + 1e-6), dim=-1)
    ke_sorted, _ = torch.sort(ke_norm, descending=True)
    if ke_sorted.shape[-1] >= 4:
        feats.append(ke_sorted[:4])                           # [4]
    else:
        pad = torch.zeros(4 - ke_sorted.shape[-1], device=motion.device)
        feats.append(torch.cat([ke_sorted, pad], dim=-1))
    # entropy
    p = ke_norm + 1e-10
    entropy = -(p * p.log()).sum().reshape(1)
    feats.append(entropy)                                      # [1]

    return torch.cat(feats, dim=-1)                            # [24]


def q_distance_torch(q1: torch.Tensor, q2: torch.Tensor, mode: str = 'cosine') -> torch.Tensor:
    """Distance between two Q vectors. mode: 'cosine' or 'l2'.

    Both q1, q2: [24] or [B, 24].
    Returns scalar.
    """
    if mode == 'cosine':
        if q1.dim() == 1:
            return 1.0 - F.cosine_similarity(q1.unsqueeze(0), q2.unsqueeze(0), dim=-1).squeeze()
        return (1.0 - F.cosine_similarity(q1, q2, dim=-1)).mean()
    return F.mse_loss(q1, q2)


def smoothness_loss_torch(motion: torch.Tensor) -> torch.Tensor:
    """Second-derivative smoothness (jerk-like) penalty.

    motion: [T, J, 13]; returns: scalar
    Penalizes only the position channels (joints' :3 + root's relevant channels).
    """
    pos = motion[..., :, :3]
    if pos.shape[0] < 3:
        return torch.zeros(1, device=motion.device).squeeze()
    accel = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    return accel.pow(2).mean()


def bone_validity_loss_torch(positions: torch.Tensor, parents: list,
                             rest_offsets: torch.Tensor) -> torch.Tensor:
    """L2 penalty: bone length deviation from rest-pose offsets.

    positions: [T, J, 3] (RECOVERED positions, NOT motion features)
    parents: list of length J (parent index per joint, -1 or 0 for root)
    rest_offsets: [J, 3] rest-pose joint offsets relative to parent

    returns: scalar
    """
    T, J, _ = positions.shape
    rest_lengths = rest_offsets.norm(dim=-1)  # [J]
    valid_mask = torch.tensor([0 <= p < J and p != j for j, p in enumerate(parents)],
                              dtype=torch.bool, device=positions.device)
    if valid_mask.sum() == 0:
        return torch.zeros(1, device=positions.device).squeeze()
    parent_idx = torch.tensor([p if (0 <= p < J and p != j) else 0
                               for j, p in enumerate(parents)],
                              dtype=torch.long, device=positions.device)
    bone_vec = positions - positions[:, parent_idx]  # [T, J, 3]
    bone_len = bone_vec.norm(dim=-1)                  # [T, J]
    target_len = rest_lengths.unsqueeze(0).expand(T, -1)
    loss = ((bone_len - target_len).pow(2) * valid_mask.float()).mean()
    return loss


def foot_skate_loss_torch(positions: torch.Tensor, contact: torch.Tensor,
                          foot_indices: list) -> torch.Tensor:
    """Foot-skate loss: when contact=1, foot velocity should be ~0.

    positions: [T, J, 3]
    contact: [T, J] (binary or soft)
    foot_indices: list of joint indices considered as feet (heuristic)
    """
    if not foot_indices or positions.shape[0] < 2:
        return torch.zeros(1, device=positions.device).squeeze()
    foot_pos = positions[:, foot_indices, :]  # [T, F, 3]
    foot_vel = velocity_torch(foot_pos)        # [T, F, 3]
    foot_contact = contact[:, foot_indices]    # [T, F]
    # When contact, velocity should be small
    skate = (foot_vel.norm(dim=-1) * foot_contact).pow(2)  # [T, F]
    return skate.mean()
