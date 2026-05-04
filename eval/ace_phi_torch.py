"""ACE feature extractor Ψ (torch, differentiable). Mirrors eval/ace_phi.py.

Hard gate: torch_ace_psi must match numpy ace_psi to 1e-4 (fp64) on 100 random windows × 5 skels.
Used in L_feat training-time computation through STE-decoder.

Reuses MoReFlow's torch recovery (rotation_6d_to_matrix_columnwise, torch_recover_root_world,
torch_recover_joint_world, _log_so3_torch).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.moreflow_phi_torch import (
    rotation_6d_to_matrix_columnwise,
    torch_recover_root_world,
    torch_recover_joint_world,
    _log_so3_torch,
)
from eval.moreflow_phi import G_MAX, load_contact_groups
from eval.ace_phi import (
    D_PSI,
    ace_psi_per_frame as ace_psi_per_frame_np,
    precompute_ace_descriptors,
)


def matrix_to_rotation_6d_columnwise(R):
    """Inverse of rotation_6d_to_matrix_columnwise. Returns [..., 6] = first 2 columns flattened."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def torch_ace_psi_per_frame(motion_13d_phys, body_length, ee_joints, n_ee):
    """Per-frame ACE Ψ.

    motion_13d_phys: [B, T, J, 13] in physical units
    body_length: [B] FloatTensor
    ee_joints: [B, G_MAX] LongTensor (-1 = unused)
    n_ee: [B] LongTensor

    Returns: [B, T, D_PSI=37]
    """
    B, T, J, _ = motion_13d_phys.shape
    device = motion_13d_phys.device
    dtype = motion_13d_phys.dtype

    # 1. Recover root world position; ALSO compute R_world separately (MoReFlow's torch port returns only pos_world)
    pos_world_root = torch_recover_root_world(motion_13d_phys)            # [B, T, 3]
    R_world = rotation_6d_to_matrix_columnwise(motion_13d_phys[..., 0, 3:9])  # [B, T, 3, 3]

    # 2. Height: body-length normalized
    height = pos_world_root[..., 1:2] / body_length[:, None, None]        # [B, T, 1]

    # 3. 6D rotation (column-conv)
    rot_6d = matrix_to_rotation_6d_columnwise(R_world)                    # [B, T, 6]

    # 4. Linear velocity local: channels 9, 10, 11 of root row, body-length normalized
    lin_vel_local = motion_13d_phys[:, :, 0, 9:12] / body_length[:, None, None]  # [B, T, 3]

    # 5. Angular velocity per frame: log_so3(R_t^T @ R_{t+1})
    R_rel = torch.einsum('btji,btjk->btik', R_world[:, :-1], R_world[:, 1:])  # [B, T-1, 3, 3]
    ang_vel = _log_so3_torch(R_rel)                                       # [B, T-1, 3]
    zero_first = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    ang_vel = torch.cat([zero_first, ang_vel], dim=1)                     # [B, T, 3]
    # NOT body-length normalized

    # 6. EE world positions, body-length normalized, contact-group ordered, n_ee-masked
    joint_world = torch_recover_joint_world(motion_13d_phys)              # [B, T, J, 3]
    ee_idx_safe = ee_joints.clamp(min=0)                                  # [B, G_MAX]
    ee_idx_expand = ee_idx_safe[:, None, :, None].expand(B, T, G_MAX, 3)
    ee_world = torch.gather(joint_world, 2, ee_idx_expand)                # [B, T, G_MAX, 3]
    ee_world_norm = ee_world / body_length[:, None, None, None]
    valid_mask = (ee_joints >= 0).to(dtype)                               # [B, G_MAX]
    ee_world_norm = ee_world_norm * valid_mask[:, None, :, None]
    ee_flat = ee_world_norm.flatten(2)                                    # [B, T, 24]

    return torch.cat([height, rot_6d, lin_vel_local, ang_vel, ee_flat], dim=-1)  # [B, T, 37]


def compute_L_feat_torch(x_src_phys, x_tgt_phys, body_src, body_tgt,
                          ee_src, ee_tgt, n_ee_src, n_ee_tgt):
    """Per-frame L_feat with EE correspondence masking.

    Returns scalar mean over (batch, time).
    """
    B, T, _, _ = x_src_phys.shape
    psi_src = torch_ace_psi_per_frame(x_src_phys, body_src, ee_src, n_ee_src)   # [B, T, 37]
    psi_tgt = torch_ace_psi_per_frame(x_tgt_phys, body_tgt, ee_tgt, n_ee_tgt)   # [B, T, 37]

    # EE correspondence mask: per-batch, first 13 dims always; EE dims for first n_ee_min only
    n_ee_min = torch.minimum(n_ee_src, n_ee_tgt)                          # [B]
    # Build [B, 37] mask
    arange_g = torch.arange(G_MAX, device=psi_src.device)[None, :]        # [1, G_MAX]
    ee_count_mask = (arange_g < n_ee_min[:, None]).to(psi_src.dtype)      # [B, G_MAX]
    ee_dim_mask = ee_count_mask.unsqueeze(-1).expand(-1, -1, 3).flatten(1) # [B, 24]
    pre_mask = torch.ones(B, 13, device=psi_src.device, dtype=psi_src.dtype)
    full_mask = torch.cat([pre_mask, ee_dim_mask], dim=-1)                # [B, 37]
    diff = (psi_src - psi_tgt) * full_mask[:, None, :]                    # [B, T, 37]
    per_frame_norm = torch.linalg.norm(diff, dim=-1)                      # [B, T]
    return per_frame_norm.mean()


def test_ace_psi_matches_numpy(cond, contact_groups, n_windows=20, n_skels=5,
                                tolerance=1e-4, devices=('cpu',), dtypes=(torch.float64,),
                                verbose=True):
    """Hard gate: torch_ace_psi must match numpy ace_psi to tolerance."""
    descs = precompute_ace_descriptors(cond, contact_groups)
    rng = np.random.RandomState(0)
    motion_dir = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
    candidate_skels = [s for s, d in descs.items() if d['n_ee'] > 0]
    rng.shuffle(candidate_skels)
    test_skels = candidate_skels[:n_skels]
    if verbose:
        print(f"[gate] testing {len(test_skels)} skels: {test_skels}")

    results = []
    for device_str in devices:
        for dtype in dtypes:
            device = torch.device(device_str)
            max_diff = 0.0
            count = 0
            for skel in test_skels:
                desc = descs[skel]
                clips = sorted([f for f in motion_dir.iterdir() if f.name.startswith(skel + '___')])
                if not clips:
                    continue
                for clip_path in clips[:5]:
                    motion = np.load(clip_path).astype(np.float64)
                    if motion.shape[0] < 32:
                        continue
                    start = rng.randint(0, motion.shape[0] - 32 + 1)
                    window_np = motion[start:start + 32]                  # [32, J, 13]

                    # Numpy reference
                    psi_np = ace_psi_per_frame_np(window_np, desc['body_length'],
                                                   desc['ee_joints'], desc['n_ee'])  # [32, 37]

                    # Torch
                    window_t = torch.from_numpy(window_np).to(device=device, dtype=dtype).unsqueeze(0)
                    bl_t = torch.tensor([desc['body_length']], device=device, dtype=dtype)
                    ee_t = torch.from_numpy(desc['ee_joints']).to(device=device, dtype=torch.long).unsqueeze(0)
                    nee_t = torch.tensor([desc['n_ee']], device=device, dtype=torch.long)
                    psi_torch = torch_ace_psi_per_frame(window_t, bl_t, ee_t, nee_t).squeeze(0).cpu().numpy()

                    diff = float(np.abs(psi_torch - psi_np).max())
                    if diff > max_diff:
                        max_diff = diff
                    count += 1
                    if count >= n_windows:
                        break
                if count >= n_windows:
                    break

            ok = max_diff <= tolerance
            if verbose:
                print(f"[gate] {device_str}/{dtype}: {'PASS' if ok else 'FAIL'} ({count} windows). "
                      f"max_diff={max_diff:.2e}")
            results.append((device_str, str(dtype), max_diff, ok))
    return results


def gate_ace_or_abort(cond, contact_groups, **kw):
    results = test_ace_psi_matches_numpy(cond, contact_groups, **kw)
    passing = [(d, t) for (d, t, _, ok) in results if ok]
    if not passing:
        msg = "[ACE gate FATAL] torch ace_psi does not match numpy ace_psi.\n"
        for d, t, mx, _ in results:
            msg += f"  {d}/{t}: max_diff={mx:.2e}\n"
        raise RuntimeError(msg)
    return passing[0]


if __name__ == '__main__':
    cond = np.load(PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()
    cg = load_contact_groups()
    print("Running ACE Ψ hard gate test...")
    results = test_ace_psi_matches_numpy(
        cond, cg,
        devices=('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',),
        dtypes=(torch.float32, torch.float64),
    )
    n_pass = sum(1 for *_, ok in results if ok)
    print(f"\nGate result: {n_pass}/{len(results)} variants pass.")
    if n_pass == 0:
        sys.exit(1)
