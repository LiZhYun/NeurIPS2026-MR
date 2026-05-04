"""MoReFlow Stage B condition descriptors Φ_c (torch, differentiable).

Mirrors eval/moreflow_phi.py exactly, with all ops differentiable for L_feat training.
HARD GATE: must match numpy phi() to 1e-4 (machine epsilon for fp64; ~1e-6 for fp32).

References:
  eval/moreflow_phi.py — numpy reference (same five conditions)
  data_loaders/truebones/truebones_utils/motion_process.py:493 — recover_from_bvh_ric_np
  utils/rotation_conversions.py:536 — rotation_6d_to_matrix_np (column convention; we mirror it in torch)
  arXiv:2509.25600 Appendix A.4

KEY: rotation_6d_to_matrix (torch) and rotation_6d_to_matrix_np use TRANSPOSED conventions.
We define `rotation_6d_to_matrix_columnwise` here to match numpy's column convention.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.moreflow_phi import (
    CONDITIONS, G_MAX, D_COND_PADDED,
    phi as phi_np,
    precompute_skel_descriptors,
    load_contact_groups,
)


# -------------------------------------------------------------------- rotation matrix (column convention, matches numpy)


def rotation_6d_to_matrix_columnwise(d6):
    """Mirror of rotation_6d_to_matrix_np (returns column-stacked matrix).

    d6: [..., 6]
    Returns: [..., 3, 3] with columns [x, y, z] where:
      x = normalize(d6[..., 0:3])
      z = normalize(cross(x, d6[..., 3:6]))
      y = cross(z, x)
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # Stack as columns: [x, y, z] along last dim
    mat = torch.stack([x, y, z], dim=-1)  # [..., 3, 3]
    return mat


# -------------------------------------------------------------------- torch recovery for EE_world / root_XY


def torch_recover_root_world(motion_13d):
    """Differentiable port of recover_root_quat_and_pos_np.

    Mirrors motion_process.py:455 exactly. `-q * v` in numpy = `R^T @ v` in matrix terms
    (where R is the column-convention matrix returned by Quaternions.transforms()).

    motion_13d: [..., T, J, 13] — full motion (uses root row [..., 0, :])
    Returns: pos_world_root [..., T, 3]
    """
    root_row = motion_13d[..., 0, :]                                      # [..., T, 13]
    R_world = rotation_6d_to_matrix_columnwise(root_row[..., 3:9])         # [..., T, 3, 3]

    leading_shape = root_row.shape[:-2]
    T = root_row.shape[-2]
    device = root_row.device
    dtype = root_row.dtype
    zero_first = torch.zeros(*leading_shape, 1, device=device, dtype=dtype)
    x_vel_shifted = torch.cat([zero_first, root_row[..., :-1, 9]], dim=-1)
    z_vel_shifted = torch.cat([zero_first, root_row[..., :-1, 11]], dim=-1)
    y_zero = torch.zeros_like(x_vel_shifted)
    xz_local = torch.stack([x_vel_shifted, y_zero, z_vel_shifted], dim=-1)  # [..., T, 3]

    # numpy `-q * v` = R^T @ v (R is column convention).
    # einsum '...tji,...tj->...ti' = (R^T) @ v per frame
    xz_world = torch.einsum('...tji,...tj->...ti', R_world, xz_local)
    pos_world_xz = torch.cumsum(xz_world, dim=-2)

    y_from_channel1 = root_row[..., 1]                                     # [..., T]
    pos_world = torch.stack([pos_world_xz[..., 0],
                              y_from_channel1,
                              pos_world_xz[..., 2]], dim=-1)
    return pos_world


def torch_recover_joint_world(motion_13d):
    """Differentiable port of recover_from_bvh_ric_np.

    Numpy reference (motion_process.py:493):
      positions = -r_rot_quat * ric_pos        # = R^T @ ric_pos
      positions[..., 0] += r_pos[..., 0:1]
      positions[..., 2] += r_pos[..., 2:3]
      concat root + non-root

    motion_13d: [..., T, J, 13]
    Returns: pos_world [..., T, J, 3]
    """
    pos_world_root = torch_recover_root_world(motion_13d)                 # [..., T, 3]
    R_world = rotation_6d_to_matrix_columnwise(motion_13d[..., 0, 3:9])   # [..., T, 3, 3]

    ric_pos = motion_13d[..., 1:, :3]                                      # [..., T, J-1, 3]
    # `-q * vec` = `R^T @ vec`
    pos_world_nonroot_local = torch.einsum('...tji,...tnj->...tni', R_world, ric_pos)
    nonroot_x = pos_world_nonroot_local[..., 0] + pos_world_root[..., None, 0]
    nonroot_y = pos_world_nonroot_local[..., 1]
    nonroot_z = pos_world_nonroot_local[..., 2] + pos_world_root[..., None, 2]
    pos_world_nonroot = torch.stack([nonroot_x, nonroot_y, nonroot_z], dim=-1)
    pos_world = torch.cat([pos_world_root.unsqueeze(-2), pos_world_nonroot], dim=-2)
    return pos_world


def _log_so3_torch(R):
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    small = torch.abs(sin_theta) < 1e-5
    factor = torch.where(small, 0.5 + theta**2 / 12.0, theta / (2.0 * sin_theta + 1e-20))
    skew_x = R[..., 2, 1] - R[..., 1, 2]
    skew_y = R[..., 0, 2] - R[..., 2, 0]
    skew_z = R[..., 1, 0] - R[..., 0, 1]
    return torch.stack([factor * skew_x, factor * skew_y, factor * skew_z], dim=-1)


# -------------------------------------------------------------------- Φ descriptors (torch, batched)


def torch_phi(motion_13d_physical, c_type, ee_joints, bone_lengths, n_ee):
    """Compute Φ_c for a BATCH of motion windows.

    motion_13d_physical: [B, T, J, 13]
    c_type: one of CONDITIONS
    ee_joints: [B, G_MAX] LongTensor (-1 = unused)
    bone_lengths: [B, G_MAX] FloatTensor (1.0 = unused; safe for division)
    n_ee: [B] LongTensor — actual number of EEs (used for masking)
    Returns: [B, D_COND_PADDED=24] FloatTensor.
    """
    B, T, J, _ = motion_13d_physical.shape
    device = motion_13d_physical.device
    dtype = motion_13d_physical.dtype
    valid_mask = (ee_joints >= 0).to(dtype)                                # [B, G_MAX]
    out = torch.zeros(B, D_COND_PADDED, device=device, dtype=dtype)

    if c_type == 'root_vel':
        lin_vel = motion_13d_physical[:, :, 0, 9:12].mean(dim=1)           # [B, 3]
        R_world = rotation_6d_to_matrix_columnwise(motion_13d_physical[:, :, 0, 3:9])  # [B, T, 3, 3]
        # Per-frame relative R_t^T @ R_{t+1}
        R_rel = torch.einsum('btji,btjk->btik', R_world[:, :-1], R_world[:, 1:])
        ang_vel = _log_so3_torch(R_rel).mean(dim=1)                        # [B, 3]
        out[:, :3] = lin_vel
        out[:, 3:6] = ang_vel
        return out

    elif c_type == 'EE_local':
        # ric_pos channels 0:3 already in root-anchored frame
        ee_idx_safe = ee_joints.clamp(min=0)                                # [B, G_MAX]
        ee_idx_expand = ee_idx_safe[:, None, :, None].expand(B, T, G_MAX, 3)
        ee_ric = torch.gather(motion_13d_physical[..., :3], 2, ee_idx_expand)  # [B, T, G_MAX, 3]
        # Per-EE bone normalization
        ee_ric = ee_ric / bone_lengths[:, None, :, None].clamp(min=1e-6)
        ee_mean = ee_ric.mean(dim=1)                                        # [B, G_MAX, 3]
        ee_mean = ee_mean * valid_mask[:, :, None]
        return ee_mean.flatten(1)

    elif c_type == 'EE_world':
        joint_world = torch_recover_joint_world(motion_13d_physical)        # [B, T, J, 3]
        ee_idx_safe = ee_joints.clamp(min=0)
        ee_idx_expand = ee_idx_safe[:, None, :, None].expand(B, T, G_MAX, 3)
        ee_world = torch.gather(joint_world, 2, ee_idx_expand)             # [B, T, G_MAX, 3]
        ee_mean = ee_world.mean(dim=1)
        ee_mean = ee_mean * valid_mask[:, :, None]
        return ee_mean.flatten(1)

    elif c_type == 'root_XY':
        joint_world = torch_recover_joint_world(motion_13d_physical)        # [B, T, J, 3]
        root_xy = joint_world[:, :, 0, :][..., [0, 2]]                      # [B, T, 2]
        out[:, :2] = (root_xy - root_xy[:, 0:1]).mean(dim=1)
        return out

    elif c_type == 'root_Z':
        out[:, 0] = motion_13d_physical[:, :, 0, 1].mean(dim=1)
        return out

    else:
        raise ValueError(f"Unknown c_type: {c_type}")


# -------------------------------------------------------------------- HARD GATE


def test_recovery_matches_numpy(cond, contact_groups, n_windows=20, n_skels=5,
                                 tolerance=1e-4, devices=('cpu',), dtypes=(torch.float64,),
                                 verbose=True):
    skel_descs = precompute_skel_descriptors(cond, contact_groups)
    rng = np.random.RandomState(0)
    motion_dir = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'

    candidate_skels = [s for s, d in skel_descs.items() if d['n_ee'] > 0]
    rng.shuffle(candidate_skels)
    test_skels = candidate_skels[:n_skels]
    if verbose:
        print(f"[gate] testing {len(test_skels)} skels: {test_skels}")

    results = []
    for device_str in devices:
        for dtype in dtypes:
            device = torch.device(device_str)
            max_diff_per_c = {c: 0.0 for c in CONDITIONS}
            count = 0
            for skel in test_skels:
                desc = skel_descs[skel]
                clips = sorted([f for f in motion_dir.iterdir() if f.name.startswith(skel + '___')])
                if not clips:
                    continue
                for clip_path in clips[:5]:
                    motion = np.load(clip_path).astype(np.float64)
                    if motion.shape[0] < 32:
                        continue
                    start = rng.randint(0, motion.shape[0] - 32 + 1)
                    window_np = motion[start:start + 32]

                    np_results = {c: phi_np(window_np, c, desc) for c in CONDITIONS}

                    window_t = torch.from_numpy(window_np).to(device=device, dtype=dtype).unsqueeze(0)
                    ee_t = torch.from_numpy(desc['ee_joints']).to(device=device, dtype=torch.long).unsqueeze(0)
                    bl_t = torch.from_numpy(desc['bone_lengths']).to(device=device, dtype=dtype).unsqueeze(0)
                    nee_t = torch.tensor([desc['n_ee']], device=device, dtype=torch.long)

                    for c in CONDITIONS:
                        torch_result = torch_phi(window_t, c, ee_t, bl_t, nee_t).squeeze(0).cpu().numpy()
                        np_padded = np_results[c]  # already padded to D_COND_PADDED
                        diff = float(np.abs(torch_result - np_padded).max())
                        if diff > max_diff_per_c[c]:
                            max_diff_per_c[c] = diff
                    count += 1
                    if count >= n_windows:
                        break
                if count >= n_windows:
                    break

            all_pass = all(d <= tolerance for d in max_diff_per_c.values())
            if verbose:
                status = "PASS" if all_pass else "FAIL"
                print(f"[gate] {device_str}/{dtype}: {status} ({count} windows). "
                      f"Max diffs: " + ", ".join(f"{c}={max_diff_per_c[c]:.2e}" for c in CONDITIONS))
            results.append((device_str, str(dtype), max_diff_per_c, all_pass))
    return results


def gate_or_abort(cond, contact_groups, **kw):
    """Run hard gate. If no variant passes, raise RuntimeError. Else return first passing variant."""
    results = test_recovery_matches_numpy(cond, contact_groups, **kw)
    passing = [(d, t) for (d, t, _, ok) in results if ok]
    if not passing:
        msg = "[gate FATAL] torch_phi does not match numpy phi to tolerance.\nVariants tried:\n"
        for d, t, mx, _ in results:
            msg += f"  {d}/{t}: " + ", ".join(f"{c}={mx[c]:.2e}" for c in CONDITIONS) + "\n"
        raise RuntimeError(msg)
    return passing[0]


# -------------------------------------------------------------------- self-test


if __name__ == '__main__':
    cond = np.load(PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()
    contact_groups = load_contact_groups()
    print("Running hard gate test (numpy ↔ torch Φ)...")
    results = test_recovery_matches_numpy(
        cond, contact_groups,
        devices=('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',),
        dtypes=(torch.float32, torch.float64),
    )
    n_pass = sum(1 for *_, ok in results if ok)
    print(f"\nGate result: {n_pass}/{len(results)} variants pass.")
    if n_pass == 0:
        print("FAIL — would abort training.")
        sys.exit(1)
    else:
        print("OK — training would proceed.")
