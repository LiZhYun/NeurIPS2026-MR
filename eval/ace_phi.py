"""ACE feature extractor Ψ (numpy reference, paper-faithful per §6.3).

Per-frame components, body-length normalized where dimensional:
  - height (1D) — root world Y, normalized by body length
  - 6D continuous rotation (6D) — from R_world (avoids quaternion sign issue)
  - root linear velocity local (3D) — channels [9,10,11] of root row, body-length normalized
  - root angular velocity (3D) — log_so3(R_t^T @ R_{t+1}), NOT body-length normalized
  - EE world positions (24D = 3 × G_max=8) — body-length normalized, contact-group ordered

Total per-frame: 37D.

L_feat is per-frame norm averaged over time (NOT mean-then-norm, which collapses temporal info).

References:
  papers/2305.14792.pdf §5.2-5.3, §6.3, §7
  data_loaders/truebones/truebones_utils/motion_process.py:455 (recover_root_quat_and_pos_np)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.motion_process import (
    recover_from_bvh_ric_np, recover_root_quat_and_pos_np,
)
from utils.rotation_conversions import rotation_6d_to_matrix_np
from eval.moreflow_phi import (
    G_MAX, load_contact_groups, select_ee_joints, compute_bone_chain_length,
)


def _recover_root_world(motion_13d):
    """Wraps the canonical numpy reference for root recovery.
    Returns (pos_world [T, 3], R_world [T, 3, 3]). R_world is column-stacked (Quaternions roundtrip).
    """
    root_row = motion_13d[:, 0, :]
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(root_row)
    R_world = r_rot_quat.transforms()
    return r_pos.astype(np.float64), R_world.astype(np.float64)

D_PSI = 1 + 6 + 3 + 3 + 3 * G_MAX   # 1 height + 6 rot6d + 3 lin_vel + 3 ang_vel + 24 EE_world = 37


def matrix_to_rotation_6d_np(R):
    """Inverse of rotation_6d_to_matrix_np (column-stacked convention).

    R: [..., 3, 3]. Returns 6D = first 2 columns flattened: [..., 6].
    """
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def _log_so3_np(R):
    """Differentiable log map of SO(3) → axis-angle. R: [..., 3, 3] → [..., 3]."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    small = np.abs(sin_theta) < 1e-5
    factor = np.where(small, 0.5 + theta**2 / 12.0, theta / (2.0 * sin_theta + 1e-20))
    skew_x = R[..., 2, 1] - R[..., 1, 2]
    skew_y = R[..., 0, 2] - R[..., 2, 0]
    skew_z = R[..., 1, 0] - R[..., 0, 1]
    return np.stack([factor * skew_x, factor * skew_y, factor * skew_z], axis=-1)


def precompute_ace_descriptors(cond_dict, contact_groups_dict):
    """One-time per-skel precomputation. Returns dict skel → {body_length, ee_joints, n_ee, ...}."""
    out = {}
    for skel, c in cond_dict.items():
        offsets = np.asarray(c['offsets'], dtype=np.float64)
        parents = np.asarray(c['parents'], dtype=np.int64)
        cg = contact_groups_dict.get(skel, {})
        ee = select_ee_joints(cg)
        # Body length: sqrt(mean(||offsets[1:]||²)). Same as MoReFlow's limb_scale.
        offset_norms = np.linalg.norm(offsets[1:], axis=-1)
        body_length = float(np.sqrt(np.mean(offset_norms**2)) + 1e-6)
        # EE indices padded to G_MAX
        ee_padded = -np.ones(G_MAX, dtype=np.int64)
        ee_padded[:min(len(ee), G_MAX)] = ee[:G_MAX]
        out[skel] = {
            'body_length': body_length,
            'ee_joints': ee_padded,
            'n_ee': min(len(ee), G_MAX),
            'parents': parents,
            'offsets': offsets,
        }
    return out


def ace_psi_per_frame(motion_13d, body_length, ee_joints, n_ee):
    """Compute ACE Ψ per frame on ONE motion window.

    motion_13d: [T, J, 13] np float (physical units, post un-normalize)
    body_length: float scalar (per-skel)
    ee_joints: np.int64 [G_MAX] (-1 = unused)
    n_ee: int

    Returns: [T, D_PSI=37] np.float64.
    """
    motion_13d = motion_13d.astype(np.float64)
    T, J, _ = motion_13d.shape

    # 1. Recover root world (uses canonical numpy reference)
    pos_world_root, R_world_quat = _recover_root_world(motion_13d)       # [T, 3], [T, 3, 3] (column-conv via Quaternions)

    # 2. Height: root world Y, body-length normalized
    height = pos_world_root[:, 1:2] / body_length                         # [T, 1]

    # 3. 6D rotation from R_world (column convention)
    rot_6d = matrix_to_rotation_6d_np(R_world_quat)                       # [T, 6]

    # 4. Root linear velocity local (channels 9, 10, 11), body-length normalized
    lin_vel_local = motion_13d[:, 0, 9:12] / body_length                  # [T, 3]

    # 5. Root angular velocity per frame: log_so3(R_t^T @ R_{t+1})
    R_rel = np.einsum('tji,tjk->tik', R_world_quat[:-1], R_world_quat[1:])  # [T-1, 3, 3]
    ang_vel = _log_so3_np(R_rel)                                          # [T-1, 3]
    ang_vel = np.concatenate([np.zeros((1, 3), dtype=np.float64), ang_vel], axis=0)  # [T, 3]
    # NOTE: do NOT body-length normalize ang_vel (radians are dimensionless)

    # 6. EE world positions, body-length normalized, contact-group ordered, n_ee-masked
    joint_world = recover_from_bvh_ric_np(motion_13d)                     # [T, J, 3]
    ee_flat = np.zeros((T, 3 * G_MAX), dtype=np.float64)
    for g in range(n_ee):
        j = int(ee_joints[g])
        ee_flat[:, g*3 : g*3+3] = joint_world[:, j, :] / body_length

    return np.concatenate([height, rot_6d, lin_vel_local, ang_vel, ee_flat], axis=-1)  # [T, 37]


def compute_L_feat_np(x_src_phys, x_tgt_phys, src_desc, tgt_desc):
    """Per-frame L_feat: ‖Ψ(x_src) − Ψ(x_tgt)‖ averaged over time. NumPy reference.

    EE-correspondence: only first min(n_ee_src, n_ee_tgt) EEs compared (zero-mask others).
    """
    psi_src = ace_psi_per_frame(x_src_phys, src_desc['body_length'],
                                 src_desc['ee_joints'], src_desc['n_ee'])  # [T, 37]
    psi_tgt = ace_psi_per_frame(x_tgt_phys, tgt_desc['body_length'],
                                 tgt_desc['ee_joints'], tgt_desc['n_ee'])  # [T, 37]
    n_ee_min = min(src_desc['n_ee'], tgt_desc['n_ee'])
    # Build mask: first 13 dims always compared; EE dims compared only for n_ee_min entries
    mask = np.ones(D_PSI, dtype=np.float64)
    mask[13 + 3*n_ee_min:] = 0.0
    diff = (psi_src - psi_tgt) * mask[None, :]
    per_frame_norm = np.linalg.norm(diff, axis=-1)                        # [T]
    return float(per_frame_norm.mean())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--skel', default='Horse')
    args = ap.parse_args()

    cond = np.load(PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()
    cg = load_contact_groups()
    descs = precompute_ace_descriptors(cond, cg)
    desc = descs[args.skel]
    print(f"[{args.skel}] body_length={desc['body_length']:.4f}, n_ee={desc['n_ee']}, "
          f"ee_joints={desc['ee_joints'][:desc['n_ee']]}")

    motion_dir = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
    clips = sorted([f for f in motion_dir.iterdir() if f.name.startswith(args.skel + '___')])
    motion = np.load(clips[0]).astype(np.float32)[:32]
    print(f"  Sample clip: {clips[0].name}, shape={motion.shape}")

    psi = ace_psi_per_frame(motion, desc['body_length'], desc['ee_joints'], desc['n_ee'])
    print(f"  Ψ per-frame shape: {psi.shape} (expect [32, 37])")
    print(f"  Per-component slices (first frame):")
    print(f"    height={psi[0, 0]:.4f}, rot_6d={psi[0, 1:7]}")
    print(f"    lin_vel={psi[0, 7:10]}, ang_vel={psi[0, 10:13]}")
    print(f"    EE_world (first 3): {psi[0, 13:16]}")

    # Self-similarity sanity: L_feat between motion and itself should be ~0
    L = compute_L_feat_np(motion, motion, desc, desc)
    print(f"  L_feat(self, self) = {L:.6e} (expect ~0)")
