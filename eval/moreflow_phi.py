"""MoReFlow Stage B condition descriptors Φ_c (numpy reference).

Five condition families per paper Appendix A.4:
  - root_vel:  root-aligned linear+angular velocity (R^6, time-averaged)
  - EE_local:  per-EE position in root-anchored frame, normalized by anchor-to-EE
               bone-chain length (R^{3·G_max}=24, padded with zeros for unused EEs)
  - EE_world:  per-EE world position (R^{24})
  - root_XY:   per-window relative root path (R^2, mean of XY relative to first frame)
  - root_Z:    root height (R^1, time-averaged)

KEY OBSERVATION (avoids the rotation-convention pitfall):
  - 13D layout (per get_bvh_cont6d_params):
      [0:3]  ric (root-relative) position. Already in root-anchored frame for non-root joints.
      [3:9]  6D rotation (per-joint, including root)
      [9:12] root-aligned-frame velocity (per-joint, including root)
      [12]   foot contact
  - root_vel.linear is literally the root row's channels 9,10,11 mean.
  - EE_local is literally the ric_pos of EEs (channels 0:3), no recovery needed.
  - root_Z is literally root row channel 1 mean.
  - Only EE_world and root_XY need full world recovery via recover_from_bvh_ric_np.
  - Angular velocity in root_vel needs per-frame root rotation matrix.

Mirror impl in eval/moreflow_phi_torch.py — both must match to 1e-5 (hard gate).
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
    recover_from_bvh_ric_np,
)
from utils.rotation_conversions import rotation_6d_to_matrix_np

CONDITIONS = ['root_vel', 'EE_local', 'EE_world', 'root_XY', 'root_Z']
G_MAX = 8                      # max EE groups per skel (padded)
D_COND_PADDED = 24             # universal pad-to size for cond_vec input to model

CONTACT_GROUPS_PATH = PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json'


# -------------------------------------------------------------------- helpers


def load_contact_groups():
    with open(CONTACT_GROUPS_PATH) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def select_ee_joints(skel_contact_groups):
    if not skel_contact_groups:
        return np.array([], dtype=np.int64)
    ee_indices = []
    for gname in sorted(skel_contact_groups.keys()):
        joint_list = skel_contact_groups[gname]
        if joint_list:
            ee_indices.append(joint_list[-1])  # deepest = last
    return np.array(ee_indices, dtype=np.int64)


def compute_bone_chain_length(offsets, parents, ee_joint_idx, anchor_idx=0):
    path = []
    j = int(ee_joint_idx)
    seen = set()
    while j != anchor_idx and j != -1:
        if j in seen:
            break
        path.append(j)
        seen.add(j)
        j = int(parents[j])
    return float(sum(np.linalg.norm(offsets[k]) for k in path) + 1e-6)


def precompute_skel_descriptors(cond_dict, contact_groups_dict):
    out = {}
    for skel, c in cond_dict.items():
        cg = contact_groups_dict.get(skel, {})
        ee = select_ee_joints(cg)
        bls = np.ones(G_MAX, dtype=np.float64)
        ee_padded = -np.ones(G_MAX, dtype=np.int64)
        for i, j in enumerate(ee[:G_MAX]):
            ee_padded[i] = j
            bls[i] = compute_bone_chain_length(c['offsets'], c['parents'], j, anchor_idx=0)
        out[skel] = {
            'ee_joints': ee_padded,
            'bone_lengths': bls,
            'parents': np.asarray(c['parents'], dtype=np.int64),
            'offsets': np.asarray(c['offsets'], dtype=np.float64),
            'n_ee': min(len(ee), G_MAX),
        }
    return out


# -------------------------------------------------------------------- log SO(3)


def _log_so3_np(R):
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


# -------------------------------------------------------------------- Φ descriptors


def phi(motion_13d, c_type, skel_desc):
    """Compute Φ_c for ONE motion window.

    motion_13d: [T, J, 13] np float
    c_type: one of CONDITIONS
    Returns: 1-D np.float64 array — root_vel: 6, EE_local/world: 24 (G_MAX*3), root_XY: 2, root_Z: 1
    """
    motion_13d = motion_13d.astype(np.float64)
    T = motion_13d.shape[0]
    ee_idx = skel_desc['ee_joints']
    n_ee = skel_desc['n_ee']
    bone_lengths = skel_desc['bone_lengths']

    if c_type == 'root_vel':
        # Linear: channels 9,10,11 of root row are root-aligned-frame velocity (from encoding).
        # Frame 0 is undefined (no prior frame), so we take frames 1..T-1 — which matches
        # `velocity = (positions[1:] - positions[:-1])` shifted into channels 9-11.
        # Per encoding: vel has T-1 entries, padded to T by repeating last in motion_features.
        # Actually: features = [pos[:-1], rot[:-1], vel, foot] all length T-1; final dim 13.
        # So all channels are aligned at length T-1. We take frames 0..T-1 mean (all valid).
        lin_vel = motion_13d[:, 0, 9:12].mean(axis=0)                    # [3]
        # Angular: per-frame root rotation matrix; relative rotation R_t^T @ R_{t+1}
        R_world = rotation_6d_to_matrix_np(motion_13d[:, 0, 3:9])         # [T, 3, 3]
        R_rel = np.einsum('tji,tjk->tik', R_world[:-1], R_world[1:])      # R_t^T @ R_{t+1}
        ang_vel = _log_so3_np(R_rel).mean(axis=0)                         # [3]
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        out[:6] = np.concatenate([lin_vel, ang_vel])
        return out

    elif c_type == 'EE_local':
        # ric_pos (channels 0:3) is already in root-anchored frame per get_rifke.
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        for g in range(n_ee):
            j = int(ee_idx[g])
            ee_pos = motion_13d[:, j, :3].mean(axis=0) / bone_lengths[g]  # [3]
            out[g*3 : g*3 + 3] = ee_pos
        return out

    elif c_type == 'EE_world':
        joint_world = recover_from_bvh_ric_np(motion_13d)                 # [T, J, 3]
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        for g in range(n_ee):
            j = int(ee_idx[g])
            out[g*3 : g*3 + 3] = joint_world[:, j, :].mean(axis=0)
        return out

    elif c_type == 'root_XY':
        joint_world = recover_from_bvh_ric_np(motion_13d)                 # [T, J, 3]
        root_xy = joint_world[:, 0, [0, 2]]                                # [T, 2]
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        out[:2] = (root_xy - root_xy[0:1]).mean(axis=0)
        return out

    elif c_type == 'root_Z':
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        out[0] = motion_13d[:, 0, 1].mean()
        return out

    else:
        raise ValueError(f"Unknown c_type: {c_type}")


# -------------------------------------------------------------------- self-test


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--skel', default='Horse')
    args = ap.parse_args()

    cond = np.load(PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond, contact_groups)
    desc = skel_descs[args.skel]
    print(f"[{args.skel}] n_ee={desc['n_ee']}, "
          f"ee_joints={desc['ee_joints'][:desc['n_ee']]}, "
          f"bone_lengths={desc['bone_lengths'][:desc['n_ee']]}")

    motion_dir = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
    clips = sorted([f for f in motion_dir.iterdir() if f.name.startswith(args.skel + '___')])
    motion = np.load(clips[0]).astype(np.float32)[:32]
    print(f"  Sample clip: {clips[0].name}, shape={motion.shape}")

    for c in CONDITIONS:
        v = phi(motion, c, desc)
        print(f"  Φ_{c}: shape={v.shape}, mean={v.mean():.4f}, "
              f"std={v.std():.4f}, max={np.abs(v).max():.4f}")
