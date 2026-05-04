"""Bridge: convert Stage-2 IK output (theta, root_pos, positions) to the
Truebones 13-dim-per-joint representation expected by Stage-3 AnyTop
projection.

Per the dataset convention (see ``motion_process.get_motion_features`` and
``get_bvh_cont6d_params``), the 13-dim slot for each joint at each frame is:
    [0:3]   root-relative position (from ``get_rifke``)
    [3:9]   6D rotation (for joint j>=1, stored value is PARENT's rotation;
            for j=0, stored value is root-yaw rotation)
    [9:12]  world velocity rotated into root-yaw frame
    [12]    foot-contact binary

For the root (j=0) slot specifically, ``get_rifke`` zeros the XZ channels and
keeps Y; the [9:12] slot actually holds linear root XZ velocity in channels
[9, 11] (with 10 as padding / root local-frame vel-Y).  ``recover_from_bvh_ric_np``
depends on [9, 11] for the XZ integration.

This bridge mimics that encoding from world-frame positions by:
  * deriving per-frame root yaw from the heading (body-frame forward)
  * filling the ric positions via the equivalent of ``get_rifke``
  * fitting 6D rotations from positions using ``animation_from_positions``
    (this yields per-joint rotation quaternions consistent with the given
    offsets; we then rearrange parent/child like ``get_bvh_cont6d_params``)
  * computing world-velocities (rotated into root-yaw frame)
  * binarising foot contact against the target skeleton's contact groups

A zero-rotation (identity 6D) fallback is available if the Animation IK is
slow or unstable; it won't hurt AnyTop's projection because the prior only
uses 6D as a conditioning signal - positions remain the dominant signal for
contact/COM constraints.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = str(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_EPS = 1e-8
POS_Y_IDX = 1
FOOT_CH_IDX = 12


# -------------------- yaw / quaternion helpers ------------------------
def _yaw_quaternion_from_forward(forward_xz: np.ndarray) -> np.ndarray:
    """Build a quaternion [w, x, y, z] that rotates world forward (-Z or +X)
    to ``target = [0, 0, 1]`` (same semantic as ``get_root_quat``).
    forward_xz: [T, 3] unit vector with y=0.
    Returns quats: [T, 4] (w, x, y, z)."""
    T = forward_xz.shape[0]
    target = np.array([0.0, 0.0, 1.0])[None].repeat(T, 0)
    # Quaternion between two vectors in xz plane rotates about +Y.
    # q = normalize( [1 + dot, cross] )
    dot = (forward_xz * target).sum(axis=-1)                  # [T]
    cross = np.cross(forward_xz, target)                      # [T, 3]
    w = 1.0 + dot                                             # [T]
    q = np.concatenate([w[:, None], cross], axis=-1)          # [T, 4]
    # Handle anti-parallel (dot ~ -1): rotate by pi about Y
    bad = w < 1e-6
    if bad.any():
        q[bad] = np.array([0.0, 0.0, 1.0, 0.0])               # 180° about Y
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + _EPS)
    return q


def _quat_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply quat [..., 4] (w, x, y, z) to vector [..., 3]."""
    w = q[..., 0]
    xyz = q[..., 1:]
    t = 2 * np.cross(xyz, v)
    return v + w[..., None] * t + np.cross(xyz, t)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    q2 = q.copy()
    q2[..., 1:] *= -1
    return q2


def _quat_to_6d_np(q: np.ndarray) -> np.ndarray:
    """Quaternion [..., 4] (w, x, y, z) -> rotation 6D [..., 6] (first two
    columns of rotation matrix: b1 (= R[:,0]) concatenated with b2 (= R[:,1])).

    Matches ``get_6d_rep`` which uses ``Quaternions.rotation_matrix(cont6d=True)``:
    the 6D layout in truebones is the first two columns of the 3x3 rotation
    matrix stacked."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R00 = 1 - 2 * (y * y + z * z)
    R10 = 2 * (x * y + w * z)
    R20 = 2 * (x * z - w * y)
    R01 = 2 * (x * y - w * z)
    R11 = 1 - 2 * (x * x + z * z)
    R21 = 2 * (y * z + w * x)
    col0 = np.stack([R00, R10, R20], axis=-1)
    col1 = np.stack([R01, R11, R21], axis=-1)
    return np.concatenate([col0, col1], axis=-1)  # [..., 6]


# ----------------- root-forward heuristic -----------------------------
def _root_forward_from_positions(positions: np.ndarray, smoothing: float = 0.9
                                  ) -> np.ndarray:
    """Body-frame forward direction per frame, derived from root-centred
    centroid velocity (xz-only) with running smoothing.
    positions: [T, J, 3].
    Returns: [T, 3] unit vector with y=0.
    """
    T = positions.shape[0]
    # Use root XZ trajectory to pick a forward; y-component zeroed.
    root = positions[:, 0, :]
    vel = np.zeros_like(root)
    vel[1:] = root[1:] - root[:-1]
    vel[0] = vel[1]
    vel[:, 1] = 0.0
    running = vel[0].copy()
    forward = np.zeros_like(vel)
    for t in range(T):
        running = smoothing * running + (1.0 - smoothing) * vel[t]
        forward[t] = running
    mag = np.linalg.norm(forward, axis=-1, keepdims=True)
    forward = np.where(mag < _EPS, np.array([[0.0, 0.0, 1.0]]), forward / (mag + _EPS))
    return forward


# ----------------- core bridge ---------------------------------------
def _fit_6d_rotations(positions: np.ndarray, parents, offsets: np.ndarray,
                      iterations: int = 3) -> Optional[np.ndarray]:
    """Fit per-joint 6D rotations from world-positions using
    ``animation_from_positions``. Returns [T, J, 6] or ``None`` on failure.
    """
    try:
        from InverseKinematics import animation_from_positions
        from Quaternions import Quaternions

        T = positions.shape[0]
        pos_np = np.asarray(positions, dtype=np.float64)
        parents_np = np.asarray(parents, dtype=np.int64)
        off_np = np.asarray(offsets, dtype=np.float64)
        anim, sorted_order, _sorted_parents = animation_from_positions(
            pos_np, parents_np, offsets=off_np.copy(), iterations=iterations)
        # anim.rotations: Quaternions [T, J]
        qs = anim.rotations.qs              # [T, J, 4] (w, x, y, z)
        inv_perm = np.zeros_like(sorted_order)
        inv_perm[sorted_order] = np.arange(len(sorted_order))
        qs = qs[:, inv_perm]
        # Convert quaternion array [T, J, 4] -> rotation matrix -> 6D.
        cont6d = _quat_to_6d_np(qs)
        return cont6d.astype(np.float32)
    except Exception as e:
        print(f'[bridge] fit_6d_rotations failed: {e}; falling back to identity 6D')
        return None


def theta_to_motion_13dim(theta: np.ndarray, root_pos: np.ndarray,
                          positions: np.ndarray,
                          target_skel: str, cond: dict,
                          contact_groups: Optional[Dict[str, Dict[str, list]]] = None,
                          foot_height_frac: float = 0.05,
                          fit_rotations: bool = True,
                          fps: int = 30) -> np.ndarray:
    """Convert Stage-2 IK output to the Truebones 13-dim-per-joint format.

    Parameters
    ----------
    theta : [T, J, 3]
        Axis-angle per-frame rotations from Stage 2 (currently unused for 6D
        construction because the joint-rotation convention in the dataset
        stores the parent rotation in each child slot; we refit rotations
        from the positions instead, which matches the dataset pipeline).
    root_pos : [T, 3]
        Root world position from Stage 2.
    positions : [T, J, 3]
        World-frame joint positions from Stage 2 (post-FK).
    target_skel : str
        Target skeleton name (key of cond).
    cond : dict
        cond[target_skel] with 'parents', 'offsets', etc.
    contact_groups : dict or None
        Contact groups JSON; used to identify foot joints for the contact
        channel.  If None, fall back to zeros.
    foot_height_frac : float
        Height (as fraction of body scale) below which a foot joint is
        considered in contact.

    Returns
    -------
    motion : [T, J, 13] float32
    """
    target_cond = cond[target_skel]
    parents = np.asarray(target_cond['parents'], dtype=np.int64)
    offsets = np.asarray(target_cond['offsets'], dtype=np.float32)
    J = offsets.shape[0]
    T = positions.shape[0]
    assert positions.shape[1] == J, (
        f'positions J={positions.shape[1]} vs cond J={J}')

    # Ensure positions are anchored at root: override root with root_pos for
    # consistency (get_rifke subtracts root xz so anchor is irrelevant for
    # other joints, but root Y channel is directly read by AnyTop).
    positions = positions.astype(np.float32).copy()
    if root_pos is not None:
        positions[:, 0, :] = root_pos.astype(np.float32)

    # --- 1. Root yaw quaternion ---
    forward = _root_forward_from_positions(positions)
    r_rot_quat = _yaw_quaternion_from_forward(forward)                   # [T, 4]
    r_rot_quat_inv = _quat_inv(r_rot_quat)

    # --- 2. RIC positions (joints 1..J-1) ---
    ric = positions.copy()
    # Subtract root XZ from all joints.
    ric[..., 0] -= positions[:, 0:1, 0]
    ric[..., 2] -= positions[:, 0:1, 2]
    # Rotate to face Z+: apply root_quat.
    # Per frame, rotate each joint position by r_rot_quat[t].
    q_b = np.broadcast_to(r_rot_quat[:, None, :], (T, J, 4))
    ric = _quat_apply_np(q_b, ric)
    # Root slot: zero XZ, keep Y.  ``get_rifke`` does exactly this.
    ric[:, 0, 0] = 0.0
    ric[:, 0, 2] = 0.0
    # Root Y is the world Y of root.
    ric[:, 0, 1] = positions[:, 0, 1]

    # --- 3. 6D rotations ---
    if fit_rotations:
        cont6d = _fit_6d_rotations(positions, parents, offsets)
    else:
        cont6d = None
    if cont6d is None:
        # Identity fallback.
        cont6d = np.zeros((T, J, 6), dtype=np.float32)
        cont6d[..., 0] = 1.0
        cont6d[..., 4] = 1.0
    # Rearrange per the dataset convention: slot j (j>=1) stores parent's
    # rotation; slot 0 stores the root-yaw 6D rotation.
    cont6d_reordered = np.zeros_like(cont6d)
    for j in range(1, J):
        p = parents[j]
        if 0 <= p < J:
            cont6d_reordered[:, j] = cont6d[:, p]
        else:
            cont6d_reordered[:, j] = cont6d[:, j]
    cont6d_reordered[:, 0] = _quat_to_6d_np(r_rot_quat)                  # [T, 6]

    # --- 4. Velocities: (global_positions[1:] - global_positions[:-1]) rotated by root quat at t+1 ---
    vel = np.zeros((T, J, 3), dtype=np.float32)
    if T >= 2:
        dpos = positions[1:] - positions[:-1]                             # [T-1, J, 3]
        q_t1 = np.broadcast_to(r_rot_quat[1:, None, :], (T - 1, J, 4))
        vel_rot = _quat_apply_np(q_t1, dpos)                              # [T-1, J, 3]
        vel[:-1] = vel_rot
        vel[-1] = vel_rot[-1]

    # Root slot stores linear xz-velocity in (9,11) only; [10] = padding.
    # ``get_motion_features`` stashes ``velocity`` (not ``local_vel``) in the
    # root slot: velocity = r_rot[1:] * (positions[1:, 0] - positions[:-1, 0]).
    # We replicate that here: ``vel[:, 0]`` above is already the root's rotated
    # delta, so channels [9], [11] pick up the X/Z components naturally.

    # --- 5. Foot contact ---
    contact = np.zeros((T, J), dtype=np.float32)
    body_scale = float(np.linalg.norm(offsets, axis=1).sum() + _EPS)
    foot_h = foot_height_frac * body_scale
    foot_joints = []
    if contact_groups is not None and target_skel in contact_groups:
        for grp in contact_groups[target_skel].values():
            for j in grp:
                if 0 <= int(j) < J:
                    foot_joints.append(int(j))
    foot_joints = sorted(set(foot_joints))
    if foot_joints:
        y = positions[:, foot_joints, 1]                                  # [T, F]
        # Velocity magnitude within the horizontal plane, per foot joint.
        horiz_vel = np.zeros_like(y)
        horiz_vel[1:] = np.linalg.norm(
            positions[1:, foot_joints, 0::2] - positions[:-1, foot_joints, 0::2],
            axis=-1)
        horiz_vel[0] = horiz_vel[1]
        is_contact = ((y < foot_h) & (horiz_vel < 0.05 * body_scale)).astype(np.float32)
        contact[:, foot_joints] = is_contact

    # --- 6. Assemble ---
    motion = np.zeros((T, J, 13), dtype=np.float32)
    motion[..., :3] = ric                                                 # root: [0, root_y, 0]
    motion[..., 3:9] = cont6d_reordered
    motion[..., 9:12] = vel                                               # root xz via channels 9, 11
    motion[..., 12] = contact
    return motion


# ------------------ sanity util --------------------------------------
def bridge_diagnostics(motion_13: np.ndarray, target_skel: str, cond: dict) -> dict:
    """Return sanity statistics about a motion produced by the bridge."""
    T, J, F = motion_13.shape
    assert F == 13
    off = np.asarray(cond[target_skel]['offsets'], dtype=np.float32)
    scale = float(np.linalg.norm(off, axis=1).sum() + _EPS)
    ric = motion_13[..., :3]
    rot = motion_13[..., 3:9]
    vel = motion_13[..., 9:12]
    contact = motion_13[..., 12]
    # ric norms relative to body scale; joints j>=1.
    ric_norms = np.linalg.norm(ric[:, 1:], axis=-1) / (scale + _EPS)
    # 6D rotation first-col norms (should all be ~1 after normalization in
    # downstream rotation_6d_to_matrix_np, but raw value should at least be
    # non-degenerate).
    rot_col0_norm = np.linalg.norm(rot[..., :3], axis=-1)
    # Root Y range relative to body scale.
    root_y = motion_13[:, 0, 1]
    return {
        'T': int(T),
        'J': int(J),
        'body_scale': scale,
        'ric_norm_median_over_scale': float(np.median(ric_norms)),
        'ric_norm_p95_over_scale': float(np.percentile(ric_norms, 95)),
        'ric_norm_max_over_scale': float(ric_norms.max()) if ric_norms.size else 0.0,
        'rot_col0_norm_median': float(np.median(rot_col0_norm)),
        'rot_col0_norm_p05': float(np.percentile(rot_col0_norm, 5)),
        'vel_mean_abs': float(np.abs(vel).mean()),
        'vel_max_abs': float(np.abs(vel).max()),
        'root_y_min': float(root_y.min()),
        'root_y_max': float(root_y.max()),
        'root_y_median_over_scale': float(np.median(root_y) / (scale + _EPS)),
        'contact_fraction': float(contact.mean()),
    }


if __name__ == '__main__':
    # Quick smoke-test: round-trip a real motion through the bridge
    # (decode via recover_from_bvh_ric_np and re-encode).
    import json
    cond = np.load(f'{PROJECT_ROOT}/dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()
    with open(f'{PROJECT_ROOT}/eval/quotient_assets/contact_groups.json') as f:
        cg = json.load(f)
    fname = 'PolarBearB___Walk_644.npy'
    skel = 'PolarBearB'
    m = np.load(f'{PROJECT_ROOT}/dataset/truebones/zoo/truebones_processed/motions/{fname}')
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    positions = recover_from_bvh_ric_np(m.astype(np.float32))
    T, J, _ = positions.shape
    theta = np.zeros((T, J, 3), dtype=np.float32)
    root_pos = positions[:, 0, :]
    m_bridge = theta_to_motion_13dim(theta, root_pos, positions, skel, cond, cg)
    print('bridge motion shape:', m_bridge.shape)
    print('diagnostics:', bridge_diagnostics(m_bridge, skel, cond))
    # Compare rotated-back positions.
    pos_rt = recover_from_bvh_ric_np(m_bridge)
    err = np.linalg.norm(pos_rt - positions, axis=-1).mean()
    print(f'round-trip position L1 error (avg): {err:.6f}')
