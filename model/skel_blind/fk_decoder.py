"""Forward-kinematics baseline decoder.

This is a placeholder that writes per-slot positions back onto their
assigned joints. It does NOT solve IK; Plan C adds the
differentiable Gauss-Newton IK layer (§2.3 of the spec). Here we only
need a pass-through decoder so the rest of the pipeline can be tested.

Rotations are left zero, velocities and contacts are copied from the
slot via the assignment mapping. Callers must not treat FK output as
kinematically consistent — use only for encoder round-trip tests and
pipeline scaffolding.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from model.skel_blind.slot_assign import assign_joints_to_slots
from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx


def fk_decode(invariant_rep: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Write invariant slot positions back to per-joint positions.

    Args:
        invariant_rep: [T, SLOT_COUNT, 8] output of encode_motion_to_invariant.
        skel_cond: AnyTop cond entry.

    Returns:
        motion: [T, J, 13] with positions filled from slot positions;
        other channels left at zero except velocity (4:7) and contact (3).
    """
    assert invariant_rep.ndim == 3 and invariant_rep.shape[1] == SLOT_COUNT
    T, _, C = invariant_rep.shape
    assert C >= 3

    J = len(skel_cond["joints_names"])
    out = np.zeros((T, J, 13), dtype=np.float32)

    asg = assign_joints_to_slots(skel_cond)
    null_idx = slot_type_to_idx("null")
    for slot_idx, joint_list in asg.slot_to_joints.items():
        if slot_idx == null_idx:
            continue
        for j in joint_list:
            if j < J:
                out[:, j, 0:3] = invariant_rep[:, slot_idx, 0:3]
                if C >= 7:
                    out[:, j, 9:12] = invariant_rep[:, slot_idx, 4:7]  # velocity
                if C >= 4:
                    out[:, j, 12] = invariant_rep[:, slot_idx, 3]       # contact
    return out
