"""Invariant motion representation encoder.

Task 4 implements the POSITION channels only. Later tasks extend to
contacts (Task 5), velocity + phase (Task 6), and full integration (Task 7).

Motion layout convention: [T, J, 13] where channels 0:3 = root-relative
position, 3:9 = 6D rotation, 9:12 = velocity, 12 = foot-contact binary.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from model.skel_blind.slot_assign import assign_joints_to_slots
from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx

POS_SLICE_IN = slice(0, 3)  # columns in the input [T,J,13] tensor
POS_SLICE_OUT = slice(0, 3)  # output channels 0..3 of the invariant rep


def encode_positions(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Aggregate per-slot positions from the motion tensor.

    Args:
        motion: [T, J, 13] joint-level motion.
        skel_cond: AnyTop cond entry (must contain 'joints_names', 'parents',
                   'object_type').

    Returns:
        positions: [T, SLOT_COUNT, 3] with each slot carrying the mean
        position of the joints assigned to it; null slots are zero.
    """
    assert motion.ndim == 3 and motion.shape[2] >= 3
    T, J, _ = motion.shape
    pos_joint = motion[:, :, POS_SLICE_IN].astype(np.float32)

    asg = assign_joints_to_slots(skel_cond)
    out = np.zeros((T, SLOT_COUNT, 3), dtype=np.float32)
    null_idx = slot_type_to_idx("null")

    for slot_idx, joint_list in asg.slot_to_joints.items():
        if slot_idx == null_idx:
            continue
        idxs = [j for j in joint_list if j < J]
        if not idxs:
            continue
        out[:, slot_idx, :] = pos_joint[:, idxs, :].mean(axis=1)
    return out


CONTACT_CH_IN = 12  # channel index in the input [T,J,13] tensor


def encode_contacts(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Aggregate per-slot contact schedule from the motion tensor.

    Slot contact = max over joints assigned to the slot (OR-gate).

    Returns: [T, SLOT_COUNT, 1] in [0, 1].
    """
    assert motion.ndim == 3 and motion.shape[2] >= 13
    T, J, _ = motion.shape
    contact_joint = motion[:, :, CONTACT_CH_IN].astype(np.float32)

    asg = assign_joints_to_slots(skel_cond)
    out = np.zeros((T, SLOT_COUNT, 1), dtype=np.float32)
    null_idx = slot_type_to_idx("null")

    for slot_idx, joint_list in asg.slot_to_joints.items():
        if slot_idx == null_idx:
            continue
        idxs = [j for j in joint_list if j < J]
        if not idxs:
            continue
        out[:, slot_idx, 0] = contact_joint[:, idxs].max(axis=1)
    return out


VEL_SLICE_IN = slice(9, 12)


def encode_velocity_phase(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Aggregate per-slot velocity (mean over joints) and derive a per-slot
    phase signal via strike-to-strike linear interpolation.

    Phase is in [0, 2π) per (frame, slot). For each slot we detect contact
    rising edges (heel-strikes); within each consecutive-strike gait cycle
    we linearly interpolate phase 0 → 2π. Outside any cycle (before first
    strike, after last strike, or when fewer than 2 strikes occur on the
    slot) we fall back to a global linear ramp over the clip.

    This is the standard biomechanics gait-cycle phase convention. See
    Codex review thread `019d95ca-0391-7382-95b3-3fab430bc4b7` for the
    derivation rationale; the prior step-with-π-jumps version was rejected
    as noise.

    Returns: [T, SLOT_COUNT, 4] where channels 0..2 = velocity, 3 = phase.
    """
    assert motion.ndim == 3 and motion.shape[2] >= 13
    T, J, _ = motion.shape
    vel_joint = motion[:, :, VEL_SLICE_IN].astype(np.float32)
    contact_joint = motion[:, :, CONTACT_CH_IN].astype(np.float32)

    asg = assign_joints_to_slots(skel_cond)
    out = np.zeros((T, SLOT_COUNT, 4), dtype=np.float32)
    null_idx = slot_type_to_idx("null")

    linear_phase = np.linspace(0.0, 2 * np.pi, T, endpoint=False, dtype=np.float32)

    for slot_idx, joint_list in asg.slot_to_joints.items():
        if slot_idx == null_idx:
            continue
        idxs = [j for j in joint_list if j < J]
        if not idxs:
            continue
        out[:, slot_idx, 0:3] = vel_joint[:, idxs, :].mean(axis=1)

        # Detect rising contact edges → strike frames.
        slot_contact = contact_joint[:, idxs].max(axis=1)
        is_contact = (slot_contact > 0.5).astype(np.int8)
        rising = np.where(np.diff(is_contact, prepend=0) > 0)[0]  # frame indices

        if rising.size < 2:
            out[:, slot_idx, 3] = linear_phase
            continue

        # Strike-to-strike linear interpolation within each gait cycle.
        phase = np.zeros(T, dtype=np.float32)
        # Pre-first-strike: project backwards using first cycle's period.
        first_cycle = rising[1] - rising[0]
        for t in range(rising[0]):
            frac = (t - rising[0]) / max(first_cycle, 1)  # negative
            phase[t] = np.mod(2 * np.pi * frac, 2 * np.pi)
        # In-cycle linear ramps.
        for k in range(len(rising) - 1):
            s, e = rising[k], rising[k + 1]
            cycle_len = max(e - s, 1)
            for t in range(s, e):
                phase[t] = 2 * np.pi * (t - s) / cycle_len
        # Post-last-strike: extend with last cycle's period.
        last_cycle = rising[-1] - rising[-2]
        for t in range(rising[-1], T):
            phase[t] = np.mod(2 * np.pi * (t - rising[-1]) / max(last_cycle, 1), 2 * np.pi)
        out[:, slot_idx, 3] = phase
    return out


CHANNEL_COUNT = 8  # pos(3) + contact(1) + vel(3) + phase(1)


def encode_motion_to_invariant(motion: np.ndarray, skel_cond: Dict[str, Any]) -> np.ndarray:
    """Concatenate per-slot channels into a single invariant representation.

    Returns: [T, SLOT_COUNT, 8] with channel layout:
      - 0:3 position
      - 3:4 contact
      - 4:7 velocity
      - 7:8 phase
    """
    pos = encode_positions(motion, skel_cond)
    con = encode_contacts(motion, skel_cond)
    vp = encode_velocity_phase(motion, skel_cond)
    out = np.concatenate([pos, con, vp[:, :, 0:3], vp[:, :, 3:4]], axis=-1)
    assert out.shape[-1] == CHANNEL_COUNT
    return out
