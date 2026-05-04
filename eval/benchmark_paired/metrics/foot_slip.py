"""Foot-slip rate: mean tangential world-velocity magnitude during contact.

World velocity = root-slot Δposition + per-slot velocity. Tangential =
horizontal components (perpendicular to up_axis, default Y).

Known limitation (Codex 2026-04-16): on Truebones-processed motions
root pos[x,z] are 0 across frames — only pos[y] carries height. Root
delta therefore contributes ~0 to world velocity in current data, so
this metric reduces to "tangential slot-relative velocity during
contact." Plan B/C should add explicit world-translation channel.
"""
from __future__ import annotations

import numpy as np

from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx, idx_to_slot_type


def _foot_slot_ids():
    foot_types = {"LF", "RF", "LH", "RH", "claw_L", "claw_R"}
    foot_types |= {f"mid_leg_{i}" for i in range(16)}
    return [i for i in range(SLOT_COUNT) if idx_to_slot_type(i) in foot_types]


_FOOT_SLOTS = _foot_slot_ids()
_ROOT_IDX = slot_type_to_idx("root")


def foot_slip_rate(inv: np.ndarray, up_axis: int = 1) -> float:
    """Mean tangential world-velocity magnitude across (slot, frame) pairs
    where contact >= 0.5, for foot-like slots.

    up_axis: 0/1/2 for X/Y/Z; default 1 (Y-up). Tangential = the OTHER two axes.
    """
    assert inv.ndim == 3 and inv.shape[1] == SLOT_COUNT
    assert up_axis in (0, 1, 2)
    T = inv.shape[0]
    if T == 0:
        return 0.0
    root_pos = inv[:, _ROOT_IDX, 0:3]
    root_world_vel = np.diff(root_pos, axis=0, prepend=root_pos[:1])
    tangential_axes = [a for a in (0, 1, 2) if a != up_axis]
    slip_vals = []
    for s in _FOOT_SLOTS:
        contact = inv[:, s, 3] >= 0.5
        if not contact.any():
            continue
        slot_rel_vel = inv[:, s, 4:7]
        world_vel = root_world_vel + slot_rel_vel
        tangential = world_vel[:, tangential_axes]
        mag = np.linalg.norm(tangential, axis=-1)
        slip_vals.extend(mag[contact].tolist())
    if not slip_vals:
        return 0.0
    return float(np.mean(slip_vals))
