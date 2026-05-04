"""End-effector path DTW distance between two invariant reps.

For each non-null slot, compute the DTW distance between the two
motions' position trajectories. Aggregate across slots by mean.
"""
from __future__ import annotations

import numpy as np

from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx

NULL_SLOT = slot_type_to_idx("null")


def _dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Classic O(T_a * T_b) DTW with L2 per-frame cost. a: [T_a, D], b: [T_b, D]."""
    T_a, T_b = a.shape[0], b.shape[0]
    if T_a == 0 or T_b == 0:
        return 0.0
    D = np.full((T_a + 1, T_b + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            c = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            D[i, j] = c + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return float(D[T_a, T_b] / (T_a + T_b))


def end_effector_dtw(inv_a: np.ndarray, inv_b: np.ndarray) -> float:
    """Mean per-slot DTW distance over slot trajectories."""
    assert inv_a.ndim == 3 and inv_b.ndim == 3
    assert inv_a.shape[1] == SLOT_COUNT and inv_b.shape[1] == SLOT_COUNT
    dists = []
    for s in range(SLOT_COUNT):
        if s == NULL_SLOT:
            continue
        traj_a = inv_a[:, s, 0:3]
        traj_b = inv_b[:, s, 0:3]
        if np.all(traj_a == 0) and np.all(traj_b == 0):
            continue
        dists.append(_dtw_cost(traj_a, traj_b))
    if not dists:
        return 0.0
    return float(np.mean(dists))
