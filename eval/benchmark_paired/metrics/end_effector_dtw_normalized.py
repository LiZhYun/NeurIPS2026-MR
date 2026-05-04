"""Body-scale-normalized end-effector DTW.

Addresses reviewer concern: current DTW compares absolute positions which are
dominated by skeleton scale/cadence rather than motion shape. This variant:
  1. Per-slot z-score normalization over the temporal axis
  2. Per-motion scale normalization by max joint-to-root distance

Returns: mean per-slot DTW on normalized trajectories.
"""
from __future__ import annotations

import numpy as np

from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx

NULL_SLOT = slot_type_to_idx("null")


def _dtw_cost(a, b):
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


def end_effector_dtw_normalized(inv_a: np.ndarray, inv_b: np.ndarray,
                                mode: str = "zscore") -> float:
    """Mean per-slot DTW on SCALE-NORMALIZED trajectories.

    mode:
      'zscore': subtract per-slot-per-motion mean, divide by std
      'scale':  divide by per-motion max-distance-from-origin
      'both':   apply zscore then scale
    """
    assert inv_a.ndim == 3 and inv_b.ndim == 3
    assert inv_a.shape[1] == SLOT_COUNT and inv_b.shape[1] == SLOT_COUNT

    def normalize(inv):
        out = np.zeros_like(inv[:, :, 0:3])
        for s in range(SLOT_COUNT):
            if s == NULL_SLOT:
                continue
            traj = inv[:, s, 0:3].copy()
            if np.all(traj == 0):
                continue
            if mode in ("zscore", "both"):
                m = traj.mean(axis=0, keepdims=True)
                sd = traj.std(axis=0, keepdims=True) + 1e-6
                traj = (traj - m) / sd
            if mode in ("scale", "both"):
                maxd = np.linalg.norm(traj, axis=1).max() + 1e-6
                traj = traj / maxd
            out[:, s, :] = traj
        return out

    norm_a = normalize(inv_a)
    norm_b = normalize(inv_b)

    dists = []
    for s in range(SLOT_COUNT):
        if s == NULL_SLOT:
            continue
        traj_a = norm_a[:, s, :]
        traj_b = norm_b[:, s, :]
        if np.all(traj_a == 0) and np.all(traj_b == 0):
            continue
        dists.append(_dtw_cost(traj_a, traj_b))
    if not dists:
        return 0.0
    return float(np.mean(dists))
