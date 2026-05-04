"""Phase consistency: cosine-similarity-based per-frame agreement.

Score in [0, 1]; 1 = identical phase, 0 = π offset.
"""
from __future__ import annotations

import numpy as np

from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx

NULL_SLOT = slot_type_to_idx("null")


def _has_gait_phase(contact_seq: np.ndarray) -> bool:
    """True if the contact sequence has ≥2 rising edges (meaningful gait cycle)."""
    binary = (contact_seq > 0.5).astype(np.int8)
    rising = np.where(np.diff(binary, prepend=0) > 0)[0]
    return rising.size >= 2


def phase_consistency(inv_pred: np.ndarray, inv_gt: np.ndarray) -> float:
    assert inv_pred.ndim == 3 and inv_gt.ndim == 3
    assert inv_pred.shape[1] == SLOT_COUNT
    T = min(inv_pred.shape[0], inv_gt.shape[0])
    scores = []
    for s in range(SLOT_COUNT):
        if s == NULL_SLOT:
            continue
        pa = inv_pred[:T, s, 7]
        pb = inv_gt[:T, s, 7]
        if np.all(pa == 0) and np.all(pb == 0):
            continue
        # Skip slots where either clip lacks meaningful gait phase
        ca = inv_pred[:T, s, 3]
        cb = inv_gt[:T, s, 3]
        if not (_has_gait_phase(ca) and _has_gait_phase(cb)):
            continue
        ea = np.stack([np.cos(pa), np.sin(pa)], axis=-1)
        eb = np.stack([np.cos(pb), np.sin(pb)], axis=-1)
        cos_sim = (ea * eb).sum(axis=-1)
        scores.append(((cos_sim + 1.0) / 2.0).mean())
    if not scores:
        return 1.0
    return float(np.mean(scores))
