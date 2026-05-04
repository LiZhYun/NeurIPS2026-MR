"""Frame-wise contact timing F1 across all non-null slots."""
from __future__ import annotations

import numpy as np

from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx

NULL_SLOT = slot_type_to_idx("null")


def contact_timing_f1(inv_pred: np.ndarray, inv_gt: np.ndarray, thresh: float = 0.5) -> float:
    assert inv_pred.ndim == 3 and inv_gt.ndim == 3
    assert inv_pred.shape[1] == SLOT_COUNT and inv_gt.shape[1] == SLOT_COUNT
    T = min(inv_pred.shape[0], inv_gt.shape[0])
    mask_slots = [s for s in range(SLOT_COUNT) if s != NULL_SLOT]
    pred = (inv_pred[:T, mask_slots, 3] >= thresh).astype(np.int8).ravel()
    gt = (inv_gt[:T, mask_slots, 3] >= thresh).astype(np.int8).ravel()
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    if tp + fp + fn == 0:
        return 1.0 if tn > 0 else 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
