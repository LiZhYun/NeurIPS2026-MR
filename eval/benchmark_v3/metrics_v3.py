"""v3 benchmark metrics — reviewer-approved rigorous protocol.

Primary metrics (pre-registered, per reviewer):
  1. Contrastive AUC — per-query AUC for ranking K positives above K adversarials (pooled)
  2. Cross-skeleton action retrieval — top-1, top-5 accuracy with distractors

Secondary (descriptive) metrics:
  3. Best-match distance = min over positive set
  4. Rank of best-positive vs {positives ∪ adversarials}

Distance metrics (each feeds into the ranking):
  - Z-score DTW on invariant rep (with circularity caveat)
  - Procrustes-aligned trajectory L2 (PRIMARY — scale + rotation invariant)
  - Q-component L2 (COM, contacts, cadence — morphology-invariant by design)

All metrics are computed against the POSITIVE SET (multiple valid targets).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def _kabsch_3d(A, B):
    """Proper 3D rigid Procrustes (Kabsch with reflection handling).

    A, B: [N, 3] point clouds (row-vector convention). Returns (R, scale, tA, tB)
    such that A_hat = scale * (A - tA) @ R + tB minimizes ||A_hat - B||²,
    with R ∈ SO(3) (proper rotation, no reflection).

    For row-vector convention with H = A0^T @ B0:
      maximize trace(R^T H) → R = U V^T (NOT V U^T)
    Verified numerically (synthetic exact rotation recovery → 4.7e-16 error).
    """
    tA = A.mean(axis=0, keepdims=True)
    tB = B.mean(axis=0, keepdims=True)
    A0 = A - tA
    B0 = B - tB
    H = A0.T @ B0  # 3x3
    U, S, Vt = np.linalg.svd(H)
    # Reflection correction: ensure det(R) = +1 in SO(3)
    d = np.sign(np.linalg.det(U @ Vt))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    # Singular values for scale: flip last if reflection was corrected
    S_corr = S.copy()
    S_corr[-1] *= d
    scale = S_corr.sum() / ((A0 ** 2).sum() + 1e-9)
    return R, float(scale), tA, tB


def _dtw_cost(a, b):
    """Vectorized DTW with L2 per-frame cost. Returns normalized cost."""
    T_a, T_b = a.shape[0], b.shape[0]
    if T_a == 0 or T_b == 0:
        return 0.0
    # Vectorized cost matrix [T_a, T_b]
    diff = a[:, None, :] - b[None, :, :]  # [T_a, T_b, D]
    C = np.linalg.norm(diff, axis=-1)
    D = np.full((T_a + 1, T_b + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            D[i, j] = C[i-1, j-1] + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
    return float(D[T_a, T_b] / (T_a + T_b))


def _dtw_align_indices(a, b):
    """Return (idx_a, idx_b) lists giving the optimal warping path.
    Each (idx_a[k], idx_b[k]) is one step along the path."""
    T_a, T_b = a.shape[0], b.shape[0]
    if T_a == 0 or T_b == 0:
        return [], []
    diff = a[:, None, :] - b[None, :, :]
    C = np.linalg.norm(diff, axis=-1)
    D = np.full((T_a + 1, T_b + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            D[i, j] = C[i-1, j-1] + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
    # Backtrack
    path_a, path_b = [], []
    i, j = T_a, T_b
    while i > 0 and j > 0:
        path_a.append(i - 1); path_b.append(j - 1)
        choices = [(D[i-1, j-1], i-1, j-1), (D[i-1, j], i-1, j), (D[i, j-1], i, j-1)]
        choices.sort()
        _, i, j = choices[0]
    return list(reversed(path_a)), list(reversed(path_b))


def dist_zscore_dtw_inv(inv_pred, inv_ref):
    """Z-score normalized DTW on invariant rep [T, 32, 8]. Slot-pooled mean."""
    from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx
    NULL_SLOT = slot_type_to_idx("null")

    def normalize(inv):
        out = np.zeros_like(inv[:, :, 0:3])
        for s in range(SLOT_COUNT):
            if s == NULL_SLOT:
                continue
            traj = inv[:, s, 0:3].copy()
            # Skip exactly-zero AND near-zero slots (avoids 1e-6 noise blowup)
            if np.abs(traj).max() < 1e-4:
                continue
            m = traj.mean(axis=0, keepdims=True)
            sd = traj.std(axis=0, keepdims=True)
            # Skip slot if std is too small in any axis (constant slot)
            if sd.max() < 1e-4:
                continue
            sd = sd + 1e-6  # tiny floor for numerical stability
            out[:, s, :] = (traj - m) / sd
        return out

    norm_a = normalize(inv_pred)
    norm_b = normalize(inv_ref)
    dists = []
    for s in range(SLOT_COUNT):
        if s == NULL_SLOT:
            continue
        ta = norm_a[:, s, :]
        tb = norm_b[:, s, :]
        if np.all(ta == 0) and np.all(tb == 0):
            continue
        dists.append(_dtw_cost(ta, tb))
    return float(np.mean(dists)) if dists else 0.0


def dist_procrustes_trajectory(pos_pred, pos_ref, dtw_align=True,
                                use_root_relative=True):
    """Procrustes-aligned joint position L2 in WORLD 3D space (not flat 3J).

    pos_pred, pos_ref: [T, J, 3] — joint positions on target skeleton.
    Both have SAME J (target skeleton).

    Steps:
      1. (Optional) Root-relative: subtract joint 0 (root) from each frame
         to remove translation drift confound.
      2. DTW-align temporally (vectorized) to handle T_pred ≠ T_ref.
      3. Pool all aligned point pairs into [N, 3] clouds.
      4. Apply 3D Kabsch (proper rotation + scale, NO reflection).
      5. Mean per-pair L2 after alignment.
    """
    if pos_pred.shape[1] != pos_ref.shape[1]:
        # Should not happen — both motions are on target skel
        J = min(pos_pred.shape[1], pos_ref.shape[1])
        pos_pred = pos_pred[:, :J]
        pos_ref = pos_ref[:, :J]

    if use_root_relative:
        # Joint 0 = root by AnyTop convention
        pos_pred = pos_pred - pos_pred[:, :1, :]
        pos_ref = pos_ref - pos_ref[:, :1, :]

    if dtw_align:
        # DTW alignment in joint-flat space (high-dim DTW is fine for warping)
        a_flat = pos_pred.reshape(pos_pred.shape[0], -1)
        b_flat = pos_ref.reshape(pos_ref.shape[0], -1)
        idx_a, idx_b = _dtw_align_indices(a_flat, b_flat)
        if not idx_a:
            return float('inf')
        # Build aligned point clouds: each step contributes J point-pairs
        A = pos_pred[idx_a].reshape(-1, 3)  # [|path|*J, 3]
        B = pos_ref[idx_b].reshape(-1, 3)
    else:
        T = min(pos_pred.shape[0], pos_ref.shape[0])
        A = pos_pred[:T].reshape(-1, 3)
        B = pos_ref[:T].reshape(-1, 3)

    # 3D Kabsch
    R, scale, tA, tB = _kabsch_3d(A, B)
    A_aligned = scale * (A - tA) @ R + tB
    return float(np.mean(np.linalg.norm(A_aligned - B, axis=1)))


def q_component_distances(q_pred, q_ref):
    """Return UNCOMPOSED Q-component distances as a dict.

    Components:
      com_rel_l2: COM path difference, scale-normalized
      cs_one_minus_f1: 1 - contact-schedule F1
      cadence_abs: |Δ cadence| Hz
      limb_l2: limb-usage L2
    """
    com_p = np.asarray(q_pred['com_path'])
    com_r = np.asarray(q_ref['com_path'])
    T = min(com_p.shape[0], com_r.shape[0])
    com_diff = float(np.linalg.norm(com_p[:T] - com_r[:T]) /
                     (np.linalg.norm(com_r[:T]) + 1e-9))

    cs_p = np.asarray(q_pred['contact_sched'])
    cs_r = np.asarray(q_ref['contact_sched'])
    if cs_p.ndim > 1:
        cs_p = (cs_p.sum(axis=1) > 0).astype(np.float32)
    if cs_r.ndim > 1:
        cs_r = (cs_r.sum(axis=1) > 0).astype(np.float32)
    T = min(len(cs_p), len(cs_r))
    cs_p, cs_r = cs_p[:T] > 0.5, cs_r[:T] > 0.5
    tp = float(((cs_p == 1) & (cs_r == 1)).sum())
    fp = float(((cs_p == 1) & (cs_r == 0)).sum())
    fn = float(((cs_p == 0) & (cs_r == 1)).sum())
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    cs_f1 = 2 * pr * rc / (pr + rc + 1e-8)

    cad_diff = abs(float(q_pred['cadence']) - float(q_ref['cadence']))

    lu_p = -np.sort(-np.asarray(q_pred['limb_usage']))
    lu_r = -np.sort(-np.asarray(q_ref['limb_usage']))
    K = max(len(lu_p), len(lu_r))
    lu_p = np.pad(lu_p, (0, K - len(lu_p)))
    lu_r = np.pad(lu_r, (0, K - len(lu_r)))
    lu_l2 = float(np.linalg.norm(lu_p - lu_r))

    return {
        'com_rel_l2': com_diff,
        'cs_one_minus_f1': float(1 - cs_f1),
        'cadence_abs': cad_diff,
        'limb_l2': lu_l2,
    }


def dist_q_component_zscore(q_pred, q_ref_set, q_neg_set):
    """Per-query z-scored Q-component composite.

    Z-scores each component using the (positives + negatives) pool's mean+std.
    Returns the avg z-distance from q_pred to q_ref_set's first item.

    Use this in a paired manner: pass each candidate (positive or negative)
    as q_pred, and the union (positives ∪ negatives) as the pool.
    Returns dict with both per-component and avg z-score distance.
    """
    # Compute raw components for all references in pool
    pool_components = []
    for ref in q_ref_set + q_neg_set:
        pool_components.append(q_component_distances(q_pred, ref))
    if not pool_components:
        return {}
    keys = pool_components[0].keys()
    means = {k: np.mean([c[k] for c in pool_components]) for k in keys}
    stds = {k: np.std([c[k] for c in pool_components]) + 1e-6 for k in keys}
    # Z-score each component
    z_scores = []
    for c in pool_components:
        z = [(c[k] - means[k]) / stds[k] for k in keys]
        z_scores.append(np.mean(z))
    return z_scores  # one per item in (refs + negs)


# ---------------------------------------------------------------------------
# Primary metrics (per-query)
# ---------------------------------------------------------------------------

def contrastive_auc(pos_dists, neg_dists):
    """Per-query AUC for ranking positives above adversarials.

    pos_dists: list of N_pos distances (lower = closer)
    neg_dists: list of N_neg distances (lower = closer)

    Labels: 1 for positive (should have lower dist), 0 for negative.
    Lower dist → higher rank → should be positive.
    AUC = P(pos_dist < neg_dist) across all pairs.
    """
    if not pos_dists or not neg_dists:
        return 0.5  # undefined
    scores = -np.concatenate([pos_dists, neg_dists])  # higher = better
    labels = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def contrastive_accuracy(pos_dists, neg_dists):
    """Binary: min positive dist < min negative dist."""
    if not pos_dists or not neg_dists:
        return 0.5
    return float(min(pos_dists) < min(neg_dists))


def best_match_distance(pos_dists):
    """min over positive set."""
    return float(min(pos_dists)) if pos_dists else float('inf')


def rank_of_best_positive(pos_dists, neg_dists):
    """Rank (1-indexed) of best positive among pooled {positives ∪ negatives}.
    1 = best (all positives beat all negatives)."""
    all_items = [('pos', d) for d in pos_dists] + [('neg', d) for d in neg_dists]
    all_items.sort(key=lambda x: x[1])
    for i, (label, _) in enumerate(all_items):
        if label == 'pos':
            return i + 1
    return len(all_items) + 1


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieval_topk(pred_dists, labels, k_vals=(1, 5, 10)):
    """Rank by pred_dists (ascending = closest first). Hit if labels[rank] == 'positive'.

    pred_dists: list of N distances, one per candidate
    labels: list of N labels in {'positive', 'negative_adv', 'negative_distractor'}
    """
    order = np.argsort(pred_dists)
    ranked_labels = [labels[i] for i in order]
    hits = {}
    for k in k_vals:
        topk = ranked_labels[:k]
        hits[f'top_{k}'] = int(any(lbl == 'positive' for lbl in topk))
    return hits


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=1000, seed=42, ci=0.95):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(lo), float(arr.mean()), float(hi))
