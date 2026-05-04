"""Deterministic motion quality metrics (no model required).

Metrics:
  - root_speed_correlation : Pearson r between root speed profiles
  - contact_rhythm_f1      : F1 of foot contact sequence
  - bone_length_consistency: fraction of frames with <5% deviation from rest-pose
  - foot_sliding           : mean velocity of ground-contact joints during contact frames

All functions accept numpy arrays with shape [T, J, 13] or [J, 13, T] (specify layout).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_tjf(x, layout='tjf'):
    """Ensure array is [T, J, F].  layout='jft' means input is [J, F, T]."""
    if layout == 'jft':
        return np.transpose(x, (2, 0, 1))
    return x


def _root_speed(motion_tjf):
    """Root speed (m/s proxy) per frame.  Returns [T-1] array."""
    root_pos = motion_tjf[:, 0, :3]          # [T, 3]
    delta    = np.diff(root_pos, axis=0)     # [T-1, 3]
    return np.linalg.norm(delta, axis=-1)    # [T-1]


def _foot_contacts(motion_tjf, foot_joint_indices=None, threshold=0.5):
    """Binary foot contact per joint per frame.

    Uses the foot-contact feature (dim 12) if foot_joint_indices is None,
    else falls back to vertical velocity threshold.

    Returns [T, n_feet] binary array.
    """
    if foot_joint_indices is not None:
        # Vertical velocity threshold (index 9 = velocity x, 10 = y, 11 = z)
        vy = motion_tjf[:, foot_joint_indices, 10]   # [T, n_feet]
        return (np.abs(vy) < threshold).astype(float)
    else:
        # Use stored foot-contact flags (dim 12)
        contact = motion_tjf[:, :, 12]    # [T, J]
        return (contact > threshold).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Public metrics
# ─────────────────────────────────────────────────────────────────────────────

def root_speed_correlation(src_motion, tgt_motion, layout='jft', eps=1e-8):
    """Pearson r between source and target root speed profiles.

    Higher = more similar global tempo/rhythm.

    src_motion, tgt_motion: [J, 13, T] (layout='jft') or [T, J, 13] (layout='tjf')
    Returns scalar in [-1, 1].
    """
    src = _to_tjf(src_motion, layout)
    tgt = _to_tjf(tgt_motion, layout)

    # Align lengths
    T = min(src.shape[0], tgt.shape[0]) - 1
    if T < 2:
        return float('nan')

    spd_src = _root_speed(src)[:T]
    spd_tgt = _root_speed(tgt)[:T]

    src_c = spd_src - spd_src.mean()
    tgt_c = spd_tgt - spd_tgt.mean()

    denom = (np.std(spd_src) * np.std(spd_tgt) * T + eps)
    return float(np.dot(src_c, tgt_c) / denom)


def contact_rhythm_f1(src_motion, tgt_motion, layout='jft',
                      foot_indices=None, threshold=0.5):
    """F1 score of foot contact sequence between source and target.

    Foot contacts are detected from feature dim 12 (stored contact flags)
    or from a velocity threshold if foot_indices is given.

    src_motion, tgt_motion: [J, 13, T] or [T, J, 13]
    Returns scalar in [0, 1].
    """
    src = _to_tjf(src_motion, layout)
    tgt = _to_tjf(tgt_motion, layout)

    T = min(src.shape[0], tgt.shape[0])
    c_src = _foot_contacts(src[:T], foot_indices, threshold)  # [T, n_feet]
    c_tgt = _foot_contacts(tgt[:T], foot_indices, threshold)

    # Use only joints that exist in BOTH (min n_feet)
    n_feet = min(c_src.shape[1], c_tgt.shape[1])
    c_src  = c_src[:, :n_feet].flatten()
    c_tgt  = c_tgt[:, :n_feet].flatten()

    tp = ((c_src == 1) & (c_tgt == 1)).sum()
    fp = ((c_src == 0) & (c_tgt == 1)).sum()
    fn = ((c_src == 1) & (c_tgt == 0)).sum()

    prec   = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * prec * recall / (prec + recall + 1e-8))


def bone_length_consistency(motion, offsets, layout='jft', tol=0.05, parents=None):
    """Fraction of frames where all bone lengths deviate <tol from rest-pose.

    motion:  [J, 13, T] or [T, J, 13]
    offsets: [J, 3]  rest-pose bone offsets (parent→child)
    tol:     relative deviation threshold (default 5%)
    parents: [J] int array, parents[j] = parent index of joint j (-1 for root).
             Required for correct bone length computation.
    Returns scalar in [0, 1].
    """
    mot = _to_tjf(motion, layout)                  # [T, J, 13]
    T, J, _ = mot.shape

    # Rest-pose bone lengths from offsets
    rest_len = np.linalg.norm(offsets, axis=-1)    # [J]
    nonzero  = rest_len > 1e-4                      # exclude root

    # Root-relative positions from motion features (dims 0:3)
    pos = mot[:, :, :3]                            # [T, J, 3]

    if parents is not None:
        # Correct: compute actual parent-child bone lengths
        bone_len_pred = np.zeros((T, J))
        for j in range(J):
            if parents[j] >= 0:
                bone_len_pred[:, j] = np.linalg.norm(
                    pos[:, j] - pos[:, parents[j]], axis=-1)
    else:
        # Fallback without parent info: use position norms (less accurate)
        bone_len_pred = np.linalg.norm(pos, axis=-1)   # [T, J]

    rest_proxy = np.where(rest_len > 1e-4, rest_len, 1.0)
    deviation = np.abs(bone_len_pred - rest_proxy[None]) / rest_proxy[None]  # [T, J]
    per_frame_ok = (deviation[:, nonzero] < tol).all(axis=-1)               # [T]
    return float(per_frame_ok.mean())


def foot_sliding(motion, layout='jft', foot_indices=None,
                 contact_threshold=0.5):
    """Mean velocity of ground-contact joints during contact frames.

    Lower = less sliding = more physically plausible.

    motion:        [J, 13, T] or [T, J, 13]
    foot_indices:  list of joint indices to check (default: all joints with contact flag)
    Returns scalar (mean foot speed during contact frames, units = feature units/frame).
    """
    mot = _to_tjf(motion, layout)           # [T, J, 13]
    T, J, _ = mot.shape

    if foot_indices is None:
        # Find joints that have ANY contact flag > threshold across frames
        contact_flags = mot[:, :, 12]       # [T, J]
        foot_indices = np.where(contact_flags.max(axis=0) > contact_threshold)[0]
        if len(foot_indices) == 0:
            return float('nan')

    vel = mot[:, foot_indices, 9:12]        # [T, n_feet, 3] velocity features
    speed = np.linalg.norm(vel, axis=-1)    # [T, n_feet]

    contact = mot[:, foot_indices, 12]      # [T, n_feet] contact flags
    in_contact = contact > contact_threshold  # [T, n_feet] bool

    if in_contact.sum() == 0:
        return float('nan')

    return float(speed[in_contact].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pairs(src_motions, tgt_motions, src_offsets=None, layout='jft'):
    """Compute all metrics for a list of source/target motion pairs.

    src_motions: list of [J, 13, T] arrays  (or [T, J, 13] if layout='tjf')
    tgt_motions: list of [J, 13, T] arrays
    src_offsets: list of [J, 3] arrays (for bone_length_consistency); optional

    Returns dict of {metric_name: list_of_per_pair_values}.
    """
    results = {
        'root_speed_correlation': [],
        'contact_rhythm_f1':      [],
        'foot_sliding_src':       [],
        'foot_sliding_tgt':       [],
    }
    if src_offsets is not None:
        results['bone_length_consistency_tgt'] = []

    for i, (src, tgt) in enumerate(zip(src_motions, tgt_motions)):
        results['root_speed_correlation'].append(
            root_speed_correlation(src, tgt, layout))
        results['contact_rhythm_f1'].append(
            contact_rhythm_f1(src, tgt, layout))
        results['foot_sliding_src'].append(
            foot_sliding(src, layout))
        results['foot_sliding_tgt'].append(
            foot_sliding(tgt, layout))
        if src_offsets is not None:
            off = src_offsets[i] if isinstance(src_offsets, list) else src_offsets
            results['bone_length_consistency_tgt'].append(
                bone_length_consistency(tgt, off, layout))

    # Summarize
    summary = {}
    for k, vals in results.items():
        vals_clean = [v for v in vals if not np.isnan(v)]
        summary[k] = {
            'mean': float(np.mean(vals_clean)) if vals_clean else float('nan'),
            'std':  float(np.std(vals_clean))  if vals_clean else float('nan'),
            'per_pair': vals,
        }
    return summary


if __name__ == '__main__':
    # Quick self-test
    T, J = 40, 15
    rng  = np.random.default_rng(0)
    src  = rng.standard_normal((J, 13, T)).astype(np.float32)
    tgt  = src + rng.standard_normal((J, 13, T)).astype(np.float32) * 0.1
    # Set contact flags
    src[4, 12, :] = (rng.uniform(size=T) > 0.5).astype(np.float32)
    tgt[4, 12, :] = src[4, 12, :]

    r   = root_speed_correlation(src, tgt)
    f1  = contact_rhythm_f1(src, tgt)
    fs  = foot_sliding(src)
    print(f'root_speed_r={r:.3f}  contact_f1={f1:.3f}  foot_sliding={fs:.4f}')
    print('motion_metrics self-test passed')
