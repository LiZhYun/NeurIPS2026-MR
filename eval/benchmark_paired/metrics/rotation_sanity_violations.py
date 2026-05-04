"""Rotation-sanity violation rate.

The 6D rotation parameterization (Zhou et al., CVPR 2019) encodes a
rotation as two 3D vectors that should be unit-length and mutually
orthogonal. Flag a (frame, joint) pair as 'violating' if magnitude is
outside [0.8, 1.2] OR cosine of vector-pair > 0.3.

NOT a joint-anatomical-limit check — it's a model-output sanity gate.
A real plausibility metric (DeepMimic-style trackability,
arXiv:1804.02717) is deferred to Plan B/C.
"""
from __future__ import annotations

import numpy as np

LOW, HIGH = 0.8, 1.2
ORTHO_TOL = 0.3


def rotation_sanity_violation_rate(motion: np.ndarray) -> float:
    """Fraction of (frame, joint) pairs with malformed 6D rotation."""
    assert motion.ndim == 3 and motion.shape[2] >= 9
    T, J, _ = motion.shape
    if T * J == 0:
        return 0.0
    rot = motion[:, :, 3:9].astype(np.float32)
    v1 = rot[:, :, 0:3]
    v2 = rot[:, :, 3:6]
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    dot = (v1 * v2).sum(axis=-1)
    denom = np.maximum(n1 * n2, 1e-6)
    cos_orth = np.abs(dot / denom)
    viol_mag = (n1 < LOW) | (n1 > HIGH) | (n2 < LOW) | (n2 > HIGH)
    viol_ortho = cos_orth > ORTHO_TOL
    viol = viol_mag | viol_ortho
    return float(viol.mean())
