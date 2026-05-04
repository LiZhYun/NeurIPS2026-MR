"""Canonicalization layer over `eval/quotient_assets/contact_groups.json`.

The source file uses 30+ heterogeneous slot keys (L, R, L1..4, L_arm,
L_hand, L_mid*, tail, front, mid, mid_back, mid_front, all, ...). This
module translates them at load time to the 32-slot vocabulary defined
in `slot_vocab.py`. The source JSON is treated as read-only (audit
trail preserved).

Three skeletons currently flagged `_unresolved` in the source (Pigeon,
Tukan, Pirrana) get explicit per-skeleton overrides authored from
joint-name evidence; the original `all` group is dropped.

See Codex review threads `019d95ca-0391-7382-95b3-3fab430bc4b7`
(canonicalization motivation) and `019d9606-f104-7e61-b003-8db318b2ae6c`
(numeric-leg + snake-segment + Pirrana fixes).
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

from model.skel_blind.slot_vocab import ALL_SLOT_TYPES

_REPO_ROOT = Path(__file__).resolve().parents[2]  # model/skel_blind/<this> -> repo root
_CONTACT_GROUPS_PATH = _REPO_ROOT / "eval/quotient_assets/contact_groups.json"
_COND_PATH = _REPO_ROOT / "dataset/truebones/zoo/truebones_processed/cond.npy"

# Canonical key passthrough set — these source keys are already in vocabulary.
_PASSTHROUGH = {
    "root", "head", "LF", "RF", "LH", "RH", "LW", "RW",
    "claw_L", "claw_R",
    "tail_0", "tail_1", "tail_2", "tail_3",
}

# Generic single-key -> canonical mapping (not skeleton-family-dependent).
_SIMPLE_RENAMES = {
    "L_hand": "claw_L",     # SpiderG manipulator
    "R_hand": "claw_R",
    "L_leg1": "LH",         # SpiderG rear leg
    "R_leg1": "RH",
}

# Numeric leg pairs (L1..N / R1..N) for Spider/Crab/HermitCrab style.
# Resolved DYNAMICALLY per-skeleton: lowest-numbered = FRONT, highest = REAR.
def _resolve_numeric_legs(raw_keys):
    L_nums = sorted(int(k[1:]) for k in raw_keys
                    if len(k) >= 2 and k[0] == "L" and k[1:].isdigit())
    R_nums = sorted(int(k[1:]) for k in raw_keys
                    if len(k) >= 2 and k[0] == "R" and k[1:].isdigit())
    out = {}
    if L_nums:
        out[f"L{L_nums[0]}"] = "LF"
        if len(L_nums) >= 2:
            out[f"L{L_nums[-1]}"] = "LH"
        for i, n in enumerate(L_nums[1:-1]):
            out[f"L{n}"] = f"mid_leg_{2 * i}"
    if R_nums:
        out[f"R{R_nums[0]}"] = "RF"
        if len(R_nums) >= 2:
            out[f"R{R_nums[-1]}"] = "RH"
        for i, n in enumerate(R_nums[1:-1]):
            out[f"R{n}"] = f"mid_leg_{2 * i + 1}"
    return out

# Insect / centipede mid-leg sequences.
_MID_PAIRS = {
    "L_mid": "mid_leg_0",   "R_mid": "mid_leg_1",
    "L_mid1": "mid_leg_2",  "R_mid1": "mid_leg_3",
    "L_mid2": "mid_leg_4",  "R_mid2": "mid_leg_5",
    "L_mid3": "mid_leg_6",  "R_mid3": "mid_leg_7",
}

# Bipedal-like arms (Cricket, Raptor families).
_ARM_PAIRS = {
    "L_arm": "LF", "R_arm": "RF",
}

# Bipeds: only one leg pair -> treat as hind.
_BIPED_PAIRS = {
    "L": "LH", "R": "RH",
}

# Snake / serpent body segments -> tail spine, front-to-back.
# Anaconda has {front, mid_front, mid_back, tail}; KingCobra has
# {front, mid, tail}. The `tail` key MUST go to tail_3 (rear-most) so it
# does not collide with `front` (head-end) at tail_0.
_SNAKE_SEGMENTS = {
    "front": "tail_0",
    "mid_front": "tail_1",
    "mid": "tail_2",
    "mid_back": "tail_2",
    "tail": "tail_3",
}

# Skeletons whose source entry is `{all: [...]}` -- author explicit overrides
# from joint-name evidence (verified 2026-04-16):
#   Pigeon (9j): 0=Hips, 1=RightArm, 2=RightForeArm, 3=RightLeg, 4=LeftLeg,
#                5=Tail01, 6=LeftArm, 7=LeftForeArm, 8=Spine
#   Tukan (18j): 0=Hips, 3=kosi(waist), 4=R_momo(thigh), 6=L_momo, 8=mune(chest),
#                9=R_kata(shoulder), 10=R_hiji(elbow), 11=L_kata, 12=L_hiji,
#                13=kao(face)/14=ago(jaw)
#   Pirrana (21j, fish): 0=N_ALL, 3=atama(head), 4=munabireR, 5=munabireL,
#                15=obire, 16=obireB, 17=obireA, 18=sebire, 19=harabireR, 20=harabireL
_UNRESOLVED_OVERRIDES: Dict[str, Dict[str, List[int]]] = {
    "Pigeon": {
        "root": [0],
        "head": [8],          # Spine -- Pigeon has no explicit head joint
        "LW": [6, 7],         # LeftArm + LeftForeArm = left wing
        "RW": [1, 2],         # RightArm + RightForeArm = right wing
        "LH": [4],            # LeftLeg
        "RH": [3],            # RightLeg
        "tail_0": [5],        # Tail01
    },
    "Tukan": {
        "root": [0],
        "head": [13, 14],     # kao + ago (face + jaw)
        "LW": [11, 12],       # L_kata + L_hiji
        "RW": [9, 10],        # R_kata + R_hiji
        "LH": [6],            # L_momo
        "RH": [4],            # R_momo
        "tail_0": [3, 5],     # kosi (waist) + o (mid-pelvis joint between thighs)
        "tail_1": [8],        # mune (chest)
    },
    "Pirrana": {
        # Pirrana's raw `all` group = [4, 5, 15, 16, 17, 18, 19, 20]
        # = munabireR/L (pectoral fins), obire/B/A (caudal-fin segments),
        # sebire (dorsal fin), harabireR/L (pelvic fins). All 8 covered:
        "root": [0],          # N_ALL
        "head": [3],          # atama (head)
        "LW": [5],            # munabireL (left pectoral fin)
        "RW": [4],            # munabireR (right pectoral fin)
        "LH": [20],           # harabireL (left pelvic fin)
        "RH": [19],           # harabireR (right pelvic fin)
        "tail_0": [18],       # sebire (dorsal fin)
        "tail_1": [16],       # obireB
        "tail_2": [17],       # obireA
        "tail_3": [15],       # obire (main caudal fin)
    },
}


def _build_canonical_for_skel(skel_object_type: str, raw_groups: Dict) -> Dict[str, List[int]]:
    """Translate one skeleton's raw group map to canonical slot keys."""
    out: Dict[str, List[int]] = {}

    # Override path: unresolved skeletons use authored mappings.
    if skel_object_type in _UNRESOLVED_OVERRIDES:
        for k, v in _UNRESOLVED_OVERRIDES[skel_object_type].items():
            out[k] = list(v)
        return out

    src = raw_groups.get(skel_object_type, {})
    raw_keys = [k for k in src if not str(k).startswith("_") and k != "all"]
    numeric_resolution = _resolve_numeric_legs(raw_keys)

    for raw_key, joint_list in src.items():
        if str(raw_key).startswith("_"):
            continue
        if raw_key == "all":
            continue
        canonical = (
            raw_key if raw_key in _PASSTHROUGH else
            _SIMPLE_RENAMES.get(raw_key) or
            numeric_resolution.get(raw_key) or
            _MID_PAIRS.get(raw_key) or
            _ARM_PAIRS.get(raw_key) or
            _BIPED_PAIRS.get(raw_key) or
            _SNAKE_SEGMENTS.get(raw_key)
        )
        if canonical is None:
            # Unknown raw key -- drop silently. Caught by test_no_passthrough_of_raw_keys.
            continue
        out.setdefault(canonical, []).extend(int(j) for j in joint_list)
    # Deduplicate and sort.
    for k in out:
        out[k] = sorted(set(out[k]))
    # Validate.
    for k in out:
        assert k in ALL_SLOT_TYPES, f"{skel_object_type}: produced non-vocab key {k!r}"
    return out


@lru_cache(maxsize=1)
def _build_canonical_contact_groups_cached() -> Dict[str, Dict[str, List[int]]]:
    """Internal cached builder — do NOT call directly from code outside this module.
    Callers must use `load_canonical_contact_groups()` which returns a deep copy.
    """
    with open(_CONTACT_GROUPS_PATH) as f:
        raw = json.load(f)
    cond = np.load(_COND_PATH, allow_pickle=True).item()

    out: Dict[str, Dict[str, List[int]]] = {}
    for skel_name, skel_cond in cond.items():
        ot = skel_cond["object_type"]
        out[ot] = _build_canonical_for_skel(ot, raw)
        if not out[ot]:
            raise RuntimeError(
                f"{ot}: empty canonical groups after translation. "
                f"Raw entry: {raw.get(ot, {})}"
            )
    return out


def load_canonical_contact_groups() -> Dict[str, Dict[str, List[int]]]:
    """Load canonicalized per-skeleton contact groups.

    Returns a deep copy so callers cannot mutate the cached state.
    """
    return copy.deepcopy(_build_canonical_contact_groups_cached())


def canonical_groups_for(object_type: str) -> Dict[str, List[int]]:
    """Convenience wrapper: get canonical groups for one skeleton."""
    g = load_canonical_contact_groups()
    if object_type not in g:
        raise KeyError(f"{object_type!r} not in canonical contact groups")
    return g[object_type]
