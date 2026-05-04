"""Slot vocabulary loader and index lookup.

Slot indices 0..31 are fixed at authoring time and must not be
renumbered without an explicit migration (hashes in pre-registered
benchmarks reference these indices).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_VOCAB_PATH = Path(__file__).parent / "config" / "slot_vocabulary.json"


@lru_cache(maxsize=1)
def load_slot_vocabulary() -> Dict:
    with open(_VOCAB_PATH) as f:
        return json.load(f)


_vocab = load_slot_vocabulary()
SLOT_COUNT: int = _vocab["slot_count"]
ALL_SLOT_TYPES: List[str] = [s["type"] for s in _vocab["slots"]]
# Canonical types = everything except mid_leg_* residual slots and null/reserved.
# This matches the spec §2.1 "canonical effector types" + "spine segment" + "global_anchor" grouping.
_RESIDUAL_CATEGORIES = {"residual_limb", "null", "reserved"}
CANONICAL_SLOT_TYPES: List[str] = [
    s["type"] for s in _vocab["slots"] if s["category"] not in _RESIDUAL_CATEGORIES
]
_TYPE_TO_IDX: Dict[str, int] = {s["type"]: s["idx"] for s in _vocab["slots"]}
_IDX_TO_TYPE: Dict[int, str] = {s["idx"]: s["type"] for s in _vocab["slots"]}


def slot_type_to_idx(slot_type: str) -> int:
    if slot_type not in _TYPE_TO_IDX:
        raise KeyError(f"Unknown slot type {slot_type!r}; known: {list(_TYPE_TO_IDX)}")
    return _TYPE_TO_IDX[slot_type]


def idx_to_slot_type(idx: int) -> str:
    if idx not in _IDX_TO_TYPE:
        raise IndexError(f"Slot idx {idx} out of range [0, {SLOT_COUNT})")
    return _IDX_TO_TYPE[idx]
