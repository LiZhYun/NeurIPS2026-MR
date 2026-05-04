"""Skeleton-blind motion generation package (Plan A foundation).

Exposes the invariant motion representation and its FK baseline decoder.
See docs/superpowers/specs/2026-04-16-cross-skel-oral-design.md §2.1.
"""
from model.skel_blind.encoder import (
    encode_motion_to_invariant,
    encode_positions,
    encode_contacts,
    encode_velocity_phase,
    CHANNEL_COUNT,
)
from model.skel_blind.slot_assign import assign_joints_to_slots, AssignmentResult
from model.skel_blind.slot_vocab import (
    SLOT_COUNT,
    ALL_SLOT_TYPES,
    CANONICAL_SLOT_TYPES,
    slot_type_to_idx,
    idx_to_slot_type,
)

__all__ = [
    "encode_motion_to_invariant",
    "encode_positions",
    "encode_contacts",
    "encode_velocity_phase",
    "assign_joints_to_slots",
    "AssignmentResult",
    "SLOT_COUNT",
    "CHANNEL_COUNT",
    "ALL_SLOT_TYPES",
    "CANONICAL_SLOT_TYPES",
    "slot_type_to_idx",
    "idx_to_slot_type",
]
