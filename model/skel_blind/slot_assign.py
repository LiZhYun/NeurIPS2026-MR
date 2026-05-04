"""Joint → slot assignment for AnyTop/Truebones skeletons.

Algorithm:
1. Get the canonical group map for this skeleton via the canonicalization
   layer (`contact_groups_canonical.canonical_groups_for(object_type)`).
   Canonical keys are root/head/LF/RF/LH/RH/LW/RW/claw_L/claw_R/tail_0..3
   plus mid_leg_0..15 for spillover.
2. For joints not covered by any canonical group, assign to
   `mid_leg_0..15` based on left/right bilateral side + graph depth.
3. Unused slots map to the `null` slot type (only via missing assignment;
   null slot itself is not actively assigned for unused-joint cases —
   joints that the heuristic cannot place go to null).

Output is an `AssignmentResult` with:
- `joint_to_slot: dict[int, int]`  (joint-idx → slot-idx)
- `slot_to_joints: dict[int, list[int]]`  (slot-idx → list of joint-idx)

**Note (2026-04-16, code review)**: the head-name fallback in
`assign_joints_to_slots` runs for ~60 of 70 skeletons because
`contact_groups.json` historically tracked only ground-contact joints
and omitted `head`. The fallback is the primary code path for head
assignment, not an exception case. Future work may move head-name
detection into `contact_groups_canonical.py` as a generic post-step
to consolidate where this rule lives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx
from model.skel_blind.contact_groups_canonical import canonical_groups_for


@dataclass
class AssignmentResult:
    joint_to_slot: Dict[int, int] = field(default_factory=dict)
    slot_to_joints: Dict[int, List[int]] = field(default_factory=dict)


def _assign_mid_legs(
    uncovered_joints: List[int], parents: List[int], joint_names: List[str]
) -> Dict[int, int]:
    """Distribute uncovered joints to mid_leg_0..15 slots by bilateral side + depth.

    Heuristic:
    - Compute graph depth of each joint from root.
    - Use joint-name bilaterality hints ('L', 'R', 'left', 'right') or position sign.
    - Interleave left/right assignments into mid_leg_0, mid_leg_1, ... in depth order.
    """
    depth_by_joint: Dict[int, int] = {0: 0}
    for j in range(1, len(parents)):
        p = int(parents[j])
        depth_by_joint[j] = depth_by_joint.get(p, 0) + 1

    left_candidates = []
    right_candidates = []
    mid_candidates = []
    for j in uncovered_joints:
        name = str(joint_names[j]).lower()
        is_left = (
            "_l_" in name
            or name.endswith("_l")
            or name.startswith("l_")
            or "left" in name
        )
        is_right = (
            "_r_" in name
            or name.endswith("_r")
            or name.startswith("r_")
            or "right" in name
        )
        if is_left and not is_right:
            left_candidates.append(j)
        elif is_right and not is_left:
            right_candidates.append(j)
        else:
            mid_candidates.append(j)

    left_candidates.sort(key=lambda j: depth_by_joint.get(j, 99))
    right_candidates.sort(key=lambda j: depth_by_joint.get(j, 99))
    mid_candidates.sort(key=lambda j: depth_by_joint.get(j, 99))

    mapping: Dict[int, int] = {}
    mid_leg_pairs = [(f"mid_leg_{2*i}", f"mid_leg_{2*i+1}") for i in range(8)]
    for i, (l_slot, r_slot) in enumerate(mid_leg_pairs):
        if i < len(left_candidates):
            mapping[left_candidates[i]] = slot_type_to_idx(l_slot)
        if i < len(right_candidates):
            mapping[right_candidates[i]] = slot_type_to_idx(r_slot)

    null_idx = slot_type_to_idx("null")
    for j in mid_candidates:
        mapping[j] = null_idx
    for j in left_candidates[len(mid_leg_pairs):]:
        mapping[j] = null_idx
    for j in right_candidates[len(mid_leg_pairs):]:
        mapping[j] = null_idx
    return mapping


_HEAD_NAME_TOKENS = ("head", "skull", "cranium", "kao")  # 'kao' = face in Japanese (Tukan)


def _find_head_joint(uncovered: List[int], joint_names: List[str]) -> int | None:
    """Return the first uncovered joint whose name contains a head-like token, or None."""
    for j in uncovered:
        name = str(joint_names[j]).lower()
        if any(tok in name for tok in _HEAD_NAME_TOKENS):
            return j
    return None


def assign_joints_to_slots(skel_cond: Dict[str, Any]) -> AssignmentResult:
    """Assign every joint of `skel_cond` to a slot index in [0, SLOT_COUNT).

    Uses `cond['object_type']` for skeleton lookup (per
    `eval/quotient_extractor.py:206` — Truebones convention).
    """
    joint_names = list(skel_cond["joints_names"])
    parents = list(skel_cond["parents"])
    if "object_type" not in skel_cond:
        raise ValueError(
            "skel_cond must carry an 'object_type' key (Truebones cond convention)"
        )
    object_type = skel_cond["object_type"]

    groups = canonical_groups_for(object_type)

    result = AssignmentResult()
    covered_joints = set()
    for slot_type, joint_list in groups.items():
        try:
            slot_idx = slot_type_to_idx(slot_type)
        except KeyError:
            continue
        for j in joint_list:
            j = int(j)
            if j < 0 or j >= len(joint_names):
                continue
            if j in covered_joints:
                continue
            result.joint_to_slot[j] = slot_idx
            result.slot_to_joints.setdefault(slot_idx, []).append(j)
            covered_joints.add(j)

    if 0 not in result.joint_to_slot:
        root_idx = slot_type_to_idx("root")
        result.joint_to_slot[0] = root_idx
        result.slot_to_joints.setdefault(root_idx, []).append(0)
        covered_joints.add(0)

    # If the canonical groups had no `head` key, try to find one by name.
    head_slot_idx = slot_type_to_idx("head")
    if head_slot_idx not in result.slot_to_joints:
        uncovered_so_far = [j for j in range(len(joint_names)) if j not in covered_joints]
        head_j = _find_head_joint(uncovered_so_far, joint_names)
        if head_j is not None:
            result.joint_to_slot[head_j] = head_slot_idx
            result.slot_to_joints.setdefault(head_slot_idx, []).append(head_j)
            covered_joints.add(head_j)

    uncovered = [j for j in range(len(joint_names)) if j not in covered_joints]
    mid_assignments = _assign_mid_legs(uncovered, parents, joint_names)
    for j, slot in mid_assignments.items():
        result.joint_to_slot[j] = slot
        result.slot_to_joints.setdefault(slot, []).append(j)

    assert len(result.joint_to_slot) == len(joint_names)
    return result
