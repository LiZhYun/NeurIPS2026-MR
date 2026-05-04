"""Heuristic contact-group builder for Idea K (M2 extension, 2026-04-14).

Authors contact_groups.json for every Truebones skeleton that is not yet
manually authored. Output merges into eval/quotient_assets/contact_groups.json
while preserving the existing 6 authored entries verbatim.

Heuristic recipe (see CLAUDE.md / idea-stage/PHASE2_6_* for context):

  1. quadruped-4 : L/R front (=Hand/Finger/Forearm/UpperArm) + L/R hind (=Foot/Toe/Thigh/Calf)
  2. biped-2     : single L/R pair (legs); used for humanoids where arms are not ground-contact
  3. biped+wings-4 (flyer-4) : L/R feet + L/R wings
  4. snake-body-N: body segments split by joint depth (front / mid_front / mid_back / tail)
  5. spider-8+claws: 8 L/R legs (L1..L4, R1..R4), optional claw_L/claw_R
  6. other/body  : fallback — central axis joints (spine+head) as single 'body' group

Per the extractor contract, only "contact-candidate" joints go in. Heuristic
candidate substrings: Toe, Foot, Paw, Claw, Finger, Hand, Tip, Nub, Wing,
and lowercase Japanese equivalents (te/ashi) where needed.

Usage:
  conda activate anytop
  python -m eval.build_contact_groups [--out <path>] [--dry-run]
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
DATASET_COND = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
CG_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'

# ----------------------------------------------------------------------------
# String-level helpers
# ----------------------------------------------------------------------------

_SIDE_PATTERNS = {
    'L': re.compile(r'(?:(?:^|_)(?:L|LEFT|Left|left)(?=$|_|\d|[A-Z]))'
                    r'|(?:Elk[L])'  # Deer: ElkLFemur, ElkLScapula
                    r'|(?:Sabrecat_Left)'
                    ),
    'R': re.compile(r'(?:(?:^|_)(?:R|RIGHT|Right|right)(?=$|_|\d|[A-Z]))'
                    r'|(?:Elk[R])'
                    r'|(?:Sabrecat_Right)'
                    ),
}


def detect_side(name: str) -> str | None:
    """Return 'L' / 'R' if a left/right token is present, else None."""
    # Fast path for very common BVH markers
    upper = name
    # Guard against 'Calf' / 'Claw' etc. that contain no L/R tokens.
    # Use regex on boundaries.
    for side, pat in _SIDE_PATTERNS.items():
        if pat.search(upper):
            return side
    return None


# Contact-candidate name substrings (case-insensitive)
_CONTACT_CANDIDATES = [
    'toe', 'foot', 'paw', 'claw', 'finger', 'hand', 'nub', 'tip', 'wing',
    # Japanese / romanised BVH suffixes used by Alligator/Tukan
    '_te', '_ashi', 'momo', 'hiji',
    # Anatomical names used by Deer/Elk skeleton
    'phalangesmanus', 'phalanxprima', 'metacarpus', 'cannon',
    # Scorpion-style
    'frontleg', 'middleleg', 'hindleg',
]

# Joints that count as "front" vs "hind" in quadruped BVH convention.
_FRONT_HINTS = ['hand', 'finger', 'forearm', 'upperarm', 'clavicle', 'wrist',
                'shoulder', 'elbow', 'arm', 'te', 'kata', 'hiji',
                'humerus', 'radius', 'metacarpus', 'phalangesmanus', 'scapula']
_HIND_HINTS = ['foot', 'toe', 'thigh', 'calf', 'horselink', 'knee', 'ankle',
               'hip_', 'leg', 'ashi', 'momo', 'hiza', 'femur', 'tibia',
               'cannon', 'phalanxprima']

# Wing-specific substrings
_WING_HINTS = ['wing']


def _contains_any(name: str, keys: list[str]) -> bool:
    lo = name.lower()
    return any(k in lo for k in keys)


def _is_contact_candidate(name: str) -> bool:
    return _contains_any(name, _CONTACT_CANDIDATES)


def _classify_fh(name: str) -> str | None:
    """Front ('F') vs Hind ('H') classification from name. None if neither."""
    lo = name.lower()
    # Wing is not a front/hind foot
    if 'wing' in lo:
        return None
    hind_hit = any(h in lo for h in _HIND_HINTS)
    front_hit = any(f in lo for f in _FRONT_HINTS)
    if hind_hit and not front_hit:
        return 'H'
    if front_hit and not hind_hit:
        return 'F'
    # Ambiguous (rare) — fall through
    return None


# ----------------------------------------------------------------------------
# Topology helpers
# ----------------------------------------------------------------------------

def _graph_depth(parents: List[int]) -> np.ndarray:
    J = len(parents)
    depth = np.zeros(J, dtype=np.int32)
    # parents are typically topologically ordered (parent idx < child idx).
    for j in range(J):
        p = parents[j]
        if 0 <= p < j:
            depth[j] = depth[p] + 1
    return depth


def _leaves(parents: List[int]) -> set[int]:
    J = len(parents)
    has_child = [False] * J
    for p in parents:
        if 0 <= p < J:
            has_child[p] = True
    return {j for j in range(J) if not has_child[j]}


# ----------------------------------------------------------------------------
# Morphology heuristics
# ----------------------------------------------------------------------------

def _count_leg_chains(kinematic_chains, names):
    """Count kinematic chains that end at a Foot/Toe/Ashi-like joint."""
    n = 0
    for chain in kinematic_chains:
        # ignore single-joint chains
        if len(chain) < 3:
            continue
        tail_name = names[chain[-1]].lower()
        if any(k in tail_name for k in ['toe', 'foot', 'ashi', 'claw', 'finger', 'paw',
                                          'phalangesmanus', 'phalanxprima']):
            n += 1
    return n


def _has_wings(names):
    for n in names:
        if 'wing' in n.lower():
            return True
    return False


def _is_snake(names, parents):
    """Snake: joints are almost entirely spine/tail/head, no L/R limbs."""
    J = len(names)
    n_lr = 0
    for n in names:
        s = detect_side(n)
        if s is not None:
            n_lr += 1
    # Very few L/R tokens and majority of joints are spine/tail/head/body
    backbone_kws = ['spine', 'spline', 'tail', 'neck', 'head', 'body', 'cog',
                    'pelvis', 'mouth', 'tongue', 'jaw', 'hips', 'tone',
                    'sippo', 'koshi']
    n_backbone = sum(1 for n in names if _contains_any(n, backbone_kws))
    return (n_lr <= 2) and (n_backbone / max(J, 1) >= 0.75)


def _count_numbered_leg_pairs(names):
    """Count *distinct* numbered leg chains per side. Returns (left_ids,
    right_ids) sets. Spider-like requires >=3 ids on each side."""
    # Collapse per-leg joints into one by keeping only the FIRST joint of a leg
    # chain. A spider's leg appears as e.g. BN_leg_L_01 (joint 1 of leg 1),
    # BN_leg_L_02 (joint 2 of leg 1). For Crab/HermitCrab the leg numbers can
    # go >4. Normalise by (side, leg_chain_idx) where chain_idx is derived
    # from the leg group pattern.
    # We detect two families:
    #   (a) jt_FrontLeg1_L / jt_HindLeg1_R (Scorpion-2): explicit leg name + idx
    #   (b) Bip01_L_Thigh_2 (Scorpion): L|R side with a numeric suffix on Thigh
    #   (c) BN_leg_L_NN (Spider/SpiderG/Crab): every joint of every leg
    left = set()
    right = set()
    for n in names:
        # Family (a): jt_FrontLeg1_L, jt_MiddleLeg1_L, jt_HindLeg1_L
        m = re.search(r'(Front|Middle|Hind)Leg\d*_(L|R)', n, re.I)
        if m:
            chain = f"{m.group(1).lower()}"
            side = m.group(2).upper()
            (left if side == 'L' else right).add(chain)
            continue
        # Family (b): Bip01_L_Thigh_2 (8-leg scorpion)
        m = re.search(r'(L|R)_Thigh_?(\d+)', n)
        if m:
            side = m.group(1).upper()
            idx = int(m.group(2))
            (left if side == 'L' else right).add(f"thigh{idx}")
            continue
        # Family (c): BN_leg_L_NN — we don't coalesce, but require multiple
        # distinct first-segment indices. Use the joint-number suffix where
        # the hundreds digit is the leg number. For Spider (J=71) the indices
        # are 01..05 for leg 1 etc. We don't have that info reliably, so we
        # rely on *counting distinct chain patterns* in kinematic_chains.
        # Leave (c) handling to the kinematic_chains based detector below.
    return left, right


def _count_leg_chains_by_name(names, kinematic_chains):
    """Count kinematic chains that are leg-like (contain 'leg' token or
    end in a Toe/Foot/Claw/Finger)."""
    n_left = 0
    n_right = 0
    for chain in kinematic_chains:
        if len(chain) < 3:
            continue
        # Does the chain contain a L or R token?
        for j in chain[1:]:  # skip root pelvis
            nm = names[j]
            if 'leg' in nm.lower() and detect_side(nm) == 'L':
                n_left += 1
                break
            if 'leg' in nm.lower() and detect_side(nm) == 'R':
                n_right += 1
                break
    return n_left, n_right


# ----------------------------------------------------------------------------
# Group builders
# ----------------------------------------------------------------------------

def build_quadruped(names):
    """Four groups: LF / RF / LH / RH of contact-candidate joints."""
    groups = {'LF': [], 'RF': [], 'LH': [], 'RH': []}
    for i, n in enumerate(names):
        side = detect_side(n)
        if side is None:
            continue
        fh = _classify_fh(n)
        if fh is None:
            continue
        if not _is_contact_candidate(n):
            continue
        # Wing belongs to wing group, not limb group.
        if 'wing' in n.lower():
            continue
        key = f'{side}{fh}'
        groups[key].append(i)
    # Prune empties
    groups = {k: sorted(v) for k, v in groups.items() if v}
    return groups


def build_biped(names, include_arms=False):
    """Two groups L / R; leg contact-candidates only (unless include_arms)."""
    groups = {'L': [], 'R': []}
    for i, n in enumerate(names):
        side = detect_side(n)
        if side is None:
            continue
        if not _is_contact_candidate(n):
            continue
        if 'wing' in n.lower():
            continue
        fh = _classify_fh(n)
        if fh == 'F' and not include_arms:
            continue
        groups[side].append(i)
    groups = {k: sorted(v) for k, v in groups.items() if v}
    return groups


def build_wings(names):
    """LW / RW wing groups (leaves of each wing chain)."""
    groups = {'LW': [], 'RW': []}
    for i, n in enumerate(names):
        side = detect_side(n)
        if side is None:
            continue
        if 'wing' not in n.lower():
            continue
        groups[f'{side}W'].append(i)
    groups = {k: sorted(v) for k, v in groups.items() if v}
    return groups


def build_flyer4(names):
    """4 groups: biped feet (L/R) + wings (LW/RW)."""
    g = build_biped(names, include_arms=False)
    g.update(build_wings(names))
    return {k: v for k, v in g.items() if v}


def build_flyer_quad(names):
    """Quadruped with wings: LF/RF/LH/RH + LW/RW. (Dragon / Bat-style)"""
    g = build_quadruped(names)
    g.update(build_wings(names))
    return {k: v for k, v in g.items() if v}


def build_spider(names, parents, kinematic_chains):
    """Spider/octopod: L1..L4 / R1..R4, plus optional claw_L / claw_R.

    Heuristic: walk kinematic_chains, group chains that pass through a
    'leg'/'thigh'-like joint with a side token, take the chain's leaf joints.
    Number them by anterior-to-posterior position (using the second joint's
    parent depth as a proxy)."""
    leaves = _leaves(parents)
    # Collect leg-chains as (side, representative_chain_leaf, chain_joint_list)
    leg_chains = []  # list of (side, chain_tuple)
    for chain in kinematic_chains:
        if len(chain) < 3:
            continue
        # Determine side by the first non-root joint on the chain
        side = None
        is_leg = False
        is_claw = False
        for j in chain[1:]:
            nm = names[j]
            lo = nm.lower()
            s = detect_side(nm)
            if s is None:
                continue
            # legs
            if any(k in lo for k in ['leg', 'thigh', 'femur', 'frontleg',
                                       'middleleg', 'hindleg']):
                side = s
                is_leg = True
                break
            # claws / pincers
            if any(k in lo for k in ['pincer', 'piers', 'pliers', 'claw_',
                                       'bigmandible', 'fangs', 'lowermandible',
                                       'crab_pincers']):
                side = s
                is_claw = True
                break
        if is_leg:
            leg_chains.append((side, chain, 'leg'))
        elif is_claw:
            leg_chains.append((side, chain, 'claw'))

    # Number legs per side by the parent-depth of the chain root joint
    groups: Dict[str, List[int]] = {}
    # Separate legs vs claws
    per_side_legs = {'L': [], 'R': []}
    claws = {'L': [], 'R': []}
    for side, chain, kind in leg_chains:
        if side is None:
            continue
        if kind == 'leg':
            per_side_legs[side].append(chain)
        else:
            claws[side].append(chain)
    for side in ['L', 'R']:
        # Sort legs by the chain's root joint index (stable ordering)
        per_side_legs[side].sort(key=lambda c: c[1])
        for i, chain in enumerate(per_side_legs[side], start=1):
            key = f'{side}{i}'
            # Take contact-candidate joints on the chain (leaves or toe/foot)
            picks = [j for j in chain
                     if j in leaves or _is_contact_candidate(names[j])]
            if not picks:
                picks = [chain[-1]]  # last joint of the chain
            groups[key] = sorted(set(picks))
        # Claws
        if claws[side]:
            all_claw = []
            for chain in claws[side]:
                all_claw.extend([j for j in chain if j in leaves
                                    or _is_contact_candidate(names[j])])
            if not all_claw:
                all_claw = [c[-1] for c in claws[side]]
            groups[f'claw_{side}'] = sorted(set(all_claw))

    return {k: v for k, v in groups.items() if v}


def build_snake(names, parents):
    """Snake body segmented by depth: front / mid_front / mid_back / tail."""
    depth = _graph_depth(parents)
    # Only include spine/tail/body joints (backbone), exclude head/eyes.
    backbone_kws = ['spine', 'spline', 'tail', 'body', 'pelvis', 'sippo',
                    'kosi', 'koshi', 'hara', 'mune', 'cog', 'hips']
    idxs = [i for i, n in enumerate(names) if _contains_any(n, backbone_kws)]
    if not idxs:
        return {}
    # Split by depth quartile
    d = depth[idxs]
    q = np.quantile(d, [0.25, 0.5, 0.75])
    groups = {'front': [], 'mid_front': [], 'mid_back': [], 'tail': []}
    for i, di in zip(idxs, d):
        if di <= q[0]:
            groups['front'].append(i)
        elif di <= q[1]:
            groups['mid_front'].append(i)
        elif di <= q[2]:
            groups['mid_back'].append(i)
        else:
            groups['tail'].append(i)
    return {k: sorted(v) for k, v in groups.items() if v}


def build_fallback(names):
    """Fallback 'body' group: central axis joints (Hips/Spine/Neck/Head)."""
    kws = ['pelvis', 'spine', 'neck', 'head', 'hips', 'cog', 'body']
    idxs = [i for i, n in enumerate(names) if _contains_any(n, kws)]
    if not idxs:
        # absolute fallback — use joint 0 only
        return {'body': [0]}
    return {'body': sorted(idxs)}


# ----------------------------------------------------------------------------
# Skeleton-level dispatch
# ----------------------------------------------------------------------------

def classify_morphology(names, parents, kinematic_chains):
    """Return a label from {spider, snake, flyer_quad, flyer_biped,
    quadruped, biped, fallback}."""
    left_legs, right_legs = _count_numbered_leg_pairs(names)
    n_leg_L, n_leg_R = _count_leg_chains_by_name(names, kinematic_chains)
    # Spider-like: >=3 distinct numbered legs on each side (via pattern) OR
    # >=3 leg-named chains on each side
    spider_by_name = len(left_legs) >= 3 and len(right_legs) >= 3
    spider_by_chain = n_leg_L >= 3 and n_leg_R >= 3
    if spider_by_name or spider_by_chain:
        return 'spider'
    if _is_snake(names, parents):
        return 'snake'
    has_wings = _has_wings(names)
    # Determine biped vs quadruped by whether Hand/Finger joints exist on
    # L+R sides AND there's independent Foot/Toe chains with L+R.
    n_leg_chains = _count_leg_chains(kinematic_chains, names)
    # Count unique (side, front/hind) buckets
    buckets = set()
    for n in names:
        side = detect_side(n)
        fh = _classify_fh(n)
        if side and fh and _is_contact_candidate(n) and 'wing' not in n.lower():
            buckets.add((side, fh))
    full_quadruped = {('L', 'F'), ('R', 'F'), ('L', 'H'), ('R', 'H')} <= buckets
    has_biped_legs = {('L', 'H'), ('R', 'H')} <= buckets
    if has_wings:
        if full_quadruped:
            return 'flyer_quad'
        if has_biped_legs or {('L', 'F'), ('R', 'F')} <= buckets:
            return 'flyer_biped'
    if full_quadruped:
        return 'quadruped'
    if has_biped_legs and not full_quadruped:
        # humanoid / biped animal: only hind legs are ground-contact
        return 'biped'
    if has_biped_legs and full_quadruped:
        return 'quadruped'
    # Check for naked quadruped without both-side Hand joints but
    # at least 4 leg-like chains: treat as quadruped
    if n_leg_chains >= 4 and not has_wings:
        # fall through to quadruped attempt
        return 'quadruped'
    if buckets & {('L', 'F'), ('R', 'F')}:
        return 'biped'
    return 'fallback'


def build_for_skeleton(name: str, entry: dict) -> dict:
    names = list(entry['joints_names'])
    parents = list(entry['parents'])
    chains = [list(c) for c in entry['kinematic_chains']]
    label = classify_morphology(names, parents, chains)
    if label == 'spider':
        g = build_spider(names, parents, chains)
    elif label == 'snake':
        g = build_snake(names, parents)
    elif label == 'flyer_quad':
        g = build_flyer_quad(names)
    elif label == 'flyer_biped':
        g = build_flyer4(names)
    elif label == 'quadruped':
        g = build_quadruped(names)
    elif label == 'biped':
        g = build_biped(names, include_arms=False)
    else:
        g = build_fallback(names)

    # If we built a partial/empty structure, fall back to body.
    if not g:
        g = build_fallback(names)
    # Sanity: prune any negative / OOB indices
    J = len(names)
    g = {k: [i for i in v if 0 <= i < J] for k, v in g.items()}
    g = {k: v for k, v in g.items() if v}
    if not g:
        g = build_fallback(names)
    return {'label': label, 'groups': g}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(CG_PATH))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--cond', default=str(DATASET_COND))
    ap.add_argument('--force-rebuild', action='store_true',
                    help='Regenerate entries even when already present. Used '
                         'for QA; does NOT overwrite the Week-1 POC entries '
                         '(Horse/Chicken/Bear/Cat/Bat/Anaconda/Spider).')
    args = ap.parse_args()

    cond = np.load(args.cond, allow_pickle=True).item()
    with open(CG_PATH) as f:
        existing = json.load(f)
    existing_skels = {k for k in existing.keys() if not k.startswith('_')}
    print(f"Existing authored skels ({len(existing_skels)}): {sorted(existing_skels)}")

    # These were hand-authored as Week-1 POC; never overwrite.
    POC_LOCKED = {'Horse', 'Chicken', 'Bear', 'Cat', 'Bat', 'Anaconda', 'Spider'}

    merged = dict(existing)
    label_counts = {}
    authored_now = []
    for skel in sorted(cond.keys()):
        if skel in POC_LOCKED:
            continue
        if skel in existing_skels and not args.force_rebuild:
            continue
        out = build_for_skeleton(skel, cond[skel])
        label = out['label']
        groups = out['groups']
        label_counts[label] = label_counts.get(label, 0) + 1
        merged[skel] = groups
        authored_now.append((skel, label, {k: len(v) for k, v in groups.items()}))
        print(f"  {skel:20s} label={label:12s} groups={ {k: len(v) for k, v in groups.items()} }")

    print("\nLabel histogram:")
    for k, v in sorted(label_counts.items()):
        print(f"  {k:12s}: {v}")

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(merged, f, indent=2)
    n_skels = sum(1 for k in merged if not k.startswith('_'))
    n_meta = len(merged) - n_skels
    print(f"\nWrote {out_path} with {n_skels} skeleton entries "
          f"({n_meta} underscore-keys preserved).")


if __name__ == '__main__':
    main()
