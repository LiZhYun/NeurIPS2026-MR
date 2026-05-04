"""Validate contact_groups.json against cond.npy.

Loads the per-skeleton joint-name lists from cond.npy and checks that every
skeleton in contact_groups.json (1) exists in the Truebones Zoo dataset,
(2) has all indices within range [0, J), and (3) has >=2 groups OR is listed
in ``_unresolved``.

Run from the project root:
    conda run -n anytop python eval/quotient_assets/validate_contact_groups.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
COND_PATH = REPO_ROOT / "dataset/truebones/zoo/truebones_processed/cond.npy"
JSON_PATH = REPO_ROOT / "eval/quotient_assets/contact_groups.json"


def main() -> int:
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(JSON_PATH, "r") as f:
        groups = json.load(f)

    meta_keys = {k for k in groups if k.startswith("_")}
    unresolved = set(groups.get("_unresolved", []))

    skeletons_in_json = [k for k in groups if k not in meta_keys]
    dataset_skeletons = set(cond.keys())

    errors = []
    summaries = []

    # Skeletons in JSON must exist in dataset.
    for name in skeletons_in_json:
        if name not in dataset_skeletons:
            errors.append(f"[unknown skeleton] {name} not in cond.npy")
            continue

        joint_names = cond[name]["joints_names"]
        J = len(joint_names)

        entry = groups[name]
        if not isinstance(entry, dict):
            errors.append(f"[bad type] {name} entry is not a dict")
            continue

        group_names = list(entry.keys())
        group_count = len(group_names)

        # Every group must have >=1 valid joint index.
        total_joints = 0
        for gname, gidx in entry.items():
            if not isinstance(gidx, list) or len(gidx) == 0:
                errors.append(f"[empty group] {name}.{gname} has no indices")
                continue
            for idx in gidx:
                if not isinstance(idx, int):
                    errors.append(f"[bad index type] {name}.{gname} has non-int {idx!r}")
                elif idx < 0 or idx >= J:
                    errors.append(
                        f"[out-of-range] {name}.{gname} idx {idx} not in [0, {J})"
                    )
                else:
                    total_joints += 1

        if group_count < 2 and name not in unresolved:
            errors.append(
                f"[insufficient groups] {name} has {group_count} groups and is not "
                f"in _unresolved"
            )

        summaries.append(
            (name, J, group_count, total_joints, group_names)
        )

    # Coverage summary.
    covered = set(skeletons_in_json)
    missing = dataset_skeletons - covered
    extra = covered - dataset_skeletons

    print("=" * 70)
    print("Per-skeleton validation summary")
    print("=" * 70)
    for name, J, gcount, nj, gnames in sorted(summaries):
        flag = " (UNRESOLVED)" if name in unresolved else ""
        print(f"  {name:18s} J={J:3d}  groups={gcount:2d}  joints={nj:3d}  "
              f"{gnames}{flag}")

    print()
    print("=" * 70)
    print("Coverage")
    print("=" * 70)
    print(f"Total skeletons in dataset: {len(dataset_skeletons)}")
    print(f"Total skeletons in JSON:    {len(covered)}")
    print(f"Missing from JSON ({len(missing)}): {sorted(missing)}")
    print(f"Extra in JSON ({len(extra)}): {sorted(extra)}")
    print(f"Unresolved ({len(unresolved)}): {sorted(unresolved)}")

    if errors:
        print()
        print("=" * 70)
        print("ERRORS")
        print("=" * 70)
        for err in errors:
            print(" - " + err)
        return 1

    print()
    print("OK: all entries valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
