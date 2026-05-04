"""Channel ablation: which invariant rep channels carry cross-skeleton information?

Zero out each channel group one at a time and measure impact on DTW.
Channel layout: pos(0:3), contact(3:4), vel(4:7), phase(7:8).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.slot_vocab import SLOT_COUNT
from eval.benchmark_paired.metrics import end_effector_dtw, contact_timing_f1, phase_consistency

PAIRS_DIR = "eval/benchmark_paired/pairs"

CHANNEL_GROUPS = {
    "all": None,
    "no_pos": (0, 3),
    "no_contact": (3, 4),
    "no_vel": (4, 7),
    "no_phase": (7, 8),
    "pos_only": {"keep": (0, 3)},
    "contact_only": {"keep": (3, 4)},
}


def ablate(inv, group_spec):
    """Zero out specified channels or keep only specified channels."""
    out = inv.copy()
    if isinstance(group_spec, tuple):
        out[:, :, group_spec[0]:group_spec[1]] = 0
    elif isinstance(group_spec, dict) and "keep" in group_spec:
        keep = group_spec["keep"]
        mask = np.zeros_like(out)
        mask[:, :, keep[0]:keep[1]] = out[:, :, keep[0]:keep[1]]
        out = mask
    return out


def run():
    with open(os.path.join(PAIRS_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    results = {g: {"dtw": [], "f1": [], "phase": []} for g in CHANNEL_GROUPS}
    n_pairs = min(50, len(manifest["pairs"]))

    for p in manifest["pairs"][:n_pairs]:
        data = np.load(os.path.join(PAIRS_DIR, p["pair_file"]))
        inv_a, inv_b = data["inv_a"], data["inv_b"]

        for group_name, spec in CHANNEL_GROUPS.items():
            a = inv_a if spec is None else ablate(inv_a, spec)
            b = inv_b if spec is None else ablate(inv_b, spec)
            results[group_name]["dtw"].append(end_effector_dtw(a, b))
            results[group_name]["f1"].append(contact_timing_f1(a, b))
            results[group_name]["phase"].append(phase_consistency(a, b))

    print(f"=== CHANNEL ABLATION ({n_pairs} pairs) ===")
    print(f"{'Condition':16s}  {'DTW↓':>8s}  {'F1↑':>8s}  {'Phase↑':>8s}")
    summary = {}
    for g in CHANNEL_GROUPS:
        d = np.mean(results[g]["dtw"])
        f = np.mean(results[g]["f1"])
        p = np.mean(results[g]["phase"])
        print(f"{g:16s}  {d:8.4f}  {f:8.4f}  {p:8.4f}")
        summary[g] = {"dtw": round(d, 4), "f1": round(f, 4), "phase": round(p, 4)}

    out_path = "idea-stage/channel_ablation.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
