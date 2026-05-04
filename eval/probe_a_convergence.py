"""Probe A: Cross-skeleton output convergence in invariant rep space.

Tests whether the invariant rep converges for the same action across
different source skeletons. If the rep is truly topology-invariant, the
invariant rep of "Walk" should be similar regardless of which skeleton
performs it.

Convergence metric: for each (action, slot) pair, compute mean pairwise DTW
across all source skeletons that have that action. Lower = more convergent.
Compare with a random permutation baseline.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.slot_vocab import SLOT_COUNT, slot_type_to_idx
from eval.benchmark_paired.metrics import end_effector_dtw

NULL_SLOT = slot_type_to_idx("null")
INVARIANT_DIR = "dataset/truebones/zoo/invariant_reps"


def load_invariant_reps():
    with open(os.path.join(INVARIANT_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    clips_by_action = defaultdict(list)
    for skel_name, info in manifest["skeletons"].items():
        npz_path = os.path.join(INVARIANT_DIR, f"{skel_name}.npz")
        data = np.load(npz_path, allow_pickle=True)
        for clip_name in info["clips"]:
            parts = clip_name.split("___")
            if len(parts) >= 2:
                action_part = parts[1]
                action = action_part.rsplit("_", 1)[0].lower()
            else:
                action = clip_name.lower()
            clips_by_action[action].append({
                "skel": skel_name,
                "clip": clip_name,
                "inv": data[clip_name],
            })

    return clips_by_action


def pairwise_dtw(group, max_pairs=20):
    n = len(group)
    if n < 2:
        return []
    rng = np.random.RandomState(42)
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(indices) > max_pairs:
        rng.shuffle(indices)
        indices = indices[:max_pairs]

    dtws = []
    for i, j in indices:
        inv_a = group[i]["inv"]
        inv_b = group[j]["inv"]
        dtws.append(end_effector_dtw(inv_a, inv_b))
    return dtws


def main():
    clips_by_action = load_invariant_reps()

    multi_skel_actions = {
        action: clips for action, clips in clips_by_action.items()
        if len(set(c["skel"] for c in clips)) >= 3
    }

    print(f"Actions with ≥3 skeletons: {len(multi_skel_actions)}")

    all_clips_flat = []
    for clips in multi_skel_actions.values():
        seen = set()
        for c in clips:
            if c["skel"] not in seen:
                seen.add(c["skel"])
                all_clips_flat.append(c)

    results = []
    rng = np.random.RandomState(42)
    for action in sorted(multi_skel_actions):
        clips = multi_skel_actions[action]
        skeletons = list(set(c["skel"] for c in clips))
        one_per_skel = []
        seen = set()
        for c in clips:
            if c["skel"] not in seen:
                seen.add(c["skel"])
                one_per_skel.append(c)

        intra_dtws = pairwise_dtw(one_per_skel, max_pairs=30)
        if not intra_dtws:
            continue

        other_actions = [a for a in multi_skel_actions if a != action]
        cross_dtws = []
        for _ in range(min(30, len(intra_dtws))):
            i = rng.randint(0, len(one_per_skel))
            other_action = other_actions[rng.randint(0, len(other_actions))]
            other_clips = multi_skel_actions[other_action]
            j = rng.randint(0, len(other_clips))
            cross_dtws.append(end_effector_dtw(
                one_per_skel[i]["inv"], other_clips[j]["inv"]))

        result = {
            "action": action,
            "n_skeletons": len(skeletons),
            "n_intra_pairs": len(intra_dtws),
            "intra_dtw": round(float(np.mean(intra_dtws)), 4),
            "cross_dtw": round(float(np.mean(cross_dtws)), 4),
            "ratio": round(float(np.mean(intra_dtws)) / float(np.mean(cross_dtws)), 4) if cross_dtws else None,
        }
        results.append(result)

    results.sort(key=lambda r: r["ratio"] or 999)

    print(f"\n{'Action':20s} {'#Skel':>5s} {'Intra':>8s} {'Cross':>8s} {'Ratio':>8s}")
    print("-" * 55)
    for r in results[:10]:
        print(f"{r['action']:20s} {r['n_skeletons']:5d} {r['intra_dtw']:8.4f} {r['cross_dtw']:8.4f} {r['ratio'] or 0:8.3f}")

    print(f"\n  ... ({len(results) - 15} more) ...\n")
    for r in results[-5:]:
        print(f"{r['action']:20s} {r['n_skeletons']:5d} {r['intra_dtw']:8.4f} {r['cross_dtw']:8.4f} {r['ratio'] or 0:8.3f}")

    valid = [r for r in results if r["ratio"] is not None]
    if valid:
        mean_ratio = np.mean([r["ratio"] for r in valid])
        converged = sum(1 for r in valid if r["ratio"] < 1.0) / len(valid) * 100
        print(f"\nOverall: {len(valid)} actions")
        print(f"  Mean intra/cross ratio: {mean_ratio:.3f} (1.0 = no convergence, <1.0 = convergent)")
        print(f"  Actions with ratio < 1.0: {converged:.0f}%")
        print(f"  Mean intra DTW: {np.mean([r['intra_dtw'] for r in valid]):.4f}")
        print(f"  Mean cross DTW: {np.mean([r['cross_dtw'] for r in valid]):.4f}")

    out_path = "eval/benchmark_paired/probe_a_convergence.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "n_actions": len(results)}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
