"""Action discriminability: can we distinguish actions from invariant reps across skeletons?

For each skeleton with 2+ action types, compute:
- Intra-action DTW (same action, different clips)
- Inter-action DTW (different actions, same skeleton)

If intra < inter, the invariant rep is action-discriminative.
Also test cross-skeleton: same action on different skeletons vs different actions on different skeletons.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.encoder import encode_motion_to_invariant
from eval.benchmark_paired.metrics import end_effector_dtw

DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
MOTION_DIR = os.path.join(DATA_ROOT, "motions")
INV_DIR = "dataset/truebones/zoo/invariant_reps"


def extract_action(filename):
    import re
    base = os.path.splitext(filename)[0]
    if "___" in base:
        action_part = base.split("___", 1)[1]
    else:
        action_part = base.split("_", 1)[1] if "_" in base else base
    action_part = re.sub(r'_\d+$', '', action_part)
    action_part = re.sub(r'\d+$', '', action_part)
    return action_part.lower().strip("_")


def run():
    manifest_path = os.path.join(INV_DIR, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build action index per skeleton
    skel_action_clips = defaultdict(lambda: defaultdict(list))
    for skel_name, info in manifest["skeletons"].items():
        data = np.load(os.path.join(INV_DIR, f"{skel_name}.npz"), allow_pickle=True)
        for clip_name in info["clips"]:
            action = extract_action(clip_name)
            inv = data[clip_name]
            skel_action_clips[skel_name][action].append(inv)

    # Within-skeleton discriminability
    intra_dists = []
    inter_dists = []
    n_skels_tested = 0

    for skel, actions in skel_action_clips.items():
        action_list = [a for a, clips in actions.items() if len(clips) >= 1]
        if len(action_list) < 2:
            continue
        n_skels_tested += 1

        # Intra-action: compare clips of the same action
        for a in action_list:
            clips = actions[a]
            for i in range(len(clips)):
                for j in range(i + 1, len(clips)):
                    d = end_effector_dtw(clips[i], clips[j])
                    intra_dists.append(d)

        # Inter-action: compare clips of different actions (sample to avoid explosion)
        rng = np.random.RandomState(42)
        pairs_done = 0
        for i in range(len(action_list)):
            for j in range(i + 1, len(action_list)):
                if pairs_done >= 50:
                    break
                a1, a2 = action_list[i], action_list[j]
                c1 = actions[a1][rng.randint(len(actions[a1]))]
                c2 = actions[a2][rng.randint(len(actions[a2]))]
                d = end_effector_dtw(c1, c2)
                inter_dists.append(d)
                pairs_done += 1

    # Cross-skeleton discriminability using benchmark pairs
    with open("eval/benchmark_paired/pairs/evaluation.json") as f:
        eval_data = json.load(f)

    cross_same_action = [p["dtw_transfer"] for p in eval_data["pairs"]]
    cross_random = [p["dtw_random"] for p in eval_data["pairs"]]

    print(f"=== ACTION DISCRIMINABILITY ===")
    print(f"\nWithin-skeleton ({n_skels_tested} skeletons):")
    print(f"  Intra-action DTW (same action, diff clips):     mean={np.mean(intra_dists):.4f}  n={len(intra_dists)}")
    print(f"  Inter-action DTW (diff action, same skeleton):  mean={np.mean(inter_dists):.4f}  n={len(inter_dists)}")
    ratio = np.mean(intra_dists) / np.mean(inter_dists) if inter_dists else float('inf')
    print(f"  Ratio (intra/inter): {ratio:.3f}  {'DISCRIMINATIVE' if ratio < 1.0 else 'NOT discriminative'}")

    print(f"\nCross-skeleton ({len(cross_same_action)} pairs):")
    print(f"  Same-action different-skeleton DTW:  mean={np.mean(cross_same_action):.4f}")
    print(f"  Random baseline DTW:                 mean={np.mean(cross_random):.4f}")
    print(f"  Ratio: {np.mean(cross_same_action)/np.mean(cross_random):.3f}")

    summary = {
        "within_skeleton": {
            "n_skeletons": n_skels_tested,
            "intra_action_dtw": round(np.mean(intra_dists), 4),
            "inter_action_dtw": round(np.mean(inter_dists), 4),
            "ratio": round(ratio, 3),
            "n_intra": len(intra_dists),
            "n_inter": len(inter_dists),
        },
        "cross_skeleton": {
            "same_action_dtw": round(np.mean(cross_same_action), 4),
            "random_dtw": round(np.mean(cross_random), 4),
            "ratio": round(np.mean(cross_same_action) / np.mean(cross_random), 3),
        },
    }
    out_path = "idea-stage/action_discriminability.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
