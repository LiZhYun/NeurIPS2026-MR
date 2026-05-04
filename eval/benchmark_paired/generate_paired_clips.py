"""Generate paired benchmark clips by encoding the same action across different skeletons.

For each action label (e.g. "Walk") shared by 2+ skeletons, encode both motions to the
32-slot invariant rep. The resulting pairs form ground-truth cross-skeleton correspondences
for the stress-test benchmark (spec §4).

Output: JSON manifest + per-pair .npz files with invariant reps.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.skel_blind.encoder import encode_motion_to_invariant

DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
MOTION_DIR = os.path.join(DATA_ROOT, "motions")
OUT_DIR = "eval/benchmark_paired/pairs"


def extract_action_label(filename):
    """Extract a normalized action label from a Truebones motion filename.

    Filenames: '{Skeleton}___{Action}_{id}.npy' or '{Skeleton}_{...}_{Action}_{id}.npy'
    """
    base = os.path.splitext(filename)[0]
    if "___" in base:
        parts = base.split("___", 1)
        action_part = parts[1]
    else:
        parts = base.split("_", 1)
        if len(parts) < 2:
            return None
        action_part = parts[1]
    action_part = re.sub(r'_\d+$', '', action_part)
    action_part = re.sub(r'\d+$', '', action_part)
    return action_part.lower().strip("_")


def make_skel_cond(cond_dict, skel_name):
    entry = cond_dict[skel_name]
    return {
        "joints_names": entry["joints_names"],
        "parents": entry["parents"],
        "object_type": skel_name,
    }


def build_action_index(cond_dict):
    """Build {action_label: [(skeleton, filename), ...]} index."""
    action_to_clips = defaultdict(list)
    for f in sorted(os.listdir(MOTION_DIR)):
        if not f.endswith(".npy"):
            continue
        skel = None
        for name in cond_dict:
            if f.startswith(name + "___") or f.startswith(name + "_"):
                if skel is None or len(name) > len(skel):
                    skel = name
        if skel is None:
            continue
        action = extract_action_label(f)
        if action:
            action_to_clips[action].append((skel, f))
    return action_to_clips


def generate_pairs(max_pairs=200, max_per_action=5):
    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()
    action_index = build_action_index(cond_dict)

    multi_skel_actions = {
        a: clips for a, clips in action_index.items()
        if len(set(c[0] for c in clips)) >= 2
    }
    print(f"Actions with 2+ skeletons: {len(multi_skel_actions)}")
    for a in sorted(multi_skel_actions.keys())[:20]:
        skels = sorted(set(c[0] for c in multi_skel_actions[a]))
        print(f"  {a}: {skels}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = []
    pair_id = 0

    rng = np.random.RandomState(42)

    for action in sorted(multi_skel_actions.keys()):
        if pair_id >= max_pairs:
            break
        clips = multi_skel_actions[action]
        skels_seen = {}
        for skel, fname in clips:
            if skel not in skels_seen:
                skels_seen[skel] = fname

        skel_list = sorted(skels_seen.keys())
        all_combos = [(skel_list[i], skel_list[j])
                      for i in range(len(skel_list))
                      for j in range(i + 1, len(skel_list))]
        rng.shuffle(all_combos)
        action_count = 0

        for s_a, s_b in all_combos:
            if action_count >= max_per_action or pair_id >= max_pairs:
                break
            f_a, f_b = skels_seen[s_a], skels_seen[s_b]

            motion_a = np.load(os.path.join(MOTION_DIR, f_a))
            motion_b = np.load(os.path.join(MOTION_DIR, f_b))
            cond_a = make_skel_cond(cond_dict, s_a)
            cond_b = make_skel_cond(cond_dict, s_b)

            inv_a = encode_motion_to_invariant(motion_a, cond_a)
            inv_b = encode_motion_to_invariant(motion_b, cond_b)

            pair_file = f"pair_{pair_id:04d}_{action}_{s_a}_{s_b}.npz"
            np.savez_compressed(
                os.path.join(OUT_DIR, pair_file),
                inv_a=inv_a, inv_b=inv_b,
                skel_a=s_a, skel_b=s_b,
                file_a=f_a, file_b=f_b,
                action=action,
            )

            pairs.append({
                "pair_id": pair_id,
                "action": action,
                "skel_a": s_a,
                "skel_b": s_b,
                "file_a": f_a,
                "file_b": f_b,
                "frames_a": int(inv_a.shape[0]),
                "frames_b": int(inv_b.shape[0]),
                "pair_file": pair_file,
            })
            pair_id += 1
            action_count += 1
            print(f"  [{pair_id}] {action}: {s_a}({motion_a.shape[1]}j) ↔ {s_b}({motion_b.shape[1]}j)")

    manifest = {
        "benchmark": "cross-skeleton paired clips v0",
        "n_pairs": len(pairs),
        "n_actions": len(set(p["action"] for p in pairs)),
        "n_skeletons": len(set(s for p in pairs for s in [p["skel_a"], p["skel_b"]])),
        "pairs": pairs,
    }

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"Generated {len(pairs)} pairs across {manifest['n_actions']} actions, {manifest['n_skeletons']} skeletons")
    print(f"Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    generate_pairs(max_pairs=50)
