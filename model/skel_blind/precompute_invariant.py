"""Pre-encode all Truebones motions to invariant representation for CFM training.

Saves one npz per skeleton type containing all its clips' invariant reps,
plus a manifest JSON for quick loading.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.skel_blind.encoder import encode_motion_to_invariant

DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
MOTION_DIR = os.path.join(DATA_ROOT, "motions")
OUT_DIR = "dataset/truebones/zoo/invariant_reps"


def make_cond(cond_dict, name):
    return {
        "joints_names": cond_dict[name]["joints_names"],
        "parents": cond_dict[name]["parents"],
        "object_type": name,
    }


def precompute():
    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = {"skeletons": {}, "total_clips": 0, "total_frames": 0}
    all_files = sorted(os.listdir(MOTION_DIR))
    processed = 0
    errors = 0

    for skel_name in sorted(cond_dict.keys()):
        cond = make_cond(cond_dict, skel_name)
        skel_clips = {}
        skel_frames = 0

        for f in all_files:
            if not (f.startswith(skel_name + "___") or f.startswith(skel_name + "_")):
                continue
            if not f.endswith(".npy"):
                continue
            # Disambiguate: e.g., "Bear" shouldn't match "BearPolar_..."
            rest = f[len(skel_name):]
            if not (rest.startswith("___") or rest.startswith("_")):
                continue
            # Extra check: if another longer skeleton name also matches, skip
            longer_match = False
            for other in cond_dict:
                if other != skel_name and f.startswith(other + "___") and len(other) > len(skel_name):
                    longer_match = True
                    break
                if other != skel_name and f.startswith(other + "_") and len(other) > len(skel_name):
                    longer_match = True
                    break
            if longer_match:
                continue

            try:
                motion = np.load(os.path.join(MOTION_DIR, f))
                inv = encode_motion_to_invariant(motion, cond)
                clip_name = os.path.splitext(f)[0]
                skel_clips[clip_name] = inv
                skel_frames += inv.shape[0]
                processed += 1
            except Exception as e:
                print(f"  ERROR {f}: {e}")
                errors += 1

        if skel_clips:
            out_path = os.path.join(OUT_DIR, f"{skel_name}.npz")
            np.savez_compressed(out_path, **skel_clips)
            manifest["skeletons"][skel_name] = {
                "n_clips": len(skel_clips),
                "total_frames": skel_frames,
                "clips": list(skel_clips.keys()),
            }
            manifest["total_clips"] += len(skel_clips)
            manifest["total_frames"] += skel_frames
            print(f"  {skel_name}: {len(skel_clips)} clips, {skel_frames} frames")

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Processed: {processed} clips, {errors} errors")
    print(f"Total frames: {manifest['total_frames']}")
    print(f"Skeletons: {len(manifest['skeletons'])}")
    print(f"Saved to: {OUT_DIR}")
    return manifest


if __name__ == "__main__":
    precompute()
