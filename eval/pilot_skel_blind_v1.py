"""C-fast zero-training pilot: does the invariant rep preserve action semantics across skeletons?

For 5 diverse skeleton pairs sharing a "Walk" action:
  1. Encode source Walk → inv_source [T, 32, 8]
  2. Encode target Walk → inv_target [T, 32, 8]
  3. Generate random baseline → inv_random [T, 32, 8]
  4. Compute 3 metrics: DTW, contact F1, phase consistency
     - transfer = metric(inv_source, inv_target)   [should be good if rep is skeleton-blind]
     - random   = metric(inv_random, inv_target)    [baseline]

If transfer >> random on DTW (lower is better) and transfer >> random on F1/phase (higher is better),
the invariant rep encodes action semantics regardless of skeleton topology.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.encoder import encode_motion_to_invariant
from model.skel_blind.slot_vocab import SLOT_COUNT
from eval.benchmark_paired.metrics import end_effector_dtw, contact_timing_f1, phase_consistency

DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
MOTION_DIR = os.path.join(DATA_ROOT, "motions")

PAIRS = [
    ("Cat", "Cat_CAT_Walk_197.npy", "Dog", None),
    ("Horse", "Horse___WalkLoop_451.npy", "Elephant", None),
    ("Spider", "Spider___Walk1_911.npy", "Ant", "Ant___Walk_46.npy"),
    ("Alligator", "Alligator___Walk1_17.npy", "Crab", None),
    ("Trex", "Trex___Walk1_986.npy", "Raptor", None),
]


def find_walk_motion(skel_name):
    candidates = []
    for f in os.listdir(MOTION_DIR):
        if f.startswith(skel_name + "___") or f.startswith(skel_name + "_"):
            if "Walk" in f or "walk" in f:
                candidates.append(f)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def load_motion(filename):
    return np.load(os.path.join(MOTION_DIR, filename))


def make_skel_cond(cond_dict, skel_name):
    entry = cond_dict[skel_name]
    return {
        "joints_names": entry["joints_names"],
        "parents": entry["parents"],
        "object_type": skel_name,
    }


def random_invariant(T):
    return np.random.randn(T, SLOT_COUNT, 8).astype(np.float32)


def run_pilot():
    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()

    resolved_pairs = []
    for src_name, src_file, tgt_name, tgt_file in PAIRS:
        if src_name not in cond_dict or tgt_name not in cond_dict:
            print(f"SKIP: {src_name} or {tgt_name} not in cond_dict")
            continue
        if not os.path.exists(os.path.join(MOTION_DIR, src_file)):
            alt = find_walk_motion(src_name)
            if alt is None:
                print(f"SKIP: no Walk motion for {src_name}")
                continue
            src_file = alt
        if tgt_file is None:
            tgt_file = find_walk_motion(tgt_name)
        if tgt_file is None or not os.path.exists(os.path.join(MOTION_DIR, tgt_file)):
            print(f"SKIP: no Walk motion for {tgt_name}")
            continue
        resolved_pairs.append((src_name, src_file, tgt_name, tgt_file))

    print(f"Resolved {len(resolved_pairs)} pairs")
    results = []
    np.random.seed(42)

    for src_name, src_file, tgt_name, tgt_file in resolved_pairs:
        print(f"\n--- {src_name} ({src_file}) → {tgt_name} ({tgt_file}) ---")
        src_motion = load_motion(src_file)
        tgt_motion = load_motion(tgt_file)
        src_cond = make_skel_cond(cond_dict, src_name)
        tgt_cond = make_skel_cond(cond_dict, tgt_name)

        print(f"  src: {src_motion.shape}, tgt: {tgt_motion.shape}")

        inv_src = encode_motion_to_invariant(src_motion, src_cond)
        inv_tgt = encode_motion_to_invariant(tgt_motion, tgt_cond)
        T_min = min(inv_src.shape[0], inv_tgt.shape[0])
        inv_rand = random_invariant(T_min)

        dtw_transfer = end_effector_dtw(inv_src, inv_tgt)
        dtw_random = end_effector_dtw(inv_rand, inv_tgt)

        f1_transfer = contact_timing_f1(inv_src, inv_tgt)
        f1_random = contact_timing_f1(inv_rand, inv_tgt)

        phase_transfer = phase_consistency(inv_src, inv_tgt)
        phase_random = phase_consistency(inv_rand, inv_tgt)

        pair_result = {
            "source": src_name,
            "target": tgt_name,
            "source_file": src_file,
            "target_file": tgt_file,
            "src_joints": int(src_motion.shape[1]),
            "tgt_joints": int(tgt_motion.shape[1]),
            "src_frames": int(src_motion.shape[0]),
            "tgt_frames": int(tgt_motion.shape[0]),
            "transfer": {
                "dtw": round(dtw_transfer, 4),
                "contact_f1": round(f1_transfer, 4),
                "phase_consistency": round(phase_transfer, 4),
            },
            "random_baseline": {
                "dtw": round(dtw_random, 4),
                "contact_f1": round(f1_random, 4),
                "phase_consistency": round(phase_random, 4),
            },
            "transfer_wins": {
                "dtw": dtw_transfer < dtw_random,
                "contact_f1": f1_transfer > f1_random,
                "phase_consistency": phase_transfer > phase_random,
            },
        }
        results.append(pair_result)

        print(f"  DTW:     transfer={dtw_transfer:.4f}  random={dtw_random:.4f}  {'WIN' if dtw_transfer < dtw_random else 'LOSE'}")
        print(f"  F1:      transfer={f1_transfer:.4f}  random={f1_random:.4f}  {'WIN' if f1_transfer > f1_random else 'LOSE'}")
        print(f"  Phase:   transfer={phase_transfer:.4f}  random={phase_random:.4f}  {'WIN' if phase_transfer > phase_random else 'LOSE'}")

    total_wins = sum(
        sum(r["transfer_wins"].values()) for r in results
    )
    total_comparisons = len(results) * 3
    win_rate = total_wins / max(total_comparisons, 1)

    summary = {
        "pilot": "C-fast zero-training invariant rep transfer",
        "n_pairs": len(results),
        "win_rate": round(win_rate, 3),
        "total_wins": total_wins,
        "total_comparisons": total_comparisons,
        "pairs": results,
    }

    print(f"\n=== SUMMARY ===")
    print(f"Win rate: {total_wins}/{total_comparisons} = {win_rate:.1%}")
    print(f"Verdict: {'POSITIVE — invariant rep preserves cross-skeleton semantics' if win_rate > 0.6 else 'NEGATIVE — invariant rep does NOT beat random baseline'}")

    out_path = os.path.join("idea-stage", "v1_zero_training_pilot.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_pilot()
