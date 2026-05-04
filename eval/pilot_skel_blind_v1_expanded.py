"""Expanded C-fast pilot with controls.

Three conditions:
  A. cross-skeleton same-action (should beat random if rep is skeleton-blind)
  B. same-skeleton same-action control (ceiling — should be near-perfect)
  C. cross-action negative (should score WORSE than same-action to show action-discriminability)
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


def find_motion(skel, keyword):
    for f in sorted(os.listdir(MOTION_DIR)):
        if (f.startswith(skel + "___") or f.startswith(skel + "_")) and keyword in f:
            return f
    return None


def load(fname):
    return np.load(os.path.join(MOTION_DIR, fname))


def make_cond(cond_dict, name):
    return {"joints_names": cond_dict[name]["joints_names"],
            "parents": cond_dict[name]["parents"],
            "object_type": name}


def evaluate_pair(inv_a, inv_b):
    T = min(inv_a.shape[0], inv_b.shape[0])
    inv_rand = np.random.randn(T, SLOT_COUNT, 8).astype(np.float32)
    return {
        "transfer": {
            "dtw": round(end_effector_dtw(inv_a, inv_b), 4),
            "contact_f1": round(contact_timing_f1(inv_a, inv_b), 4),
            "phase": round(phase_consistency(inv_a, inv_b), 4),
        },
        "random": {
            "dtw": round(end_effector_dtw(inv_rand, inv_b), 4),
            "contact_f1": round(contact_timing_f1(inv_rand, inv_b), 4),
            "phase": round(phase_consistency(inv_rand, inv_b), 4),
        },
    }


CROSS_SKEL_PAIRS = [
    ("Cat", "Walk", "Hound", "Walk"),
    ("Cat", "Walk", "Lynx", "Walk"),
    ("Horse", "SlowWalk", "Elephant", "walk"),
    ("Horse", "SlowWalk", "Camel", "SlowWalk"),
    ("Spider", "Walk", "Ant", "Walk"),
    ("Spider", "Walk", "Crab", "Walk"),
    ("Alligator", "Walk1", "Crocodile", "Walk"),
    ("Trex", "Walk1", "Raptor", "FastWalk"),
    ("Bear", "SlowWalk", "BrownBear", "SlowWalk"),
    ("Lion", "Walk", "Jaguar", "Walk"),
    ("Deer", "Walk", "Gazelle", "Walk"),
    ("Fox", "Walk", "Coyote", "Walk"),
]

SAME_SKEL_CONTROLS = [
    ("Alligator", "Walk1", "Alligator", "Walk2"),
    ("Spider", "Walk1", "Spider", "Walk2"),
    ("Bear", "SlowWalk", "Bear", "WalkForward"),
    ("Deer", "Walk", "Deer", "WalkBack"),
]

CROSS_ACTION_NEGATIVES = [
    ("Cat", "Walk", "Cat", "Attack"),
    ("Horse", "SlowWalk", "Horse", "Gallop"),
    ("Spider", "Walk", "Spider", "Attack"),
    ("Trex", "Walk1", "Trex", "Attack"),
]


def run():
    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()
    np.random.seed(42)
    all_results = {"cross_skeleton": [], "same_skeleton": [], "cross_action": []}

    for label, pairs in [("cross_skeleton", CROSS_SKEL_PAIRS),
                          ("same_skeleton", SAME_SKEL_CONTROLS),
                          ("cross_action", CROSS_ACTION_NEGATIVES)]:
        print(f"\n=== {label.upper()} ===")
        for sa, ka, sb, kb in pairs:
            if sa not in cond_dict or sb not in cond_dict:
                print(f"  SKIP: {sa} or {sb} not in cond_dict")
                continue
            fa = find_motion(sa, ka)
            fb = find_motion(sb, kb)
            if fa is None or fb is None:
                print(f"  SKIP: no '{ka}' for {sa} or '{kb}' for {sb}")
                continue

            inv_a = encode_motion_to_invariant(load(fa), make_cond(cond_dict, sa))
            inv_b = encode_motion_to_invariant(load(fb), make_cond(cond_dict, sb))
            res = evaluate_pair(inv_a, inv_b)
            res["pair"] = f"{sa}({ka})→{sb}({kb})"
            all_results[label].append(res)

            t, r = res["transfer"], res["random"]
            dtw_w = "W" if t["dtw"] < r["dtw"] else "L"
            f1_w = "W" if t["contact_f1"] > r["contact_f1"] else "L"
            ph_w = "W" if t["phase"] > r["phase"] else "L"
            print(f"  {res['pair']:40s}  DTW {t['dtw']:.3f}/{r['dtw']:.3f}[{dtw_w}]  F1 {t['contact_f1']:.3f}/{r['contact_f1']:.3f}[{f1_w}]  Ph {t['phase']:.3f}/{r['phase']:.3f}[{ph_w}]")

    def summarize(entries):
        if not entries:
            return {}
        dtw_wins = sum(1 for e in entries if e["transfer"]["dtw"] < e["random"]["dtw"])
        f1_wins = sum(1 for e in entries if e["transfer"]["contact_f1"] > e["random"]["contact_f1"])
        ph_wins = sum(1 for e in entries if e["transfer"]["phase"] > e["random"]["phase"])
        n = len(entries)
        return {
            "n": n,
            "dtw_win_rate": round(dtw_wins / n, 3),
            "f1_win_rate": round(f1_wins / n, 3),
            "phase_win_rate": round(ph_wins / n, 3),
            "mean_dtw_transfer": round(np.mean([e["transfer"]["dtw"] for e in entries]), 4),
            "mean_dtw_random": round(np.mean([e["random"]["dtw"] for e in entries]), 4),
            "mean_f1_transfer": round(np.mean([e["transfer"]["contact_f1"] for e in entries]), 4),
            "mean_f1_random": round(np.mean([e["random"]["contact_f1"] for e in entries]), 4),
            "mean_phase_transfer": round(np.mean([e["transfer"]["phase"] for e in entries]), 4),
            "mean_phase_random": round(np.mean([e["random"]["phase"] for e in entries]), 4),
        }

    summary = {k: summarize(v) for k, v in all_results.items()}
    print("\n=== AGGREGATE ===")
    for k, s in summary.items():
        if s:
            print(f"  {k}: n={s['n']}  DTW_wr={s['dtw_win_rate']:.0%}  F1_wr={s['f1_win_rate']:.0%}  Ph_wr={s['phase_win_rate']:.0%}")
            print(f"    DTW mean: {s['mean_dtw_transfer']:.4f} vs {s['mean_dtw_random']:.4f}")
            print(f"    F1  mean: {s['mean_f1_transfer']:.4f} vs {s['mean_f1_random']:.4f}")
            print(f"    Ph  mean: {s['mean_phase_transfer']:.4f} vs {s['mean_phase_random']:.4f}")

    output = {"summary": summary, "detailed": all_results}
    out_path = "idea-stage/v1_zero_training_pilot_expanded.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
