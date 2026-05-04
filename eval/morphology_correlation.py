"""Morphological similarity vs transfer quality correlation.

Question: do topologically similar skeletons transfer better?
Uses joint-count ratio and shared-slot overlap as similarity proxies.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.slot_assign import assign_joints_to_slots
from model.skel_blind.slot_vocab import slot_type_to_idx

DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
PAIRS_DIR = "eval/benchmark_paired/pairs"


def slot_overlap(cond_a, cond_b):
    """Fraction of non-null slots that both skeletons populate."""
    null_idx = slot_type_to_idx("null")
    asg_a = assign_joints_to_slots(cond_a)
    asg_b = assign_joints_to_slots(cond_b)
    active_a = set(s for s in asg_a.slot_to_joints if s != null_idx and asg_a.slot_to_joints[s])
    active_b = set(s for s in asg_b.slot_to_joints if s != null_idx and asg_b.slot_to_joints[s])
    union = active_a | active_b
    if not union:
        return 0.0
    return len(active_a & active_b) / len(union)


def run():
    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()

    with open(os.path.join(PAIRS_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(PAIRS_DIR, "evaluation.json")) as f:
        eval_data = json.load(f)

    eval_by_id = {p["pair_id"]: p for p in eval_data["pairs"]}
    results = []

    for p in manifest["pairs"]:
        pid = p["pair_id"]
        if pid not in eval_by_id:
            continue
        e = eval_by_id[pid]
        sa, sb = p["skel_a"], p["skel_b"]
        if sa not in cond_dict or sb not in cond_dict:
            continue

        cond_a = {"joints_names": cond_dict[sa]["joints_names"],
                  "parents": cond_dict[sa]["parents"], "object_type": sa}
        cond_b = {"joints_names": cond_dict[sb]["joints_names"],
                  "parents": cond_dict[sb]["parents"], "object_type": sb}

        ja = len(cond_dict[sa]["joints_names"])
        jb = len(cond_dict[sb]["joints_names"])
        joint_ratio = min(ja, jb) / max(ja, jb)
        overlap = slot_overlap(cond_a, cond_b)

        results.append({
            "pair_id": pid,
            "action": p["action"],
            "skel_a": sa, "skel_b": sb,
            "joints_a": ja, "joints_b": jb,
            "joint_ratio": round(joint_ratio, 3),
            "slot_overlap": round(overlap, 3),
            "dtw": e["dtw_transfer"],
            "f1": e["f1_transfer"],
            "dtw_win": e["dtw_win"],
        })

    overlaps = np.array([r["slot_overlap"] for r in results])
    dtws = np.array([r["dtw"] for r in results])
    f1s = np.array([r["f1"] for r in results])
    ratios = np.array([r["joint_ratio"] for r in results])

    corr_overlap_dtw = np.corrcoef(overlaps, dtws)[0, 1]
    corr_overlap_f1 = np.corrcoef(overlaps, f1s)[0, 1]
    corr_ratio_dtw = np.corrcoef(ratios, dtws)[0, 1]

    print(f"=== MORPHOLOGY vs TRANSFER QUALITY ({len(results)} pairs) ===")
    print(f"Correlation (slot_overlap, DTW↓):  r={corr_overlap_dtw:.3f}")
    print(f"Correlation (slot_overlap, F1↑):   r={corr_overlap_f1:.3f}")
    print(f"Correlation (joint_ratio, DTW↓):   r={corr_ratio_dtw:.3f}")

    # Bin by overlap quartiles
    quartiles = np.percentile(overlaps, [25, 50, 75])
    bins = [("Q1 (low overlap)", overlaps <= quartiles[0]),
            ("Q2", (overlaps > quartiles[0]) & (overlaps <= quartiles[1])),
            ("Q3", (overlaps > quartiles[1]) & (overlaps <= quartiles[2])),
            ("Q4 (high overlap)", overlaps > quartiles[2])]

    print(f"\nPer-quartile breakdown:")
    print(f"{'Quartile':25s}  {'N':>3s}  {'DTW↓':>8s}  {'F1↑':>8s}  {'DTW_wr':>8s}")
    for label, mask in bins:
        n = mask.sum()
        if n == 0:
            continue
        d = dtws[mask].mean()
        f = f1s[mask].mean()
        w = np.array([r["dtw_win"] for r in results])[mask].mean()
        print(f"{label:25s}  {n:3d}  {d:8.4f}  {f:8.4f}  {w:8.1%}")

    summary = {
        "n_pairs": len(results),
        "corr_overlap_dtw": round(corr_overlap_dtw, 3),
        "corr_overlap_f1": round(corr_overlap_f1, 3),
        "corr_ratio_dtw": round(corr_ratio_dtw, 3),
    }
    out_path = "idea-stage/morphology_correlation.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "pairs": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
