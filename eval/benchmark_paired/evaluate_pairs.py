"""Evaluate all paired clips from the benchmark manifest.

Computes DTW, contact F1, and phase consistency for each pair,
plus a random baseline. Reports per-action and overall statistics.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.skel_blind.slot_vocab import SLOT_COUNT
from eval.benchmark_paired.metrics import end_effector_dtw, contact_timing_f1, phase_consistency

PAIRS_DIR = "eval/benchmark_paired/pairs"


def run():
    manifest_path = os.path.join(PAIRS_DIR, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    np.random.seed(42)
    results = []

    for p in manifest["pairs"]:
        data = np.load(os.path.join(PAIRS_DIR, p["pair_file"]))
        inv_a, inv_b = data["inv_a"], data["inv_b"]
        T = min(inv_a.shape[0], inv_b.shape[0])
        inv_rand = np.random.randn(T, SLOT_COUNT, 8).astype(np.float32)

        dtw_t = end_effector_dtw(inv_a, inv_b)
        dtw_r = end_effector_dtw(inv_rand, inv_b)
        f1_t = contact_timing_f1(inv_a, inv_b)
        f1_r = contact_timing_f1(inv_rand, inv_b)
        ph_t = phase_consistency(inv_a, inv_b)
        ph_r = phase_consistency(inv_rand, inv_b)

        results.append({
            "pair_id": p["pair_id"],
            "action": p["action"],
            "skel_a": p["skel_a"],
            "skel_b": p["skel_b"],
            "dtw_transfer": round(dtw_t, 4),
            "dtw_random": round(dtw_r, 4),
            "f1_transfer": round(f1_t, 4),
            "f1_random": round(f1_r, 4),
            "phase_transfer": round(ph_t, 4),
            "phase_random": round(ph_r, 4),
            "dtw_win": dtw_t < dtw_r,
            "f1_win": f1_t > f1_r,
            "phase_win": ph_t > ph_r,
        })

    n = len(results)
    dtw_wins = sum(r["dtw_win"] for r in results)
    f1_wins = sum(r["f1_win"] for r in results)
    ph_wins = sum(r["phase_win"] for r in results)

    summary = {
        "n_pairs": n,
        "dtw_win_rate": round(dtw_wins / n, 3),
        "f1_win_rate": round(f1_wins / n, 3),
        "phase_win_rate": round(ph_wins / n, 3),
        "mean_dtw_transfer": round(np.mean([r["dtw_transfer"] for r in results]), 4),
        "mean_dtw_random": round(np.mean([r["dtw_random"] for r in results]), 4),
        "mean_f1_transfer": round(np.mean([r["f1_transfer"] for r in results]), 4),
        "mean_f1_random": round(np.mean([r["f1_random"] for r in results]), 4),
        "mean_phase_transfer": round(np.mean([r["phase_transfer"] for r in results]), 4),
        "mean_phase_random": round(np.mean([r["phase_random"] for r in results]), 4),
    }

    print(f"=== BENCHMARK EVALUATION ({n} pairs) ===")
    print(f"DTW:   win rate {summary['dtw_win_rate']:.0%}  mean {summary['mean_dtw_transfer']:.4f} vs {summary['mean_dtw_random']:.4f}")
    print(f"F1:    win rate {summary['f1_win_rate']:.0%}  mean {summary['mean_f1_transfer']:.4f} vs {summary['mean_f1_random']:.4f}")
    print(f"Phase: win rate {summary['phase_win_rate']:.0%}  mean {summary['mean_phase_transfer']:.4f} vs {summary['mean_phase_random']:.4f}")

    per_action = {}
    for r in results:
        per_action.setdefault(r["action"], []).append(r)
    print(f"\nPer-action breakdown:")
    for a in sorted(per_action.keys()):
        rs = per_action[a]
        na = len(rs)
        dw = sum(r["dtw_win"] for r in rs)
        print(f"  {a}: n={na}  DTW_wr={dw/na:.0%}  mean_dtw={np.mean([r['dtw_transfer'] for r in rs]):.3f}")

    out = {"summary": summary, "per_action": {a: len(rs) for a, rs in per_action.items()}, "pairs": results}
    out_path = os.path.join(PAIRS_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
