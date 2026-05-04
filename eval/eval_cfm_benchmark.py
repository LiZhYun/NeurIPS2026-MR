"""Evaluate trained CFM on the 200-pair benchmark.

For each pair (skel_A action X, skel_B action X):
1. Encode source motion → inv_src
2. Generate via CFM conditioned on inv_src → inv_gen
3. Compare inv_gen vs inv_tgt (ground truth) using DTW, F1, phase
4. Also compare zero-training baseline (inv_src vs inv_tgt) and random

Three conditions: CFM_generated, zero_training_transfer, random_baseline.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.cfm_model import InvariantCFM, SLOT_COUNT, CHANNEL_COUNT
from model.skel_blind.encoder import encode_motion_to_invariant
from eval.benchmark_paired.metrics import end_effector_dtw, contact_timing_f1, phase_consistency
from sample.generate_cfm import load_model, sample_euler

DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
PAIRS_DIR = "eval/benchmark_paired/pairs"


def make_cond(cond_dict, name):
    return {"joints_names": cond_dict[name]["joints_names"],
            "parents": cond_dict[name]["parents"],
            "object_type": name}


def pad_or_crop(inv, window):
    T = inv.shape[0]
    if T > window:
        return inv[:window]
    elif T < window:
        pad = np.zeros((window - T, SLOT_COUNT, CHANNEL_COUNT), dtype=np.float32)
        return np.concatenate([inv, pad], axis=0)
    return inv


def evaluate_checkpoint(ckpt_path, max_pairs=50, n_steps=30, cfg_weight=2.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(ckpt_path, device)
    window = model_args.get("window", 40)

    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()

    with open(os.path.join(PAIRS_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    np.random.seed(42)
    results = []

    for i, p in enumerate(manifest["pairs"][:max_pairs]):
        data = np.load(os.path.join(PAIRS_DIR, p["pair_file"]))
        inv_a = data["inv_a"]  # source skeleton
        inv_b = data["inv_b"]  # target skeleton (ground truth)

        inv_a_w = pad_or_crop(inv_a, window)
        inv_b_w = pad_or_crop(inv_b, window)

        z_src = torch.from_numpy(inv_a_w).float().unsqueeze(0).to(device)
        with torch.no_grad():
            inv_gen = sample_euler(model, z_src, n_steps=n_steps, cfg_weight=cfg_weight)
        inv_gen_np = inv_gen[0].cpu().numpy()

        T_eval = min(inv_a.shape[0], inv_b.shape[0], window)
        inv_rand = np.random.randn(T_eval, SLOT_COUNT, CHANNEL_COUNT).astype(np.float32)

        inv_a_eval = inv_a_w[:T_eval]
        inv_b_eval = inv_b_w[:T_eval]
        inv_gen_eval = inv_gen_np[:T_eval]

        result = {
            "pair_id": p["pair_id"],
            "action": p["action"],
            "skel_a": p["skel_a"],
            "skel_b": p["skel_b"],
            "cfm": {
                "dtw": round(end_effector_dtw(inv_gen_eval, inv_b_eval), 4),
                "f1": round(contact_timing_f1(inv_gen_eval, inv_b_eval), 4),
                "phase": round(phase_consistency(inv_gen_eval, inv_b_eval), 4),
            },
            "zero_training": {
                "dtw": round(end_effector_dtw(inv_a_eval, inv_b_eval), 4),
                "f1": round(contact_timing_f1(inv_a_eval, inv_b_eval), 4),
                "phase": round(phase_consistency(inv_a_eval, inv_b_eval), 4),
            },
            "random": {
                "dtw": round(end_effector_dtw(inv_rand, inv_b_eval), 4),
                "f1": round(contact_timing_f1(inv_rand, inv_b_eval), 4),
                "phase": round(phase_consistency(inv_rand, inv_b_eval), 4),
            },
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{min(max_pairs, len(manifest['pairs']))}] {p['action']}: {p['skel_a']}→{p['skel_b']}")

    def agg(key):
        vals = [r[key] for r in results]
        return {
            "mean_dtw": round(np.mean([v["dtw"] for v in vals]), 4),
            "mean_f1": round(np.mean([v["f1"] for v in vals]), 4),
            "mean_phase": round(np.mean([v["phase"] for v in vals]), 4),
            "dtw_vs_random_win": round(sum(1 for r in results if r[key]["dtw"] < r["random"]["dtw"]) / len(results), 3),
        }

    summary = {
        "checkpoint": ckpt_path,
        "n_pairs": len(results),
        "cfm": agg("cfm"),
        "zero_training": agg("zero_training"),
        "random": agg("random"),
    }

    print(f"\n=== EVALUATION ({len(results)} pairs, ckpt={os.path.basename(ckpt_path)}) ===")
    for cond_name in ["cfm", "zero_training", "random"]:
        s = summary[cond_name]
        print(f"  {cond_name:15s}: DTW={s['mean_dtw']:.4f}  F1={s['mean_f1']:.4f}  Ph={s['mean_phase']:.4f}  DTW_wr={s['dtw_vs_random_win']:.0%}")

    cfm_better = summary["cfm"]["mean_dtw"] < summary["zero_training"]["mean_dtw"]
    print(f"\n  CFM {'IMPROVES' if cfm_better else 'DOES NOT improve'} over zero-training baseline")

    out = {"summary": summary, "pairs": results}
    out_name = f"cfm_eval_{os.path.splitext(os.path.basename(ckpt_path))[0]}.json"
    out_path = os.path.join(PAIRS_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--cfg_weight", type=float, default=2.0)
    args = parser.parse_args()
    evaluate_checkpoint(args.ckpt, args.max_pairs, args.n_steps, args.cfg_weight)
