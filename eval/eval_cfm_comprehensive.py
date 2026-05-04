"""Comprehensive CFM evaluation with paper-ready breakdowns.

Produces: per-action, per-morphology, statistical significance (bootstrap CI),
and DTW vs slot_overlap scatter data.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark_paired.metrics import end_effector_dtw, contact_timing_f1, phase_consistency
from eval.build_contact_groups import classify_morphology
from model.skel_blind.cfm_model import SLOT_COUNT, CHANNEL_COUNT
from model.skel_blind.invariant_dataset import TEST_SKELETONS
from model.skel_blind.slot_vocab import slot_type_to_idx
from sample.generate_cfm import load_model, sample_euler

import torch

DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
PAIRS_DIR = "eval/benchmark_paired/pairs"
NULL_SLOT = slot_type_to_idx("null")


def get_morphology(cond_dict, skel_name):
    entry = cond_dict[skel_name]
    names = list(entry["joints_names"])
    parents = list(entry["parents"])
    chains = [list(c) for c in entry["kinematic_chains"]]
    return classify_morphology(names, parents, chains)


def slot_overlap(inv_a, inv_b):
    active_a = set()
    active_b = set()
    for s in range(SLOT_COUNT):
        if s == NULL_SLOT:
            continue
        if not np.all(inv_a[:, s, :3] == 0):
            active_a.add(s)
        if not np.all(inv_b[:, s, :3] == 0):
            active_b.add(s)
    if not active_a and not active_b:
        return 0.0
    return len(active_a & active_b) / len(active_a | active_b)


def pad_or_crop(inv, window):
    T = inv.shape[0]
    if T > window:
        return inv[:window]
    elif T < window:
        pad = np.zeros((window - T, SLOT_COUNT, CHANNEL_COUNT), dtype=np.float32)
        return np.concatenate([inv, pad], axis=0)
    return inv


def bootstrap_ci(values, n_boot=1000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    means = [rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(lo), float(np.mean(values)), float(hi)


def evaluate_comprehensive(ckpt_path, max_pairs=200, n_steps=30, cfg_weight=2.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(ckpt_path, device)
    window = model_args.get("window", 40)

    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()

    with open(os.path.join(PAIRS_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    np.random.seed(42)
    pairs = manifest["pairs"][:max_pairs]

    morph_cache = {}
    def get_morph(name):
        if name not in morph_cache:
            morph_cache[name] = get_morphology(cond_dict, name)
        return morph_cache[name]

    results = []
    for i, p in enumerate(pairs):
        data = np.load(os.path.join(PAIRS_DIR, p["pair_file"]))
        inv_a = data["inv_a"]
        inv_b = data["inv_b"]

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

        overlap = slot_overlap(inv_a[:T_eval], inv_b[:T_eval])
        morph_a = get_morph(p["skel_a"])
        morph_b = get_morph(p["skel_b"])
        morph_pair = f"{morph_a}→{morph_b}" if morph_a != morph_b else f"{morph_a}(same)"

        a_test = p["skel_a"] in TEST_SKELETONS
        b_test = p["skel_b"] in TEST_SKELETONS
        if a_test and b_test:
            split_cat = "test_test"
        elif not a_test and not b_test:
            split_cat = "train_train"
        else:
            split_cat = "mixed"

        result = {
            "pair_id": p["pair_id"],
            "action": p["action"],
            "skel_a": p["skel_a"],
            "skel_b": p["skel_b"],
            "morph_a": morph_a,
            "morph_b": morph_b,
            "morph_pair": morph_pair,
            "split_cat": split_cat,
            "slot_overlap": round(overlap, 4),
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

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}] {p['action']}: {p['skel_a']}→{p['skel_b']}")

    report = build_report(results, ckpt_path)
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_path = os.path.join(PAIRS_DIR, f"comprehensive_{ckpt_stem}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")

    print_report(report)
    return report


def build_report(results, ckpt_path):
    def compute_group(group, key):
        dtws = [r[key]["dtw"] for r in group]
        f1s = [r[key]["f1"] for r in group]
        phases = [r[key]["phase"] for r in group]
        dtw_lo, dtw_mean, dtw_hi = bootstrap_ci(dtws)
        f1_lo, f1_mean, f1_hi = bootstrap_ci(f1s)
        ph_lo, ph_mean, ph_hi = bootstrap_ci(phases)
        return {
            "n": len(group),
            "dtw": {"mean": round(dtw_mean, 4), "ci95": [round(dtw_lo, 4), round(dtw_hi, 4)]},
            "f1": {"mean": round(f1_mean, 4), "ci95": [round(f1_lo, 4), round(f1_hi, 4)]},
            "phase": {"mean": round(ph_mean, 4), "ci95": [round(ph_lo, 4), round(ph_hi, 4)]},
        }

    def dtw_win_rate(group, a_key, b_key):
        wins = sum(1 for r in group if r[a_key]["dtw"] < r[b_key]["dtw"])
        return round(wins / len(group) * 100, 1) if group else 0

    overall = {
        "cfm": compute_group(results, "cfm"),
        "zero_training": compute_group(results, "zero_training"),
        "random": compute_group(results, "random"),
        "cfm_vs_zt_dtw_win": dtw_win_rate(results, "cfm", "zero_training"),
        "cfm_vs_random_dtw_win": dtw_win_rate(results, "cfm", "random"),
    }

    by_action = defaultdict(list)
    for r in results:
        by_action[r["action"]].append(r)

    action_table = {}
    for action, group in sorted(by_action.items()):
        action_table[action] = {
            "n": len(group),
            "cfm_dtw": round(np.mean([r["cfm"]["dtw"] for r in group]), 4),
            "zt_dtw": round(np.mean([r["zero_training"]["dtw"] for r in group]), 4),
            "improvement": round(
                (np.mean([r["zero_training"]["dtw"] for r in group])
                 - np.mean([r["cfm"]["dtw"] for r in group]))
                / max(np.mean([r["zero_training"]["dtw"] for r in group]), 1e-8) * 100, 1),
        }

    by_morph = defaultdict(list)
    for r in results:
        by_morph[r["morph_a"]].append(r)
        if r["morph_a"] != r["morph_b"]:
            by_morph[r["morph_b"]].append(r)

    morph_table = {}
    for morph, group in sorted(by_morph.items()):
        morph_table[morph] = {
            "n": len(group),
            "cfm_dtw": round(np.mean([r["cfm"]["dtw"] for r in group]), 4),
            "zt_dtw": round(np.mean([r["zero_training"]["dtw"] for r in group]), 4),
            "cfm_f1": round(np.mean([r["cfm"]["f1"] for r in group]), 4),
            "cfm_vs_zt_win": dtw_win_rate(group, "cfm", "zero_training"),
        }

    cross_vs_same = {"cross": [], "same": []}
    for r in results:
        if r["morph_a"] == r["morph_b"]:
            cross_vs_same["same"].append(r)
        else:
            cross_vs_same["cross"].append(r)

    morph_cross = {}
    for key, group in cross_vs_same.items():
        if group:
            morph_cross[key] = {
                "n": len(group),
                "cfm_dtw": round(np.mean([r["cfm"]["dtw"] for r in group]), 4),
                "zt_dtw": round(np.mean([r["zero_training"]["dtw"] for r in group]), 4),
                "cfm_vs_zt_win": dtw_win_rate(group, "cfm", "zero_training"),
            }

    scatter = [
        {"overlap": r["slot_overlap"],
         "cfm_dtw": r["cfm"]["dtw"],
         "zt_dtw": r["zero_training"]["dtw"],
         "improvement": (r["zero_training"]["dtw"] - r["cfm"]["dtw"]) / max(r["zero_training"]["dtw"], 1e-8) * 100}
        for r in results
    ]

    by_split = {}
    for cat in ["train_train", "mixed", "test_test"]:
        group = [r for r in results if r.get("split_cat") == cat]
        if group:
            by_split[cat] = {
                "n": len(group),
                "cfm_dtw": round(np.mean([r["cfm"]["dtw"] for r in group]), 4),
                "zt_dtw": round(np.mean([r["zero_training"]["dtw"] for r in group]), 4),
                "cfm_vs_zt_win": dtw_win_rate(group, "cfm", "zero_training"),
            }

    return {
        "checkpoint": ckpt_path,
        "n_pairs": len(results),
        "overall": overall,
        "by_action": action_table,
        "by_morphology": morph_table,
        "cross_vs_same_morph": morph_cross,
        "by_split": by_split,
        "scatter_data": scatter,
        "pairs": results,
    }


def print_report(report):
    o = report["overall"]
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE EVALUATION — {report['n_pairs']} pairs")
    print(f"Checkpoint: {report['checkpoint']}")
    print(f"{'='*70}")

    print(f"\n--- Overall ---")
    for key in ["cfm", "zero_training", "random"]:
        s = o[key]
        print(f"  {key:15s}: DTW={s['dtw']['mean']:.4f} [{s['dtw']['ci95'][0]:.4f}, {s['dtw']['ci95'][1]:.4f}]  "
              f"F1={s['f1']['mean']:.4f}  Ph={s['phase']['mean']:.4f}  (N={s['n']})")
    print(f"  CFM vs zero-training DTW win: {o['cfm_vs_zt_dtw_win']}%")
    print(f"  CFM vs random DTW win: {o['cfm_vs_random_dtw_win']}%")

    print(f"\n--- By Morphology ---")
    print(f"  {'Morphology':15s} {'N':>4s} {'CFM DTW':>8s} {'ZT DTW':>8s} {'Win%':>6s}")
    for morph, s in sorted(report["by_morphology"].items()):
        print(f"  {morph:15s} {s['n']:4d} {s['cfm_dtw']:8.4f} {s['zt_dtw']:8.4f} {s['cfm_vs_zt_win']:5.1f}%")

    if report.get("cross_vs_same_morph"):
        print(f"\n--- Cross vs Same Morphology ---")
        for key, s in report["cross_vs_same_morph"].items():
            print(f"  {key:15s}: N={s['n']}, CFM DTW={s['cfm_dtw']:.4f}, ZT DTW={s['zt_dtw']:.4f}, Win={s['cfm_vs_zt_win']}%")

    if report.get("by_split"):
        print(f"\n--- By Train/Test Split ---")
        for cat, s in report["by_split"].items():
            print(f"  {cat:15s}: N={s['n']}, CFM DTW={s['cfm_dtw']:.4f}, ZT DTW={s['zt_dtw']:.4f}, Win={s['cfm_vs_zt_win']}%")

    print(f"\n--- Top 5 Actions (by CFM improvement) ---")
    actions = sorted(report["by_action"].items(), key=lambda x: x[1]["improvement"], reverse=True)
    print(f"  {'Action':20s} {'N':>3s} {'CFM DTW':>8s} {'ZT DTW':>8s} {'Δ%':>7s}")
    for action, s in actions[:5]:
        print(f"  {action:20s} {s['n']:3d} {s['cfm_dtw']:8.4f} {s['zt_dtw']:8.4f} {s['improvement']:+6.1f}%")

    print(f"\n--- Bottom 5 Actions ---")
    for action, s in actions[-5:]:
        print(f"  {action:20s} {s['n']:3d} {s['cfm_dtw']:8.4f} {s['zt_dtw']:8.4f} {s['improvement']:+6.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--cfg_weight", type=float, default=2.0)
    args = parser.parse_args()
    evaluate_comprehensive(args.ckpt, args.max_pairs, args.n_steps, args.cfg_weight)
