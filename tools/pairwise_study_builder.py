"""Build a blinded human pairwise preference study manifest.

Reads idea-stage/eval_pairs.json and the renders produced by
tools/render_method_outputs.py, samples pairs stratified by family_gap x
support-same-label presence, and for each sampled pair selects K method-vs-method
comparisons (default 3 of C(5,2)=10). Left/right method assignment is
randomized, and study_id is opaque so ratings are blinded.

Writes idea-stage/human_study_manifest.json with records of shape:
  {
    "study_id": str,        # blinded, opaque
    "pair_id": int,
    "source_skel": str,
    "target_skel": str,
    "source_action": str,
    "strata": {"family_gap": str, "support_present": bool, "support_count": int},
    "source_path": str,
    "left":  {"method_code": "A|B|...", "path": str},
    "right": {"method_code": "A|B|...", "path": str},
    "hidden": {"left_method": str, "right_method": str}
  }

The viewer reads study_id + paths + strata; it does not read `hidden`
(keys are kept in the manifest so the analysis tool can unblind later).
"""

import argparse
import itertools
import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PAIRS_PATH = PROJECT_ROOT / "idea-stage" / "eval_pairs.json"
RENDER_ROOT = PROJECT_ROOT / "eval" / "results" / "k_compare" / "renders"
MANIFEST_PATH = PROJECT_ROOT / "idea-stage" / "human_study_manifest.json"

METHODS = ["K", "q_retrieval", "q_label_retrieval", "motion2motion", "label_random"]


def opaque_id(rng, length=10):
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def strata_key(pair):
    return (pair.get("family_gap", "?"), int(pair.get("support_same_label", 0)) > 0)


def stratified_sample(pairs, n_target, rng):
    """Sample n_target pairs, proportional to each stratum's size."""
    buckets = defaultdict(list)
    for p in pairs:
        buckets[strata_key(p)].append(p)
    total = len(pairs)
    selected = []
    # Proportional allocation with a floor of 1 per non-empty stratum
    for key, items in buckets.items():
        n_bucket = max(1, round(len(items) / total * n_target))
        rng.shuffle(items)
        selected.extend(items[:n_bucket])
    # Trim or grow to exactly n_target
    if len(selected) > n_target:
        rng.shuffle(selected)
        selected = selected[:n_target]
    elif len(selected) < n_target:
        remaining = [p for p in pairs if p not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n_target - len(selected)])
    selected.sort(key=lambda p: p["pair_id"])
    return selected


def build_manifest(args):
    rng = random.Random(args.seed)
    pairs = json.load(open(EVAL_PAIRS_PATH))["pairs"]

    # Require renders exist
    renders_available = {m: {p.name for p in (RENDER_ROOT / m).glob("pair_*.mp4")} for m in METHODS + ["source"]}

    # Filter to pairs that have all methods + source rendered (if --require-renders)
    if args.require_renders:
        keep = []
        for p in pairs:
            pid = p["pair_id"]
            fname = f"pair_{pid:02d}.mp4"
            missing = [m for m in METHODS + ["source"] if fname not in renders_available[m]]
            if missing:
                print(f"[skip pair_{pid:02d}] missing renders: {missing}")
                continue
            keep.append(p)
        pairs = keep

    n_pairs = min(args.n_pairs, len(pairs))
    sampled = stratified_sample(pairs, n_pairs, rng)

    # All method pair combinations
    all_combos = list(itertools.combinations(METHODS, 2))  # 10

    entries = []
    per_method_counts = Counter()
    per_pair_combos_counts = Counter()
    for pair in sampled:
        pid = pair["pair_id"]
        combos = rng.sample(all_combos, k=min(args.combos_per_pair, len(all_combos)))
        for combo in combos:
            a, b = combo
            # Randomize left/right
            if rng.random() < 0.5:
                left_method, right_method = a, b
            else:
                left_method, right_method = b, a
            entry = {
                "study_id": opaque_id(rng),
                "pair_id": pid,
                "source_skel": pair["source_skel"],
                "target_skel": pair["target_skel"],
                "source_action": pair["source_label"],
                "strata": {
                    "family_gap": pair.get("family_gap", "unknown"),
                    "support_present": int(pair.get("support_same_label", 0)) > 0,
                    "support_count": int(pair.get("support_same_label", 0)),
                },
                "source_path": str(Path("eval/results/k_compare/renders/source") / f"pair_{pid:02d}.mp4"),
                "left": {
                    "method_code": "L",
                    "path": str(Path(f"eval/results/k_compare/renders/{left_method}") / f"pair_{pid:02d}.mp4"),
                },
                "right": {
                    "method_code": "R",
                    "path": str(Path(f"eval/results/k_compare/renders/{right_method}") / f"pair_{pid:02d}.mp4"),
                },
                "hidden": {
                    "left_method": left_method,
                    "right_method": right_method,
                },
            }
            entries.append(entry)
            per_method_counts[left_method] += 1
            per_method_counts[right_method] += 1
            per_pair_combos_counts[pid] += 1

    # Balance check
    balance_ratio = 0.0
    if per_method_counts:
        counts = list(per_method_counts.values())
        balance_ratio = min(counts) / max(counts)

    manifest = {
        "meta": {
            "seed": args.seed,
            "n_pairs_sampled": len(sampled),
            "combos_per_pair": args.combos_per_pair,
            "n_comparisons": len(entries),
            "methods": METHODS,
            "blinded": True,
            "per_method_appearance": dict(per_method_counts),
            "method_balance_min_over_max": round(balance_ratio, 3),
            "strata_coverage": dict(Counter((p.get("family_gap", "?"), int(p.get("support_same_label", 0)) > 0) for p in sampled)),
        },
        "questions": [
            {"key": "natural", "text": "Which clip looks more natural as target-skeleton motion?",
             "choices": ["left", "right", "equal"]},
            {"key": "preserve", "text": "Which clip better preserves the source action?",
             "choices": ["left", "right", "equal"]},
        ],
        "entries": entries,
    }

    # Strata coverage json-serialisable
    manifest["meta"]["strata_coverage"] = {f"{k[0]}|sup={k[1]}": v for k, v in manifest["meta"]["strata_coverage"].items()}

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {MANIFEST_PATH}")
    print(f"  sampled pairs: {len(sampled)}, comparisons: {len(entries)}, balance min/max: {balance_ratio:.2f}")
    print(f"  per-method appearances: {dict(per_method_counts)}")
    print(f"  strata coverage: {manifest['meta']['strata_coverage']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-pairs", type=int, default=30,
                    help="Number of source pairs to sample for comparison")
    ap.add_argument("--combos-per-pair", type=int, default=3,
                    help="Method-pair combos per sampled pair (max C(5,2)=10)")
    ap.add_argument("--require-renders", action="store_true", default=True,
                    help="Skip pairs missing any method render")
    args = ap.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
