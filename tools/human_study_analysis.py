"""Aggregate exported pairwise-study responses (from the HTML viewer's
'Export responses' button) into per-method win-rate tables, bucketed by
strata.

Usage:
  python tools/human_study_analysis.py \\
    --manifest idea-stage/human_study_manifest.json \\
    --responses anytop_pairwise_responses.json \\
    --out idea-stage/human_study_results.json

The responses file is a dict keyed by study_id; each value has
{question_key: choice, ..., meta: {...}} where choice is
"left" | "right" | "equal".
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def aggregate(manifest, responses):
    entries_by_id = {e["study_id"]: e for e in manifest["entries"]}
    questions = [q["key"] for q in manifest["questions"]]

    # totals[method][question]: wins, losses, ties
    totals = defaultdict(lambda: defaultdict(lambda: Counter()))
    # stratum_totals[stratum_key][method][question] -> Counter
    stratum_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: Counter())))

    answered = 0
    for study_id, resp in responses.items():
        entry = entries_by_id.get(study_id)
        if entry is None:
            continue
        left_method = entry["hidden"]["left_method"]
        right_method = entry["hidden"]["right_method"]
        strata = entry["strata"]
        skey = f"{strata['family_gap']}|sup={'yes' if strata['support_present'] else 'no'}"
        any_answered = False
        for qk in questions:
            choice = resp.get(qk)
            if choice not in {"left", "right", "equal"}:
                continue
            any_answered = True
            if choice == "left":
                totals[left_method][qk]["win"] += 1
                totals[right_method][qk]["loss"] += 1
                stratum_totals[skey][left_method][qk]["win"] += 1
                stratum_totals[skey][right_method][qk]["loss"] += 1
            elif choice == "right":
                totals[right_method][qk]["win"] += 1
                totals[left_method][qk]["loss"] += 1
                stratum_totals[skey][right_method][qk]["win"] += 1
                stratum_totals[skey][left_method][qk]["loss"] += 1
            else:  # equal
                totals[left_method][qk]["tie"] += 1
                totals[right_method][qk]["tie"] += 1
                stratum_totals[skey][left_method][qk]["tie"] += 1
                stratum_totals[skey][right_method][qk]["tie"] += 1
        if any_answered:
            answered += 1

    def to_rate(counter):
        total = counter["win"] + counter["loss"] + counter["tie"]
        if total == 0:
            return {"n": 0, "win_rate": None, "loss_rate": None, "tie_rate": None}
        return {
            "n": total,
            "win_rate": round(counter["win"] / total, 3),
            "loss_rate": round(counter["loss"] / total, 3),
            "tie_rate": round(counter["tie"] / total, 3),
        }

    global_summary = {
        m: {qk: to_rate(totals[m][qk]) for qk in questions} for m in totals
    }
    per_stratum = {
        skey: {m: {qk: to_rate(stratum_totals[skey][m][qk]) for qk in questions} for m in stratum_totals[skey]}
        for skey in stratum_totals
    }
    return {
        "n_comparisons_total": len(manifest["entries"]),
        "n_comparisons_answered": answered,
        "questions": questions,
        "global_win_rates": global_summary,
        "per_stratum_win_rates": per_stratum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("idea-stage/human_study_manifest.json"))
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("idea-stage/human_study_results.json"))
    args = ap.parse_args()
    manifest = load_json(args.manifest)
    responses = load_json(args.responses)
    result = aggregate(manifest, responses)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
