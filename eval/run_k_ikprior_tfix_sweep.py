"""t_fix sweep for Idea N (K_ikprior). Reviewer's Option C diagnostic.

For each t_fix in {5, 15, 30, 60, 90}, run K_ikprior on all 30 eval pairs and
preserve metrics.json under a per-t_fix filename. Then aggregate prior_first /
prior_last trajectories + skating + contact F1 per t_fix.

Output: idea-stage/idea_N_tfix_sweep.json

30 pairs × 5 t_fix × ~10 s/pair = ~25 min total.
"""
from __future__ import annotations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json, os, shutil, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

SWEEP_OUT = ROOT / 'idea-stage/idea_N_tfix_sweep.json'
METRICS_SRC = ROOT / 'eval/results/k_compare/K_ikprior/metrics.json'
PRESERVED_DIR = ROOT / 'eval/results/k_compare/K_ikprior_tfix_sweep'
PRESERVED_DIR.mkdir(parents=True, exist_ok=True)

T_FIXES = [5, 15, 30, 60, 90]


def summarise(metrics: dict) -> dict:
    per_pair = [p for p in metrics.get('per_pair', []) if p.get('status') == 'ok'
                and p.get('prior_first') is not None and p.get('prior_last') is not None]
    rel = [(p['prior_last'] - p['prior_first']) / max(p['prior_first'], 1e-9)
           for p in per_pair]
    decreased = sum(1 for p in per_pair if p['prior_last'] < p['prior_first'])
    skate = [p['skating_proxy'] for p in per_pair if p.get('skating_proxy') is not None]
    cf1 = [p['contact_f1'] for p in per_pair if p.get('contact_f1') is not None]
    return {
        'n_ok': len(per_pair),
        'n_decreased': decreased,
        'frac_decreased': decreased / max(len(per_pair), 1),
        'mean_rel_change_pct': float(np.mean(rel) * 100) if rel else None,
        'median_rel_change_pct': float(np.median(rel) * 100) if rel else None,
        'mean_skating': float(np.mean(skate)) if skate else None,
        'mean_contact_f1': float(np.mean(cf1)) if cf1 else None,
        'mean_prior_first': float(np.mean([p['prior_first'] for p in per_pair])) if per_pair else None,
        'mean_prior_last': float(np.mean([p['prior_last'] for p in per_pair])) if per_pair else None,
    }


def main():
    from eval.run_k_ikprior_30pairs import run as run_ikprior

    results = {}
    t_start = time.time()
    for tf in T_FIXES:
        print(f"\n{'='*70}\n  t_fix = {tf}  \n{'='*70}")
        t0 = time.time()
        run_ikprior(w_prior=0.3, prior_every=25, t_fix=tf, n_iters=400)
        dur = time.time() - t0
        # Preserve metrics.json
        dst = PRESERVED_DIR / f'metrics_tfix_{tf}.json'
        shutil.copy(METRICS_SRC, dst)
        metrics = json.load(open(dst))
        summary = summarise(metrics)
        summary['wall_time_s'] = dur
        results[str(tf)] = summary
        print(f"\n  t_fix={tf}: decreased {summary['n_decreased']}/{summary['n_ok']}  "
              f"mean rel Δ={summary['mean_rel_change_pct']:+.1f}%  "
              f"skate={summary['mean_skating']:.2f}  "
              f"cF1={summary['mean_contact_f1']:.3f}  "
              f"wall={dur:.1f}s")

    total_s = time.time() - t_start
    aggregate = {
        'total_runtime_s': total_s,
        't_fixes': T_FIXES,
        'fixed_params': {'w_prior': 0.3, 'prior_every': 25, 'n_iters': 400},
        'n_pairs_per_t_fix': 30,
        'by_t_fix': results,
    }
    SWEEP_OUT.write_text(json.dumps(aggregate, indent=2))
    print(f"\n=== DONE: total {total_s:.1f}s ===")
    print(f"saved: {SWEEP_OUT}")
    print("\n | t_fix | n_dec | frac_dec | mean Δ% | median Δ% | skate | cF1   | prior_first | prior_last |")
    print(" |-------|-------|----------|---------|-----------|-------|-------|-------------|------------|")
    for tf in T_FIXES:
        s = results[str(tf)]
        print(f" |  {tf:3d}  |  {s['n_decreased']:2d}   |  {s['frac_decreased']:.2f}    | "
              f"{s['mean_rel_change_pct']:+7.1f} | {s['median_rel_change_pct']:+8.1f}  |"
              f" {s['mean_skating']:.2f}  | {s['mean_contact_f1']:.3f} | "
              f"{s['mean_prior_first']:.2e} | {s['mean_prior_last']:.2e} |")


if __name__ == '__main__':
    main()
