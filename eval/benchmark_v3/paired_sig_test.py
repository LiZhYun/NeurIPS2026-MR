"""Paired significance test: per-query AUC of method A vs method B.

Computes:
  - Win-count: how many queries does A score higher than B
  - Sign test p-value (binomial)
  - Wilcoxon signed-rank test
  - Block-bootstrap 95% CI on mean difference (block by skel_pair)

Usage:
  python -m eval.benchmark_v3.paired_sig_test \
      --method_a save/m3/m3_rerank_v1 --name_a M3_PhaseA \
      --method_b eval/results/baselines/k_retrieve_v5 --name_b k_retrieve \
      --folds 42 43
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import binom_test, wilcoxon


def load_per_query(method_dir: Path, fold: int, distance: str):
    """Return dict {query_id: {cluster_auc, exact_auc, split, skel_pair}}."""
    p = method_dir / f'fold_{fold}' / f'v5_eval_{distance}.json'
    if not p.exists():
        # try alt path
        p2 = method_dir / f'v3_fold_{fold}' / f'v5_eval_{distance}.json'
        if not p2.exists(): return {}
        p = p2
    data = json.load(open(p))
    out = {}
    for r in data['per_query']:
        qid = r['query_id']
        out[qid] = {
            'cluster_auc': r.get('cluster_auc'),
            'exact_auc': r.get('exact_auc'),
            'split': r['split'],
            'skel_pair': f"{r['skel_a']}__{r['skel_b']}",
            'cluster': r['cluster'],
        }
    return out


def block_bootstrap_ci(values_with_blocks, n_boot=1000, ci=0.95, seed=42):
    if not values_with_blocks: return (0.0, 0.0, 0.0, 0)
    rng = np.random.RandomState(seed)
    block_to_values = defaultdict(list)
    for v, b in values_with_blocks:
        if v is None: continue
        block_to_values[b].append(v)
    blocks = list(block_to_values.keys())
    if not blocks: return (0.0, 0.0, 0.0, 0)
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(len(blocks), len(blocks), replace=True)
        all_vals = []
        for bi in sampled:
            all_vals.extend(block_to_values[blocks[bi]])
        means.append(np.mean(all_vals) if all_vals else 0.0)
    means = np.array(means)
    raw_mean = float(np.mean([v for v, _ in values_with_blocks if v is not None]))
    return (
        float(np.percentile(means, (1 - ci) / 2 * 100)),
        raw_mean,
        float(np.percentile(means, (1 + ci) / 2 * 100)),
        len([v for v, _ in values_with_blocks if v is not None]),
    )


def paired_test_one_fold(qa_per_query, qb_per_query, tier='cluster_auc',
                         splits=('test_test',)):
    """Pair matching queries (must exist in both A and B), compute diffs, sig tests."""
    pairs = []
    for qid, ra in qa_per_query.items():
        if qid not in qb_per_query: continue
        rb = qb_per_query[qid]
        if ra['split'] not in splits: continue
        a = ra.get(tier); b = rb.get(tier)
        if a is None or b is None: continue
        pairs.append({
            'qid': qid, 'a': a, 'b': b, 'diff': a - b,
            'split': ra['split'], 'skel_pair': ra['skel_pair'],
            'cluster': ra['cluster'],
        })
    if not pairs:
        return {'n': 0}

    diffs = np.array([p['diff'] for p in pairs])
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses

    sign_p = binom_test(wins, n, p=0.5, alternative='greater') if n > 0 else 1.0

    try:
        wilcoxon_p = float(wilcoxon(diffs, alternative='greater').pvalue) if (diffs != 0).any() else 1.0
    except Exception:
        wilcoxon_p = 1.0

    diff_ci = block_bootstrap_ci([(p['diff'], p['skel_pair']) for p in pairs])
    a_ci = block_bootstrap_ci([(p['a'], p['skel_pair']) for p in pairs])
    b_ci = block_bootstrap_ci([(p['b'], p['skel_pair']) for p in pairs])

    return {
        'n': len(pairs),
        'wins': wins, 'losses': losses, 'ties': len(pairs) - n,
        'mean_diff': float(diffs.mean()),
        'median_diff': float(np.median(diffs)),
        'sign_test_p': float(sign_p),
        'wilcoxon_p': wilcoxon_p,
        'win_rate': wins / max(n, 1),
        'diff_ci': diff_ci,    # (lo, mean, hi, n)
        'a_ci': a_ci,
        'b_ci': b_ci,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_a', required=True, help='Path to method A (with fold_NN/v5_eval_X.json)')
    parser.add_argument('--name_a', default='A')
    parser.add_argument('--method_b', required=True, help='Path to method B')
    parser.add_argument('--name_b', default='B')
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--distance', default='all')
    parser.add_argument('--splits', nargs='+', default=['test_test'])
    parser.add_argument('--out_path', default=None)
    args = parser.parse_args()

    distances = ('procrustes', 'zscore_dtw', 'q_component') if args.distance == 'all' else [args.distance]

    method_a = Path(args.method_a) if Path(args.method_a).is_absolute() else PROJECT_ROOT / args.method_a
    method_b = Path(args.method_b) if Path(args.method_b).is_absolute() else PROJECT_ROOT / args.method_b

    results = {}
    for fold in args.folds:
        for dist in distances:
            qa = load_per_query(method_a, fold, dist)
            qb = load_per_query(method_b, fold, dist)
            if not qa or not qb:
                print(f"fold {fold} {dist}: SKIP (missing data: {len(qa)} vs {len(qb)})")
                continue
            for tier in ('cluster_auc', 'exact_auc'):
                key = f"fold_{fold}_{dist}_{tier}"
                results[key] = paired_test_one_fold(qa, qb, tier=tier, splits=args.splits)
                r = results[key]
                if r['n'] == 0:
                    print(f"  {key}: n=0 (no overlapping queries)")
                    continue
                print(f"  {key}: n={r['n']} {args.name_a} {r['a_ci'][1]:.3f} vs {args.name_b} {r['b_ci'][1]:.3f}")
                print(f"    diff={r['mean_diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][2]:+.3f}], "
                      f"win-count={r['wins']}/{r['n']} ({100*r['win_rate']:.0f}%), "
                      f"sign_p={r['sign_test_p']:.4f}, wilcoxon_p={r['wilcoxon_p']:.4f}")

    out_path = Path(args.out_path) if args.out_path else None
    if out_path:
        with open(out_path, 'w') as f:
            json.dump({'method_a': args.name_a, 'method_b': args.name_b,
                       'folds': args.folds, 'splits': args.splits,
                       'results': results}, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
