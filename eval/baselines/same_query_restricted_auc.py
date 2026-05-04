"""Same-query restricted AUC comparison (Codex R2 NEW W1 fix).

For all baselines that have V5 procrustes results, compute cluster-tier AUC
restricted to AL-Flow's 159 OK queries per fold. This addresses the codex critique:
"AL-Flow looks like an outlier on the full 300 query set, but on the same 159 OK
subset it is approximately tied with ActionBridge_v2."

The eval_v5.py script computes overall AUC over all OK queries; we filter the
input to only the AL-Flow OK queries first.

Usage:
  python -m eval.baselines.same_query_restricted_auc
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHODS = {
    'anytop_v5':              'eval/results/baselines/anytop_v5/fold_{fold}',
    'm2m_lite':               'eval/results/baselines/m2m_lite_v5/fold_{fold}',
    'm2m_official':           'eval/results/baselines/m2m_official_v5/fold_{fold}',
    'ace_primary_70':         'save/ace_inference/v3/ace_primary_70_v5/fold_{fold}',
    'ace_inductive_60':       'save/ace_inference/v3/ace_inductive_60_v5/fold_{fold}',
    'actionbridge_v2_final':  'save/actionbridge_inference/v2_final_v5/fold_{fold}',
    'dpg_v1':                 'save/dpg/dpg_v1/fold_{fold}',
    'dpg_sb_v2_step12k':      'save/dpg_sb/dpg_sb_v2_step12k/fold_{fold}',
    'dpg_sb_v3':              'save/dpg_sb/dpg_sb_v3/fold_{fold}',
    'anchor_label_flow':      'eval/results/baselines/anchor_label_flow_v5/fold_{fold}',
    'anchor_label_flow_euler20': 'eval/results/baselines/anchor_label_flow_v5_euler20/fold_{fold}',
    'm3A_rerank_noaction_v2': 'save/m3/m3A_rerank_noaction_v2/fold_{fold}',
    'm3_rerank_v1':           'save/m3/m3_rerank_v1/fold_{fold}',
    'm3_cqpred':              'save/m3/m3_cqpred/fold_{fold}',
    'i5_action_classifier_v5': 'save/oracles/v3/i5_action_classifier_v5/fold_{fold}',
    'action_oracle_v5':       'save/oracles/v3/action_oracle_v5/fold_{fold}',
    'self_positive_v5':       'save/oracles/v3/self_positive_v5/fold_{fold}',
    'random_skel_b_v5':       'save/oracles/v3/random_skel_b_v5/fold_{fold}',
}


def get_ok_qids(method_dir: Path) -> set:
    """Return set of query_id values whose status == 'ok' in metrics.json."""
    mf = method_dir / 'metrics.json'
    if not mf.exists():
        return set()
    try:
        m = json.load(open(mf))
        if 'per_query' in m:
            return {q['query_id'] for q in m['per_query'] if q.get('status') == 'ok'}
        return set()
    except Exception:
        return set()


def restrict_per_query_auc(json_path: Path, target_qids: set) -> dict:
    """Read v5_eval_procrustes.json's per_query, restrict to target_qids,
    and compute cluster-tier overall AUC + bootstrap CI on the restricted set.
    """
    if not json_path.exists():
        return {'auc': None, 'lo': None, 'hi': None, 'n': 0}
    d = json.load(open(json_path))
    per_q = d.get('per_query', [])
    # Filter
    filtered = [q for q in per_q if q['query_id'] in target_qids]
    aucs = [q.get('cluster_auc') for q in filtered if q.get('cluster_auc') is not None]
    if not aucs:
        return {'auc': None, 'lo': None, 'hi': None, 'n': 0}
    import numpy as np
    rng = np.random.RandomState(0)
    n_boot = 1000
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(len(aucs), len(aucs), replace=True)
        boot.append(np.mean([aucs[i] for i in idx]))
    boot = np.sort(boot)
    return {
        'auc': float(np.mean(aucs)),
        'lo': float(boot[int(n_boot * 0.025)]),
        'hi': float(boot[int(n_boot * 0.975)]),
        'n': len(aucs),
    }


def main():
    print("Same-query restricted cluster-tier AUC comparison")
    print("Restriction set: queries where AL-Flow status == 'ok' (each fold separately)")
    print()

    for fold in [42, 43]:
        # AL-Flow's OK qids define the restriction
        alflow_dir = PROJECT_ROOT / f'eval/results/baselines/anchor_label_flow_v5/fold_{fold}'
        target_qids = get_ok_qids(alflow_dir)
        print(f"=== fold {fold} | AL-Flow OK qids: {len(target_qids)} ===")
        print(f"{'Method':32s} {'AUC':>20s}  {'CI':>20s}  n")

        rows = []
        for name, dir_template in METHODS.items():
            mdir = PROJECT_ROOT / dir_template.format(fold=fold)
            jf = mdir / 'v5_eval_procrustes.json'
            r = restrict_per_query_auc(jf, target_qids)
            rows.append((name, r))

        # Sort by AUC descending
        rows = sorted(rows, key=lambda x: -(x[1].get('auc') or -1))
        for name, r in rows:
            if r['auc'] is None:
                print(f"{name:32s} {'(missing)':>20s}")
                continue
            print(f"{name:32s} {r['auc']:>20.4f}  [{r['lo']:.3f},{r['hi']:.3f}]  n={r['n']}")
        print()

    # Also write a JSON summary
    summary = {}
    for fold in [42, 43]:
        target_qids = get_ok_qids(PROJECT_ROOT / f'eval/results/baselines/anchor_label_flow_v5/fold_{fold}')
        summary[f'fold_{fold}'] = {
            'n_target_qids': len(target_qids),
            'methods': {}
        }
        for name, dir_template in METHODS.items():
            mdir = PROJECT_ROOT / dir_template.format(fold=fold)
            r = restrict_per_query_auc(mdir / 'v5_eval_procrustes.json', target_qids)
            summary[f'fold_{fold}']['methods'][name] = r

    out_path = PROJECT_ROOT / 'eval/results/same_query_restricted_auc.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
