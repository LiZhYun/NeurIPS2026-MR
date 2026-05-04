"""Figure: Same-query restricted AUC across method families (gens vs retrieval vs oracles).

Visualizes Codex R3+R4's central concession + author's reframe:
- ALL generators (label-conditioned or not, source-conditioned or not) cluster at 0.50-0.55
- ALL retrieval methods (with same labels) cluster at 0.77-0.79
- Oracles at 0.87-0.90
- The structural gap is NOT explained by metadata availability.

Reads: eval/results/same_query_restricted_auc.json
Output: figures/fig_same_query_auc.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / 'eval/results/same_query_restricted_auc.json'
OUT_PDF = PROJECT_ROOT / 'figures/fig_same_query_auc.pdf'
OUT_PNG = PROJECT_ROOT / 'figures/fig_same_query_auc.png'

# Method ordering and family labels — drives bar color groups
FAMILIES = [
    # (display_name, internal_key, family)
    ('M2M (lite)',          'm2m_lite',                    'gen_no_labels'),
    ('M2M (official)',      'm2m_official',                'gen_no_labels'),
    ('AnyTop',              'anytop_v5',                   'gen_no_labels'),
    ('ACE (inductive)',     'ace_inductive_60',            'gen_no_labels'),
    ('ACE (primary)',       'ace_primary_70',              'gen_no_labels'),
    ('DPG-v1',              'dpg_v1',                      'gen_label_only'),
    ('ActionBridge-v2',     'actionbridge_v2_final',       'gen_label_only'),
    ('LSB-v2',              'dpg_sb_v2_step12k',           'gen_src+label'),
    ('LSB-v3 (NEW)',        'dpg_sb_v3',                   'gen_src+label'),
    ('AL-Flow (NEW)',       'anchor_label_flow',           'gen_label_only'),
    ('AL-Flow 20-step (NEW)', 'anchor_label_flow_euler20', 'gen_label_only'),
    ('Random',              'random_skel_b_v5',            'reference'),
    ('M3-A retrieval',      'm3A_rerank_noaction_v2',      'retrieval'),
    ('M3 retrieval',        'm3_rerank_v1',                'retrieval'),
    ('M3-CqPred',           'm3_cqpred',                   'retrieval'),
    ('I-5 classifier',      'i5_action_classifier_v5',     'retrieval'),
    ('Self-positive',       'self_positive_v5',            'oracle'),
    ('Action oracle',       'action_oracle_v5',            'oracle'),
]

FAMILY_COLOR = {
    'gen_no_labels':   '#1f77b4',  # blue — unconditioned generators
    'gen_label_only':  '#ff7f0e',  # orange — label-cond generators
    'gen_src+label':   '#9467bd',  # purple — source+label-cond generators
    'reference':       '#7f7f7f',  # gray — random reference
    'retrieval':       '#2ca02c',  # green — retrieval methods
    'oracle':          '#d62728',  # red — oracle upper bounds
}
FAMILY_LABEL = {
    'gen_no_labels':   'Generator (no labels)',
    'gen_label_only':  'Generator (label-cond)',
    'gen_src+label':   'Generator (src+label)',
    'reference':       'Random reference',
    'retrieval':       'Retrieval (label)',
    'oracle':          'Oracle upper bound',
}


def main():
    d = json.load(open(INPUT))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    for ax_i, fold in enumerate([42, 43]):
        ax = axes[ax_i]
        data = d[f'fold_{fold}']['methods']
        n_qid = d[f'fold_{fold}']['n_target_qids']

        rows = []
        for name, key, fam in FAMILIES:
            r = data.get(key, {})
            if r.get('auc') is None:
                continue
            rows.append((name, fam, r['auc'], r['lo'], r['hi'], r['n']))
        # Sort by AUC ascending so bars go left-to-right from worst to best
        rows = sorted(rows, key=lambda x: x[2])

        names = [r[0] for r in rows]
        aucs = [r[2] for r in rows]
        lows = [r[3] for r in rows]
        highs = [r[4] for r in rows]
        colors = [FAMILY_COLOR[r[1]] for r in rows]

        y = np.arange(len(names))
        ax.barh(y, aucs, color=colors, edgecolor='black', linewidth=0.5)
        ax.errorbar(aucs, y,
                    xerr=[np.array(aucs) - np.array(lows),
                          np.array(highs) - np.array(aucs)],
                    fmt='none', color='black', lw=1.0, capsize=2.5)
        ax.axvline(0.5, color='gray', linestyle=':', lw=1, alpha=0.6, label='chance (AUC=0.5)')
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel('cluster-tier AUC (procrustes)', fontsize=10)
        ax.set_xlim(0.40, 0.95)
        ax.set_title(f'V5 fold {fold} — n={n_qid} (AL-Flow OK queries)', fontsize=11, weight='bold')
        ax.grid(True, axis='x', alpha=0.3, ls=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend on first axis
    handles = [plt.Rectangle((0,0),1,1, color=c, label=FAMILY_LABEL[f])
               for f, c in FAMILY_COLOR.items()]
    handles.insert(0, plt.Line2D([0],[0], color='gray', linestyle=':', label='chance'))
    axes[0].legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9, ncol=1)

    fig.suptitle('Same-query restricted comparison: ALL generators (with or without labels) sit at chance,\n'
                 'while retrieval methods (with same labels) reach 0.77-0.81. The gap is structural, not metadata-driven.',
                 fontsize=11.5, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches='tight', dpi=200)
    plt.savefig(OUT_PNG, bbox_inches='tight', dpi=180)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == '__main__':
    main()
