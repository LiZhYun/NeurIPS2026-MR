"""Figure: Synthetic 2x2 baseline ladder + cell-mean ceiling visualization.

Shows the structural cell-mean degeneracy directly:
  - Zero / random Gaussian baselines: MSE ≈ 22-24 (target variance scale)
  - Random clip same (skel, action): MSE ≈ 5.3 (no learning, just sampling within cell)
  - Source-blind cell-mean predictor E[T*(x_a) | skel_b, action]: MSE ≈ 2.0 — the
    irreducible source-blind ceiling (the cell-mean degeneracy)
  - Trained SmallFlowGen (4k or 10k steps): MSE ≈ 5.1 (engineering loss above ceiling)
  - Oracle with x_a (true T*(x_a)): MSE = 0.0 (perfect with source conditioning)

Headline: the gap from cell-mean (2.0) to oracle (0.0) is the structural cell-mean
degeneracy — eliminable only by conditioning on the source motion.

Reads: save/synthetic_2x2/oracle_baselines_summary.json + summary_2x2.json
Output: figures/fig_synth_2x2_ladder.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_FILE = PROJECT_ROOT / 'save/synthetic_2x2/oracle_baselines_summary.json'
SUMMARY_FILE = PROJECT_ROOT / 'save/synthetic_2x2/summary_2x2.json'
OUT_PDF = PROJECT_ROOT / 'figures/fig_synth_2x2_ladder.pdf'
OUT_PNG = PROJECT_ROOT / 'figures/fig_synth_2x2_ladder.png'


def main():
    oracle = json.load(open(ORACLE_FILE))
    trained_summary = json.load(open(SUMMARY_FILE))

    cells = ['sparse_unpaired', 'sparse_paired', 'dense_unpaired', 'dense_paired']
    cell_label = {
        'sparse_unpaired': 'sparse / unpaired',
        'sparse_paired':   'sparse / paired',
        'dense_unpaired':  'dense / unpaired',
        'dense_paired':    'dense / paired',
    }

    # Build per-cell (baseline → MSE) ladders + add trained-model marker
    bars = ['zero', 'random_gaussian', 'random_same_action_clip',
            'source_blind_cell_mean', 'oracle_with_xa']
    bar_labels = {
        'zero':                    'zero',
        'random_gaussian':         'random Gaussian',
        'random_same_action_clip': 'random clip\nin same cell',
        'source_blind_cell_mean':  'source-blind\ncell mean (ceiling)',
        'oracle_with_xa':          'oracle\nwith $x_a$',
    }
    bar_color = {
        'zero':                    '#cccccc',
        'random_gaussian':         '#999999',
        'random_same_action_clip': '#7f7f7f',
        'source_blind_cell_mean':  '#ff7f0e',  # orange — cell-mean ceiling
        'oracle_with_xa':          '#2ca02c',  # green — oracle
    }

    # Trained-model MSE per cell (from summary_2x2.json — list of dicts)
    trained_mse = {m['cell']: m['overall_macro_mse'] for m in trained_summary}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_bars = len(bars)
    n_cells = len(cells)
    bar_width = 0.16
    x_base = np.arange(n_cells)

    for bi, b in enumerate(bars):
        vals = [oracle[c][b]['overall_macro_mse'] for c in cells]
        x_pos = x_base + (bi - n_bars/2 + 0.5) * bar_width
        ax.bar(x_pos, vals, width=bar_width, color=bar_color[b],
               edgecolor='black', linewidth=0.4, label=bar_labels[b])

    # Trained-model marker per cell (red diamond)
    trained_x = x_base
    trained_y = [trained_mse[c] for c in cells]
    ax.scatter(trained_x, trained_y, marker='D', s=120, color='#d62728',
               edgecolors='black', linewidths=1.2, zorder=10,
               label='trained SmallFlowGen\n(256d×6L, 10k steps)')

    # Annotate the structural gap (cell-mean → oracle)
    cm_y = [oracle[c]['source_blind_cell_mean']['overall_macro_mse'] for c in cells]
    for ci, c in enumerate(cells):
        ax.annotate('', xy=(x_base[ci] + 1.5*bar_width, 0.0),
                    xytext=(x_base[ci] + 1.5*bar_width, cm_y[ci]),
                    arrowprops=dict(arrowstyle='|-|', color='red', lw=1.2, mutation_scale=4))
        ax.text(x_base[ci] + 1.7*bar_width, cm_y[ci]/2, f'$\\Delta\\approx${cm_y[ci]:.1f}',
                fontsize=8, color='red', va='center')

    ax.set_xticks(x_base)
    ax.set_xticklabels([cell_label[c] for c in cells], fontsize=10)
    ax.set_ylabel('overall macro MSE (cross-skel transport recovery)', fontsize=10)
    ax.set_title('Synthetic 2$\\times$2 cell-mean degeneracy: structural source-blind ceiling = 2.0 MSE\n'
                 '(oracle with $x_a$ = 0.0; gap is the irreducible cell-mean averaging error)',
                 fontsize=11.5, pad=8)
    ax.set_ylim(0, 27)
    ax.legend(loc='upper center', ncol=3, fontsize=9, framealpha=0.92)
    ax.grid(True, axis='y', ls=':', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches='tight', dpi=200)
    plt.savefig(OUT_PNG, bbox_inches='tight', dpi=180)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == '__main__':
    main()
