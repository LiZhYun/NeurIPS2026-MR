"""Paired Wilcoxon tests on per-query P3 rotation/noise ratios.

Tests run:
  Q1. ACE-T per-query ratio > 1.0 (NON-DEGENERATE vs noise floor)
  Q2. ACE-T per-query ratio > matched no_L_adv ratio (paired by seed)
  Q3. ACE-I per-query ratio > matched no_L_adv ratio
  Q4. no_L_adv ratio ≠ 1.0 (statistical distinguishability from noise floor)
  Q5. MoReFlow / AL-Flow ratio > 1.0
  Q6. Cross-method comparisons (MoReFlow vs ACE; AL-Flow vs MoReFlow; ACE vs no_zsrc)
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent / 'results'

def load_ratios(method):
    f = RESULTS_DIR / f'gauge_p3_{method}.json'
    d = json.loads(f.read_text())
    return np.array([q['rotation_over_noise_ratio'] for q in d['per_query']])

def main():
    methods = {
        'ACE-T_s42': load_ratios('ACE-T_s42'),
        'ACE-T_s43': load_ratios('ACE-T_s43'),
        'ACE-T_s44': load_ratios('ACE-T_s44'),
        'ACE-I_s42': load_ratios('ACE-I_s42'),
        'ACE-I_s43': load_ratios('ACE-I_s43'),
        'ACE-I_s44': load_ratios('ACE-I_s44'),
        'no_Ladv_s42': load_ratios('no_Ladv_s42'),
        'no_Ladv_s43': load_ratios('no_Ladv_s43'),
        'no_Ladv_s44': load_ratios('no_Ladv_s44'),
        'no_zsrc_s42': load_ratios('no_zsrc_s42'),
        'MoReFlow-T': load_ratios('MoReFlow-T'),
        'MoReFlow-I': load_ratios('MoReFlow-I'),
        'AL-Flow': load_ratios('AL-Flow'),
    }
    print(f"{'method':<20} {'n':>4} {'median':>8} {'mean':>8}")
    for k, v in methods.items():
        print(f"{k:<20} {len(v):>4} {np.median(v):>8.3f} {v.mean():>8.3f}")
    print()
    print('Q1. ACE-T ratio > 1.0:')
    for s in ['s42', 's43', 's44']:
        a = methods[f'ACE-T_{s}']
        r = wilcoxon(a - 1.0, alternative='greater')
        print(f'  ACE-T_{s} > 1.0: W={r.statistic:.0f} p={r.pvalue:.3e}')
    print('Q2. ACE-T > no_L_adv (paired by seed):')
    for s in ['s42', 's43', 's44']:
        a = methods[f'ACE-T_{s}']
        b = methods[f'no_Ladv_{s}']
        r = wilcoxon(a, b, alternative='greater')
        print(f'  ACE-T_{s} > no_Ladv_{s}: W={r.statistic:.0f} p={r.pvalue:.3e}')
    print('Q3. ACE-I > no_L_adv (paired by seed):')
    for s in ['s42', 's43', 's44']:
        a = methods[f'ACE-I_{s}']
        b = methods[f'no_Ladv_{s}']
        r = wilcoxon(a, b, alternative='greater')
        print(f'  ACE-I_{s} > no_Ladv_{s}: W={r.statistic:.0f} p={r.pvalue:.3e}')
    print('Q4. no_L_adv ratio ≠ 1.0 (= noise floor):')
    for s in ['s42', 's43', 's44']:
        b = methods[f'no_Ladv_{s}']
        r = wilcoxon(b - 1.0, alternative='two-sided')
        print(f'  no_Ladv_{s}: W={r.statistic:.0f} p={r.pvalue:.3e}  median(r-1)={np.median(b-1):.3f}')
    print('Q5. MoReFlow / AL-Flow ratio > 1.0:')
    for k in ['MoReFlow-T', 'MoReFlow-I', 'AL-Flow']:
        v = methods[k]
        r = wilcoxon(v - 1.0, alternative='greater')
        print(f'  {k}: W={r.statistic:.0f} p={r.pvalue:.3e}  median(r-1)={np.median(v-1):.3f}')
    print('Q6. Cross-method (paired by query_id):')
    pairs = [
        ('MoReFlow-T', 'ACE-T_s42'),
        ('MoReFlow-I', 'ACE-I_s42'),
        ('AL-Flow', 'MoReFlow-T'),
        ('AL-Flow', 'ACE-T_s42'),
        ('ACE-T_s42', 'no_zsrc_s42'),
        ('ACE-I_s42', 'no_zsrc_s42'),
    ]
    for a_key, b_key in pairs:
        a = methods[a_key]
        b = methods[b_key]
        if len(a) == len(b):
            r_two = wilcoxon(a, b, alternative='two-sided')
            r_grt = wilcoxon(a, b, alternative='greater')
            print(f'  {a_key} vs {b_key}: median(a-b)={np.median(a-b):+.3f}  two-sided p={r_two.pvalue:.3e}  one-sided(a>b) p={r_grt.pvalue:.3e}')


if __name__ == '__main__':
    main()
