"""Compute L2 differences between Motion-Inversion output modes across the 30 eval pairs."""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
from pathlib import Path
import numpy as np

ROOT = Path(str(PROJECT_ROOT))
DIR_MI = ROOT / 'eval/results/k_compare/motion_inversion'
DIR_MI_DIS = ROOT / 'eval/results/k_compare/motion_inversion_disabled'
DIR_PLAIN = ROOT / 'eval/results/k_compare/motion_inversion_plainv/plain_v'
DIR_DIFF = ROOT / 'eval/results/k_compare/motion_inversion_plainv/diff_v'

with open(ROOT / 'idea-stage/eval_pairs.json') as f:
    pairs = json.load(f)['pairs']


def load_pair(d, pid, src_skel, tgt_skel):
    p = d / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
    if not p.exists():
        return None
    return np.load(p)


def l2(a, b):
    if a is None or b is None:
        return None
    n = min(a.shape[0], b.shape[0])
    return float(np.linalg.norm(a[:n] - b[:n]))


results = {'enabled_vs_disabled_per_pair': [],
           'plain_v_vs_diff_v_per_pair': []}
for p in pairs:
    pid = p['pair_id']
    src_skel = p['source_skel']
    tgt_skel = p['target_skel']
    me = load_pair(DIR_MI, pid, src_skel, tgt_skel)
    md = load_pair(DIR_MI_DIS, pid, src_skel, tgt_skel)
    d = l2(me, md)
    if d is not None:
        results['enabled_vs_disabled_per_pair'].append({'pair_id': pid, 'l2': d})

# Ablation pairs
abl_ids = [0, 10, 19]
for pid in abl_ids:
    p = next(pp for pp in pairs if pp['pair_id'] == pid)
    pv = load_pair(DIR_PLAIN, pid, p['source_skel'], p['target_skel'])
    dv = load_pair(DIR_DIFF, pid, p['source_skel'], p['target_skel'])
    d = l2(pv, dv)
    if d is not None:
        results['plain_v_vs_diff_v_per_pair'].append({'pair_id': pid, 'l2': d})

vals_ed = [e['l2'] for e in results['enabled_vs_disabled_per_pair']]
vals_pd = [e['l2'] for e in results['plain_v_vs_diff_v_per_pair']]
results['enabled_vs_disabled_summary'] = {
    'n': len(vals_ed), 'mean': float(np.mean(vals_ed)) if vals_ed else None,
    'min': float(np.min(vals_ed)) if vals_ed else None,
    'max': float(np.max(vals_ed)) if vals_ed else None,
}
results['plain_v_vs_diff_v_summary'] = {
    'n': len(vals_pd), 'mean': float(np.mean(vals_pd)) if vals_pd else None,
    'min': float(np.min(vals_pd)) if vals_pd else None,
    'max': float(np.max(vals_pd)) if vals_pd else None,
}
out = ROOT / 'save/B1_motion_inversion/ablation_l2.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print('Enabled vs disabled L2 (30 pairs):')
print(f"  mean={results['enabled_vs_disabled_summary']['mean']:.4f}  "
      f"min={results['enabled_vs_disabled_summary']['min']:.4f}  "
      f"max={results['enabled_vs_disabled_summary']['max']:.4f}")
print('Plain-V vs Differential-V L2 (3 pairs):')
print(f"  mean={results['plain_v_vs_diff_v_summary']['mean']:.4f}")
print('Saved:', out)
