"""Compute extended metrics for vmc_temporal_lora outputs:
  - contact_f1_vs_source (stratified)
  - acceleration smoothness: mean |a_t| where a_t = pos_{t+1}-2*pos_t+pos_{t-1}
  - jerk: mean |j_t| where j_t = a_{t+1} - a_t
  - Pull v2-classifier label/behavior rates from unified JSON

Also reports K, B1 in_context_retarget, and idea_F_topo_lora for reference.
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

K_COMPARE = ROOT / 'eval/results/k_compare'
EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
UNIFIED = ROOT / 'idea-stage/unified_method_comparison_v2_full.json'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'

METHODS = ['vmc_temporal_lora', 'K', 'in_context_retarget',
           'idea_F_topo_lora', 'motion2motion']


def contact_f1(pred, gt, thresh=0.5):
    p = (np.asarray(pred) >= thresh).astype(np.int8).ravel()
    g = (np.asarray(gt) >= thresh).astype(np.int8).ravel()
    n = min(p.size, g.size)
    p, g = p[:n], g[:n]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
    return float(2 * pr * rc / (pr + rc + 1e-8))


def grouped_contact_schedule(motion, tgt_skel, contact_groups):
    ch12 = motion[..., 12]
    contacts = (ch12 > 0.5).astype(np.float32)
    if tgt_skel not in contact_groups:
        return contacts.sum(axis=1, keepdims=True)
    groups = contact_groups[tgt_skel]
    names = sorted(groups.keys())
    T, J = contacts.shape
    sched = np.zeros((T, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        idxs = [int(j) for j in groups[name] if 0 <= int(j) < J]
        if idxs:
            sched[:, i] = contacts[:, idxs].mean(axis=1)
    return sched


def motion_to_positions(motion_13):
    """motion_13 [T, J, 13] -> [T, J, 3] root-relative positions (channels 0:3)."""
    return motion_13[..., 0:3].astype(np.float32)


def accel_jerk(positions):
    """positions [T, J, 3] → (mean|a|, mean|j|) over all joints × time."""
    T = positions.shape[0]
    if T < 4:
        return 0.0, 0.0
    a = positions[2:] - 2 * positions[1:-1] + positions[:-2]   # [T-2, J, 3]
    j = a[1:] - a[:-1]                                          # [T-3, J, 3]
    a_mag = np.linalg.norm(a, axis=-1)
    j_mag = np.linalg.norm(j, axis=-1)
    return float(a_mag.mean()), float(j_mag.mean())


def stratum_of(p):
    if p['support_same_label'] == 0:
        return 'absent'
    if p['family_gap'] == 'near':
        return 'near_present'
    return p['family_gap']  # moderate | extreme


def main():
    from eval.quotient_extractor import extract_quotient

    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)
    motion_dir = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'

    # Cache source contact schedules
    src_sched_cache = {}

    all_results = {}
    for method in METHODS:
        method_dir = K_COMPARE / method
        if not method_dir.exists():
            continue

        per_pair = []
        for p in pairs:
            pid = p['pair_id']
            path = method_dir / f"pair_{pid:02d}_{p['source_skel']}_to_{p['target_skel']}.npy"
            if not path.exists():
                continue
            try:
                motion = np.load(path)
            except Exception:
                continue
            # Tolerate either [T,J,13] or [J,T,13]
            if motion.ndim == 3 and motion.shape[2] == 13:
                m13 = motion  # assume [T,J,13]
            else:
                continue
            g_sched = grouped_contact_schedule(m13, p['target_skel'], contact_groups)
            pred_contact = g_sched.sum(axis=1)

            if p['source_fname'] not in src_sched_cache:
                try:
                    q = extract_quotient(p['source_fname'], cond[p['source_skel']],
                                         contact_groups=contact_groups,
                                         motion_dir=str(motion_dir))
                    src_sched_cache[p['source_fname']] = np.asarray(q['contact_sched']).sum(axis=1)
                except Exception:
                    continue
            src_contact = src_sched_cache[p['source_fname']]
            Ts = src_contact.shape[0]; Tp = pred_contact.shape[0]
            if Ts < 4 or Tp < 4:
                continue
            idx = np.clip(np.linspace(0, Ts - 1, Tp).astype(int), 0, Ts - 1)
            src_resampled = src_contact[idx]
            c_f1 = contact_f1(pred_contact, src_resampled)

            pos = motion_to_positions(m13)
            a_mean, j_mean = accel_jerk(pos)

            per_pair.append({
                'pair_id': pid, 'stratum': stratum_of(p),
                'contact_f1_vs_source': c_f1,
                'accel': a_mean, 'jerk': j_mean,
            })

        buckets = defaultdict(list)
        for e in per_pair:
            buckets[e['stratum']].append(e)
        buckets['all'] = list(per_pair)

        out = {}
        for name, entries in buckets.items():
            if not entries:
                continue
            out[name] = {
                'n': len(entries),
                'contact_f1_vs_source': float(np.mean([e['contact_f1_vs_source'] for e in entries])),
                'accel_mean':          float(np.mean([e['accel'] for e in entries])),
                'jerk_mean':           float(np.mean([e['jerk'] for e in entries])),
            }
        all_results[method] = out

    # V2 classifier pull
    v2 = {}
    if UNIFIED.exists():
        uni = json.load(open(UNIFIED))
        for m in METHODS:
            if m in uni.get('methods', {}):
                v2[m] = uni['methods'][m]

    # Pretty print
    strata_order = ['near_present', 'absent', 'moderate', 'extreme', 'all']
    print('\n=== contact_f1_vs_source | accel | jerk  (stratified) ===')
    header = f"{'method':25s}" + ''.join(f"{s:>12s}" for s in strata_order)
    print(header)
    for m in METHODS:
        if m not in all_results:
            continue
        row_cf1 = f"{m:25s}"
        for s in strata_order:
            b = all_results[m].get(s)
            row_cf1 += (f"{b['contact_f1_vs_source']:12.3f}" if b else f"{'—':>12s}")
        print(row_cf1 + '  [contact_f1]')
        row_a = f"{'':25s}"
        for s in strata_order:
            b = all_results[m].get(s)
            row_a += (f"{b['accel_mean']:12.4f}" if b else f"{'—':>12s}")
        print(row_a + '  [accel]')
        row_j = f"{'':25s}"
        for s in strata_order:
            b = all_results[m].get(s)
            row_j += (f"{b['jerk_mean']:12.4f}" if b else f"{'—':>12s}")
        print(row_j + '  [jerk]')

    print('\n=== v2 classifier (from unified JSON) ===')
    for m in METHODS:
        if m not in v2:
            continue
        by_s = v2[m].get('by_stratum', {})
        print(f"  {m}:")
        for s in ['near', 'absent', 'moderate', 'extreme', 'all']:
            b = by_s.get(s)
            if b:
                print(f"    {s:10s} n={b.get('n',0):>2}  "
                      f"lbl={b.get('label_match_rate',0):.3f}  "
                      f"beh={b.get('behavior_preserved_rate',0):.3f}")

    out_path = K_COMPARE / 'vmc_temporal_lora' / 'metrics_extended.json'
    out_path.write_text(json.dumps({
        'methods': all_results,
        'v2_classifier': {m: v2[m] for m in v2},
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
