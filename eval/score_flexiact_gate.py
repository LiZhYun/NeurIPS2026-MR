"""Pilot B analysis — score flexiact_gate outputs and summarise the pilot.

Reads the trained gate's g(t) curve, the training log for matched-vs-null gap
pre vs post, and the 30-pair outputs. Reports:
  - Learned g(t) curve shape (min/max/argmax).
  - C6 gap (matched vs null) at step 1 vs last step.
  - stratified contact_f1_vs_source (reuses eval.score_vmc_temporal_lora helpers).
  - v2 classifier label/behavior by re-running k_action_accuracy_v2_full.py
    (which auto-discovers the flexiact_gate dir).

Output: eval/results/k_compare/flexiact_gate/metrics_extended.json.
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

K_COMPARE = ROOT / 'eval/results/k_compare'
EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
UNIFIED = ROOT / 'idea-stage/unified_method_comparison_v2_full.json'
GATE_DIR = ROOT / 'save/B1_flexiact_gate'
METHODS = ['flexiact_gate', 'B1_no_gate', 'K', 'vmc_temporal_lora']


def load_gate_curve():
    p = GATE_DIR / 'g_of_t.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_train_log():
    p = GATE_DIR / 'train_log.jsonl'
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def describe_gate(curve):
    if not curve:
        return None
    g = np.asarray(curve['g'], dtype=np.float32)
    t = np.arange(len(g))
    flat_dev = float(np.max(np.abs(g - 0.5)))
    return {
        'num_timesteps': int(curve['num_timesteps']),
        'g_min': float(g.min()),
        'g_max': float(g.max()),
        'g_mean': float(g.mean()),
        'argmax_t': int(t[g.argmax()]),
        'argmin_t': int(t[g.argmin()]),
        'max_abs_dev_from_0p5': flat_dev,
        'g_first': float(g[0]),
        'g_last': float(g[-1]),
        'g_quartiles_t25_50_75': [float(g[len(g)//4]), float(g[len(g)//2]), float(g[3*len(g)//4])],
    }


def describe_c6(log):
    if not log:
        return None
    first = log[0]
    last = log[-1]
    return {
        'step_first': first.get('step'),
        'step_last': last.get('step'),
        'l_matched_first': first.get('l_matched'),
        'l_null_first': first.get('l_null'),
        'null_gap_first_pct': first.get('null_gap', 0.0) * 100.0,
        'l_matched_last': last.get('l_matched'),
        'l_null_last': last.get('l_null'),
        'null_gap_last_pct': last.get('null_gap', 0.0) * 100.0,
        'gap_grew_from_to_pct': [first.get('null_gap', 0.0) * 100.0,
                                 last.get('null_gap', 0.0) * 100.0],
    }


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


def stratum_of(p):
    if p['support_same_label'] == 0:
        return 'absent'
    if p['family_gap'] == 'near':
        return 'near_present'
    return p['family_gap']


def compute_contact_f1_stratified(methods):
    """Adapted from eval/score_vmc_temporal_lora.py."""
    from eval.quotient_extractor import extract_quotient

    COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
    CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
    motion_dir = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'

    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    src_sched_cache = {}

    results = {}
    for method in methods:
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
            if not (motion.ndim == 3 and motion.shape[2] == 13):
                continue
            m13 = motion  # [T,J,13]
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
            per_pair.append({'pair_id': pid, 'stratum': stratum_of(p),
                             'contact_f1_vs_source': c_f1})

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
                'contact_f1_vs_source': float(np.mean([e['contact_f1_vs_source']
                                                       for e in entries])),
            }
        results[method] = out
    return results


def run_v2_classifier():
    """Re-run the unified classifier comparison so flexiact_gate + B1_no_gate appear."""
    cmd = [sys.executable, '-m', 'eval.k_action_accuracy_v2_full']
    print('Running:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    # 1. Gate curve
    curve = load_gate_curve()
    gate_summary = describe_gate(curve)

    # 2. Training log (C6 gap pre vs post)
    log = load_train_log()
    c6_summary = describe_c6(log)

    # 3. Refresh unified classifier JSON so flexiact_gate + B1_no_gate get scored
    if (K_COMPARE / 'flexiact_gate').exists() or (K_COMPARE / 'B1_no_gate').exists():
        try:
            run_v2_classifier()
        except subprocess.CalledProcessError as e:
            print(f"WARN: k_action_accuracy_v2_full.py failed: {e}", flush=True)

    # 4. Pull v2 classifier numbers for our row and neighbours
    v2 = {}
    if UNIFIED.exists():
        uni = json.load(open(UNIFIED))
        for m in METHODS:
            if m in uni.get('methods', {}):
                v2[m] = uni['methods'][m]

    # 5. contact_f1_vs_source stratified
    cf1 = compute_contact_f1_stratified(METHODS)

    out = {
        'gate_curve_summary': gate_summary,
        'c6_gap_pre_vs_post': c6_summary,
        'contact_f1_vs_source_stratified': cf1,
        'v2_classifier_stratified': v2,
    }
    out_dir = K_COMPARE / 'flexiact_gate'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'metrics_extended.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}", flush=True)

    # Pretty summary to stdout
    print('\n=== Gate g(t) summary ===')
    if gate_summary:
        for k, v in gate_summary.items():
            print(f"  {k}: {v}")
    else:
        print("  (no gate curve found)")

    print('\n=== C6 gap pre vs post (matched − null / null, %) ===')
    if c6_summary:
        print(f"  step {c6_summary['step_first']}: gap={c6_summary['null_gap_first_pct']:+.2f}%  "
              f"(matched={c6_summary['l_matched_first']:.4f}, null={c6_summary['l_null_first']:.4f})")
        print(f"  step {c6_summary['step_last']}: gap={c6_summary['null_gap_last_pct']:+.2f}%  "
              f"(matched={c6_summary['l_matched_last']:.4f}, null={c6_summary['l_null_last']:.4f})")
    else:
        print("  (no training log)")

    print('\n=== contact_f1_vs_source (stratified) ===')
    strata_order = ['near_present', 'absent', 'moderate', 'extreme', 'all']
    header = f"{'method':22s}" + ''.join(f"{s:>14s}" for s in strata_order)
    print(header)
    for m in METHODS:
        if m not in cf1:
            continue
        row = f"{m:22s}"
        for s in strata_order:
            b = cf1[m].get(s)
            row += (f"{b['contact_f1_vs_source']:14.3f}" if b else f"{'—':>14s}")
        print(row)

    print('\n=== v2 classifier (label_match / behavior_preserved) ===')
    for m in METHODS:
        if m not in v2:
            continue
        by_s = v2[m].get('by_stratum', {})
        print(f"  {m}:")
        for s in ['near', 'absent', 'moderate', 'extreme', 'all']:
            b = by_s.get(s)
            if b:
                print(f"    {s:10s} n={b.get('n', 0):>2}  "
                      f"lbl={b.get('label_match_rate', 0):.3f}  "
                      f"beh={b.get('behavior_preserved_rate', 0):.3f}")
        pcs = v2[m].get('per_class_pred_overall', {})
        print(f"    per_class_pred_overall: {pcs}")


if __name__ == '__main__':
    main()
