"""Compute contact_f1_vs_source stratified metrics for in_context_retarget outputs.

Follows the same metric definition as run_retrieve_refine_v2:
  - Binarize (pred_motion's aggregated contact fraction >= 0.5)
  - vs source's aggregate contact_sched (from quotient cache).
  - F1 between binary time series (aligned to min length).

Also aggregates the v2-classifier labels already computed in
idea-stage/unified_method_comparison_v2_full.json.
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

METHOD_DIR = ROOT / 'eval/results/k_compare/in_context_retarget'
EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
Q_CACHE = ROOT / 'idea-stage/quotient_cache.npz'
UNIFIED = ROOT / 'idea-stage/unified_method_comparison_v2_full.json'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'


def contact_f1(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> float:
    p = (np.asarray(pred) >= thresh).astype(np.int8).ravel()
    g = (np.asarray(gt) >= thresh).astype(np.int8).ravel()
    n = min(p.size, g.size)
    p, g = p[:n], g[:n]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    return float(2 * pr * rc / (pr + rc + 1e-8))


def agg_contact_from_motion(motion: np.ndarray) -> np.ndarray:
    """motion [T, J, 13] → per-frame contact fraction = mean(ch12 > 0.5).

    Matches compute_contact_schedule_aggregate from eval.quotient_extractor:
      contacts.sum(axis=1) / J  with J = motion.shape[1].
    """
    ch12 = motion[..., 12]  # [T, J]
    contacts = (ch12 > 0.5).astype(np.float32)
    return contacts.sum(axis=1) / max(motion.shape[1], 1)  # [T]


def grouped_contact_schedule(motion: np.ndarray, tgt_skel: str, contact_groups: dict) -> np.ndarray:
    """Compute grouped contact schedule [T, C] for generated motion on target skeleton."""
    ch12 = motion[..., 12]  # [T, J]
    contacts = (ch12 > 0.5).astype(np.float32)
    if tgt_skel not in contact_groups:
        return contacts.sum(axis=1, keepdims=True)  # [T, 1] fallback
    groups = contact_groups[tgt_skel]
    names = sorted(groups.keys())
    T, J = contacts.shape
    sched = np.zeros((T, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        idxs = [int(j) for j in groups[name] if 0 <= int(j) < J]
        if idxs:
            sched[:, i] = contacts[:, idxs].mean(axis=1)
    return sched


def main():
    from eval.quotient_extractor import extract_quotient

    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)
    motion_dir = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'

    # Cache source grouped contact schedules
    src_sched_cache = {}

    per_pair = []
    for p in pairs:
        pid = p['pair_id']
        method_path = METHOD_DIR / f"pair_{pid:02d}_{p['source_skel']}_to_{p['target_skel']}.npy"
        if not method_path.exists():
            per_pair.append({'pair_id': pid, 'error': 'missing'})
            continue
        motion = np.load(method_path)  # [T, J, 13]
        # Grouped contact schedule on TARGET skel → sum across C groups
        g_sched = grouped_contact_schedule(motion, p['target_skel'], contact_groups)
        pred_contact = g_sched.sum(axis=1)  # [T]

        # Source: extract_quotient on source motion with src contact_groups
        if p['source_fname'] not in src_sched_cache:
            try:
                q = extract_quotient(p['source_fname'], cond[p['source_skel']],
                                     contact_groups=contact_groups,
                                     motion_dir=str(motion_dir))
                src_sched_cache[p['source_fname']] = np.asarray(q['contact_sched']).sum(axis=1)
            except Exception as e:
                per_pair.append({'pair_id': pid, 'error': f'src_qx:{e}'})
                continue
        src_contact = src_sched_cache[p['source_fname']]

        # Time-align
        T_min = min(pred_contact.shape[0], src_contact.shape[0])
        if T_min < 4:
            per_pair.append({'pair_id': pid, 'error': 'too_short'})
            continue
        Ts = src_contact.shape[0]
        Tp = pred_contact.shape[0]
        idx = np.clip(np.linspace(0, Ts - 1, Tp).astype(int), 0, Ts - 1)
        src_resampled = src_contact[idx]

        c_f1 = contact_f1(pred_contact, src_resampled)

        # Stratum
        family = p['family_gap']
        strat = 'near_present' if family == 'near' else family
        if p['support_same_label'] == 0:
            strat2 = 'absent'
        else:
            strat2 = strat

        per_pair.append({
            'pair_id': pid,
            'src_skel': p['source_skel'],
            'tgt_skel': p['target_skel'],
            'src_label': p['source_label'],
            'family_gap': family,
            'support_same_label': p['support_same_label'],
            'stratum': strat2,
            'contact_f1_vs_source': c_f1,
        })

    # Stratify
    buckets = defaultdict(list)
    for e in per_pair:
        if 'error' in e:
            continue
        buckets[e['stratum']].append(e)
    buckets['all'] = [e for e in per_pair if 'error' not in e]

    summary = {}
    for name, entries in buckets.items():
        vals = [e['contact_f1_vs_source'] for e in entries]
        summary[name] = {
            'n': len(entries),
            'contact_f1_vs_source_mean': float(np.mean(vals)) if vals else None,
            'contact_f1_vs_source_std': float(np.std(vals)) if vals else None,
        }

    # Pull v2 classifier label_match/behavior_preserved from unified json, stratified
    if UNIFIED.exists():
        with open(UNIFIED) as f:
            uni = json.load(f)
        icr_block = uni['methods'].get('in_context_retarget', {})
        v2_by_stratum = icr_block.get('by_stratum', {})
    else:
        v2_by_stratum = {}

    # Write metrics alongside the npy files
    out = {
        'method': 'in_context_retarget',
        'variant': 'concat_behavior_tokens_B1_inference_only',
        'n_pairs': len(pairs),
        'n_valid': len(buckets['all']),
        'contact_f1_vs_source_stratified': summary,
        'v2_classifier_stratified': v2_by_stratum,
        'per_pair': per_pair,
    }
    out_path = METHOD_DIR / 'metrics_extended.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")

    # Pretty print
    print('\n=== CONTACT_F1_VS_SOURCE — in_context_retarget ===')
    for s in ['near_present', 'absent', 'moderate', 'extreme', 'all']:
        if s in summary:
            print(f"  {s:14s} n={summary[s]['n']:>2}  "
                  f"c_f1_vs_src={summary[s]['contact_f1_vs_source_mean']:.3f}")
    print('\n=== V2 CLASSIFIER RATES — in_context_retarget (from unified) ===')
    for s in ['near', 'absent', 'moderate', 'extreme', 'all']:
        b = v2_by_stratum.get(s, {})
        if b:
            print(f"  {s:14s} n={b.get('n', 0):>2}  "
                  f"lbl_match={b.get('label_match_rate', 0):.3f}  "
                  f"beh_preserved={b.get('behavior_preserved_rate', 0):.3f}")


if __name__ == '__main__':
    main()
