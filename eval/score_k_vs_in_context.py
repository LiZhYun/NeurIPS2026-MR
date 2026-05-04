"""Compare contact_f1_vs_source for K baseline and in_context_retarget using the
SAME grouped-contact formulation (apples-to-apples).
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

METHODS = ['K', 'in_context_retarget', 'retrieve_refine_v4', 'classifier_rerank',
           'motion2motion', 'label_random']
COMPARE_DIR = ROOT / 'eval/results/k_compare'
EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTION_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'


def contact_f1(pred, gt, th=0.5):
    p = (np.asarray(pred) >= th).astype(np.int8).ravel()
    g = (np.asarray(gt) >= th).astype(np.int8).ravel()
    n = min(p.size, g.size)
    p, g = p[:n], g[:n]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
    return float(2 * pr * rc / (pr + rc + 1e-8))


def grouped_contact(motion, skel, contact_groups):
    ch12 = motion[..., 12]
    contacts = (ch12 > 0.5).astype(np.float32)
    if skel not in contact_groups:
        return contacts.sum(axis=1, keepdims=True)
    groups = contact_groups[skel]
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
        cg = json.load(f)

    src_sched_cache = {}
    def src_contact(p):
        if p['source_fname'] not in src_sched_cache:
            q = extract_quotient(p['source_fname'], cond[p['source_skel']],
                                 contact_groups=cg, motion_dir=str(MOTION_DIR))
            src_sched_cache[p['source_fname']] = np.asarray(q['contact_sched']).sum(axis=1)
        return src_sched_cache[p['source_fname']]

    all_results = {}
    for method in METHODS:
        mdir = COMPARE_DIR / method
        if not mdir.exists():
            continue
        per = []
        for p in pairs:
            pid = p['pair_id']
            matches = list(mdir.glob(f'pair_{pid:02d}_*.npy'))
            if not matches:
                continue
            motion = np.load(matches[0])
            g = grouped_contact(motion, p['target_skel'], cg)
            pred = g.sum(axis=1)
            try:
                sc = src_contact(p)
            except Exception as e:
                print(f"src fail pair {pid}: {e}")
                continue
            idx = np.clip(np.linspace(0, sc.shape[0]-1, pred.shape[0]).astype(int), 0, sc.shape[0]-1)
            sc_al = sc[idx]
            c = contact_f1(pred, sc_al)
            strat = 'absent' if p['support_same_label'] == 0 else (
                'near_present' if p['family_gap'] == 'near' else p['family_gap'])
            per.append({'pair_id': pid, 'c_f1': c, 'stratum': strat})

        strat_map = defaultdict(list)
        for r in per:
            strat_map[r['stratum']].append(r['c_f1'])
        strat_map['all'] = [r['c_f1'] for r in per]
        summary = {}
        for name, vals in strat_map.items():
            summary[name] = {
                'n': len(vals),
                'c_f1_mean': float(np.mean(vals)) if vals else None,
            }
        all_results[method] = summary

    # Print table
    strata_order = ['near_present', 'absent', 'moderate', 'extreme', 'all']
    header = f"{'Method':<24} " + ' '.join(f"{s[:10]:>12}" for s in strata_order)
    print(header)
    print('-' * len(header))
    for method, summ in all_results.items():
        row = f"{method:<24} "
        for s in strata_order:
            b = summ.get(s)
            if b and b['c_f1_mean'] is not None:
                row += f"{b['c_f1_mean']:>12.3f}"
            else:
                row += f"{'—':>12}"
        print(row)

    out_path = COMPARE_DIR / 'contact_f1_grouped_comparison.json'
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
