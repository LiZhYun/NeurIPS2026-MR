"""Multi-anchor ensemble for minimal-correspondence escape on 12 absent pairs.

For each absent pair, run Variant-A minimal-corr retargeting against EVERY
candidate anchor clip on the target skeleton. Classify each output via V2
classifier. Report decision rules:

 - oracle_upper_bound: at least one anchor produced the correct source label.
 - keyword_heuristic: the current minimal_corr_full choice (baseline).
 - majority_vote: modal prediction across anchors = source label.
 - non_idle_first: first non-Idle/non-Die anchor that produces source label.
 - qdist_closest: anchor whose output Q is closest to source Q.

Goal: identify whether a classifier-free decision rule can push 3/12 → 6+/12
on the absent stratum, which would push the router overall past McNemar bar.
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse components from the single-anchor runner.
from eval.run_minimal_corr_full import (  # type: ignore
    MOTION_DIR, COND_PATH, CONTACT_GROUPS_PATH, EVAL_PAIRS_PATH,
    AUTHORING_JSON, CLF_V2,
    ACTION_FALLBACK_KEYWORDS,
    _motion_id, _motion_frames,
    load_motion, denormalize_motion, body_scale,
    retarget_dtw_project, classify_motion, ma_smooth_positions,
)

OUT_DIR = ROOT / 'eval/results/k_compare/minimal_corr_multi_anchor'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = ROOT / 'idea-stage/minimal_corr_multi_anchor_results.json'


def enumerate_candidate_anchors(target_skel: str):
    """All candidate anchor clips for target_skel, sorted by motion id."""
    all_files = sorted(os.listdir(MOTION_DIR))
    p_tri = f'{target_skel}___'
    p_dbl = f'{target_skel}_{target_skel}_'
    cands = [f for f in all_files
             if (f.startswith(p_tri) or f.startswith(p_dbl)) and f.endswith('.npy')]
    if target_skel == 'PolarBear':
        cands = [f for f in cands if not f.startswith('PolarBearB___')
                 and not f.startswith('PolarBearB_PolarBearB_')]
    return sorted(cands, key=lambda f: _motion_id(f))


def keyword_heuristic_pick(cands, preferred_keywords):
    def keyword_rank(fname: str):
        low = fname.lower()
        for i, kw in enumerate(preferred_keywords):
            if kw.lower() in low:
                return i
        return len(preferred_keywords)
    return sorted(cands, key=lambda f: (keyword_rank(f), _motion_id(f), -_motion_frames(f)))[0]


def action_from_fname(fname: str) -> str:
    # Strip skel prefix and trailing _id.npy
    m = re.match(r'([A-Za-z]+)___([^_]+(?:_[^_]+)*)_\d+\.npy$', fname)
    if m:
        return m.group(2)
    m = re.match(r'([A-Za-z]+)_[A-Za-z]+_([^_]+(?:_[^_]+)*)_\d+\.npy$', fname)
    if m:
        return m.group(2)
    return fname


def main():
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)
    with open(EVAL_PAIRS_PATH) as f:
        eval_pairs = json.load(f)['pairs']
    pair_by_id = {p['pair_id']: p for p in eval_pairs}
    with open(AUTHORING_JSON) as f:
        authoring = json.load(f)

    from eval.external_classifier import extract_classifier_features, ACTION_CLASSES  # noqa
    from eval.train_external_classifier_v2 import V2Classifier
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.quotient_extractor import extract_quotient

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = V2Classifier(str(CLF_V2), device=device)
    print(f'Loaded V2 classifier ({clf.arch}) on {device}')

    ordered_keys = [k for k in authoring.keys() if k.startswith('pair_')]
    results = {
        'config': {'method': 'minimal_corr_multi_anchor_variantA_only',
                   'classifier_ckpt': str(CLF_V2)},
        'pairs': [],
    }

    t_all = time.time()
    for key in ordered_keys:
        entry = authoring[key]
        m = re.match(r'pair_(\d+)_', key)
        pid = int(m.group(1)) if m else -1
        src_skel = entry['source_skel']; tgt_skel = entry['target_skel']
        src_action = entry['source_action']
        pair_meta = pair_by_id.get(pid, {})
        src_fname = pair_meta.get('source_fname', '')
        family_gap = pair_meta.get('family_gap', 'unknown')

        pairs_list = entry['pairs']
        src_idxs = [p['src'] for p in pairs_list]
        tgt_idxs = [p['tgt'] for p in pairs_list]

        print(f'\n=== {key}: {src_skel}({src_action}) -> {tgt_skel} [{family_gap}] ===')

        try:
            anchors = enumerate_candidate_anchors(tgt_skel)
        except Exception as e:
            print(f'  !! enumeration failed: {e}')
            continue
        print(f'  {len(anchors)} candidate anchors')

        # Load source motion once.
        try:
            src_norm = load_motion(src_fname, cond[src_skel])
            src_phys = denormalize_motion(src_norm, cond[src_skel])
            src_q = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTION_DIR))
            src_pred = classify_motion(clf, src_phys, cond[src_skel],
                                       extract_classifier_features, recover_from_bvh_ric_np)
        except Exception as e:
            print(f'  !! source motion load/classify failed: {e}')
            continue

        src_bs = body_scale(cond[src_skel]); tgt_bs = body_scale(cond[tgt_skel])

        # Keyword-heuristic baseline pick (current method).
        pref_kws = ACTION_FALLBACK_KEYWORDS.get(key, [src_action])
        heuristic_pick = keyword_heuristic_pick(anchors, pref_kws)

        anchor_results = []
        for anchor_fname in anchors:
            t0 = time.time()
            try:
                anc_norm = load_motion(anchor_fname, cond[tgt_skel])
                anc_phys = denormalize_motion(anc_norm, cond[tgt_skel])
                retarget, mapping, stretch, plen = retarget_dtw_project(
                    src_phys, anc_phys, src_idxs, tgt_idxs, src_bs, tgt_bs,
                )
                tgt_pred = classify_motion(clf, retarget, cond[tgt_skel],
                                           extract_classifier_features, recover_from_bvh_ric_np)
                # Q features of retarget
                tmp_fname = f'__mc_mult_{tgt_skel}_{anchor_fname[:-4]}.npy'
                tmp_path = MOTION_DIR / tmp_fname
                q_dist = None
                try:
                    np.save(tmp_path, retarget.astype(np.float32))
                    q_r = extract_quotient(tmp_fname, cond[tgt_skel],
                                           contact_groups=contact_groups,
                                           motion_dir=str(MOTION_DIR))
                    # Simple Q-distance = L2 on com_path + contact_sched
                    sc = np.asarray(src_q['com_path']).astype(np.float32)
                    rc = np.asarray(q_r['com_path']).astype(np.float32)
                    T_ = min(sc.shape[0], rc.shape[0])
                    # body-scale-normalize
                    com_l2 = float(np.linalg.norm(
                        sc[:T_] / (body_scale(cond[src_skel]) + 1e-6) -
                        rc[:T_] / (tgt_bs + 1e-6)))
                    ss = np.asarray(src_q['contact_sched']).astype(np.float32)
                    rs = np.asarray(q_r['contact_sched']).astype(np.float32)
                    T2 = min(ss.shape[0], rs.shape[0])
                    # Aggregate each side across joints (mean over dim-1)
                    ss_m = ss[:T2].mean(-1) if ss.ndim > 1 else ss[:T2]
                    rs_m = rs[:T2].mean(-1) if rs.ndim > 1 else rs[:T2]
                    cs_l2 = float(np.linalg.norm(ss_m - rs_m))
                    q_dist = com_l2 + cs_l2
                finally:
                    if tmp_path.exists():
                        try: tmp_path.unlink()
                        except Exception: pass

                label_match = bool(tgt_pred == src_action)
                anchor_results.append({
                    'anchor': anchor_fname,
                    'anchor_action': action_from_fname(anchor_fname),
                    'anchor_frames': int(anc_phys.shape[0]),
                    'tgt_pred': tgt_pred,
                    'label_match': label_match,
                    'q_dist': q_dist,
                    'stretch_factor': stretch,
                    'wall_time_s': time.time() - t0,
                })
            except Exception as e:
                anchor_results.append({
                    'anchor': anchor_fname,
                    'error': str(e),
                    'wall_time_s': time.time() - t0,
                })

        # Decision rules
        oks = [r for r in anchor_results if 'error' not in r]
        matches = [r for r in oks if r['label_match']]
        counts = Counter(r['tgt_pred'] for r in oks)
        majority_pred = counts.most_common(1)[0][0] if counts else None

        # keyword-heuristic baseline
        kw_row = next((r for r in oks if r['anchor'] == heuristic_pick), None)

        # non-idle-first: first anchor whose name doesn't contain Idle/Die that
        # produces correct label
        non_idle_match = next(
            (r for r in oks
             if 'idle' not in r['anchor'].lower()
             and 'die' not in r['anchor'].lower()
             and r['label_match']),
            None,
        )

        # qdist-closest: anchor whose Q is closest to source Q
        qd_valid = [r for r in oks if r.get('q_dist') is not None]
        qd_pick = min(qd_valid, key=lambda r: r['q_dist']) if qd_valid else None

        pair_rec = {
            'pair_id': pid, 'pair_key': key,
            'src_skel': src_skel, 'src_action': src_action, 'tgt_skel': tgt_skel,
            'family_gap': family_gap,
            'src_classifier_pred': src_pred,
            'n_anchors': len(anchors),
            'n_evaluated': len(oks),
            'anchor_pred_counts': dict(counts),
            'majority_pred': majority_pred,
            'majority_correct': bool(majority_pred == src_action),
            'oracle_upper_bound_correct': bool(matches),  # any anchor produces src_label
            'oracle_matching_anchors': [r['anchor'] for r in matches],
            'keyword_heuristic_anchor': heuristic_pick,
            'keyword_heuristic_correct': bool(kw_row and kw_row['label_match']),
            'non_idle_first_correct': bool(non_idle_match is not None),
            'qdist_pick_anchor': qd_pick['anchor'] if qd_pick else None,
            'qdist_pick_correct': bool(qd_pick and qd_pick['label_match']),
            'anchor_results': anchor_results,
        }
        results['pairs'].append(pair_rec)

        # Print concise
        print(f'  counts: {dict(counts)}')
        print(f'  oracle: {bool(matches)} ({len(matches)}/{len(oks)} anchors correct)')
        print(f'  keyword: {bool(kw_row and kw_row["label_match"])}')
        print(f'  majority: {bool(majority_pred == src_action)} (={majority_pred})')
        print(f'  non_idle_first: {non_idle_match is not None}')
        print(f'  qdist_pick: {qd_pick["anchor"] if qd_pick else None} -> {qd_pick["tgt_pred"] if qd_pick else None} correct={bool(qd_pick and qd_pick["label_match"])}')

    # Aggregate
    n = len(results['pairs'])
    agg = {
        'oracle_upper_bound': sum(1 for p in results['pairs'] if p['oracle_upper_bound_correct']) / max(n, 1),
        'keyword_heuristic': sum(1 for p in results['pairs'] if p['keyword_heuristic_correct']) / max(n, 1),
        'majority_vote': sum(1 for p in results['pairs'] if p['majority_correct']) / max(n, 1),
        'non_idle_first': sum(1 for p in results['pairs'] if p['non_idle_first_correct']) / max(n, 1),
        'qdist_pick': sum(1 for p in results['pairs'] if p['qdist_pick_correct']) / max(n, 1),
    }
    results['summary'] = {
        'n_absent_pairs': n,
        'wall_time_total_s': time.time() - t_all,
        'decision_rule_accuracy': agg,
    }
    print('\n=== SUMMARY ===')
    for k, v in agg.items():
        print(f'  {k}: {v:.3f} ({int(v*n)}/{n})')

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
