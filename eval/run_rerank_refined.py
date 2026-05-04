"""Idea S_v4 — classifier_rerank + source-anchored AnyTop projection.

Pipeline per pair:
  1. Use the already-picked top-1 from classifier_rerank (metrics.json) as
     x_init. (Skip re-running rerank; motions are saved verbatim in
     eval/results/k_compare/classifier_rerank/.)
  2. Extract source Q.
  3. Build source-anchored hard constraints via H_v2's
     build_source_anchored_contact_mask + source COM path rescaled by
     target_body_scale / source_body_scale.
  4. anytop_project(x_init=rerank_candidate, target_skel=tgt, hard_constraints,
                     t_init=0.3, n_steps=20, lambda_com=1.0).
  5. Save refined motion + metrics.json in
     eval/results/k_compare/rerank_refined/.

Hypothesis: refining on the BEST retrieval candidate (classifier_rerank's
0.533 label-match) pulls post-refinement closer to source Q while keeping
semantic match higher than pure-Q-retrieval-refinement (H_v4).

Usage:  conda run -n anytop python -m eval.run_rerank_refined
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse helpers from H_v2 and classifier_rerank WITHOUT modifying them.
from eval.run_retrieve_refine_v2 import (  # noqa: E402
    build_source_anchored_contact_mask,
    q_component_l2,
    contact_f1 as _contact_f1_binary,
)
from eval.run_classifier_rerank import (  # noqa: E402
    q_components_and_contact_f1,
    classify_motion_probs,
    stratum_of,
)
from eval.anytop_projection import anytop_project  # noqa: E402
from eval.quotient_extractor import extract_quotient  # noqa: E402
from eval.external_classifier import ACTION_CLASSES  # noqa: E402
from eval.train_external_classifier_v2 import V2Classifier  # noqa: E402

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
CLF_CKPT = ROOT / 'save/external_classifier_v2.pt'
RERANK_DIR = ROOT / 'eval/results/k_compare/classifier_rerank'
RERANK_METRICS = RERANK_DIR / 'metrics.json'

OUT_DIR = ROOT / 'eval/results/k_compare/rerank_refined'
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD = 'rerank_refined'
T_INIT = 0.3
N_STEPS = 20
LAMBDA_COM = 1.0


NUMERIC_KEYS = [
    'label_match', 'behavior_preserved',
    'q_com_path_l2', 'q_heading_vel_l2', 'q_cadence_abs_diff',
    'q_contact_sched_aggregate_l2', 'q_limb_usage_top5_l2',
    'contact_f1_vs_source',
    'q_com_path_l2_pre', 'q_heading_vel_l2_pre', 'q_cadence_abs_diff_pre',
    'q_contact_sched_aggregate_l2_pre', 'q_limb_usage_top5_l2_pre',
    'contact_f1_vs_source_pre',
    'q_com_path_delta', 'q_heading_vel_delta', 'q_cadence_delta',
    'q_contact_sched_aggregate_delta', 'q_limb_usage_top5_delta',
    'contact_f1_delta',
    'refine_runtime_s', 'wall_time_s',
]


def stratified_means(per_pair_entries):
    buckets = defaultdict(list)
    for e in per_pair_entries:
        for s in stratum_of(e['family_gap'], e['support_same_label']):
            buckets[s].append(e)
    out = {}
    for s in ['all', 'near_present', 'absent', 'moderate', 'extreme']:
        es = buckets.get(s, [])
        stats = {'n': len(es)}
        for k in NUMERIC_KEYS:
            vals = [e[k] for e in es if e.get(k) is not None
                    and not (isinstance(e[k], float) and np.isnan(e[k]))]
            stats[k] = float(np.mean(vals)) if vals else None
        # Class prediction distribution.
        preds = [e.get('action_classifier_pred') for e in es
                 if e.get('action_classifier_pred')]
        cnt = {}
        for p in preds:
            cnt[p] = cnt.get(p, 0) + 1
        stats['per_class_pred'] = cnt
        out[s] = stats
    return out


def main():
    t_start_all = time.time()

    print('[rerank_refined] loading caches...')
    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(RERANK_METRICS) as f:
        rerank_metrics = json.load(f)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    # map pair_id -> rerank entry (retrieved_fname, output_file)
    rerank_by_pair = {e['pair_id']: e for e in rerank_metrics['per_pair']}
    print(f"  loaded {len(rerank_by_pair)} classifier_rerank entries")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[rerank_refined] loading v2 classifier on {device}...')
    clf = V2Classifier(str(CLF_CKPT), device=device)

    per_pair = []
    for p in pairs:
        pid = int(p['pair_id'])
        src_fname = p['source_fname']
        src_skel = p['source_skel']
        src_label = p['source_label']
        tgt_skel = p['target_skel']
        family_gap = p['family_gap']
        support = int(p['support_same_label'])
        strat = {'near': 'near_present'}.get(family_gap, family_gap)

        rec = {
            'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
            'src_label': src_label, 'tgt_skel': tgt_skel,
            'family_gap': strat, 'support_same_label': support,
            'status': 'pending', 'error': None,
        }
        t_pair0 = time.time()
        print(f"\n=== pair {pid:02d} {src_skel}->{tgt_skel}  gap={strat}  supp={support} ===")

        try:
            if pid not in rerank_by_pair:
                raise RuntimeError(f'no classifier_rerank entry for pair {pid}')
            rr = rerank_by_pair[pid]
            retr_fname = rr['retrieved_fname']
            rec['retrieved_fname'] = retr_fname
            rec['retrieved_coarse_label'] = rr.get('retrieved_coarse_label')
            rec['rerank_p_source_action'] = rr.get('p_source_action')

            if tgt_skel not in cond_dict or src_skel not in cond_dict:
                raise RuntimeError('missing cond for src/tgt')

            # x_init = verbatim rerank-chosen candidate
            rerank_motion_path = RERANK_DIR / rr['output_file']
            if not rerank_motion_path.exists():
                rerank_motion_path = MOTIONS_DIR / retr_fname
            x_init = np.load(rerank_motion_path).astype(np.float32)
            T_retr, J_retr, F_retr = x_init.shape

            # --- Source Q + pre-refinement Q distances ---
            src_q = extract_quotient(src_fname, cond_dict[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTIONS_DIR))
            pre_q = extract_quotient(retr_fname, cond_dict[tgt_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTIONS_DIR))
            pre_stats = q_components_and_contact_f1(src_q, pre_q)
            rec['q_com_path_l2_pre']               = pre_stats['q_com_path_l2']
            rec['q_heading_vel_l2_pre']            = pre_stats['q_heading_vel_l2']
            rec['q_cadence_abs_diff_pre']          = pre_stats['q_cadence_abs_diff']
            rec['q_contact_sched_aggregate_l2_pre']= pre_stats['q_contact_sched_aggregate_l2']
            rec['q_limb_usage_top5_l2_pre']        = pre_stats['q_limb_usage_top5_l2']
            rec['contact_f1_vs_source_pre']        = pre_stats['contact_f1_vs_source']

            # --- Source-anchored hard constraints (H_v2 style) ---
            tgt_groups = contact_groups.get(tgt_skel, {})
            tgt_groups_clean = {k: v for k, v in tgt_groups.items()
                                if not str(k).startswith('_')}
            source_contact_mask = build_source_anchored_contact_mask(
                src_q, tgt_skel, tgt_groups_clean,
                contact_groups.get(src_skel, {}),
                n_frames_target=T_retr, n_joints_target=J_retr,
            )
            src_com = np.asarray(src_q['com_path']).astype(np.float32)
            src_bs = float(src_q['body_scale'])
            tgt_bs = float(np.linalg.norm(cond_dict[tgt_skel]['offsets'], axis=1).sum() + 1e-6)
            scale_ratio = tgt_bs / max(src_bs, 1e-6)
            idx = np.clip(np.linspace(0, src_com.shape[0] - 1, T_retr).astype(int),
                          0, src_com.shape[0] - 1)
            src_com_resampled = src_com[idx] * scale_ratio

            hard_con = {'contact_positions': source_contact_mask,
                        'com_path': src_com_resampled}

            # --- Project ---
            t_ref0 = time.time()
            proj = anytop_project(x_init, tgt_skel,
                                  hard_constraints=hard_con,
                                  t_init=T_INIT, n_steps=N_STEPS,
                                  lambda_com=LAMBDA_COM,
                                  device=str(device))
            x_refined = proj['x_refined']
            rec['refine_runtime_s'] = float(proj['runtime_seconds'])

            # --- Save ---
            out_fname = f"pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy"
            np.save(OUT_DIR / out_fname, x_refined.astype(np.float32))
            rec['output_file'] = out_fname

            # --- Post-refinement Q (via temporary file so extract_quotient
            #     can run its usual pipeline). ---
            tmp_name = f'__rerank_refined_tmp_pair_{pid:02d}.npy'
            tmp_path = MOTIONS_DIR / tmp_name
            try:
                np.save(tmp_path, x_refined.astype(np.float32))
                post_q = extract_quotient(tmp_name, cond_dict[tgt_skel],
                                          contact_groups=contact_groups,
                                          motion_dir=str(MOTIONS_DIR))
            finally:
                if tmp_path.exists():
                    try: tmp_path.unlink()
                    except Exception: pass
            post_stats = q_components_and_contact_f1(src_q, post_q)
            rec['q_com_path_l2']               = post_stats['q_com_path_l2']
            rec['q_heading_vel_l2']            = post_stats['q_heading_vel_l2']
            rec['q_cadence_abs_diff']          = post_stats['q_cadence_abs_diff']
            rec['q_contact_sched_aggregate_l2']= post_stats['q_contact_sched_aggregate_l2']
            rec['q_limb_usage_top5_l2']        = post_stats['q_limb_usage_top5_l2']
            rec['contact_f1_vs_source']        = post_stats['contact_f1_vs_source']

            def _d(a, b):
                if a is None or b is None: return None
                return float(a - b)
            rec['q_com_path_delta']    = _d(rec['q_com_path_l2'], rec['q_com_path_l2_pre'])
            rec['q_heading_vel_delta'] = _d(rec['q_heading_vel_l2'], rec['q_heading_vel_l2_pre'])
            rec['q_cadence_delta']     = _d(rec['q_cadence_abs_diff'], rec['q_cadence_abs_diff_pre'])
            rec['q_contact_sched_aggregate_delta'] = _d(
                rec['q_contact_sched_aggregate_l2'], rec['q_contact_sched_aggregate_l2_pre'])
            rec['q_limb_usage_top5_delta'] = _d(
                rec['q_limb_usage_top5_l2'], rec['q_limb_usage_top5_l2_pre'])
            rec['contact_f1_delta'] = _d(rec['contact_f1_vs_source'],
                                         rec['contact_f1_vs_source_pre'])

            # --- Classifier label_match + behavior_preserved ---
            probs_chosen, pred_chosen, _ = classify_motion_probs(
                x_refined, cond_dict[tgt_skel], clf)
            src_motion = np.load(MOTIONS_DIR / src_fname)
            probs_src, pred_src, _ = classify_motion_probs(
                src_motion, cond_dict[src_skel], clf)
            rec['action_classifier_pred']  = pred_chosen
            rec['source_classifier_pred']  = pred_src
            rec['label_match'] = (
                int(pred_chosen == src_label) if pred_chosen is not None else None)
            rec['behavior_preserved'] = (
                int(pred_chosen == pred_src)
                if (pred_chosen is not None and pred_src is not None) else None)
            if probs_chosen is not None:
                from eval.external_classifier import ACTION_TO_IDX
                if src_label in ACTION_TO_IDX:
                    rec['p_source_action'] = float(probs_chosen[ACTION_TO_IDX[src_label]])

            rec['wall_time_s'] = float(time.time() - t_pair0)
            rec['status'] = 'ok'

            cf_pre = rec['contact_f1_vs_source_pre']
            cf_post = rec['contact_f1_vs_source']
            print(
                f"  ok  retr={retr_fname}  "
                f"lm={rec['label_match']}  bp={rec['behavior_preserved']}  "
                f"pred={pred_chosen or '--':6s}  "
                f"cf1 {cf_pre:.2f}->{cf_post:.2f}  "
                f"q_com {rec['q_com_path_l2_pre']:.2f}->{rec['q_com_path_l2']:.2f}  "
                f"refine={rec['refine_runtime_s']:.2f}s  "
                f"wall={rec['wall_time_s']:.2f}s"
            )

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['wall_time_s'] = float(time.time() - t_pair0)
            print(f"  FAILED: {e}")

        per_pair.append(rec)

    ok = [r for r in per_pair if r['status'] == 'ok']
    stratified = stratified_means(ok)

    out = {
        'method': METHOD,
        'hparams': {'t_init': T_INIT, 'n_steps': N_STEPS,
                    'lambda_com': LAMBDA_COM,
                    'x_init': 'classifier_rerank_top1'},
        'classifier_ckpt': str(CLF_CKPT),
        'classifier_arch': clf.arch,
        'n_pairs': len(pairs),
        'n_ok': len(ok),
        'n_failed': len(per_pair) - len(ok),
        'total_wall_time_s': time.time() - t_start_all,
        'per_pair': per_pair,
        'stratified': stratified,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print('\n' + '=' * 78)
    print(f'RERANK_REFINED SUMMARY  (t_init={T_INIT}, n_steps={N_STEPS}, lambda_com={LAMBDA_COM})')
    print('=' * 78)
    hdr = (f"{'stratum':<14} {'n':>3} {'lbl':>5} {'beh':>5} "
           f"{'cf1(pre)':>9} {'cf1(pst)':>9} {'Δcf1':>7} {'Δcom':>7} {'Δhv':>7}")
    print(hdr)
    for s in ['all', 'near_present', 'absent', 'moderate', 'extreme']:
        v = stratified.get(s, {})
        n = v.get('n', 0)
        def fmt(x, w, sign=False):
            if x is None: return f"{'--':>{w}}"
            if sign:      return f"{x:>{w}+.3f}"
            return f"{x:>{w}.3f}"
        print(f"{s:<14} {n:>3} "
              f"{fmt(v.get('label_match'), 5)} "
              f"{fmt(v.get('behavior_preserved'), 5)} "
              f"{fmt(v.get('contact_f1_vs_source_pre'), 9)} "
              f"{fmt(v.get('contact_f1_vs_source'), 9)} "
              f"{fmt(v.get('contact_f1_delta'), 7, sign=True)} "
              f"{fmt(v.get('q_com_path_delta'), 7, sign=True)} "
              f"{fmt(v.get('q_heading_vel_delta'), 7, sign=True)}")
    print(f"\nTotal wall time: {time.time() - t_start_all:.1f}s")
    print(f"metrics: {OUT_DIR/'metrics.json'}")
    print(f"outputs: {OUT_DIR} ({len(ok)} .npy files)")


if __name__ == '__main__':
    main()
