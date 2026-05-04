"""Idea S — classifier-reranked retrieval.

For each of the 30 canonical eval pairs:
  1. Retrieve the top-K (default 20) target-skeleton clips by Q-signature cosine
     similarity to the source Q signature.
  2. For each candidate, load the motion, extract topology-normalised features
     via ``eval.external_classifier.extract_classifier_features`` and run the
     v2 external classifier to obtain a 12-class probability distribution.
  3. Pick the candidate that maximises ``P(source_action_class)``; break ties
     by Q-cosine similarity.
  4. Save the candidate motion VERBATIM (no generation, no refinement) as the
     retargeted output under
     ``eval/results/k_compare/classifier_rerank/pair_<id>_<src>_to_<tgt>.npy``
     and record a metrics.json with Q-component distances, contact_f1, wall time.

Why this matters: v1 external classifier was too weak (val-acc 0.35) for prior
methods to trust it as a re-ranker. v2 reports val-acc 0.52 / macro-F1 0.51,
so using it as a re-ranker over the top-K Q-retrieval shortlist is a simple
drop-in upgrade over plain Q retrieval.

Usage:
    conda run -n anytop python -m eval.run_classifier_rerank
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.external_classifier import (  # noqa: E402
    ACTION_CLASSES,
    ACTION_TO_IDX,
    extract_classifier_features,
)
from eval.pilot_Q_experiments import q_signature  # noqa: E402
from eval.quotient_extractor import extract_quotient  # noqa: E402
from eval.train_external_classifier_v2 import (  # noqa: E402
    V2Classifier,
    resample_along_time,
)

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
CLF_CKPT = ROOT / 'save/external_classifier_v2.pt'

OUT_DIR = ROOT / 'eval/results/k_compare/classifier_rerank'
METHOD = 'classifier_rerank'

TOP_K = 20
CONTACT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Small helpers (local copies — do NOT modify upstream modules)
# ---------------------------------------------------------------------------

def l2_norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def cosine_sim(a, b):
    return l2_norm(a) @ l2_norm(b).T


def build_q_sig_array(qc):
    """Build [N, D] Q-signature matrix in qc['meta'] order."""
    N = len(qc['meta'])
    sigs = []
    for i in range(N):
        q = {
            'com_path': qc['com_path'][i],
            'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        sigs.append(q_signature(q))
    return np.stack(sigs)


@torch.no_grad()
def classifier_probabilities(clf: V2Classifier, features: np.ndarray) -> np.ndarray | None:
    """Return softmax probabilities over ACTION_CLASSES for a single clip.

    Mirrors V2Classifier.predict without changing the class, but averages
    softmax probabilities across ensemble members (rather than averaging logits)
    — this gives a proper distribution usable for re-ranking.
    """
    if features is None:
        return None
    feats = resample_along_time(features, clf.target_T).astype(np.float32)
    feats = (feats - clf.feat_mean) / (clf.feat_std + 1e-6)
    x = torch.from_numpy(feats).unsqueeze(0).to(clf.device)
    probs_sum = None
    for m in clf.models:
        logits = m(x)
        probs = F.softmax(logits, dim=-1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    probs_sum = probs_sum / max(len(clf.models), 1)
    return probs_sum.squeeze(0).cpu().numpy().astype(np.float32)


def classify_motion_probs(motion_13dim: np.ndarray, cond_skel: dict,
                          clf: V2Classifier):
    """Return (probs [12], pred_label_str, denorm_ok) for a [T, J, 13] motion tensor."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    J_skel = cond_skel['offsets'].shape[0]
    m = motion_13dim
    if m.shape[1] > J_skel:
        m = m[:, :J_skel]
    # Auto-detect normalised vs denormalised input: dataset motions are stored
    # normalised (per-joint z-score), so values typically lie in [-3, 3].
    if np.abs(m).max() < 5:
        mean = cond_skel['mean'][:J_skel]
        std = cond_skel['std'][:J_skel]
        m = m.astype(np.float32) * std + mean
    try:
        positions = recover_from_bvh_ric_np(m.astype(np.float32))
    except Exception:
        return None, None, False
    parents = cond_skel['parents'][:J_skel]
    feats = extract_classifier_features(positions, parents)
    if feats is None or feats.shape[0] < 4:
        return None, None, True
    probs = classifier_probabilities(clf, feats)
    if probs is None:
        return None, None, True
    idx = int(probs.argmax())
    return probs, ACTION_CLASSES[idx], True


# ---------------------------------------------------------------------------
# Q-component distances + contact F1 (identical recipe to score_unified_q)
# ---------------------------------------------------------------------------

def _resample_time(a: np.ndarray, T: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 0 or a.shape[0] == T:
        return a
    T_a = a.shape[0]
    idx = np.clip(np.round(np.linspace(0, T_a - 1, T)).astype(int), 0, T_a - 1)
    return a[idx]


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) + np.linalg.norm(b) + 1e-6
    return float(np.linalg.norm(a - b) / denom)


def q_components_and_contact_f1(q_src: dict, q_tgt: dict) -> dict:
    # com_path
    T = min(q_src['com_path'].shape[0], q_tgt['com_path'].shape[0])
    out_com = _rel_l2(_resample_time(q_src['com_path'], T),
                      _resample_time(q_tgt['com_path'], T))

    # heading_vel
    T = min(q_src['heading_vel'].shape[0], q_tgt['heading_vel'].shape[0])
    out_hv = _rel_l2(_resample_time(q_src['heading_vel'], T),
                     _resample_time(q_tgt['heading_vel'], T))

    out_cad = float(abs(float(q_src['cadence']) - float(q_tgt['cadence'])))

    cs_src = np.asarray(q_src['contact_sched'])
    cs_tgt = np.asarray(q_tgt['contact_sched'])
    agg_src = cs_src.sum(axis=1) if cs_src.ndim == 2 else cs_src
    agg_tgt = cs_tgt.sum(axis=1) if cs_tgt.ndim == 2 else cs_tgt
    T = min(len(agg_src), len(agg_tgt))
    agg_src_r = _resample_time(agg_src, T)
    agg_tgt_r = _resample_time(agg_tgt, T)
    out_contact = _rel_l2(agg_src_r, agg_tgt_r)

    lu_src = -np.sort(-np.asarray(q_src['limb_usage']))[:5]
    lu_tgt = -np.sort(-np.asarray(q_tgt['limb_usage']))[:5]
    K = max(len(lu_src), len(lu_tgt), 5)
    lu_src = np.pad(lu_src, (0, K - len(lu_src)))
    lu_tgt = np.pad(lu_tgt, (0, K - len(lu_tgt)))
    out_lu = _rel_l2(lu_src, lu_tgt)

    # Binarised contact F1 on summed-over-groups schedule
    bin_src = (agg_src_r >= CONTACT_THRESHOLD).astype(np.int8)
    bin_tgt = (agg_tgt_r >= CONTACT_THRESHOLD).astype(np.int8)
    tp = int(((bin_tgt == 1) & (bin_src == 1)).sum())
    fp = int(((bin_tgt == 1) & (bin_src == 0)).sum())
    fn = int(((bin_tgt == 0) & (bin_src == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    return {
        'q_com_path_l2': out_com,
        'q_heading_vel_l2': out_hv,
        'q_cadence_abs_diff': out_cad,
        'q_contact_sched_aggregate_l2': out_contact,
        'q_limb_usage_top5_l2': out_lu,
        'contact_f1_vs_source': float(f1),
    }


# ---------------------------------------------------------------------------
# Stratified summary
# ---------------------------------------------------------------------------

NUMERIC_KEYS = [
    'label_match', 'behavior_preserved', 'action_file_label_match',
    'p_source_action', 'q_cosine_top1',
    'q_com_path_l2', 'q_heading_vel_l2', 'q_cadence_abs_diff',
    'q_contact_sched_aggregate_l2', 'q_limb_usage_top5_l2',
    'contact_f1_vs_source', 'wall_time_s',
]


def stratum_of(family_gap: str, support: int) -> list:
    strata = []
    if family_gap in ('near', 'near_present'):
        strata.append('near_present')
    elif family_gap == 'moderate':
        strata.append('moderate')
    elif family_gap == 'extreme':
        strata.append('extreme')
    if support == 0:
        strata.append('absent')
    strata.append('all')
    return strata


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
        out[s] = stats
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start_all = time.time()

    print('[classifier_rerank] loading caches...')
    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    fname_to_meta_idx = {m['fname']: i for i, m in enumerate(meta)}
    q_meta = list(qc['meta'])
    fname_to_q_idx = {m['fname']: i for i, m in enumerate(q_meta)}

    print(f"  meta: {len(meta)}  q_meta: {len(q_meta)}  pairs: {len(pairs)}")

    print('[classifier_rerank] building Q signatures...')
    q_sigs = build_q_sig_array(qc)
    print(f"  Q-sig shape: {q_sigs.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[classifier_rerank] loading v2 classifier on {device}...')
    clf = V2Classifier(str(CLF_CKPT), device=device)
    print(f"  arch={clf.arch}  target_T={clf.target_T}  n_classes={len(clf.classes)}")

    skel_to_meta_idx = defaultdict(list)
    for i, m in enumerate(meta):
        skel_to_meta_idx[m['skeleton']].append(i)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_pair = []
    candidate_trace = []  # for audit

    for pair in pairs:
        t_pair_start = time.time()
        pair_id = pair['pair_id']
        src_fname = pair['source_fname']
        src_skel = pair['source_skel']
        src_label = pair['source_label']
        tgt_skel = pair['target_skel']
        family_gap = pair['family_gap']
        support = pair['support_same_label']
        strat = {'near': 'near_present'}.get(family_gap, family_gap)

        if src_fname not in fname_to_q_idx or src_fname not in fname_to_meta_idx:
            print(f"  pair {pair_id:02d}: source not in caches, skipping")
            continue
        if tgt_skel not in cond_dict:
            print(f"  pair {pair_id:02d}: target {tgt_skel} not in cond, skipping")
            continue
        if src_label not in ACTION_TO_IDX:
            # source_label must be a known action class; 'other' is in list
            print(f"  pair {pair_id:02d}: unknown src_label {src_label}, skipping")
            continue

        src_q_idx = fname_to_q_idx[src_fname]
        src_q_sig = q_sigs[src_q_idx]

        # Target pool on target skeleton (must also exist in Q cache)
        tgt_pool_meta = [i for i in skel_to_meta_idx[tgt_skel]
                        if meta[i]['fname'] != src_fname
                        and meta[i]['fname'] in fname_to_q_idx]
        if not tgt_pool_meta:
            print(f"  pair {pair_id:02d}: no target pool for {tgt_skel}, skipping")
            continue

        pool_qi = [fname_to_q_idx[meta[i]['fname']] for i in tgt_pool_meta]
        sims_q = cosine_sim(src_q_sig[None], q_sigs[pool_qi])[0]
        # top-K indices by Q-cosine
        K = min(TOP_K, len(sims_q))
        topk_order = np.argsort(-sims_q)[:K]

        # Score each candidate with classifier
        src_class_idx = ACTION_TO_IDX[src_label]
        cand_records = []
        for rank_i, pool_pos in enumerate(topk_order):
            meta_i = tgt_pool_meta[int(pool_pos)]
            cand_fname = meta[meta_i]['fname']
            cand_motion = np.load(MOTIONS_DIR / cand_fname)
            probs, pred_str, ok = classify_motion_probs(
                cand_motion, cond_dict[tgt_skel], clf)
            if not ok or probs is None:
                # feats extraction failed — assign near-zero probability but keep candidate
                p_src = -1.0
                probs_list = None
            else:
                p_src = float(probs[src_class_idx])
                probs_list = [float(x) for x in probs]
            cand_records.append({
                'rank': int(rank_i),
                'meta_idx': meta_i,
                'fname': cand_fname,
                'coarse_label': meta[meta_i]['coarse_label'],
                'q_cosine': float(sims_q[int(pool_pos)]),
                'p_source_action': p_src,
                'pred_label': pred_str,
                'probs': probs_list,
            })

        # Rerank: primary key P(source_action); tiebreak by Q-cosine (both descending).
        # Candidates with failed feature extraction (p_src==-1) are pushed to the back.
        valid = [c for c in cand_records if c['p_source_action'] >= 0]
        fallback = [c for c in cand_records if c['p_source_action'] < 0]
        valid.sort(key=lambda c: (-c['p_source_action'], -c['q_cosine']))
        fallback.sort(key=lambda c: -c['q_cosine'])
        ordered = valid + fallback
        best = ordered[0]

        # Save chosen motion VERBATIM
        out_fname = f"pair_{pair_id:02d}_{src_skel}_to_{tgt_skel}.npy"
        chosen_motion = np.load(MOTIONS_DIR / best['fname'])
        np.save(OUT_DIR / out_fname, chosen_motion.astype(np.float32))

        # Metrics
        # (1) classifier-dependent: label_match (chosen-clip classifier vs src_label)
        #     + behavior_preserved (chosen-clip classifier vs source-clip classifier)
        chosen_motion_for_clf = chosen_motion
        probs_chosen, pred_chosen, _ = classify_motion_probs(
            chosen_motion_for_clf, cond_dict[tgt_skel], clf)
        src_motion = np.load(MOTIONS_DIR / src_fname)
        probs_src, pred_src, _ = classify_motion_probs(
            src_motion, cond_dict[src_skel], clf)
        label_match = int(pred_chosen == src_label) if pred_chosen is not None else None
        behavior_preserved = (
            int(pred_chosen == pred_src)
            if (pred_chosen is not None and pred_src is not None) else None
        )

        # (2) classifier-independent: Q components + contact F1
        src_q = extract_quotient(src_fname, cond_dict[src_skel],
                                 contact_groups=contact_groups,
                                 motion_dir=str(MOTIONS_DIR))
        tgt_q = extract_quotient(best['fname'], cond_dict[tgt_skel],
                                 contact_groups=contact_groups,
                                 motion_dir=str(MOTIONS_DIR))
        q_dists = q_components_and_contact_f1(src_q, tgt_q)

        retrieved_meta = meta[best['meta_idx']]
        wall_time = time.time() - t_pair_start
        entry = {
            'pair_id': pair_id,
            'src_fname': src_fname,
            'src_skel': src_skel,
            'src_label': src_label,
            'tgt_skel': tgt_skel,
            'family_gap': strat,
            'support_same_label': support,
            'retrieved_fname': best['fname'],
            'retrieved_coarse_label': retrieved_meta['coarse_label'],
            'retrieved_rank_by_q': best['rank'],
            'output_file': out_fname,
            'p_source_action': best['p_source_action'],
            'q_cosine_top1': float(np.max(sims_q)),
            'q_cosine_of_chosen': best['q_cosine'],
            'action_classifier_pred': pred_chosen,
            'source_classifier_pred': pred_src,
            'action_file_label_match': int(retrieved_meta['coarse_label'] == src_label),
            'label_match': label_match,
            'behavior_preserved': behavior_preserved,
            **q_dists,
            'wall_time_s': float(wall_time),
        }
        per_pair.append(entry)
        candidate_trace.append({
            'pair_id': pair_id,
            'src_fname': src_fname,
            'src_skel': src_skel,
            'tgt_skel': tgt_skel,
            'src_label': src_label,
            'candidates': cand_records,
            'chosen_rank': best['rank'],
            'chosen_fname': best['fname'],
        })

        print(
            f"  pair {pair_id:02d} [{src_skel}->{tgt_skel}] src_label={src_label:6s} "
            f"-> chosen rank={best['rank']:>2}  p={best['p_source_action']:.3f}  "
            f"q_cos={best['q_cosine']:.3f}  file_lbl={retrieved_meta['coarse_label']:6s}  "
            f"clf_pred={pred_chosen or '--':6s}  "
            f"lm={label_match}  bp={behavior_preserved}  "
            f"cf1={q_dists['contact_f1_vs_source']:.2f}  "
            f"{wall_time:.2f}s"
        )

    stratified = stratified_means(per_pair)
    metrics_path = OUT_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'method': METHOD,
            'top_k': TOP_K,
            'classifier_ckpt': str(CLF_CKPT),
            'classifier_arch': clf.arch,
            'n_pairs': len(per_pair),
            'total_wall_time_s': time.time() - t_start_all,
            'per_pair': per_pair,
            'stratified': stratified,
        }, f, indent=2)
    trace_path = OUT_DIR / 'candidate_trace.json'
    with open(trace_path, 'w') as f:
        json.dump(candidate_trace, f, indent=2)

    print('\n' + '=' * 78)
    print(f'CLASSIFIER-RERANK SUMMARY (top-K={TOP_K}, v2 classifier)')
    print('=' * 78)
    hdr = f"{'stratum':<14} {'n':>3} {'lbl_match':>9} {'behav_pres':>10} {'file_lbl':>9} {'p_src':>7} {'cf1':>5}"
    print(hdr)
    for s in ['all', 'near_present', 'absent', 'moderate', 'extreme']:
        v = stratified.get(s, {})
        n = v.get('n', 0)
        lm = v.get('label_match')
        bp = v.get('behavior_preserved')
        fl = v.get('action_file_label_match')
        ps = v.get('p_source_action')
        cf = v.get('contact_f1_vs_source')

        def fmt(x, w):
            return f"{x:>{w}.3f}" if x is not None else f"{'--':>{w}}"
        print(f"{s:<14} {n:>3} {fmt(lm,9)} {fmt(bp,10)} {fmt(fl,9)} "
              f"{fmt(ps,7)} {fmt(cf,5)}")
    print(f"\nTotal wall time: {time.time() - t_start_all:.1f} s")
    print(f"metrics.json:   {metrics_path}")
    print(f"candidates:     {trace_path}")
    print(f"outputs:        {OUT_DIR} ({len(per_pair)} .npy files)")


if __name__ == '__main__':
    main()
