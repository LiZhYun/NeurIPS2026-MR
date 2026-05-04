"""Retrieval baselines for K-comparison (CPU-only).

Generates cross-skeleton motion retargeting OUTPUTS for 30 canonical eval pairs
using four training-free retrieval methods, and saves both the retrieved motion
tensors (shape [T, J, 13] in normalized Truebones format) and aggregated metrics.

Methods:
    label_random     — random target-skel clip, preferring matching source_label.
    psi_retrieval    — ψ-mean cosine retrieval on target-skel pool (no label filter).
    q_retrieval      — Q-signature (19-dim) cosine retrieval (no label filter).
    q_label_retrieval — Q-signature retrieval restricted to same-label pool.

Metrics per pair:
    - Action accuracy via external classifier (save/external_classifier.pt)
    - Action file-label match (retrieved coarse_label == source_label)
    - Q per-component L2 distance to source Q
    - ψ mean cosine distance to source ψ
    - Wall time

Stratified means by family_gap (near / absent / moderate / extreme) saved to metrics.json.

Usage:
    conda run -n anytop python -m eval.run_baselines
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))

# Make sure we can import from repo
sys.path.insert(0, str(ROOT))

from eval.external_classifier import (  # noqa: E402
    ACTION_CLASSES,
    ACTION_TO_IDX,
    ActionClassifier,
    extract_classifier_features,
)
from eval.pilot_Q_experiments import q_signature  # noqa: E402
from eval.quotient_extractor import extract_quotient  # noqa: E402

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
CLASSIFIER_CKPT = ROOT / 'save/external_classifier.pt'

OUT_DIR = ROOT / 'eval/results/k_compare'
METHODS = ['label_random', 'psi_retrieval', 'q_retrieval', 'q_label_retrieval']

SEED = 42


def l2_norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def cosine_sim(a, b):
    return l2_norm(a) @ l2_norm(b).T


def resample_features_to_T(feat, T=64):
    Tf = feat.shape[0]
    if Tf < T:
        feat = np.pad(feat, ((0, T - Tf), (0, 0), (0, 0)))
    else:
        idx = np.linspace(0, Tf - 1, T).astype(int)
        feat = feat[idx]
    return feat


def classify_motion(motion_raw, info, classifier, device='cpu'):
    """Run external classifier on a [T, J, 13] *normalized* motion tensor.

    Returns (pred_label_idx, pred_label_str).
    """
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    n_joints = len(info['joints_names'])
    parents = np.array(info['parents'][:n_joints], dtype=np.int64)
    mean = info['mean'][:n_joints]
    std = info['std'][:n_joints] + 1e-6

    motion_denorm = motion_raw[:, :n_joints] * std + mean
    positions = recover_from_bvh_ric_np(motion_denorm)
    feat = extract_classifier_features(positions, parents)
    if feat is None:
        return None, None
    feat = resample_features_to_T(feat, 64)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(x)
    pred = int(logits.argmax(-1).item())
    return pred, ACTION_CLASSES[pred]


def q_component_l2(q_src, q_tgt):
    """Per-component L2 distance between two Q dicts."""
    def _l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            # Pad along time dimension when shape differs
            if a.ndim == b.ndim and a.ndim >= 1:
                T = min(a.shape[0], b.shape[0])
                a = a[:T]
                b = b[:T]
            else:
                return None
        return float(np.linalg.norm((a - b).reshape(-1)))

    out = {
        'com_path': _l2(q_src['com_path'], q_tgt['com_path']),
        'heading_vel': _l2(q_src['heading_vel'], q_tgt['heading_vel']),
        'cadence': float(abs(float(q_src['cadence']) - float(q_tgt['cadence']))),
    }
    # contact_sched and limb_usage may differ in dims across skeletons; collapse
    cs_src = np.asarray(q_src['contact_sched']).reshape(q_src['contact_sched'].shape[0], -1).sum(axis=-1)
    cs_tgt = np.asarray(q_tgt['contact_sched']).reshape(q_tgt['contact_sched'].shape[0], -1).sum(axis=-1)
    T = min(len(cs_src), len(cs_tgt))
    out['contact_sched_aggregate'] = float(np.linalg.norm(cs_src[:T] - cs_tgt[:T]))
    # limb usage: compare sorted distributions (top-k)
    lu_src = -np.sort(-np.asarray(q_src['limb_usage']))[:5]
    lu_tgt = -np.sort(-np.asarray(q_tgt['limb_usage']))[:5]
    K = min(len(lu_src), len(lu_tgt))
    lu_src = np.pad(lu_src[:K], (0, max(0, 5 - K)))
    lu_tgt = np.pad(lu_tgt[:K], (0, max(0, 5 - K)))
    out['limb_usage_top5'] = float(np.linalg.norm(lu_src - lu_tgt))
    return out


def psi_mean_cos_distance(psi_src, psi_tgt):
    """1 - cosine similarity between mean-ψ."""
    a = psi_src.mean(axis=0)
    b = psi_tgt.mean(axis=0)
    sim = float(l2_norm(a) @ l2_norm(b))
    return 1.0 - sim


def build_q_sig_array(qc):
    """Build [N, 19] Q-signature matrix in q_meta order."""
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


def stratified_summary(all_entries):
    """Group metrics by family_gap bucket and by support_absent flag.

    The spec names four strata: near_present / absent / moderate / extreme.
    eval_pairs.json uses family_gap in {near, moderate, extreme} and a
    separate support_same_label integer (0 => 'absent' stratum).
    We therefore report four strata:
        near_present — family_gap=='near' (i.e. close families)
        absent       — support_same_label == 0  (no matching label on target)
        moderate     — family_gap=='moderate' (regardless of support)
        extreme      — family_gap=='extreme' (regardless of support)
    and an 'all' bucket.
    """
    buckets = defaultdict(list)
    for e in all_entries:
        if e['family_gap'] == 'near_present':
            buckets['near_present'].append(e)
        if e['family_gap'] == 'moderate':
            buckets['moderate'].append(e)
        if e['family_gap'] == 'extreme':
            buckets['extreme'].append(e)
        if e['support_same_label'] == 0:
            buckets['absent'].append(e)
    buckets['all'] = list(all_entries)

    numeric_keys = [
        'action_classifier_match',
        'action_file_label_match',
        'psi_mean_cos_distance',
        'q_com_path_l2',
        'q_heading_vel_l2',
        'q_contact_sched_l2',
        'q_cadence_abs_diff',
        'q_limb_usage_top5_l2',
        'wall_time_s',
    ]
    summary = {}
    for stratum, entries in buckets.items():
        summary[stratum] = {'n': len(entries)}
        for k in numeric_keys:
            vals = [e[k] for e in entries if e.get(k) is not None]
            summary[stratum][k] = float(np.mean(vals)) if vals else None
    return summary


def save_method_outputs(method, pair_outputs, per_pair_entries):
    method_dir = OUT_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)
    for fname, motion in pair_outputs:
        np.save(method_dir / fname, motion)
    stratified = stratified_summary(per_pair_entries)
    with open(method_dir / 'metrics.json', 'w') as f:
        json.dump(
            {
                'method': method,
                'per_pair': per_pair_entries,
                'stratified': stratified,
            },
            f,
            indent=2,
        )
    return stratified


def main():
    t_start_all = time.time()
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # ── Load all caches ──
    print('Loading caches...')
    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    psi_all = np.load(PSI_PATH)  # [1070, 64, 62]
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    fname_to_meta_idx = {m['fname']: i for i, m in enumerate(meta)}
    q_meta = list(qc['meta'])
    fname_to_q_idx = {m['fname']: i for i, m in enumerate(q_meta)}
    psi_mean = psi_all.mean(axis=1)  # [N, 62]

    print(f"  ψ cache: {psi_all.shape}  Q cache: {len(q_meta)}  meta: {len(meta)}  pairs: {len(pairs)}")

    # Build pools
    skel_to_meta_idx = defaultdict(list)
    skel_label_to_meta_idx = defaultdict(list)
    for i, m in enumerate(meta):
        skel_to_meta_idx[m['skeleton']].append(i)
        skel_label_to_meta_idx[(m['skeleton'], m['coarse_label'])].append(i)

    # Build Q signatures aligned to q_meta order
    print('Building Q signatures...')
    q_sigs = build_q_sig_array(qc)  # [N_q, 19]
    q_sig_dim = q_sigs.shape[1]
    print(f"  Q-sig dim: {q_sig_dim}")

    # Load classifier
    print('Loading external classifier on CPU...')
    device = 'cpu'
    classifier = ActionClassifier()
    ckpt = torch.load(CLASSIFIER_CKPT, map_location=device)
    classifier.load_state_dict(ckpt['model'])
    classifier.to(device).eval()
    print(f"  classifier val_acc during training: {ckpt.get('val_acc', 'unknown')}")

    # ── Precompute per-source context ──
    # For metrics, we need source Q and source ψ. Filter out pairs that miss caches.
    method_entries = {m: [] for m in METHODS}
    method_motions = {m: [] for m in METHODS}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        (OUT_DIR / method).mkdir(parents=True, exist_ok=True)

    # ── Iterate pairs ──
    for pair in pairs:
        pair_id = pair['pair_id']
        src_fname = pair['source_fname']
        src_skel = pair['source_skel']
        src_label = pair['source_label']
        tgt_skel = pair['target_skel']
        family_gap = pair['family_gap']
        support = pair['support_same_label']

        # Normalize family_gap buckets — eval_pairs.json uses 'near', 'moderate', 'absent', 'extreme'
        # We map 'near' → 'near_present' to match spec
        strat = {'near': 'near_present'}.get(family_gap, family_gap)

        if src_fname not in fname_to_meta_idx or src_fname not in fname_to_q_idx:
            print(f"  pair {pair_id}: source not in caches, skipping")
            continue
        if tgt_skel not in cond_dict:
            print(f"  pair {pair_id}: target {tgt_skel} not in cond, skipping")
            continue

        src_meta_idx = fname_to_meta_idx[src_fname]
        src_q_idx = fname_to_q_idx[src_fname]

        # Source features
        src_psi = psi_all[src_meta_idx]
        src_q_sig = q_sigs[src_q_idx]
        src_q_dict = extract_quotient(src_fname, cond_dict[src_skel],
                                      contact_groups=contact_groups,
                                      motion_dir=str(MOTIONS_DIR))

        # Target pool
        tgt_pool_meta = [i for i in skel_to_meta_idx[tgt_skel]
                        if meta[i]['fname'] != src_fname]  # defensive
        tgt_pool_label = [i for i in skel_label_to_meta_idx[(tgt_skel, src_label)]
                          if meta[i]['fname'] != src_fname]

        if not tgt_pool_meta:
            print(f"  pair {pair_id}: no target clips for {tgt_skel}, skipping")
            continue

        tgt_info = cond_dict[tgt_skel]

        # ── Method 1: label_random ──
        t0 = time.time()
        if tgt_pool_label:
            rand_idx = int(rng.choice(tgt_pool_label))
        else:
            rand_idx = int(rng.choice(tgt_pool_meta))
        label_random_fname = meta[rand_idx]['fname']
        label_random_time = time.time() - t0

        # ── Method 2: psi_retrieval ──
        t0 = time.time()
        pool_psi = psi_mean[tgt_pool_meta]  # [P, 62]
        sims = cosine_sim(psi_mean[src_meta_idx:src_meta_idx + 1], pool_psi)[0]
        psi_idx = tgt_pool_meta[int(np.argmax(sims))]
        psi_fname = meta[psi_idx]['fname']
        psi_time = time.time() - t0

        # ── Method 3: q_retrieval ──
        t0 = time.time()
        tgt_pool_q = [(i, fname_to_q_idx[meta[i]['fname']])
                      for i in tgt_pool_meta
                      if meta[i]['fname'] in fname_to_q_idx]
        if not tgt_pool_q:
            print(f"  pair {pair_id}: no q-pool for {tgt_skel}, skipping method 3")
            continue
        pool_idx_q = [m for m, _ in tgt_pool_q]
        pool_qi = [q for _, q in tgt_pool_q]
        sims_q = cosine_sim(src_q_sig[None], q_sigs[pool_qi])[0]
        q_idx = pool_idx_q[int(np.argmax(sims_q))]
        q_fname = meta[q_idx]['fname']
        q_time = time.time() - t0

        # ── Method 4: q_label_retrieval ──
        t0 = time.time()
        tgt_pool_q_label = [(i, fname_to_q_idx[meta[i]['fname']])
                            for i in tgt_pool_label
                            if meta[i]['fname'] in fname_to_q_idx]
        if tgt_pool_q_label:
            pool_idx_ql = [m for m, _ in tgt_pool_q_label]
            pool_qi_ql = [q for _, q in tgt_pool_q_label]
            sims_ql = cosine_sim(src_q_sig[None], q_sigs[pool_qi_ql])[0]
            ql_idx = pool_idx_ql[int(np.argmax(sims_ql))]
            ql_used_fallback = False
        else:
            ql_idx = q_idx  # fallback to q_retrieval
            ql_used_fallback = True
        ql_fname = meta[ql_idx]['fname']
        ql_time = time.time() - t0

        # ── Load motions, compute metrics, save ──
        method_retrievals = {
            'label_random': (label_random_fname, label_random_time),
            'psi_retrieval': (psi_fname, psi_time),
            'q_retrieval': (q_fname, q_time),
            'q_label_retrieval': (ql_fname, ql_time),
        }

        for method, (retr_fname, wall) in method_retrievals.items():
            t_metric_start = time.time()
            out_fname = f"pair_{pair_id:02d}_{src_skel}_to_{tgt_skel}.npy"
            # Load target motion VERBATIM
            motion_path = MOTIONS_DIR / retr_fname
            motion = np.load(motion_path)  # [T, J, 13] normalized
            method_motions[method].append((out_fname, motion.astype(np.float32)))

            # Metrics
            pred_idx, pred_str = classify_motion(motion, tgt_info, classifier, device=device)
            src_label_idx = ACTION_TO_IDX.get(src_label, ACTION_TO_IDX['other'])
            action_clf_match = None if pred_idx is None else int(pred_idx == src_label_idx)

            retr_meta = meta[fname_to_meta_idx[retr_fname]]
            action_file_match = int(retr_meta['coarse_label'] == src_label)

            # Q distance — need to extract Q for retrieved clip (on target skel)
            tgt_q_dict = extract_quotient(retr_fname, tgt_info,
                                          contact_groups=contact_groups,
                                          motion_dir=str(MOTIONS_DIR))
            q_dists = q_component_l2(src_q_dict, tgt_q_dict)

            # ψ distance
            retr_psi = psi_all[fname_to_meta_idx[retr_fname]]
            psi_cos_dist = psi_mean_cos_distance(src_psi, retr_psi)

            entry = {
                'pair_id': pair_id,
                'src_fname': src_fname,
                'src_skel': src_skel,
                'src_label': src_label,
                'tgt_skel': tgt_skel,
                'family_gap': strat,
                'support_same_label': support,
                'retrieved_fname': retr_fname,
                'retrieved_coarse_label': retr_meta['coarse_label'],
                'output_file': out_fname,
                'action_classifier_pred': pred_str,
                'action_classifier_match': action_clf_match,
                'action_file_label_match': action_file_match,
                'psi_mean_cos_distance': psi_cos_dist,
                'q_com_path_l2': q_dists.get('com_path'),
                'q_heading_vel_l2': q_dists.get('heading_vel'),
                'q_contact_sched_l2': q_dists.get('contact_sched_aggregate'),
                'q_cadence_abs_diff': q_dists.get('cadence'),
                'q_limb_usage_top5_l2': q_dists.get('limb_usage_top5'),
                'wall_time_s': wall + (time.time() - t_metric_start),
                'q_label_used_fallback': ql_used_fallback if method == 'q_label_retrieval' else None,
            }
            method_entries[method].append(entry)
            print(f"  pair {pair_id:02d} [{method:18s}] src={src_label:8s} -> "
                  f"retr={retr_meta['coarse_label']:8s} clf={pred_str or '--':8s} "
                  f"clf_match={action_clf_match} file_match={action_file_match}")

    # ── Save per-method outputs + metrics.json ──
    print('\nSaving per-method outputs + metrics.json...')
    per_method_strat = {}
    for method in METHODS:
        strat = save_method_outputs(method, method_motions[method], method_entries[method])
        per_method_strat[method] = strat

    # ── Cross-method summary ──
    summary = {
        'methods': METHODS,
        'total_pairs_processed': max(len(method_entries[m]) for m in METHODS),
        'total_wall_time_s': time.time() - t_start_all,
        'per_method_stratified': per_method_strat,
    }
    with open(OUT_DIR / 'baselines_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Console report
    print('\n' + '=' * 70)
    print('BASELINE SUMMARY')
    print('=' * 70)
    for method in METHODS:
        s = per_method_strat[method]
        print(f"\n-- {method} --")
        for stratum in ['all', 'near_present', 'absent', 'moderate', 'extreme']:
            if stratum not in s:
                continue
            st = s[stratum]
            n = st['n']
            clf = st.get('action_classifier_match')
            file_ = st.get('action_file_label_match')
            qcom = st.get('q_com_path_l2')
            print(f"  {stratum:14s} n={n:2d}  clf_acc={clf if clf is None else f'{clf:.3f}'}  "
                  f"file_acc={file_ if file_ is None else f'{file_:.3f}'}  "
                  f"q_com={qcom if qcom is None else f'{qcom:.3f}'}")
    print(f"\nTotal wall time: {summary['total_wall_time_s']:.1f} s")
    print(f"Summary: {OUT_DIR / 'baselines_summary.json'}")


if __name__ == '__main__':
    main()
