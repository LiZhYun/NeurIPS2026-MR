"""Multiple rapid Q-based experiments (Phase 2.6 extension, 2026-04-14).

Exp_1 (Q retrieval vs ψ retrieval): same protocol as Pilot C but using Q instead of ψ.
Exp_2 (Q-support characterization): distribution of Q-similarity stratified by support.
Exp_3 (Q salvage of support-absent): in the support-absent regime, is the nearest Q-neighbor
  on the target skeleton still a plausible match?
Exp_4 (Regime characterization): which actions dominate the support-absent regime, by skeleton
  and action class.
Exp_5 (Q-vs-psi action discrimination): can Q alone predict action class without label?

All experiments load both the ψ cache (existing) and the Q cache (just built) and operate
at metadata + signature level — no motion generation required.
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
OUT = ROOT / 'idea-stage/pilot_Q_results.json'
SEED = 42
N_SOURCES = 300


def q_signature(q_entry, k_limb=5):
    """Fixed-dim summary of a Q entry for retrieval.

    q_entry is a dict-like with: com_path [T,3], heading_vel [T], contact_sched [T] or [T,C],
    cadence scalar, limb_usage [K].
    Returns: 1-D feature vector.
    """
    com_path = q_entry['com_path'] if not hasattr(q_entry, 'item') else q_entry.item()['com_path']
    if isinstance(q_entry, np.ndarray) and q_entry.dtype == object:
        q = q_entry.item() if q_entry.ndim == 0 else q_entry
    else:
        q = q_entry

    # com_path summary (6-dim): total displacement magnitude, peak speed, x/y/z range
    com = np.asarray(q['com_path'])
    if com.size == 0:
        com_sum = np.zeros(6)
    else:
        disp = com[-1] - com[0]
        vel = np.diff(com, axis=0) if com.shape[0] > 1 else np.zeros((1, 3))
        peak_speed = np.linalg.norm(vel, axis=-1).max()
        com_sum = np.array([
            np.linalg.norm(disp),
            peak_speed,
            np.ptp(com[:, 0]),  # NumPy 2 compat (ndarray.ptp() removed)
            np.ptp(com[:, 1]),
            np.ptp(com[:, 2]),
            com[:, 1].mean(),  # mean height
        ])

    # heading_vel summary (4-dim): mean, std, max, min
    hv = np.asarray(q['heading_vel'])
    hv_sum = np.array([hv.mean(), hv.std(), hv.max(), hv.min()]) if hv.size > 0 else np.zeros(4)

    # contact_sched summary (4-dim): mean, std, contact-fraction (>0.5 frames), range
    cs = np.asarray(q['contact_sched'])
    if cs.ndim > 1:
        cs = cs.mean(axis=1)  # collapse to aggregate
    if cs.size > 0:
        cs_sum = np.array([cs.mean(), cs.std(), (cs > 0.5).mean(), np.ptp(cs)])
    else:
        cs_sum = np.zeros(4)

    # cadence scalar
    cad = float(q['cadence'])

    # limb_usage top-k (padded to k_limb)
    lu = np.asarray(q['limb_usage'])
    if lu.size > 0:
        top = -np.sort(-lu)[:k_limb]
        if top.size < k_limb:
            top = np.concatenate([top, np.zeros(k_limb - top.size)])
    else:
        top = np.zeros(k_limb)

    return np.concatenate([com_sum, hv_sum, cs_sum, [cad], top]).astype(np.float32)


def cosine(a, b):
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return an @ bn.T


def main():
    print("Loading caches...")
    psi_all = np.load(PSI_PATH)  # [1070, 64, 62]
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    q_meta = qc['meta']
    print(f"  ψ cache: {psi_all.shape}; Q cache: {len(q_meta)} clips; metadata: {len(meta)} clips")

    # Build q_signatures [N, D] for the Q cache order
    q_fname_to_idx = {m['fname']: i for i, m in enumerate(q_meta)}
    meta_to_q_idx = [q_fname_to_idx.get(m['fname'], -1) for m in meta]

    q_sig_list = []
    for i in range(len(q_meta)):
        q = {
            'com_path': qc['com_path'][i],
            'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        q_sig_list.append(q_signature(q))
    q_sigs = np.stack(q_sig_list)  # [N, D]
    D = q_sigs.shape[1]
    print(f"  Q signature dim: {D}")

    # Reindex q_sigs into meta-order (use -1 for missing; we filter)
    n_meta = len(meta)
    # Use psi_mean for ψ retrieval
    psi_mean = psi_all.mean(axis=1)  # [1070, 62]

    # Index by skeleton, by (skel, label)
    skel_to_idx = defaultdict(list)
    skel_label_to_idx = defaultdict(list)
    label_counts_by_skel = defaultdict(Counter)
    for i, m in enumerate(meta):
        skel_to_idx[m['skeleton']].append(i)
        skel_label_to_idx[(m['skeleton'], m['coarse_label'])].append(i)
        label_counts_by_skel[m['skeleton']][m['coarse_label']] += 1
    skeletons = sorted(skel_to_idx.keys())

    # Sample source clips from val (same as Pilot C)
    rng = np.random.default_rng(SEED)
    val_idx = [i for i, m in enumerate(meta) if m['split'] == 'val']
    source_indices = rng.choice(val_idx, size=min(N_SOURCES, len(val_idx)), replace=False)

    all_results = []
    for src_i in source_indices:
        src_m = meta[src_i]
        src_skel = src_m['skeleton']
        src_label = src_m['coarse_label']
        other_skels = [s for s in skeletons if s != src_skel]
        tgt_skel = str(rng.choice(other_skels))

        support_same_label = label_counts_by_skel[tgt_skel][src_label]
        tgt_pool = skel_to_idx[tgt_skel]
        tgt_pool_labels = [meta[j]['coarse_label'] for j in tgt_pool]

        # Fetch signatures — may miss if Q cache dropped this clip
        src_q_idx = meta_to_q_idx[src_i]
        tgt_pool_q = [meta_to_q_idx[j] for j in tgt_pool]
        valid_pool = [(j, qi) for j, qi in zip(tgt_pool, tgt_pool_q) if qi >= 0]
        if not valid_pool or src_q_idx < 0:
            continue
        pool_idx = [j for j, _ in valid_pool]
        pool_qi = [qi for _, qi in valid_pool]

        # ψ-only retrieval (no label filter)
        sims_psi = cosine(psi_mean[src_i:src_i+1], psi_mean[pool_idx])[0]
        psi_top1 = pool_idx[int(np.argmax(sims_psi))]
        psi_top1_match = meta[psi_top1]['coarse_label'] == src_label

        # Q-only retrieval (no label filter)
        sims_q = cosine(q_sigs[src_q_idx:src_q_idx+1], q_sigs[pool_qi])[0]
        q_top1_q_idx = pool_qi[int(np.argmax(sims_q))]
        q_top1_meta_idx = pool_idx[int(np.argmax(sims_q))]
        q_top1_match = meta[q_top1_meta_idx]['coarse_label'] == src_label

        # Top-3 Q retrieval
        q_order = np.argsort(-sims_q)[:3]
        q_top3_matches = [meta[pool_idx[o]]['coarse_label'] == src_label for o in q_order]

        # Combined (Q + action filter): restrict pool to same-label first
        match_pool_idx = [j for j in pool_idx if meta[j]['coarse_label'] == src_label]
        match_pool_qi = [meta_to_q_idx[j] for j in match_pool_idx if meta_to_q_idx[j] >= 0]
        match_pool_idx = [j for j in match_pool_idx if meta_to_q_idx[j] >= 0]
        if match_pool_idx:
            sims_q_filt = cosine(q_sigs[src_q_idx:src_q_idx+1], q_sigs[match_pool_qi])[0]
            q_filt_top1 = match_pool_idx[int(np.argmax(sims_q_filt))]
            used_filter = True
        else:
            q_filt_top1 = q_top1_meta_idx  # fallback
            used_filter = False

        # In-action Q discrimination: how close is src_q to same-action vs diff-action clips
        # across the WHOLE dataset (within-skel class-separability proxy)?
        all_q = q_sigs
        all_labels = np.array([m['coarse_label'] for m in meta])
        all_skels_same = np.array([m['skeleton'] == src_skel for m in meta])
        # NOT used for retrieval — just for characterization
        # Compute: mean sim to (same-label, any-skel != src) vs mean sim to (diff-label, any-skel != src)
        same_label_mask = (all_labels == src_label) & (~all_skels_same)
        diff_label_mask = (all_labels != src_label) & (~all_skels_same)
        # Protect against no-support case: same_label_mask may be empty
        if same_label_mask.sum() > 0:
            q_indices_same = [meta_to_q_idx[j] for j in np.where(same_label_mask)[0] if meta_to_q_idx[j] >= 0]
            q_sim_same = cosine(q_sigs[src_q_idx:src_q_idx+1], q_sigs[q_indices_same])[0].mean()
        else:
            q_sim_same = float('nan')
        q_indices_diff = [meta_to_q_idx[j] for j in np.where(diff_label_mask)[0] if meta_to_q_idx[j] >= 0]
        q_sim_diff = cosine(q_sigs[src_q_idx:src_q_idx+1], q_sigs[q_indices_diff])[0].mean()

        all_results.append({
            'src_fname': src_m['fname'],
            'src_skel': src_skel,
            'src_label': src_label,
            'tgt_skel': tgt_skel,
            'support_same_label': support_same_label,
            'psi_top1_match': psi_top1_match,
            'psi_top1_label': meta[psi_top1]['coarse_label'],
            'q_top1_match': q_top1_match,
            'q_top1_label': meta[q_top1_meta_idx]['coarse_label'],
            'q_top3_match_rate': float(np.mean(q_top3_matches)),
            'q_filtered_top1_is_valid': used_filter,
            'q_sim_same_label': float(q_sim_same),
            'q_sim_diff_label': float(q_sim_diff),
            'q_sim_ratio': float(q_sim_same / q_sim_diff) if (q_sim_diff != 0 and not np.isnan(q_sim_same)) else float('nan'),
        })

    n = len(all_results)
    print(f"\nCollected {n} valid queries")
    # Stratify by support-present/absent
    by_support = defaultdict(list)
    for r in all_results:
        key = 'present' if r['support_same_label'] > 0 else 'absent'
        by_support[key].append(r)
    print(f"  support-present: {len(by_support['present'])}")
    print(f"  support-absent:  {len(by_support['absent'])}")

    def summarize(bucket, keys):
        if not bucket:
            return {k: None for k in keys}
        out = {}
        for k in keys:
            vals = [r[k] for r in bucket if not (isinstance(r[k], float) and np.isnan(r[k]))]
            out[k] = float(np.mean(vals)) if vals else None
        return out

    # EXPERIMENT SUMMARIES

    # Exp_1: psi retrieval top-1 action match vs Q retrieval top-1 action match (no label filter)
    exp_1 = {
        'description': 'Q retrieval vs ψ retrieval: top-1 action match rate WITHOUT label filter',
        'overall_psi_top1_match': float(np.mean([r['psi_top1_match'] for r in all_results])),
        'overall_q_top1_match': float(np.mean([r['q_top1_match'] for r in all_results])),
        'by_support': {
            k: {
                'n': len(v),
                'psi_top1_match': float(np.mean([r['psi_top1_match'] for r in v])),
                'q_top1_match': float(np.mean([r['q_top1_match'] for r in v])),
                'q_top3_match': float(np.mean([r['q_top3_match_rate'] for r in v])),
            } for k, v in by_support.items()
        }
    }

    # Exp_2: Q-similarity stratification — is Q higher for same-label pairs than diff-label?
    exp_2 = {
        'description': 'Q-sim to same-label cross-skel clips vs diff-label cross-skel clips',
        'overall': {
            'q_sim_same_label': float(np.nanmean([r['q_sim_same_label'] for r in all_results])),
            'q_sim_diff_label': float(np.mean([r['q_sim_diff_label'] for r in all_results])),
            'q_sim_ratio (same/diff, >1 means Q discriminates)': float(np.nanmean([r['q_sim_ratio'] for r in all_results])),
        }
    }

    # Exp_3: for support-absent pairs, what does Q-retrieval give us?
    exp_3 = {
        'description': 'In support-absent regime: Q-retrieval picks WHICH action labels on target?',
        'n_absent': len(by_support['absent']),
        'q_top1_label_distribution': dict(Counter(r['q_top1_label'] for r in by_support['absent'])),
        'source_label_distribution_in_absent': dict(Counter(r['src_label'] for r in by_support['absent'])),
    }

    # Exp_4: Characterize support-absent regime — which actions/skeletons?
    exp_4 = {
        'description': 'Support-absent regime characterization',
        'src_label_counts_absent': dict(Counter(r['src_label'] for r in by_support['absent'])),
        'src_label_counts_present': dict(Counter(r['src_label'] for r in by_support['present'])),
        'src_label_absent_rate': {
            lab: (by_support['absent']
                  and sum(r['src_label'] == lab for r in by_support['absent']) / max(sum(r['src_label'] == lab for r in all_results), 1)) or 0.0
            for lab in set(r['src_label'] for r in all_results)
        },
    }

    # Exp_5: Q's intrinsic same-vs-diff label gap (per source)
    gap_per_source = [
        (r['q_sim_same_label'] - r['q_sim_diff_label'])
        for r in all_results if not np.isnan(r['q_sim_same_label'])
    ]
    exp_5 = {
        'description': 'Q-sim gap: (sim-to-same-label) - (sim-to-diff-label), across all valid sources',
        'n': len(gap_per_source),
        'mean_gap': float(np.mean(gap_per_source)) if gap_per_source else None,
        'std_gap': float(np.std(gap_per_source)) if gap_per_source else None,
        'frac_positive_gap': float(np.mean([g > 0 for g in gap_per_source])) if gap_per_source else None,
    }

    summary = {
        'seed': SEED,
        'n_queries': n,
        'q_sig_dim': int(D),
        'exp_1_q_vs_psi_retrieval': exp_1,
        'exp_2_q_sim_label_stratified': exp_2,
        'exp_3_support_absent_q_retrieval_labels': exp_3,
        'exp_4_support_absent_regime': exp_4,
        'exp_5_q_same_diff_gap': exp_5,
    }

    OUT.write_text(json.dumps(summary, indent=2))

    # Console report
    print("\n" + "=" * 70)
    print("EXP_1: Q vs ψ retrieval (top-1 action match, no label filter)")
    print("=" * 70)
    print(f"  Overall ψ top-1: {exp_1['overall_psi_top1_match']:.3f}")
    print(f"  Overall Q top-1: {exp_1['overall_q_top1_match']:.3f}")
    for k, v in exp_1['by_support'].items():
        print(f"  support-{k} (n={v['n']}): ψ={v['psi_top1_match']:.3f}  Q={v['q_top1_match']:.3f}  Q@3={v['q_top3_match']:.3f}")

    print("\n" + "=" * 70)
    print("EXP_2: Q discrimination of same-label vs diff-label cross-skeleton")
    print("=" * 70)
    for k, v in exp_2['overall'].items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 70)
    print("EXP_3: Support-absent regime — what does Q retrieve?")
    print("=" * 70)
    print(f"  n support-absent: {exp_3['n_absent']}")
    print(f"  Top retrieved labels (when source has no support on target):")
    for lab, count in sorted(exp_3['q_top1_label_distribution'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {lab}: {count}")
    print(f"  Source labels in support-absent regime:")
    for lab, count in sorted(exp_3['source_label_distribution_in_absent'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {lab}: {count}")

    print("\n" + "=" * 70)
    print("EXP_4: Support-absent rate per source action label")
    print("=" * 70)
    for lab, rate in sorted(exp_4['src_label_absent_rate'].items(), key=lambda x: -x[1])[:12]:
        n_abs = exp_4['src_label_counts_absent'].get(lab, 0)
        n_pres = exp_4['src_label_counts_present'].get(lab, 0)
        print(f"  {lab:10}: absent-rate={rate:.2f}  (absent={n_abs}, present={n_pres})")

    print("\n" + "=" * 70)
    print("EXP_5: Q-sim gap (same-label vs diff-label) per-source statistics")
    print("=" * 70)
    print(f"  n valid queries: {exp_5['n']}")
    print(f"  mean gap: {exp_5['mean_gap']:.4f}  (std {exp_5['std_gap']:.4f})")
    print(f"  fraction of queries with positive gap: {exp_5['frac_positive_gap']:.3f}")

    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
