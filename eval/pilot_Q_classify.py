"""Exp_8: Q vs ψ as action-classification features (within-skeleton, held-out).

For each skeleton with >= 6 clips, split its clips train/val, train a linear classifier
on ψ-mean and on Q-signature, and measure held-out accuracy. Aggregates across skeletons.

Purpose: does the 5-component Q (≈19-dim) discriminate actions at least as well as
the 62-dim ψ? This tests whether K's foundation is rich enough to carry classification
signal across different skeletons.

All on CPU, scikit-learn logistic regression with default settings.
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(str(PROJECT_ROOT))
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
OUT = ROOT / 'idea-stage/pilot_Q_classify_results.json'
SEED = 42


def q_signature(q, k_limb=5):
    com = np.asarray(q['com_path'])
    if com.size == 0:
        com_sum = np.zeros(6)
    else:
        disp = com[-1] - com[0]
        vel = np.diff(com, axis=0) if com.shape[0] > 1 else np.zeros((1, 3))
        peak_speed = np.linalg.norm(vel, axis=-1).max()
        com_sum = np.array([
            np.linalg.norm(disp), peak_speed,
            com[:, 0].ptp(), com[:, 1].ptp(), com[:, 2].ptp(), com[:, 1].mean(),
        ])
    hv = np.asarray(q['heading_vel'])
    hv_sum = np.array([hv.mean(), hv.std(), hv.max(), hv.min()]) if hv.size > 0 else np.zeros(4)
    cs = np.asarray(q['contact_sched'])
    if cs.ndim > 1:
        cs = cs.mean(axis=1)
    if cs.size > 0:
        cs_sum = np.array([cs.mean(), cs.std(), (cs > 0.5).mean(), cs.ptp()])
    else:
        cs_sum = np.zeros(4)
    cad = float(q['cadence'])
    lu = np.asarray(q['limb_usage'])
    if lu.size > 0:
        top = -np.sort(-lu)[:k_limb]
        if top.size < k_limb:
            top = np.concatenate([top, np.zeros(k_limb - top.size)])
    else:
        top = np.zeros(k_limb)
    return np.concatenate([com_sum, hv_sum, cs_sum, [cad], top]).astype(np.float32)


def main():
    psi_all = np.load(PSI_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    q_meta = qc['meta']
    q_fname_to_idx = {m['fname']: i for i, m in enumerate(q_meta)}

    # Build Q signatures
    q_sigs = {}
    for i in range(len(q_meta)):
        q = {
            'com_path': qc['com_path'][i],
            'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        q_sigs[q_meta[i]['fname']] = q_signature(q)

    psi_mean_all = psi_all.mean(axis=1)

    # Per-skeleton experiments (within-skeleton action classification)
    by_skel = defaultdict(list)
    for i, m in enumerate(meta):
        by_skel[m['skeleton']].append(i)

    results_per_skel = {}
    rng = np.random.default_rng(SEED)

    psi_accs, q_accs, n_test_total = [], [], 0
    # Macro across skeletons with enough clips
    for skel, idxs in by_skel.items():
        if len(idxs) < 6:
            continue
        # Labels
        labels = np.array([meta[i]['coarse_label'] for i in idxs])
        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            continue
        # Stratified 80/20 split
        idxs_np = np.array(idxs)
        train_i, test_i = [], []
        for lab in unique_labels:
            mask = labels == lab
            pool = idxs_np[mask]
            rng.shuffle(pool)
            n = len(pool)
            n_test = max(1, n // 5)
            test_i.extend(pool[:n_test].tolist())
            train_i.extend(pool[n_test:].tolist())
        if len(train_i) < 3 or len(test_i) < 1:
            continue

        # Build feature matrices
        try:
            X_psi_tr = psi_mean_all[train_i]
            y_tr = [meta[i]['coarse_label'] for i in train_i]
            X_psi_te = psi_mean_all[test_i]
            y_te = [meta[i]['coarse_label'] for i in test_i]

            # Q features — may be missing if Q cache dropped clip
            q_tr = [q_sigs.get(meta[i]['fname']) for i in train_i]
            q_te = [q_sigs.get(meta[i]['fname']) for i in test_i]
            if any(q is None for q in q_tr + q_te):
                continue
            X_q_tr = np.stack(q_tr)
            X_q_te = np.stack(q_te)
        except Exception as e:
            continue

        # Standardize
        scaler_psi = StandardScaler().fit(X_psi_tr)
        scaler_q = StandardScaler().fit(X_q_tr)
        X_psi_tr_s = scaler_psi.transform(X_psi_tr)
        X_psi_te_s = scaler_psi.transform(X_psi_te)
        X_q_tr_s = scaler_q.transform(X_q_tr)
        X_q_te_s = scaler_q.transform(X_q_te)

        # Train + eval
        try:
            clf_psi = LogisticRegression(max_iter=500, random_state=SEED).fit(X_psi_tr_s, y_tr)
            psi_acc = clf_psi.score(X_psi_te_s, y_te)
        except Exception:
            psi_acc = float('nan')
        try:
            clf_q = LogisticRegression(max_iter=500, random_state=SEED).fit(X_q_tr_s, y_tr)
            q_acc = clf_q.score(X_q_te_s, y_te)
        except Exception:
            q_acc = float('nan')

        results_per_skel[skel] = {
            'n_train': len(train_i),
            'n_test': len(test_i),
            'n_classes': len(unique_labels),
            'psi_acc': float(psi_acc),
            'q_acc': float(q_acc),
            'chance': 1.0 / len(unique_labels),
        }
        if not (np.isnan(psi_acc) or np.isnan(q_acc)):
            psi_accs.append(psi_acc)
            q_accs.append(q_acc)
            n_test_total += len(test_i)

    summary = {
        'n_skeletons_evaluated': len(results_per_skel),
        'n_test_total': n_test_total,
        'mean_psi_acc_macro': float(np.mean(psi_accs)) if psi_accs else None,
        'mean_q_acc_macro': float(np.mean(q_accs)) if q_accs else None,
        'mean_chance_macro': float(np.mean([v['chance'] for v in results_per_skel.values()])),
        'q_wins': int(sum(1 for p, q in zip(psi_accs, q_accs) if q > p)),
        'psi_wins': int(sum(1 for p, q in zip(psi_accs, q_accs) if p > q)),
        'ties': int(sum(1 for p, q in zip(psi_accs, q_accs) if p == q)),
        'per_skeleton': results_per_skel,
    }

    OUT.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("EXP_8: Within-skeleton action classification — Q vs ψ logistic regression")
    print("=" * 70)
    print(f"  Skeletons evaluated: {summary['n_skeletons_evaluated']}")
    print(f"  Total held-out clips: {summary['n_test_total']}")
    print(f"  Chance baseline (macro): {summary['mean_chance_macro']:.3f}")
    print(f"  ψ accuracy (macro):     {summary['mean_psi_acc_macro']:.3f}")
    print(f"  Q accuracy (macro):     {summary['mean_q_acc_macro']:.3f}")
    print(f"  Q wins: {summary['q_wins']}  ψ wins: {summary['psi_wins']}  ties: {summary['ties']}")
    print(f"\n  Per-skel details (top 10 by n_test):")
    sorted_skels = sorted(results_per_skel.items(), key=lambda x: -x[1]['n_test'])[:10]
    for skel, r in sorted_skels:
        print(f"    {skel:22s}: n_test={r['n_test']:3d}  classes={r['n_classes']:2d}  ψ={r['psi_acc']:.2f}  Q={r['q_acc']:.2f}  chance={r['chance']:.2f}")
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
