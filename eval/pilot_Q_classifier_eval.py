"""Exp_9: Q-retrieval action accuracy as measured by the EXTERNAL classifier.

Protocol:
  - For each (source ∈ val, target_skel) pair, retrieve top-1 on target by Q-signature
    (no action-label filter — true cross-skeleton retrieval).
  - Load the retrieved motion.
  - Run the external classifier on the retrieved motion.
  - Measure: what fraction of retrievals get classified as the SOURCE action?
  - Stratify by support-present / support-absent.

This is a stronger quality signal than file-label match in Exp_1, because the
classifier evaluates the actual motion content via topology-normalised depth-bin
features — independent of our ψ and of file naming conventions.

Baselines:
  - ψ-retrieval (top-1 by cosine of ψ_mean)
  - Random retrieval (uniform over target skel)
  - Label-matched random retrieval (ceiling for support-present)
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
CLF_CKPT = ROOT / 'save/external_classifier.pt'
OUT = ROOT / 'idea-stage/pilot_Q_classifier_eval.json'
SEED = 42
N_PAIRS = 150  # keep tight for speed

sys.path.insert(0, str(ROOT))
from eval.external_classifier import ActionClassifier, extract_classifier_features, ACTION_CLASSES
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np


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


def cosine(a, b):
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return an @ bn.T


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load external classifier
    state = torch.load(str(CLF_CKPT), map_location='cpu', weights_only=False)
    clf = ActionClassifier().to(device).eval()
    clf.load_state_dict(state['model'])
    print(f"Loaded external classifier: {CLF_CKPT}")

    # Load caches
    psi_all = np.load(PSI_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    q_meta = qc['meta']
    q_fname_to_idx = {m['fname']: i for i, m in enumerate(q_meta)}
    meta_to_q_idx = [q_fname_to_idx.get(m['fname'], -1) for m in meta]

    # Build Q sig matrix
    q_sig_list = [q_signature({
        'com_path': qc['com_path'][i],
        'heading_vel': qc['heading_vel'][i],
        'contact_sched': qc['contact_sched'][i],
        'cadence': float(qc['cadence'][i]),
        'limb_usage': qc['limb_usage'][i],
    }) for i in range(len(q_meta))]
    q_sigs = np.stack(q_sig_list)
    psi_mean = psi_all.mean(axis=1)

    # Indexing
    skel_to_idx = defaultdict(list)
    label_counts_by_skel = defaultdict(Counter)
    for i, m in enumerate(meta):
        skel_to_idx[m['skeleton']].append(i)
        label_counts_by_skel[m['skeleton']][m['coarse_label']] += 1
    skeletons = sorted(skel_to_idx.keys())

    # Load cond for parents (needed by classifier)
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'), allow_pickle=True).item()
    motion_dir = os.path.join(DATASET_DIR, 'motions')

    def classify(fname, skel):
        """Load motion, run external classifier, return predicted action label."""
        m = np.load(os.path.join(motion_dir, fname))  # [T, J, 13]
        J_skel = cond[skel]['offsets'].shape[0]
        if m.shape[1] > J_skel:
            m = m[:, :J_skel]
        positions = recover_from_bvh_ric_np(m.astype(np.float32))  # [T, J, 3]
        parents = cond[skel]['parents']
        feats = extract_classifier_features(positions, parents)
        if feats is None:
            return None
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            logits = clf(x)
            pred_idx = int(logits.argmax(-1).item())
        return ACTION_CLASSES[pred_idx]

    rng = np.random.default_rng(SEED)
    val_idx = [i for i, m in enumerate(meta) if m['split'] == 'val']
    source_indices = rng.choice(val_idx, size=min(N_PAIRS, len(val_idx)), replace=False)

    results = []
    for k, src_i in enumerate(source_indices):
        if k % 20 == 0:
            print(f"  {k}/{len(source_indices)}")
        src_m = meta[src_i]
        src_skel = src_m['skeleton']
        src_label = src_m['coarse_label']
        other_skels = [s for s in skeletons if s != src_skel]
        tgt_skel = str(rng.choice(other_skels))
        support_same = label_counts_by_skel[tgt_skel][src_label]

        src_q_idx = meta_to_q_idx[src_i]
        if src_q_idx < 0:
            continue
        tgt_pool = skel_to_idx[tgt_skel]
        tgt_pool_q = [meta_to_q_idx[j] for j in tgt_pool]
        valid = [(j, qi) for j, qi in zip(tgt_pool, tgt_pool_q) if qi >= 0]
        if not valid:
            continue
        pool_idx = [j for j, _ in valid]
        pool_qi = [qi for _, qi in valid]

        # Q-retrieval (no label filter)
        sims_q = cosine(q_sigs[src_q_idx:src_q_idx+1], q_sigs[pool_qi])[0]
        q_top1 = pool_idx[int(np.argmax(sims_q))]
        q_top1_fname = meta[q_top1]['fname']

        # ψ-retrieval
        sims_psi = cosine(psi_mean[src_i:src_i+1], psi_mean[pool_idx])[0]
        psi_top1 = pool_idx[int(np.argmax(sims_psi))]
        psi_top1_fname = meta[psi_top1]['fname']

        # Random target
        rand_idx = pool_idx[rng.integers(len(pool_idx))]
        rand_fname = meta[rand_idx]['fname']

        # Classify the source motion itself (sanity — should usually match src_label)
        # and each retrieved target motion
        src_pred = classify(src_m['fname'], src_skel)
        q_pred = classify(q_top1_fname, tgt_skel)
        psi_pred = classify(psi_top1_fname, tgt_skel)
        rand_pred = classify(rand_fname, tgt_skel)

        results.append({
            'src_fname': src_m['fname'],
            'src_skel': src_skel,
            'src_label': src_label,
            'tgt_skel': tgt_skel,
            'support_same_label': support_same,
            'src_classifier_pred': src_pred,
            'q_retrieved_fname': q_top1_fname,
            'q_classifier_pred': q_pred,
            'psi_retrieved_fname': psi_top1_fname,
            'psi_classifier_pred': psi_pred,
            'random_pred': rand_pred,
            'q_match_src': src_pred is not None and q_pred == src_pred,
            'psi_match_src': src_pred is not None and psi_pred == src_pred,
            'random_match_src': src_pred is not None and rand_pred == src_pred,
            'q_match_label': q_pred == src_label,
            'psi_match_label': psi_pred == src_label,
            'random_match_label': rand_pred == src_label,
        })

    n = len(results)
    print(f"\nCollected {n} valid evaluations")
    by_support = defaultdict(list)
    for r in results:
        key = 'present' if r['support_same_label'] > 0 else 'absent'
        by_support[key].append(r)

    def rate(bucket, key):
        if not bucket:
            return None
        return float(np.mean([r[key] for r in bucket]))

    summary = {
        'n_pairs': n,
        'by_support': {
            k: {
                'n': len(v),
                'q_match_src_classifier_pred_rate': rate(v, 'q_match_src'),
                'psi_match_src_classifier_pred_rate': rate(v, 'psi_match_src'),
                'random_match_src_classifier_pred_rate': rate(v, 'random_match_src'),
                'q_match_src_file_label_rate': rate(v, 'q_match_label'),
                'psi_match_src_file_label_rate': rate(v, 'psi_match_label'),
                'random_match_src_file_label_rate': rate(v, 'random_match_label'),
            } for k, v in by_support.items()
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("EXP_9: Q-retrieval action accuracy via independent external classifier")
    print("=" * 70)
    print("  (classifier-pred match means: classifier predicts the SAME action on")
    print("   retrieved target motion as it does on source motion)")
    for strat, s in summary['by_support'].items():
        print(f"\n  support-{strat} (n={s['n']}):")
        print(f"    random            | src-classifier-pred-match: {s['random_match_src_classifier_pred_rate']:.3f} | file-label-match: {s['random_match_src_file_label_rate']:.3f}")
        print(f"    ψ-retrieval       | src-classifier-pred-match: {s['psi_match_src_classifier_pred_rate']:.3f} | file-label-match: {s['psi_match_src_file_label_rate']:.3f}")
        print(f"    Q-retrieval       | src-classifier-pred-match: {s['q_match_src_classifier_pred_rate']:.3f} | file-label-match: {s['q_match_src_file_label_rate']:.3f}")

    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
