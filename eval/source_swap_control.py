"""Source-swap control: did the model learn clip-specific behavior or only action-level?

For pair (source_A, target_B), generate via two source clips of the SAME action class:
  - source_A → motion_AB
  - source_A' → motion_A'B (different source, same action class)

If outputs are nearly identical → method only conditions on action label, not clip-specific.

Locked metric (per EMPIRICAL_PLAN.md):
  - Embedding space: TMR feature space (or our external classifier penultimate)
  - Distance: cosine distance
  - Pre-registered threshold: max(0.15, 0.5 × median pairwise distance among real target-skel clips)

Usage:
    conda run -n anytop python -m eval.source_swap_control \
        --gen_a path/to/source_A_motions.npz --gen_b path/to/source_Aprime_motions.npz
"""
import os
import json
import argparse
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gen_a', required=True, help='Generated motions for source A')
    p.add_argument('--gen_b', required=True, help='Generated motions for source A_prime')
    p.add_argument('--classifier_ckpt', default='save/external_classifier.pt')
    p.add_argument('--threshold', type=float, default=None,
                   help='Pre-registered threshold; if None, computed from real data')
    p.add_argument('--out', default=None)
    return p.parse_args()


@torch.no_grad()
def get_classifier_features(motion_arr, parents, n_joints, classifier, device):
    """Get penultimate-layer features from external classifier as embedding."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.external_classifier import extract_classifier_features

    positions = recover_from_bvh_ric_np(motion_arr[:, :n_joints])
    feats = extract_classifier_features(positions, parents)
    if feats is None:
        return None
    if feats.shape[0] < 64:
        feats = np.pad(feats, ((0, 64 - feats.shape[0]), (0, 0), (0, 0)))
    else:
        idx = np.linspace(0, feats.shape[0] - 1, 64).astype(int)
        feats = feats[idx]
    x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    # Run through everything except final classification layer
    B, T, D, F_ = x.shape
    x = x.permute(0, 2, 3, 1).reshape(B, D * F_, T)
    # Conv layers up to but not including final Linear:
    h = x
    for layer in classifier.conv[:-1]:  # all but final classifier head
        h = layer(h)
    return h.cpu().numpy().flatten()


def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b))


def compute_real_threshold(target_skels, cond_dict, motion_dir, classifier, device, n_per_skel=5):
    """Compute median pairwise cosine distance among real target-skel clips for threshold calibration."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    distances = []
    for skel in target_skels:
        if skel not in cond_dict:
            continue
        info = cond_dict[skel]
        n_joints = len(info['joints_names'])
        parents = np.array(info['parents'][:n_joints], dtype=np.int64)
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints] + 1e-6

        files = sorted(f for f in os.listdir(motion_dir) if f.startswith(f'{skel}_'))[:n_per_skel]
        embs = []
        for fn in files:
            try:
                raw = np.load(os.path.join(motion_dir, fn))
                m_denorm = raw[:, :n_joints] * std + mean
                emb = get_classifier_features(m_denorm, parents, n_joints, classifier, device)
                if emb is not None:
                    embs.append(emb)
            except Exception:
                continue
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                distances.append(cosine_distance(embs[i], embs[j]))
    if not distances:
        return 0.15
    return float(np.median(distances))


def main():
    args = parse_args()
    if args.out is None:
        args.out = args.gen_a.replace('.npz', '_swap_eval.json')

    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from eval.external_classifier import ActionClassifier

    fixseed(42)
    dist_util.setup_dist(0)
    device = dist_util.dev()

    print(f"Loading classifier from {args.classifier_ckpt}")
    classifier = ActionClassifier()
    state = torch.load(args.classifier_ckpt, map_location='cpu', weights_only=False)
    classifier.load_state_dict(state['model'])
    classifier.to(device).eval()

    print(f"Loading source A: {args.gen_a}")
    a = np.load(args.gen_a, allow_pickle=True)
    print(f"Loading source A_prime: {args.gen_b}")
    b = np.load(args.gen_b, allow_pickle=True)

    pairs_a = a['pairs']
    motions_a = a['motions']
    pairs_b = b['pairs']
    motions_b = b['motions']

    # Build target_skel → embedding pairs for matching
    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    # Compute embeddings
    print("\nComputing embeddings...")
    emb_a, emb_b = [], []
    for (pair_a, m_a), (pair_b, m_b) in zip(zip(pairs_a, motions_a), zip(pairs_b, motions_b)):
        if pair_a[2] != pair_b[2]:
            continue  # target must match
        tgt_skel = pair_a[2]
        info = cond_dict[tgt_skel]
        n_j = len(info['joints_names'])
        parents = np.array(info['parents'][:n_j], dtype=np.int64)
        try:
            ea = get_classifier_features(np.array(m_a), parents, n_j, classifier, device)
            eb = get_classifier_features(np.array(m_b), parents, n_j, classifier, device)
            if ea is not None and eb is not None:
                emb_a.append((tgt_skel, ea))
                emb_b.append((tgt_skel, eb))
        except Exception:
            continue

    if not emb_a:
        print("No valid embedding pairs — exit")
        return

    # Compute threshold from real data if not provided
    if args.threshold is None:
        target_skels = list(set(p[0] for p in emb_a))
        print(f"\nComputing threshold from real clips of {len(target_skels)} skeletons...")
        median_real = compute_real_threshold(target_skels, cond_dict, opt.motion_dir, classifier, device)
        threshold = max(0.15, 0.5 * median_real)
        print(f"  Median real pairwise distance: {median_real:.4f}")
        print(f"  Threshold (max(0.15, 0.5 × median)): {threshold:.4f}")
    else:
        threshold = args.threshold
        print(f"\nUsing fixed threshold: {threshold:.4f}")

    # Compute pairwise distances
    distances = []
    for (s_a, ea), (s_b, eb) in zip(emb_a, emb_b):
        d = cosine_distance(ea, eb)
        distances.append({'target_skel': s_a, 'distance': d, 'above_threshold': d > threshold})

    distances_arr = np.array([d['distance'] for d in distances])
    n_above = sum(1 for d in distances if d['above_threshold'])
    pass_rate = n_above / len(distances)

    print(f"\n{'='*50}")
    print(f"SOURCE-SWAP DIFFERENTIATION CONTROL")
    print(f"{'='*50}")
    print(f"N pairs: {len(distances)}")
    print(f"Mean cosine distance: {distances_arr.mean():.4f} ± {distances_arr.std():.4f}")
    print(f"Median distance:      {np.median(distances_arr):.4f}")
    print(f"Pass rate (above {threshold:.3f}): {n_above}/{len(distances)} = {pass_rate:.2%}")
    pre_reg_pass = pass_rate >= 0.80
    print(f"\nPre-registered criterion (≥80% above threshold): {'PASS' if pre_reg_pass else 'FAIL'}")
    if not pre_reg_pass:
        print("→ Method may only condition on action label, not clip-specific behavior")

    out_data = {
        'threshold': threshold,
        'mean_distance': float(distances_arr.mean()),
        'pass_rate': float(pass_rate),
        'pre_registered_pass': bool(pre_reg_pass),
        'distances': distances,
    }
    with open(args.out, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
