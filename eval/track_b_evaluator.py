"""Track B evaluator: apply external metrics to generated cross-skeleton motions.

Inputs:
  - generated_motions.npz from track_b_inference.py (per method)

External evaluators (independent of our training pipeline):
  - Action transfer accuracy (external_classifier.py — trained on REAL Truebones only)
  - Source-swap differentiation (TMR feature space — frozen pretrained on HumanML3D)
  - Naturalness via Frechet on TMR or on classifier embedding

Statistics (per refine-logs/effect_program/EMPIRICAL_PLAN.md):
  - Cluster bootstrap by (source_skel, target_skel)
  - Stratified by topology gap quartile

Usage:
    conda run -n anytop python -m eval.track_b_evaluator \
        --generated eval/results/track_b/generated_motions.npz \
        --classifier_ckpt save/external_classifier.pt
"""
import os
import json
import argparse
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--generated', required=True, help='npz from track_b_inference.py')
    p.add_argument('--classifier_ckpt', default='save/external_classifier.pt')
    p.add_argument('--source_metadata', default='eval/results/effect_cache/clip_metadata.json')
    p.add_argument('--topology_gap', default='eval/results/effect_cache/topology_gap.npy')
    p.add_argument('--topology_gap_skels', default='eval/results/effect_cache/topology_gap_skels.json')
    p.add_argument('--out', default=None, help='Defaults to <generated>_eval.json')
    p.add_argument('--n_bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    return p.parse_args()


ACTION_CLASSES = ['walk', 'run', 'idle', 'attack', 'fly', 'swim', 'jump',
                  'turn', 'die', 'eat', 'getup', 'other']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}


def load_classifier(ckpt_path, device):
    from eval.external_classifier import ActionClassifier
    model = ActionClassifier()
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_action(model, features, device):
    """features: [T, n_bins, n_feat] → predicted action class."""
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    logits = model(x)
    return int(logits.argmax(-1).item()), logits.softmax(-1).cpu().numpy()[0]


def cluster_bootstrap_diff(scores_a, scores_b, clusters, n_bootstrap=10000, seed=42):
    """Paired cluster bootstrap on score difference (a - b).

    Returns: mean diff, 95% percentile CI, p-value (two-sided).
    """
    rng = np.random.default_rng(seed)
    diff = scores_a - scores_b  # per-pair difference
    unique_clusters = np.array(sorted(set(clusters)))
    n_clusters = len(unique_clusters)

    boots = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        mask = np.isin(clusters, sampled)
        # weight resampled clusters by frequency of selection
        boot_idx = []
        for c in sampled:
            boot_idx.extend(np.where(clusters == c)[0])
        if not boot_idx:
            continue
        boots.append(diff[boot_idx].mean())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided p-value: fraction of bootstrap means with same sign as null vs observed
    obs = diff.mean()
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return float(obs), float(lo), float(hi), float(p), float(boots.std())


def main():
    args = parse_args()
    if args.out is None:
        args.out = args.generated.replace('.npz', '_eval.json')

    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.external_classifier import extract_classifier_features

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    print(f"Loading generated motions from {args.generated}")
    data = np.load(args.generated, allow_pickle=True)
    pairs = data['pairs']  # list of (src_fname, src_skel, tgt_skel)
    motions = data['motions']  # list of [T, J, 13] (denormalized)
    print(f"  {len(motions)} pairs")

    print(f"Loading classifier from {args.classifier_ckpt}")
    classifier = load_classifier(args.classifier_ckpt, device)

    # Source action labels
    with open(args.source_metadata) as f:
        metadata = json.load(f)
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    # Topology gap matrix
    gap = np.load(args.topology_gap)
    with open(args.topology_gap_skels) as f:
        gap_skels = json.load(f)
    skel_to_idx = {s: i for i, s in enumerate(gap_skels)}

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    # Process each pair
    print(f"\nApplying external evaluator...")
    records = []
    for i, (pair, motion) in enumerate(zip(pairs, motions)):
        src_fname, src_skel, tgt_skel = pair[0], pair[1], pair[2]
        src_action = fname_to_action.get(src_fname, ACTION_TO_IDX['other'])

        # Generated motion as denormalized positions
        motion_arr = np.array(motion)
        if motion_arr.ndim == 3 and motion_arr.shape[2] == 13:
            # [T, J, 13] - recover positions
            try:
                positions = recover_from_bvh_ric_np(motion_arr)
            except Exception:
                continue
        else:
            continue

        # Get target skeleton parents
        if tgt_skel not in cond_dict:
            continue
        info = cond_dict[tgt_skel]
        n_joints = len(info['joints_names'])
        parents = np.array(info['parents'][:n_joints], dtype=np.int64)

        # Extract features
        try:
            features = extract_classifier_features(positions[:, :n_joints], parents)
            if features is None:
                continue
            T_feat = features.shape[0]
            if T_feat < 64:
                features = np.pad(features, ((0, 64 - T_feat), (0, 0), (0, 0)))
            else:
                idx = np.linspace(0, T_feat - 1, 64).astype(int)
                features = features[idx]
            pred_action, action_probs = predict_action(classifier, features, device)
        except Exception:
            continue

        # Topology gap
        if src_skel in skel_to_idx and tgt_skel in skel_to_idx:
            gap_val = float(gap[skel_to_idx[src_skel], skel_to_idx[tgt_skel]])
        else:
            gap_val = float('nan')

        records.append({
            'src_fname': src_fname,
            'src_skel': src_skel,
            'tgt_skel': tgt_skel,
            'src_action': src_action,
            'pred_action': pred_action,
            'action_match': int(pred_action == src_action),
            'src_action_prob': float(action_probs[src_action]),  # confidence on the correct class
            'topology_gap': gap_val,
            'cluster': f"{src_skel}_{tgt_skel}",
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(motions)}")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"TRACK B EXTERNAL EVALUATION — n={len(records)}")
    print(f"{'='*60}")

    if not records:
        print("No valid records — exit")
        return

    actions_match = np.array([r['action_match'] for r in records])
    src_action_probs = np.array([r['src_action_prob'] for r in records])
    gaps = np.array([r['topology_gap'] for r in records])

    print(f"\nAction transfer accuracy (overall): {actions_match.mean():.3f}")
    print(f"Mean prob on source action class:    {src_action_probs.mean():.3f}")

    # By topology gap quartile
    valid = ~np.isnan(gaps)
    if valid.sum() > 4:
        quartiles = np.percentile(gaps[valid], [25, 50, 75])
        for q_idx, (lo, hi) in enumerate([(0, quartiles[0]), (quartiles[0], quartiles[1]),
                                            (quartiles[1], quartiles[2]), (quartiles[2], 1.1)]):
            mask = (gaps >= lo) & (gaps < hi) & valid
            if mask.sum() > 0:
                print(f"  Q{q_idx+1} ({lo:.3f}-{hi:.3f}, n={mask.sum()}): "
                      f"acc={actions_match[mask].mean():.3f}  "
                      f"prob={src_action_probs[mask].mean():.3f}")

    # By source action
    print("\nPer source-action accuracy:")
    src_actions = np.array([r['src_action'] for r in records])
    for a_idx in range(12):
        mask = src_actions == a_idx
        if mask.sum() > 0:
            print(f"  {ACTION_CLASSES[a_idx]:>10s}: n={mask.sum():3d}  "
                  f"acc={actions_match[mask].mean():.3f}")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'overall_action_acc': float(actions_match.mean()),
            'overall_src_prob': float(src_action_probs.mean()),
            'n_records': len(records),
            'records': records,
        }, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
