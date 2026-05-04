"""DIFT null-baseline probe: random-init AnyTop vs pretrained.

Tests whether the partial DIFT signal (motion_probe=0.489, skel_probe=0.285)
comes from LEARNED diffusion features or from skeleton geometry leaking
through the random network architecture.

We create an AnyTop model with the SAME architecture as the pretrained one
but with random initialization (no weight loading), then run the identical
DIFT probe pipeline from eval/dift_probe.py.

If probes still show signal (e.g., motion_probe > 0.3), the signal is
structural (architecture leaks skeleton geometry).
If they drop to chance, the pretrained features carry real learned semantics.

Usage:
    conda run -n anytop python -m eval.dift_probe_null_baseline \
        --t_seg 3 --layer 3 --n_clips 200
"""
import os
import json
import argparse
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--args_path', type=str,
                   default='save/all_model_dataset_truebones_bs_16_latentdim_128/args.json',
                   help='Path to args.json from the pretrained model directory')
    p.add_argument('--t_seg', type=int, default=3, help='Diffusion timestep for DIFT extraction')
    p.add_argument('--layer', type=int, default=3, help='Encoder layer to extract from')
    p.add_argument('--pool', choices=['time_joint', 'time_only', 'joint_only', 'none_flat'],
                   default='time_joint', help='Pooling strategy')
    p.add_argument('--n_clips', type=int, default=200, help='Number of clips to probe')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', type=str, default='eval/results/dift_probe_null_baseline.json')
    return p.parse_args()


def create_random_anytop(args_path, device):
    """Create an AnyTop model with random weights (no pretrained loading)."""
    from utils.model_util import create_model_and_diffusion_general_skeleton

    with open(args_path) as f:
        args_d = json.load(f)

    class NS:
        def __init__(self, d):
            self.__dict__.update(d)
    args = NS(args_d)

    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    # NO weight loading -- this is the key difference from dift_probe.py
    model.to(device)
    model.eval()
    return model, diffusion, args


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Import probe helpers from dift_probe (reuse without modification)
    from eval.dift_probe import encode_dift_multi, pool_features

    # Load split & labels
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)
    with open('dataset/truebones/zoo/truebones_processed/train_val_split.json') as f:
        split = json.load(f)
    val_files = split['val'][:args.n_clips]
    print(f"[NULL BASELINE] Probing {len(val_files)} val clips")

    # Create randomly-initialized model (same architecture, no trained weights)
    print(f"Creating RANDOM-INIT AnyTop from {args.args_path}")
    model, diffusion, m_args = create_random_anytop(args.args_path, device)
    print(f"  num_layers={m_args.layers}, latent_dim={m_args.latent_dim}, num_frames={m_args.num_frames}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,} (ALL random)")

    # Load data
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    # T5 for joint name embeddings
    from model.conditioners import T5Conditioner
    t5 = T5Conditioner(name=m_args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device='cuda')

    n_frames = m_args.num_frames

    def encode_motion(fname):
        if fname not in label_map:
            return None
        skel = label_map[fname]['skeleton']
        if skel not in cond_dict:
            return None
        raw = np.load(pjoin(opt.motion_dir, fname))
        T, J_src, _ = raw.shape
        if T < n_frames:
            pad = np.zeros((n_frames - T, J_src, 13))
            raw = np.concatenate([raw, pad], axis=0)
        else:
            raw = raw[(T - n_frames)//2:(T - n_frames)//2 + n_frames]

        info = cond_dict[skel]
        mean = info['mean']
        std = info['std'] + 1e-6
        norm = np.nan_to_num((raw - mean[None, :]) / std[None, :])

        from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
        from data_loaders.tensors import truebones_batch_collate, create_padded_relation

        max_joints = opt.max_joints
        n_joints = J_src
        feature_len = opt.feature_len

        # Pad motion
        x = np.zeros((n_frames, max_joints, feature_len))
        x[:, :n_joints, :] = norm
        x_t = torch.tensor(x).permute(1, 2, 0).float().unsqueeze(0).to(device)  # [1, J, 13, T]

        # tpos
        tpos_raw = info['tpos_first_frame']
        tpos = np.zeros((max_joints, feature_len))
        tpos[:n_joints] = (tpos_raw - mean) / std
        tpos = np.nan_to_num(tpos)
        tpos_t = torch.tensor(tpos).float().unsqueeze(0).to(device)

        # joint names
        names = info['joints_names']
        names_emb = t5(t5.tokenize(names)).detach().cpu().numpy()
        names_padded = np.zeros((max_joints, names_emb.shape[1]))
        names_padded[:n_joints] = names_emb
        names_t = torch.tensor(names_padded).float().unsqueeze(0).to(device)

        # graph_dist + joint_relations
        gd = create_padded_relation(info['joints_graph_dist'], max_joints, n_joints)
        jr = create_padded_relation(info['joint_relations'], max_joints, n_joints)
        gd_t = torch.tensor(gd).long().unsqueeze(0).to(device)
        jr_t = torch.tensor(jr).long().unsqueeze(0).to(device)

        # joints_mask
        jmask_5d = torch.zeros(1, 1, 1, max_joints + 1, max_joints + 1, device=device)
        jmask_5d[0, 0, 0, :n_joints + 1, :n_joints + 1] = 1.0

        # temporal_mask
        tmask = create_temporal_mask_for_window(31, n_frames)
        tmask_t = torch.tensor(tmask).unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(device)

        y = {
            'joints_mask':       jmask_5d,
            'mask':              tmask_t,
            'tpos_first_frame':  tpos_t,
            'joints_names_embs': names_t,
            'graph_dist':        gd_t,
            'joints_relations':  jr_t,
            'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
            'n_joints':          torch.tensor([n_joints]),
        }
        return x_t, y, skel, label_map[fname]['coarse_label'], n_joints

    layers = [args.layer]
    print(f"Extracting layers: {layers}, pool={args.pool}")

    # Extract DIFT features for all val clips
    all_feats, all_skels, all_motions = [], [], []
    n_done = 0
    n_failed = 0
    for fname in val_files:
        try:
            res = encode_motion(fname)
            if res is None:
                n_failed += 1
                continue
            x, y, skel, motion, n_joints = res
            feat_dict = encode_dift_multi(model, diffusion, x, y, args.t_seg, layers)
            v = pool_features(feat_dict, n_joints, args.pool)
            all_feats.append(v)
            all_skels.append(skel)
            all_motions.append(motion)
            n_done += 1
            if n_done % 50 == 0:
                print(f"  done {n_done}/{len(val_files)} (failed {n_failed})")
        except Exception as e:
            n_failed += 1
            if n_failed < 5:
                import traceback
                print(f"  fail on {fname}: {e}")
                traceback.print_exc()

    print(f"\nExtracted {n_done} DIFT vectors (random-init), {n_failed} failures")
    if n_done < 20:
        print("Too few -- aborting probes")
        return

    all_feats = np.array(all_feats)
    print(f"Feature shape: {all_feats.shape}")

    # Run probes (identical to dift_probe.py)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
    N, D = all_feats.shape

    le_s = LabelEncoder().fit(all_skels)
    y_s = le_s.transform(all_skels)
    skel_acc = 0.0
    if len(set(y_s)) > 1:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        skel_acc = cross_val_score(clf, all_feats, y_s, cv=min(5, len(set(y_s))), scoring='accuracy').mean()

    # motion probe (excluding "other")
    valid = [i for i, m in enumerate(all_motions) if m != 'other']
    motion_acc = 0.0
    if len(valid) > 20:
        z_v = all_feats[valid]
        m_v = [all_motions[i] for i in valid]
        le_m = LabelEncoder().fit(m_v)
        y_m = le_m.transform(m_v)
        if len(set(y_m)) > 2:
            clf2 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            motion_acc = cross_val_score(clf2, z_v, y_m, cv=min(5, len(set(y_m))), scoring='accuracy').mean()

    # Cosine analyses
    z_norm = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8)
    cos = z_norm @ z_norm.T
    same_s, cross_s = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (same_s if all_skels[i] == all_skels[j] else cross_s).append(cos[i, j])
    skel_gap = (np.mean(same_s) - np.mean(cross_s)) if same_s and cross_s else 0
    mean_cos = float(np.mean(cos[~np.eye(N, dtype=bool)]))

    # P@5
    hits, total = 0, 0
    for i in range(N):
        if all_motions[i] == 'other':
            continue
        sims = [(j, cos[i, j]) for j in range(N) if j != i and all_skels[j] != all_skels[i]]
        sims.sort(key=lambda x: -x[1])
        top5 = sims[:5]
        hits += sum(1 for j, _ in top5 if all_motions[j] == all_motions[i])
        total += len(top5)
    p5 = hits / max(total, 1)

    # Load pretrained results for comparison
    pretrained_path = 'eval/results/dift_t3_l3.json'
    pretrained = {}
    if os.path.exists(pretrained_path):
        with open(pretrained_path) as f:
            pretrained = json.load(f)

    results = {
        'experiment': 'null_baseline_random_init',
        'args_path': args.args_path,
        't_seg': args.t_seg,
        'layer': args.layer,
        'n_val': N,
        'feature_dim': D,
        'skel_probe_holdout': float(skel_acc),
        'motion_probe_holdout': float(motion_acc),
        'z_mean_cos_holdout': mean_cos,
        'skel_cos_gap_holdout': float(skel_gap),
        'cross_skel_p5_holdout': float(p5),
    }

    print()
    print("=" * 60)
    print(f"NULL BASELINE RESULTS (random-init, t={args.t_seg}, layer={args.layer})")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # Side-by-side comparison
    if pretrained:
        print()
        print("-" * 60)
        print("COMPARISON: Pretrained vs Random-Init")
        print("-" * 60)
        metrics = [
            ('skel_probe_holdout',  'skel_probe',  'lower is better'),
            ('motion_probe_holdout','motion_probe', 'higher is better'),
            ('z_mean_cos_holdout',  'z_mean_cos',   ''),
            ('skel_cos_gap_holdout','skel_cos_gap', 'lower is better'),
            ('cross_skel_p5_holdout','cross_skel_p5','higher is better'),
        ]
        for key, short, note in metrics:
            pre_val = pretrained.get(key, float('nan'))
            null_val = results[key]
            delta = null_val - pre_val
            print(f"  {short:20s}  pretrained={pre_val:.4f}  random={null_val:.4f}  delta={delta:+.4f}  ({note})")

        # Verdict
        print()
        motion_drop = pretrained.get('motion_probe_holdout', 0) - results['motion_probe_holdout']
        skel_drop = pretrained.get('skel_probe_holdout', 0) - results['skel_probe_holdout']
        if results['motion_probe_holdout'] > 0.30:
            print("VERDICT: Random network STILL shows motion signal (>0.30).")
            print("         => Signal is STRUCTURAL (architecture leaks geometry).")
            print("         => Pretrained features may NOT carry real learned semantics.")
        else:
            print("VERDICT: Random network motion probe DROPPED below 0.30.")
            print(f"         => Motion signal drop: {motion_drop:.4f}")
            print("         => Pretrained DIFT features carry REAL learned semantics.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
