"""DIFT (Diffusion Features) extraction probe for AnyTop pretrained model.

Tests the hypothesis: AnyTop's trained denoiser has implicit cross-skeleton
motion semantics in its intermediate features. Following AnyTop §5.6, we
extract features at diffusion step t_seg=3 from layer l_seg=1, then run
the same skeleton/motion probes we use for our Stage 1 encoder.

If DIFT features pass our Stage 1 targets out-of-the-box, we can SKIP
training a Stage 1 encoder entirely and use them as conditioning for Stage 2.

Usage:
    conda run -n anytop python -m eval.dift_probe \
        --model_path save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt \
        --t_seg 3 --layer 1
"""
import os
import json
import argparse
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--t_seg', type=int, default=3, help='Diffusion timestep for DIFT extraction')
    p.add_argument('--layer', type=int, default=1, help='Encoder layer to extract from (-1 = concat all)')
    p.add_argument('--pool', choices=['time_joint', 'time_only', 'joint_only', 'none_flat'],
                   default='time_joint', help='Pooling strategy')
    p.add_argument('--layers_concat', type=str, default='',
                   help='Comma-sep list of layers to concat (e.g. "2,3"). Empty = use --layer only.')
    p.add_argument('--n_clips', type=int, default=400, help='Number of clips to probe')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', type=str, default='eval/results/dift_probe.json')
    return p.parse_args()


def load_pretrained_anytop(model_path, device):
    from utils.fixseed import fixseed
    from utils import dist_util
    from utils.model_util import create_model_and_diffusion_general_skeleton, load_model

    with open(pjoin(os.path.dirname(model_path), 'args.json')) as f:
        args_d = json.load(f)

    class NS:
        def __init__(self, d):
            self.__dict__.update(d)
    args = NS(args_d)

    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    state = torch.load(model_path, map_location='cpu')
    load_model(model, state)
    # Note: AnyTop.to() returns None, so call separately and don't chain
    model.to(device)
    model.eval()
    return model, diffusion, args


def encode_dift_multi(model, diffusion, x, y, t, layers):
    """Run forward diffusion and extract activations at multiple layers.

    Returns dict {layer: [T+1, B, J, D]}
    """
    device = next(model.parameters()).device
    bs = x.shape[0]
    t_vec = torch.full((bs,), t, dtype=torch.long, device=device)

    noise = torch.randn_like(x)
    x_t = diffusion.q_sample(x, t_vec, noise=noise)

    # get_layer_activation in AnyTop only supports a single layer at a time.
    # Run once per requested layer.
    outs = {}
    for layer in layers:
        with torch.no_grad():
            out = model(x_t, t_vec, get_layer_activation=layer, y=y)
        if isinstance(out, tuple):
            _, activations = out
            outs[layer] = activations[layer]
        else:
            outs[layer] = out
    return outs


def pool_features(feat_dict, n_joints, pool_mode):
    """Pool multi-layer features into a single vector.

    feat_dict: {layer: [T+1, B, J, D]}
    Returns: numpy vector
    """
    pooled = []
    for layer in sorted(feat_dict.keys()):
        f = feat_dict[layer]
        if f.dim() == 4:  # [T+1, B, J, D]
            f = f[1:, 0, :n_joints, :]  # [T, J_real, D]
            if pool_mode == 'time_joint':
                v = f.mean(dim=(0, 1))  # [D]
            elif pool_mode == 'time_only':
                # Keep per-joint info, take mean over time
                v = f.mean(dim=0)  # [J_real, D]
                v = v.reshape(-1)   # [J_real*D]
            elif pool_mode == 'joint_only':
                # Keep temporal info
                v = f.mean(dim=1)   # [T, D]
                v = v.reshape(-1)
            else:  # none_flat
                v = f.reshape(-1)
        else:
            v = f.reshape(-1)
        pooled.append(v.cpu().numpy())
    # Concatenate across layers
    # Handle variable-length time_only case by truncating to min J
    if pool_mode in ('time_only', 'none_flat') and len(pooled) > 1:
        min_len = min(v.shape[0] for v in pooled)
        pooled = [v[:min_len] for v in pooled]
    return np.concatenate(pooled, axis=0)


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Load split & labels
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)
    with open('dataset/truebones/zoo/truebones_processed/train_val_split.json') as f:
        split = json.load(f)
    val_files = split['val'][:args.n_clips]
    print(f"Probing {len(val_files)} val clips")

    # Load model
    print(f"Loading {args.model_path}")
    model, diffusion, m_args = load_pretrained_anytop(args.model_path, device)
    print(f"  num_layers={m_args.layers}, latent_dim={m_args.latent_dim}, num_frames={m_args.num_frames}")

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

        # Build the y dict expected by the AnyTop model
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

        # joints_mask: [B, ?, ?, J+1, J+1] float — anytop expects float
        # 1.0 = valid pair, 0.0 = padding pair
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

    # Determine which layers to extract
    if args.layers_concat:
        layers = [int(x) for x in args.layers_concat.split(',')]
    else:
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

    print(f"\nExtracted {n_done} DIFT vectors, {n_failed} failures")
    if n_done < 20:
        print("Too few — aborting probes")
        return

    all_feats = np.array(all_feats)
    print(f"Feature shape: {all_feats.shape}")

    # Run probes
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

    results = {
        'model_path': args.model_path,
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
    print("=" * 50)
    print(f"DIFT PROBE RESULTS (t={args.t_seg}, layer={args.layer})")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print()
    print("Targets: skel<=0.10  motion>=0.50  gap<=0.03  P@5>=0.40")
    relaxed_pass = (results['skel_probe_holdout'] <= 0.10 and
                    results['motion_probe_holdout'] >= 0.50 and
                    results['skel_cos_gap_holdout'] <= 0.03 and
                    results['cross_skel_p5_holdout'] >= 0.40)
    print(f"  PASSES RELAXED TARGETS: {'YES — Stage 1 may be unnecessary!' if relaxed_pass else 'no'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
