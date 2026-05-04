"""DIFT Adapter: activate AnyTop pretrained features for cross-skeleton conditioning.

Pipeline:
  1. Freeze AnyTop pretrained backbone (all_model_...)
  2. Extract DIFT features for all train+val clips ONCE and cache to disk
     - Forward diffusion to t=t_seg, extract layer l_seg activations
     - Pool over time+joint → [D] vector per clip (fast to iterate)
  3. Train a small MLP adapter on cached features
     - Input: DIFT feature [D_dift]
     - Output: z [D_z]
     - Loss: cross-skeleton InfoNCE on projection head
  4. Evaluate adapter output with the same probes as our trained Stage 1

Usage:
    # Step 1: cache features
    conda run -n anytop python -m eval.dift_adapter cache \
        --model_path save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt \
        --t_seg 3 --layer 3

    # Step 2: train adapter
    conda run -n anytop python -m eval.dift_adapter train \
        --cache eval/results/dift_cache_t3_l3.pt \
        --save_dir save/dift_adapter_l3
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Adapter model
# ─────────────────────────────────────────────────────────────────────────────

class DIFTAdapter(nn.Module):
    """Tiny MLP that maps frozen DIFT features → task-aligned z."""
    def __init__(self, d_in, d_hidden=256, d_z=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_z),
        )
        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, d_z),
        )

    def forward(self, x, return_proj=False):
        z = self.backbone(x)
        if return_proj:
            p = self.proj(z)
            return z, p
        return z


# ─────────────────────────────────────────────────────────────────────────────
# Feature caching — extract DIFT once
# ─────────────────────────────────────────────────────────────────────────────

def cache_dift_features(args):
    from utils.fixseed import fixseed
    from utils import dist_util
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Import here so unit tests don't need GPU deps
    from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation
    from model.conditioners import T5Conditioner

    # Load model
    with open(pjoin(os.path.dirname(args.model_path), 'args.json')) as f:
        args_d = json.load(f)
    class NS:
        def __init__(self, d):
            self.__dict__.update(d)
    m_args = NS(args_d)
    print(f"Loading AnyTop backbone: {args.model_path}")
    model, diffusion = create_model_and_diffusion_general_skeleton(m_args)
    state = torch.load(args.model_path, map_location='cpu')
    load_model(model, state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=m_args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device='cuda')

    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)
    with open('dataset/truebones/zoo/truebones_processed/train_val_split.json') as f:
        split = json.load(f)

    # Cache joint-name embeddings per skeleton (they're expensive — T5 is slow)
    print("Caching joint name embeddings per skeleton...")
    name_emb_cache = {}
    for skel, info in cond_dict.items():
        names = info['joints_names']
        name_emb_cache[skel] = t5(t5.tokenize(names)).detach().cpu().numpy()

    n_frames = m_args.num_frames
    max_joints = opt.max_joints

    def extract_one(fname):
        if fname not in label_map:
            return None
        skel = label_map[fname]['skeleton']
        if skel not in cond_dict:
            return None
        info = cond_dict[skel]
        n_joints = info['offsets'].shape[0]
        if n_joints >= max_joints:
            return None

        raw = np.load(pjoin(opt.motion_dir, fname))
        T, J_src, _ = raw.shape
        if T < n_frames:
            pad = np.zeros((n_frames - T, J_src, 13))
            raw = np.concatenate([raw, pad], axis=0)
        else:
            start = max((T - n_frames) // 2, 0)
            raw = raw[start:start + n_frames]

        mean = info['mean']
        std = info['std'] + 1e-6
        norm = np.nan_to_num((raw - mean[None, :]) / std[None, :])

        x_np = np.zeros((n_frames, max_joints, opt.feature_len))
        x_np[:, :n_joints, :] = norm
        x = torch.tensor(x_np).permute(1, 2, 0).float().unsqueeze(0).to(device)

        tpos_raw = info['tpos_first_frame']
        tpos = np.zeros((max_joints, opt.feature_len))
        tpos[:n_joints] = (tpos_raw - mean) / std
        tpos = np.nan_to_num(tpos)
        tpos_t = torch.tensor(tpos).float().unsqueeze(0).to(device)

        names_padded = np.zeros((max_joints, name_emb_cache[skel].shape[1]))
        names_padded[:n_joints] = name_emb_cache[skel]
        names_t = torch.tensor(names_padded).float().unsqueeze(0).to(device)

        gd = create_padded_relation(info['joints_graph_dist'], max_joints, n_joints)
        jr = create_padded_relation(info['joint_relations'], max_joints, n_joints)
        gd_t = torch.tensor(gd).long().unsqueeze(0).to(device)
        jr_t = torch.tensor(jr).long().unsqueeze(0).to(device)

        jmask_5d = torch.zeros(1, 1, 1, max_joints + 1, max_joints + 1, device=device)
        jmask_5d[0, 0, 0, :n_joints + 1, :n_joints + 1] = 1.0

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

        t_vec = torch.full((1,), args.t_seg, dtype=torch.long, device=device)
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t_vec, noise=noise)

        with torch.no_grad():
            out = model(x_t, t_vec, get_layer_activation=args.layer, y=y)
        if isinstance(out, tuple):
            _, activations = out
            feat = activations[args.layer]
        else:
            feat = out

        # feat: [T+1, 1, J, D] — pool as time_joint → [D]
        f = feat[1:, 0, :n_joints, :].mean(dim=(0, 1))  # [D]
        return f.cpu().numpy(), skel, label_map[fname]['coarse_label']

    all_files = sorted(set(split['train'] + split['val']))
    print(f"Extracting DIFT features for {len(all_files)} clips (t={args.t_seg}, layer={args.layer})...")
    features, skels, motions, names = [], [], [], []
    for fname in tqdm(all_files):
        res = extract_one(fname)
        if res is None:
            continue
        f, skel, motion = res
        features.append(f)
        skels.append(skel)
        motions.append(motion)
        names.append(fname)

    features = np.array(features)
    split_ids = np.array(['train' if n in set(split['train']) else 'val' for n in names])

    print(f"Cached {len(features)} features, dim={features.shape[1]}")
    torch.save({
        'features':   features,
        'skels':      skels,
        'motions':    motions,
        'names':      names,
        'split_ids':  split_ids,
        't_seg':      args.t_seg,
        'layer':      args.layer,
    }, args.out)
    print(f"Saved to {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# Adapter training
# ─────────────────────────────────────────────────────────────────────────────

def train_adapter(args):
    from utils.fixseed import fixseed
    fixseed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    feats = cache['features']  # np.ndarray [N, D]
    skels = cache['skels']     # list[str]
    motions = cache['motions'] # list[str]
    split_ids = cache['split_ids']  # ['train'|'val']

    train_idx = np.where(split_ids == 'train')[0]
    val_idx = np.where(split_ids == 'val')[0]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Feat dim: {feats.shape[1]}")

    # Standardize features (important — DIFT features are not normalized)
    train_feats = feats[train_idx]
    feat_mean = train_feats.mean(axis=0, keepdims=True)
    feat_std = train_feats.std(axis=0, keepdims=True) + 1e-6
    feats_norm = (feats - feat_mean) / feat_std

    X = torch.tensor(feats_norm, dtype=torch.float32, device=device)
    X_train = X[train_idx]
    X_val = X[val_idx]

    # Build adapter
    adapter = DIFTAdapter(d_in=feats.shape[1], d_hidden=args.d_hidden, d_z=args.d_z).to(device)
    print(f"Adapter params: {sum(p.numel() for p in adapter.parameters())/1e3:.1f}K")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)

    # Eval helper
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    def evaluate(step):
        adapter.eval()
        with torch.no_grad():
            z_val = adapter(X_val).cpu().numpy()
        N = len(z_val)
        val_skels = [skels[i] for i in val_idx]
        val_motions = [motions[i] for i in val_idx]

        # skel probe
        le_s = LabelEncoder().fit(val_skels)
        y_s = le_s.transform(val_skels)
        skel_acc = 0.0
        if len(set(y_s)) > 1:
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            skel_acc = cross_val_score(clf, z_val, y_s, cv=min(5, len(set(y_s))), scoring='accuracy').mean()

        # motion probe (exclude 'other')
        valid = [i for i, m in enumerate(val_motions) if m != 'other']
        motion_acc = 0.0
        if len(valid) > 20:
            z_v = z_val[valid]
            m_v = [val_motions[i] for i in valid]
            le_m = LabelEncoder().fit(m_v)
            y_m = le_m.transform(m_v)
            if len(set(y_m)) > 2:
                clf2 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
                motion_acc = cross_val_score(clf2, z_v, y_m, cv=min(5, len(set(y_m))), scoring='accuracy').mean()

        # cosine analysis
        z_norm = z_val / (np.linalg.norm(z_val, axis=1, keepdims=True) + 1e-8)
        cos = z_norm @ z_norm.T
        same_s, cross_s = [], []
        for i in range(N):
            for j in range(i + 1, N):
                (same_s if val_skels[i] == val_skels[j] else cross_s).append(cos[i, j])
        skel_gap = (np.mean(same_s) - np.mean(cross_s)) if same_s and cross_s else 0

        # P@5 cross-skel same-label
        hits, total = 0, 0
        for i in range(N):
            if val_motions[i] == 'other':
                continue
            sims = [(j, cos[i, j]) for j in range(N) if j != i and val_skels[j] != val_skels[i]]
            sims.sort(key=lambda x: -x[1])
            top5 = sims[:5]
            hits += sum(1 for j, _ in top5 if val_motions[j] == val_motions[i])
            total += len(top5)
        p5 = hits / max(total, 1)

        adapter.train()
        return {
            'step': step,
            'skel_probe_holdout': float(skel_acc),
            'motion_probe_holdout': float(motion_acc),
            'skel_cos_gap_holdout': float(skel_gap),
            'cross_skel_p5_holdout': float(p5),
        }

    # Contrastive loss
    def supcon_loss(proj, labels, skels_batch, temperature=0.07):
        """Cross-skeleton InfoNCE: positives = same motion label, different skeleton."""
        z = F.normalize(proj, dim=-1)
        sim = (z @ z.T) / temperature
        B = z.shape[0]
        pos_mask = torch.zeros(B, B, dtype=torch.bool, device=z.device)
        for i in range(B):
            for j in range(B):
                if i != j and labels[i] == labels[j] \
                   and skels_batch[i] != skels_batch[j] \
                   and labels[i] not in ('unk', 'other'):
                    pos_mask[i, j] = True
        if not pos_mask.any():
            return torch.tensor(0.0, device=z.device)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        log_sum_exp = torch.logsumexp(sim + torch.where(self_mask, 0.0, -1e9), dim=1)
        loss = 0.0
        count = 0
        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) > 0:
                loss = loss - (sim[i, pos_idx] - log_sum_exp[i]).mean()
                count += 1
        return loss / max(count, 1)

    train_skels = [skels[i] for i in train_idx]
    train_motions = [motions[i] for i in train_idx]

    eval_history = []
    best_composite = -1e9
    best_metrics = None
    best_step = 0

    print(f"Training adapter for {args.num_steps} steps, bs={args.batch_size}")
    rng = np.random.RandomState(args.seed)
    adapter.train()
    for step in tqdm(range(args.num_steps)):
        idx = rng.choice(len(X_train), size=args.batch_size, replace=False)
        x_b = X_train[idx]
        batch_motions = [train_motions[i] for i in idx]
        batch_skels = [train_skels[i] for i in idx]

        z, proj = adapter(x_b, return_proj=True)
        loss = supcon_loss(proj, batch_motions, batch_skels, temperature=args.temperature)

        optimizer.zero_grad()
        if loss.item() > 0:
            loss.backward()
            optimizer.step()

        if (step + 1) % args.eval_interval == 0:
            m = evaluate(step + 1)
            eval_history.append(m)
            comp = m['motion_probe_holdout'] + m['cross_skel_p5_holdout'] \
                 - 3 * m['skel_cos_gap_holdout'] - m['skel_probe_holdout']
            relaxed_pass = (m['skel_probe_holdout'] <= 0.10 and
                           m['motion_probe_holdout'] >= 0.50 and
                           m['skel_cos_gap_holdout'] <= 0.03 and
                           m['cross_skel_p5_holdout'] >= 0.40)
            flag = 'RELAXED-PASS' if relaxed_pass else ''
            tqdm.write(f"[{step+1}] loss={loss.item():.3f} "
                       f"skel={m['skel_probe_holdout']:.3f} "
                       f"mo={m['motion_probe_holdout']:.3f} "
                       f"gap={m['skel_cos_gap_holdout']:.4f} "
                       f"P@5={m['cross_skel_p5_holdout']:.3f} "
                       f"comp={comp:.3f} {flag}")
            if comp > best_composite:
                best_composite = comp
                best_metrics = m
                best_step = step + 1
                torch.save({
                    'adapter': adapter.state_dict(),
                    'feat_mean': feat_mean,
                    'feat_std': feat_std,
                    'd_in': feats.shape[1],
                    'd_hidden': args.d_hidden,
                    'd_z': args.d_z,
                    't_seg': cache['t_seg'],
                    'layer': cache['layer'],
                    'metrics': m,
                }, pjoin(args.save_dir, 'adapter_best.pt'))

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        'adapter': adapter.state_dict(),
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'eval_history': eval_history,
        'best_metrics': best_metrics,
        'best_step': best_step,
        'args': vars(args),
    }, pjoin(args.save_dir, 'adapter_final.pt'))
    with open(pjoin(args.save_dir, 'eval_history.json'), 'w') as f:
        json.dump({'eval_history': eval_history, 'best_metrics': best_metrics, 'best_step': best_step}, f, indent=2)

    print()
    print("=" * 60)
    print(f"BEST CHECKPOINT @ step {best_step}")
    print(f"  skel_probe: {best_metrics['skel_probe_holdout']:.4f}")
    print(f"  motion_probe: {best_metrics['motion_probe_holdout']:.4f}")
    print(f"  skel_cos_gap: {best_metrics['skel_cos_gap_holdout']:.4f}")
    print(f"  cross_skel_P@5: {best_metrics['cross_skel_p5_holdout']:.4f}")
    print()
    print("Targets (relaxed): skel<=0.10 mo>=0.50 gap<=0.03 P@5>=0.40")
    passes = (best_metrics['skel_probe_holdout'] <= 0.10 and
              best_metrics['motion_probe_holdout'] >= 0.50 and
              best_metrics['skel_cos_gap_holdout'] <= 0.03 and
              best_metrics['cross_skel_p5_holdout'] >= 0.40)
    print(f"PASSES: {'YES — STAGE 1 COMPLETE!' if passes else 'no'}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    pc = sub.add_parser('cache')
    pc.add_argument('--model_path', required=True)
    pc.add_argument('--t_seg', type=int, default=3)
    pc.add_argument('--layer', type=int, default=3)
    pc.add_argument('--seed', type=int, default=10)
    pc.add_argument('--device', type=int, default=0)
    pc.add_argument('--out', type=str, default='eval/results/dift_cache_t3_l3.pt')

    pt = sub.add_parser('train')
    pt.add_argument('--cache', required=True)
    pt.add_argument('--save_dir', required=True)
    pt.add_argument('--d_hidden', type=int, default=256)
    pt.add_argument('--d_z', type=int, default=64)
    pt.add_argument('--lr', type=float, default=3e-4)
    pt.add_argument('--batch_size', type=int, default=32)
    pt.add_argument('--num_steps', type=int, default=10000)
    pt.add_argument('--eval_interval', type=int, default=500)
    pt.add_argument('--temperature', type=float, default=0.07)
    pt.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()
    if args.cmd == 'cache':
        cache_dift_features(args)
    elif args.cmd == 'train':
        os.makedirs(args.save_dir, exist_ok=True)
        train_adapter(args)


if __name__ == '__main__':
    main()
