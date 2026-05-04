"""Stage 1: Train motion auto-encoder variants and evaluate z quality.

Variants:
  AE-raw:          raw 13D input, no constraints
  AE-norm:         canonicalized input, continuous bottleneck
  AE-norm-supcon:  + supervised contrastive on motion labels (cross-skeleton positives)
  AE-norm-cycle:   + cross-skeleton latent cycle consistency

Usage:
    conda run -n anytop python -m train.train_autoencoder \
        --variant norm-cycle --num_steps 50000 --batch_size 16 \
        --save_dir save/S1_norm_cycle
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict

from utils.fixseed import fixseed
from utils import dist_util
from model.motion_autoencoder import MotionAutoEncoder
from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
from data_loaders.truebones.truebones_utils.get_opt import get_opt


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def masked_mse(pred, target, mask):
    """MSE on real joints only. mask: [B, J] bool."""
    mask_4d = mask.unsqueeze(-1).unsqueeze(-1).expand_as(target).float()
    return (((pred - target) ** 2) * mask_4d).sum() / mask_4d.sum().clamp(min=1)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning (standard practice from SimCLR)."""
    def __init__(self, d_in, d_proj=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_proj), nn.GELU(), nn.Linear(d_proj, d_proj))

    def forward(self, z_pool):
        return self.net(z_pool)


def vicreg_var_cov_loss(z_pool, gamma=0.5, eps=1e-4):
    """VICReg variance + covariance regularization on encoder pooled latent.

    z_pool: [B, D] — mean-pooled encoder z
    Returns (var_loss, cov_loss)

    Variance: each dim's std should be >= gamma (prevents collapse to constant)
    Covariance: off-diagonal of normalized cov should be ~0 (decorrelate dims)
    """
    B, D = z_pool.shape
    z_centered = z_pool - z_pool.mean(dim=0, keepdim=True)

    # Variance term — push each dim toward std >= gamma
    std = torch.sqrt(z_centered.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()

    # Covariance term — penalize off-diagonal of cov matrix
    if B > 1:
        cov = (z_centered.T @ z_centered) / (B - 1)  # [D, D]
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / D
    else:
        cov_loss = torch.tensor(0.0, device=z_pool.device)

    return var_loss, cov_loss


@torch.no_grad()
def latent_diagnostics(z_pool):
    """Diagnostics for verifying collapse prevention.

    Returns dict with: effective_rank, std_mean, std_min, std_p10
    """
    z_centered = z_pool - z_pool.mean(dim=0, keepdim=True)
    std = torch.sqrt(z_centered.var(dim=0) + 1e-6).cpu().numpy()
    cov = (z_centered.T @ z_centered) / max(z_pool.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
    eigvals = eigvals.clamp(min=1e-8).cpu().numpy()
    p = eigvals / eigvals.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return {
        'std_mean': float(std.mean()),
        'std_min': float(std.min()),
        'std_p10': float(np.percentile(std, 10)),
        'effective_rank': eff_rank,
    }


def skel_mean_invariance_loss(z_seq, skel_labels):
    """Penalize per-skeleton mean drift — directly minimizes skel_cos_gap.

    For each skeleton with >=2 samples in batch, compute mean z.
    Loss = mean ||per_skel_mean||^2, encouraging zero per-skeleton bias.
    """
    z_pool = z_seq.mean(dim=(1, 2))  # [B, d_z]
    z_norm = F.normalize(z_pool, dim=-1)
    B = z_pool.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_seq.device)

    # Group samples by skeleton
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(skel_labels):
        groups[s].append(i)

    losses = []
    for s, idx in groups.items():
        if len(idx) >= 2:
            mean_z = z_norm[idx].mean(dim=0)
            losses.append((mean_z ** 2).sum())

    if not losses:
        return torch.tensor(0.0, device=z_seq.device)
    return torch.stack(losses).mean()


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class AdversarialSkelClassifier(nn.Module):
    """Skeleton classifier with gradient reversal — pushes z toward skeleton invariance."""
    def __init__(self, d_in, n_skel, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_skel))

    def forward(self, z_pool, lambda_=1.0):
        z_rev = grad_reverse(z_pool, lambda_)
        return self.net(z_rev)


def cross_skeleton_infonce(proj_head, z_seq, motion_labels, skel_labels, temperature=0.07):
    """Cross-skeleton InfoNCE on projected z with per-anchor logsumexp.

    Uses projection head (standard SupCon practice) rather than raw z.
    Positives = same COARSE motion label on DIFFERENT skeletons.
    """
    B = z_seq.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_seq.device)

    z_pool = z_seq.mean(dim=(1, 2))  # [B, d_z]
    z_proj = proj_head(z_pool)        # [B, d_proj]
    z_norm = F.normalize(z_proj, dim=-1)
    sim = z_norm @ z_norm.T / temperature

    # Positives: same motion label AND different skeleton
    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=z_seq.device)
    for i in range(B):
        for j in range(B):
            if i != j and motion_labels[i] == motion_labels[j] \
               and skel_labels[i] != skel_labels[j] \
               and motion_labels[i] not in ('unk', 'other'):
                pos_mask[i, j] = True

    if not pos_mask.any():
        return torch.tensor(0.0, device=z_seq.device)

    # Per-anchor SupCon loss
    self_mask = ~torch.eye(B, dtype=torch.bool, device=z_seq.device)
    log_sum_exp = torch.logsumexp(sim + torch.where(self_mask, 0.0, -1e9), dim=1)
    loss = 0.0
    count = 0
    for i in range(B):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        if len(pos_idx) > 0:
            loss = loss - (sim[i, pos_idx] - log_sum_exp[i]).mean()
            count += 1
    return loss / max(count, 1)


def cycle_consistency_loss(model, z, all_skel_data, device):
    """Cross-skeleton latent cycle: z → decode(T_random) → re-encode → z'.

    Penalize ||z - z'|| to force skeleton-invariance.
    all_skel_data: dict of {skel_name: (offsets [J,3], mask [J] bool)}
    """
    B = z.shape[0]
    if not all_skel_data or B == 0:
        return torch.tensor(0.0, device=device)

    skel_names = list(all_skel_data.keys())
    T = z.shape[1] * 4  # approximate original T from T/4

    # Pick random target skeletons (one per sample)
    losses = []
    max_joints = next(iter(all_skel_data.values()))[0].shape[0]

    for b in range(B):
        tgt_name = random.choice(skel_names)
        tgt_offsets, tgt_mask = all_skel_data[tgt_name]
        tgt_off = tgt_offsets.unsqueeze(0).to(device)   # [1, J, 3]
        tgt_msk = tgt_mask.unsqueeze(0).to(device)      # [1, J]

        # Decode z[b] on random target skeleton
        z_b = z[b:b+1]  # [1, T/4, K, d_z]
        x_t = model.decode(z_b, tgt_off, tgt_msk, T)   # [1, J, 13, T]

        # Re-encode on target skeleton
        z_prime = model.encode(x_t, tgt_off, tgt_msk)   # [1, T/4, K, d_z]

        losses.append(F.mse_loss(z_b, z_prime))

    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation probes
# ─────────────────────────────────────────────────────────────────────────────

def _parse_source_skel(name):
    """Extract SOURCE skeleton from source_name (e.g., 'Horse___SlowWalk_432.npy' → 'Horse')."""
    return name.split('___')[0].split('__')[0] if '___' in name else name.split('__')[0]


def _parse_motion_label(name):
    """Extract motion label (e.g., 'Horse___SlowWalk_432.npy' → 'SlowWalk')."""
    parts = name.replace('.npy', '').split('___')
    return parts[1].split('_')[0] if len(parts) > 1 else 'unk'


@torch.no_grad()
def evaluate_z_holdout(model, val_files, label_map, device, opt, n_frames=120):
    """Evaluate z quality on a FIXED held-out validation set.

    Reads val_files directly from disk with deterministic crops (no random).
    Returns probe accuracies on cross-validated logistic regression.
    """
    import numpy as np
    from os.path import join as pjoin
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    motion_dir = opt.motion_dir
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    model.eval()

    all_z, all_skels, all_motions = [], [], []
    for fname in val_files:
        if fname not in label_map:
            continue
        skel = label_map[fname]['skeleton']
        if skel not in cond_dict:
            continue

        raw = np.load(pjoin(motion_dir, fname))
        T, J_src, _ = raw.shape
        if T < n_frames:
            pad = np.zeros((n_frames - T, J_src, 13))
            raw_p = np.concatenate([raw, pad], axis=0)
        else:
            # Deterministic center crop
            start = (T - n_frames) // 2
            raw_p = raw[start:start + n_frames]

        mean = cond_dict[skel]['mean']
        std = cond_dict[skel]['std'] + 1e-6
        norm = np.nan_to_num((raw_p - mean[None, :]) / std[None, :])

        sm = np.zeros((n_frames, opt.max_joints, 13))
        sm[:, :J_src, :] = norm
        so = np.zeros((opt.max_joints, 3))
        so[:J_src, :] = cond_dict[skel]['offsets']

        sm_t = torch.tensor(sm).permute(1, 2, 0).float().unsqueeze(0).to(device)
        so_t = torch.tensor(so).float().unsqueeze(0).to(device)
        msk = torch.zeros(1, opt.max_joints, dtype=torch.bool, device=device)
        msk[0, :J_src] = True

        z = model.encode(sm_t, so_t, msk)
        all_z.append(z[0].cpu().numpy().reshape(-1))
        all_skels.append(skel)
        all_motions.append(label_map[fname]['coarse_label'])

    model.train()
    if len(all_z) < 20:
        return {}

    all_z = np.array(all_z)
    N = len(all_z)

    # Skeleton probe (5-fold CV)
    le_s = LabelEncoder().fit(all_skels)
    y_s = le_s.transform(all_skels)
    if len(set(y_s)) > 1:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        skel_acc = cross_val_score(clf, all_z, y_s, cv=min(5, len(set(y_s))), scoring='accuracy').mean()
    else:
        skel_acc = 0.0

    # Motion probe (excluding 'other' to focus on real categories)
    valid = [i for i, m in enumerate(all_motions) if m != 'other']
    if len(valid) > 20:
        z_v = all_z[valid]
        m_v = [all_motions[i] for i in valid]
        le_m = LabelEncoder().fit(m_v)
        y_m = le_m.transform(m_v)
        if len(set(y_m)) > 2:
            clf2 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            motion_acc = cross_val_score(clf2, z_v, y_m, cv=min(5, len(set(y_m))), scoring='accuracy').mean()
        else:
            motion_acc = 0.0
    else:
        motion_acc = 0.0

    # Cosine analyses
    z_norm = all_z / (np.linalg.norm(all_z, axis=1, keepdims=True) + 1e-8)
    cos = z_norm @ z_norm.T
    same_s, cross_s = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (same_s if all_skels[i] == all_skels[j] else cross_s).append(cos[i, j])
    skel_gap = (np.mean(same_s) - np.mean(cross_s)) if same_s and cross_s else 0
    mean_cos = float(np.mean(cos[~np.eye(N, dtype=bool)]))

    # Cross-skeleton same-label P@5 (excluding 'other')
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

    # Latent diagnostics — eff rank and std stats on pooled z
    diag = {}
    try:
        # Recompute z_pool from all_z which is flattened [N, T/4*K*d_z]
        # Reshape back: assume model has self.d_z attribute
        flat_dim = all_z.shape[1]
        # Approximate: use variance over samples as a proxy
        z_t = torch.tensor(all_z)
        z_centered = z_t - z_t.mean(dim=0, keepdim=True)
        std = torch.sqrt(z_centered.var(dim=0) + 1e-6).numpy()
        # Effective rank via cov spectrum (subsample to make it tractable)
        sub_dim = min(flat_dim, 256)
        idx = np.random.RandomState(0).permutation(flat_dim)[:sub_dim]
        z_sub = z_centered[:, idx].numpy()
        cov = (z_sub.T @ z_sub) / max(N - 1, 1)
        eigvals = np.linalg.eigvalsh(cov + 1e-6 * np.eye(sub_dim))
        eigvals = np.clip(eigvals, 1e-8, None)
        p = eigvals / eigvals.sum()
        eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        diag = {
            'std_mean': float(std.mean()),
            'std_min': float(std.min()),
            'std_p10': float(np.percentile(std, 10)),
            'effective_rank': eff_rank,
        }
    except Exception as e:
        diag = {}

    return {
        'skel_probe_holdout': float(skel_acc),
        'motion_probe_holdout': float(motion_acc),
        'z_mean_cos_holdout': mean_cos,
        'skel_cos_gap_holdout': float(skel_gap),
        'cross_skel_p5_holdout': float(p5),
        'n_val': N,
        **diag,
    }


@torch.no_grad()
def evaluate_z(model, data_loader, device, n_batches=50):
    """Evaluate z quality with CORRECT source labels and train/test split."""
    model.eval()
    all_z, all_skels, all_motions = [], [], []

    for i, (_, cond) in enumerate(data_loader):
        if i >= n_batches:
            break
        y = cond['y']
        motion = y['source_motion'].to(device)
        offsets = y['source_offsets'].to(device)
        mask = y['source_joints_mask'].to(device)

        z = model.encode(motion, offsets, mask)
        # Use FULL sequence z flattened, not just mean-pooled
        z_flat = z.reshape(z.shape[0], -1).cpu()  # [B, T/4*K*d_z]
        all_z.append(z_flat)

        # Use clean labels from precomputed map, fallback to filename parsing
        for name in y.get('source_name', ['unk'] * motion.shape[0]):
            fname = name if name.endswith('.npy') else name + '.npy'
            if hasattr(evaluate_z, '_label_map') and fname in evaluate_z._label_map:
                all_skels.append(evaluate_z._label_map[fname]['skeleton'])
                all_motions.append(evaluate_z._label_map[fname]['coarse_label'])
            else:
                all_skels.append(_parse_source_skel(name))
                all_motions.append(_parse_motion_label(name))

    all_z = torch.cat(all_z, dim=0).numpy()
    N = len(all_z)
    model.train()

    if N < 20:
        return {}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    # 1. Skeleton-ID probe — cross-validated (NOT train-on-train)
    le_skel = LabelEncoder().fit(all_skels)
    y_skel = le_skel.transform(all_skels)
    if len(set(y_skel)) > 1:
        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        skel_acc = cross_val_score(clf, all_z, y_skel, cv=min(5, len(set(y_skel))),
                                   scoring='accuracy').mean()
    else:
        skel_acc = 0.0

    # 2. Motion-type probe — cross-validated
    le_mot = LabelEncoder().fit(all_motions)
    y_mot = le_mot.transform(all_motions)
    if len(set(y_mot)) > 2:
        clf2 = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        motion_acc = cross_val_score(clf2, all_z, y_mot, cv=min(5, len(set(y_mot))),
                                     scoring='accuracy').mean()
    else:
        motion_acc = 0.0

    # 3. Z diversity: mean pairwise cosine (on pooled z for efficiency)
    z_pool = all_z.reshape(N, -1)
    z_norm = z_pool / (np.linalg.norm(z_pool, axis=1, keepdims=True) + 1e-8)
    sub = min(N, 300)
    cos = z_norm[:sub] @ z_norm[:sub].T
    mask_diag = ~np.eye(sub, dtype=bool)
    mean_cos = cos[mask_diag].mean()

    # 4. Same-skel vs cross-skel cosine gap
    same_s, cross_s = [], []
    for i in range(sub):
        for j in range(i + 1, sub):
            (same_s if all_skels[i] == all_skels[j] else cross_s).append(cos[i, j])
    skel_gap = (np.mean(same_s) - np.mean(cross_s)) if same_s and cross_s else 0

    # 5. Cross-skeleton same-label precision@5
    hits, total = 0, 0
    for i in range(sub):
        sims = [(j, cos[i, j]) for j in range(sub) if j != i and all_skels[j] != all_skels[i]]
        sims.sort(key=lambda x: -x[1])
        top5 = sims[:5]
        if top5 and all_motions[i] != 'unk':
            hits += sum(1 for j, _ in top5 if all_motions[j] == all_motions[i])
            total += len(top5)
    cross_precision = hits / max(total, 1)

    return {
        'skel_probe_cv': skel_acc,
        'motion_probe_cv': motion_acc,
        'z_mean_cos': float(mean_cos),
        'skel_cos_gap': float(skel_gap),
        'cross_skel_p5': cross_precision,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument('--variant', choices=['raw', 'norm', 'norm-supcon', 'norm-cycle'],
                        default='norm')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_z', type=int, default=32)
    parser.add_argument('--num_queries', type=int, default=4)
    parser.add_argument('--lambda_supcon', type=float, default=0.1)
    parser.add_argument('--lambda_cycle', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=0.0,
                        help='Weight for gradient-reversal adversarial skel classifier')
    parser.add_argument('--lambda_skelmean', type=float, default=0.0,
                        help='Weight for per-skeleton mean invariance loss')
    parser.add_argument('--lambda_var', type=float, default=0.0,
                        help='VICReg variance loss weight on z_pool')
    parser.add_argument('--lambda_cov', type=float, default=0.0,
                        help='VICReg covariance loss weight on z_pool')
    parser.add_argument('--vicreg_gamma', type=float, default=0.5,
                        help='VICReg variance threshold')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from this Stage 1 checkpoint')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    opt = get_opt(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Stage 1 variant: {args.variant}")
    canonical = args.variant != 'raw'

    # Load train/val split — TRAIN LOADER MUST ONLY SEE TRAIN FILES
    train_files = None
    split_path = os.path.join(opt.data_root, 'truebones/zoo/truebones_processed/train_val_split.json')
    if not os.path.exists(split_path):
        # Try alternate path relative to motion_dir
        split_path = os.path.join(os.path.dirname(opt.motion_dir), 'train_val_split.json')
    if os.path.exists(split_path):
        with open(split_path) as f:
            split_data = json.load(f)
        train_files = split_data['train']
        print(f"Restricting train loader to {len(train_files)} files")

    print("Creating data loader...")
    data = get_dataset_loader_conditioned(
        batch_size=args.batch_size, num_frames=120,
        temporal_window=31, t5_name='t5-base',
        balanced=False, objects_subset='all',
        split_files=train_files)

    print("Creating auto-encoder...")
    model = MotionAutoEncoder(
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        d_model=args.d_model, d_z=args.d_z,
        num_queries=args.num_queries, canonical=canonical,
    ).to(dist_util.dev())

    if args.resume_from:
        print(f"Resuming model weights from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

    # Projection head for InfoNCE (trained but NOT part of the encoder)
    proj_head = None
    if 'supcon' in args.variant:
        proj_head = ProjectionHead(args.d_z, d_proj=64).to(dist_util.dev())

    # Load precomputed clean motion labels + train/val split
    label_map = {}
    label_path = os.path.join(os.path.dirname(opt.motion_dir), 'motion_labels.json')
    if os.path.exists(label_path):
        with open(label_path) as f:
            label_map = json.load(f)
        print(f"Loaded {len(label_map)} clean motion labels from {label_path}")

    val_files = []
    split_path = os.path.join(os.path.dirname(opt.motion_dir), 'train_val_split.json')
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        val_files = split['val']
        print(f"Loaded held-out val set: {len(val_files)} files")

    # Adversarial skeleton classifier (gradient reversal)
    adv_clf = None
    skel_to_idx = {}
    if args.lambda_adv > 0 and label_map:
        all_skels_list = sorted(set(v['skeleton'] for v in label_map.values()))
        skel_to_idx = {s: i for i, s in enumerate(all_skels_list)}
        adv_clf = AdversarialSkelClassifier(args.d_z, len(all_skels_list)).to(dist_util.dev())
        print(f"Adversarial skel classifier: {len(all_skels_list)} classes, lambda={args.lambda_adv}")

    all_params = list(model.parameters())
    if proj_head: all_params += list(proj_head.parameters())
    if adv_clf: all_params += list(adv_clf.parameters())
    print(f"Params: {sum(p.numel() for p in all_params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps)

    # Inject label_map into evaluate_z for clean labels
    evaluate_z._label_map = label_map

    data_iter = iter(data)
    log_losses = defaultdict(float)
    eval_history = []

    print(f"Training {args.num_steps} steps...")
    for step in tqdm(range(args.num_steps)):
        try:
            _, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(data)
            _, cond = next(data_iter)

        y = cond['y']
        motion = y['source_motion'].to(dist_util.dev())
        offsets = y['source_offsets'].to(dist_util.dev())
        mask = y['source_joints_mask'].to(dist_util.dev())

        recon, z = model(motion, offsets, mask)

        # Reconstruction loss
        loss_recon = masked_mse(recon, motion, mask)
        loss = loss_recon
        log_losses['recon'] += loss_recon.item()

        # Cross-skeleton InfoNCE with projection head and clean labels
        if 'supcon' in args.variant and proj_head is not None:
            source_names = y.get('source_name', ['unk'] * motion.shape[0])
            motion_labels = []
            skel_labels = []
            for name in source_names:
                fname = name if name.endswith('.npy') else name + '.npy'
                if fname in label_map:
                    motion_labels.append(label_map[fname]['coarse_label'])
                    skel_labels.append(label_map[fname]['skeleton'])
                else:
                    motion_labels.append(_parse_motion_label(name))
                    skel_labels.append(_parse_source_skel(name))
            loss_sc = cross_skeleton_infonce(proj_head, z, motion_labels, skel_labels)
            loss = loss + args.lambda_supcon * loss_sc
            log_losses['infonce'] += loss_sc.item()

        # VICReg variance + covariance regularization on encoder pooled latent
        if args.lambda_var > 0 or args.lambda_cov > 0:
            z_pool_enc = z.mean(dim=(1, 2))  # [B, d_z]
            var_l, cov_l = vicreg_var_cov_loss(z_pool_enc, gamma=args.vicreg_gamma)
            if args.lambda_var > 0:
                loss = loss + args.lambda_var * var_l
                log_losses['var'] += var_l.item()
            if args.lambda_cov > 0:
                loss = loss + args.lambda_cov * cov_l
                log_losses['cov'] += cov_l.item()

        # Skel-mean invariance loss (directly minimizes per-skeleton bias)
        if args.lambda_skelmean > 0:
            source_names = y.get('source_name', ['unk'] * motion.shape[0])
            skel_labels_sm = []
            for name in source_names:
                fname = name if name.endswith('.npy') else name + '.npy'
                if fname in label_map:
                    skel_labels_sm.append(label_map[fname]['skeleton'])
                else:
                    skel_labels_sm.append(_parse_source_skel(name))
            loss_sm = skel_mean_invariance_loss(z, skel_labels_sm)
            loss = loss + args.lambda_skelmean * loss_sm
            log_losses['skelmean'] += loss_sm.item()

        # Adversarial skeleton classifier (gradient reversal)
        if adv_clf is not None and args.lambda_adv > 0:
            source_names = y.get('source_name', ['unk'] * motion.shape[0])
            skel_idx = []
            valid_b = []
            for bi, name in enumerate(source_names):
                fname = name if name.endswith('.npy') else name + '.npy'
                if fname in label_map and label_map[fname]['skeleton'] in skel_to_idx:
                    skel_idx.append(skel_to_idx[label_map[fname]['skeleton']])
                    valid_b.append(bi)
            if len(valid_b) > 0:
                z_pool = z.mean(dim=(1, 2))[valid_b]  # [B', d_z]
                skel_targets = torch.tensor(skel_idx, device=dist_util.dev())
                logits = adv_clf(z_pool, lambda_=args.lambda_adv)
                loss_adv = F.cross_entropy(logits, skel_targets)
                loss = loss + loss_adv
                log_losses['adv'] += loss_adv.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        log_losses['total'] += loss.item()

        # Log
        if (step + 1) % args.log_interval == 0:
            msg = f"[{step+1}]"
            for k, v in log_losses.items():
                msg += f" {k}={v/args.log_interval:.4f}"
            tqdm.write(msg)
            log_losses = defaultdict(float)

        # Evaluate z quality (train-stream + held-out)
        if (step + 1) % args.eval_interval == 0:
            metrics = evaluate_z(model, data, dist_util.dev(), n_batches=30)
            if val_files:
                holdout = evaluate_z_holdout(model, val_files, label_map, dist_util.dev(), opt)
                metrics.update(holdout)
            eval_history.append({'step': step + 1, **metrics})
            tqdm.write(f"  TRAIN: skel={metrics.get('skel_probe_cv',0):.3f} "
                        f"mot={metrics.get('motion_probe_cv',0):.3f} "
                        f"P@5={metrics.get('cross_skel_p5',0):.3f}")
            if val_files:
                tqdm.write(f"  HOLDOUT: skel={metrics.get('skel_probe_holdout',0):.3f} "
                            f"mot={metrics.get('motion_probe_holdout',0):.3f} "
                            f"gap={metrics.get('skel_cos_gap_holdout',0):.4f} "
                            f"P@5={metrics.get('cross_skel_p5_holdout',0):.3f}")

        # Save
        if (step + 1) % args.save_interval == 0:
            torch.save({
                'model': model.state_dict(),
                'encoder_state': {k: v for k, v in model.state_dict().items()
                                  if k.startswith(('feat_extract.', 'attn_pool.', 'temporal_down.', 'to_latent.'))},
                'step': step + 1,
                'eval_history': eval_history,
            }, os.path.join(args.save_dir, f'model{step+1:09d}.pt'))

    # Final eval + save
    final_metrics = evaluate_z(model, data, dist_util.dev(), n_batches=50)
    if val_files:
        final_metrics.update(evaluate_z_holdout(model, val_files, label_map, dist_util.dev(), opt))
    eval_history.append({'step': args.num_steps, **final_metrics, 'final': True})

    torch.save({
        'model': model.state_dict(),
        'encoder_state': {k: v for k, v in model.state_dict().items()
                          if k.startswith(('feat_extract.', 'attn_pool.', 'temporal_down.', 'to_latent.'))},
        'eval_history': eval_history,
    }, os.path.join(args.save_dir, 'final.pt'))

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {args.variant}")
    print(f"{'='*60}")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved to {args.save_dir}/final.pt")

    with open(os.path.join(args.save_dir, 'eval_history.json'), 'w') as f:
        json.dump(eval_history, f, indent=2)


if __name__ == '__main__':
    main()
