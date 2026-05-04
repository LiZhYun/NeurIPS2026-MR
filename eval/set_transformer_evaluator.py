"""Independent Set Transformer Evaluator for motion-conditioned generation.

Architecture:
  Root branch  : velocity, height, yaw rate  → 1D temporal conv → 64D root embedding
  Joint branch : per-joint DeepSets (root-relative position + velocity, body-scale
                 normalized) → Set Transformer → 64D joint embedding
  Fusion       : cat(root, joint) → 2-layer temporal transformer → 128D sequence
  Pooling      : mean over time → 128D motion embedding

Trained with view-consistency on Truebones motions, then frozen for evaluation.

Usage:
  # Train:
    python eval/set_transformer_evaluator.py --train --data_root <path>
      --save_path eval/checkpoints/st_evaluator.pt

  # Evaluate retrieval:
    python eval/set_transformer_evaluator.py --eval --checkpoint <path>
      --source_npy <...> --target_npy <...>
"""

import argparse
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class RootFeatureExtractor(nn.Module):
    """Extract root motion features: velocity, height, and yaw rate.

    Input : x_root  [B, 13, T]   (root joint features from motion tensor)
    Output: [B, 64, T]
    """
    def __init__(self, d_out=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, d_out, kernel_size=3, padding=1),
            nn.GELU(),
        )

    @staticmethod
    def _root_features(x_root):
        """x_root: [B, 13, T] → [B, 4, T] (root_x, root_z, height, yaw rate).

        Note: features [0:2] are root XZ position (not velocity). The heading
        direction is derived from finite differences on these positions.
        """
        pos   = x_root[:, :3, :]    # root position  [B, 3, T]
        xz    = pos[:, :2, :]       # xz position    [B, 2, T]
        height = pos[:, 2:3, :]     # height         [B, 1, T]

        # Heading direction from position trajectory
        px = xz[:, 0:1, :]
        pz = xz[:, 1:2, :]
        yaw = torch.atan2(pz, px + 1e-8)                         # [B, 1, T]
        dyaw = torch.cat([yaw[:, :, 1:] - yaw[:, :, :-1],
                          torch.zeros_like(yaw[:, :, :1])], dim=-1)

        return torch.cat([xz, height, dyaw], dim=1)              # [B, 4, T]

    def forward(self, x_root):
        feats = self._root_features(x_root)                       # [B, 4, T]
        # re-declare proj to accept 4 → d_out
        return self.proj(feats)                                    # [B, d_out, T]


class DeepSetJoint(nn.Module):
    """DeepSets encoder for a single joint: φ per point → pooling → ρ.

    Input : [B*T, D_in]  — root-relative position+velocity (6D)
    Output: [B*T, d_out]
    """
    def __init__(self, d_in=6, d_hidden=64, d_out=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_hidden, d_out), nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.rho(self.phi(x))


class SetAttentionBlock(nn.Module):
    """One Set Attention Block (Inducing Point Self-Attention)."""
    def __init__(self, d_model, num_heads, num_inducing=16, dropout=0.1):
        super().__init__()
        self.attn_q2x = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_x2q = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # x: [B, J, d_model]
        B = x.shape[0]
        I = self.inducing.expand(B, -1, -1)
        # Inducing queries attend over joint tokens (mask out padding joints)
        h, _ = self.attn_q2x(I, x, x, key_padding_mask=key_padding_mask)
        h = self.norm(I + h)
        # Joint tokens attend over inducing points (no padding — all inducing are valid)
        out, _ = self.attn_x2q(x, h, h)
        out = self.norm(x + out)
        out = self.norm2(out + self.ff(out))
        return out


class JointBranchEncoder(nn.Module):
    """Encode all joints per frame via DeepSets + Set Transformer → joint summary.

    Input:
      x_joints: [B, J, 13, T]  — all joint features
      joints_mask: [B, J]       — True = real joint

    Output: [B, d_out, T]
    """
    def __init__(self, d_joint=64, d_out=64, num_heads=4, num_inducing=16):
        super().__init__()
        self.deep_set = DeepSetJoint(d_in=6, d_hidden=d_joint, d_out=d_joint)
        self.sab1 = SetAttentionBlock(d_joint, num_heads, num_inducing)
        self.sab2 = SetAttentionBlock(d_joint, num_heads, num_inducing)
        self.pool = nn.Linear(d_joint, d_out)   # pooled output

    @staticmethod
    def _joint_features(x_joints, body_scale):
        """Extract root-relative position + velocity, body-scale normalized.

        x_joints:   [B, J, 13, T]  (root is dim 0)
        body_scale: [B, 1, 1, 1]   (normalizer)
        Returns:    [B, J, 6, T]
        """
        root_pos = x_joints[:, 0:1, :3, :]               # [B,1,3,T]
        rel_pos  = (x_joints[:, :, :3, :] - root_pos) / (body_scale + 1e-6)
        vel      = x_joints[:, :, 9:12, :] / (body_scale + 1e-6)   # velocity features
        return torch.cat([rel_pos, vel], dim=2)           # [B, J, 6, T]

    def forward(self, x_joints, joints_mask):
        B, J, _, T = x_joints.shape

        # Body scale: mean limb length as normalizer (per sample)
        limb_len = x_joints[:, :, :3, 0].norm(dim=-1).mean(dim=-1, keepdim=True)  # [B,1]
        body_scale = limb_len[:, :, None, None]                                     # [B,1,1,1]

        feats = self._joint_features(x_joints, body_scale)   # [B, J, 6, T]

        # Process each frame independently
        # → [B*T, J, 6] → DeepSets → [B*T, J, d_joint]
        feats = feats.permute(0, 3, 1, 2).reshape(B * T, J, 6)  # [B*T, J, 6]
        flat  = feats.reshape(B * T * J, 6)
        h     = self.deep_set(flat).view(B * T, J, -1)          # [B*T, J, d_joint]

        key_mask = (~joints_mask).unsqueeze(0).expand(T, -1, -1).reshape(B * T, J)  # True=pad
        h = self.sab1(h, key_padding_mask=key_mask)
        h = self.sab2(h, key_padding_mask=key_mask)

        # Pool over real joints (masked mean)
        real = joints_mask.unsqueeze(0).expand(T, -1, -1).reshape(B * T, J, 1).float()
        pooled = (h * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)   # [B*T, d_joint]
        out    = self.pool(pooled).view(B, T, -1).permute(0, 2, 1)      # [B, d_out, T]
        return out


class TemporalEncoder(nn.Module):
    """2-layer temporal transformer over fused root+joint representation → 128D embedding."""
    def __init__(self, d_in=128, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.d_model = d_model

    def forward(self, x):
        # x: [B, d_in, T]
        x = x.permute(0, 2, 1)        # [B, T, d_in]
        x = self.proj_in(x)            # [B, T, d_model]
        T = x.shape[1]
        pos = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, T]
        pos_emb = _sinusoidal_pe(pos, self.d_model)
        x = x + pos_emb
        x = self.encoder(x)            # [B, T, d_model]
        return x.mean(dim=1)           # [B, d_model]  — mean pool over time


def _sinusoidal_pe(positions, d_model):
    """positions: [B, T] → [B, T, d_model]"""
    d = d_model
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(0, d, 2, device=positions.device, dtype=positions.dtype) / d
    )
    pos = positions.unsqueeze(-1)          # [B, T, 1]
    args = pos * freq.unsqueeze(0).unsqueeze(0)  # [B, T, d/2]
    pe = torch.zeros(*positions.shape, d, device=positions.device, dtype=positions.dtype)
    pe[..., 0::2] = torch.sin(args)
    pe[..., 1::2] = torch.cos(args)
    return pe


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SetTransformerEvaluator(nn.Module):
    """Topology-agnostic motion content evaluator.

    Trained independently (not part of AnyTop) with view-consistency loss.
    Frozen at evaluation time.

    Input:
      motion:      [B, J, 13, T]
      joints_mask: [B, J]        True=real joint

    Output: 128D motion embedding (L2-normalized).
    """
    EMBED_DIM = 128

    def __init__(self, d_root=64, d_joint=64, d_temp=128):
        super().__init__()
        self.root_branch  = RootFeatureExtractor(d_out=d_root)
        self.joint_branch = JointBranchEncoder(d_joint=d_joint, d_out=d_joint)
        # Re-declare root proj to accept 4 channels (after _root_features)
        self.root_branch.proj = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, d_root, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal = TemporalEncoder(d_in=d_root + d_joint, d_model=d_temp)

    def forward(self, motion, joints_mask):
        # motion: [B, J, 13, T];  joints_mask: [B, J]
        x_root = motion[:, 0, :, :]                     # [B, 13, T]
        root_emb  = self.root_branch(x_root)             # [B, d_root, T]
        joint_emb = self.joint_branch(motion, joints_mask)  # [B, d_joint, T]

        fused = torch.cat([root_emb, joint_emb], dim=1)  # [B, d_root+d_joint, T]
        emb = self.temporal(fused)                        # [B, 128]
        return F.normalize(emb, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# View-consistency training (self-supervised)
# ─────────────────────────────────────────────────────────────────────────────

def _augment_eval_view(motion):
    """Light augmentation for evaluator self-supervised training.

    motion: [B, J, 13, T]
    """
    B, J, nf, T = motion.shape
    out = motion.clone()

    # Feature noise
    out = out + torch.randn_like(out) * 0.02

    # Temporal warp ±10%
    scale = torch.empty(1).uniform_(0.9, 1.1).item()
    T_new = max(4, int(T * scale))
    flat = out.view(B, J * nf, T)
    warped = F.interpolate(flat, size=T_new, mode='linear', align_corners=False)
    out = F.interpolate(warped, size=T, mode='linear', align_corners=False).view(B, J, nf, T)

    return out


def view_consistency_loss_eval(emb1, emb2, temperature=0.07):
    """NT-Xent (InfoNCE) contrastive loss (original, for backward compat).

    emb1, emb2: [B, D] L2-normalized embeddings from two views of the same motions.
    Positives: (emb1[i], emb2[i]).  Negatives: all other pairs in the batch.
    """
    B = emb1.shape[0]
    embs = torch.cat([emb1, emb2], dim=0)
    sim  = (embs @ embs.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=embs.device)
    sim  = sim.masked_fill(mask, -1e9)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=embs.device),
        torch.arange(0, B,     device=embs.device),
    ])
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical motion label mapping
# ─────────────────────────────────────────────────────────────────────────────

import re

# Map raw filename labels → canonical semantic groups
_CANONICAL = {
    # Locomotion
    'walk': 'walk', 'slowwalk': 'walk', 'walkloop': 'walk', 'walkforward': 'walk',
    'steadywalk': 'walk', 'risingwalk': 'walk', 'walkandrise': 'walk',
    'walkforward': 'walk', 'stopwalk': 'walk', 'injuredwalk': 'walk',
    'walkleft': 'walk', 'walkright': 'walk', 'walkback': 'walk',
    'walker': 'walk', 'stride': 'walk',
    'run': 'run', 'runloop': 'run', 'runfast': 'run', 'running': 'run',
    'landrun': 'run', 'runfall': 'run', 'standingrun': 'run',
    'trot': 'run', 'trotting': 'run',
    'march': 'walk',
    # Fly
    'fly': 'fly', 'flyloop': 'fly', 'flying': 'fly', 'flyglide': 'fly',
    'flyslow': 'fly', 'flyfast': 'fly', 'slowfly': 'fly', 'flyforward': 'fly',
    'screamfly': 'fly', 'slowloop': 'fly', 'circlefly': 'fly', 'float': 'fly',
    'takeoff': 'takeoff',
    # Idle
    'idle': 'idle', 'idle2': 'idle', 'idle3': 'idle', 'idleloop': 'idle',
    'slowidle': 'idle', 'standidle': 'idle', 'idlelook': 'idle',
    'idleears': 'idle', 'idlepissed': 'idle', 'alertidle': 'idle',
    'restless': 'idle', 'restless2': 'idle', 'restless3': 'idle',
    'twitching': 'idle', 'agitated': 'idle',
    # Attack
    'attack': 'attack', 'attack1': 'attack', 'attack2': 'attack',
    'attack3': 'attack', 'attack4': 'attack', 'attack5': 'attack',
    'attack6': 'attack', 'bite': 'attack', 'bite2': 'attack',
    'circlebite': 'attack', 'strike': 'attack', 'strike2': 'attack',
    'stinger': 'attack', 'swat': 'attack', 'angeryswat': 'attack',
    # Die
    'die': 'die', 'die2': 'die', 'dieloop': 'die', 'death': 'die',
    'deathloop': 'die', 'dying': 'die', 'dieup': 'die',
    'fall': 'die', 'bellydeath1': 'die',
    # Recovery
    'getup': 'getup', 'standup': 'getup',
    # Turn
    'turnleft': 'turn', 'turnright': 'turn',
    'turn90left': 'turn', 'turn90right': 'turn',
    # Hit reaction
    'knockedback': 'hit', 'hit': 'hit', 'recoil': 'hit', 'recoil2': 'hit',
    'flipped': 'hit',
    # Other
    'yawn': 'yawn', 'shake': 'shake', 'roar': 'roar',
    'rearing': 'rear', 'rearing2': 'rear', 'bucking': 'rear',
    'downloop': 'lying', 'lying': 'lying', 'downin': 'lying',
}


def canonical_label(filename):
    """Extract canonical motion label from a filename like 'Horse___SlowWalk_432.npy'."""
    m = re.match(r'[^_]+___(.+?)_?\d*\.npy', filename)
    if not m:
        return None
    raw = m.group(1).lower().replace('_', '')
    return _CANONICAL.get(raw, None)


def supcon_loss(emb1, emb2, labels, temperature=0.07):
    """Supervised Contrastive loss (Khosla et al. 2020), fully vectorized.

    emb1, emb2: [B, D] L2-normalized embeddings from two views
    labels:     [B] integer class labels (-1 = no label, excluded from SupCon)

    For labeled samples: all samples with the same label are positives.
    For unlabeled samples (-1): fall back to NT-Xent (only augmented self is positive).
    """
    B = emb1.shape[0]
    embs = torch.cat([emb1, emb2], dim=0)  # [2B, D]
    sim  = (embs @ embs.T) / temperature    # [2B, 2B]

    # Mask out self-similarity
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=embs.device)
    sim = sim.masked_fill(self_mask, -1e9)

    # Build positive mask [2B, 2B] — vectorized
    full_labels = torch.cat([labels, labels])  # [2B]
    has_label = (full_labels >= 0)             # [2B]

    # Same-label positives (only for labeled samples)
    label_match = (full_labels.unsqueeze(0) == full_labels.unsqueeze(1))  # [2B, 2B]
    label_pos   = label_match & has_label.unsqueeze(0) & has_label.unsqueeze(1)

    # Augmented-self positives (always)
    aug_pos = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=embs.device)
    idx = torch.arange(B, device=embs.device)
    aug_pos[idx, idx + B] = True
    aug_pos[idx + B, idx] = True

    # For unlabeled samples: only augmented self
    # For labeled samples: label match + augmented self
    pos_mask = torch.where(
        has_label.unsqueeze(1).expand(2 * B, 2 * B),
        label_pos | aug_pos,
        aug_pos,
    )
    pos_mask.fill_diagonal_(False)  # never self

    # SupCon: -1/|P(i)| * sum_p [ sim(i,p) - log(sum_a exp(sim(i,a))) ]
    log_denom = torch.logsumexp(sim, dim=1)            # [2B]
    log_prob  = sim - log_denom.unsqueeze(1)           # [2B, 2B]

    # Masked mean over positives per anchor
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1)  # [2B]
    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
    return loss_per_anchor.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_embeddings(model, motions, masks, batch_size=32, device='cuda'):
    """Compute embeddings for a list of motions.

    motions: list of [J, 13, T] tensors or single [N, J, 13, T]
    masks:   list of [J] bool tensors or single [N, J]
    Returns: [N, 128] numpy array
    """
    model.eval()
    if isinstance(motions, list):
        motions = torch.stack(motions)
        masks   = torch.stack(masks)

    N = motions.shape[0]
    embs = []
    for i in range(0, N, batch_size):
        b_mot  = motions[i:i+batch_size].to(device)
        b_mask = masks[i:i+batch_size].to(device)
        e = model(b_mot, b_mask)
        embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0)


def retrieval_at_k(query_embs, key_embs, k=1):
    """Nearest-neighbor retrieval: fraction of queries where ground-truth key
    ranks within top-k (queries[i] ↔ keys[i] are paired).

    query_embs, key_embs: [N, D] numpy arrays (L2-normalized).
    """
    sim = query_embs @ key_embs.T   # [N, N]
    ranks = np.argsort(-sim, axis=1)
    hits = sum(1 for i in range(len(query_embs)) if i in ranks[i, :k])
    return hits / len(query_embs)


def cosine_similarity_paired(embs_a, embs_b):
    """Mean cosine similarity between paired embeddings."""
    return float((embs_a * embs_b).sum(axis=1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Simple dataset wrapper for standalone evaluator training
# ─────────────────────────────────────────────────────────────────────────────

class EvalMotionDataset(Dataset):
    """Wraps pre-extracted motion numpy arrays for evaluator training.

    Expected npy format: dict with keys 'motions' [N,J,13,T], 'masks' [N,J], 'names' [N].
    Returns (motion, mask, label_id) where label_id is a canonical motion class (-1 if unmapped).
    """
    def __init__(self, npy_path, max_joints=143, max_frames=40):
        data = np.load(npy_path, allow_pickle=True).item()
        self.motions = data['motions']   # [N, J, 13, T] (already padded)
        self.masks   = data['masks']     # [N, J] bool
        self.names   = data.get('names', np.array([''] * len(self.motions)))

        # Build canonical label IDs
        label_set = {}
        self.label_ids = np.full(len(self.motions), -1, dtype=np.int64)
        for i, name in enumerate(self.names):
            cl = canonical_label(str(name))
            if cl is not None:
                if cl not in label_set:
                    label_set[cl] = len(label_set)
                self.label_ids[i] = label_set[cl]
        self.label_names = {v: k for k, v in label_set.items()}
        n_labeled = (self.label_ids >= 0).sum()
        print(f'  SupCon labels: {len(label_set)} classes, '
              f'{n_labeled}/{len(self.motions)} labeled ({100*n_labeled/len(self.motions):.0f}%)')

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = torch.tensor(self.motions[idx], dtype=torch.float32)
        mask   = torch.tensor(self.masks[idx],   dtype=torch.bool)
        label  = int(self.label_ids[idx])
        return motion, mask, label


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_evaluator(args):
    # Use CPU if GPU is nearly full (another experiment may be running)
    if torch.cuda.is_available():
        free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        device = torch.device('cuda') if free_mb > 6000 else torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(f'Training evaluator on {device}')
    model  = SetTransformerEvaluator().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dataset = EvalMotionDataset(args.data_npy)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, drop_last=True)

    use_supcon = hasattr(args, 'use_supcon') and args.use_supcon

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            motion, mask, labels = batch
            motion = motion.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)

            view2 = _augment_eval_view(motion)

            emb1 = model(motion, mask)
            emb2 = model(view2,  mask)

            if use_supcon:
                loss = supcon_loss(emb1, emb2, labels)
            else:
                loss = view_consistency_loss_eval(emb1, emb2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / max(1, len(loader))
        print(f'Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'loss': best_loss}, args.save_path)
            print(f'  Saved checkpoint → {args.save_path}')

    print(f'Training done. Best loss={best_loss:.4f}')


def eval_retrieval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SetTransformerEvaluator().to(device)
    ckpt   = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    src_data = np.load(args.source_npy, allow_pickle=True).item()
    tgt_data = np.load(args.target_npy, allow_pickle=True).item()

    src_motions = torch.tensor(src_data['motions'], dtype=torch.float32)
    src_masks   = torch.tensor(src_data['masks'],   dtype=torch.bool)
    tgt_motions = torch.tensor(tgt_data['motions'], dtype=torch.float32)
    tgt_masks   = torch.tensor(tgt_data['masks'],   dtype=torch.bool)

    src_embs = compute_embeddings(model, src_motions, src_masks, device=device)
    tgt_embs = compute_embeddings(model, tgt_motions, tgt_masks, device=device)

    r1  = retrieval_at_k(src_embs, tgt_embs, k=1)
    r5  = retrieval_at_k(src_embs, tgt_embs, k=5)
    cos = cosine_similarity_paired(src_embs, tgt_embs)
    print(f'R@1={r1:.3f}  R@5={r5:.3f}  CosSim={cos:.3f}')
    return {'R@1': r1, 'R@5': r5, 'CosSim': cos}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    tr = subparsers.add_parser('train')
    tr.add_argument('--data_npy',   required=True)
    tr.add_argument('--save_path',  default='eval/checkpoints/st_evaluator.pt')
    tr.add_argument('--epochs',     type=int, default=100)
    tr.add_argument('--batch_size', type=int, default=32)
    tr.add_argument('--use_supcon', action='store_true',
                    help='Use SupCon loss with canonical motion labels (recommended)')

    ev = subparsers.add_parser('eval')
    ev.add_argument('--checkpoint',  required=True)
    ev.add_argument('--source_npy',  required=True)
    ev.add_argument('--target_npy',  required=True)

    args = parser.parse_args()
    if args.cmd == 'train':
        train_evaluator(args)
    elif args.cmd == 'eval':
        eval_retrieval(args)
    else:
        parser.print_help()
