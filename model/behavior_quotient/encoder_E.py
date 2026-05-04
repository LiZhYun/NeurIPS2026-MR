"""E Stage A — Skeleton-blind behavior-code encoder for cross-skel retargeting.

Per FINAL_PROPOSAL V4 (Codex 7.5/10): low-leakage encoder + multi-head
auxiliary skeleton-agnostic supervision + IB regularization + GRL skel head.
NOT architecturally skeleton-blind (joint mask still leaks skel-id; we
measure residual leakage via probe, not block it).

Architecture:
  Input:  motion [B, T, J_max=143, 13]  (no skel-id, no graph features)
          mask   [B, J_max]               (joint validity mask — known leakage source)
  Output: behavior_code z [B, D=128]      (clip-level pooled latent)
          + auxiliary head outputs:
            - action_logits [B, n_clusters=10]
            - q_pred [B, 22]
            - contact_density [B] (mean per-frame contact density)
            - com_heading [B, 4]   (mean COM xyz + mean heading)
            - skel_logits [B, n_skels=70]   (gradient-reversed; for invariance pressure)
          + IB stats: mu, logvar (for KL term)

Loss summary:
  L = w_action * CE(action_logits, action_label)
    + w_Q * MSE(q_pred, Q_22d)
    + w_contact * MSE(contact_density, contact_density_target)
    + w_COM * MSE(com_heading, com_heading_target)
    + w_KL * KL(N(mu, sigma) || N(0, I))
    + w_grl * (-CE(skel_logits, skel_id))      # gradient-reversed
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal layer (Ganin et al., 2015)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class JointMaskedAttention(nn.Module):
    """Single attention block over joints with mask (per-frame)."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: [B, J, d]; mask: [B, J] bool (True = valid)
        h = self.ln1(x)
        if mask is not None:
            key_padding_mask = ~mask  # True = padding (invalid)
        else:
            key_padding_mask = None
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class TemporalTransformerBlock(nn.Module):
    """Standard transformer block over time."""
    def __init__(self, d_model: int, n_heads: int = 4, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class BehaviorEncoderE(nn.Module):
    """E Stage A encoder + auxiliary heads + IB + GRL skel head.

    Pipeline:
      motion [B, T, J, 13] + mask [B, J]
        -> joint_proj [B, T, J, d]
        -> JointAttention (masked) per frame: [B, T, J, d]
        -> masked-mean over joints: [B, T, d]
        -> TemporalTransformer × N: [B, T, d]
        -> mean-pool over T: [B, d]
        -> mu, logvar (IB) → reparam → z [B, D]
        -> heads
    """
    def __init__(self,
                 d_in: int = 13,
                 d_model: int = 256,
                 n_joint_layers: int = 2,
                 n_temporal_layers: int = 4,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 d_z: int = 128,
                 n_clusters: int = 10,
                 n_skels: int = 70,
                 q_dim: int = 22):
        super().__init__()
        self.d_model = d_model
        self.d_z = d_z

        self.joint_proj = nn.Linear(d_in, d_model)
        self.joint_attn = nn.ModuleList([
            JointMaskedAttention(d_model, n_heads, dropout)
            for _ in range(n_joint_layers)
        ])
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_temporal_layers)
        ])
        # IB: predict mu and logvar
        self.mu_head = nn.Linear(d_model, d_z)
        self.logvar_head = nn.Linear(d_model, d_z)

        # Auxiliary heads (operate on z)
        self.action_head = nn.Sequential(
            nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, n_clusters))
        self.q_head = nn.Sequential(
            nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, q_dim))
        self.contact_head = nn.Sequential(
            nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, 1))
        self.com_head = nn.Sequential(
            nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, 4))

        # GRL skel head (gradient reversed)
        self.skel_head = nn.Sequential(
            nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, n_skels))

    def forward(self, motion: torch.Tensor, joint_mask: torch.Tensor,
                grl_alpha: float = 1.0, return_features: bool = False):
        """
        motion:     [B, T, J, 13]
        joint_mask: [B, J]   bool (True = valid)

        Returns dict with keys:
          z, mu, logvar, action_logits, q_pred, contact_pred, com_pred, skel_logits
        Optional: features (intermediate temporal mean-pool [B, d])
        """
        B, T, J, _ = motion.shape

        # 1. Joint feature projection
        h = self.joint_proj(motion)        # [B, T, J, d]

        # 2. Joint attention per frame (collapse B*T into one batch)
        h = h.reshape(B * T, J, self.d_model)
        m = joint_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J)
        for blk in self.joint_attn:
            h = blk(h, mask=m)
        # Masked mean over joints
        m_f = m.float().unsqueeze(-1)        # [B*T, J, 1]
        h_pool = (h * m_f).sum(dim=1) / (m_f.sum(dim=1) + 1e-6)  # [B*T, d]
        h_pool = h_pool.reshape(B, T, self.d_model)              # [B, T, d]

        # 3. Temporal transformer
        for blk in self.temporal_blocks:
            h_pool = blk(h_pool)             # [B, T, d]

        # 4. Mean-pool over time
        feat = h_pool.mean(dim=1)            # [B, d]

        # 5. IB: mu, logvar → z (reparam during train)
        mu = self.mu_head(feat)              # [B, D]
        logvar = self.logvar_head(feat).clamp(min=-10, max=10)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu

        # 6. Aux heads
        action_logits = self.action_head(z)
        q_pred = self.q_head(z)
        contact_pred = self.contact_head(z).squeeze(-1)
        com_pred = self.com_head(z)
        # GRL skel head — gradient reversed for adversarial invariance
        z_rev = gradient_reversal(z, alpha=grl_alpha)
        skel_logits = self.skel_head(z_rev)

        out = {
            'z': z, 'mu': mu, 'logvar': logvar,
            'action_logits': action_logits,
            'q_pred': q_pred,
            'contact_pred': contact_pred,
            'com_pred': com_pred,
            'skel_logits': skel_logits,
        }
        if return_features:
            out['features'] = feat
        return out


def kl_loss(mu, logvar):
    """KL(N(mu, sigma) || N(0, I)) per-batch mean."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    enc = BehaviorEncoderE(d_z=128, n_clusters=10, n_skels=70, q_dim=22)
    print(f"params: {count_parameters(enc)/1e6:.2f}M")
    motion = torch.randn(4, 100, 50, 13)
    mask = torch.ones(4, 50, dtype=torch.bool)
    mask[:, 30:] = False
    out = enc(motion, mask, grl_alpha=0.5)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
