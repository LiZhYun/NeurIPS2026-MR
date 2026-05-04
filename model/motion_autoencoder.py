"""Stage 1 auto-encoder: encoder + skeleton-aware decoder for direct MSE training.

Design choices (from external review):
  - Encoder input is CANONICALIZED: rotations + scale-normalized root + contact.
    Raw positions are excluded to reduce morphology leakage.
  - Decoder is skeleton-AWARE: receives offsets, joint mask, tpos.
    This lets z focus on motion intent rather than spatial layout.
  - Latent is CONTINUOUS (not FSQ). Discretize later once representation is validated.
  - Supports 3 training modes: raw, norm, norm+SupCon, norm+Cycle.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# Encoder components
# ─────────────────────────────────────────────────────────────────────────────

class CanonicalFeatureExtractor(nn.Module):
    """Extract morphology-reduced features from raw 13D motion.

    Input:  motion [B, J, 13, T], offsets [B, J, 3]
    Output: canonical features [B, T, J, D_canon]

    Keeps: 6D rotation (skeleton-agnostic local articulation)
           scale-normalized root velocity
           contact flags
    Drops: raw root-relative positions (morphology-heavy)
    """
    def __init__(self, d_canon=64):
        super().__init__()
        # STRIPPED morphology cues per Round 5 review:
        #   - removed root height (scales with morphology)
        #   - removed joint-indexed contact (joint identity leaks)
        # Joint: 6D rotation(6) + velocity(3) = 9 per joint
        # Root: xz_vel(2) + vert_vel(1) + yaw_rate(1) = 4
        self.joint_proj = nn.Linear(9, d_canon)
        self.root_proj = nn.Linear(4, d_canon)

    def forward(self, motion, offsets, mask):
        """
        motion:  [B, J, 13, T]
        offsets: [B, J, 3]
        mask:    [B, J] bool
        Returns: [B, T, J, D_canon]
        """
        x = motion.permute(0, 3, 1, 2)  # [B, T, J, 13]
        B, T, J, _ = x.shape

        # Body scale for velocity normalization
        bone_len = offsets.norm(dim=-1).clamp(min=1e-6)
        body_scale = (bone_len * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
        body_scale = body_scale.view(B, 1, 1, 1).clamp(min=1e-4)

        # Root features: ONLY velocity-derived (no absolute height)
        root_pos = x[:, :, 0, :3]  # [B, T, 3]
        root_vel = torch.zeros_like(root_pos)
        root_vel[:, 1:] = (root_pos[:, 1:] - root_pos[:, :-1])
        root_vel = root_vel / body_scale.squeeze(-1)

        yaw = torch.atan2(root_vel[:, :, 2:3], root_vel[:, :, 0:1] + 1e-8)
        dyaw = torch.zeros_like(yaw)
        dyaw[:, 1:] = yaw[:, 1:] - yaw[:, :-1]

        root_feat = torch.cat([
            root_vel[:, :, [0, 2]],  # xz velocity
            root_vel[:, :, 1:2],     # vertical velocity
            dyaw,                     # yaw rate
        ], dim=-1)  # [B, T, 4]
        root_emb = self.root_proj(root_feat).unsqueeze(2)  # [B, T, 1, D]

        # Joint features: rotation + velocity ONLY (no contact)
        joint_rot = x[:, :, 1:, 3:9]                                # [B, T, J-1, 6]
        joint_vel = x[:, :, 1:, 9:12] / body_scale                  # [B, T, J-1, 3]
        joint_feat = torch.cat([joint_rot, joint_vel], dim=-1)      # [B, T, J-1, 9]
        joint_emb = self.joint_proj(joint_feat)                      # [B, T, J-1, D]

        out = torch.cat([root_emb, joint_emb], dim=2)  # [B, T, J, D]
        return out


class RawFeatureExtractor(nn.Module):
    """Baseline: embed raw 13D features without canonicalization."""
    def __init__(self, d_canon=64, feature_len=13):
        super().__init__()
        self.proj = nn.Linear(feature_len, d_canon)

    def forward(self, motion, offsets, mask):
        x = motion.permute(0, 3, 1, 2)  # [B, T, J, 13]
        return self.proj(x)  # [B, T, J, D]


class AttentionPool(nn.Module):
    """K learned queries compress J joints → K slots per frame."""
    def __init__(self, d_model, num_queries, num_heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)

    def forward(self, h, mask):
        # h: [B, T, J, D], mask: [B, J] True=real
        B, T, J, D = h.shape
        h_flat = h.reshape(B * T, J, D)
        q = self.queries.unsqueeze(0).expand(B * T, -1, -1)
        key_pad = ~mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J)
        out, _ = self.attn(q, h_flat, h_flat, key_padding_mask=key_pad)
        K = self.queries.shape[0]
        return out.view(B, T, K, D)


class TemporalCNN(nn.Module):
    """Downsample T → T/4."""
    def __init__(self, d_model):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1, stride=2), nn.GELU(),
        )

    def forward(self, s):
        B, T, K, D = s.shape
        x = s.permute(0, 2, 3, 1).reshape(B * K, D, T)
        x = self.convs(x)
        T2 = x.shape[-1]
        return x.view(B, K, D, T2).permute(0, 3, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder components (skeleton-aware, disposable after Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalUpsampleCNN(nn.Module):
    """T/4 → T via transposed convolutions."""
    def __init__(self, d_model):
        super().__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, 4, padding=1, stride=2), nn.GELU(),
            nn.ConvTranspose1d(d_model, d_model, 4, padding=1, stride=2), nn.GELU(),
        )

    def forward(self, s, T_target):
        B, T4, K, D = s.shape
        x = s.permute(0, 2, 3, 1).reshape(B * K, D, T4)
        x = self.convs(x)
        if x.shape[-1] > T_target:
            x = x[:, :, :T_target]
        elif x.shape[-1] < T_target:
            x = F.pad(x, (0, T_target - x.shape[-1]))
        return x.view(B, K, D, T_target).permute(0, 3, 1, 2)


class SkeletonAwareDecoder(nn.Module):
    """Decode z → motion using skeleton geometry.

    Receives: z slots [B, T, K, D] + skeleton offsets [B, J, 3] + joint mask [B, J]
    Produces: motion [B, T, J, 13]
    """
    def __init__(self, d_model, max_joints, feature_len=13, num_heads=4):
        super().__init__()
        self.offset_emb = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, feature_len)

    def forward(self, slots, offsets, mask):
        # slots: [B, T, K, D], offsets: [B, J, 3], mask: [B, J]
        B, T, K, D = slots.shape
        J = offsets.shape[1]

        # Joint queries from skeleton geometry
        joint_q = self.offset_emb(offsets)  # [B, J, D]
        joint_q = joint_q.unsqueeze(1).expand(B, T, J, D).reshape(B * T, J, D)
        kv = slots.reshape(B * T, K, D)

        out, _ = self.cross_attn(joint_q, kv, kv)
        out = self.norm1(out + joint_q)
        out = self.norm2(out + self.ffn(out))
        out = out.view(B, T, J, D)
        return self.out_proj(out)  # [B, T, J, 13]


# ─────────────────────────────────────────────────────────────────────────────
# Full auto-encoder
# ─────────────────────────────────────────────────────────────────────────────

class MotionAutoEncoder(nn.Module):
    """Stage 1 auto-encoder with configurable encoder input and constraints.

    Modes:
      canonical=False: raw 13D features (AE-raw baseline)
      canonical=True:  canonicalized features (AE-norm)
      + SupCon or Cycle losses added externally in training loop
    """
    def __init__(self, max_joints=143, feature_len=13, d_model=128,
                 d_z=32, num_queries=4, num_heads=4, canonical=True):
        super().__init__()
        self.d_z = d_z
        self.d_model = d_model
        self.canonical = canonical

        # Encoder
        if canonical:
            self.feat_extract = CanonicalFeatureExtractor(d_canon=d_model)
        else:
            self.feat_extract = RawFeatureExtractor(d_canon=d_model, feature_len=feature_len)

        self.attn_pool = AttentionPool(d_model, num_queries, num_heads)
        self.temporal_down = TemporalCNN(d_model)

        # Continuous bottleneck (not FSQ — validate representation first)
        self.to_latent = nn.Linear(d_model, d_z)
        self.from_latent = nn.Linear(d_z, d_model)

        # Decoder (skeleton-aware)
        self.temporal_up = TemporalUpsampleCNN(d_model)
        self.decoder = SkeletonAwareDecoder(d_model, max_joints, feature_len, num_heads)

    def encode(self, motion, offsets, mask):
        """Encode motion → z [B, T/4, K, d_z]."""
        h = self.feat_extract(motion, offsets, mask)  # [B, T, J, D]
        pooled = self.attn_pool(h, mask)               # [B, T, K, D]
        down = self.temporal_down(pooled)               # [B, T/4, K, D]
        z = self.to_latent(down)                        # [B, T/4, K, d_z]
        return z

    def decode(self, z, offsets, mask, T_target):
        """Decode z → motion [B, J, 13, T] using target skeleton geometry."""
        z_proj = self.from_latent(z)                    # [B, T/4, K, D]
        up = self.temporal_up(z_proj, T_target)         # [B, T, K, D]
        recon = self.decoder(up, offsets, mask)          # [B, T, J, 13]
        return recon.permute(0, 2, 3, 1)                # [B, J, 13, T]

    def forward(self, motion, offsets, mask):
        """Full encode-decode. Returns (reconstruction, z)."""
        T = motion.shape[-1]
        z = self.encode(motion, offsets, mask)
        recon = self.decode(z, offsets, mask, T)
        return recon, z

    def z_pooled(self, z):
        """Pool z to a single vector per sample for probing/contrastive loss."""
        return z.mean(dim=(1, 2))  # [B, d_z]
