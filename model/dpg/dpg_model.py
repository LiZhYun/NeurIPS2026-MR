"""DPG (Direct Paired Generative) — supervised paired generative oracle.

Per dual-review (local + Codex xhigh, 2026-04-24):
  - Source encoder outputs time-varying tokens [T_src, d] (NOT pooled vector)
  - Generator uses cross-attention from target tokens to source tokens
  - Action label + target skel + target length/mask all explicit conditioning
  - Same-skel self-recon auxiliary branch (handled in training script)

Architecture (~10-15M params):
  SourceEncoder: motion [T_src, J_max, 13] + masks → tokens [T_src, d]
  TargetGenerator: noise [T_tgt, J_max, 13] + cross-attn to source → velocity
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_emb(t: torch.Tensor, d: int) -> torch.Tensor:
    """Standard sinusoidal embedding for diffusion time (t in [0,1]) → [B, d]."""
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None] * freqs[None, :] * (2 * math.pi)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if d % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Cross-attention from x → source_tokens, plus self-attention + FFN."""
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x, src_tokens, x_mask=None, src_mask=None):
        # x: [B, L_x, d], src_tokens: [B, L_src, d]
        # x_mask: [B, L_x] padding (True=pad), src_mask: [B, L_src] padding
        h = self.ln1(x)
        sa, _ = self.self_attn(h, h, h, key_padding_mask=x_mask, need_weights=False)
        x = x + sa
        h = self.ln2(x)
        ca, _ = self.cross_attn(h, src_tokens, src_tokens,
                                 key_padding_mask=src_mask, need_weights=False)
        x = x + ca
        x = x + self.ff(self.ln3(x))
        return x


class DPGSourceEncoder(nn.Module):
    """Encode source motion [T, J, 13] + masks → time-varying tokens [T, d]."""
    def __init__(self, d_model: int = 256, n_layers: int = 3, n_heads: int = 4,
                 max_J: int = 143, max_T: int = 200, n_clusters: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_J = max_J
        self.max_T = max_T

        # Per (frame, joint) feature embedding
        self.joint_feat_proj = nn.Linear(13, d_model)
        # Joint position embedding (each joint slot has identity)
        self.joint_pos_emb = nn.Parameter(torch.randn(max_J, d_model) * 0.02)
        # Time position embedding
        self.time_pos_emb = nn.Parameter(torch.randn(max_T, d_model) * 0.02)

        # Joint attention (1 layer, per-frame)
        self.joint_attn = SelfAttnBlock(d_model, n_heads, d_model * 2)
        # Temporal transformer
        self.time_blocks = nn.ModuleList([
            SelfAttnBlock(d_model, n_heads, d_model * 2) for _ in range(n_layers)
        ])

        # Auxiliary cluster head (mean pool over T)
        self.aux_cluster = nn.Linear(d_model, n_clusters)

    def forward(self, motion: torch.Tensor, joint_mask: torch.Tensor,
                t_mask: torch.Tensor):
        """
        motion: [B, T, J, 13], joint_mask: [B, J] bool, t_mask: [B, T] bool
        Returns: tokens [B, T, d], aux_cluster_logits [B, n_clusters]
        """
        B, T, J, _ = motion.shape
        # Project + add joint pos emb
        h = self.joint_feat_proj(motion) + self.joint_pos_emb[:J].unsqueeze(0).unsqueeze(0)
        # Joint attention per frame: collapse B*T into one dim
        h_jt = h.reshape(B * T, J, self.d_model)
        m_jt = (~joint_mask).unsqueeze(1).expand(B, T, J).reshape(B * T, J)
        h_jt = self.joint_attn(h_jt, key_padding_mask=m_jt)
        # Masked mean over joints
        valid_j = joint_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J).float().unsqueeze(-1)
        h_pool = (h_jt * valid_j).sum(dim=1) / (valid_j.sum(dim=1) + 1e-6)  # [B*T, d]
        h_pool = h_pool.reshape(B, T, self.d_model)
        # Add time pos emb
        h_pool = h_pool + self.time_pos_emb[:T].unsqueeze(0)
        # Temporal transformer (with t_mask)
        t_pad_mask = ~t_mask  # True = pad
        for blk in self.time_blocks:
            h_pool = blk(h_pool, key_padding_mask=t_pad_mask)
        # Auxiliary cluster: masked mean over time
        valid_t = t_mask.float().unsqueeze(-1)
        h_clip = (h_pool * valid_t).sum(dim=1) / (valid_t.sum(dim=1) + 1e-6)  # [B, d]
        aux_logits = self.aux_cluster(h_clip)
        return h_pool, aux_logits


class DPGGenerator(nn.Module):
    """Flow matching generator: noisy target motion + cross-attn to source → velocity.

    Output: per-joint per-frame velocity in motion-feature space.
    """
    def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 4,
                 max_J: int = 143, max_T: int = 200, n_skels: int = 70, n_exact_actions: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_J = max_J
        self.max_T = max_T

        self.joint_feat_proj = nn.Linear(13, d_model)
        self.joint_pos_emb = nn.Parameter(torch.randn(max_J, d_model) * 0.02)
        self.time_pos_emb = nn.Parameter(torch.randn(max_T, d_model) * 0.02)

        # Conditioning: time, action, skel
        self.t_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.action_emb = nn.Embedding(n_exact_actions, d_model)
        self.skel_emb = nn.Embedding(n_skels, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        # Joint attention block (1 layer)
        self.joint_attn = SelfAttnBlock(d_model, n_heads, d_model * 2)
        # Temporal generator blocks with cross-attention to source
        self.gen_blocks = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, d_model * 2) for _ in range(n_layers)
        ])

        # Per-joint output projection
        self.out_proj = nn.Linear(d_model, 13)

    def forward(self, z_t: torch.Tensor, t_diff: torch.Tensor,
                src_tokens: torch.Tensor, src_t_mask: torch.Tensor,
                tgt_joint_mask: torch.Tensor, tgt_t_mask: torch.Tensor,
                action_idx: torch.Tensor, skel_idx: torch.Tensor):
        """
        z_t:           [B, T, J, 13]   noisy target motion at time t_diff
        t_diff:        [B]              diffusion time in [0,1]
        src_tokens:    [B, T_src, d]   source encoder output
        src_t_mask:    [B, T_src] bool valid source frames
        tgt_joint_mask:[B, J]  bool   valid target joints
        tgt_t_mask:    [B, T]  bool   valid target frames
        action_idx:    [B]              exact action index
        skel_idx:      [B]              target skel id

        Returns: velocity [B, T, J, 13]
        """
        B, T, J, _ = z_t.shape
        # Embed motion features
        h = self.joint_feat_proj(z_t) + self.joint_pos_emb[:J].unsqueeze(0).unsqueeze(0)
        # Per-frame joint attention
        h_jt = h.reshape(B * T, J, self.d_model)
        m_jt = (~tgt_joint_mask).unsqueeze(1).expand(B, T, J).reshape(B * T, J)
        h_jt = self.joint_attn(h_jt, key_padding_mask=m_jt)
        # Masked mean over joints — reduce per frame to [B, T, d]
        valid_j = tgt_joint_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J).float().unsqueeze(-1)
        h_t = (h_jt * valid_j).sum(dim=1) / (valid_j.sum(dim=1) + 1e-6)
        h_t = h_t.reshape(B, T, self.d_model)
        # Add time pos emb
        h_t = h_t + self.time_pos_emb[:T].unsqueeze(0)

        # Conditioning vector (time, action, skel)
        t_emb = self.t_proj(sinusoidal_time_emb(t_diff, self.d_model))  # [B, d]
        a_emb = self.action_emb(action_idx)  # [B, d]
        s_emb = self.skel_emb(skel_idx)      # [B, d]
        cond = self.cond_proj(torch.cat([t_emb, a_emb, s_emb], dim=-1))  # [B, d]
        # Add as conditioning to each frame
        h_t = h_t + cond.unsqueeze(1)  # broadcast [B, T, d]

        # Generator blocks with cross-attention to source
        tgt_pad_mask = ~tgt_t_mask
        src_pad_mask = ~src_t_mask
        for blk in self.gen_blocks:
            h_t = blk(h_t, src_tokens, x_mask=tgt_pad_mask, src_mask=src_pad_mask)

        # Now broadcast back to per-joint to produce velocities
        # Reuse joint feat embedding to inject joint-specific info
        joint_pos = self.joint_pos_emb[:J].unsqueeze(0).unsqueeze(0)  # [1, 1, J, d]
        h_full = h_t.unsqueeze(2) + joint_pos  # [B, T, J, d]
        # Project to 13-d velocity
        velocity = self.out_proj(h_full)  # [B, T, J, 13]
        # Mask out invalid joints/frames
        joint_v = tgt_joint_mask.unsqueeze(1).unsqueeze(-1).float()
        time_v = tgt_t_mask.unsqueeze(-1).unsqueeze(-1).float()
        velocity = velocity * joint_v * time_v
        return velocity


class DPGModel(nn.Module):
    """Combined source encoder + target generator."""
    def __init__(self, d_model: int = 256, n_layers_src: int = 3, n_layers_gen: int = 6,
                 n_heads: int = 4, max_J: int = 143, max_T: int = 200,
                 n_skels: int = 70, n_exact_actions: int = 100, n_clusters: int = 10):
        super().__init__()
        self.encoder = DPGSourceEncoder(
            d_model=d_model, n_layers=n_layers_src, n_heads=n_heads,
            max_J=max_J, max_T=max_T, n_clusters=n_clusters)
        self.generator = DPGGenerator(
            d_model=d_model, n_layers=n_layers_gen, n_heads=n_heads,
            max_J=max_J, max_T=max_T, n_skels=n_skels,
            n_exact_actions=n_exact_actions)

    def encode(self, src_motion, src_joint_mask, src_t_mask):
        return self.encoder(src_motion, src_joint_mask, src_t_mask)

    def generate(self, z_t, t_diff, src_tokens, src_t_mask,
                 tgt_joint_mask, tgt_t_mask, action_idx, skel_idx):
        return self.generator(z_t, t_diff, src_tokens, src_t_mask,
                               tgt_joint_mask, tgt_t_mask, action_idx, skel_idx)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = DPGModel(d_model=256, n_layers_src=3, n_layers_gen=6,
                 n_heads=4, max_J=143, max_T=200, n_skels=70, n_exact_actions=100)
    print(f"Total params: {count_params(m)/1e6:.1f}M")
    print(f"Encoder: {count_params(m.encoder)/1e6:.1f}M")
    print(f"Generator: {count_params(m.generator)/1e6:.1f}M")

    B = 2
    src_motion = torch.randn(B, 100, 50, 13)
    src_jmask = torch.ones(B, 50, dtype=torch.bool); src_jmask[:, 30:] = False
    src_tmask = torch.ones(B, 100, dtype=torch.bool); src_tmask[:, 60:] = False
    tokens, aux = m.encode(src_motion, src_jmask, src_tmask)
    print(f"Source tokens: {tokens.shape}, aux: {aux.shape}")

    z_t = torch.randn(B, 80, 60, 13)
    t = torch.rand(B)
    tgt_jmask = torch.ones(B, 60, dtype=torch.bool); tgt_jmask[:, 40:] = False
    tgt_tmask = torch.ones(B, 80, dtype=torch.bool); tgt_tmask[:, 50:] = False
    aid = torch.zeros(B, dtype=torch.long)
    sid = torch.zeros(B, dtype=torch.long)
    v = m.generate(z_t, t, tokens, src_tmask, tgt_jmask, tgt_tmask, aid, sid)
    print(f"Velocity: {v.shape}")
