"""DPG-SB-v2 — Schrödinger-Bridge generative oracle in MoReFlow latent space.

Per Codex's 4 mandatory tweaks (Round 4+ DPG-evolution review):
  1. SB in latent space (per-skel VQ z-tokens), NOT raw motion features
  2. Single SHARED conditioned bridge (skel_a + skel_b + action), not per-pair
  3. Hard target-manifold anchoring: discriminator + bone/contact/jerk on decoded
  4. Retrieval-initialized noise: init z_b = retrieved candidate z + noise

Architecture (~5-8M params):
  SourceEncoder: source z [B, 8, 256] → tokens [B, 8, d_model]
  BridgeGenerator: takes (z_b_t, t_diff, src_tokens, conds) → velocity for z_b
  Discriminator: takes (z_b, skel_b_emb) → real/fake logit
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sin_emb(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None] * freqs[None, :] * (2 * math.pi)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if d % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SelfAttn(nn.Module):
    def __init__(self, d, h, ff_mult=2, p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.GELU(),
                                 nn.Linear(d * ff_mult, d))

    def forward(self, x):
        h = self.ln1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.ff(self.ln2(x))
        return x


class CrossAttn(nn.Module):
    def __init__(self, d, h, ff_mult=2, p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.GELU(),
                                 nn.Linear(d * ff_mult, d))

    def forward(self, x, src):
        h = self.ln1(x)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa
        h = self.ln2(x)
        ca, _ = self.cross_attn(h, src, src, need_weights=False)
        x = x + ca
        x = x + self.ff(self.ln3(x))
        return x


class BridgeGenerator(nn.Module):
    """SB generator over per-skel VQ z-tokens.

    Inputs:
      z_b_t:    [B, 8, 256]  noisy target z at diffusion time t_diff
      t_diff:   [B]           diffusion time in [0, 1]
      z_a:      [B, 8, 256]   source z (clean)
      action_id: [B]          exact action label
      skel_a_id: [B]          source skel id
      skel_b_id: [B]          target skel id

    Output:
      v: [B, 8, 256]  predicted velocity / target_z
    """
    def __init__(self, codebook_dim: int = 256, n_tokens: int = 8,
                 d_model: int = 384, n_layers: int = 6, n_heads: int = 6,
                 n_skels: int = 70, n_exact_actions: int = 100,
                 src_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Project source + target z to d_model
        self.z_proj_src = nn.Linear(codebook_dim, d_model)
        self.z_proj_tgt = nn.Linear(codebook_dim, d_model)
        # Position embeddings for tokens
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

        # Source encoder
        self.src_blocks = nn.ModuleList([
            SelfAttn(d_model, n_heads, p=dropout) for _ in range(src_layers)
        ])

        # Conditioning: time, skel_a, skel_b, action
        self.skel_emb = nn.Embedding(n_skels, d_model)
        self.action_emb = nn.Embedding(n_exact_actions, d_model)
        self.t_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                     nn.Linear(d_model, d_model))
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))

        # Generator blocks (target queries → cross-attend source)
        self.gen_blocks = nn.ModuleList([
            CrossAttn(d_model, n_heads, p=dropout) for _ in range(n_layers)
        ])

        # Output projection
        self.out_proj = nn.Linear(d_model, codebook_dim)

    def encode_source(self, z_a):
        # z_a: [B, 8, 256] → src tokens [B, 8, d_model]
        h = self.z_proj_src(z_a) + self.pos_emb.unsqueeze(0)
        for blk in self.src_blocks:
            h = blk(h)
        return h

    def forward(self, z_b_t, t_diff, src_tokens, action_id, skel_a_id, skel_b_id):
        B = z_b_t.shape[0]
        # Embed target z + pos
        h = self.z_proj_tgt(z_b_t) + self.pos_emb.unsqueeze(0)

        # Conditioning: time + skel_a + skel_b + action
        t_emb = self.t_proj(sin_emb(t_diff, self.d_model))     # [B, d]
        sa_emb = self.skel_emb(skel_a_id)                       # [B, d]
        sb_emb = self.skel_emb(skel_b_id)                       # [B, d]
        ae_emb = self.action_emb(action_id)                     # [B, d]
        cond = self.cond_proj(torch.cat([t_emb, sa_emb, sb_emb, ae_emb], dim=-1))
        # Add as global conditioning to each token
        h = h + cond.unsqueeze(1)

        # Generator blocks: target queries cross-attend source
        for blk in self.gen_blocks:
            h = blk(h, src_tokens)

        # Predict velocity in z space
        v = self.out_proj(h)  # [B, 8, 256]
        return v


class Discriminator(nn.Module):
    """Conditional discriminator over per-skel z-tokens."""
    def __init__(self, codebook_dim: int = 256, n_tokens: int = 8,
                 d_model: int = 256, n_layers: int = 3, n_heads: int = 4,
                 n_skels: int = 70, dropout: float = 0.1):
        super().__init__()
        self.z_proj = nn.Linear(codebook_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.skel_emb = nn.Embedding(n_skels, d_model)

        self.blocks = nn.ModuleList([
            SelfAttn(d_model, n_heads, p=dropout) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, 1)

    def forward(self, z, skel_id):
        h = self.z_proj(z) + self.pos_emb.unsqueeze(0)
        h = h + self.skel_emb(skel_id).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h)
        # Pool + head
        h_pool = h.mean(dim=1)
        logit = self.head(h_pool).squeeze(-1)
        return logit  # [B] real/fake logit


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    G = BridgeGenerator(d_model=384, n_layers=6, n_heads=6,
                         n_skels=70, n_exact_actions=80)
    D = Discriminator(d_model=256, n_layers=3, n_heads=4, n_skels=70)
    print(f"Generator: {count_params(G)/1e6:.1f}M params")
    print(f"Discriminator: {count_params(D)/1e6:.1f}M params")

    B = 4
    z_a = torch.randn(B, 8, 256)
    z_b = torch.randn(B, 8, 256)
    t = torch.rand(B)
    aid = torch.randint(0, 80, (B,))
    sa = torch.randint(0, 70, (B,))
    sb = torch.randint(0, 70, (B,))

    src = G.encode_source(z_a)
    print(f"Source tokens: {src.shape}")
    v = G(z_b, t, src, aid, sa, sb)
    print(f"Generator output: {v.shape}")
    d_logit = D(z_b, sb)
    print(f"Discriminator logit: {d_logit.shape}")
