"""ACE Discriminator: scores decoded target motion transitions (real vs generated).

Per ACE_DESIGN_V3 §3.2, paper-faithful (D operates on (x_prev, x_cur), not latent z):

  Input:   x_prev_tgt [B, 32, J_tgt, 13], x_cur_tgt [B, 32, J_tgt, 13],
           tgt_skel_id [B], tgt_graph [B, 128], joint_mask [B, J_max=142]
  Output:  d_logit [B, 1] — pre-sigmoid; bigger = more "real"

Architecture:
  - Concat (x_prev, x_cur) along time axis → [B, 64, J_max, 13]
  - Per-frame segment embedding (binary "prev/cur" added to per-frame learned embedding)
  - Per-joint feature projection: Linear(13, 128) → [B, 64, J_max, 128]
  - Joint-axis transformer (3 layers × 4 heads × 128) with FiLM conditioning
  - Joint pooling: masked mean → [B, 64, 128]
  - Temporal transformer (2 layers × 4 heads × 128) with FiLM
  - Temporal pooling: mean → [B, 128]
  - Final MLP → 1 logit

R1 GP applied externally on real input. No spectral norm by default (paper uses GP only).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

J_MAX = 142  # max joint count across 70 Truebones skels (Dragon=142)


class FiLMTransformerLayer(nn.Module):
    """Standard Transformer encoder layer with FiLM modulation injected on the residual paths."""

    def __init__(self, d_model, n_heads, dim_ff, dropout, d_cond):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        # FiLM: produces (γ_attn, β_attn, γ_ffn, β_ffn) per d_model channel.
        # Zero-init so FiLM starts as identity: (1+0)·h + 0 = h.
        self.film = nn.Linear(d_cond, 4 * d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond, attn_mask=None, key_padding_mask=None):
        """
        x:     [B, T, d_model]
        cond:  [B, d_cond]
        Returns: [B, T, d_model]
        """
        film_params = self.film(cond)                                     # [B, 4*d_model]
        gamma_a, beta_a, gamma_f, beta_f = film_params.chunk(4, dim=-1)
        # Broadcast over sequence
        gamma_a = gamma_a.unsqueeze(1)
        beta_a = beta_a.unsqueeze(1)
        gamma_f = gamma_f.unsqueeze(1)
        beta_f = beta_f.unsqueeze(1)

        # Self-attention sublayer
        h = self.norm1(x)
        h = (1 + gamma_a) * h + beta_a                                    # FiLM modulation
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out

        # FFN sublayer
        h = self.norm2(x)
        h = (1 + gamma_f) * h + beta_f                                    # FiLM modulation
        ffn_out = self.ffn(h)
        x = x + ffn_out
        return x


class ACEDiscriminator(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_joint_layers=3, n_temporal_layers=2,
                 dim_ff=512, dropout=0.1, T_total=64,
                 n_skels=70, d_skel_id_emb=64, d_graph=128, d_cond=192):
        super().__init__()
        self.d_model = d_model
        self.T_total = T_total                                            # 32 prev + 32 cur

        # Per-joint feature projection
        self.joint_proj = nn.Linear(13, d_model)
        # Per-frame segment embedding (binary: 0=prev, 1=cur)
        self.segment_emb = nn.Embedding(2, d_model)
        # Per-frame positional encoding (length T_total = 64)
        self.frame_pe = nn.Parameter(torch.zeros(T_total, d_model))
        nn.init.trunc_normal_(self.frame_pe, std=0.02)

        # Conditioning: tgt_skel_id + tgt_graph → d_cond
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.cond_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Joint-axis transformer (operates over J_max with masking)
        self.joint_layers = nn.ModuleList([
            FiLMTransformerLayer(d_model, n_heads, dim_ff, dropout, d_cond)
            for _ in range(n_joint_layers)
        ])

        # Temporal transformer (operates over T_total = 64 after joint pooling)
        self.temporal_layers = nn.ModuleList([
            FiLMTransformerLayer(d_model, n_heads, dim_ff, dropout, d_cond)
            for _ in range(n_temporal_layers)
        ])

        # Final head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_prev, x_cur, tgt_skel_id, tgt_graph, joint_mask):
        """
        x_prev:      [B, 32, J_tgt, 13] — decoded prev target window (physical units)
        x_cur:       [B, 32, J_tgt, 13] — decoded current target window
        tgt_skel_id: [B]
        tgt_graph:   [B, d_graph]
        joint_mask:  [B, J_max] — full mask matrix; we slice to actual J_pad below

        Returns: d_logit [B, 1]
        """
        B = x_prev.shape[0]
        T_half = x_prev.shape[1]                                          # 32
        # Use dynamic J_pad = actual joint count of input (single-skel per iter; batch is same skel)
        # This avoids materializing [B*T, H, J_MAX=142, J_MAX=142] attention scores when
        # J_tgt << J_MAX. Major memory saving for skels with few joints.
        J_pad = x_prev.shape[2]
        # Concat along time → [B, 64, J_pad, 13]
        x = torch.cat([x_prev, x_cur], dim=1)                             # [B, T_total, J_pad, 13]
        # Project per joint per frame
        h = self.joint_proj(x)                                            # [B, T_total, J_pad, d_model]

        # Segment + frame positional encoding
        seg_ids = torch.cat([
            torch.zeros(T_half, device=h.device, dtype=torch.long),
            torch.ones(T_half, device=h.device, dtype=torch.long),
        ])                                                                # [T_total]
        seg_h = self.segment_emb(seg_ids).unsqueeze(0).unsqueeze(2)       # [1, T_total, 1, d_model]
        h = h + seg_h
        h = h + self.frame_pe.unsqueeze(0).unsqueeze(2)                   # [1, T_total, 1, d_model]

        # Conditioning vector
        skel_id_h = self.skel_id_emb(tgt_skel_id)                         # [B, d_skel_id_emb]
        cond = self.cond_proj(torch.cat([skel_id_h, tgt_graph], dim=-1))  # [B, d_cond]

        # Joint-axis attention: reshape to [B*T_total, J_pad, d_model], apply transformer over J
        h_joint = h.reshape(B * self.T_total, J_pad, self.d_model)        # [B*T_total, J_pad, d_model]
        # Slice mask to J_pad columns only
        joint_mask_pad = joint_mask[:, :J_pad]                            # [B, J_pad]
        joint_mask_rep = joint_mask_pad.unsqueeze(1).expand(B, self.T_total, -1).reshape(
            B * self.T_total, J_pad)                                      # [B*T_total, J_pad]
        # MultiheadAttention key_padding_mask: True = ignore
        key_padding_mask = (joint_mask_rep < 0.5)                         # [B*T_total, J_pad]
        # Cond per (B*T_total): broadcast cond over time
        cond_per_step = cond.unsqueeze(1).expand(B, self.T_total, -1).reshape(
            B * self.T_total, -1)                                         # [B*T_total, d_cond]
        for layer in self.joint_layers:
            h_joint = layer(h_joint, cond_per_step, key_padding_mask=key_padding_mask)
        h_joint = h_joint.reshape(B, self.T_total, J_pad, self.d_model)

        # Joint pooling (masked mean)
        joint_mask_b = joint_mask_pad.unsqueeze(1).unsqueeze(-1)          # [B, 1, J_pad, 1]
        h_pooled = (h_joint * joint_mask_b).sum(dim=2) / joint_mask_b.sum(dim=2).clamp(min=1.0)
        # [B, T_total, d_model]

        # Temporal transformer
        for layer in self.temporal_layers:
            h_pooled = layer(h_pooled, cond)                              # [B, T_total, d_model]

        # Temporal pooling (mean)
        h_final = h_pooled.mean(dim=1)                                    # [B, d_model]

        # Final head
        return self.head(h_final)                                         # [B, 1]

    @staticmethod
    def _pad_joints(x, J_max):
        """Pad joint axis from J_tgt to J_max with zeros. x: [B, T, J_tgt, 13]."""
        B, T, J, F = x.shape
        if J == J_max:
            return x
        pad = torch.zeros(B, T, J_max - J, F, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    D = ACEDiscriminator()
    n = count_parameters(D)
    print(f"Discriminator params: {n:,} ({n/1e6:.1f}M)")

    B = 4
    x_prev = torch.randn(B, 32, 79, 13)  # Horse-sized
    x_cur = torch.randn(B, 32, 79, 13)
    tgt_id = torch.randint(0, 70, (B,))
    tgt_g = torch.randn(B, 128)
    joint_mask = torch.zeros(B, J_MAX)
    joint_mask[:, :79] = 1.0  # Horse has 79 joints, rest padded

    d_logit = D(x_prev, x_cur, tgt_id, tgt_g, joint_mask)
    print(f"d_logit shape: {d_logit.shape} (expect [{B}, 1])")
    print(f"σ(d_logit): {torch.sigmoid(d_logit).squeeze().tolist()}")

    # Test gradient flow for R1 (D should produce gradients w.r.t. input)
    x_real = torch.randn(B, 32, 79, 13, requires_grad=True)
    x_real_prev = torch.randn(B, 32, 79, 13, requires_grad=True)
    d_real = D(x_real_prev, x_real, tgt_id, tgt_g, joint_mask)
    grad_outputs = torch.ones_like(d_real)
    grad_real = torch.autograd.grad(d_real, [x_real_prev, x_real],
                                     grad_outputs=grad_outputs,
                                     create_graph=True, retain_graph=True)
    print(f"R1 grad norm (prev): {grad_real[0].pow(2).sum().sqrt():.4f}")
    print(f"R1 grad norm (cur):  {grad_real[1].pow(2).sum().sqrt():.4f}")
    print(f"R1 grad does propagate (create_graph=True works): OK")
