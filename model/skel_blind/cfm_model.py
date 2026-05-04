"""Target-predictive CFM generator on invariant motion representation.

Architecture: factored slot ⊗ temporal transformer, per spec §2.2.
No skeleton input enters the generator (condition C1).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.skel_blind.slot_vocab import SLOT_COUNT
from model.skel_blind.encoder import CHANNEL_COUNT


class SinusoidalTimestepEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class SlotTemporalBlock(nn.Module):
    """One layer of factored attention: slot-attention then temporal-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.slot_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.slot_norm1 = nn.LayerNorm(d_model)
        self.slot_ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.slot_norm2 = nn.LayerNorm(d_model)

        self.temp_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temp_norm1 = nn.LayerNorm(d_model)
        self.temp_ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.temp_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, S, D]"""
        B, T, S, D = x.shape

        # Slot attention: attend across slots within each frame
        x_flat = x.reshape(B * T, S, D)
        h = self.slot_norm1(x_flat)
        h = x_flat + self.slot_attn(h, h, h, need_weights=False)[0]
        h = h + self.slot_ff(self.slot_norm2(h))
        x = h.reshape(B, T, S, D)

        # Temporal attention: attend across frames within each slot
        x_flat = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        h = self.temp_norm1(x_flat)
        h = x_flat + self.temp_attn(h, h, h, need_weights=False)[0]
        h = h + self.temp_ff(self.temp_norm2(h))
        x = h.reshape(B, S, T, D).permute(0, 2, 1, 3)

        return x


class InvariantCFM(nn.Module):
    """Target-predictive Conditional Flow Matching on [T, 32, 8] invariant reps.

    Predicts clean target x_1 from noisy input x_t conditioned on source rep z_src.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection: concat [x_t, z_src] along channel dim → project
        self.input_proj = nn.Linear(CHANNEL_COUNT * 2, d_model)

        # Timestep embedding
        self.time_emb = SinusoidalTimestepEmb(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Positional embeddings
        self.slot_pos_emb = nn.Embedding(SLOT_COUNT, d_model)
        self.temp_pos_emb = nn.Embedding(max_frames, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SlotTemporalBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, CHANNEL_COUNT)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_src: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, T, S, C] noisy invariant rep at time t
            t: [B] diffusion timestep in [0, 1]
            z_src: [B, T, S, C] source conditioning invariant rep
            cond_mask: [B] bool mask for classifier-free guidance (True = drop condition)

        Returns:
            x_1_pred: [B, T, S, C] predicted clean target
        """
        B, T, S, C = x_t.shape

        if cond_mask is not None:
            z_src = z_src.clone()
            z_src[cond_mask] = 0.0

        h = self.input_proj(torch.cat([x_t, z_src], dim=-1))

        t_emb = self.time_mlp(self.time_emb(t))
        h = h + t_emb[:, None, None, :]

        slot_ids = torch.arange(S, device=h.device)
        frame_ids = torch.arange(T, device=h.device)
        h = h + self.slot_pos_emb(slot_ids)[None, None, :, :]
        h = h + self.temp_pos_emb(frame_ids)[None, :, None, :]

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        return self.output_proj(h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = InvariantCFM()
    print(f"Parameters: {count_parameters(model):,}")

    B, T, S, C = 2, 40, SLOT_COUNT, CHANNEL_COUNT
    x_t = torch.randn(B, T, S, C)
    t = torch.rand(B)
    z_src = torch.randn(B, T, S, C)

    out = model(x_t, t, z_src)
    print(f"Input: {x_t.shape} → Output: {out.shape}")
    assert out.shape == x_t.shape
    print("Forward pass OK")
