"""ACE Generator: predicts target latent given source motion + previous target context.

Per ACE_DESIGN_V3 §3.1:
  Input:  z_src [B, 8, 256], prev_z_tgt [B, 8, 256], src/tgt skel id, src/tgt graph
  Output: z_pred [B, 8, 256] (target latent for current chunk)

Architecture:
  - Concat (z_src, prev_z_tgt) → [B, 16, 256] sequence
  - Positional embedding distinguishes src-half (positions 0-7) from prev-tgt-half (8-15)
  - Skel/graph conditioning: additive at input + FiLM at every layer
  - 6×8×512 transformer encoder
  - Output head returns 8 tokens (positions 8-15 = the "predicted target" half)

START token: when prev_z_tgt unavailable (clip start), substitute target-conditioned START
              vector. See ACEStartTokens module.
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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class ACEStartTokens(nn.Module):
    """Target-conditioned START tokens for clip-start chunks.

    START[tgt_skel] = mean(z_real_target_pool[tgt_skel]) + learned_offset[tgt_skel]

    For inductive ablation: held-out skels (id >= n_train_skels) get zero offset.
    Pre-computed mean must be passed at construction (computed once from cache).

    Args:
        n_skels: total number of skel IDs (70 for transductive, 70 for inductive but only first
                 n_train_skels have meaningful learned_offset)
        n_train_skels: number of skels actually trained (60 for inductive, 70 for transductive)
        codebook_dim: 256
        n_tokens: 8
        z_means: Tensor[n_skels, n_tokens, codebook_dim] — pre-computed per-skel mean of real z_tgt
    """
    def __init__(self, n_skels, n_train_skels, codebook_dim, n_tokens, z_means):
        super().__init__()
        assert z_means.shape == (n_skels, n_tokens, codebook_dim), \
            f"z_means shape {z_means.shape} != ({n_skels}, {n_tokens}, {codebook_dim})"
        self.register_buffer('z_means', z_means.float())                  # frozen pre-computed
        self.n_skels = n_skels
        self.n_train_skels = n_train_skels
        # Learned offset per skel; init to zeros so START = mean at start
        self.offset = nn.Parameter(torch.zeros(n_skels, n_tokens, codebook_dim))

    def forward(self, tgt_skel_id):
        """tgt_skel_id: [B] LongTensor → returns [B, n_tokens, codebook_dim]."""
        mean = self.z_means[tgt_skel_id]                                  # [B, n_tokens, d]
        offset = self.offset[tgt_skel_id]                                 # [B, n_tokens, d]
        # Mask offset to zero for held-out (inductive) skels — id >= n_train_skels
        held_out = (tgt_skel_id >= self.n_train_skels).unsqueeze(-1).unsqueeze(-1)
        offset = torch.where(held_out, torch.zeros_like(offset), offset)
        return mean + offset


class ACEGenerator(nn.Module):
    """Window-level autoregressive generator: G(z_src, prev_z_tgt, skels, graphs) → z_pred."""

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, n_tokens=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Token projection
        self.token_proj = nn.Linear(codebook_dim, d_model)
        # Positional encoding over 16-token sequence (8 src + 8 prev_tgt)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=2 * n_tokens)
        # Segment embedding: 0 = src half, 1 = prev_tgt half
        self.segment_emb = nn.Embedding(2, d_model)

        # Skel + graph conditioning (additive at input; same as MoReFlow)
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(2 * (d_skel_id_emb + d_graph), d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: project back to codebook_dim for the predicted target tokens
        self.out_proj = nn.Linear(d_model, codebook_dim)

    def forward(self, z_src, prev_z_tgt, src_skel_id, tgt_skel_id, src_graph, tgt_graph):
        """
        z_src:        [B, n_tokens=8, codebook_dim=256]
        prev_z_tgt:   [B, n_tokens=8, codebook_dim=256]
        src_skel_id:  [B]
        tgt_skel_id:  [B]
        src_graph:    [B, d_graph=128]
        tgt_graph:    [B, d_graph=128]

        Returns:
          z_pred:     [B, n_tokens=8, codebook_dim=256]
        """
        B = z_src.shape[0]
        T = self.n_tokens

        # Concat src + prev_tgt along token axis → [B, 16, codebook_dim]
        seq = torch.cat([z_src, prev_z_tgt], dim=1)                       # [B, 16, 256]
        h = self.token_proj(seq)                                          # [B, 16, 512]

        # Positional + segment embeddings
        h = self.pos_enc(h)
        seg_ids = torch.cat([
            torch.zeros(T, device=h.device, dtype=torch.long),
            torch.ones(T, device=h.device, dtype=torch.long),
        ])                                                                # [16]
        seg = self.segment_emb(seg_ids).unsqueeze(0)                      # [1, 16, 512]
        h = h + seg

        # Skel + graph additive conditioning
        src_id_emb = self.skel_id_emb(src_skel_id)                        # [B, 128]
        tgt_id_emb = self.skel_id_emb(tgt_skel_id)
        skel_in = torch.cat([src_id_emb, src_graph, tgt_id_emb, tgt_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)                                  # [B, 512]
        h = h + skel_h.unsqueeze(1)                                       # broadcast over 16 tokens

        # Transformer encoder
        h = self.encoder(h)                                               # [B, 16, 512]

        # Output: take the prev_tgt half (positions 8-15) and project to codebook_dim
        # Rationale: tokens 8-15 attended to all of src + prev_tgt; they are the "predicted target".
        h_target = h[:, T:, :]                                            # [B, 8, 512]
        z_pred = self.out_proj(h_target)                                  # [B, 8, 256]
        return z_pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    G = ACEGenerator()
    n = count_parameters(G)
    print(f"Generator params: {n:,} ({n/1e6:.1f}M)")

    B = 4
    z_src = torch.randn(B, 8, 256)
    prev_z_tgt = torch.randn(B, 8, 256)
    src_id = torch.randint(0, 70, (B,))
    tgt_id = torch.randint(0, 70, (B,))
    src_g = torch.randn(B, 128)
    tgt_g = torch.randn(B, 128)
    z_pred = G(z_src, prev_z_tgt, src_id, tgt_id, src_g, tgt_g)
    print(f"z_pred shape: {z_pred.shape} (expect [{B}, 8, 256])")
    print(f"z_pred stats: mean={z_pred.mean():.4f}, std={z_pred.std():.4f}")

    # Test ACEStartTokens
    z_means = torch.randn(70, 8, 256)
    starts = ACEStartTokens(n_skels=70, n_train_skels=60, codebook_dim=256, n_tokens=8, z_means=z_means)
    n_starts = sum(p.numel() for p in starts.parameters() if p.requires_grad)
    print(f"ACEStartTokens learnable params: {n_starts:,}  (expected 70*8*256={70*8*256:,})")
    test_ids = torch.tensor([5, 10, 65, 30])  # 5,10,30 in train; 65 held-out
    out = starts(test_ids)
    print(f"START output shape: {out.shape} (expect [4, 8, 256])")
    # Held-out should equal mean (offset masked to zero)
    held_out_diff = (out[2] - z_means[65]).abs().max().item()
    print(f"Held-out (id=65) diff from pure mean: {held_out_diff:.6e} (expect ~0)")
