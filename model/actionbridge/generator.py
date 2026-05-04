"""VQ-ActionBridge generator: conditional flow matching on per-skel VQ latent space.

Architecture (adapted from MoReFlow Stage B DiscreteFlowTransformer):
  Input:  z_q [B, T=8, codebook_dim=256] interpolant
          q [B] flow time
          tgt_skel_id [B], tgt_graph [B, d_graph]
          action_id [B] ∈ {0..N_clusters-1}
  Output: v_psi [B, T=8, codebook_dim=256] continuous velocity

Differences from MoReFlow Stage B:
  - DROPS src_skel_id, src_graph (we don't use source coupling)
  - DROPS phi-feature conditioning
  - ADDS action embedding as primary conditioning
  - Supports CFG via action-dropout (not phi-dropout)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


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


class TimeEmbedding(nn.Module):
    def __init__(self, d_model, max_period=1000.0):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        half = d_model // 2
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)

    def forward(self, q):
        args = q.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class ActionBridgeGenerator(nn.Module):
    """Conditional flow matching generator: P(z_target | action, skel_b)."""

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128,
                 n_actions=10, d_action_emb=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.time_emb = TimeEmbedding(d_model)

        # Target skel conditioning (drop src side from MoReFlow)
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Action conditioning (n_actions includes a NULL idx at index 0 for CFG drop)
        # Class indices 1..n_actions-1 are real actions
        self.action_emb = nn.Embedding(n_actions, d_action_emb)
        self.action_proj = nn.Sequential(
            nn.Linear(d_action_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_skel_id, tgt_graph, action_id, action_mask=None):
        """
        z_q:          [B, T=8, codebook_dim]
        q:            [B] flow time ∈ [0, 1]
        tgt_skel_id:  [B]
        tgt_graph:    [B, d_graph]
        action_id:    [B] ∈ {0..n_actions-1}, 0 = NULL/CFG-drop
        action_mask:  [B] bool — True = drop action conditioning (CFG drop). Sets action to NULL.
        Returns v: [B, T=8, codebook_dim]
        """
        B, T, _ = z_q.shape
        h = self.token_proj(z_q)
        h = self.pos_enc(h)

        time_h = self.time_emb(q)
        skel_in = torch.cat([self.skel_id_emb(tgt_skel_id), tgt_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)

        if action_mask is not None:
            action_id = torch.where(action_mask, torch.zeros_like(action_id), action_id)
        act_h = self.action_proj(self.action_emb(action_id))

        cond_sum = (time_h + skel_h + act_h).unsqueeze(1)
        h = h + cond_sum
        h = self.encoder(h)
        return self.vocab_head(h)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    g = ActionBridgeGenerator(n_skels=70, n_actions=11)
    print(f"params: {count_parameters(g):,}")
    z = torch.randn(4, 8, 256)
    q = torch.rand(4)
    tid = torch.randint(0, 70, (4,))
    tg = torch.randn(4, 128)
    aid = torch.randint(0, 11, (4,))
    out = g(z, q, tid, tg, aid)
    print('out shape:', out.shape)
