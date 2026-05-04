"""ActionBridge v2 generator: source-conditioned flow matching on per-skel VQ z.

Differences from v1:
  - Replace `action_emb` lookup with `behavior_tokens` from encoder (shape [B, 8, 256])
  - Cross-attention: z queries attend to behavior_tokens
  - Skel-id NULL support (set tgt_skel_id_emb=0 when held-out)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


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
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class ActionBridgeGeneratorV2(nn.Module):
    """Cross-attention generator: z attends to behavior_tokens for source-conditioning."""

    def __init__(self, codebook_dim=256, d_model=384, n_layers=6, n_heads=8,
                 dim_ff=1024, dropout=0.1, n_tokens=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128, d_behavior=256):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.n_tokens = n_tokens

        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        self.time_emb = TimeEmbedding(d_model)

        # Target skel conditioning
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Behavior token projection (from encoder output dim)
        self.behavior_proj = nn.Linear(d_behavior, d_model)

        # Self-attention layers (Tx Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=1)
            for _ in range(n_layers)
        ])
        # Cross-attention layers (z queries → behavior keys/values)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_skel_id, tgt_graph, behavior_tokens, drop_skel=False, drop_behavior=False):
        """
        z_q:               [B, 8, codebook_dim]  flow interpolant
        q:                 [B]                    flow time
        tgt_skel_id:       [B]
        tgt_graph:         [B, d_graph]
        behavior_tokens:   [B, 8, d_behavior]    from encoder
        drop_skel:         bool — set skel_id_emb=0 (for inductive)
        drop_behavior:     bool — set behavior_tokens=0 (for CFG)
        """
        B, T, _ = z_q.shape
        h = self.token_proj(z_q)
        h = h + self.pos_enc.unsqueeze(0)

        # Time
        time_h = self.time_emb(q)

        # Skel conditioning
        sid_emb = self.skel_id_emb(tgt_skel_id)
        if drop_skel:
            sid_emb = torch.zeros_like(sid_emb)
        skel_in = torch.cat([sid_emb, tgt_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)

        h = h + (time_h + skel_h).unsqueeze(1)

        # Behavior tokens (CFG drop)
        bt = behavior_tokens
        if drop_behavior:
            bt = torch.zeros_like(bt)
        bt_proj = self.behavior_proj(bt)                        # [B, 8, d_model]

        # Interleaved self-attention + cross-attention layers
        for sa, ca, cn in zip(self.self_attn_layers, self.cross_attn_layers, self.cross_norms):
            h = sa(h)
            # Cross-attention: z queries attend to behavior_tokens
            cross_out, _ = ca(cn(h), bt_proj, bt_proj, need_weights=False)
            h = h + cross_out

        return self.vocab_head(h)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    g = ActionBridgeGeneratorV2(n_skels=70)
    print(f"params: {count_parameters(g)/1e6:.1f}M")
    z = torch.randn(4, 8, 256); q = torch.rand(4)
    tid = torch.randint(0, 70, (4,)); tg = torch.randn(4, 128)
    bt = torch.randn(4, 8, 256)
    out = g(z, q, tid, tg, bt)
    print(f"out: {out.shape}")
