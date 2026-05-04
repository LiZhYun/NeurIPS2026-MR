"""ActionBridge v2 encoder: motion + skel → behavior_tokens [8, 256].

Per refine-logs/ACTIONBRIDGE_V2_DESIGN.md.

Architecture:
  Input:  z_src [B, T_src, 8, 256] — source z-tokens from MoReFlow Stage A (already encoded)
          src_skel_id [B], src_graph [B, 128]
  Output: behavior_tokens [B, 8, 256] — skel-invariant action-bearing
          + action_logits [B, n_clusters] — auxiliary classifier

Motion enters as VQ z-tokens (already extracted by MoReFlow Stage A — same as generator's target).
Encoder operates on z-token sequence, conditioning on src skel features.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class ActionBridgeEncoder(nn.Module):
    def __init__(self, codebook_dim=256, d_model=384, n_layers=4, n_heads=8,
                 dim_ff=1024, dropout=0.1, n_tokens=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128, n_clusters=11):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Project z-tokens
        self.token_proj = nn.Linear(codebook_dim, d_model)
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.zeros(n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Source skel conditioning
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Encoder transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output: 8 behavior tokens (project back to codebook_dim for compatibility with generator)
        self.behavior_proj = nn.Linear(d_model, codebook_dim)

        # Auxiliary action classifier (mean-pool over tokens)
        self.action_head = nn.Linear(d_model, n_clusters)

    def forward(self, z_src, src_skel_id, src_graph, drop_skel=False):
        """
        z_src:        [B, n_tokens=8, codebook_dim=256]
        src_skel_id:  [B]
        src_graph:    [B, d_graph]
        drop_skel:    if True, zero out skel_id_emb (for inductive)

        Returns:
          behavior_tokens: [B, n_tokens=8, codebook_dim=256]
          action_logits:   [B, n_clusters]
        """
        B = z_src.shape[0]
        h = self.token_proj(z_src)                          # [B, 8, d_model]
        h = h + self.pos_enc.unsqueeze(0)

        # Skel conditioning (additive)
        sid_emb = self.skel_id_emb(src_skel_id)
        if drop_skel:
            sid_emb = torch.zeros_like(sid_emb)
        skel_in = torch.cat([sid_emb, src_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)                    # [B, d_model]
        h = h + skel_h.unsqueeze(1)

        h = self.encoder(h)                                 # [B, 8, d_model]

        behavior_tokens = self.behavior_proj(h)             # [B, 8, codebook_dim]
        action_logits = self.action_head(h.mean(dim=1))     # [B, n_clusters]
        return behavior_tokens, action_logits


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    e = ActionBridgeEncoder(n_skels=70, n_clusters=11)
    print(f"params: {count_parameters(e)/1e6:.1f}M")
    z = torch.randn(4, 8, 256)
    sid = torch.randint(0, 70, (4,))
    sg = torch.randn(4, 128)
    bt, al = e(z, sid, sg)
    print(f"behavior_tokens: {bt.shape}, action_logits: {al.shape}")
