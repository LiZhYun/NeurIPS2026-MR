"""ANCHOR-Label-Flow generator: ActionBridge + exact-action conditioning.

This is the supervision-matched generative comparator to ANCHOR. ANCHOR uses
both the coarse action-cluster label and the fine exact-action label from the
dataset filenames; this generator gets both labels through FiLM-style
conditioning. If it still loses to ANCHOR on V5 cluster-tier AUC, the failure
is structural (Theorem 1 / Proposition 1), not metadata-availability.

Architecture:
  - Re-uses ActionBridge's conditional flow-matching backbone in per-skeleton
    VQ latent space (z in R^[8,256]).
  - Adds an exact-action embedding alongside the cluster embedding. CFG
    dropout drops the exact-action label independently from the cluster label.

Differences from ActionBridge generator.py:
  - Adds nn.Embedding(n_exact_actions, d_exact_emb) and a separate projection.
  - forward() takes both action_id (cluster) and exact_id (exact action), with
    independent CFG drop masks.
  - Cluster + exact conditioning are summed into the same scalar cond vector
    that the original generator already adds to every token.

Usage:
    from model.actionbridge.generator_anchor import AnchorLabelFlowGenerator
    G = AnchorLabelFlowGenerator(n_skels=70, n_clusters=10+1, n_exact_actions=90+1,
                                  d_model=512, n_layers=6, n_heads=8)
    v = G(z_q, q, tgt_skel_id, tgt_graph,
          cluster_id, exact_id,
          cluster_mask=cdrop, exact_mask=edrop)

Both NULL labels are at index 0; supplying NULL via mask is equivalent to
unconditional generation on that channel.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from model.actionbridge.generator import (
    ActionBridgeGenerator, SinusoidalPositionalEncoding, TimeEmbedding,
)


class AnchorLabelFlowGenerator(nn.Module):
    """Conditional flow-matching generator P(z_target | cluster, exact_action, skel_b).

    The generator is supervision-matched to ANCHOR: same coarse + fine action
    labels, same target-skeleton conditioning. This is the comparator that
    isolates label-availability from structural identifiability failure.
    """

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128,
                 n_clusters=11, d_cluster_emb=128,
                 n_exact_actions=125, d_exact_emb=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Tokens + position
        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.time_emb = TimeEmbedding(d_model)

        # Target skeleton conditioning (same as ActionBridge)
        self.skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Coarse-action (cluster) embedding — index 0 = NULL/CFG-drop
        self.cluster_emb = nn.Embedding(n_clusters, d_cluster_emb)
        self.cluster_proj = nn.Sequential(
            nn.Linear(d_cluster_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Fine exact-action embedding — index 0 = NULL/CFG-drop
        self.exact_emb = nn.Embedding(n_exact_actions, d_exact_emb)
        self.exact_proj = nn.Sequential(
            nn.Linear(d_exact_emb, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_skel_id, tgt_graph,
                cluster_id, exact_id,
                cluster_mask=None, exact_mask=None):
        """
        z_q:          [B, T=8, codebook_dim]
        q:            [B] flow time in [0, 1]
        tgt_skel_id:  [B] long
        tgt_graph:    [B, d_graph]
        cluster_id:   [B] long, in {0..n_clusters-1} (0 = NULL)
        exact_id:     [B] long, in {0..n_exact_actions-1} (0 = NULL)
        cluster_mask: [B] bool, True drops cluster to NULL (CFG drop)
        exact_mask:   [B] bool, True drops exact action to NULL (CFG drop)
        Returns v:    [B, T=8, codebook_dim]
        """
        B, T, _ = z_q.shape
        h = self.token_proj(z_q)
        h = self.pos_enc(h)

        time_h = self.time_emb(q)
        skel_in = torch.cat([self.skel_id_emb(tgt_skel_id), tgt_graph], dim=-1)
        skel_h = self.skel_proj(skel_in)

        if cluster_mask is not None:
            cluster_id = torch.where(cluster_mask, torch.zeros_like(cluster_id), cluster_id)
        cluster_h = self.cluster_proj(self.cluster_emb(cluster_id))

        if exact_mask is not None:
            exact_id = torch.where(exact_mask, torch.zeros_like(exact_id), exact_id)
        exact_h = self.exact_proj(self.exact_emb(exact_id))

        # All conditioning is broadcast-added to every token (FiLM-style global cond).
        cond_sum = (time_h + skel_h + cluster_h + exact_h).unsqueeze(1)
        h = h + cond_sum

        h = self.encoder(h)
        return self.vocab_head(h)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(42)
    G = AnchorLabelFlowGenerator(n_skels=70, n_clusters=11, n_exact_actions=125,
                                  d_model=512, n_layers=6, n_heads=8)
    print(f"params: {count_parameters(G):,}")
    G.eval()
    z = torch.randn(4, 8, 256)
    q = torch.rand(4)
    tid = torch.randint(0, 70, (4,))
    tg = torch.randn(4, 128)
    cid = torch.randint(1, 11, (4,))   # avoid NULL=0 to make drop semantics meaningful
    eid = torch.randint(1, 125, (4,))
    out_full = G(z, q, tid, tg, cid, eid)
    print('out shape:', out_full.shape)
    cmask = torch.zeros(4, dtype=torch.bool); cmask[1] = True
    emask = torch.zeros(4, dtype=torch.bool); emask[2] = True
    out_drop = G(z, q, tid, tg, cid, eid, cluster_mask=cmask, exact_mask=emask)

    # Per code-review (2026-04-26): assert NULL-drop semantics actually exercise the path.
    assert torch.allclose(out_full[0], out_drop[0]), "row 0 (no drop) must be identical"
    assert torch.allclose(out_full[3], out_drop[3]), "row 3 (no drop) must be identical"
    assert not torch.allclose(out_full[1], out_drop[1]), "row 1 cluster_drop must differ"
    assert not torch.allclose(out_full[2], out_drop[2]), "row 2 exact_drop must differ"
    cid2 = cid.clone(); cid2[1] = 0
    out_id_eq = G(z, q, tid, tg, cid2, eid)
    assert torch.allclose(out_drop[1], out_id_eq[1]), "cluster_mask=True ≡ cluster_id=0"
    eid2 = eid.clone(); eid2[2] = 0
    out_id_eq2 = G(z, q, tid, tg, cid, eid2)
    assert torch.allclose(out_drop[2], out_id_eq2[2]), "exact_mask=True ≡ exact_id=0"
    print('cfg-drop assertions PASS — NULL-drop semantics verified.')
