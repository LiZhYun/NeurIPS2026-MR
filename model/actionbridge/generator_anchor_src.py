"""ANCHOR-Label-Flow-Src: source-conditioned generator (Codex R4 final blocker).

Extends AnchorLabelFlowGenerator with full source-motion conditioning:
  - source z_a (per-skel VQ latents [T, codebook_dim]) projected and concatenated
    with target tokens; transformer encoder sees both as a 2T-length sequence
    with source-token-type / target-token-type embeddings
  - source skeleton id + source graph features added as global FiLM conditioning
    (matching target's skel + graph treatment)
  - separate CFG dropout for source (tests "what if source is hidden?")

Architecture:
  16-token sequence = 8 src tokens || 8 tgt tokens
  global FiLM cond = time + tgt_skel + src_skel + cluster + exact
  output head = vocab_head applied to the last 8 (target) positions only

This is the supervision-matched generator that takes EVERY input ANCHOR's
retrieval-side has access to (predicted cluster + exact action + source motion +
source skeleton graph + target skeleton graph). If this generator still scores in
the random regime on V5, the metadata-availability defense is fully refuted.

Per Codex R3: also addresses the "70-skeleton or held-out-target-capable" requirement
by using the skel-graph encoder for both sides (not skeleton ID embeddings as the
sole representation), so held-out skeletons fall back gracefully.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model.actionbridge.generator import (
    SinusoidalPositionalEncoding, TimeEmbedding,
)


class AnchorLabelFlowSrcGenerator(nn.Module):
    """Conditional flow-matching generator
    P(z_target | source_motion z_a, source_skel, target_skel, cluster, exact_action).
    """

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 n_skels=70, d_skel_id_emb=128, d_graph=128,
                 n_clusters=11, d_cluster_emb=128,
                 n_exact_actions=125, d_exact_emb=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len  # tokens per skel (8 for MoReFlow Stage A)

        # Token projection (shared between src and tgt)
        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.time_emb = TimeEmbedding(d_model)

        # Token-type embeddings: distinguish src vs tgt within the concatenated sequence
        self.tok_type_emb = nn.Embedding(2, d_model)  # 0 = src, 1 = tgt

        # Target skeleton conditioning
        self.tgt_skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.tgt_skel_proj = nn.Sequential(
            nn.Linear(d_skel_id_emb + d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Source skeleton conditioning (separate embedding, same graph_dim)
        self.src_skel_id_emb = nn.Embedding(n_skels, d_skel_id_emb)
        self.src_skel_proj = nn.Sequential(
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

        # NULL src token (learnable, used when src is dropped via CFG)
        self.src_null_token = nn.Parameter(torch.randn(max_seq_len, codebook_dim) * 0.02)

        # Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_skel_id, tgt_graph,
                src_z, src_skel_id, src_graph,
                cluster_id, exact_id,
                cluster_mask=None, exact_mask=None, src_mask=None):
        """
        z_q:          [B, T=8, codebook_dim]  noisy target latent at flow time q
        q:            [B] flow time in [0, 1]
        tgt_skel_id:  [B] long
        tgt_graph:    [B, d_graph]
        src_z:        [B, T=8, codebook_dim]  source motion latent (clean)
        src_skel_id:  [B] long
        src_graph:    [B, d_graph]
        cluster_id:   [B] long, in {0..n_clusters-1} (0 = NULL)
        exact_id:     [B] long, in {0..n_exact_actions-1} (0 = NULL)
        cluster_mask: [B] bool, True drops cluster to NULL
        exact_mask:   [B] bool, True drops exact action to NULL
        src_mask:     [B] bool, True drops source to NULL token (src CFG-drop)
        Returns v:    [B, T=8, codebook_dim]  predicted velocity for target tokens
        """
        B, T, _ = z_q.shape

        # Source CFG drop: replace dropped source samples with NULL token
        if src_mask is not None:
            null_expanded = self.src_null_token.unsqueeze(0).expand(B, -1, -1)
            src_z = torch.where(src_mask.view(B, 1, 1), null_expanded, src_z)

        # Project both src and tgt tokens; add per-position positional encoding;
        # add token-type embeddings (src=0, tgt=1)
        src_h = self.token_proj(src_z)
        tgt_h = self.token_proj(z_q)
        src_h = self.pos_enc(src_h)
        tgt_h = self.pos_enc(tgt_h)
        src_h = src_h + self.tok_type_emb(torch.zeros(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)
        tgt_h = tgt_h + self.tok_type_emb(torch.ones(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)

        # Concatenate: [src, tgt] along token dim → [B, 2T, d_model]
        h = torch.cat([src_h, tgt_h], dim=1)

        # Global FiLM conditioning: time + tgt_skel + src_skel + cluster + exact
        time_h = self.time_emb(q)

        tgt_skel_in = torch.cat([self.tgt_skel_id_emb(tgt_skel_id), tgt_graph], dim=-1)
        tgt_skel_h = self.tgt_skel_proj(tgt_skel_in)

        src_skel_in = torch.cat([self.src_skel_id_emb(src_skel_id), src_graph], dim=-1)
        src_skel_h = self.src_skel_proj(src_skel_in)

        if cluster_mask is not None:
            cluster_id = torch.where(cluster_mask, torch.zeros_like(cluster_id), cluster_id)
        cluster_h = self.cluster_proj(self.cluster_emb(cluster_id))

        if exact_mask is not None:
            exact_id = torch.where(exact_mask, torch.zeros_like(exact_id), exact_id)
        exact_h = self.exact_proj(self.exact_emb(exact_id))

        cond_sum = (time_h + tgt_skel_h + src_skel_h + cluster_h + exact_h).unsqueeze(1)
        h = h + cond_sum  # broadcast over all 2T tokens

        h = self.encoder(h)

        # Output head only on target half (last T tokens)
        target_h = h[:, T:, :]  # [B, T, d_model]
        return self.vocab_head(target_h)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(42)
    G = AnchorLabelFlowSrcGenerator(n_skels=70, n_clusters=11, n_exact_actions=125,
                                     d_model=512, n_layers=6, n_heads=8)
    print(f"params: {count_parameters(G):,}")
    G.eval()
    z = torch.randn(4, 8, 256)
    q = torch.rand(4)
    tid = torch.randint(0, 70, (4,))
    tg = torch.randn(4, 128)
    src_z = torch.randn(4, 8, 256)
    sid = torch.randint(0, 70, (4,))
    sg = torch.randn(4, 128)
    cid = torch.randint(1, 11, (4,))
    eid = torch.randint(1, 125, (4,))

    out_full = G(z, q, tid, tg, src_z, sid, sg, cid, eid)
    print('out shape:', out_full.shape)

    # CFG-drop semantics tests
    cmask = torch.zeros(4, dtype=torch.bool); cmask[1] = True
    emask = torch.zeros(4, dtype=torch.bool); emask[2] = True
    smask = torch.zeros(4, dtype=torch.bool); smask[3] = True
    out_drop = G(z, q, tid, tg, src_z, sid, sg, cid, eid,
                 cluster_mask=cmask, exact_mask=emask, src_mask=smask)
    assert torch.allclose(out_full[0], out_drop[0]), "row 0 (no drop) must be identical"
    assert not torch.allclose(out_full[1], out_drop[1]), "row 1 cluster_drop must differ"
    assert not torch.allclose(out_full[2], out_drop[2]), "row 2 exact_drop must differ"
    assert not torch.allclose(out_full[3], out_drop[3]), "row 3 src_drop must differ"
    cid2 = cid.clone(); cid2[1] = 0
    out_id_eq = G(z, q, tid, tg, src_z, sid, sg, cid2, eid)
    assert torch.allclose(out_drop[1], out_id_eq[1]), "cluster_mask=True ≡ cluster_id=0"
    eid2 = eid.clone(); eid2[2] = 0
    out_id_eq2 = G(z, q, tid, tg, src_z, sid, sg, cid, eid2)
    assert torch.allclose(out_drop[2], out_id_eq2[2]), "exact_mask=True ≡ exact_id=0"
    src_z3 = src_z.clone()
    src_z3[3] = G.src_null_token.detach()
    out_id_eq3 = G(z, q, tid, tg, src_z3, sid, sg, cid, eid)
    assert torch.allclose(out_drop[3], out_id_eq3[3]), "src_mask=True ≡ src_z=null_token"
    print('cfg-drop assertions PASS — NULL-drop semantics verified for all 3 channels.')
