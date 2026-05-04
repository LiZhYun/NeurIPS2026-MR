"""ANCHOR-Label-Flow-Src-Graph: graph-only variant of AL-Flow-Src (Codex R5 final blocker).

Removes per-skel ID embeddings entirely. The model now sees ONLY:
  - source motion latent z_a (8 tokens × codebook_dim)
  - source skeleton graph features (via SkelGraphEncoder, handles any skeleton)
  - target skeleton graph features (via SkelGraphEncoder, handles any skeleton)
  - cluster id (from I-5 RandomForest on source Q-features)
  - exact action id (from manifest filename)
  - flow time q

Why this matters (Codex R5): the previous AL-Flow-Src used `nn.Embedding(n_skels=60, d_skel_id_emb=128)`
for both source and target skeleton ID embeddings. At inference time on test_v3 skeletons (held-out),
the embedding lookup would either fail or use a fallback ID. By removing ID embeddings entirely and
relying only on the graph encoder (which builds features from per-joint properties), the model handles
ANY skeleton including held-out ones with the same architectural pathway.

This is the "natural last counterfactual" Codex demanded. If V5 test_test AUC is still ≈ random
(~0.50) on held-out target skeletons with NO fallback, the structural identifiability claim is airtight.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model.actionbridge.generator import (
    SinusoidalPositionalEncoding, TimeEmbedding,
)


class AnchorLabelFlowSrcGraphGenerator(nn.Module):
    """Graph-only conditional flow-matching generator
    P(z_target | source_motion z_a, source_skel_graph, target_skel_graph, cluster, exact_action).

    NO learned skeleton ID embeddings — handles held-out skeletons via graph features only.
    """

    def __init__(self, codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
                 dim_ff=2048, dropout=0.1, max_seq_len=8,
                 d_graph=128,
                 n_clusters=11, d_cluster_emb=128,
                 n_exact_actions=125, d_exact_emb=128):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token projection (shared between src and tgt)
        self.token_proj = nn.Linear(codebook_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.time_emb = TimeEmbedding(d_model)

        # Token-type embeddings: distinguish src vs tgt (graph features alone may not — token type makes it explicit)
        self.tok_type_emb = nn.Embedding(2, d_model)  # 0 = src, 1 = tgt

        # Target skeleton conditioning — GRAPH ONLY (no ID embedding)
        self.tgt_skel_proj = nn.Sequential(
            nn.Linear(d_graph, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Source skeleton conditioning — GRAPH ONLY (no ID embedding)
        self.src_skel_proj = nn.Sequential(
            nn.Linear(d_graph, d_model),
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

        # NULL src token (learnable) for source CFG-drop
        self.src_null_token = nn.Parameter(torch.randn(max_seq_len, codebook_dim) * 0.02)

        # Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vocab_head = nn.Linear(d_model, codebook_dim)

    def forward(self, z_q, q, tgt_graph,
                src_z, src_graph,
                cluster_id, exact_id,
                cluster_mask=None, exact_mask=None, src_mask=None):
        """
        z_q:          [B, T=8, codebook_dim]
        q:            [B] flow time in [0, 1]
        tgt_graph:    [B, d_graph]   (from SkelGraphEncoder applied to target skeleton)
        src_z:        [B, T=8, codebook_dim]
        src_graph:    [B, d_graph]   (from SkelGraphEncoder applied to source skeleton)
        cluster_id, exact_id: [B] long
        cluster_mask, exact_mask, src_mask: [B] bool, drops to NULL when True
        Returns v:    [B, T=8, codebook_dim]
        """
        B, T, _ = z_q.shape

        if src_mask is not None:
            null_expanded = self.src_null_token.unsqueeze(0).expand(B, -1, -1)
            src_z = torch.where(src_mask.view(B, 1, 1), null_expanded, src_z)

        src_h = self.token_proj(src_z)
        tgt_h = self.token_proj(z_q)
        src_h = self.pos_enc(src_h)
        tgt_h = self.pos_enc(tgt_h)
        src_h = src_h + self.tok_type_emb(torch.zeros(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)
        tgt_h = tgt_h + self.tok_type_emb(torch.ones(B, dtype=torch.long, device=z_q.device)).unsqueeze(1)
        h = torch.cat([src_h, tgt_h], dim=1)

        # Global FiLM conditioning: time + tgt_graph + src_graph + cluster + exact (no skel IDs)
        time_h = self.time_emb(q)
        tgt_skel_h = self.tgt_skel_proj(tgt_graph)
        src_skel_h = self.src_skel_proj(src_graph)

        if cluster_mask is not None:
            cluster_id = torch.where(cluster_mask, torch.zeros_like(cluster_id), cluster_id)
        cluster_h = self.cluster_proj(self.cluster_emb(cluster_id))

        if exact_mask is not None:
            exact_id = torch.where(exact_mask, torch.zeros_like(exact_id), exact_id)
        exact_h = self.exact_proj(self.exact_emb(exact_id))

        cond_sum = (time_h + tgt_skel_h + src_skel_h + cluster_h + exact_h).unsqueeze(1)
        h = h + cond_sum

        h = self.encoder(h)
        target_h = h[:, T:, :]
        return self.vocab_head(target_h)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(42)
    G = AnchorLabelFlowSrcGraphGenerator(
        n_clusters=11, n_exact_actions=125, d_model=512, n_layers=6, n_heads=8)
    print(f"params: {count_parameters(G):,}")
    G.eval()
    z = torch.randn(4, 8, 256)
    q = torch.rand(4)
    tg = torch.randn(4, 128)
    src_z = torch.randn(4, 8, 256)
    sg = torch.randn(4, 128)
    cid = torch.randint(1, 11, (4,))
    eid = torch.randint(1, 125, (4,))

    out_full = G(z, q, tg, src_z, sg, cid, eid)
    print('out shape:', out_full.shape)

    cmask = torch.zeros(4, dtype=torch.bool); cmask[1] = True
    emask = torch.zeros(4, dtype=torch.bool); emask[2] = True
    smask = torch.zeros(4, dtype=torch.bool); smask[3] = True
    out_drop = G(z, q, tg, src_z, sg, cid, eid,
                 cluster_mask=cmask, exact_mask=emask, src_mask=smask)
    assert torch.allclose(out_full[0], out_drop[0])
    assert not torch.allclose(out_full[1], out_drop[1])
    assert not torch.allclose(out_full[2], out_drop[2])
    assert not torch.allclose(out_full[3], out_drop[3])
    print('all 3 CFG-drop assertions PASS — graph-only generator semantics verified.')
