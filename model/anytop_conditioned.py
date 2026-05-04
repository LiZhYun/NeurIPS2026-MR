import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from torch import Tensor

from model.anytop import AnyTop, create_sin_embedding
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder
from model.motion_encoder import MotionEncoder


class ConditionedGraphMotionDecoderLayer(GraphMotionDecoderLayer):
    """Extends GraphMotionDecoderLayer with a cross-attention sub-layer after spatial attention.

    Decoder tokens attend to the encoder's latent z, giving each joint access to
    the skeleton-agnostic motion signal while preserving all topology conditioning.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.cross_attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm_cross   = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

    def _cross_attn_block(self, x, z_embed):
        # x:       [T+1, B, J, D]
        # z_embed: [B, N', K, D']
        T1, B, J, D = x.shape
        Np, K = z_embed.shape[1], z_embed.shape[2]

        # Add temporal PE to z so decoder can align to coarse temporal positions
        positions = torch.arange(Np, device=z_embed.device).view(1, Np, 1).float()
        z_pe = z_embed + create_sin_embedding(positions, D).unsqueeze(2)  # [1, Np, 1, D] broadcasts

        # [B, N', K, D] → [N'*K, B, D]
        z_kv = z_pe.permute(1, 2, 0, 3).reshape(Np * K, B, D)

        # [T+1, B, J, D] → [T+1, J, B, D] → [(T+1)*J, B, D]
        q = x.permute(0, 2, 1, 3).reshape(T1 * J, B, D)

        out, _ = self.cross_attn(q, z_kv, z_kv)   # [(T+1)*J, B, D]

        # [(T+1)*J, B, D] → [T+1, J, B, D] → [T+1, B, J, D]
        out = out.view(T1, J, B, D).permute(0, 2, 1, 3)
        return self.dropout_cross(out)

    def forward(self,
                tgt: Tensor,
                timesteps_emb: Tensor,
                topology_rel: Tensor,
                edge_rel: Tensor,
                edge_key_emb,
                edge_query_emb,
                edge_value_emb,
                topo_key_emb,
                topo_query_emb,
                topo_value_emb,
                spatial_mask: Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                y=None,
                z_embed: Optional[Tensor] = None) -> Tensor:
        x  = tgt
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        x = self.norm1(x + self._spatial_mha_block(
            x, topology_rel, edge_rel,
            edge_key_emb, edge_query_emb, edge_value_emb,
            topo_key_emb, topo_query_emb, topo_value_emb,
            spatial_mask, tgt_key_padding_mask, y))
        if z_embed is not None:
            x = self.norm_cross(x + self._cross_attn_block(x, z_embed))
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, tgt_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x


class ConditionedGraphMotionDecoder(GraphMotionDecoder):
    """Extends GraphMotionDecoder to thread z_embed through every layer."""

    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor,
                spatial_mask: Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                y=None, get_layer_activation=-1,
                z_embed: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, dict]]:
        topology_rel = y['graph_dist'].long().to(tgt.device)
        edge_rel     = y['joints_relations'].long().to(tgt.device)
        output = tgt
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = dict()
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb     = self.edge_value_emb     if self.value_emb_flag else None
            topology_value_emb = self.topology_value_emb if self.value_emb_flag else None
            output = mod(
                output, timesteps_embs, topology_rel, edge_rel,
                self.edge_key_emb, self.edge_query_emb, edge_value_emb,
                self.topology_key_emb, self.topology_query_emb, topology_value_emb,
                spatial_mask, temporal_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                y, z_embed=z_embed)
            if layer_ind == get_layer_activation:
                activations[layer_ind] = output.clone()
        if self.norm is not None:
            output = self.norm(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output


class AnyTopConditioned(AnyTop):
    """AnyTop extended with a source motion encoder and cross-attention conditioning.

    Training (self-supervised): encode X on S → z → decode X on S
    Inference:                  encode X on S_src → z → decode on S_tgt
    """
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", t5_out_dim=512, root_input_feats=13,
                 enc_num_queries=4, enc_fsq_dims=4, enc_fsq_levels=5,
                 enc_use_vae=True, enc_d_z=64, enc_mode=None,
                 z_drop_prob=0.1,
                 geom_drop_prob=0.3, geom_jitter_prob=0.2,
                 topo_drop_prob=0.15,
                 z_norm_target=None,
                 no_rest_pe=False,
                 **kargs):
        super().__init__(max_joints, feature_len, latent_dim, ff_size, num_layers,
                         num_heads, dropout, activation, t5_out_dim, root_input_feats, **kargs)

        # Replace decoder with conditioned version
        conditioned_layer = ConditionedGraphMotionDecoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation=activation)
        self.seqTransDecoder = ConditionedGraphMotionDecoder(
            conditioned_layer, num_layers=num_layers, value_emb=self.value_emb)

        # Source motion encoder — same d_model as decoder (no projection needed)
        self.encoder = MotionEncoder(
            feature_len=feature_len,
            d_model=latent_dim,
            num_queries=enc_num_queries,
            num_heads=num_heads,
            fsq_dims=enc_fsq_dims,
            fsq_levels=enc_fsq_levels,
            dropout=dropout,
            use_vae=enc_use_vae,
            d_z=enc_d_z,
            enc_mode=enc_mode,
            no_rest_pe=no_rest_pe)

        # CFG null embedding — initialized with small random values (NOT zeros)
        # Zero init causes z ≈ null_z during training → decoder ignores z
        self.null_z = nn.Parameter(torch.randn(1, 1, enc_num_queries, latent_dim) * 0.02)
        self.z_drop_prob    = z_drop_prob
        # Z normalization: project all z_embed (real + null) to a fixed L2 norm
        # before cross-attention injection.  Eliminates magnitude shortcut where
        # the decoder uses ||z|| as a binary "conditioned vs unconditioned" flag.
        # z_norm_target = sqrt(latent_dim) gives unit-variance per-dimension.
        self.z_norm_target = z_norm_target if z_norm_target is not None else (latent_dim ** 0.5)
        # Geometry dropout: drop/jitter rest-pose geometry to prevent decoder
        # from ignoring z.  Graph structure (R_S, D_S) is NEVER affected.
        self.geom_drop_prob   = geom_drop_prob
        self.geom_jitter_prob = geom_jitter_prob
        # Topology dropout: drop graph structure (R_S, D_S) AND geometry together.
        # When fired, the decoder loses ALL skeleton-specific info and MUST use z.
        self.topo_drop_prob   = topo_drop_prob

    def _apply_geom_dropout(self, tpos_first_frame, skip_mask=None):
        """Apply per-sample geometry dropout or jitter to rest-pose features.

        tpos_first_frame: [1, B, J, D]  (unsqueezed for InputProcess)
        skip_mask: [B] bool, True = skip this sample (already handled by topo dropout)
        Returns tensor of same shape with fine geometry dropped or jittered.
        Graph structure (R_S / D_S) lives elsewhere and is NEVER touched here.
        """
        B = tpos_first_frame.shape[1]
        r = torch.rand(B, device=tpos_first_frame.device)
        if skip_mask is not None:
            r[skip_mask] = 1.0  # force no-op for already-dropped samples
        drop_mask   = r < self.geom_drop_prob
        jitter_mask = (~drop_mask) & (r < self.geom_drop_prob + self.geom_jitter_prob)

        out = tpos_first_frame.clone()
        if drop_mask.any():
            out[:, drop_mask] = 0.0
        if jitter_mask.any():
            noise = torch.randn_like(out[:, jitter_mask]) * 0.1
            out[:, jitter_mask] = out[:, jitter_mask] + noise
        return out

    def _apply_topo_dropout(self, y, topo_drop_mask):
        """Replace topology (graph_dist, joints_relations) with neutral values for dropped samples.

        When topology is dropped, the decoder loses all skeleton-specific graph structure.
        Combined with geometry dropout, this forces complete reliance on z.

        topo_drop_mask: [B] bool tensor, True = drop topology for this sample
        """
        if not topo_drop_mask.any():
            return y
        y = dict(y)  # shallow copy to avoid mutating caller's dict
        gd = y['graph_dist'].clone()
        jr = y['joints_relations'].clone()
        # Set dropped samples to neutral: distance=1 (all equally close), relation=0 (no edges)
        gd[topo_drop_mask] = 1
        jr[topo_drop_mask] = 0
        y['graph_dist'] = gd
        y['joints_relations'] = jr
        return y

    def forward(self, x, timesteps, get_layer_activation=-1, y=None):
        # x: [B, J, 13, T]
        joints_mask      = y['joints_mask'].to(x.device)
        temp_mask        = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape

        # --- Conditioning dropout (hierarchical) ---
        # Level 1 (strongest): Topology dropout — drops graph structure + geometry + names
        #   → Decoder has NO skeleton info, must rely entirely on z
        # Level 2: Geometry-only dropout — drops rest-pose features, keeps graph structure
        # Level 3: Geometry jitter — adds noise to rest-pose features
        if self.training:
            topo_drop_mask = torch.rand(bs, device=x.device) < self.topo_drop_prob
            if topo_drop_mask.any():
                y = self._apply_topo_dropout(y, topo_drop_mask)
                # Also zero geometry and names for topo-dropped samples
                tpos_first_frame[:, topo_drop_mask] = 0.0

            # Geometry dropout for non-topo-dropped samples
            if self.geom_drop_prob > 0 or self.geom_jitter_prob > 0:
                remaining = ~topo_drop_mask
                if remaining.any():
                    tpos_first_frame = self._apply_geom_dropout(
                        tpos_first_frame, skip_mask=topo_drop_mask)

        # Retrieve or fall back to null embedding
        z_embed = y.get('z', None)
        if z_embed is not None:
            z_embed = z_embed.to(x.device)
        else:
            z_embed = self.null_z.expand(bs, 1, self.encoder.num_queries, self.latent_dim)

        # CFG dropout: replace z with null for a random fraction of training samples
        if self.training and self.z_drop_prob > 0:
            drop = torch.rand(bs, device=x.device) < self.z_drop_prob   # [B]
            null = self.null_z.expand_as(z_embed)
            z_embed = torch.where(drop[:, None, None, None], null, z_embed)

        # L2-normalize z_embed per token to fixed target norm.
        # Eliminates magnitude shortcut (||z|| >> ||null_z||).
        z_token_norms = z_embed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z_embed = z_embed / z_token_norms * self.z_norm_target

        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind'])

        spatial_mask = 1.0 - joints_mask[:, 0, 0, 1:, 1:]
        spatial_mask = (spatial_mask.unsqueeze(1).unsqueeze(1)
                        .repeat(1, nframes + 1, self.num_heads, 1, 1)
                        .reshape(-1, self.num_heads, njoints, njoints))
        temporal_mask = (1.0 - temp_mask.repeat(1, njoints, self.num_heads, 1, 1)
                         .reshape(-1, nframes + 1, nframes + 1).float())
        spatial_mask[spatial_mask == 1.0]   = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9

        output = self.seqTransDecoder(
            tgt=x, timesteps_embs=timesteps_emb, memory=None,
            spatial_mask=spatial_mask, temporal_mask=temporal_mask,
            y=y, get_layer_activation=get_layer_activation,
            z_embed=z_embed)

        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output      = output[0]
        output = self.output_process(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output
