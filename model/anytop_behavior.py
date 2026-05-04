"""AnyTop with Behavior-Conditioning channel.

Replaces AnyTopConditioned's MotionEncoder (which produces unhelpful z per V1/V6) with a
behavior recognizer R(x) = (a(x), ψ(x), [r(x)]) that produces a topology-normalized
behavior representation. The behavior is mapped to per-token cross-attention conditioning.

Key design (per refine-logs/effect_program/EMPIRICAL_PLAN.md after Round 7 fixes):
- a(x): action class embedding (12-class one-hot → 32-dim)
- ψ(x): analytic dynamics (64×62, topology-normalized — extracted by eval/effect_program.py)
- r(x): optional tiny learned residual (≤16 dim, ablated)
- ALL topology-normalized — no per-joint signals leak source skeleton identity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.anytop import AnyTop, create_sin_embedding
from model.anytop_conditioned import (
    ConditionedGraphMotionDecoderLayer,
    ConditionedGraphMotionDecoder,
)


class BehaviorRecognizer(nn.Module):
    """R(x) → behavior representation B = (a, ψ, [r]).

    Inputs:
        psi: precomputed analytic dynamics ψ(x) ∈ [B, 64, 62] (from eval.effect_program)
        action_label: action class index ∈ [B], or None at inference
        source_motion (optional): for r(x) learned residual — only used if use_residual=True

    Outputs:
        behavior_tokens: [B, T_b, d_model] (T_b = number of behavior tokens after projection)
    """
    def __init__(self,
                 d_model=256,
                 n_actions=12,
                 action_emb_dim=32,
                 psi_temporal_frames=64,
                 psi_dim=62,
                 use_residual=False,
                 residual_dim=16,
                 n_behavior_tokens=8):
        super().__init__()
        self.d_model = d_model
        self.use_residual = use_residual
        self.n_behavior_tokens = n_behavior_tokens

        # Action embedding
        self.action_embedding = nn.Embedding(n_actions + 1, action_emb_dim)  # +1 for "unknown"
        self.action_proj = nn.Linear(action_emb_dim, d_model)

        # ψ projection: (64, 62) → (64, d_model)
        self.psi_per_frame_proj = nn.Linear(psi_dim, d_model)
        # Then attention pool ψ time axis to n_behavior_tokens
        self.psi_query = nn.Parameter(torch.randn(n_behavior_tokens, d_model) * 0.02)
        self.psi_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        # Optional residual head — small MLP on flattened motion summary
        if use_residual:
            self.residual_head = nn.Sequential(
                nn.Linear(64 * psi_dim, 128), nn.GELU(),
                nn.Linear(128, residual_dim),
            )
            self.residual_proj = nn.Linear(residual_dim, d_model)

        # Final layer norm to stabilize behavior tokens
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, psi, action_label=None):
        B = psi.shape[0]
        device = psi.device

        # ψ per-frame embed → attention pool to n_behavior_tokens
        psi_emb = self.psi_per_frame_proj(psi)  # [B, 64, d_model]
        # Add temporal positional encoding
        T = psi.shape[1]
        positions = torch.arange(T, device=device).view(1, T, 1).float()
        psi_emb = psi_emb + create_sin_embedding(positions, self.d_model)
        # Attention pool
        queries = self.psi_query.unsqueeze(0).expand(B, -1, -1)  # [B, n_tokens, d_model]
        psi_pooled, _ = self.psi_attn(queries, psi_emb, psi_emb, need_weights=False)
        # psi_pooled: [B, n_behavior_tokens, d_model]

        # Action token (one extra token)
        if action_label is None:
            action_label = torch.full((B,), self.action_embedding.num_embeddings - 1,
                                      dtype=torch.long, device=device)
        action_emb = self.action_embedding(action_label)  # [B, action_emb_dim]
        action_token = self.action_proj(action_emb).unsqueeze(1)  # [B, 1, d_model]

        # Optional residual token
        if self.use_residual:
            psi_flat = psi.reshape(B, -1)  # [B, 64*62]
            r = self.residual_head(psi_flat)  # [B, residual_dim]
            r_token = self.residual_proj(r).unsqueeze(1)  # [B, 1, d_model]
            tokens = torch.cat([action_token, psi_pooled, r_token], dim=1)
        else:
            tokens = torch.cat([action_token, psi_pooled], dim=1)

        return self.out_norm(tokens)


class AnyTopBehavior(AnyTop):
    """AnyTop with behavior-conditioning channel (replaces AnyTopConditioned).

    Training: source x → R(x) = behavior_tokens → decoder reconstructs x on x's own skeleton.
    Inference: source x_s on skeleton s → R(x_s) → decoder generates on skeleton t.

    The behavior representation is topology-normalized by construction (uses analytic ψ
    + action label, optional learned residual). Replaces the MotionEncoder which produced
    skeleton-specific z (verified worse than null in V1/V6).
    """
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", t5_out_dim=512, root_input_feats=13,
                 # Behavior recognizer config
                 n_actions=12,
                 action_emb_dim=32,
                 psi_temporal_frames=64,
                 psi_dim=62,
                 use_residual=False,
                 residual_dim=16,
                 n_behavior_tokens=8,
                 # CFG / dropout
                 behavior_drop_prob=0.1,
                 geom_drop_prob=0.3, geom_jitter_prob=0.2,
                 topo_drop_prob=0.15,
                 z_norm_target=None,
                 **kargs):
        super().__init__(max_joints, feature_len, latent_dim, ff_size, num_layers,
                         num_heads, dropout, activation, t5_out_dim, root_input_feats, **kargs)

        # Replace decoder with conditioned version (cross-attention layer)
        cond_layer = ConditionedGraphMotionDecoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation=activation)
        self.seqTransDecoder = ConditionedGraphMotionDecoder(
            cond_layer, num_layers=num_layers, value_emb=self.value_emb)

        # Behavior recognizer (replaces MotionEncoder)
        self.behavior_recognizer = BehaviorRecognizer(
            d_model=latent_dim,
            n_actions=n_actions,
            action_emb_dim=action_emb_dim,
            psi_temporal_frames=psi_temporal_frames,
            psi_dim=psi_dim,
            use_residual=use_residual,
            residual_dim=residual_dim,
            n_behavior_tokens=n_behavior_tokens,
        )

        # Token shape passed to cross-attention: [B, T_b=1, n_tokens, latent_dim]
        # to match AnyTopConditioned interface (which expects [B, T_b, K, D])
        self.n_total_tokens = n_behavior_tokens + 1 + (1 if use_residual else 0)

        # CFG null tokens — learnable, NOT zero (zero causes decoder to ignore conditioning)
        self.null_behavior = nn.Parameter(
            torch.randn(1, 1, self.n_total_tokens, latent_dim) * 0.02)
        self.behavior_drop_prob = behavior_drop_prob
        self.z_norm_target = z_norm_target if z_norm_target is not None else (latent_dim ** 0.5)

        self.geom_drop_prob = geom_drop_prob
        self.geom_jitter_prob = geom_jitter_prob
        self.topo_drop_prob = topo_drop_prob

    def _apply_geom_dropout(self, tpos_first_frame, skip_mask=None):
        B = tpos_first_frame.shape[1]
        r = torch.rand(B, device=tpos_first_frame.device)
        if skip_mask is not None:
            r[skip_mask] = 1.0
        drop_mask = r < self.geom_drop_prob
        jitter_mask = (~drop_mask) & (r < self.geom_drop_prob + self.geom_jitter_prob)
        out = tpos_first_frame.clone()
        if drop_mask.any():
            out[:, drop_mask] = 0.0
        if jitter_mask.any():
            noise = torch.randn_like(out[:, jitter_mask]) * 0.1
            out[:, jitter_mask] = out[:, jitter_mask] + noise
        return out

    def _apply_topo_dropout(self, y, topo_drop_mask):
        if not topo_drop_mask.any():
            return y
        y = dict(y)
        gd = y['graph_dist'].clone()
        jr = y['joints_relations'].clone()
        gd[topo_drop_mask] = 1
        jr[topo_drop_mask] = 0
        y['graph_dist'] = gd
        y['joints_relations'] = jr
        return y

    def forward(self, x, timesteps, get_layer_activation=-1, y=None):
        joints_mask      = y['joints_mask'].to(x.device)
        temp_mask        = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape

        # Conditioning dropout (kept identical to AnyTopConditioned)
        if self.training:
            topo_drop_mask = torch.rand(bs, device=x.device) < self.topo_drop_prob
            if topo_drop_mask.any():
                y = self._apply_topo_dropout(y, topo_drop_mask)
                tpos_first_frame[:, topo_drop_mask] = 0.0
            if self.geom_drop_prob > 0 or self.geom_jitter_prob > 0:
                tpos_first_frame = self._apply_geom_dropout(
                    tpos_first_frame, skip_mask=topo_drop_mask)

        # Behavior tokens: precomputed externally OR computed from y['psi'] + y['action_label']
        behavior_tokens = y.get('behavior_tokens', None)
        if behavior_tokens is None:
            # Compute from ψ + action
            psi = y['psi'].to(x.device) if 'psi' in y else None
            action = y.get('action_label', None)
            if action is not None:
                action = action.to(x.device)
            if psi is None:
                # No conditioning info — use null
                behavior_tokens = self.null_behavior.expand(bs, 1, self.n_total_tokens, self.latent_dim)
            else:
                tokens = self.behavior_recognizer(psi, action)  # [B, n_total_tokens, D]
                behavior_tokens = tokens.unsqueeze(1)  # [B, 1, n_total_tokens, D]
        else:
            behavior_tokens = behavior_tokens.to(x.device)

        # CFG dropout: replace with null for a fraction of training samples
        if self.training and self.behavior_drop_prob > 0:
            drop = torch.rand(bs, device=x.device) < self.behavior_drop_prob
            null = self.null_behavior.expand_as(behavior_tokens)
            behavior_tokens = torch.where(drop[:, None, None, None], null, behavior_tokens)

        # L2 normalize per token to fixed target norm (eliminates magnitude shortcut)
        token_norms = behavior_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        behavior_tokens = behavior_tokens / token_norms * self.z_norm_target

        timesteps_emb = create_sin_embedding(
            timesteps.view(1, -1, 1), self.latent_dim)[0]
        x_emb = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind'])

        spatial_mask = 1.0 - joints_mask[:, 0, 0, 1:, 1:]
        spatial_mask = (spatial_mask.unsqueeze(1).unsqueeze(1)
                        .repeat(1, nframes + 1, self.num_heads, 1, 1)
                        .reshape(-1, self.num_heads, njoints, njoints))
        temporal_mask = (1.0 - temp_mask.repeat(1, njoints, self.num_heads, 1, 1)
                         .reshape(-1, nframes + 1, nframes + 1).float())
        spatial_mask[spatial_mask == 1.0]   = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9

        output = self.seqTransDecoder(
            tgt=x_emb, timesteps_embs=timesteps_emb, memory=None,
            spatial_mask=spatial_mask, temporal_mask=temporal_mask,
            y=y, get_layer_activation=get_layer_activation,
            z_embed=behavior_tokens)

        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output      = output[0]
        output = self.output_process(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output
