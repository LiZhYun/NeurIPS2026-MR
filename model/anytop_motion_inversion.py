"""Pilot M-INV — Motion-Inversion-style conditioning for AnyTopBehavior.

Based on the verified mechanism of Motion Inversion (arXiv 2403.20193):

1. A 1D *temporal* motion embedding (no spatial dims) is derived from the
   behavior tokens. It gates the temporal-attention module only.
2. Motion-QK: the temporal attention's Q and K receive an additive bias
   derived from the motion embedding — restricted to the time axis, broadcast
   across joints.
3. Motion-V: the temporal attention's V uses a *differential* transform
   (V_t := V_t - V_{t+1}) to debias appearance and keep only the frame-to-frame
   change signal.

The module subclasses AnyTopBehavior and rewires the decoder's cross-attention
so behavior tokens no longer leak through the spatially-entangled path:
``cross_attn(z_embed)`` is replaced with the (constant) *null* behavior token
during training AND inference. The only route by which behavior affects output
is the motion-inversion temporal-attention hook.

Trainable parameters when frozen backbone is used:
  * Motion-Embed module (pool + per-layer projections, tiny)
  * LoRA rank-r adapters on temporal_attn Q/K/V/O (same as vmc_lora)
  * null_behavior learned parameter (retained from B1 init)

Everything else — spatial attention, cross-attention weights, feed-forward,
input/output process, behavior_recognizer — is frozen.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.anytop_behavior import AnyTopBehavior
from model.anytop_conditioned import ConditionedGraphMotionDecoderLayer


# ---------------------------------------------------------------------------
# Motion-Embed module
# ---------------------------------------------------------------------------

class MotionEmbed(nn.Module):
    """Project behavior tokens [B, 1, n_tok, d] to a 1D temporal motion signal.

    Output: [B, T_frames, d] — a per-time-frame bias broadcast across joints
    when injected into temporal attention Q/K.

    Construction (minimum-viable per Motion Inversion recipe):
      1. Average-pool n_tok behavior tokens → [B, d].
      2. A small learnable "motion template" [T_frames, d] provides temporal
         structure; it is modulated additively by the pooled behavior signal.
      3. A per-layer projection head produces the bias actually added into each
         STT layer's temporal Q and K (see InversionTemporalMHA).
    """

    def __init__(self, d_model: int, n_frames: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.n_frames = n_frames
        self.num_layers = num_layers

        # Temporal template (learned). Small init so the embedding starts near
        # zero and grows during training.
        self.temporal_template = nn.Parameter(
            torch.randn(n_frames, d_model) * 0.02)
        # Projection: behavior_token_mean → per-frame FiLM-style modulation.
        self.pool_proj = nn.Linear(d_model, d_model)
        # Per-layer heads produce a dedicated bias for each decoder layer.
        self.per_layer_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_layers)
        ])
        # Zero-init the last linear of each head so the motion-embed adds
        # nothing at step 0 (clean init — no behavior surgery on a frozen
        # backbone).
        for head in self.per_layer_head:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # Scalar gate so training can grow the contribution smoothly.
        # Init to a small positive value so every trainable parameter
        # receives a nonzero gradient on step 1.
        self.scale = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, behavior_tokens: Tensor) -> List[Tensor]:
        """behavior_tokens: [B, 1, n_tok, d] → list of [B, T_frames, d] biases."""
        B = behavior_tokens.shape[0]
        # Pool across token axis only; Np=1 by construction in B1.
        pooled = behavior_tokens.mean(dim=(1, 2))          # [B, d]
        bias = self.pool_proj(pooled)                       # [B, d]
        # Broadcast along time, add the learned template.
        T = self.n_frames
        tmpl = self.temporal_template.unsqueeze(0).expand(B, T, -1)  # [B,T,d]
        merged = tmpl + bias.unsqueeze(1)                   # [B,T,d]
        # One bias per decoder layer.
        out = [head(merged) * self.scale for head in self.per_layer_head]
        return out


# ---------------------------------------------------------------------------
# Inversion temporal attention (Motion-QK + differential Motion-V + LoRA)
# ---------------------------------------------------------------------------

class InversionTemporalMHA(nn.Module):
    """Temporal MHA with Motion-QK bias, differential Motion-V, and LoRA.

    Wraps a *frozen* ``nn.MultiheadAttention`` (the original temporal_attn).
    Compute path (per frame t, joint j):

        q_in[t] = x[t] + m[t]       (motion embedding added into Q projection)
        k_in[t] = x[t] + m[t]       (motion embedding added into K projection)
        v_in[t] = x[t]              (V is computed from x; see below)

    LoRA deltas (rank r) on Q/K/V/O reuse the VMC-LoRA pattern.

    After V = Wv(v_in), apply the differential operation
        V_motion[t] = V[t] - V[t+1]
    (last frame gets V[T-1] - V[T-1] = 0). This is what Motion Inversion calls
    Motion-V: it removes the appearance (DC) component, keeping only the
    frame-to-frame change that carries motion information.

    The differential transform is gated by ``use_differential_v`` so we can
    ablate it. When False the V path is left unchanged (plain V).
    """

    def __init__(self, base_mha: nn.MultiheadAttention, rank: int = 16,
                 alpha: int = 32, dropout: float = 0.0,
                 use_differential_v: bool = True):
        super().__init__()
        assert isinstance(base_mha, nn.MultiheadAttention)
        assert base_mha._qkv_same_embed_dim, "Only qkv_same_embed_dim MHA supported"
        self.base = base_mha
        for p in self.base.parameters():
            p.requires_grad_(False)

        d = base_mha.embed_dim
        self.embed_dim = d
        self.num_heads = base_mha.num_heads
        self.head_dim = d // self.num_heads
        self.dropout_p = base_mha.dropout
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.batch_first = base_mha.batch_first
        self.lora_drop = nn.Dropout(dropout)
        self.use_differential_v = use_differential_v
        # Optional runtime override of the V-path (set by callers for ablation).
        self._force_plain_v: bool = False

        def _mkA():
            A = nn.Parameter(torch.empty(rank, d))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            return A

        def _mkB():
            return nn.Parameter(torch.zeros(d, rank))

        self.A_q = _mkA(); self.B_q = _mkB()
        self.A_k = _mkA(); self.B_k = _mkB()
        self.A_v = _mkA(); self.B_v = _mkB()
        self.A_o = _mkA(); self.B_o = _mkB()

        # Motion-embedding plumbing — populated per forward by the decoder.
        # Stored here (not passed through the decoder signature) so we do not
        # need to modify the frozen GraphMotionDecoderLayer.forward signature.
        self._motion_embed: Optional[Tensor] = None   # [B, T, D] or None
        self._enabled: bool = True                    # gate: turn module off
        self._njoints: Optional[int] = None           # for reshaping the bias

    def lora_params(self):
        return [self.A_q, self.B_q, self.A_k, self.B_k,
                self.A_v, self.B_v, self.A_o, self.B_o]

    def set_motion_embed(self, m: Optional[Tensor], njoints: Optional[int]):
        """Called per forward by the wrapper; cleared afterwards."""
        self._motion_embed = m
        self._njoints = njoints

    def clear_motion_embed(self):
        self._motion_embed = None
        self._njoints = None

    def forward(self, query, key, value, attn_mask=None,
                key_padding_mask=None, need_weights: bool = True,
                average_attn_weights: bool = True, is_causal: bool = False):
        # The caller reshapes to [T+1, bs*njoints, D]. The motion embedding
        # (if supplied) has shape [B, T+1, D] — broadcast it across njoints.
        T, BJ, D = query.shape

        if self._motion_embed is not None and self._enabled and self._njoints is not None:
            m = self._motion_embed            # [B, T_m, D]
            B_from_m = m.shape[0]
            # If T (model's frames+1) differs from T_m, interpolate along time
            T_m = m.shape[1]
            if T_m != T:
                m = F.interpolate(
                    m.permute(0, 2, 1), size=T, mode='linear', align_corners=False
                ).permute(0, 2, 1)
            # Broadcast across joints: [B, T, D] → [T, B, njoints, D] → [T, B*J, D]
            njoints = self._njoints
            m_exp = m.permute(1, 0, 2).unsqueeze(2).expand(T, B_from_m, njoints, D)
            m_exp = m_exp.reshape(T, B_from_m * njoints, D)
            q_in = query + m_exp
            k_in = key + m_exp
        else:
            q_in = query
            k_in = key

        v_in = value

        # Build effective in-proj weight = base + LoRA deltas (stacked).
        W = self.base.in_proj_weight  # (3d, d)
        b = self.base.in_proj_bias    # (3d,)
        d = self.embed_dim

        dWq = (self.B_q @ self.A_q) * self.scaling
        dWk = (self.B_k @ self.A_k) * self.scaling
        dWv = (self.B_v @ self.A_v) * self.scaling

        # Compute Q, K, V separately so we can apply the differential to V.
        Wq = W[:d]      + dWq
        Wk = W[d:2*d]   + dWk
        Wv = W[2*d:]    + dWv
        bq = b[:d]
        bk = b[d:2*d]
        bv = b[2*d:]

        Q = F.linear(q_in, Wq, bq)   # [T, BJ, D]
        K = F.linear(k_in, Wk, bk)
        V = F.linear(v_in, Wv, bv)

        use_diff = self.use_differential_v and not self._force_plain_v \
                   and self._enabled and self._motion_embed is not None
        if use_diff:
            # V_motion[t] = V[t] - V[t+1] ; last frame → zero difference.
            V_shift = torch.roll(V, shifts=-1, dims=0)
            V_shift[-1] = V[-1]  # avoid wrap: last frame reuses own V → diff=0
            V = V - V_shift

        # Reshape for multi-head: [T, BJ, D] → [BJ, nhead, T, head_dim]
        nhead = self.num_heads
        hd = self.head_dim
        def _split(x):
            return x.view(T, BJ, nhead, hd).permute(1, 2, 0, 3)
        Qh = _split(Q)
        Kh = _split(K)
        Vh = _split(V)

        # Attention scores
        scale = 1.0 / math.sqrt(hd)
        attn = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale  # [BJ, nhead, T, T]
        if attn_mask is not None:
            # The caller (anytop_conditioned.forward) pre-flattens the
            # temporal mask to [BJ * nhead, T, T] so PyTorch's MHA can consume
            # it directly. We reshape back to [BJ, nhead, T, T] to broadcast
            # onto our per-head attention scores.
            if attn_mask.dim() == 2:
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3 and attn_mask.shape[0] == BJ * nhead:
                attn = attn + attn_mask.view(BJ, nhead, T, T)
            elif attn_mask.dim() == 3 and attn_mask.shape[0] == BJ:
                attn = attn + attn_mask.unsqueeze(1)
            else:
                attn = attn + attn_mask  # trust-but-fail (shape-error path)
        if key_padding_mask is not None:
            # [BJ, T] → broadcast
            kpm = key_padding_mask.float().masked_fill(key_padding_mask, float('-inf'))
            attn = attn + kpm.unsqueeze(1).unsqueeze(1)
        attn = torch.softmax(attn, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = torch.matmul(attn, Vh)              # [BJ, nhead, T, head_dim]
        out = out.permute(2, 0, 1, 3).reshape(T, BJ, D)

        out_W = self.base.out_proj.weight
        out_b = self.base.out_proj.bias
        out_W_eff = out_W + (self.B_o @ self.A_o) * self.scaling
        attn_out = F.linear(out, out_W_eff, out_b)

        if need_weights:
            if average_attn_weights:
                attn_w = attn.mean(dim=1)
            else:
                attn_w = attn
        else:
            attn_w = None

        return attn_out, attn_w


# ---------------------------------------------------------------------------
# Decoder wrapping + model
# ---------------------------------------------------------------------------

def _patched_temporal_block(layer, lora_attn: InversionTemporalMHA, n_frames: int):
    """Wrap layer._temporal_mha_block_sin_joint so njoints is supplied to the LoRA.

    The original signature is ``(x, attn_mask, key_padding_mask)`` where
    ``x`` has shape [frames, bs, njoints, feats]. We use this call-site to
    capture ``njoints`` and hand it to the InversionTemporalMHA before it
    runs, then flatten to [frames, bs*njoints, feats] as before.
    """
    original = layer._temporal_mha_block_sin_joint  # noqa: unused

    def new_block(x: Tensor, attn_mask, key_padding_mask):
        frames, bs, njoints, feats = x.size()
        lora_attn.set_motion_embed(lora_attn._motion_embed, njoints)
        x_flat = x.view(frames, bs * njoints, feats)
        out_attn, _ = lora_attn(
            x_flat, x_flat, x_flat, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, need_weights=False)
        out_attn = out_attn.view(frames, bs, njoints, feats)
        lora_attn.clear_motion_embed()
        return layer.dropout2(out_attn)

    layer._temporal_mha_block_sin_joint = new_block


def inject_motion_inversion(model: AnyTopBehavior, n_frames: int,
                             rank: int = 16, alpha: int = 32,
                             dropout: float = 0.0,
                             use_differential_v: bool = True
                             ) -> List[InversionTemporalMHA]:
    """Replace each decoder layer's ``temporal_attn`` with InversionTemporalMHA."""
    lora_mods: List[InversionTemporalMHA] = []
    decoder = model.seqTransDecoder
    for layer in decoder.layers:
        base = layer.temporal_attn
        lora = InversionTemporalMHA(
            base, rank=rank, alpha=alpha, dropout=dropout,
            use_differential_v=use_differential_v)
        layer.temporal_attn = lora
        # The temporal block wrapper sets njoints on each forward call.
        _patched_temporal_block(layer, lora, n_frames=n_frames)
        lora_mods.append(lora)
    return lora_mods


class AnyTopMotionInversion(AnyTopBehavior):
    """AnyTopBehavior re-routed to inject behavior via temporal-attention only.

    On forward:
      1. Behavior tokens are computed as in B1.
      2. They are passed to MotionEmbed → per-layer temporal biases.
      3. The per-layer biases are handed to each InversionTemporalMHA's
         ``_motion_embed`` slot so Motion-QK + differential-V fire inside the
         temporal block.
      4. The decoder's cross-attention receives the *null* behavior tokens
         (so the spatially-entangled path stays un-informative — the only
         real behavior signal is the temporal-attention hook).

    The full graph shares B1's null_behavior parameter (allowing us to load
    from a B1 checkpoint cleanly).
    """

    def __init__(self, *args, n_frames: int = 120,
                 mi_rank: int = 16, mi_alpha: int = 32, mi_dropout: float = 0.0,
                 use_differential_v: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_frames = n_frames
        self.motion_embed = MotionEmbed(
            d_model=self.latent_dim, n_frames=n_frames + 1,
            num_layers=self.num_layers)
        self.mi_lora_mods = inject_motion_inversion(
            self, n_frames=n_frames + 1, rank=mi_rank, alpha=mi_alpha,
            dropout=mi_dropout, use_differential_v=use_differential_v)
        # State flag: whether the motion-inversion hook is active at all.
        self.mi_enabled = True

    # ---------- enable / disable the inversion module --------------------
    def set_mi_enabled(self, flag: bool):
        self.mi_enabled = flag
        for m in self.mi_lora_mods:
            m._enabled = flag

    def set_force_plain_v(self, flag: bool):
        for m in self.mi_lora_mods:
            m._force_plain_v = flag

    # ---------- training / eval ------------------------------------------
    def forward(self, x, timesteps, get_layer_activation=-1, y=None):
        joints_mask      = y['joints_mask'].to(x.device)
        temp_mask        = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape

        if self.training:
            topo_drop_mask = torch.rand(bs, device=x.device) < self.topo_drop_prob
            if topo_drop_mask.any():
                y = self._apply_topo_dropout(y, topo_drop_mask)
                tpos_first_frame[:, topo_drop_mask] = 0.0
            if self.geom_drop_prob > 0 or self.geom_jitter_prob > 0:
                tpos_first_frame = self._apply_geom_dropout(
                    tpos_first_frame, skip_mask=topo_drop_mask)

        # Compute behavior_tokens (same as B1) -------------------------------
        behavior_tokens = y.get('behavior_tokens', None)
        if behavior_tokens is None:
            psi = y['psi'].to(x.device) if 'psi' in y else None
            action = y.get('action_label', None)
            if action is not None:
                action = action.to(x.device)
            if psi is None:
                behavior_tokens = self.null_behavior.expand(bs, 1, self.n_total_tokens, self.latent_dim)
            else:
                tokens = self.behavior_recognizer(psi, action)
                behavior_tokens = tokens.unsqueeze(1)
        else:
            behavior_tokens = behavior_tokens.to(x.device)

        if self.training and self.behavior_drop_prob > 0:
            drop = torch.rand(bs, device=x.device) < self.behavior_drop_prob
            null = self.null_behavior.expand_as(behavior_tokens)
            behavior_tokens = torch.where(drop[:, None, None, None], null, behavior_tokens)

        # L2 normalize
        token_norms = behavior_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        behavior_tokens_real = behavior_tokens / token_norms * self.z_norm_target

        # Motion embedding ----------------------------------------------------
        if self.mi_enabled:
            per_layer_m = self.motion_embed(behavior_tokens_real)
        else:
            per_layer_m = [None] * self.num_layers
        for lora, m in zip(self.mi_lora_mods, per_layer_m):
            lora._motion_embed = m

        # Re-route cross-attention to the *null* behavior token (neutralize
        # the spatial-entangled path). Use the model's learned null_behavior,
        # renormalized like B1 does.
        null_base = self.null_behavior.expand(bs, 1, self.n_total_tokens, self.latent_dim)
        null_norms = null_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        null_z = null_base / null_norms * self.z_norm_target
        cross_z = null_z if self.mi_enabled else behavior_tokens_real

        # Standard decoder plumbing ------------------------------------------
        from model.anytop import create_sin_embedding
        timesteps_emb = create_sin_embedding(
            timesteps.view(1, -1, 1), self.latent_dim)[0]
        x_emb = self.input_process(
            x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind'])

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
            z_embed=cross_z)

        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output      = output[0]
        output = self.output_process(output)

        # Hygiene: clear motion-embed from each LoRA so stale state doesn't
        # leak into a subsequent forward (matters for the ablation eval path).
        for lora in self.mi_lora_mods:
            lora.clear_motion_embed()

        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output


def freeze_everything_but_motion_inversion(model: AnyTopMotionInversion):
    """Freeze the backbone; leave MotionEmbed + LoRA deltas trainable.

    Returns the list of trainable tensors for passing to the optimizer.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    # LoRA deltas
    for m in model.mi_lora_mods:
        for p in m.lora_params():
            p.requires_grad_(True)
            trainable.append(p)
    # MotionEmbed module
    for p in model.motion_embed.parameters():
        p.requires_grad_(True)
        trainable.append(p)
    return trainable
