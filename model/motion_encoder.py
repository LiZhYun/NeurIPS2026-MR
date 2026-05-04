import math
import torch
import torch.nn as nn
from model.anytop import create_sin_embedding


class RestPE(nn.Module):
    """Encode rest-pose bone offsets into D'-dimensional vectors.

    Applies sinusoidal encoding per spatial coordinate (no topology information),
    then maps through a 2-layer MLP.
    """
    def __init__(self, d_model, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        enc_dim = 3 * 2 * num_frequencies  # 48
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, offsets):
        # offsets: [B, J, 3]
        freqs = 2.0 ** torch.arange(self.num_frequencies, device=offsets.device, dtype=offsets.dtype)
        enc = offsets[..., None] * math.pi * freqs   # [B, J, 3, num_freq]
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)  # [B, J, 3, 2*num_freq]
        enc = enc.flatten(-2)                         # [B, J, 48]
        return self.mlp(enc)                          # [B, J, D']


class AttentionPool(nn.Module):
    """Compress J joint tokens into K functional slots per frame via learned cross-attention.

    K query vectors are global parameters shared across all frames and all skeletons.
    """
    def __init__(self, d_model, num_queries, num_heads, dropout):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, h, joints_mask, return_attn=False):
        # h:           [B, T, J, D']
        # joints_mask: [B, J]  True=real joint, False=padding
        B, T, J, D = h.shape
        h_flat = h.reshape(B * T, J, D)
        queries = self.queries.unsqueeze(0).expand(B * T, -1, -1)             # [B*T, K, D']
        key_padding_mask = ~joints_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J)
        out, attn = self.attn(queries, h_flat, h_flat, key_padding_mask=key_padding_mask,
                              need_weights=return_attn, average_attn_weights=True)
        if return_attn:
            return out.view(B, T, self.num_queries, D), attn.view(B, T, self.num_queries, J)
        return out.view(B, T, self.num_queries, D)                             # [B, T, K, D']


class TemporalCNN(nn.Module):
    """Downsample temporal resolution N→N/4, processing each spatial slot independently."""
    def __init__(self, d_model):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
        )

    def forward(self, s):
        # s: [B, T, K, D']
        B, T, K, D = s.shape
        x = s.permute(0, 2, 3, 1).reshape(B * K, D, T)   # [B*K, D', T]
        x = self.convs(x)                                  # [B*K, D', T/4]
        T_out = x.shape[-1]
        return x.view(B, K, D, T_out).permute(0, 3, 1, 2)  # [B, T/4, K, D']


class FSQ(nn.Module):
    """Finite Scalar Quantization with straight-through gradient estimator.

    No learnable parameters; no commitment loss required.
    Quantizes each dimension to L uniformly spaced levels in [-1, 1].
    """
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

    def forward(self, x, return_codes=False):
        x_bounded = torch.tanh(x)
        x_scaled  = (x_bounded + 1) / 2 * (self.levels - 1)      # [0, L-1]
        x_round   = torch.round(x_scaled)
        x_st      = x_scaled + (x_round - x_scaled).detach()      # straight-through
        out = x_st / (self.levels - 1) * 2 - 1                    # back to [-1, 1]
        if return_codes:
            return out, x_round.long()
        return out


class MotionEncoder(nn.Module):
    """Encode source motion into a skeleton-agnostic latent z.

    Receives motion features and rest-pose geometry only — no topology (R_S, D_S).
    Output z_embed is used by the conditioned decoder's cross-attention layers.

    Three bottleneck modes (controlled by enc_mode; use_vae is a convenience alias):
      enc_mode='vae'    (A1, default): variational — returns (z_out, mu, logvar)
      enc_mode='fsq'    (FSQ ablation): discrete   — returns z_out
      enc_mode='direct' (A2 baseline): no bottleneck, direct linear — returns z_out
    """
    def __init__(self, feature_len=13, d_model=128, num_queries=4,
                 num_heads=4, fsq_dims=4, fsq_levels=5, dropout=0.1,
                 use_vae=True, d_z=64, enc_mode=None, no_rest_pe=False):
        super().__init__()
        # enc_mode takes priority over use_vae flag
        if enc_mode is not None:
            self.enc_mode = enc_mode
        else:
            self.enc_mode = 'vae' if use_vae else 'fsq'
        self.use_vae     = (self.enc_mode == 'vae')
        self.d_model     = d_model
        self.num_queries = num_queries
        self.d_z         = d_z if self.use_vae else (fsq_dims if self.enc_mode == 'fsq' else d_model)
        self.no_rest_pe  = no_rest_pe

        self.root_emb     = nn.Linear(feature_len, d_model)
        self.joint_emb    = nn.Linear(feature_len, d_model)
        if not no_rest_pe:
            self.rest_pe  = RestPE(d_model)
        self.attn_pool    = AttentionPool(d_model, num_queries, num_heads, dropout)
        self.temporal_cnn = TemporalCNN(d_model)

        if self.enc_mode == 'vae':
            self.mu_head     = nn.Linear(d_model, d_z)
            self.logvar_head = nn.Linear(d_model, d_z)
            # z_proj with LayerNorm prevents projection collapse (z_embed ≈ null_z)
            self.z_proj      = nn.Sequential(
                nn.LayerNorm(d_z),
                nn.Linear(d_z, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        elif self.enc_mode == 'fsq':
            self.pre_fsq  = nn.Linear(d_model, fsq_dims)
            self.fsq      = FSQ(fsq_levels)
            self.post_fsq = nn.Linear(fsq_dims, d_model)
        else:  # direct
            self.direct_proj = nn.Linear(d_model, d_model)

    def _embed(self, source_motion, source_offsets, source_mask, return_attn=False):
        """Shared encoder backbone up to temporal CNN output."""
        x = source_motion.permute(0, 3, 1, 2)                       # [B, T, J, 13]
        root  = self.root_emb(x[:, :, 0:1])                         # [B, T, 1, D']
        rest  = self.joint_emb(x[:, :, 1:])                         # [B, T, J-1, D']
        x_emb = torch.cat([root, rest], dim=2)                       # [B, T, J, D']

        if not self.no_rest_pe:
            x_emb = x_emb + self.rest_pe(source_offsets).unsqueeze(1)  # [B, 1, J, D']

        T = x_emb.shape[1]
        positions = torch.arange(T, device=x_emb.device).view(1, T, 1).float()
        x_emb = x_emb + create_sin_embedding(positions, self.d_model).unsqueeze(2)

        s, attn_w = self.attn_pool(x_emb, source_mask, return_attn=True)
        t_out = self.temporal_cnn(s)                                 # [B, T/4, K, D']
        if return_attn:
            return t_out, attn_w
        return t_out

    def forward(self, source_motion, source_offsets, source_mask, return_intermediates=False):
        # source_motion:  [B, J, 13, T]
        # source_offsets: [B, J, 3]
        # source_mask:    [B, J]  True=real joint, False=padding
        #
        # VAE mode    returns: (z_out, mu, logvar) or (z_out, mu, logvar, dict)
        # FSQ mode    returns: z_out              or (z_out, dict)
        # Direct mode returns: z_out              or (z_out, dict)

        if return_intermediates:
            t_out, attn_weights = self._embed(source_motion, source_offsets, source_mask, return_attn=True)
        else:
            t_out = self._embed(source_motion, source_offsets, source_mask, return_attn=False)

        if self.enc_mode == 'vae':
            mu     = self.mu_head(t_out)      # [B, T/4, K, d_z]
            logvar = self.logvar_head(t_out)  # [B, T/4, K, d_z]
            if self.training:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            else:
                z = mu                        # deterministic at test time
            z_out = self.z_proj(z)            # [B, T/4, K, d_model]
            if return_intermediates:
                return z_out, mu, logvar, {'attn_weights': attn_weights}
            return z_out, mu, logvar

        elif self.enc_mode == 'fsq':
            z_pre = self.pre_fsq(t_out)
            if return_intermediates:
                z_quant, z_codes = self.fsq(z_pre, return_codes=True)
                z_out = self.post_fsq(z_quant)
                return z_out, {'attn_weights': attn_weights, 'z_codes': z_codes}
            return self.post_fsq(self.fsq(z_pre))

        else:  # direct — no bottleneck
            z_out = self.direct_proj(t_out)   # [B, T/4, K, d_model]
            if return_intermediates:
                return z_out, {'attn_weights': attn_weights}
            return z_out
