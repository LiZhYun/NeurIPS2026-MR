"""AnyTopBehavior with FlexiAct-style timestep-dependent z-gate.

Pilot B: adds a 4k-param MLP g(t) that gates the behavior tokens against the
learnable null tokens, replacing the fixed CFG-dropout null-mix at the
injection site (L229-236 of model/anytop_behavior.py).

Rule under test: the FlexiAct/FAE paper SUPPORTS that the action encoder is
called during denoising and its contribution should be heterogeneous across
timesteps, but the "early-t = motion, late-t = appearance" shape is an
extrapolation. We therefore initialize g(t) ≡ 0.5 (sigmoid(0)) at all t and
let training discover the actual shape.

Gate MLP: sinusoidal_time_embed(t, d=64) → Linear(64,64) → GELU → Linear(64,1)
(bias 0) → Sigmoid. Total: 64*64 + 64 + 64*1 + 1 = 4225 params.
"""
import torch
import torch.nn as nn

from model.anytop_behavior import AnyTopBehavior
from model.anytop import create_sin_embedding


class TimestepGate(nn.Module):
    """g(t) ∈ (0, 1) per-sample gate driven by the diffusion timestep."""

    def __init__(self, time_emb_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.fc1 = nn.Linear(time_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # Initialize so that g(t) ≡ sigmoid(0) = 0.5 at all t.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.act = nn.GELU()

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B] (any dtype); produce embedding [B, time_emb_dim]
        positions = timesteps.view(1, -1, 1).float()
        emb = create_sin_embedding(positions, self.time_emb_dim)[0]  # [B, time_emb_dim]
        h = self.act(self.fc1(emb))
        g = torch.sigmoid(self.fc2(h))  # [B, 1]
        return g


class AnyTopBehaviorFlexiGate(AnyTopBehavior):
    """AnyTopBehavior whose null-mix weight is a learned function of t."""

    def __init__(self, *args, gate_time_emb_dim: int = 64, gate_hidden_dim: int = 64,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = TimestepGate(gate_time_emb_dim, gate_hidden_dim)

    def forward(self, x, timesteps, get_layer_activation=-1, y=None):
        # --- mirror AnyTopBehavior.forward up through behavior-token prep ---
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

        behavior_tokens = y.get('behavior_tokens', None)
        if behavior_tokens is None:
            psi = y['psi'].to(x.device) if 'psi' in y else None
            action = y.get('action_label', None)
            if action is not None:
                action = action.to(x.device)
            if psi is None:
                behavior_tokens = self.null_behavior.expand(
                    bs, 1, self.n_total_tokens, self.latent_dim)
            else:
                tokens = self.behavior_recognizer(psi, action)
                behavior_tokens = tokens.unsqueeze(1)
        else:
            behavior_tokens = behavior_tokens.to(x.device)

        # --- GATE INJECTION (replaces CFG dropout null-mix at L229-236) ---
        # g(t) is sample-dependent via the per-sample timestep; shape [B, 1].
        g = self.gate(timesteps)  # [B, 1]
        null = self.null_behavior.expand_as(behavior_tokens)
        g_expanded = g.view(bs, 1, 1, 1)
        behavior_tokens = g_expanded * behavior_tokens + (1.0 - g_expanded) * null

        # L2 re-norm (kept identical to parent).
        token_norms = behavior_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        behavior_tokens = behavior_tokens / token_norms * self.z_norm_target

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
            z_embed=behavior_tokens)

        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output      = output[0]
        output = self.output_process(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output
