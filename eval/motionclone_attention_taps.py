"""Runtime attention taps for MotionClone on frozen AnyTop (A3).

MotionClone (arXiv 2406.05338, ICLR 2025 — verified mechanism per
`idea-stage/LIT_REVIEW_VERIFICATION_2026_04_15.md`):
  * Sparse temporal-attention weights from a frozen video diffusion model.
  * Single denoising-step extraction.
  * Location-aware semantic guidance (only at positions where the sparse
    source map is non-zero).
  * Training-free.

CLAUDE.md rule — "NEVER modify AnyTop's existing base modules in-place
without explicit approval — create new files or subclass." — means we
must monkey-patch at runtime rather than editing
`model/motion_transformer.py`. This file installs a thin wrapper on the
`temporal_attn` module of every `GraphMotionDecoderLayer`, which:

  1. forwards the call exactly like `nn.MultiheadAttention`,
  2. also computes attention weights with `need_weights=True,
     average_attn_weights=False`,
  3. stores the most-recent attention tensor in `.last_attn_weights`.

Nothing else in the model is touched; gradient flow, dtypes, mask shapes
are preserved.

Typical usage
-------------

>>> from eval.motionclone_attention_taps import install_temporal_taps, uninstall_temporal_taps
>>> taps = install_temporal_taps(model)  # list[TemporalAttnTap] — len == n_layers
>>> model(x_noisy, t, y=y)               # triggers forward, fills tap.last_attn_weights
>>> attn_maps = [tap.last_attn_weights for tap in taps]  # each [B*J, H, T, T]
>>> uninstall_temporal_taps(model, taps) # restore original modules
"""
from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn


class TemporalAttnTap(nn.Module):
    """Drop-in wrapper for `nn.MultiheadAttention`.

    Preserves the call signature used inside
    `GraphMotionDecoderLayer._temporal_mha_block_sin_joint`, but internally
    requests per-head attention weights. The latest weight tensor is
    cached on `self.last_attn_weights`.

    Attributes
    ----------
    wrapped : nn.MultiheadAttention
        The original temporal-attention module (parameters unchanged).
    last_attn_weights : torch.Tensor or None
        Shape ``[B*J, H, T, T]`` after a forward pass (None before the
        first forward).
    enabled : bool
        If False, behaves exactly like the wrapped module (no tap overhead).
    """

    def __init__(self, wrapped: nn.MultiheadAttention):
        super().__init__()
        self.wrapped = wrapped
        self.last_attn_weights: Optional[torch.Tensor] = None
        self.enabled: bool = True
        # Mirror attributes that downstream code may read.
        self.embed_dim = wrapped.embed_dim
        self.num_heads = wrapped.num_heads

    def forward(self, query, key, value,
                key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        # AnyTop always calls us like `self.temporal_attn(x, x, x, attn_mask=..., key_padding_mask=...)`
        # (see `_temporal_mha_block_sin_joint`), unpacking two values.  We must
        # therefore return (attn_output, attn_weights) — same as wrapped MHA.
        if self.enabled:
            out, w = self.wrapped(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=False,
                is_causal=is_causal,
            )
            # w: [B*J, H, T, T] when average_attn_weights=False
            self.last_attn_weights = w.detach() if not w.requires_grad else w
            # Recompute the averaged tensor only if the caller wanted it;
            # AnyTop's call-site discards `w` anyway, so this stays cheap.
            w_out = w.mean(dim=1) if average_attn_weights else w
            return out, w_out
        else:
            return self.wrapped(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )


def install_temporal_taps(model: nn.Module) -> List[TemporalAttnTap]:
    """Replace `layer.temporal_attn` with `TemporalAttnTap` on each decoder layer.

    Idempotent: if taps are already installed, returns the existing list.
    """
    # Find the decoder (AnyTop / AnyTopBehavior both expose `.seqTransDecoder`).
    if not hasattr(model, 'seqTransDecoder'):
        raise AttributeError("model has no `seqTransDecoder`; unsupported architecture.")
    decoder = model.seqTransDecoder
    taps: List[TemporalAttnTap] = []
    for i, layer in enumerate(decoder.layers):
        tattn = getattr(layer, 'temporal_attn', None)
        if tattn is None:
            raise AttributeError(f"decoder layer {i} has no `temporal_attn`.")
        if isinstance(tattn, TemporalAttnTap):
            taps.append(tattn)
            continue
        tap = TemporalAttnTap(tattn)
        # Place tap on same device / dtype as wrapped module, without copying.
        device = next(tattn.parameters()).device
        tap.to(device)
        layer.temporal_attn = tap
        taps.append(tap)
    return taps


def uninstall_temporal_taps(model: nn.Module, taps: List[TemporalAttnTap]) -> None:
    """Restore the original `nn.MultiheadAttention` modules."""
    if not hasattr(model, 'seqTransDecoder'):
        return
    decoder = model.seqTransDecoder
    for i, layer in enumerate(decoder.layers):
        tattn = getattr(layer, 'temporal_attn', None)
        if isinstance(tattn, TemporalAttnTap):
            layer.temporal_attn = tattn.wrapped


def collect_last_attn_weights(taps: List[TemporalAttnTap]) -> List[torch.Tensor]:
    """Return the cached attention tensors from each tap.

    Raises RuntimeError if any tap has not seen a forward pass yet.
    """
    out = []
    for i, tap in enumerate(taps):
        if tap.last_attn_weights is None:
            raise RuntimeError(f"tap {i} has no cached attention; run a forward pass first.")
        out.append(tap.last_attn_weights)
    return out


def clear_last_attn_weights(taps: List[TemporalAttnTap]) -> None:
    for tap in taps:
        tap.last_attn_weights = None


# ------------------------- sparse-map helpers --------------------------------
def sparsify_topk(attn_weights: torch.Tensor, k: int,
                  valid_len: Optional[int] = None) -> torch.Tensor:
    """Top-k sparsify per query row. Zero-out everything else.

    Parameters
    ----------
    attn_weights : [B*J, H, T, T]
    k : number of key entries to retain per (head, query frame).
    valid_len : int or None
        If given, only the first `valid_len` keys (and queries) are considered;
        the rest are set to zero in the returned map.

    Returns
    -------
    sparse : [B*J, H, T, T] with exactly k non-zero entries per (batch, head, query).
    """
    a = attn_weights
    T = a.size(-1)
    if valid_len is not None and valid_len < T:
        mask = torch.zeros_like(a)
        mask[..., :valid_len, :valid_len] = 1.0
        a = a * mask
    k_eff = min(k, T)
    top_vals, top_idx = torch.topk(a, k=k_eff, dim=-1)
    sparse = torch.zeros_like(a)
    sparse.scatter_(-1, top_idx, top_vals)
    # Optional: re-normalise each query row to sum to 1 over its k picks,
    # matching the semantic-guidance formulation. Keep raw mass for simplicity.
    return sparse


def interpolate_time(attn_weights: torch.Tensor, T_out: int) -> torch.Tensor:
    """Bilinear interpolate the last two axes of a [..., T_in, T_in] attn map
    to `[..., T_out, T_out]`.

    Attention maps are on a discrete [0, T_in) x [0, T_in) grid.  Since
    sparse MotionClone maps are indexed by frame, we lift to a continuous
    representation via 2-D bilinear interpolation, then regrid to the
    target frame-count.
    """
    shape = attn_weights.shape
    *lead, T_in, T_in2 = shape
    assert T_in == T_in2, "attn map must be square in time"
    if T_in == T_out:
        return attn_weights
    a = attn_weights.reshape(-1, 1, T_in, T_in)   # [N, 1, T_in, T_in]
    a = torch.nn.functional.interpolate(a, size=(T_out, T_out),
                                        mode='bilinear', align_corners=True)
    return a.reshape(*lead, T_out, T_out)
