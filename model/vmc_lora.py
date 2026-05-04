"""VMC-style temporal-attention LoRA for AnyTopBehavior.

Wraps each ConditionedGraphMotionDecoderLayer.temporal_attn (a torch
nn.MultiheadAttention) with rank-r LoRA deltas on Q, K, V, and output
projections. The frozen base MHA weights are preserved; only the LoRA
A/B matrices are trainable.

Per VMC (arXiv 2312.00845), we fine-tune *only* temporal attention and
add a residual-vector (frame-difference) loss alongside the standard
motion loss. This file provides the module; the residual loss lives in
the training script.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRATemporalMHA(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with LoRA deltas.

    Wraps a *frozen* source MHA and adds four low-rank adapters:
      - ΔW_q, ΔW_k, ΔW_v  (each d×d = B_x @ A_x with rank r)
      - ΔW_o              (out projection)
    The forward signature matches the call site used in
    ``GraphMotionDecoderLayer._temporal_mha_block_sin_joint``:
        attn(x, x, x, attn_mask=..., key_padding_mask=...)
    and returns (attn_output, attn_weights) like nn.MHA.
    """

    def __init__(self, base_mha: nn.MultiheadAttention, rank: int = 16,
                 alpha: int = 32, dropout: float = 0.0):
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

    def lora_params(self):
        return [self.A_q, self.B_q, self.A_k, self.B_k,
                self.A_v, self.B_v, self.A_o, self.B_o]

    def _delta(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (B @ A) * self.scaling  # (d, d)

    def forward(self, query, key, value, attn_mask=None,
                key_padding_mask=None, need_weights: bool = True,
                average_attn_weights: bool = True, is_causal: bool = False):
        # We rebuild in_proj_weight by adding LoRA deltas stacked as [ΔWq;ΔWk;ΔWv].
        W = self.base.in_proj_weight  # (3d, d)
        b = self.base.in_proj_bias    # (3d,)
        d = self.embed_dim

        dWq = self._delta(self.A_q, self.B_q)
        dWk = self._delta(self.A_k, self.B_k)
        dWv = self._delta(self.A_v, self.B_v)
        W_eff = W + torch.cat([dWq, dWk, dWv], dim=0)

        out_W = self.base.out_proj.weight  # (d, d)
        out_b = self.base.out_proj.bias
        out_W_eff = out_W + self._delta(self.A_o, self.B_o)

        # need_weights=False path of F.multi_head_attention_forward is faster; we rarely need weights.
        attn_out, attn_w = F.multi_head_attention_forward(
            query=query, key=key, value=value,
            embed_dim_to_check=d, num_heads=self.num_heads,
            in_proj_weight=W_eff, in_proj_bias=b,
            bias_k=self.base.bias_k, bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.dropout_p if self.training else 0.0,
            out_proj_weight=out_W_eff, out_proj_bias=out_b,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
        )
        return attn_out, attn_w


def inject_temporal_lora(model, rank: int = 16, alpha: int = 32,
                         dropout: float = 0.0) -> List[LoRATemporalMHA]:
    """Replace each decoder layer's ``temporal_attn`` with a LoRATemporalMHA.

    Returns the list of inserted LoRA modules so the caller can collect
    trainable parameters.
    """
    lora_mods: List[LoRATemporalMHA] = []
    decoder = model.seqTransDecoder
    for layer in decoder.layers:
        base = layer.temporal_attn
        lora = LoRATemporalMHA(base, rank=rank, alpha=alpha, dropout=dropout)
        layer.temporal_attn = lora
        lora_mods.append(lora)
    return lora_mods


def freeze_all_but_lora(model, lora_modules: List[LoRATemporalMHA]):
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in lora_modules:
        for p in m.lora_params():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable
