"""Per-character VQ-VAE motion tokenizer (MoReFlow Stage A).

Architecture (per arXiv:2509.25600 §3.1, §B):
  Encoder: 1D conv stack — 2 stride-2 downsamples → factor 4 → 32-frame window → 8 tokens
  Quantizer: K=512 codebook, dim=128 (humanoids); K=256, dim=256 (Spot quadruped)
  Decoder: mirror of encoder

Input: motion [B, T, J, C] reshaped to [B, J*C, T]
Output: same shape via decode

Trained per-character on that character's motion clips (no cross-skeleton signal).

References:
  - T2M-GPT (Zhang et al. 2023a) for the core VQ-VAE backbone
  - Dead code reset is EMA-based (threshold_ema_dead_code=2 in vector_quantize_pytorch)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


_ACT = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}


class ResidualBlock1D(nn.Module):
    """Residual block with two 1D convs.

    Paper (Appendix A.1): ReLU activation. Default 'relu' matches paper; 'silu'/'gelu'
    available for ablation.
    """
    def __init__(self, channels, kernel_size=3, dilation=1, activation='relu'):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, dilation=dilation)
        self.act = _ACT[activation]()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class MoReFlowVQVAE(nn.Module):
    """Per-character VQ-VAE for motion tokenization (paper-faithful defaults).

    Defaults follow MoReFlow Appendix A.1 (Spot quadruped config — better fit for
    Truebones non-humanoid morphologies than the humanoid config):
      hidden=512, codebook_dim=256, n_resblocks=3, activation='relu'

    Parameters:
      input_dim: per-frame feature dim (J*C — varies per character)
      hidden: hidden channel size (paper: 256 for humanoid, 512 for spot)
      codebook_size: K (paper: 512 humanoid, 256 spot; here adapted per-skel)
      codebook_dim: embedding dim per codebook entry (paper: 128 humanoid, 256 spot)
      n_resblocks: residual blocks per stage (paper: 3)
      n_downsample: stride-2 downsamples (paper: 2 → 32-frame window → 8 tokens)
      activation: 'relu' (paper) | 'silu' | 'gelu'
      dead_code_threshold: EMA threshold below which a code is replaced from random
                           batch features (paper: periodic re-init; library equiv = 5)
    """
    def __init__(self, input_dim, hidden=512, codebook_size=256, codebook_dim=256,
                 n_resblocks=3, n_downsample=2, activation='relu',
                 dead_code_threshold=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.n_downsample = n_downsample
        self.activation = activation
        self.downsample_factor = 2 ** n_downsample  # default 4: 32-frame window → 8 tokens

        # Encoder: input_proj → [resblock+downsample]^n_downsample → bottleneck
        self.input_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)
        enc_layers = []
        for _ in range(n_downsample):
            for _ in range(n_resblocks):
                enc_layers.append(ResidualBlock1D(hidden, activation=activation))
            enc_layers.append(nn.Conv1d(hidden, hidden, kernel_size=4, stride=2, padding=1))
        self.encoder = nn.Sequential(*enc_layers)
        self.enc_proj = nn.Conv1d(hidden, codebook_dim, kernel_size=1)

        # VQ — commitment_weight=0.02 per MoReFlow paper Appendix A.1 Table 2.
        # threshold_ema_dead_code=5 approximates the paper's periodic re-init from
        # random batch features (Codex retracted M7 advice for 0.25 in prior round).
        self.vq = VectorQuantize(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=0.99,
            commitment_weight=0.02,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=dead_code_threshold,
        )

        # Decoder: mirror
        self.dec_proj = nn.Conv1d(codebook_dim, hidden, kernel_size=1)
        dec_layers = []
        for _ in range(n_downsample):
            dec_layers.append(nn.ConvTranspose1d(hidden, hidden, kernel_size=4,
                                                   stride=2, padding=1))
            for _ in range(n_resblocks):
                dec_layers.append(ResidualBlock1D(hidden, activation=activation))
        self.decoder = nn.Sequential(*dec_layers)
        self.output_proj = nn.Conv1d(hidden, input_dim, kernel_size=1)

    def encode(self, x):
        """x: [B, T, input_dim] → tokens [B, T//8], indices into codebook."""
        x = x.transpose(1, 2)  # [B, input_dim, T]
        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.enc_proj(h)  # [B, codebook_dim, T//8]
        h = h.transpose(1, 2)  # [B, T//8, codebook_dim]
        z_q, indices, vq_loss = self.vq(h)
        return z_q, indices, vq_loss

    def decode(self, z_q):
        """z_q: [B, T//8, codebook_dim] → motion [B, T, input_dim]"""
        h = z_q.transpose(1, 2)  # [B, codebook_dim, T//8]
        h = self.dec_proj(h)
        h = self.decoder(h)
        h = self.output_proj(h)
        return h.transpose(1, 2)  # [B, T, input_dim]

    def forward(self, x):
        """Full encode-decode for training. Returns (recon, vq_loss, indices)."""
        z_q, indices, vq_loss = self.encode(x)
        recon = self.decode(z_q)
        return recon, vq_loss, indices

    def decode_indices(self, indices):
        """Decode from token indices [B, T//8] → motion."""
        # Get embeddings from codebook
        z_q = self.vq.get_codes_from_indices(indices)  # [B, T//8, codebook_dim]
        return self.decode(z_q)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
