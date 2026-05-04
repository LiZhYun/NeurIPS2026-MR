"""Stage A tokenizer registry for MoReFlow Stage B.

Loads 70 frozen Stage A VQ-VAE tokenizers; manages encode/decode/codebook access.

Memory: ~3.4 GB resident in fp16 (vs 6.7 GB in fp32). Decoders for L_feat backprop
must run in fp32 — we materialize fp32 copies on demand, or always run in fp32 and
accept the 6.7 GB cost. Default: fp32 (simpler, fits 12 GB 4070 with margin).

Per-clip mean/std normalization handled internally; outputs are differentiable
through the un-normalize step for L_feat.

References:
  model/moreflow/vqvae.py — Stage A model class
  save/moreflow_vqvae/<skel>/{ckpt_best.pt, ckpt_final.pt, args.json}
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.moreflow.vqvae import MoReFlowVQVAE

CKPT_ROOT = PROJECT_ROOT / 'save/moreflow_vqvae'


def _load_one_tokenizer(skel_name, device, prefer_best=True, model_dtype=torch.float32):
    """Load one Stage A tokenizer. Returns (model, mean, std, K_eff, J, downsample_factor)."""
    save_dir = CKPT_ROOT / skel_name
    if prefer_best and (save_dir / 'ckpt_best.pt').exists():
        ckpt_path = save_dir / 'ckpt_best.pt'
    else:
        ckpt_path = save_dir / 'ckpt_final.pt'
    sd = torch.load(ckpt_path, map_location=device)
    a = sd['args']
    model = MoReFlowVQVAE(
        input_dim=sd['feat_dim'],
        hidden=a['hidden'],
        codebook_size=sd['codebook_size_effective'],
        codebook_dim=a['codebook_dim'],
        n_resblocks=a['n_resblocks'],
        n_downsample=a['n_downsample'],
        activation=a['activation'],
        dead_code_threshold=a.get('dead_code_threshold', 5),
    ).to(device)
    model.load_state_dict(sd['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if model_dtype == torch.float16:
        model = model.half()
    mean = torch.from_numpy(np.asarray(sd['mean'], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(sd['std'], dtype=np.float32) + 1e-6).to(device)
    return {
        'model': model,
        'model_dtype': model_dtype,
        'mean': mean,                           # [J, 13]
        'std': std,                             # [J, 13]
        'K_eff': sd['codebook_size_effective'],
        'n_joints': sd['n_joints'],
        'feat_dim': sd['feat_dim'],
        'downsample_factor': sd['downsample_factor'],
    }


class StageARegistry(nn.Module):
    """Holds all 70 frozen Stage A tokenizers; manages encode/decode/codebook access.

    Usage:
        reg = StageARegistry(skels=['Horse', 'Bear', ...], device='cuda')
        z, idx, cb = reg.encode_window('Horse', motion_13d_normalized)
        x_recon = reg.decode_tokens('Horse', z)              # differentiable through D
        x_phys = reg.unnormalize('Horse', x_recon)           # differentiable
    """
    def __init__(self, skels, device='cuda', model_dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.model_dtype = model_dtype
        self.tokenizers = {}                    # skel_name → dict
        for skel in skels:
            try:
                self.tokenizers[skel] = _load_one_tokenizer(skel, self.device, model_dtype=model_dtype)
            except Exception as e:
                print(f"  WARN: failed to load tokenizer for {skel}: {e}")
        self.skels = list(self.tokenizers.keys())
        self.skel_to_id = {s: i for i, s in enumerate(self.skels)}
        # Aggregate stats
        K_effs = [t['K_eff'] for t in self.tokenizers.values()]
        feat_dims = [t['feat_dim'] for t in self.tokenizers.values()]
        n_joints = [t['n_joints'] for t in self.tokenizers.values()]
        print(f"[StageARegistry] loaded {len(self.tokenizers)} tokenizers")
        print(f"  K_eff: min={min(K_effs)}, max={max(K_effs)}")
        print(f"  feat_dim: min={min(feat_dims)}, max={max(feat_dims)}")
        print(f"  n_joints: min={min(n_joints)}, max={max(n_joints)}")
        # Approximate VRAM (fp32)
        total_params = sum(sum(p.numel() for p in t['model'].parameters()) for t in self.tokenizers.values())
        print(f"  total params (fp32): {total_params / 1e6:.1f}M ≈ {total_params * 4 / (1024**3):.2f} GB")

    @property
    def n_skels(self):
        return len(self.skels)

    def get(self, skel_name):
        return self.tokenizers[skel_name]

    def codebook(self, skel_name, padded_to=256):
        """Returns codebook embeddings as a fp32 tensor [K_eff, codebook_dim].

        If padded_to > K_eff, pad with zeros to padded_to.
        """
        t = self.tokenizers[skel_name]
        cb = t['model'].vq._codebook.embed.squeeze(0)             # [K_eff, codebook_dim]
        if padded_to > cb.shape[0]:
            pad = torch.zeros(padded_to - cb.shape[0], cb.shape[1],
                              device=cb.device, dtype=cb.dtype)
            cb = torch.cat([cb, pad], dim=0)
        return cb                                                  # [padded_to, codebook_dim]

    def normalize(self, skel_name, motion_phys_13d):
        """motion_phys_13d: [..., T, J, 13] in physical units → normalized."""
        t = self.tokenizers[skel_name]
        return (motion_phys_13d - t['mean']) / t['std']

    def unnormalize(self, skel_name, motion_norm_13d):
        """motion_norm_13d: [..., T, J, 13] normalized → physical units. Differentiable."""
        t = self.tokenizers[skel_name]
        return motion_norm_13d * t['std'] + t['mean']

    @torch.no_grad()
    def encode_window(self, skel_name, motion_13d_normalized):
        """Encode a (batch of) windows → (z_continuous fp32, indices). Cast to model dtype internally."""
        t = self.tokenizers[skel_name]
        if motion_13d_normalized.dim() == 3:
            motion_13d_normalized = motion_13d_normalized.unsqueeze(0)
        B, T, J, _ = motion_13d_normalized.shape
        flat = motion_13d_normalized.reshape(B, T, J * 13).to(t['model_dtype'])
        z, idx, _ = t['model'].encode(flat)
        return z.float(), idx                                                # always return fp32 z

    def decode_tokens(self, skel_name, z_or_indices):
        """Decode token sequence → fp32 [B, T, J, 13] normalized motion (differentiable)."""
        t = self.tokenizers[skel_name]
        if z_or_indices.dim() == 2:
            z = t['model'].vq.get_codes_from_indices(z_or_indices)           # [B, T_token, codebook_dim]
        else:
            z = z_or_indices
        # Cast input to model dtype (fp16 if half) but preserve gradient via .to()
        z_cast = z.to(t['model_dtype'])
        flat_recon = t['model'].decode(z_cast)
        B, T, _ = flat_recon.shape
        J = t['n_joints']
        # Cast output back to fp32 for L_feat / unnormalize compute
        return flat_recon.float().reshape(B, T, J, 13)

    def ste_quantize(self, skel_name, z_continuous):
        """STE quantize: z_continuous in fp32 → z_q in fp32 (codebook upcast)."""
        cb = self.codebook(skel_name, padded_to=0).float()                   # ensure fp32 for stable cdist
        dists = torch.cdist(z_continuous, cb.unsqueeze(0).expand(z_continuous.shape[0], -1, -1))
        idx = dists.argmin(dim=-1)
        z_q = cb[idx]
        return z_continuous + (z_q - z_continuous).detach()


if __name__ == '__main__':
    # Smoke test on 3 skels
    test_skels = ['Horse', 'Bat', 'Anaconda']
    reg = StageARegistry(test_skels, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nSkels: {reg.skels}")
    print(f"skel_to_id: {reg.skel_to_id}")
    # Try encode/decode round-trip on Horse
    motion_dir = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
    horse_clip = sorted([f for f in motion_dir.iterdir() if f.name.startswith('Horse___')])[0]
    m = np.load(horse_clip).astype(np.float32)[:32]                      # [T=32, J=79, 13]
    m_t = torch.from_numpy(m).to(reg.device)
    m_norm = reg.normalize('Horse', m_t)
    z, idx = reg.encode_window('Horse', m_norm)
    print(f"  Horse: z {z.shape}, idx {idx.shape}, indices range [{idx.min()}, {idx.max()}]")
    x_recon = reg.decode_tokens('Horse', z)
    print(f"  decoded shape: {x_recon.shape}")
    diff = (x_recon.squeeze(0) - m_norm).abs().mean()
    print(f"  recon mean abs diff (normalized): {diff.item():.4f}")
    # Codebook
    cb = reg.codebook('Horse', padded_to=256)
    print(f"  codebook (Horse): {cb.shape}, K_eff={reg.get('Horse')['K_eff']}")
    cb_bat = reg.codebook('Bat', padded_to=256)
    print(f"  codebook (Bat):   {cb_bat.shape}, K_eff={reg.get('Bat')['K_eff']}")
    # STE quantize
    z_q = reg.ste_quantize('Horse', z)
    print(f"  STE z_q shape: {z_q.shape}, "
          f"forward equal to nearest: {torch.allclose(z_q.detach(), reg.codebook('Horse', padded_to=0)[idx])}")
