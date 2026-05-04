"""Generate cross-skeleton motion transfers using trained CFM model.

Given source motion on skeleton A, generates the corresponding motion
in invariant space. The invariant rep can then be decoded on any target
skeleton via FK or IK.

Sampling: Euler ODE integration from noise x_0 to clean x_1.
Classifier-free guidance: x_1_pred = (1 + w) * cond_pred - w * uncond_pred
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.cfm_model import InvariantCFM, SLOT_COUNT, CHANNEL_COUNT
from model.skel_blind.encoder import encode_motion_to_invariant


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    model = InvariantCFM(
        d_model=args.get("d_model", 384),
        n_layers=args.get("n_layers", 12),
        n_heads=args.get("n_heads", 8),
        max_frames=args.get("window", 40),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args


@torch.no_grad()
def sample_euler(model, z_src, n_steps=50, cfg_weight=2.0):
    """Euler ODE integration for target-predictive CFM.

    In target-predictive CFM, the velocity at time t is:
      v(x_t, t) = (x_1_pred(x_t, t) - x_t) / (1 - t)

    We integrate from t=0 (noise) to t=1 (clean).
    """
    B, T, S, C = z_src.shape
    device = z_src.device
    x = torch.randn(B, T, S, C, device=device)

    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t_tensor = torch.full((B,), t_val, device=device)

        # Conditional prediction
        x_1_cond = model(x, t_tensor, z_src)

        if cfg_weight > 0:
            # Unconditional prediction
            z_null = torch.zeros_like(z_src)
            x_1_uncond = model(x, t_tensor, z_null)
            x_1_pred = (1 + cfg_weight) * x_1_cond - cfg_weight * x_1_uncond
        else:
            x_1_pred = x_1_cond

        # Velocity and step
        denom = max(1.0 - t_val, 1e-5)
        v = (x_1_pred - x) / denom
        x = x + v * dt

    return x


def generate_transfer(model, source_inv, device, n_steps=50, cfg_weight=2.0):
    """Generate invariant rep conditioned on source invariant rep."""
    z_src = torch.from_numpy(source_inv).float().unsqueeze(0).to(device)
    out = sample_euler(model, z_src, n_steps=n_steps, cfg_weight=cfg_weight)
    return out[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--source_skel", type=str, required=True)
    parser.add_argument("--source_motion", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--cfg_weight", type=float, default=2.0)
    parser.add_argument("--out_dir", type=str, default="save/cfm_v1/samples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.ckpt, device)
    window = model_args.get("window", 40)

    DATA_ROOT = "dataset/truebones/zoo/truebones_processed"
    cond_dict = np.load(os.path.join(DATA_ROOT, "cond.npy"), allow_pickle=True).item()
    cond = {
        "joints_names": cond_dict[args.source_skel]["joints_names"],
        "parents": cond_dict[args.source_skel]["parents"],
        "object_type": args.source_skel,
    }

    motion = np.load(os.path.join(DATA_ROOT, "motions", args.source_motion))
    inv_src = encode_motion_to_invariant(motion, cond)

    T_src = inv_src.shape[0]
    if T_src > window:
        inv_src = inv_src[:window]
    elif T_src < window:
        pad = np.zeros((window - T_src, SLOT_COUNT, CHANNEL_COUNT), dtype=np.float32)
        inv_src = np.concatenate([inv_src, pad], axis=0)

    inv_gen = generate_transfer(model, inv_src, device,
                                n_steps=args.n_steps, cfg_weight=args.cfg_weight)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"gen_{args.source_skel}_{os.path.splitext(args.source_motion)[0]}.npz")
    np.savez_compressed(out_path, inv_src=inv_src, inv_gen=inv_gen,
                        source_skel=args.source_skel, source_motion=args.source_motion)
    print(f"Generated: {out_path}, shape={inv_gen.shape}")


if __name__ == "__main__":
    main()
