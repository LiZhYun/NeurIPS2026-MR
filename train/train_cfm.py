"""Training loop for target-predictive CFM on invariant motion reps.

Target-predictive CFM (FlowMotion arXiv:2504.01338):
  x_t = t * x_1 + (1-t) * x_0    (linear interpolation, x_0 ~ N(0,1))
  Model predicts x_1 from (x_t, t, z_src)
  Loss = ||x_1_pred - x_1||^2

Self-reconstruction: z_src = x_1 (source = target, unpaired training).
Classifier-free guidance: randomly drop z_src with probability p_uncond.

Phase 1: CFM + recon loss.
Phase 2 (later): add contact BCE + phase MSE auxiliary heads.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.skel_blind.cfm_model import InvariantCFM, count_parameters
from model.skel_blind.invariant_dataset import InvariantMotionDataset

DEFAULT_SAVE_DIR = "save/cfm_v1"


def inverse_lipschitz_penalty(model, x_1_pred, z_src, alpha=0.1):
    """Soft inverse-Lipschitz penalty on G_theta output (spec §2.2, condition C3).

    Uses the already-computed output x_1_pred to avoid extra forward passes.
    Computes on adjacent batch pairs: ||out_i - out_{i+1}|| / ||z_i - z_{i+1}||.
    """
    B = x_1_pred.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=x_1_pred.device)

    out_diff = (x_1_pred[:-1] - x_1_pred[1:]).reshape(B - 1, -1).norm(dim=1)
    with torch.no_grad():
        z_diff = (z_src[:-1] - z_src[1:]).reshape(B - 1, -1).norm(dim=1)

    ratio = out_diff / (z_diff + 1e-8)
    penalty = torch.clamp(alpha - ratio, min=0).mean()
    return penalty


def train(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    dataset = InvariantMotionDataset(window=args.window, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InvariantCFM(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_frames=args.window,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Dataset: {len(dataset)} clips, batch {args.batch_size}, window {args.window}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    start_time = time.time()
    log_interval = 100
    save_interval = 5000
    running_loss = 0.0
    running_cfm = 0.0
    running_il = 0.0

    log_path = os.path.join(save_dir, "training_log.jsonl")

    model.train()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            x_1 = batch.to(device)  # [B, T, S, C]

            t = torch.rand(x_1.shape[0], device=device)
            t = torch.clamp(t, 1e-5, 1 - 1e-5)

            x_0 = torch.randn_like(x_1)
            x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0

            z_src = x_1.clone()
            cond_mask = torch.rand(x_1.shape[0], device=device) < args.p_uncond
            z_src[cond_mask] = 0.0

            with torch.amp.autocast('cuda', dtype=torch.float16):
                x_1_pred = model(x_t, t, z_src)
                pos_loss = nn.functional.mse_loss(x_1_pred[..., :3], x_1[..., :3])
                vel_loss = nn.functional.mse_loss(x_1_pred[..., 4:7], x_1[..., 4:7])
                phase_loss = (1 - torch.cos(x_1_pred[..., 7:8] - x_1[..., 7:8])).mean()
                if args.contact_bce:
                    contact_loss = nn.functional.binary_cross_entropy_with_logits(
                        x_1_pred[..., 3:4], x_1[..., 3:4])
                else:
                    contact_loss = nn.functional.mse_loss(x_1_pred[..., 3:4], x_1[..., 3:4])
                cfm_loss = (pos_loss + vel_loss
                            + args.lambda_contact * contact_loss
                            + args.lambda_phase * phase_loss)

            il_loss = torch.tensor(0.0, device=device)
            if args.lambda_il > 0 and step > 1000:
                il_loss = inverse_lipschitz_penalty(model, x_1_pred, z_src)

            total_loss = cfm_loss + args.lambda_il * il_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += total_loss.item()
            running_cfm += cfm_loss.item()
            running_il += il_loss.item()
            step += 1

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_cfm = running_cfm / log_interval
                avg_il = running_il / log_interval
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta_hours = (args.max_steps - step) / steps_per_sec / 3600

                entry = {
                    "step": step,
                    "loss": round(avg_loss, 5),
                    "cfm": round(avg_cfm, 5),
                    "il": round(avg_il, 5),
                    "lr": round(scheduler.get_last_lr()[0], 7),
                    "steps_per_sec": round(steps_per_sec, 2),
                    "eta_h": round(eta_hours, 2),
                }
                print(f"[{step:6d}/{args.max_steps}] loss={avg_loss:.5f} cfm={avg_cfm:.5f} il={avg_il:.5f} lr={scheduler.get_last_lr()[0]:.2e} ({steps_per_sec:.1f} it/s, ETA {eta_hours:.1f}h)")

                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                running_loss = 0.0
                running_cfm = 0.0
                running_il = 0.0

            if step % save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_{step:06d}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

    final_path = os.path.join(save_dir, "ckpt_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--lambda_il", type=float, default=0.01)
    parser.add_argument("--lambda_contact", type=float, default=1.0)
    parser.add_argument("--lambda_phase", type=float, default=1.0)
    parser.add_argument("--contact_bce", action="store_true")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    train(args)
