"""Supervised paired CFM training — upper bound reference.

Same architecture as train_cfm.py but trained on actual (source → target)
cross-skeleton pairs. The model sees paired invariant reps of the same action
performed by different skeletons.

This serves as the supervised upper bound in the paper: it measures the
maximum achievable cross-skeleton transfer quality given paired supervision.
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
from model.skel_blind.paired_dataset import PairedInvariantDataset

DEFAULT_SAVE_DIR = "save/cfm_supervised"


def train(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    dataset = PairedInvariantDataset(window=args.window, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InvariantCFM(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_frames=args.window,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Dataset: {len(dataset)} pairs, batch {args.batch_size}, window {args.window}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    start_time = time.time()
    running_loss = 0.0

    log_path = os.path.join(save_dir, "training_log.jsonl")

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model.train()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            z_src, x_1 = batch  # source inv rep, target inv rep
            z_src = z_src.to(device)
            x_1 = x_1.to(device)

            t = torch.rand(x_1.shape[0], device=device)
            t = torch.clamp(t, 1e-5, 1 - 1e-5)

            x_0 = torch.randn_like(x_1)
            x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0

            cond_mask = torch.rand(x_1.shape[0], device=device) < args.p_uncond
            z_src[cond_mask] = 0.0

            with torch.amp.autocast('cuda', dtype=torch.float16):
                x_1_pred = model(x_t, t, z_src)
                pos_loss = nn.functional.mse_loss(x_1_pred[..., :3], x_1[..., :3])
                vel_loss = nn.functional.mse_loss(x_1_pred[..., 4:7], x_1[..., 4:7])
                phase_loss = (1 - torch.cos(x_1_pred[..., 7:8] - x_1[..., 7:8])).mean()
                contact_loss = nn.functional.mse_loss(x_1_pred[..., 3:4], x_1[..., 3:4])
                total_loss = pos_loss + vel_loss + contact_loss + phase_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += total_loss.item()
            step += 1

            if step % 100 == 0:
                avg_loss = running_loss / 100
                elapsed = time.time() - start_time
                sps = step / elapsed
                eta_h = (args.max_steps - step) / sps / 3600

                print(f"[{step:6d}/{args.max_steps}] loss={avg_loss:.5f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"({sps:.1f} it/s, ETA {eta_h:.1f}h)")

                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "loss": round(avg_loss, 5),
                        "lr": round(scheduler.get_last_lr()[0], 7),
                    }) + "\n")
                running_loss = 0.0

            if step % 5000 == 0:
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
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    train(args)
