"""prop:cm empirical demonstration via paired-support ladder + per-cell variance.

Hypothesis (prop:cm): under squared-error sparse paired regression, the trained
predictor converges to E[y | (skel_a, skel_b, action)] (the cell mean), so
different source clips within the same cell collapse to the same target output.

Empirical test:
  Train an MLP regressor f_theta(x_a, skel_a, skel_b, action) on the synthetic
  2x2 transport problem with paired support sizes
      M ∈ {1, 2, 4, 8, 16, 32, 50}  pairs per cell
  After training, generate outputs for held-out source clips that share a
  (skel_b, action) cell with multiple training sources. Measure:

    var_per_cell  : avg over (skel_b, action) cells of var{f_theta(x_a)} where
                    x_a varies over distinct source clips of the same cell
    norm_var_ratio: var_per_cell / var_per_cell_oracle (where oracle = T*(x_a))

  prop:cm prediction:
    M=1   →  norm_var_ratio ≈ 0   (only one paired example per cell, model
                                    converges to that exact example, but unseen
                                    sources collapse to its prediction)
    M=2   →  small
    M=50  →  norm_var_ratio approaches 1  (the model learns source-conditional
                                            transport because each source has a
                                            distinct paired target)

Output: JSON with the variance ladder per M.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.synthetic_2x2.build_synth_dataset import (
    K_SKELETONS, JOINTS_PER_SKEL, ACTIONS, T_FRAMES, action_trajectory, true_transport,
)


# ---------- Tiny squared-error regressor ----------

class TinyRegressor(nn.Module):
    """Direct-output MSE regressor f(x_a, sa, sb, action) -> x_b.
    Uses average pooling + per-skel embeddings + small MLP. Sufficient for the
    synthetic transport problem (which is deterministic per cell)."""
    def __init__(self, j_max, t_frames, n_skels, n_actions, d_emb=32, hidden=128):
        super().__init__()
        self.j_max = j_max
        self.t_frames = t_frames
        self.skel_emb = nn.Embedding(n_skels, d_emb)
        self.action_emb = nn.Embedding(n_actions, d_emb)
        flat = j_max * t_frames * 3
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 3 * d_emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, flat),
        )

    def forward(self, x_a, sa, sb, action):
        b = x_a.shape[0]
        x = x_a.view(b, -1)
        h = self.encoder(x)
        cond = torch.cat([self.skel_emb(sa), self.skel_emb(sb), self.action_emb(action)], -1)
        out = self.head(torch.cat([h, cond], -1))
        return out.view(b, self.t_frames, self.j_max, 3)


# ---------- Build paired data with M pairs per cell ----------

def build_paired_data(n_pairs_per_cell: int, j_max: int, seed: int):
    """Build paired (x_a, x_b, sa, sb, action) data with `n_pairs_per_cell`
    distinct source clips per (sa, sb, action) cell. Each source clip gets paired
    with exactly one target via T*."""
    rng = np.random.RandomState(seed)
    src_list = []
    tgt_list = []
    sa_list = []
    sb_list = []
    a_list = []
    cell_id = []  # (sb, action) tuple — used for variance evaluation
    for sa in range(K_SKELETONS):
        for sb in range(K_SKELETONS):
            if sa == sb:
                continue
            for ai, action in enumerate(ACTIONS):
                for k in range(n_pairs_per_cell):
                    cs = seed * 100003 + sa * 1009 + sb * 53 + ai * 7 + k * 3
                    xa = action_trajectory(action, JOINTS_PER_SKEL[sa], T_FRAMES, cs,
                                            add_instance_noise=True)
                    xb = true_transport(xa, JOINTS_PER_SKEL[sb], action)
                    pad_a = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_a[:, :xa.shape[1], :] = xa
                    pad_b = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_b[:, :xb.shape[1], :] = xb
                    src_list.append(pad_a)
                    tgt_list.append(pad_b)
                    sa_list.append(sa)
                    sb_list.append(sb)
                    a_list.append(ai)
                    cell_id.append((sb, ai))
    return (
        np.stack(src_list), np.stack(tgt_list),
        np.array(sa_list), np.array(sb_list), np.array(a_list),
        cell_id,
    )


def build_held_out_eval(n_eval_per_cell: int, j_max: int, seed: int):
    """Held-out source clips (different RNG seed offset). For each (sa, sb, action)
    cell, build n_eval_per_cell distinct held-out source clips. Used to measure
    output variance on UNSEEN sources within the same cell."""
    rng = np.random.RandomState(seed + 9999)
    by_cell = {}  # (sa, sb, ai) -> list of (xa_padded, xa_real_J, xb_target)
    for sa in range(K_SKELETONS):
        for sb in range(K_SKELETONS):
            if sa == sb:
                continue
            for ai, action in enumerate(ACTIONS):
                samples = []
                for k in range(n_eval_per_cell):
                    cs = (seed + 9999) * 100003 + sa * 1009 + sb * 53 + ai * 7 + k * 3
                    xa = action_trajectory(action, JOINTS_PER_SKEL[sa], T_FRAMES, cs,
                                            add_instance_noise=True)
                    xb = true_transport(xa, JOINTS_PER_SKEL[sb], action)
                    pad_a = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_a[:, :xa.shape[1], :] = xa
                    pad_b = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_b[:, :xb.shape[1], :] = xb
                    samples.append((pad_a, JOINTS_PER_SKEL[sa], pad_b, JOINTS_PER_SKEL[sb]))
                by_cell[(sa, sb, ai)] = samples
    return by_cell


# ---------- Train + eval ----------

def train_and_measure(n_pairs_per_cell: int, max_steps: int, batch_size: int,
                       device: str, seed: int):
    j_max = max(JOINTS_PER_SKEL)
    src, tgt, sa, sb, ai, _ = build_paired_data(n_pairs_per_cell, j_max, seed)
    held_eval = build_held_out_eval(n_eval_per_cell=8, j_max=j_max, seed=seed)
    print(f"  M={n_pairs_per_cell}: paired N = {len(src)}; held-out cells = {len(held_eval)}")

    dev = torch.device(device)
    model = TinyRegressor(j_max, T_FRAMES, K_SKELETONS, len(ACTIONS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    src_t = torch.from_numpy(src).to(dev)
    tgt_t = torch.from_numpy(tgt).to(dev)
    sa_t = torch.from_numpy(sa).long().to(dev)
    sb_t = torch.from_numpy(sb).long().to(dev)
    ai_t = torch.from_numpy(ai).long().to(dev)
    n = len(src)
    rng = np.random.RandomState(seed + 1)
    t0 = time.time()
    for step in range(max_steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        x_pred = model(src_t[idx], sa_t[idx], sb_t[idx], ai_t[idx])
        loss = F.mse_loss(x_pred, tgt_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, max_steps // 4) == 0:
            print(f"    step {step+1}/{max_steps}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")

    # Eval: per-cell output variance over held-out sources
    model.eval()
    out_vars = []
    src_vars = []
    oracle_vars = []
    output_minus_oracle_mse = []
    cell_count = 0
    with torch.no_grad():
        for (sa_c, sb_c, ai_c), samples in held_eval.items():
            if len(samples) < 2:
                continue
            preds = []
            sources = []
            oracles = []
            for pad_a, real_J_a, pad_b_oracle, real_J_b in samples:
                xa_t = torch.from_numpy(pad_a).unsqueeze(0).to(dev)
                pred = model(xa_t,
                              torch.tensor([sa_c], device=dev).long(),
                              torch.tensor([sb_c], device=dev).long(),
                              torch.tensor([ai_c], device=dev).long())
                pred_np = pred[0, :, :real_J_b, :].cpu().numpy()  # crop to real joints
                preds.append(pred_np)
                sources.append(pad_a[:, :real_J_a, :])
                oracles.append(pad_b_oracle[:, :real_J_b, :])
            # Per-cell variance: average per-frame, per-joint variance across sources
            pred_arr = np.stack(preds)        # [n_src, T, J_b, 3]
            src_arr = np.stack(sources)       # [n_src, T, J_a, 3]  (different J_a per skel; constant within sa_c)
            oracle_arr = np.stack(oracles)    # [n_src, T, J_b, 3]
            out_vars.append(float(np.mean(pred_arr.var(axis=0))))
            src_vars.append(float(np.mean(src_arr.var(axis=0))))
            oracle_vars.append(float(np.mean(oracle_arr.var(axis=0))))
            mse = float(np.mean((pred_arr - oracle_arr) ** 2))
            output_minus_oracle_mse.append(mse)
            cell_count += 1
    out_var = float(np.mean(out_vars)) if out_vars else float('nan')
    src_var = float(np.mean(src_vars)) if src_vars else float('nan')
    oracle_var = float(np.mean(oracle_vars)) if oracle_vars else float('nan')
    return {
        'M_pairs_per_cell': n_pairs_per_cell,
        'cells_evaluated': cell_count,
        'out_var_per_cell': out_var,
        'oracle_var_per_cell': oracle_var,
        'src_var_per_cell': src_var,
        'norm_var_ratio_out_over_oracle': out_var / max(oracle_var, 1e-9),
        'mse_out_vs_oracle': float(np.mean(output_minus_oracle_mse)) if output_minus_oracle_mse else float('nan'),
        'training_time_s': time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_json', required=True)
    p.add_argument('--ladder', type=int, nargs='+',
                    default=[1, 2, 4, 8, 16, 32, 50])
    p.add_argument('--max_steps', type=int, default=1500)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    print(f"[prop:cm ladder] device={args.device}  ladder={args.ladder}  max_steps={args.max_steps}")
    rows = []
    for M in args.ladder:
        print(f"\n=== M = {M} pairs per cell ===")
        try:
            row = train_and_measure(M, args.max_steps, args.batch_size, args.device, args.seed)
            rows.append(row)
            print(f"  → out_var={row['out_var_per_cell']:.5f}  oracle_var={row['oracle_var_per_cell']:.5f}  "
                  f"ratio={row['norm_var_ratio_out_over_oracle']:.4f}  mse_vs_oracle={row['mse_out_vs_oracle']:.4f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    out = {
        'description': 'prop:cm empirical demonstration. Squared-error regressor on synthetic 2x2 transport with M pairs per (skel_a, skel_b, action) cell. Held-out sources tested for per-cell output variance.',
        'config': {
            'max_steps': args.max_steps, 'batch_size': args.batch_size,
            'seed': args.seed, 'ladder': args.ladder,
        },
        'rows': rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[prop:cm ladder] saved {args.out_json}")
    print(f"\nLadder summary (M : norm_var_ratio):")
    for r in rows:
        print(f"  M={r['M_pairs_per_cell']:4d}  ratio={r['norm_var_ratio_out_over_oracle']:.4f}  mse_vs_oracle={r['mse_out_vs_oracle']:.4f}")


if __name__ == '__main__':
    main()
