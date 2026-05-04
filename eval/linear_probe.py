"""Skeleton-ID linear probe (Experiment E4).

Trains a linear classifier on frozen z embeddings to predict skeleton ID.
Lower accuracy = more skeleton-invariant latent = better.

Usage:
  # Extract z embeddings from a trained checkpoint, then probe:
  python eval/linear_probe.py \
      --checkpoint save/A1_full_method_bs_4_latentdim_256/model_final.pt \
      --data_split train \
      --probe_epochs 50

  # Compare A1 (with L_inv) vs B1 (without L_inv):
  python eval/linear_probe.py \
      --checkpoint save/A1_full_method_bs_4_latentdim_256/model_final.pt \
      --compare    save/B1_vae_no_linv_bs_4_latentdim_256/model_final.pt \
      --data_split val
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Z embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_z_embeddings(checkpoint_path, split='val', batch_size=16, device='cuda'):
    """Load a trained AnyTopConditioned checkpoint and extract mean-pooled z.

    Returns:
      z_embs:       [N, D]   numpy float32 — mean-pooled z per motion
      skeleton_ids: [N]      numpy int      — integer skeleton label
      label_names:  list[str]               — label_names[i] = skeleton name for id i
    """
    from utils.model_util import create_conditioned_model_and_diffusion, get_conditioned_args
    from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
    import json, os

    ckpt_dir  = os.path.dirname(checkpoint_path)
    args_path = os.path.join(ckpt_dir, 'args.json')
    with open(args_path) as f:
        args_dict = json.load(f)

    # Reconstruct args namespace
    from argparse import Namespace
    args = Namespace(**args_dict)

    print(f'Loading model from {checkpoint_path}...')
    model, _ = create_conditioned_model_and_diffusion(args)
    state = torch.load(checkpoint_path, map_location='cpu')
    # Handle both raw state dict and wrapped checkpoint dicts
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)

    print(f'Loading {split} data...')
    loader = get_dataset_loader_conditioned(
        batch_size=batch_size, num_frames=args.num_frames,
        temporal_window=args.temporal_window, t5_name='t5-base',
        split=split, balanced=False, objects_subset='all')

    z_list, label_list = [], []
    label_map = {}  # str → int

    for _, cond in loader:
        cond_y = {k: v.to(device) if torch.is_tensor(v) else v
                  for k, v in cond['y'].items()}
        enc_out = model.encoder(
            cond_y['source_motion'],
            cond_y['source_offsets'],
            cond_y['source_joints_mask'],
        )
        z = enc_out[0] if isinstance(enc_out, tuple) else enc_out  # [B, T', K, D]
        z_mean = z.mean(dim=(1, 2)).cpu().numpy()                   # [B, D]
        z_list.append(z_mean)

        for ot in cond_y['object_type']:
            # Strip augmentation suffixes: "Horse__remove[1,2]" → "Horse"
            base_skel = ot.split('__')[0]
            if base_skel not in label_map:
                label_map[base_skel] = len(label_map)
            label_list.append(label_map[base_skel])

    z_embs       = np.concatenate(z_list, axis=0)
    skeleton_ids = np.array(label_list, dtype=np.int64)
    label_names  = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    print(f'  Extracted {len(z_embs)} embeddings, {len(label_names)} skeleton classes')
    return z_embs, skeleton_ids, label_names


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_probe(z_train, y_train, n_classes, epochs=50, lr=1e-3, device='cpu'):
    """Train a single linear layer on frozen z embeddings."""
    D = z_train.shape[1]
    probe = nn.Linear(D, n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.tensor(z_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0:
            print(f'  Probe epoch {ep+1}/{epochs}  loss={total/len(loader):.4f}')

    return probe


def eval_linear_probe(probe, z_test, y_test, device='cpu'):
    """Evaluate probe accuracy on held-out embeddings."""
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.tensor(z_test, dtype=torch.float32).to(device))
        preds  = logits.argmax(dim=-1).cpu().numpy()
    acc = (preds == y_test).mean()
    return float(acc)


def run_probe(checkpoint_path, split='val', probe_epochs=50, train_frac=0.8, seed=42):
    """Full probe pipeline: extract → train/val split → probe → report."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    z_embs, y, label_names = extract_z_embeddings(checkpoint_path, split, device=device)
    n_classes = len(label_names)

    # Train/val split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z_embs))
    n_train = int(len(z_embs) * train_frac)
    i_train, i_val = idx[:n_train], idx[n_train:]

    print(f'\nTraining linear probe ({n_train} train, {len(i_val)} val, {n_classes} classes)...')
    probe = train_linear_probe(z_embs[i_train], y[i_train], n_classes,
                               epochs=probe_epochs, device='cpu')
    acc   = eval_linear_probe(probe, z_embs[i_val], y[i_val])

    print(f'\nSkeleton-ID probe accuracy: {acc:.3f}  '
          f'(chance={1/n_classes:.3f}, lower=more invariant)')
    return {'accuracy': acc, 'chance': 1 / n_classes, 'n_classes': n_classes}


# ─────────────────────────────────────────────────────────────────────────────
# Compare two checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def compare_probes(ckpt_a, ckpt_b, name_a='A1 (full)', name_b='B1 (-L_inv)',
                   split='val', probe_epochs=50):
    """Run probe on two checkpoints and print comparison."""
    print(f'\n=== {name_a} ===')
    res_a = run_probe(ckpt_a, split, probe_epochs)

    print(f'\n=== {name_b} ===')
    res_b = run_probe(ckpt_b, split, probe_epochs)

    print('\n─── Comparison ───')
    print(f'  {name_a}: acc={res_a["accuracy"]:.3f}')
    print(f'  {name_b}: acc={res_b["accuracy"]:.3f}')
    print(f'  Chance:     {res_a["chance"]:.3f}')
    print(f'\n  Lower acc = more skeleton-invariant latent.')
    print(f'  Δacc = {res_b["accuracy"] - res_a["accuracy"]:+.3f}  '
          f'(positive means {name_b} is MORE skeleton-specific)')
    return res_a, res_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to trained AnyTopConditioned model checkpoint')
    parser.add_argument('--compare',      default=None,
                        help='Optional second checkpoint to compare against')
    parser.add_argument('--name_a',       default='A1 (full method)')
    parser.add_argument('--name_b',       default='B1 (VAE -L_inv)')
    parser.add_argument('--data_split',   default='val')
    parser.add_argument('--probe_epochs', type=int, default=50)
    args = parser.parse_args()

    if args.compare:
        compare_probes(args.checkpoint, args.compare,
                       name_a=args.name_a, name_b=args.name_b,
                       split=args.data_split, probe_epochs=args.probe_epochs)
    else:
        run_probe(args.checkpoint, args.data_split, args.probe_epochs)
