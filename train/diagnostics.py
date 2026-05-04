import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fsq_stats(z_codes):
    """
    z_codes: [B, N', K, fsq_dims] long tensor, values in 0..L-1.
    Returns dict with fsq_utilization and fsq_entropy_norm.
    """
    L = int(z_codes.max().item()) + 1
    fsq_dims = z_codes.shape[-1]

    # Per-dim marginal entropy, normalized by log(L)
    entropies = []
    for d in range(fsq_dims):
        vals = z_codes[..., d].reshape(-1)
        counts = torch.bincount(vals, minlength=L).float()
        probs = counts / counts.sum()
        probs = probs.clamp(min=1e-10)
        h = -(probs * probs.log()).sum().item()
        entropies.append(h / math.log(L))

    # Fraction of L^K_fsq possible code tuples that appear in this batch
    total_possible = L ** fsq_dims
    flat = z_codes.view(-1, fsq_dims)
    multipliers = torch.tensor(
        [L ** i for i in range(fsq_dims)], device=z_codes.device, dtype=torch.long)
    codes_int = (flat * multipliers).sum(dim=1)
    utilization = codes_int.unique().numel() / total_possible

    return {
        'fsq_utilization': float(utilization),
        'fsq_entropy_norm': float(np.mean(entropies)),
    }


def slot_diversity(z_embed):
    """
    z_embed: [B, N', K, D]
    Returns mean pairwise cosine distance across K slots (higher = more diverse).
    """
    B, Np, K, D = z_embed.shape
    z = z_embed.detach().view(B * Np, K, D)
    z_norm = F.normalize(z, dim=-1)
    sim = torch.bmm(z_norm, z_norm.transpose(1, 2))  # [B*N', K, K]
    off_diag = ~torch.eye(K, dtype=torch.bool, device=z.device)
    return (1.0 - sim[:, off_diag].mean()).item()


def null_z_divergence(null_z, z_embed):
    """
    null_z:  [1, 1, K, D] learnable parameter
    z_embed: [B, N', K, D]
    Returns mean L2 norm of (z_embed - null_z).
    """
    return (z_embed.detach() - null_z.detach()).norm(dim=-1).mean().item()


def collect_z_embeddings(model, data_loader, device, n_batches=30):
    """
    Iterate data_loader, encode source motion, return mean-pooled z vectors and labels.
    Caller must set model.eval() before calling.

    Returns: (z_flat [N, D], labels [N]) or (None, None) if no data.
    """
    z_list, label_list = [], []
    with torch.no_grad():
        for i, (_, cond) in enumerate(data_loader):
            if i >= n_batches:
                break
            cond_y = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in cond['y'].items()}
            enc_out = model.encoder(
                cond_y['source_motion'],
                cond_y['source_offsets'],
                cond_y['source_joints_mask'],
            )
            # VAE mode returns (z_out, mu, logvar); FSQ mode returns z_out
            z = enc_out[0] if isinstance(enc_out, tuple) else enc_out  # [B, N', K, D]
            z_list.append(z.mean(dim=(1, 2)).cpu().numpy())  # [B, D]
            labels = cond_y['object_type']
            label_list.extend(labels if isinstance(labels, list) else [labels])
    if not z_list:
        return None, None
    return np.concatenate(z_list, axis=0), label_list


def vae_stats(mu, logvar):
    """
    mu, logvar: [B, N', K, D_z]
    Returns dict with mu_var (latent variance — proxy for collapse) and mean_kl.
    """
    import torch
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return {
        'mu_var':  float(mu.var().item()),
        'mean_kl': float(kl.sum(-1).mean().item()),
    }


def z_pca_figure(z_flat, labels):
    """
    PCA scatter of z embeddings, colored by animal label.
    z_flat: [N, D] numpy array, labels: list of N strings.
    Returns matplotlib Figure.
    """
    if len(z_flat) < 3:
        return None
    z_centered = z_flat - z_flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z2 = z_centered @ Vt[:2].T  # [N, 2]
    var = np.var(z2, axis=0)
    total_var = np.var(z_centered)
    var_ratio = var / (total_var + 1e-8)

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        idxs = [j for j, l in enumerate(labels) if l == label]
        ax.scatter(z2[idxs, 0], z2[idxs, 1], label=label, s=20, alpha=0.7, color=colors[i])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.set_title('z PCA — colored by source animal')
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%} var)')
    fig.tight_layout()
    return fig


def slot_attn_figure(attn_weights, joint_labels):
    """
    Heatmap of attention pool weights.
    attn_weights: [T, K, J] or [K, J] tensor or ndarray
    joint_labels: list of J label strings
    Returns matplotlib Figure.
    """
    if isinstance(attn_weights, torch.Tensor):
        attn = attn_weights.detach().cpu().numpy()
    else:
        attn = attn_weights
    if attn.ndim == 3:
        attn = attn.mean(axis=0)  # [K, J]
    K, J = attn.shape

    fig, ax = plt.subplots(figsize=(max(6, J * 0.25), max(2, K * 0.8)))
    im = ax.imshow(attn, aspect='auto', cmap='Blues')
    ax.set_xticks(range(J))
    ax.set_xticklabels(joint_labels[:J], rotation=90, fontsize=7)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f'slot {k}' for k in range(K)])
    ax.set_title('Attention pool: slots → joints')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
