import functools
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusion.resample import LossAwareSampler
from utils import dist_util
from train.training_loop import TrainLoop, log_loss_dict
from train.diagnostics import (
    slot_diversity, null_z_divergence,
    collect_z_embeddings, z_pca_figure, slot_attn_figure,
)


def _augment_view(source_motion, source_mask):
    """Create a second view of source_motion with STRONG stochastic augmentations.

    source_motion: [B, J, 13, T]
    source_mask:   [B, J]  True=real joint

    Augmentations — stronger than v1 but not destructive.
    Target: cosine sim ~0.4-0.7 between original and augmented.

      - Feature noise:    σ=0.03 (was 0.01)
      - Token dropout:    zero 15-25% of joint tokens (was 10-15%)
      - Frame drop:       zero 10% of frames (was 5%)
      - Temporal warp:    scale s ~ U[0.75, 1.25] (was 0.85-1.15)
      - Temporal shift:   circular shift by ±8% of T
    """
    B, J, nf, T = source_motion.shape
    out = source_motion.clone()

    # Feature noise
    out = out + torch.randn_like(out) * 0.03

    # Token dropout: zero ~20% of real joints (was ~12%)
    drop_rate = torch.empty(B, device=out.device).uniform_(0.15, 0.25)
    for b in range(B):
        real_joints = source_mask[b].nonzero(as_tuple=True)[0]
        n_drop = max(1, int(drop_rate[b].item() * len(real_joints)))
        perm = torch.randperm(len(real_joints), device=out.device)[:n_drop]
        drop_idx = real_joints[perm]
        out[b, drop_idx] = 0.0

    # Frame drop: zero ~10% of frames (was 5%)
    n_drop_frames = max(1, int(0.10 * T))
    drop_frames = torch.randperm(T, device=out.device)[:n_drop_frames]
    out[:, :, :, drop_frames] = 0.0

    # Temporal shift: circular shift by ±8% of T
    max_shift = max(1, T // 12)
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    out = torch.roll(out, shifts=shift, dims=-1)

    # Temporal warp (wider range)
    scale = torch.empty(1, device=out.device).uniform_(0.75, 1.25).item()
    T_new = max(4, int(T * scale))
    flat = out.view(B, J * nf, T)
    warped = F.interpolate(flat, size=T_new, mode='linear', align_corners=False)
    out = F.interpolate(warped, size=T, mode='linear', align_corners=False).view(B, J, nf, T)

    return out


def _free_bits_kl(mu, logvar, tau=0.5):
    """Per-token KL with free-bits threshold tau (nats).

    Returns mean KL per token, clamped to >= tau (encourages each token
    to use at least tau nats before the loss forces it to zero).

    mu, logvar: [B, T', K, D_z]
    Returns scalar.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, T', K, D_z]
    kl_per_token = kl_per_dim.sum(dim=-1)                          # [B, T', K]
    return kl_per_token.clamp(min=tau).mean()


class MomentumBank:
    """Fixed-size queue of past embeddings for scaling InfoNCE negatives.

    Maintains a FIFO buffer of L2-normalized embeddings. Updated each step
    with new embeddings pushed in and oldest entries evicted.
    """
    def __init__(self, dim, size=512):
        self.size = size
        self.dim = dim
        self.bank = None  # lazy init on first push
        self.ptr = 0

    @torch.no_grad()
    def push(self, embeddings):
        """Push [N, D] normalized embeddings into the bank."""
        embeddings = embeddings.detach()
        N = embeddings.shape[0]
        if self.bank is None:
            self.bank = torch.zeros(self.size, self.dim, device=embeddings.device)
        for i in range(N):
            self.bank[self.ptr] = embeddings[i]
            self.ptr = (self.ptr + 1) % self.size

    def get(self):
        """Return current bank contents [size, D] as an independent copy.

        Must clone to avoid version conflicts: push() modifies self.bank
        in-place, which would bump the version of any view/detach sharing
        the same storage — breaking autograd if the returned tensor is used
        in a computation graph.
        """
        if self.bank is None:
            return None
        return self.bank.detach().clone()


def _view_consistency_loss(mu1, mu2, temperature=0.1, bank=None):
    """InfoNCE contrastive loss with optional memory bank for more negatives.

    Positives: (mu1[i], mu2[i]) — two views of the same motion.
    Negatives: all other motions in the batch + memory bank entries.

    With B=4 and bank_size=512, this provides ~516 negatives instead of 6.

    mu1, mu2: [B, T', K, D_z]
    bank: MomentumBank or None
    Returns scalar.
    """
    B = mu1.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=mu1.device)

    v1 = F.normalize(mu1.reshape(B, -1), dim=-1)  # [B, D]
    v2 = F.normalize(mu2.reshape(B, -1), dim=-1)  # [B, D]

    if bank is not None:
        bank_embs = bank.get()  # [M, D] or None
    else:
        bank_embs = None

    if bank_embs is not None and bank_embs.abs().sum() > 0:
        # InfoNCE with memory bank: each anchor scores against its positive + all negatives
        # Negatives = other in-batch views + entire bank
        bank_embs = bank_embs.to(v1.device)
        M = bank_embs.shape[0]

        # For v1[i], positive is v2[i], negatives are v2[j!=i] + bank
        # Compute all scores
        pos_scores = (v1 * v2).sum(dim=-1, keepdim=True) / temperature  # [B, 1]
        neg_in_batch = (v1 @ v2.T) / temperature  # [B, B]
        neg_bank = (v1 @ bank_embs.T) / temperature  # [B, M]

        # Mask out positive from in-batch negatives
        mask = torch.eye(B, dtype=torch.bool, device=v1.device)
        neg_in_batch = neg_in_batch.masked_fill(mask, -1e9)

        # logits: [B, 1 + (B-1) + M] — positive first
        logits = torch.cat([pos_scores, neg_in_batch, neg_bank], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=v1.device)  # positive is at index 0

        loss_v1 = F.cross_entropy(logits, labels)

        # Symmetric: v2 anchors
        pos_scores2 = (v2 * v1).sum(dim=-1, keepdim=True) / temperature
        neg_in_batch2 = (v2 @ v1.T) / temperature
        neg_bank2 = (v2 @ bank_embs.T) / temperature
        neg_in_batch2 = neg_in_batch2.masked_fill(mask, -1e9)
        logits2 = torch.cat([pos_scores2, neg_in_batch2, neg_bank2], dim=1)
        loss_v2 = F.cross_entropy(logits2, labels)

        # Update bank with current embeddings
        bank.push(torch.cat([v1, v2], dim=0))

        return (loss_v1 + loss_v2) / 2
    else:
        # Fallback: standard InfoNCE (small batch, no bank yet)
        embs = torch.cat([v1, v2], dim=0)
        sim = (embs @ embs.T) / temperature
        mask = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(mask, -1e9)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=sim.device),
            torch.arange(0, B, device=sim.device),
        ])
        # Bootstrap bank
        if bank is not None:
            bank.push(torch.cat([v1, v2], dim=0))
        return F.cross_entropy(sim, labels)


class TrainLoopConditioned(TrainLoop):
    """Extends TrainLoop to encode source motion into z before each diffusion step.

    Loss:
        L = L_recon(diffusion, high-noise bias)
          + beta * L_KL(free-bits tau=0.5)
          + lambda_inv * L_inv(InfoNCE with memory bank on posterior means)
          + lambda_rank * L_rank(matched z must outperform shuffled z at denoising)

    Key v5 fixes over v4:
      - Contrastive denoising loss: forces decoder to use z content, not just magnitude
      - Z norm equalization (model-side): eliminates ||z|| >> ||null_z|| shortcut
      - Memory-bank InfoNCE (512 negatives instead of 6)
      - High-noise timestep bias (Beta(2,1) → biases toward high t)
      - Topology dropout (model-side, makes z necessary)
    """
    KL_TAU         = 0.5   # free-bits threshold (nats)
    BETA_MAX       = 0.05  # final KL weight after warmup (was 2e-4 — way too small)
    BETA_WARMUP    = 2000  # steps to ramp from 0 → BETA_MAX
    LAMBDA_INV     = 1.0   # view-consistency loss weight (override via args.lambda_inv)
    LAMBDA_RANK    = 1.0   # contrastive denoising loss weight
    RANK_MARGIN    = 0.05  # margin: L_recon(matched) should be < L_recon(shuffled) - margin
    BANK_SIZE      = 512   # memory bank size for InfoNCE negatives

    MASK_RATIO     = 0.0    # fraction of joints to mask in x_0 before noising (0=disabled)

    def __init__(self, args, *loop_args, **loop_kwargs):
        super().__init__(args, *loop_args, **loop_kwargs)
        self.freeze_encoder = getattr(args, 'freeze_encoder', False)
        if self.freeze_encoder:
            self.model.encoder.requires_grad_(False)
            print(f"[Stage 2] Encoder frozen ({sum(p.numel() for p in self.model.encoder.parameters())/1e6:.1f}M params)")
        if hasattr(args, 'mask_ratio') and args.mask_ratio is not None:
            self.MASK_RATIO = args.mask_ratio
        if hasattr(args, 'lambda_inv') and args.lambda_inv is not None:
            self.LAMBDA_INV = args.lambda_inv
        if hasattr(args, 'beta_max') and args.beta_max is not None:
            self.BETA_MAX = args.beta_max
        if hasattr(args, 'beta_warmup') and args.beta_warmup is not None:
            self.BETA_WARMUP = args.beta_warmup
        if hasattr(args, 'lambda_rank') and args.lambda_rank is not None:
            self.LAMBDA_RANK = args.lambda_rank
        if hasattr(args, 'rank_margin') and args.rank_margin is not None:
            self.RANK_MARGIN = args.rank_margin
        # Memory bank for contrastive loss — populated lazily on first forward
        self._mu_bank = None

    def forward_backward(self, batch, cond, epoch=-1):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            micro      = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # ------- View 1: encode source motion -------
            enc_ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
            with enc_ctx:
                enc_out1 = self.ddp_model.encoder(
                    micro_cond['y']['source_motion'],
                    micro_cond['y']['source_offsets'],
                    micro_cond['y']['source_joints_mask'],
                )

            if self.model.encoder.use_vae:
                z_embed, mu1, logvar1 = enc_out1
            else:
                z_embed = enc_out1
                mu1 = logvar1 = None

            micro_cond['y']['z'] = z_embed

            # ------- Sample timesteps (high-noise bias) -------
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            num_diffusion_steps = self.diffusion.num_timesteps
            beta_t = torch.distributions.Beta(2.0, 1.0).sample((micro.shape[0],))
            t = (beta_t * num_diffusion_steps).long().clamp(0, num_diffusion_steps - 1).to(dist_util.dev())
            noise = torch.randn_like(micro)

            # ------- Joint masking on x_0 (before noising) -------
            # Mask random joints in x_0 so x_t has no data signal for them.
            # Model must use FSQ tokens to predict masked joints.
            # Implementation: adjust noise so that for masked joints,
            # x_t = sqrt(1-α_t) · ε (pure noise), while x_start stays
            # as the original x_0 (correct target for loss).
            if self.MASK_RATIO > 0:
                B, J = micro.shape[0], micro.shape[1]
                n_joints_per = micro_cond['y']['n_joints']  # [B]
                # Sample mask rate ~ U[0, MASK_RATIO] per sample for diversity
                mask_rates = torch.rand(B, device=micro.device) * self.MASK_RATIO
                joint_mask = torch.zeros(B, J, dtype=torch.bool, device=micro.device)
                for b in range(B):
                    nj = int(n_joints_per[b].item())
                    n_mask = max(1, int(mask_rates[b].item() * nj))
                    perm = torch.randperm(nj, device=micro.device)[:n_mask]
                    joint_mask[b, perm] = True
                # Compute noise adjustment: for masked joints, cancel x_0 signal in x_t
                # x_t = sqrt(α) * x_0 + sqrt(1-α) * noise_adj = sqrt(1-α) * ε
                # → noise_adj = ε - sqrt(α)/sqrt(1-α) * x_0
                alphas_cumprod = torch.tensor(
                    [self.diffusion.alphas_cumprod[ti.item()] for ti in t],
                    device=micro.device, dtype=micro.dtype)
                sqrt_a = alphas_cumprod.sqrt().view(B, 1, 1, 1)
                sqrt_1ma = (1 - alphas_cumprod).sqrt().clamp(min=1e-8).view(B, 1, 1, 1)
                adjustment = (sqrt_a / sqrt_1ma) * micro
                mask_4d = joint_mask[:, :, None, None].expand_as(noise)
                noise = torch.where(mask_4d, noise - adjustment, noise)

            # ------- Shuffled z reference loss (no grad, for ranking) -------
            # Compute BEFORE matched forward to avoid in-place graph conflicts.
            # Uses eval mode + no_grad to isolate completely from the training graph.
            loss_rank = torch.zeros(1, device=micro.device)
            loss_recon_shuffled = None
            use_ranking = (self.LAMBDA_RANK > 0 and z_embed is not None
                           and micro.shape[0] >= 2)
            if use_ranking:
                B = micro.shape[0]
                perm = torch.roll(torch.arange(B, device=micro.device), 1)
                y_shuf = {}
                for k, v in micro_cond['y'].items():
                    y_shuf[k] = v.clone() if torch.is_tensor(v) else v
                y_shuf['z'] = z_embed[perm].detach().clone()

                self.model.eval()
                with torch.no_grad():
                    losses_shuf = self.diffusion.training_losses(
                        self.ddp_model, micro.clone(), t.clone(),
                        model_kwargs={'y': y_shuf}, noise=noise.clone())
                    loss_recon_shuffled = (losses_shuf["loss"] * weights).mean().item()
                self.model.train()

            # ------- Matched z denoising loss -------
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                noise=noise,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss_recon_matched = (losses["loss"] * weights).mean()
            loss = loss_recon_matched

            # Hinge ranking: matched z should beat shuffled z by margin
            if loss_recon_shuffled is not None:
                loss_rank = torch.clamp(
                    loss_recon_matched - torch.tensor(loss_recon_shuffled, device=micro.device)
                    + self.RANK_MARGIN, min=0.0)
                loss = loss + self.LAMBDA_RANK * loss_rank

            # ------- KL loss (VAE mode only) -------
            if mu1 is not None and logvar1 is not None:
                step = self.total_step()
                beta = self.BETA_MAX * min(1.0, step / max(1, self.BETA_WARMUP))
                loss_kl = _free_bits_kl(mu1, logvar1, tau=self.KL_TAU)
                loss = loss + beta * loss_kl
            else:
                loss_kl = torch.zeros(1, device=micro.device)
                beta = 0.0

            # ------- View-consistency loss (VAE mode only) -------
            loss_inv = torch.zeros(1, device=micro.device)
            if mu1 is not None and self.LAMBDA_INV > 0:
                src_motion  = micro_cond['y']['source_motion']
                src_offsets = micro_cond['y']['source_offsets']
                src_mask    = micro_cond['y']['source_joints_mask']

                src_motion_v2 = _augment_view(src_motion, src_mask)

                enc_out2 = self.ddp_model.encoder(
                    src_motion_v2, src_offsets, src_mask)
                _, mu2, _ = enc_out2

                # Lazily init memory bank with correct dimension
                if self._mu_bank is None:
                    flat_dim = mu1.reshape(mu1.shape[0], -1).shape[1]
                    self._mu_bank = MomentumBank(flat_dim, self.BANK_SIZE)
                loss_inv = _view_consistency_loss(mu1, mu2, bank=self._mu_bank)
                loss = loss + self.LAMBDA_INV * loss_inv

            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

            # Log our custom losses to the diffusion logger (appears in progress.csv)
            from diffusion import logger as diff_logger
            diff_logger.logkv_mean('loss_kl', loss_kl.item() if torch.is_tensor(loss_kl) else loss_kl)
            diff_logger.logkv_mean('loss_inv', loss_inv.item() if torch.is_tensor(loss_inv) else loss_inv)
            diff_logger.logkv_mean('loss_rank', loss_rank.item() if torch.is_tensor(loss_rank) else loss_rank)
            diff_logger.logkv_mean('beta', beta)
            diff_logger.logkv_mean('loss_total', loss.item())

            self.mp_trainer.backward(loss)

        # Encoder diagnostics at log_interval
        if self.total_step() % self.log_interval == 0:
            with torch.no_grad():
                if self.model.encoder.use_vae:
                    z_diag, mu_d, _lv, inter = self.ddp_model.encoder(
                        micro_cond['y']['source_motion'],
                        micro_cond['y']['source_offsets'],
                        micro_cond['y']['source_joints_mask'],
                        return_intermediates=True,
                    )
                else:
                    z_diag, inter = self.ddp_model.encoder(
                        micro_cond['y']['source_motion'],
                        micro_cond['y']['source_offsets'],
                        micro_cond['y']['source_joints_mask'],
                        return_intermediates=True,
                    )

            self.train_platform.report_scalar(
                'slot_diversity', slot_diversity(z_diag), self.total_step(), 'Encoder')
            self.train_platform.report_scalar(
                'null_z_divergence', null_z_divergence(self.model.null_z, z_diag),
                self.total_step(), 'Encoder')
            # Log all losses to wandb
            self.train_platform.report_scalar('loss_recon', loss_recon_matched.item(),
                                              self.total_step(), 'Loss')
            self.train_platform.report_scalar('loss_rank', loss_rank.item() if torch.is_tensor(loss_rank) else 0,
                                              self.total_step(), 'Loss')
            self.train_platform.report_scalar('loss_total', loss.item(), self.total_step(), 'Loss')
            if mu1 is not None:
                self.train_platform.report_scalar('loss_kl', loss_kl.item(), self.total_step(), 'Loss')
                self.train_platform.report_scalar('beta', beta, self.total_step(), 'Loss')
                self.train_platform.report_scalar('beta_x_kl', beta * loss_kl.item(), self.total_step(), 'Loss')
                self.train_platform.report_scalar('loss_inv', loss_inv.item() if torch.is_tensor(loss_inv) else loss_inv,
                                                  self.total_step(), 'Loss')
                self.train_platform.report_scalar(
                    'mu_var', mu_d.var().item(), self.total_step(), 'Encoder')
                # Collapse detector: mean pairwise cosine of mu in batch (>0.95 = collapsing)
                mu_flat = mu_d.view(mu_d.shape[0], -1)
                mu_norm = F.normalize(mu_flat, dim=-1)
                mu_cos = (mu_norm @ mu_norm.T).fill_diagonal_(0).sum() / (mu_d.shape[0] * (mu_d.shape[0] - 1))
                self.train_platform.report_scalar(
                    'mu_pairwise_cos', mu_cos.item(), self.total_step(), 'Encoder')
                # z_embed vs null_z cosine (critical: should be < 0.5 for z to matter)
                z_pooled = z_diag.mean(dim=1).view(z_diag.shape[0], -1)  # [B, K*D]
                null_flat = self.model.null_z.squeeze().view(-1).detach()  # [K*D]
                z_norm_d = F.normalize(z_pooled, dim=-1)
                null_norm_d = F.normalize(null_flat.unsqueeze(0), dim=-1)
                z_null_cos = (z_norm_d * null_norm_d).sum(dim=-1).mean()
                self.train_platform.report_scalar(
                    'z_null_cos', z_null_cos.item(), self.total_step(), 'Encoder')
                # z_embed pairwise cosine (should be < 0.9 for diversity)
                z_embed_norm = F.normalize(z_pooled, dim=-1)
                z_pair_cos = (z_embed_norm @ z_embed_norm.T).fill_diagonal_(0).sum() / (z_diag.shape[0] * (z_diag.shape[0] - 1))
                self.train_platform.report_scalar(
                    'z_embed_pairwise_cos', z_pair_cos.item(), self.total_step(), 'Encoder')

    def evaluate(self):
        step = self.total_step() + 1
        self.model.eval()

        # PCA of z embeddings over ~30 batches
        z_flat, labels = collect_z_embeddings(self.model, self.data, self.device, n_batches=30)
        if z_flat is not None:
            fig = z_pca_figure(z_flat, labels)
            if fig is not None:
                self.train_platform.report_figure('z_pca', fig, step)
                plt.close(fig)

        # Slot attention heatmap for one sample
        with torch.no_grad():
            for _, cond in self.data:
                cond_y = {k: v.to(self.device) if torch.is_tensor(v) else v
                          for k, v in cond['y'].items()}
                enc_out = self.model.encoder(
                    cond_y['source_motion'][:1],
                    cond_y['source_offsets'][:1],
                    cond_y['source_joints_mask'][:1],
                    return_intermediates=True,
                )
                if self.model.encoder.use_vae:
                    _, _mu, _lv, inter = enc_out
                else:
                    _, inter = enc_out
                attn = inter['attn_weights'][0]   # [T, K, J] for first sample
                n_joints = int(cond_y['source_joints_mask'][0].sum().item())
                joint_labels = [str(j) for j in range(n_joints)]
                fig = slot_attn_figure(attn, joint_labels)
                self.train_platform.report_figure('slot_attn', fig, step)
                plt.close(fig)
                break

        self.model.train()
