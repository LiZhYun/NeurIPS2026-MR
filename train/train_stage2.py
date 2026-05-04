"""Stage 2: Train AnyTop diffusion conditioned on frozen Stage 1 encoder.

Loads the Stage 1 MotionAutoEncoder encoder (frozen), encodes each batch,
projects z to decoder latent_dim, and trains the AnyTop diffusion decoder.

Usage:
    conda run -n anytop python -m train.train_stage2 \
        --stage1_path save/S1v2_norm_infonce/final.pt \
        --batch_size 2 --num_steps 600000 \
        --save_dir save/S2_infonce_600k
"""
import os
import json
import torch
import torch.nn as nn
import functools
from argparse import ArgumentParser

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion_general_skeleton
from model.motion_autoencoder import MotionAutoEncoder
from model.anytop_conditioned import ConditionedGraphMotionDecoderLayer, ConditionedGraphMotionDecoder
from model.anytop import AnyTop, create_sin_embedding
from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from diffusion.resample import UniformSampler
from train.training_loop import TrainLoop, log_loss_dict
from utils.ml_platforms import WandBPlatform, NoPlatform


class AnyTopWithExternalZ(AnyTop):
    """AnyTop decoder that receives pre-computed z via cross-attention.

    Same as AnyTopConditioned but WITHOUT a built-in encoder.
    z_embed is passed via y['z'].
    """
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace decoder with conditioned version (adds cross-attention)
        cond_layer = ConditionedGraphMotionDecoderLayer(
            d_model=self.latent_dim, nhead=4,
            dim_feedforward=1024, dropout=0.1, activation='gelu')
        self.seqTransDecoder = ConditionedGraphMotionDecoder(
            cond_layer, num_layers=kwargs.get('num_layers', 4),
            value_emb=self.value_emb)

        # Project Stage 1 z (d_z=32) → decoder latent_dim (256)
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
        )
        # Null z for CFG
        self.null_z = nn.Parameter(torch.randn(1, 1, 4, self.latent_dim) * 0.02)
        self.z_drop_prob = 0.1

    def forward(self, x, timesteps, get_layer_activation=-1, y=None):
        joints_mask = y['joints_mask'].to(x.device)
        temp_mask = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)
        bs, njoints, nfeats, nframes = x.shape

        # Get external z and project
        z_raw = y.get('z', None)
        if z_raw is not None:
            z_embed = self.z_proj(z_raw.to(x.device))  # [B, T/4, K, latent_dim]
        else:
            z_embed = self.null_z.expand(bs, 1, 4, self.latent_dim)

        # CFG dropout
        if self.training and self.z_drop_prob > 0:
            drop = torch.rand(bs, device=x.device) < self.z_drop_prob
            null = self.null_z.expand_as(z_embed)
            z_embed = torch.where(drop[:, None, None, None], null, z_embed)

        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind'])

        spatial_mask = 1.0 - joints_mask[:, 0, 0, 1:, 1:]
        spatial_mask = (spatial_mask.unsqueeze(1).unsqueeze(1)
                        .repeat(1, nframes + 1, self.num_heads, 1, 1)
                        .reshape(-1, self.num_heads, njoints, njoints))
        temporal_mask = (1.0 - temp_mask.repeat(1, njoints, self.num_heads, 1, 1)
                         .reshape(-1, nframes + 1, nframes + 1).float())
        spatial_mask[spatial_mask == 1.0] = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9

        output = self.seqTransDecoder(
            tgt=x, timesteps_embs=timesteps_emb, memory=None,
            spatial_mask=spatial_mask, temporal_mask=temporal_mask,
            y=y, get_layer_activation=get_layer_activation,
            z_embed=z_embed)

        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output = output[0]
        output = self.output_process(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output


def main():
    parser = ArgumentParser()
    parser.add_argument('--stage1_path', required=True, help='Path to Stage 1 final.pt')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=600000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--d_z', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    args = parser.parse_args()

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    opt = get_opt(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load frozen Stage 1 encoder
    print(f"Loading Stage 1 encoder from {args.stage1_path}")
    s1_ckpt = torch.load(args.stage1_path, map_location='cpu')
    stage1 = MotionAutoEncoder(
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        d_model=128, d_z=args.d_z, num_queries=4, canonical=True)
    stage1.load_state_dict(s1_ckpt['model'])
    stage1.to(dist_util.dev())
    stage1.eval()
    stage1.requires_grad_(False)
    print(f"  Stage 1 encoder loaded and frozen")

    # Create Stage 2 decoder
    print("Creating Stage 2 decoder...")
    from utils.model_util import get_gmdm_args, create_gaussian_diffusion

    class FakeArgs:
        t5_name = 't5-base'
        latent_dim = args.latent_dim
        layers = args.layers
        cond_mask_prob = 0.1
        skip_t5 = False
        value_emb = False
        noise_schedule = 'cosine'
        sigma_small = True
        diffusion_steps = 100
        lambda_fs = 0.0
        lambda_geo = 0.0

    base_args = get_gmdm_args(FakeArgs())
    model = AnyTopWithExternalZ(z_dim=args.d_z, **base_args)
    diffusion = create_gaussian_diffusion(FakeArgs())
    model.to(dist_util.dev())

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Trainable params: {n_params:.2f}M (decoder only)")

    # Data
    print("Creating data loader...")
    data = get_dataset_loader_conditioned(
        batch_size=args.batch_size, num_frames=120,
        temporal_window=31, t5_name='t5-base',
        balanced=False, objects_subset='all')

    # Training
    ml_platform = NoPlatform(save_dir=args.save_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    schedule_sampler = UniformSampler(diffusion)

    print(f"Training Stage 2 for {args.num_steps} steps...")
    data_iter = iter(data)
    running_loss = 0

    for step in range(args.num_steps):
        try:
            batch, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(data)
            batch, cond = next(data_iter)

        batch = batch.to(dist_util.dev())
        # Move all cond tensors to GPU
        for k, v in cond['y'].items():
            if torch.is_tensor(v):
                cond['y'][k] = v.to(dist_util.dev())

        y = cond['y']

        # Encode with frozen Stage 1
        with torch.no_grad():
            z = stage1.encode(
                y['source_motion'],
                y['source_offsets'],
                y['source_joints_mask'])
        y['z'] = z  # [B, T/4, K, d_z]

        # Standard diffusion training
        t, weights = schedule_sampler.sample(batch.shape[0], dist_util.dev())
        losses = diffusion.training_losses(model, batch, t, model_kwargs=cond)
        loss = (losses['loss'] * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % args.log_interval == 0:
            avg = running_loss / args.log_interval
            print(f"[{step+1}/{args.num_steps}] loss={avg:.4f}")
            running_loss = 0

        if (step + 1) % args.save_interval == 0:
            ckpt = {
                'model': model.state_dict(),
                'stage1_path': args.stage1_path,
                'step': step + 1,
            }
            path = os.path.join(args.save_dir, f'model{step+1:09d}.pt')
            torch.save(ckpt, path)
            print(f"  Saved {path}")

    print("Stage 2 training complete.")


if __name__ == '__main__':
    main()
