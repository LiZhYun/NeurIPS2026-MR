"""
Sample from a trained AnyTopConditioned model.

Given a source motion on source_object_type, encode it to z, then decode on
target_object_type. Source and target can be the same (reconstruction) or
different (cross-skeleton retargeting).

Example:
    conda run -n anytop python -m sample.generate_conditioned \
        --model_path save/mini_overfit_bs_2_latentdim_128/model000001999.pt \
        --source_object_type Horse \
        --target_object_type Horse
"""
import os
import json
import numpy as np
import torch
from argparse import ArgumentParser

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_conditioned_model_and_diffusion, load_model
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_batch_collate, create_padded_relation
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from model.conditioners import T5Conditioner
from os.path import join as pjoin
import BVH
from InverseKinematics import animation_from_positions


def load_args_from_checkpoint(model_path):
    args_path = pjoin(os.path.dirname(model_path), 'args.json')
    with open(args_path) as f:
        return json.load(f)


def build_source_tensors(source_object_type, cond_dict, opt, n_frames, temporal_window):
    """Load the first stored motion for source_object_type, normalize, and pad."""
    motion_dir = opt.motion_dir
    motion_files = sorted(
        f for f in os.listdir(motion_dir) if f.startswith(f'{source_object_type}_'))
    assert motion_files, f"No motion files found for {source_object_type}"

    raw = np.load(pjoin(motion_dir, motion_files[0]))  # [T, J_src, 13]
    T, J_src, _ = raw.shape

    mean = cond_dict[source_object_type]['mean']   # [J_src, 13]
    std  = cond_dict[source_object_type]['std'] + 1e-6
    norm = (raw - mean[None, :]) / std[None, :]    # [T, J_src, 13]
    norm = np.nan_to_num(norm)

    # Temporal crop / pad to n_frames
    if T >= n_frames:
        norm = norm[:n_frames]
    else:
        pad = np.zeros((n_frames - T, J_src, 13))
        norm = np.concatenate([norm, pad], axis=0)

    max_joints = opt.max_joints
    offsets_raw = cond_dict[source_object_type]['offsets']  # [J_src, 3]

    # Pad to max_joints
    source_motion = np.zeros((n_frames, max_joints, 13))
    source_motion[:, :J_src, :] = norm
    source_offsets = np.zeros((max_joints, 3))
    source_offsets[:J_src, :] = offsets_raw

    # [J_max, 13, T] and [J_max, 3]
    source_motion_t = torch.tensor(source_motion).permute(1, 2, 0).float().unsqueeze(0)  # [1, J_max, 13, T]
    source_offsets_t = torch.tensor(source_offsets).float().unsqueeze(0)                 # [1, J_max, 3]
    source_mask = torch.zeros(1, max_joints, dtype=torch.bool)
    source_mask[0, :J_src] = True                                                         # [1, J_max]

    return source_motion_t, source_offsets_t, source_mask


def build_target_condition(target_object_type, cond_dict, n_frames, temporal_window,
                           t5_conditioner, max_joints, feature_len):
    """Build model_kwargs for the target skeleton (same as generate.py's create_condition)."""
    obj = cond_dict[target_object_type]
    parents = obj['parents']
    n_joints = len(parents)
    mean     = obj['mean']
    std      = obj['std']
    tpos     = (obj['tpos_first_frame'] - mean) / (std + 1e-6)
    tpos     = np.nan_to_num(tpos)
    joints_names_embs = encode_joints_names(obj['joints_names'], t5_conditioner)

    batch = [
        np.zeros((n_frames, n_joints, feature_len)),                  # motion (ignored, noised)
        n_frames,                                                      # m_length
        parents,
        tpos,
        obj['offsets'],
        create_temporal_mask_for_window(temporal_window, n_frames),
        obj['joints_graph_dist'],
        obj['joint_relations'],
        target_object_type,
        joints_names_embs,
        0,                                                             # crop_start_ind
        mean,
        std,
        max_joints,
    ]
    return truebones_batch_collate([batch])


def encode_joints_names(joints_names, t5_conditioner):
    tokens = t5_conditioner.tokenize(joints_names)
    return t5_conditioner(tokens).detach().cpu().numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--source_object_type', required=True)
    parser.add_argument('--target_object_type', required=True)
    parser.add_argument('--motion_length', default=5.0, type=float,
                        help='Output motion length in seconds.')
    parser.add_argument('--num_repetitions', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--output_dir', default='', type=str)
    args = parser.parse_args()

    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    # Load training args from checkpoint dir
    ckpt_args = load_args_from_checkpoint(args.model_path)

    class Namespace:
        def __init__(self, d):
            self.__dict__.update(d)
    ckpt = Namespace(ckpt_args)

    opt = get_opt(args.device)
    n_frames = int(args.motion_length * opt.fps)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    out_path = args.output_dir or pjoin(
        os.path.dirname(args.model_path),
        f'samples_{args.source_object_type}_to_{args.target_object_type}_seed{args.seed}')
    os.makedirs(out_path, exist_ok=True)

    print("Creating model and diffusion...")
    model, diffusion = create_conditioned_model_and_diffusion(ckpt)
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    print("Loading T5...")
    t5_conditioner = T5Conditioner(
        name=ckpt.t5_name, finetune=False, word_dropout=0.0,
        normalize_text=False, device='cuda')

    # Build source tensors and encode
    print(f"Encoding source motion: {args.source_object_type}")
    source_motion, source_offsets, source_mask = build_source_tensors(
        args.source_object_type, cond_dict, opt, n_frames, ckpt.temporal_window)
    source_motion  = source_motion.to(dist_util.dev())
    source_offsets = source_offsets.to(dist_util.dev())
    source_mask    = source_mask.to(dist_util.dev())
    with torch.no_grad():
        enc_out = model.encoder(source_motion, source_offsets, source_mask)
        if isinstance(enc_out, tuple):
            z = enc_out[0]  # VAE mode: (z_out, mu, logvar)
        else:
            z = enc_out
    print(f"  z shape: {z.shape}")

    # Build target condition
    print(f"Target skeleton: {args.target_object_type}")
    _, model_kwargs = build_target_condition(
        args.target_object_type, cond_dict, n_frames, ckpt.temporal_window,
        t5_conditioner, opt.max_joints, opt.feature_len)
    model_kwargs['y']['z'] = z

    for rep_i in range(args.num_repetitions):
        print(f"Sampling repetition {rep_i}...")
        sample = diffusion.p_sample_loop(
            model,
            (1, opt.max_joints, opt.feature_len, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        tgt = cond_dict[args.target_object_type]
        n_joints = len(tgt['parents'])
        motion = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()  # [T, J, 13]
        motion = motion * tgt['std'][None, :] + tgt['mean'][None, :]

        global_positions = recover_from_bvh_ric_np(motion)
        pref = f'{args.source_object_type}_to_{args.target_object_type}_rep{rep_i}'
        mp4_path = pjoin(out_path, f'{pref}.mp4')
        npy_path = pjoin(out_path, f'{pref}.npy')

        plot_general_skeleton_3d_motion(
            mp4_path, tgt['parents'], global_positions,
            title=pref, fps=opt.fps)
        np.save(npy_path, motion)
        print(f"  Saved: {npy_path}, {mp4_path}")

        # Optionally save BVH
        try:
            out_anim, _, _ = animation_from_positions(
                positions=global_positions, parents=tgt['parents'],
                offsets=tgt['offsets'], iterations=150)
            if out_anim is not None:
                BVH.save(pjoin(out_path, f'{pref}.bvh'), out_anim, tgt['joints_names'])
        except Exception as e:
            print(f"  BVH export skipped: {e}")


if __name__ == '__main__':
    main()
