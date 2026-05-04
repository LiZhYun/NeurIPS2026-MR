"""Phase 0 — Kill-Switch Teacher Audit for decoder-aligned latent training.

Pre-registered gate: before training any new encoder through the frozen
AnyTopConditioned teacher, verify that the teacher actually *uses* z in a
non-degenerate way. If the DDPM loss is indifferent to matched vs null z,
the whole decoder-aligned training plan cannot work — abandon and switch
to the architectural branch.

PASS CRITERION (pre-registered, see refine-logs/stage1_refine/FINAL_PROPOSAL.md):
    gap_rel   = (mean_null_loss - mean_matched_loss) / mean_matched_loss
    gap_sigma = mean_null_loss - mean_matched_loss, in units of across-skel σ
    PASS iff  gap_rel >= 0.02  AND  gap_sigma >= 2.0

Usage:
    conda run -n anytop python -m eval.stage1_teacher_audit \\
        --ckpt save/A1v3_infonce_bs_4_latentdim_256/model000199999.pt \\
        --args_json save/A1v3_infonce_bs_4_latentdim_256/args.json
"""
import os
import json
import argparse
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='save/A2_direct_cond_bs_4_latentdim_256/model000049999.pt')
    p.add_argument('--args_json', default='save/A2_direct_cond_bs_4_latentdim_256/args.json')
    p.add_argument('--n_clips', type=int, default=200)
    p.add_argument('--timesteps', type=str, default='1,10,25,50,99',
                   help='Comma-separated diffusion timesteps to probe')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', type=str, default='eval/results/stage1_teacher_audit.json')
    return p.parse_args()


def load_teacher(ckpt_path, args_json_path, device):
    with open(args_json_path) as f:
        args_d = json.load(f)

    # Provide safe defaults for keys that may be missing on older checkpoints
    defaults = {
        'topo_drop_prob': 0.15,
        'z_norm_target': None,
        'no_rest_pe': False,
    }
    for k, v in defaults.items():
        args_d.setdefault(k, v)

    class NS:
        def __init__(self, d):
            self.__dict__.update(d)
    args = NS(args_d)

    from utils.model_util import create_conditioned_model_and_diffusion, load_model
    model, diffusion = create_conditioned_model_and_diffusion(args)

    print(f"Loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')

    # Handle A1v3-era z_proj shape: old Linear → new Sequential(LN, Linear, GELU, LN)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    if 'encoder.z_proj.weight' in ckpt_keys and 'encoder.z_proj.1.weight' in model_keys:
        print("  Remapping old z_proj Linear → new z_proj Sequential")
        state['encoder.z_proj.1.weight'] = state.pop('encoder.z_proj.weight')
        state['encoder.z_proj.1.bias'] = state.pop('encoder.z_proj.bias')
        # Init LayerNorms (indices 0 and 3) from model defaults — identity-like
        for k in model_keys:
            if k.startswith('encoder.z_proj.0.') or k.startswith('encoder.z_proj.3.'):
                state[k] = model.state_dict()[k]

    load_model(model, state)

    # Note: AnyTopConditioned.to() returns None — do not chain with .eval()
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, diffusion, args


def build_encode_motion_fn(opt, cond_dict, label_map, t5, n_frames, device):
    """Return a closure that encodes a val filename → (x, y_dict, z_matched, skel).

    x is the normalized motion tensor on device, y is the full conditioning dict,
    z_matched is the model.encoder output (computed outside this function), and
    source_* are pre-built for the encoder call.
    """
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation

    max_joints = opt.max_joints
    feature_len = opt.feature_len

    def prepare(fname):
        if fname not in label_map:
            return None
        skel = label_map[fname]['skeleton']
        if skel not in cond_dict:
            return None

        raw = np.load(pjoin(opt.motion_dir, fname))
        T, J_src, _ = raw.shape
        if T < n_frames:
            pad = np.zeros((n_frames - T, J_src, 13))
            raw = np.concatenate([raw, pad], axis=0)
        else:
            start = (T - n_frames) // 2
            raw = raw[start:start + n_frames]

        info = cond_dict[skel]
        mean = info['mean']
        std = info['std'] + 1e-6
        norm = np.nan_to_num((raw - mean[None, :]) / std[None, :])

        n_joints = J_src
        # Decoder input: [1, max_joints, 13, T]
        x_np = np.zeros((n_frames, max_joints, feature_len))
        x_np[:, :n_joints, :] = norm
        x = torch.tensor(x_np).permute(1, 2, 0).float().unsqueeze(0).to(device)

        # Encoder inputs — MotionEncoder uses source_motion [B, J, 13, T], offsets [B, J, 3], mask [B, J] bool
        src_motion = x  # same 4D tensor, shared with decoder
        offsets_np = np.zeros((max_joints, 3))
        offsets_np[:n_joints] = info['offsets']
        src_offsets = torch.tensor(offsets_np).float().unsqueeze(0).to(device)
        src_mask = torch.zeros(1, max_joints, dtype=torch.bool, device=device)
        src_mask[0, :n_joints] = True

        # Rest-pose / topology / names for the decoder-side y dict
        tpos_raw = info['tpos_first_frame']
        tpos = np.zeros((max_joints, feature_len))
        tpos[:n_joints] = (tpos_raw - mean) / std
        tpos = np.nan_to_num(tpos)
        tpos_t = torch.tensor(tpos).float().unsqueeze(0).to(device)

        names = info['joints_names']
        names_emb = t5(t5.tokenize(names)).detach().cpu().numpy()
        names_padded = np.zeros((max_joints, names_emb.shape[1]))
        names_padded[:n_joints] = names_emb
        names_t = torch.tensor(names_padded).float().unsqueeze(0).to(device)

        gd = create_padded_relation(info['joints_graph_dist'], max_joints, n_joints)
        jr = create_padded_relation(info['joint_relations'], max_joints, n_joints)
        gd_t = torch.tensor(gd).long().unsqueeze(0).to(device)
        jr_t = torch.tensor(jr).long().unsqueeze(0).to(device)

        jmask_5d = torch.zeros(1, 1, 1, max_joints + 1, max_joints + 1, device=device)
        jmask_5d[0, 0, 0, :n_joints + 1, :n_joints + 1] = 1.0

        tmask = create_temporal_mask_for_window(31, n_frames)
        tmask_t = torch.tensor(tmask).unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(device)

        y = {
            'joints_mask':       jmask_5d,
            'mask':              tmask_t,
            'tpos_first_frame':  tpos_t,
            'joints_names_embs': names_t,
            'graph_dist':        gd_t,
            'joints_relations':  jr_t,
            'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
            'n_joints':          torch.tensor([n_joints]),
        }

        return {
            'fname': fname,
            'skel': skel,
            'x': x,
            'y': y,
            'src_motion': src_motion,
            'src_offsets': src_offsets,
            'src_mask': src_mask,
            'n_joints': n_joints,
        }

    return prepare


def encoder_z(model, item):
    """Run the built-in MotionEncoder to get z_matched with shape [B, T/4, K, latent_dim]."""
    out = model.encoder(item['src_motion'], item['src_offsets'], item['src_mask'])
    if isinstance(out, tuple):
        # VAE mode returns (z_out, mu, logvar)
        return out[0]
    return out


def masked_loss(pred, target, joints_mask_5d):
    """Per-sample masked MSE over valid joints, averaged over time × features.

    pred, target: [B, J, 13, T]
    joints_mask_5d: [B, 1, 1, J+1, J+1] float (diagonal(1:,1:) gives per-joint validity)
    Returns per-sample scalar loss [B].
    """
    jv = joints_mask_5d[:, 0, 0, 1:, 1:].diagonal(dim1=-2, dim2=-1)  # [B, J]
    se_per_joint = ((pred - target) ** 2).mean(dim=(2, 3))           # [B, J]
    num = (se_per_joint * jv).sum(dim=1)
    den = jv.sum(dim=1).clamp(min=1.0)
    return num / den


@torch.no_grad()
def run_audit(model, diffusion, items, timesteps, device):
    """For each item × each timestep × each mode, compute masked DDPM loss."""
    rng = np.random.default_rng(42)

    # Stage 1: compute z_matched for every clip
    print("Stage 1: computing z_matched for all items")
    for i, it in enumerate(items):
        it['z_matched'] = encoder_z(model, it)     # [1, T/4, K, D]
        if (i + 1) % 50 == 0:
            print(f"  z_matched {i+1}/{len(items)}")

    n_items = len(items)
    T4, K, D = items[0]['z_matched'].shape[1:]
    print(f"z shape: [1, {T4}, {K}, {D}]  null_z param shape: {tuple(model.null_z.shape)}")

    # Null z expanded to match encoder output shape — same distribution CFG uses at train time
    null_z_full = model.null_z.expand(1, T4, K, D)

    # Stage 2: loss sweep
    records = []  # list of dicts {fname, skel, t, mode, loss}
    print("Stage 2: loss sweep over modes × timesteps")
    for idx, it in enumerate(items):
        x = it['x']
        y_base = it['y']
        skel = it['skel']
        fname = it['fname']

        # Pick a random OTHER clip for shuffled
        j = int(rng.integers(0, n_items - 1))
        if j >= idx:
            j += 1
        z_shuffled = items[j]['z_matched']

        for t_val in timesteps:
            t_vec = torch.tensor([t_val], dtype=torch.long, device=device)
            # Fix noise per (clip, t) so all three modes see the same x_t — isolates the z effect
            g = torch.Generator(device=device).manual_seed(t_val * 10_000 + idx)
            noise = torch.empty_like(x).normal_(generator=g)
            x_t = diffusion.q_sample(x, t_vec, noise=noise)

            z_negated = -it['z_matched']
            z_scaled  = it['z_matched'] * 100.0
            z_random  = torch.randn_like(it['z_matched'])

            for mode, z in (
                ('matched',  it['z_matched']),
                ('shuffled', z_shuffled),
                ('null',     null_z_full),
                ('negated',  z_negated),
                ('scaled100x', z_scaled),
                ('random',   z_random),
            ):
                y = {k: v for k, v in y_base.items()}
                y['z'] = z
                pred = model(x_t, t_vec, y=y)
                loss = masked_loss(pred, x, y['joints_mask']).item()
                records.append({'fname': fname, 'skel': skel, 't': int(t_val),
                                'mode': mode, 'loss': float(loss)})

        if (idx + 1) % 25 == 0:
            print(f"  loss sweep {idx+1}/{n_items}")

    return records


def aggregate(records):
    """Compute the pass/fail statistics."""
    all_modes = sorted(set(r['mode'] for r in records))
    mean_loss = {m: 0.0 for m in all_modes}
    count = {m: 0 for m in all_modes}
    per_skel = {}  # skel -> {mode -> [losses]}

    for r in records:
        mean_loss[r['mode']] += r['loss']
        count[r['mode']] += 1
        per_skel.setdefault(r['skel'], {m: [] for m in all_modes})
        per_skel[r['skel']][r['mode']].append(r['loss'])

    for m in mean_loss:
        mean_loss[m] /= max(count[m], 1)

    # Per-skel means (matched only → used for σ)
    per_skel_matched_mean = {s: float(np.mean(per_skel[s]['matched']))
                             for s in per_skel if per_skel[s]['matched']}
    skel_means = np.array(list(per_skel_matched_mean.values()))
    sigma_across_skel = float(np.std(skel_means)) if len(skel_means) > 1 else 0.0

    gap_null = mean_loss['null'] - mean_loss['matched']
    gap_shuffled = mean_loss['shuffled'] - mean_loss['matched']
    rel_null = gap_null / max(mean_loss['matched'], 1e-8)
    rel_shuffled = gap_shuffled / max(mean_loss['matched'], 1e-8)
    sigma_null = gap_null / max(sigma_across_skel, 1e-8)
    sigma_shuffled = gap_shuffled / max(sigma_across_skel, 1e-8)

    # Per-timestep breakdown
    per_t = {}
    for r in records:
        per_t.setdefault(r['t'], {m: [] for m in all_modes})
        per_t[r['t']][r['mode']].append(r['loss'])
    per_t_summary = {}
    for t, mdict in per_t.items():
        per_t_summary[str(t)] = {m: float(np.mean(v)) for m, v in mdict.items() if v}

    return {
        'mean_loss':              {m: float(v) for m, v in mean_loss.items()},
        'gap_null':               float(gap_null),
        'gap_shuffled':           float(gap_shuffled),
        'relative_gap_null':      float(rel_null),
        'relative_gap_shuffled':  float(rel_shuffled),
        'sigma_across_skel':      float(sigma_across_skel),
        'sigma_multiples_null':   float(sigma_null),
        'sigma_multiples_shuffled': float(sigma_shuffled),
        'per_skel_matched_mean':  per_skel_matched_mean,
        'per_t':                  per_t_summary,
    }


def verdict(stats):
    """Apply the pre-registered pass criterion."""
    rel = stats['relative_gap_null']
    sig = stats['sigma_multiples_null']
    passes = (rel >= 0.02) and (sig >= 2.0)
    return {
        'pass_criterion': {
            'relative_gap_null_>=_0.02': rel >= 0.02,
            'sigma_multiples_null_>=_2.0': sig >= 2.0,
        },
        'PASSES': bool(passes),
        'action': (
            "PROCEED with decoder-aligned Stage 1 training (S1v10)."
            if passes else
            "ABANDON decoder-aligned route; switch to architectural branch (SAME / UniMoGen)."
        ),
    }


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Load teacher
    model, diffusion, m_args = load_teacher(args.ckpt, args.args_json, device)
    print(f"Teacher: layers={getattr(m_args, 'layers', '?')} "
          f"latent_dim={getattr(m_args, 'latent_dim', '?')} "
          f"enc_num_queries={getattr(m_args, 'enc_num_queries', '?')} "
          f"num_frames={getattr(m_args, 'num_frames', '?')}")

    # Val split + labels
    with open('dataset/truebones/zoo/truebones_processed/train_val_split.json') as f:
        split = json.load(f)
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)
    val_files = split['val'][:args.n_clips]
    print(f"Val clips requested: {len(val_files)}")

    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    from model.conditioners import T5Conditioner
    t5 = T5Conditioner(name=m_args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device='cuda')

    n_frames = getattr(m_args, 'num_frames', 120)
    prepare = build_encode_motion_fn(opt, cond_dict, label_map, t5, n_frames, device)

    # Build items list
    items = []
    n_skipped = 0
    for fname in val_files:
        item = prepare(fname)
        if item is None:
            n_skipped += 1
            continue
        items.append(item)
    print(f"Prepared {len(items)} items ({n_skipped} skipped)")

    if len(items) < 10:
        print("Too few items — aborting audit")
        return

    timesteps = [int(x) for x in args.timesteps.split(',')]
    print(f"Timesteps: {timesteps}")

    records = run_audit(model, diffusion, items, timesteps, device)
    stats = aggregate(records)
    v = verdict(stats)

    # Report
    print("\n" + "=" * 60)
    print("PHASE 0 — KILL-SWITCH AUDIT RESULTS")
    print("=" * 60)
    print(f"Teacher: {args.ckpt}")
    print(f"N val items: {len(items)}")
    print(f"Timesteps: {timesteps}")
    print()
    print("Mean DDPM loss by conditioning mode:")
    for m in ('matched', 'shuffled', 'null'):
        print(f"  {m:8s}: {stats['mean_loss'][m]:.6f}")
    print()
    print(f"  gap (null  - matched): {stats['gap_null']:.6f}")
    print(f"  relative gap (null):    {stats['relative_gap_null']*100:.3f}%")
    print(f"  σ across skeletons:     {stats['sigma_across_skel']:.6f}")
    print(f"  gap in σ units (null):  {stats['sigma_multiples_null']:.2f}σ")
    print()
    print(f"  gap (shuffled - matched): {stats['gap_shuffled']:.6f}")
    print(f"  relative gap (shuffled):  {stats['relative_gap_shuffled']*100:.3f}%")
    print(f"  gap in σ units (shuffled):{stats['sigma_multiples_shuffled']:.2f}σ")
    print()
    print("Pass criterion (pre-registered):")
    for k, val in v['pass_criterion'].items():
        print(f"  {k}: {val}")
    print()
    print(f"VERDICT: {'PASS' if v['PASSES'] else 'FAIL'}")
    print(f"ACTION:  {v['action']}")
    print()
    print("Per-timestep mean losses:")
    for t in sorted(stats['per_t'].keys(), key=lambda x: int(x)):
        row = stats['per_t'][t]
        print(f"  t={t:>3}: matched={row['matched']:.5f}  "
              f"shuffled={row['shuffled']:.5f}  null={row['null']:.5f}")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {
        'ckpt':        args.ckpt,
        'n_items':     len(items),
        'timesteps':   timesteps,
        'stats':       stats,
        'verdict':     v,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
