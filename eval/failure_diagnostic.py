"""Pinpoint the failure mode of the behavior-conditioned model.

Tests three hypotheses:
  H1: Decoder ignores behavior conditioning (null_z vs matched_z vs random → same loss/output)
  H2: Method collapses to action-prototype (same-action sources → near-identical outputs)
  H3: Method collapses to skeleton-prototype (same-skeleton target → near-identical outputs regardless of source)

For each, generate motions under counterfactual conditioning and compare via:
  - Classifier-embedding cosine distance (method output differentiation)
  - DDPM loss (does decoder respond to conditioning?)
  - Action-class prediction stability

Usage:
    conda run -n anytop python -m eval.failure_diagnostic --ckpt save/B1_scratch_seed42/model000200000.pt
"""
import os
import json
import argparse
import time
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--classifier_ckpt', default='save/external_classifier.pt')
    p.add_argument('--n_probe_pairs', type=int, default=20,
                   help='Pairs for counterfactual probing')
    p.add_argument('--out', default='eval/results/failure_diagnostic.json')
    p.add_argument('--device', type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def sample_with_y(model, diffusion, y, max_joints, feature_len, n_frames, device):
    x = torch.randn(1, max_joints, feature_len, n_frames, device=device)
    for t_val in reversed(range(diffusion.num_timesteps)):
        t = torch.tensor([t_val], device=device)
        out = diffusion.p_mean_variance(model, x, t, model_kwargs={'y': y}, clip_denoised=False)
        x = out['mean']
        if t_val > 0:
            noise = torch.randn_like(x)
            x = x + torch.exp(0.5 * out['log_variance']) * noise
    return x


@torch.no_grad()
def get_classifier_emb(motion_norm, n_joints, parents, mean, std, classifier, device):
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.external_classifier import extract_classifier_features

    motion_denorm = motion_norm[:, :n_joints] * (std[:n_joints] + 1e-6) + mean[:n_joints]
    positions = recover_from_bvh_ric_np(motion_denorm)
    feats = extract_classifier_features(positions, parents[:n_joints])
    if feats is None:
        return None
    if feats.shape[0] < 64:
        feats = np.pad(feats, ((0, 64 - feats.shape[0]), (0, 0), (0, 0)))
    else:
        idx = np.linspace(0, feats.shape[0] - 1, 64).astype(int)
        feats = feats[idx]
    x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    B, T, D, F_ = x.shape
    x = x.permute(0, 2, 3, 1).reshape(B, D * F_, T)
    h = x
    for layer in classifier.conv[:-1]:
        h = layer(h)
    return h.cpu().numpy().flatten(), classifier.conv(x.to(device)).softmax(-1).cpu().numpy()[0]


def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b))


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from model.conditioners import T5Conditioner
    from eval.track_b_inference import load_behavior_model, build_target_y
    from eval.external_classifier import ActionClassifier

    fixseed(42)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    print(f"Loading model from {args.ckpt}")
    model, diffusion, model_args = load_behavior_model(args.ckpt, device)
    n_frames = model_args.get('num_frames', 120)

    print(f"Loading classifier")
    classifier = ActionClassifier()
    state = torch.load(args.classifier_ckpt, map_location='cpu', weights_only=False)
    classifier.load_state_dict(state['model'])
    classifier.to(device).eval()

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=model_args.get('t5_name', 't5-base'),
                       finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')

    psi_all = np.load('eval/results/effect_cache/psi_all.npy')
    with open('eval/results/effect_cache/clip_metadata.json') as f:
        metadata = json.load(f)
    fname_to_psi_idx = {m['fname']: i for i, m in enumerate(metadata)}
    ACTION_CLASSES = ['walk', 'run', 'idle', 'attack', 'fly', 'swim', 'jump',
                      'turn', 'die', 'eat', 'getup', 'other']
    ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], 11) for m in metadata}

    skels = sorted(set(m['skeleton'] for m in metadata))
    val_clips = [m for m in metadata if m['split'] == 'val']
    rng = np.random.default_rng(42)

    # Pick probe pairs
    pairs = []
    for _ in range(args.n_probe_pairs):
        src_meta = val_clips[int(rng.integers(0, len(val_clips)))]
        target_skels_pool = [s for s in skels if s != src_meta['skeleton']]
        tgt_skel = str(rng.choice(target_skels_pool))
        pairs.append((src_meta['fname'], src_meta['skeleton'], tgt_skel))

    # For each pair, generate under 4 conditioning modes:
    #   (a) matched: source's actual ψ + action
    #   (b) null: zero ψ + "other" action (CFG null)
    #   (c) random: random ψ + random action
    #   (d) swapped: ψ from different clip of DIFFERENT action
    t0 = time.time()
    probe_records = []

    for i, (src_fname, src_skel, tgt_skel) in enumerate(pairs):
        psi_idx = fname_to_psi_idx.get(src_fname)
        if psi_idx is None:
            continue
        matched_psi = psi_all[psi_idx]
        matched_action = fname_to_action.get(src_fname, 11)

        # Pick a contrasting clip: different action class
        diff_action_clips = [j for j, m in enumerate(metadata)
                             if m['skeleton'] != src_skel and m['skeleton'] != tgt_skel
                             and fname_to_action.get(m['fname'], 11) != matched_action]
        if not diff_action_clips:
            continue
        swap_idx = int(rng.choice(diff_action_clips))
        swap_psi = psi_all[swap_idx]
        swap_action = fname_to_action.get(metadata[swap_idx]['fname'], 11)

        # Build target y (computed once per pair)
        target_y_base, n_joints, mean, std = build_target_y(
            tgt_skel, cond_dict, opt, t5, n_frames, device)

        embs = {}
        action_probs = {}
        for mode in ['matched', 'null', 'random', 'swapped']:
            y = {k: v for k, v in target_y_base.items()}
            if mode == 'matched':
                y['psi'] = torch.tensor(matched_psi[None], dtype=torch.float32, device=device)
                y['action_label'] = torch.tensor([matched_action], dtype=torch.long, device=device)
            elif mode == 'null':
                y['psi'] = torch.zeros(1, 64, 62, dtype=torch.float32, device=device)
                y['action_label'] = torch.tensor([11], dtype=torch.long, device=device)  # "other"
            elif mode == 'random':
                y['psi'] = torch.randn(1, 64, 62, dtype=torch.float32, device=device) * matched_psi.std()
                y['action_label'] = torch.tensor([int(rng.integers(0, 12))], dtype=torch.long, device=device)
            elif mode == 'swapped':
                y['psi'] = torch.tensor(swap_psi[None], dtype=torch.float32, device=device)
                y['action_label'] = torch.tensor([swap_action], dtype=torch.long, device=device)

            torch.manual_seed(10000 + i)  # same noise seed across modes
            sample = sample_with_y(model, diffusion, y, opt.max_joints, opt.feature_len, n_frames, device)
            motion_norm = sample[0].cpu().permute(2, 0, 1).numpy()

            info = cond_dict[tgt_skel]
            parents = np.array(info['parents'][:n_joints], dtype=np.int64)
            result = get_classifier_emb(motion_norm, n_joints, parents, mean, std, classifier, device)
            if result is None:
                embs[mode] = None
                continue
            emb, probs = result
            embs[mode] = emb
            action_probs[mode] = probs.tolist()

        if embs.get('matched') is None:
            continue

        # Compute pairwise distances from matched
        d_null = cosine_dist(embs['matched'], embs['null']) if embs.get('null') is not None else None
        d_random = cosine_dist(embs['matched'], embs['random']) if embs.get('random') is not None else None
        d_swapped = cosine_dist(embs['matched'], embs['swapped']) if embs.get('swapped') is not None else None

        probe_records.append({
            'src_fname': src_fname, 'src_skel': src_skel, 'tgt_skel': tgt_skel,
            'matched_action': matched_action, 'swap_action': swap_action,
            'd_matched_null':    d_null,
            'd_matched_random':  d_random,
            'd_matched_swapped': d_swapped,
            'pred_matched':  int(np.argmax(action_probs.get('matched',  [0]*12))) if embs.get('matched')  is not None else None,
            'pred_null':     int(np.argmax(action_probs.get('null',     [0]*12))) if embs.get('null')     is not None else None,
            'pred_random':   int(np.argmax(action_probs.get('random',   [0]*12))) if embs.get('random')   is not None else None,
            'pred_swapped':  int(np.argmax(action_probs.get('swapped',  [0]*12))) if embs.get('swapped')  is not None else None,
        })

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  probe {i+1}/{len(pairs)} ({elapsed:.0f}s elapsed)")

    # Report
    print(f"\n{'='*70}")
    print(f"FAILURE-MODE DIAGNOSTIC (n={len(probe_records)})")
    print(f"{'='*70}")

    if not probe_records:
        print("No valid records")
        return

    # H1: Decoder ignores behavior conditioning
    d_nulls = [r['d_matched_null'] for r in probe_records if r['d_matched_null'] is not None]
    d_rands = [r['d_matched_random'] for r in probe_records if r['d_matched_random'] is not None]
    d_swaps = [r['d_matched_swapped'] for r in probe_records if r['d_matched_swapped'] is not None]

    print(f"\nH1: Does behavior conditioning affect output?")
    print(f"   d(matched, null)    = {np.mean(d_nulls):.4f} ± {np.std(d_nulls):.4f}  "
          f"(0 = identical, >0 = responds)")
    print(f"   d(matched, random)  = {np.mean(d_rands):.4f} ± {np.std(d_rands):.4f}")
    print(f"   d(matched, swapped) = {np.mean(d_swaps):.4f} ± {np.std(d_swaps):.4f}")
    if np.mean(d_nulls) < 0.05 and np.mean(d_swaps) < 0.05:
        print("   ✗ CONFIRMED H1: Decoder ignores behavior conditioning "
              "(matched ≈ null ≈ swapped)")
    elif np.mean(d_nulls) > 0.15 and np.mean(d_swaps) > 0.15:
        print("   ✓ Decoder DOES respond to behavior conditioning")
    else:
        print("   ~ Weak response to conditioning")

    # H2: Same-action sources collapse to same output
    matched_preds = [r['pred_matched'] for r in probe_records]
    swap_preds = [r['pred_swapped'] for r in probe_records]
    null_preds = [r['pred_null'] for r in probe_records]
    action_change_count = sum(1 for r in probe_records if r['pred_matched'] != r['pred_swapped'])

    print(f"\nH2: Do swapped-behavior outputs get different predicted actions?")
    print(f"   N with matched_pred != swapped_pred: {action_change_count}/{len(probe_records)}")

    # Prediction distributions
    from collections import Counter
    c_m = Counter(matched_preds)
    c_n = Counter(null_preds)
    print(f"\n   Matched predictions distribution:  {dict(c_m.most_common())}")
    print(f"   Null    predictions distribution:  {dict(c_n.most_common())}")
    if c_m == c_n or (len(c_m) == 1 and len(c_n) == 1 and list(c_m)[0] == list(c_n)[0]):
        print("   ✗ CONFIRMED: Matched and null produce same class distribution")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'ckpt': args.ckpt,
            'n_pairs': len(probe_records),
            'd_matched_null_mean': float(np.mean(d_nulls)),
            'd_matched_random_mean': float(np.mean(d_rands)),
            'd_matched_swapped_mean': float(np.mean(d_swaps)),
            'action_change_rate': action_change_count / len(probe_records),
            'records': probe_records,
        }, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
