"""Extract AnyTop encoder z (the source-encoded latent that conditions diffusion).

For P1 isotropy and (if multi-seed available later) P2 Procrustes alignment.
Only the encoder is run; diffusion sampling is skipped — the latent of interest
is z = model.encoder(source_motion, source_offsets, source_mask).

Usage:
    python /tmp/save_anytop_latents.py \
        --model_path save/A1v5_znorm_rank_bs_4_latentdim_256/model000175000.pt \
        --manifest eval/benchmark_v3/queries_sif_intersection/manifest.json \
        --out_dir /tmp/anytop_latents/A1v5_step175k

NOTE: AnyTop has only one published seed for the 14-method evaluation. P2 cannot
be run unless multiple seeds are trained (each ~17 GPU-hr). P1 (isotropy) and P3
(rotation perturbation, see §5.23) are testable from the single seed.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from os.path import dirname, join
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_args_from_checkpoint(model_path):
    args_path = join(dirname(model_path), 'args.json')
    with open(args_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--motion_length', type=float, default=5.0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from utils import dist_util
    from utils.fixseed import fixseed
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from utils.model_util import create_conditioned_model_and_diffusion, load_model
    from sample.generate_conditioned import build_source_tensors

    fixseed(42)
    dist_util.setup_dist(args.device)
    ckpt_args_dict = load_args_from_checkpoint(args.model_path)

    class Namespace:
        def __init__(self, d): self.__dict__.update(d)
    ckpt = Namespace(ckpt_args_dict)
    opt = get_opt(args.device)
    n_frames = int(args.motion_length * opt.fps)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    print(f"[anytop-latents] Creating model from {args.model_path}")
    model, _ = create_conditioned_model_and_diffusion(ckpt)
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    print(f"[anytop-latents] Loading manifest: {args.manifest}")
    manifest = json.loads(Path(args.manifest).read_text())
    queries = manifest.get('queries', manifest.get('triples', []))
    if args.limit:
        queries = queries[:args.limit]
    print(f"[anytop-latents] Processing {len(queries)} queries...")

    triple_meta = []
    t0 = time.time()
    with torch.no_grad():
        for i, q in enumerate(queries):
            try:
                src_skel = q['skel_a']
                tgt_skel = q['skel_b']
                src_motion, src_offsets, src_mask = build_source_tensors(
                    src_skel, cond_dict, opt, n_frames, ckpt.temporal_window)
                src_motion = src_motion.to(dist_util.dev())
                src_offsets = src_offsets.to(dist_util.dev())
                src_mask = src_mask.to(dist_util.dev())

                enc_out = model.encoder(src_motion, src_offsets, src_mask)
                if isinstance(enc_out, tuple):
                    z = enc_out[0]
                else:
                    z = enc_out

                # z shape varies by AnyTop version: usually [B=1, J_pad, F_lat, T] or [B=1, J_pad, D].
                # Save as-is plus shape for downstream gauge_analysis.py to interpret.
                z_np = z.squeeze(0).cpu().numpy()
                np.save(out_dir / f'z_query_{i:04d}.npy', z_np)
                triple_meta.append({
                    'query_id': i, 'src_skel': src_skel, 'tgt_skel': tgt_skel,
                    'z_shape': list(z_np.shape),
                })
            except Exception as e:
                print(f"  [{i+1}/{len(queries)}] FAILED ({q.get('skel_a','?')}->{q.get('skel_b','?')}): {type(e).__name__}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(queries)}] done ({time.time() - t0:.0f}s)")

    (out_dir / 'meta.json').write_text(json.dumps(triple_meta, indent=2))
    print(f"[anytop-latents] saved {len(triple_meta)} queries to {out_dir} in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
