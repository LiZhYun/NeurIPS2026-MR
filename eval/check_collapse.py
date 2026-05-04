"""Quick collapse check for a trained checkpoint."""
import torch, numpy as np, json, sys
from argparse import Namespace
from utils.model_util import create_conditioned_model_and_diffusion
from data_loaders.truebones.truebones_utils.get_opt import get_opt

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else 'save/A1v3_infonce_bs_4_latentdim_256'
ckpt_file = sorted([f for f in __import__('os').listdir(ckpt_dir) if f.startswith('model')])[-1]

with open(f'{ckpt_dir}/args.json') as f:
    args = Namespace(**json.load(f))
model, _ = create_conditioned_model_and_diffusion(args)
state = torch.load(f'{ckpt_dir}/{ckpt_file}', map_location='cpu')
model.load_state_dict(state, strict=False)
model.eval()

d = np.load('eval/data/truebones_train.npy', allow_pickle=True).item()
motions = torch.tensor(d['motions'][:160], dtype=torch.float32)
masks = torch.tensor(d['masks'][:160], dtype=torch.bool)
object_types = d['object_types'][:160]

# Load per-skeleton offsets from cond_dict; pad to J_max for the encoder
opt = get_opt(None)
cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
J_max = motions.shape[1]

def get_offsets_batch(indices):
    """Build [B, J_max, 3] offset tensor for the given sample indices."""
    batch_offs = []
    for idx in indices:
        base_skel = object_types[idx].split('__')[0]
        raw_offs = cond_dict[base_skel]['offsets']          # [J_real, 3]
        pad = J_max - raw_offs.shape[0]
        padded = np.concatenate([raw_offs, np.zeros((pad, 3))], axis=0)
        batch_offs.append(padded)
    return torch.tensor(np.stack(batch_offs), dtype=torch.float32)

mus = []
with torch.no_grad():
    for i in range(0, 160, 8):
        offs = get_offsets_batch(range(i, i + 8))
        _, mu, _ = model.encoder(motions[i:i+8], offs, masks[i:i+8])
        mus.append(mu.view(8, -1).numpy())

mus = np.concatenate(mus, axis=0)
norms = np.linalg.norm(mus, axis=1, keepdims=True) + 1e-8
mus_n = mus / norms
sim = mus_n @ mus_n.T
np.fill_diagonal(sim, 0)
n = sim.shape[0]

print(f"Checkpoint: {ckpt_dir}/{ckpt_file}")
print(f"mu per-dim std:     {mus.std(axis=0).mean():.6f}")
print(f"mu pairwise cos:    {sim.sum()/(n*(n-1)):.4f}  (0.994=collapsed, <0.9=good)")
print(f"mu norm mean:       {np.linalg.norm(mus, axis=1).mean():.4f}")

# z vs null_z — project both to same space via z_proj
null_z_proj = model.encoder.z_proj(model.null_z.squeeze()).view(-1).detach().numpy()
# Get projected z for comparison
with torch.no_grad():
    z_sample, _, _ = model.encoder(motions[:8], get_offsets_batch(range(8)), masks[:8])
z_flat = z_sample.view(8, -1).numpy()
z_mean = z_flat.mean(axis=0)
z_n = z_mean / (np.linalg.norm(z_mean) + 1e-8)
null_n = null_z_proj / (np.linalg.norm(null_z_proj) + 1e-8)
print(f"z vs null_z cos:    {np.dot(z_n, null_n):.4f}  (0.87=bad, <0.5=good)")
