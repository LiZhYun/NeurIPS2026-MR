"""Check if z codes cluster by SKELETON (bad) or by MOTION TYPE (good)."""
import os, numpy as np, torch
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_conditioned_model_and_diffusion, load_model
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from sample.generate_conditioned import load_args_from_checkpoint
from os.path import join as pjoin

fixseed(10)
dist_util.setup_dist(0)

ckpt_args = load_args_from_checkpoint('save/C_FSQ_mask_bs_2_latentdim_256/model000599999.pt')

class NS:
    def __init__(self, d):
        self.__dict__.update(d)

ckpt = NS(ckpt_args)
opt = get_opt(0)
cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

model, _ = create_conditioned_model_and_diffusion(ckpt)
state = torch.load('save/C_FSQ_mask_bs_2_latentdim_256/model000599999.pt', map_location='cpu')
load_model(model, state)
model.to('cuda')
model.eval()

test_skels = ['Horse', 'Bear', 'Crab', 'Eagle', 'Anaconda', 'Bat', 'Lion', 'Scorpion']
motion_dir = opt.motion_dir
n_frames = 120
all_z, all_labels, all_mtypes = [], [], []

for skel in test_skels:
    if skel not in cond_dict:
        continue
    J = cond_dict[skel]['offsets'].shape[0]
    files = sorted(f for f in os.listdir(motion_dir)
                   if f.startswith(skel + '_') and f.endswith('.npy'))[:5]
    for mf in files:
        raw = np.load(pjoin(motion_dir, mf))
        T = raw.shape[0]
        mean = cond_dict[skel]['mean']
        std = cond_dict[skel]['std'] + 1e-6
        norm = np.nan_to_num((raw - mean[None, :]) / std[None, :])
        if T >= n_frames:
            norm = norm[:n_frames]
        else:
            norm = np.concatenate([norm, np.zeros((n_frames - T, J, 13))], 0)

        sm = np.zeros((n_frames, opt.max_joints, 13))
        sm[:, :J, :] = norm
        so = np.zeros((opt.max_joints, 3))
        so[:J, :] = cond_dict[skel]['offsets']

        sm_t = torch.tensor(sm).permute(1, 2, 0).float().unsqueeze(0).cuda()
        so_t = torch.tensor(so).float().unsqueeze(0).cuda()
        mask = torch.zeros(1, opt.max_joints, dtype=torch.bool).cuda()
        mask[0, :J] = True

        with torch.no_grad():
            enc = model.encoder(sm_t, so_t, mask)
            z = enc[0] if isinstance(enc, tuple) else enc

        all_z.append(z[0].cpu().numpy().reshape(-1))
        all_labels.append(skel)
        parts = mf.replace('.npy', '').split('___')
        all_mtypes.append(parts[1].split('_')[0] if len(parts) > 1 else 'unk')

all_z = np.array(all_z)
print(f'Encoded {len(all_z)} motions, z dim={all_z.shape[1]}')

zn = all_z / (np.linalg.norm(all_z, axis=1, keepdims=True) + 1e-8)
cos = zn @ zn.T

ss, cs = [], []
for i in range(len(all_z)):
    for j in range(i + 1, len(all_z)):
        (ss if all_labels[i] == all_labels[j] else cs).append(cos[i, j])

print(f'\nSame skeleton:   {np.mean(ss):.3f} +/- {np.std(ss):.3f}')
print(f'Cross skeleton:  {np.mean(cs):.3f} +/- {np.std(cs):.3f}')
print(f'Skeleton gap:    {np.mean(ss) - np.mean(cs):.3f}')

sm2, dm = [], []
for i in range(len(all_z)):
    for j in range(i + 1, len(all_z)):
        (sm2 if all_mtypes[i] == all_mtypes[j] else dm).append(cos[i, j])

if sm2:
    print(f'\nSame motion:     {np.mean(sm2):.3f} +/- {np.std(sm2):.3f}')
if dm:
    print(f'Diff motion:     {np.mean(dm):.3f} +/- {np.std(dm):.3f}')
if sm2 and dm:
    print(f'Motion gap:      {np.mean(sm2) - np.mean(dm):.3f}')

print(f'\nMotion types: {sorted(set(all_mtypes))}')
print(f'Labels per skel: {[(s, sum(1 for l in all_labels if l==s)) for s in test_skels if s in set(all_labels)]}')
