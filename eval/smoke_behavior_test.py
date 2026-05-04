import torch, traceback
from utils.fixseed import fixseed
from utils import dist_util
from model.anytop_behavior import AnyTopBehavior
fixseed(42)
dist_util.setup_dist(0)
device = dist_util.dev()

model = AnyTopBehavior(
    max_joints=143, feature_len=13, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4,
    t5_out_dim=768, n_actions=12, use_residual=False, n_behavior_tokens=8,
    skip_t5=False,  # required for InputProcess
)
model.to(device)
model.eval()
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
B, T = 1, 120
x = torch.randn(B, 143, 13, T, device=device)
t = torch.tensor([5], device=device)

# Match the mask shape from dift_probe.py
tmask_np = create_temporal_mask_for_window(31, T)  # [T+1, T+1]
tmask_t = torch.tensor(tmask_np).unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(device)

y = {
    'joints_mask': torch.ones(B, 1, 1, 144, 144, device=device),
    'mask': tmask_t,
    'tpos_first_frame': torch.randn(B, 143, 13, device=device),
    'joints_names_embs': torch.randn(B, 143, 768, device=device),
    'graph_dist': torch.zeros(B, 143, 143, dtype=torch.long, device=device),
    'joints_relations': torch.zeros(B, 143, 143, dtype=torch.long, device=device),
    'crop_start_ind': torch.zeros(B, dtype=torch.long, device=device),
    'n_joints': torch.tensor([30]),
    'psi': torch.randn(B, 64, 62, device=device),
    'action_label': torch.tensor([3], device=device),
}
try:
    with torch.no_grad():
        out = model(x, t, y=y)
    print(f'OK Output: {out.shape}')
except Exception as e:
    traceback.print_exc()
