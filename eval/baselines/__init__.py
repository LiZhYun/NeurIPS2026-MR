"""Baselines reproduction infrastructure for cross-skeleton motion retargeting.

Each baseline produces output matching the unified contract:
  eval/results/baselines/<method>/pair_<id:04d>.npy → [T_out, J_b, 13]
where joint ordering matches cond_dict[skel_b]['joints_names'].

See refine-logs/EXPERIMENT_PLAN_V2.md for the full plan.
"""
