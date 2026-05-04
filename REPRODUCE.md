# Reproduction Guide

This document walks through reproducing every quantitative claim in the paper. It is structured by claim, not by method — start at the result you want to verify.

> **Time / compute budget**:
> - **Verify-only (no retraining)**: ~10 min, CPU is fine. Just regenerates tables and figures from the shipped `results/` JSONs.
> - **Synthetic 2×2 calibration**: ~5 min on CPU, ~2 min on GPU.
> - **Full retrain of one method**: 12–18 h on a single 24 GB GPU.
> - **Full retrain of all 14 methods**: ~10 GPU-days.

---

## 0. Setup

```bash
# Conda env
conda env create -f environment.yaml
conda activate anytop

# Set the data root (anywhere you have ~5 GB of free space)
export ANYTOP_DATA=/your/path/here

# (Optional) for W&B logging during training
export WANDB_ENTITY=your-entity
export WANDB_PROJECT=anytop-anon
```

The codebase computes `PROJECT_ROOT` automatically from each script's location, so no env var is needed for the source tree.

---

## 1. Truebones Zoo dataset

**Source**: Truebones Zoo motion library (request access via the original distributor — we do not redistribute it).

After obtaining the raw `.bvh` files, run:

```bash
mkdir -p ${ANYTOP_DATA}/dataset/truebones/zoo
# place raw BVH files under ${ANYTOP_DATA}/dataset/truebones/zoo/raw/

python -m data_loaders.truebones.preprocess \
    --raw_dir ${ANYTOP_DATA}/dataset/truebones/zoo/raw \
    --out_dir ${ANYTOP_DATA}/dataset/truebones/zoo/truebones_processed
```

This produces normalized motion `.npy` files and the per-skeleton `cond.npy` metadata used everywhere downstream.

---

## 2. Verify the headline tables (no retraining)

The shipped `results/` directory contains the per-method SIF JSONs and the aggregate robustness analysis. To regenerate the four headline tables and figures from them:

```bash
# tab:sif (primary, max-support n up to 49)
python -m figures.scripts.fig_sif_max_support

# tab:sif_intersect (cross-method 37-triple intersection)
python -m figures.scripts.fig_sif_intersection

# tab:propcm_ladder (paired-support variance ladder)
python -m figures.scripts.fig_propcm_ladder

# tab:label_only (action-only retrieval baselines)
python -m figures.scripts.fig_label_only
```

The output PDFs land in `figures/`. Compare against the paper's tables — every number traces to a `results/baselines/<method>_sif/sif_metric*.json` field.

---

## 3. Synthetic 2×2 calibration

Reproduces ρ_SIF = +0.996 (oracle) and ≈ 0 (source-blind), and the `prop:cm` variance ladder.

```bash
# Build the four 2×2 cells (sparse/dense × paired/unpaired)
python -m eval.synthetic_2x2.build_synth_dataset --out save/synthetic_2x2 --seed 0

# Train oracle + source-blind baselines on each cell, compute SIF
python -m eval.synthetic_2x2.train_and_eval_synth \
    --data_root save/synthetic_2x2 \
    --out_json results/synth_2x2_sif_validation.json

# prop:cm: variance-ratio ladder over M ∈ {1, 2, 4, 8, 16, 32, 50}
for SEED in 42 43 44; do
    python -m eval.synthetic_2x2.propcm_variance_ladder \
        --seed $SEED \
        --out_json results/propcm_variance_ladder_seed${SEED}.json
done

# Render the four MP4 demos
python -m eval.synthetic_2x2.render_demos
```

Outputs:
- `results/synth_2x2_sif_validation.json` — oracle + baseline ρ_SIF (matches `tab:synthetic_sif_calibration`)
- `results/propcm_variance_ladder_seed{42,43,44}.json` — variance ratios per M (matches `fig:propcm_ladder`)
- `save/synthetic_2x2/demos/{actions_overview, skeleton_scale, oracle_transport, instance_noise}.mp4` — visualisations

---

## 4. Train and evaluate the 14 methods

Each method has its own training script under `train/` and its own evaluation runner under `eval/baselines/`. The pattern is identical:

```bash
# 1. Train (one-time, ~12-18 h on 1× 24 GB GPU)
python -m train.train_<method> --run_name <method>_v3 --max_steps <N>

# 2. Inference on the SIF benchmark fold
python -m eval.baselines.run_<method>_v5 \
    --ckpt save/<method>/<method>_v3/final.pt \
    --manifest eval/benchmark_v3/queries_sif/manifest.json \
    --output_dir eval/results/baselines/<method>_sif

# 3. Compute SIF on the saved outputs
python -m eval.baselines.run_anchor_sif \
    --predictions_dir eval/results/baselines/<method>_sif \
    --output_json eval/results/baselines/<method>_sif/sif_metric.json
```

Per-method details:

| Method | Train script | Inference runner | Train time |
|---|---|---|---|
| AnyTop | `train.train_conditioned` | `eval.baselines.run_anytop_v5` | ~17 h |
| ACE-T (transductive) | `train.train_ace --transductive` | `eval.baselines.run_ace_v3 --variant T` | ~14 h |
| ACE-I (inductive) | `train.train_ace` | `eval.baselines.run_ace_v3 --variant I` | ~14 h |
| MoReFlow-T | `train.train_moreflow --transductive` | `eval.baselines.run_moreflow_v3 --variant T` | ~12 h |
| MoReFlow-I | `train.train_moreflow` | `eval.baselines.run_moreflow_v3 --variant I` | ~12 h |
| AL-Flow | `train.train_anchor_label_flow` | `eval.baselines.run_anchor_label_flow_v5` | ~15 h |
| AL-Flow-Src | `train.train_anchor_label_flow_src` | `eval.baselines.run_anchor_label_flow_src_v5` | ~15 h |
| AL-Flow-Src-G | `train.train_anchor_label_flow_src_g` | `eval.baselines.run_anchor_label_flow_src_g_v5` | ~15 h |
| DPG-SB-v3 | `train.train_dpg_sb_v3` | `eval.baselines.run_dpg_sb_v2 --ckpt <v3-ckpt>` | ~30–60 min (uses frozen MoReFlow latents) |
| Motion2Motion-Direct | (inference-only, requires upstream M2M ckpt) | `eval.baselines.run_m2m_direct_v5` | n/a |
| Motion2Motion-BVH | (inference-only, BVH I/O wrapper) | `eval.baselines.run_m2m_official_v5` | n/a |
| ANCHOR | (deterministic retrieval) | `eval.baselines.run_anchor_sif` | n/a |
| random-same-cluster | (label-only baseline) | `eval.baselines.run_random_same_cluster` | n/a |
| random-same-exact-action | (label-only baseline) | `eval.baselines.run_random_same_exact_action` | n/a |

---

## 5. Aggregate to the cross-method robustness table

After running per-method SIF, the aggregate analysis (LOMO sensitivity, paired Wilcoxon, max-support and 37-triple intersection):

```bash
python -m eval.aggregate_sif_robustness \
    --baseline_dir eval/results/baselines \
    --output_json results/sif_robustness_analysis.json
```

This file is what `tab:sif`, `tab:sif_intersect`, and the LOMO-sensitivity sentence in §5 are built from.

---

## 6. Decoder-nondegeneracy (gauge probe on real architectures)

```bash
# Single-cell pilot
python -m eval.gauge_analysis.decoder_nondegeneracy \
    --model anytop --ckpt save/<your-anytop-ckpt> \
    --output_json results/decoder_nondegeneracy_anytop.json

# 72-cell sweep across (skel × action × diffusion timestep × hook layer)
python -m eval.gauge_analysis.decoder_nondegeneracy_sweep \
    --model anytop --ckpt save/<your-anytop-ckpt> \
    --output_json results/decoder_nondegeneracy_anytop_sweep.json

# Synthetic VAE counterexample (small bottleneck)
python -m eval.gauge_analysis.decoder_nondegeneracy_synth \
    --output_json results/decoder_nondegeneracy_synth.json
```

These reproduce the §5.23 result that AnyTop's decoder is strongly non-degenerate (≈ 408 % gauge sensitivity vs noise floor) while a small KL-regularised VAE is approximately gauge-invariant.

---

## 7. Pre-trained checkpoints (optional)

To skip retraining, the trained checkpoints for all 14 methods are available at the following anonymous mirror:

> **TODO (paper authors)**: upload checkpoints to a single anonymous Drive / Zenodo / Hugging Face entry and paste the URL here.
>
> Suggested package layout:
> ```
> anytop-checkpoints.tar.gz   (~3 GB)
>   ├── anytop_v5/
>   ├── ace_t_v5/  ace_i_v5/
>   ├── moreflow_t_v5/  moreflow_i_v5/
>   ├── anchor_label_flow/  anchor_label_flow_src/  anchor_label_flow_src_g/
>   └── dpg_sb_v3/
> ```
>
> Reviewers download, untar into `save/`, and skip §4 training.

---

## 8. Notes and known gotchas

- **Inductive vs transductive training**: AnyTop and ACE-I use transductive Stage A pre-training on all 70 skeletons; AL-Flow variants are strictly inductive. AL-Flow therefore *cleanly skips* queries on held-out test skeletons (Spider, Elephant, Crab as targets; plus more for AL-Flow-Src). This is reported transparently in `eval/results/baselines/anchor_label_flow*_sif/metrics.json` (`status: skipped_skel_*_not_in_train`) and is the reason the cross-method intersection is n=37 rather than n=49.
- **DPG-SB-v3 uses the v2 inference adapter**: by design, the v3 training script writes a v2-API-compatible checkpoint. Use `run_dpg_sb_v2.py --ckpt save/dpg_sb/dpg_sb_v3/final.pt`.
- **`m3_rerank_v1`**: an additional retrieval-with-rerank baseline; included for completeness but not part of the 14-method headline.
- **Random seeds**: every script accepts `--seed`; defaults are 42 (matches paper). The synthetic 2×2 ladder uses three seeds (42, 43, 44) for stability.
