# Why Cross-Skeleton Retargeting Is Non-Identifiable: Structural Limits of Generative Motion Models

Anonymous code release accompanying the NeurIPS 2026 submission.

This repository contains the implementation of:

1. **Source-Instance Fidelity (SIF)** — a metric that measures whether a cross-skeleton motion generator preserves source-instance identity, on top of the standard contrastive retrieval AUC.
2. **All 14 evaluated methods** (AnyTop, ACE-T/I, MoReFlow-T/I, Motion2Motion-Direct/BVH, AL-Flow / AL-Flow-Src / AL-Flow-Src-G, DPG-SB-v3, ANCHOR retrieval, and two random label-only baselines).
3. **The synthetic 2×2 calibration** that anchors SIF at +1 (oracle) and 0 (source-blind), and the empirical demonstration of `prop:cm` (paired-support variance ladder).
4. **The decoder-nondegeneracy probe** used to validate the gauge non-identifiability theorem on real architectures.
5. **Headline result JSONs** under `results/` so reviewers can verify the numbers cited in the paper without retraining.

---

## Repository Layout

```
model/                   Architecture code (transformers, STT, AnyTop / ACE / MoReFlow / DPG-SB / AL-Flow / skel-blind variants)
train/                   Training scripts for every method we evaluate
sample/                  Inference / generation drivers
data_loaders/            Truebones Zoo dataset loader and topology utilities
diffusion/               DDPM utilities (used by AnyTop family)
eval/
  baselines/             Per-method evaluation runners (run_<method>_v5.py)
  benchmark_v3/          Benchmark construction (queries_v5, queries_sif, action taxonomy)
  synthetic_2x2/         Synthetic calibration: dataset, oracle, prop:cm ladder, demo videos
  gauge_analysis/        P3 rotation perturbation, L-SIF latent SIF, latent dump scripts
  metrics/               Distance metrics and retrieval primitives
figures/scripts/         Paper figure generation scripts
results/                 Headline result JSONs (subset, ~600 KB)
scripts/                 Shell pipelines for end-to-end runs
utils/                   Argument parsing, model creation helpers
tools/                   Small auxiliary tools
environment.yaml         Conda environment specification
REPRODUCE.md             Step-by-step reproduction guide
LICENSE                  MIT
```

---

## Quick Start

### 1. Environment

```bash
conda env create -f environment.yaml
conda activate anytop
```

Python 3.8, PyTorch 1.13+, CUDA 11.6+. A single 12 GB GPU is sufficient for inference and the synthetic calibration. Full training of AnyTop/ACE/MoReFlow requires 1× 24 GB GPU and 12–18 hours per method.

### 2. Set the data root

The repository expects environment variables for paths outside the source tree:

```bash
export ANYTOP_DATA=/path/to/where/you/keep/data    # for the Truebones dataset
```

`PROJECT_ROOT` is computed automatically inside each script via `Path(__file__).resolve().parents[N]`.

### 3. Get the data

The Truebones Zoo dataset must be obtained from its original source (instructions in `REPRODUCE.md` § 1). Place it at `${ANYTOP_DATA}/dataset/truebones/zoo/` and run the preprocessing scripts.

### 4. Run the synthetic 2×2 calibration (no checkpoints needed, ~5 min)

```bash
python -m eval.synthetic_2x2.build_synth_dataset --out save/synthetic_2x2 --seed 0
python -m eval.synthetic_2x2.train_and_eval_synth   # oracle + baselines
python -m eval.synthetic_2x2.propcm_variance_ladder  # paired-support ladder
python -m eval.synthetic_2x2.render_demos            # MP4 demos under save/synthetic_2x2/demos/
```

This reproduces the calibration block of the paper (oracle ρ_SIF = +0.996, source-blind ≈ 0, prop:cm collapse) and writes four MP4 videos visualising the toy world.

### 5. Verify the headline numbers

The paper's `tab:sif`, `tab:sif_intersect`, `tab:propcm_ladder`, and `tab:label_only` numbers can be re-derived from the JSONs already shipped under `results/` — without running any method:

```bash
python -m figures.scripts.fig_sif_max_support      # tab:sif primary
python -m figures.scripts.fig_sif_intersection     # tab:sif intersection
python -m figures.scripts.fig_propcm_ladder        # tab:propcm_ladder
python -m figures.scripts.fig_label_only           # tab:label_only
```

For the full reproduction (training + inference for all 14 methods), see **`REPRODUCE.md`**.

---

## License

MIT, matching the upstream AnyTop release. See `LICENSE`.

---

## Anonymity Note

This repository has been mechanically anonymised: paths, author names, emails, and W&B entity strings have been stripped. If you find any residual identifying information, it is unintentional — please flag it via the OpenReview discussion.
