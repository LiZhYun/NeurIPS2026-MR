#!/bin/bash
# Full eval pipeline for DPG. All metrics: V5 standard + B1 gold + Realism (Tier 1+2) + paired sig tests.
# Usage: bash eval/run_dpg_full_eval.sh <ckpt_path> <out_tag>
set -e
CKPT="${1:-save/dpg/dpg_v1/final.pt}"
TAG="${2:-dpg_v1}"
ROOT=${PROJECT_ROOT}
cd "$ROOT"

echo "============================================================"
echo "Full DPG eval pipeline"
echo "  ckpt: $CKPT"
echo "  tag: $TAG"
echo "============================================================"

# Step 1: Inference on V5 fold 42 + 43
echo "--- Step 1: Inference on V5 ---"
conda run -n anytop python -u -m eval.baselines.run_dpg_v5 \
  --folds 42 43 --ckpt "$CKPT" --out_tag "$TAG" --n_steps 20

# Step 2: V5 standard eval (3 distances × 2 folds)
echo "--- Step 2: V5 standard eval ---"
for fold in 42 43; do
  for dist in procrustes zscore_dtw q_component; do
    conda run -n anytop python -m eval.benchmark_v3.eval_v5 \
      --method_dir "save/dpg/$TAG/fold_$fold" \
      --fold $fold --method_name "$TAG" --distance $dist 2>&1 \
      | grep -E "(cluster-tier|exact-tier)" | head -4 | sed "s|^|[V5 f$fold $dist] |"
  done
done

# Step 3: B1 gold subset eval (3 distances)
echo "--- Step 3: B1 gold subset eval ---"
for dist in procrustes zscore_dtw q_component; do
  conda run -n anytop python -m eval.benchmark_v3.eval_b1_gold \
    --method_root "save/dpg/$TAG" \
    --method_name "$TAG" --distance $dist 2>&1 \
    | grep -E "(=== B1|exact-tier AUC|hit@1|hit@all_pos|MRR)" | head -6 | sed "s|^|[B1 $dist] |"
done

# Step 4: Realism eval (Tier 1 + Tier 2 with FID)
echo "--- Step 4: Realism eval (Tier 1 + Tier 2) ---"
conda run -n anytop python -u -m eval.benchmark_v3.eval_v5_realism \
  --method_dir "save/dpg/$TAG" --method_name "$TAG" --include_fid \
  --folds 42 43 2>&1 | tail -20

# Step 5: Paired sig tests vs key baselines
echo "--- Step 5: Paired sig tests ---"
for OPP_NAME in M3_PhaseA_v1 i5_action_classifier k_retrieve action_oracle; do
  case "$OPP_NAME" in
    M3_PhaseA_v1) OPP_DIR="save/m3/m3_rerank_v1" ;;
    i5_action_classifier) OPP_DIR="save/oracles/v3/i5_action_classifier_v5" ;;
    k_retrieve) OPP_DIR="eval/results/baselines/k_retrieve_v5" ;;
    action_oracle) OPP_DIR="save/oracles/v3/action_oracle_v5" ;;
  esac
  echo "    DPG vs $OPP_NAME"
  conda run -n anytop python -m eval.benchmark_v3.paired_sig_test \
    --method_a "save/dpg/$TAG" --name_a "$TAG" \
    --method_b "$OPP_DIR" --name_b "$OPP_NAME" \
    --folds 42 43 --distance all --splits test_test \
    --out_path "save/dpg/$TAG/paired_vs_${OPP_NAME}.json" 2>&1 \
    | grep -E "diff=|wins=" | sed "s|^|        |"
done

echo "============================================================"
echo "Full DPG eval pipeline complete. Outputs in save/dpg/$TAG/"
echo "============================================================"
