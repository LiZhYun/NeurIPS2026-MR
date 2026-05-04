#!/bin/bash
# Auto-pipeline: waits for v1 training, runs eval, launches v2
# Usage: nohup bash scripts/auto_pipeline.sh > scripts/pipeline.log 2>&1 &

set -e
PYTHON=${ANYTOP_DATA}/miniconda3/envs/anytop/bin/python
cd ${PROJECT_ROOT}

echo "[$(date)] Pipeline started"

# Step 1: Wait for v1 training to finish
echo "[$(date)] Waiting for v1 training (PID 2475170)..."
while kill -0 2475170 2>/dev/null; do
    sleep 60
done
echo "[$(date)] v1 training complete"

# Step 2: Run comprehensive 200-pair eval on final checkpoint
echo "[$(date)] Running comprehensive eval on v1 final checkpoint..."
CKPT="save/cfm_v1/ckpt_final.pt"
if [ ! -f "$CKPT" ]; then
    # Fall back to latest numbered checkpoint
    CKPT=$(ls -t save/cfm_v1/ckpt_*.pt | head -1)
fi
echo "[$(date)] Using checkpoint: $CKPT"
$PYTHON eval/eval_cfm_comprehensive.py --ckpt "$CKPT" --max_pairs 200 --n_steps 30 --cfg_weight 2.0
echo "[$(date)] v1 eval complete"

# Step 3: Launch v2 training (proper train split + circular phase)
echo "[$(date)] Launching v2 training..."
mkdir -p save/cfm_v2
$PYTHON train/train_cfm.py \
    --max_steps 200000 --batch_size 8 --window 40 \
    --d_model 384 --n_layers 12 --n_heads 8 \
    --lr 1e-4 --lambda_il 0.01 \
    --save_dir save/cfm_v2
echo "[$(date)] v2 training complete"

# Step 4: Eval v2
echo "[$(date)] Running comprehensive eval on v2..."
CKPT_V2="save/cfm_v2/ckpt_final.pt"
if [ ! -f "$CKPT_V2" ]; then
    CKPT_V2=$(ls -t save/cfm_v2/ckpt_*.pt | head -1)
fi
$PYTHON eval/eval_cfm_comprehensive.py --ckpt "$CKPT_V2" --max_pairs 200 --n_steps 30 --cfg_weight 2.0
echo "[$(date)] v2 eval complete"

# Step 5: Launch v3 (Phase 2: contact BCE + weighted phase)
echo "[$(date)] Launching v3 Phase 2 training..."
mkdir -p save/cfm_v3
$PYTHON train/train_cfm.py \
    --max_steps 200000 --batch_size 8 --window 40 \
    --d_model 384 --n_layers 12 --n_heads 8 \
    --lr 1e-4 --lambda_il 0.01 \
    --contact_bce --lambda_contact 0.5 --lambda_phase 2.0 \
    --save_dir save/cfm_v3
echo "[$(date)] v3 training complete"

# Step 6: Eval v3
echo "[$(date)] Running comprehensive eval on v3..."
CKPT_V3="save/cfm_v3/ckpt_final.pt"
if [ ! -f "$CKPT_V3" ]; then
    CKPT_V3=$(ls -t save/cfm_v3/ckpt_*.pt | head -1)
fi
$PYTHON eval/eval_cfm_comprehensive.py --ckpt "$CKPT_V3" --max_pairs 200 --n_steps 30 --cfg_weight 2.0
echo "[$(date)] v3 eval complete"

echo "[$(date)] Pipeline finished. All results in eval/benchmark_paired/pairs/comprehensive_*.json"
