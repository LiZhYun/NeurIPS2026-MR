#!/bin/bash
# Auto-pipeline: wait for Idea K eval → train supervised CFM → eval

set -e
PYTHON=${ANYTOP_DATA}/miniconda3/envs/anytop/bin/python
IDEA_K_PID=${1:-0}

# Step 1: Wait for Idea K eval to finish (if PID provided)
if [ "$IDEA_K_PID" -gt 0 ] 2>/dev/null; then
    echo "[$(date)] Waiting for Idea K eval (PID $IDEA_K_PID)..."
    while kill -0 "$IDEA_K_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date)] Idea K eval complete"
fi

# Step 2: Train supervised CFM (100k steps, ~4h)
echo "[$(date)] Starting supervised CFM training..."
$PYTHON train/train_cfm_supervised.py \
    --max_steps 100000 --batch_size 8 --window 40 \
    --d_model 384 --n_layers 12 --n_heads 8 \
    --lr 1e-4 --save_dir save/cfm_supervised
echo "[$(date)] Supervised training complete"

# Step 3: Comprehensive eval
echo "[$(date)] Running comprehensive eval on supervised CFM..."
$PYTHON eval/eval_cfm_comprehensive.py \
    --ckpt save/cfm_supervised/ckpt_final.pt \
    --max_pairs 200 --n_steps 30 --cfg_weight 2.0
echo "[$(date)] Supervised eval complete"
