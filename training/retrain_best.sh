#!/usr/bin/env bash
#
# Retrain the best preference model — H from sweep 2 (pairwise-difference head).
#
# Best-of-everything configuration documented in reports/01_baseline_sweep.md:
#   head:   pairwise-difference fusion (--pairwise-head)
#           h128 / dropout 0.2 / weight_decay 0.05
#   label:  per-evaluator hard labels with agreement weighting (defaults)
#   prompt: no criterion conditioning (stacking CCP+pairwise regresses)
#   sched:  MLP-only (cached embeddings), patience 50
#
# Expected results (from sweep 2 H):
#   val_acc ≈ 0.611
#   unanim  ≈ 0.65 / majority ≈ 0.60 / split ≈ 0.60
#   train-val gap ≈ 0.15
#   best epoch typically lands around ~150
#
# Output checkpoint will be written to ./checkpoints/best_pairwise/best/
# (containing heads.pt, halluc_head.pt, meta.json — the format the
# taste-scorer inference package expects).
#
# Usage:
#   bash retrain_best.sh
#
# Override data paths / project from the env if needed, e.g.:
#   TRAIN_CSV=~/data/battles_train.csv  bash retrain_best.sh

set -uo pipefail

TRAIN_CSV=${TRAIN_CSV:-data/battles_train.csv}
VAL_CSV=${VAL_CSV:-data/battles_val.csv}
HALLUC_TRAIN_CSV=${HALLUC_TRAIN_CSV:-data/halluc_train.csv}
HALLUC_VAL_CSV=${HALLUC_VAL_CSV:-data/halluc_val.csv}
IMAGE_DIR=${IMAGE_DIR:-data/images}
WANDB_PROJECT=${WANDB_PROJECT:-taste}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/best_pairwise}

mkdir -p "$OUTPUT_DIR"

python train.py \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --halluc-train-csv "$HALLUC_TRAIN_CSV" \
    --halluc-val-csv "$HALLUC_VAL_CSV" \
    --image-dir "$IMAGE_DIR" \
    --batch-size 32 \
    --epochs 1000 \
    --patience 50 \
    --head-hidden-dim 128 \
    --head-dropout 0.2 \
    --weight-decay 0.05 \
    --pairwise-head \
    --seed 42 \
    --wandb-project "$WANDB_PROJECT" \
    --output-dir "$OUTPUT_DIR"
