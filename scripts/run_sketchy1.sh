#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/content/sketchy/Sketchy}"
WORKERS="${WORKERS:-8}"
ADAPTER_DIR="${ADAPTER_DIR:-teacher_adapter_runs/dfn5b_sketchy1}"
EXP_NAME="${EXP_NAME:-sketchy1_dfn5b_kd}"

python -m src.finetune_dfn5b_adapter \
  --root "$ROOT" \
  --dataset sketchy_1 \
  --epochs 1 \
  --encode_batch_size 64 \
  --batch_size 128 \
  --workers "$WORKERS" \
  --fp16_backbone \
  --bottleneck_dim 32 \
  --lr 1e-4 \
  --temperature 0.07 \
  --lambda_retrieval 1 \
  --lambda_semantic 0.5 \
  --seed 42 \
  --output_dir "$ADAPTER_DIR"

python -m src.main_train \
  --root "$ROOT" \
  --dataset sketchy_1 \
  --epochs 3 \
  --teacher_adapter_ckpt "$ADAPTER_DIR/best.pt" \
  --workers "$WORKERS" \
  --batch_size 64 \
  --progress \
  --lr 4e-5 \
  --quantize_fp16 \
  --seed 42 \
  --lambda_kd 3 \
  --kd_temperature 0.07 \
  --n_ctx 1 \
  --lambda_cls 1 \
  --lambda_triplet 1 \
  --exp_name "$EXP_NAME"
