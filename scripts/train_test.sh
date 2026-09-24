#!/usr/bin/env bash
# Train for 100 epochs, then test the saved weights.
# Usage: scripts/train_test.sh [DATASET]
set -euo pipefail
cd "$(dirname "$0")/.."

dataset="${1:-TwoPatterns}"
model_dir="models"

args=(
    --dataset_name "$dataset"
    --use_ucr2018
    --patch_ratio 10
    --learning_rate 1e-4
    --batch_size 256
    --model_dir "$model_dir"
    --output_dir out
    --patching_method derivative   # normal | derivative
    --model_arch vit            # vit | patchtst
    --curvature_scale 0      # derivative only; typically 0-1.5
    # --patchtst_stride 5       # patchtst only; patch size // 2 if omitted

)

python main.py --train --epochs 100 --run_id 0 "${args[@]}"
python main.py --test_model_path "$(tail -n 1 "models/${model_dir}_${dataset}/modelspath.txt")" "${args[@]}"
