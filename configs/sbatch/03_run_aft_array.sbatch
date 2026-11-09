#!/bin/bash
#SBATCH --job-name=tfa_p3_aft
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=sched_mit_psfc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%4

set -euo pipefail

# Ensure working directory is repository root
cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p logs checkpoints/aft_runs

# Load Miniforge module on Engaging worker node
module load miniforge
source ~/.bashrc
conda activate tfa_exp

# Define the 12-arm factorial matrix (6 architectures x 2 target policies)
ARCHS=(
    "frozen_shared" "frozen_shared"
    "frozen_diff"   "frozen_diff"
    "train_shared"  "train_shared"
    "train_diff"    "train_diff"
    "lora4"         "lora4"
    "fft"           "fft"
)

TARGETS=(
    "zero" "overcorrect"
    "zero" "overcorrect"
    "zero" "overcorrect"
    "zero" "overcorrect"
    "zero" "overcorrect"
    "zero" "overcorrect"
)

CHOSEN_ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
CHOSEN_TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${CHOSEN_ARCH}_${CHOSEN_TARGET}"
OUT_DIR="checkpoints/aft_runs/${RUN_NAME}"

echo "=========================================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID | Arm: $RUN_NAME"
echo "Architecture: $CHOSEN_ARCH | Target Policy: $CHOSEN_TARGET"
echo "Output Directory: $OUT_DIR"
echo "=========================================================="

python scripts/03_run_aft.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_path data/omega_train.jsonl \
    --directions_path data/domain_directions.pt \
    --arch "$CHOSEN_ARCH" \
    --target "$CHOSEN_TARGET" \
    --omega 1.0 \
    --beta 0.1 \
    --target_layer 16 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --output_dir "$OUT_DIR"

echo "Arm $RUN_NAME completed successfully."