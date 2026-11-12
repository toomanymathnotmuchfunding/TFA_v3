#!/usr/bin/env bash
#SBATCH --job-name=tfa_p3_aft
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# ... (keep your standard setup) ...

# 4 Architectures x 2 Intensities = 8 Arms
ARCHS=(
    "frozen_shared" "frozen_shared"
    "frozen_diff"   "frozen_diff"
    "train_shared"  "train_shared"
    "lora4"         "lora4"
)
TARGETS=(
    "max_align" "max_align"
    "max_align" "max_align"
    "max_align" "max_align"
    "max_align" "max_align"
)
EPOCHS=(
    10 50
    10 50
    10 50
    10 50
)

CHOSEN_ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
CHOSEN_TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
CHOSEN_EPOCHS="${EPOCHS[$SLURM_ARRAY_TASK_ID]}"

# Standard vs Overtrained
if [ "$CHOSEN_EPOCHS" -eq 10 ]; then
    RUN_NAME="${CHOSEN_ARCH}_standard"
else
    RUN_NAME="${CHOSEN_ARCH}_overtrained"
fi

OUT_DIR="checkpoints/aft_runs/${RUN_NAME}"

python scripts/03_run_aft.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_path data/omega_train.jsonl \
    --directions_path data/spectral_subspaces_L16.pt \
    --arch "$CHOSEN_ARCH" \
    --target "$CHOSEN_TARGET" \
    --epochs "$CHOSEN_EPOCHS" \
    --learning_rate 2e-3 \
    --output_dir "$OUT_DIR"