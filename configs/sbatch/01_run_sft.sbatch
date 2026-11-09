#!/usr/bin/env bash
#SBATCH --job-name=tfa_p1_sft
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# --- Robust Cluster Environment Activation ---
if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---------------------------------------------

echo "=== Phase 1 Step 1: Conversational SFT ==="
python scripts/01_conversational_sft.py \
    --model_id meta-llama/Llama-3.1-8B \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --dataset_split "train_sft[:20000]" \
    --output_dir checkpoints/llama3_1_8b_sft \
    --max_seq_length 1024 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 1

echo "=== Phase 1 Step 2: Merge Adapter into Base ==="
python scripts/01b_merge_sft.py \
    --base_model_id meta-llama/Llama-3.1-8B \
    --adapter_dir checkpoints/llama3_1_8b_sft \
    --output_dir checkpoints/llama3_1_8b_sft_merged

echo "Phase 1 Complete."