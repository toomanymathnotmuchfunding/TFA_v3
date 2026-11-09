#!/bin/bash
#SBATCH --job-name=tfa_p1_sft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=sched_mit_psfc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p logs checkpoints

source ~/.zshrc || source ~/.bashrc
conda activate tfa_exp

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