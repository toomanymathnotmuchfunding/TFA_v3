#!/bin/bash
#SBATCH --job-name=tfa_p2_profile
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=sched_mit_psfc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00

set -euo pipefail

# Ensure working directory is repository root
cd "${SLURM_SUBMIT_DIR:-.}"

mkdir -p logs data results

# Load Miniforge module on Engaging worker node
module load miniforge
source ~/.bashrc
conda activate msm_env

echo "=== Phase 2 Step 1: Extract Domain Pools ==="
python scripts/02a_extract_data.py \
    --dataset_name PKU-Alignment/BeaverTails \
    --dataset_split 330k_train \
    --output_dir data \
    --max_pairs_per_domain 2500

echo "=== Phase 2 Step 2: Profile b0 and Select Omega ==="
python scripts/02b_profile_b0.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_dir data \
    --results_dir results \
    --omega_size 100

echo "=== Phase 2 Step 3: Extract Geometric Subspaces ==="
python scripts/02c_extract_subspaces.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_dir data \
    --results_dir results \
    --target_layer 16 \
    --max_samples 300 \
    --output_file data/domain_directions.pt

echo "Phase 2 Complete."