#!/usr/bin/env bash
#SBATCH --job-name=tfa_p2_profile
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"
mkdir -p logs data results

# --- Robust Cluster Environment Activation ---
if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---------------------------------------------

echo "=== Phase 2 Step 1: Extract Domain Pools ==="
python scripts/02a_extract_data.py \
    --dataset_name PKU-Alignment/BeaverTails \
    --dataset_split 330k_train \
    --output_dir data \
    --max_pairs_per_domain 2500 \
    --seed 42

echo "=== Phase 2 Step 2: Profile b0 and Select Omega ==="
python scripts/02b_profile_b0.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_dir data \
    --results_dir results \
    --omega_size 100

echo "=== Phase 2 Step 3: Spectral Decomposition of R_Omega ==="
python scripts/02c_extract_subspaces.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_dir data \
    --results_dir results \
    --target_layer 16 \
    --rank_k 4 \
    --output_file data/spectral_subspaces.pt

echo "Phase 2 Complete."