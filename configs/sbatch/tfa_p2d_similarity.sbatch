#!/usr/bin/env bash
#SBATCH --job-name=tfa_p2d_sim
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0                # CPU-only (H_A/H_B cached by 02c)
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"
mkdir -p logs

if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

echo "=== Phase 2 Step 4: Cross-Domain Cosine Similarity Analysis ==="
python scripts/02d_domain_similarity.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_dir data \
    --target_layer 16 \
    --rank_k 4 \
    --omega_size 100 \
    --cache_file data/hidden_states.pt

echo "Phase 2d Complete."