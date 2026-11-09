#!/usr/bin/env bash
#SBATCH --job-name=tfa_p4_eval
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"
mkdir -p logs results

# --- Robust Cluster Environment Activation ---
if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---------------------------------------------

echo "=== Phase 4: Evaluating All Checkpoints and Running Regressions ==="
python scripts/04_evaluate_and_regress.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --checkpoints_dir checkpoints/aft_runs \
    --data_dir data \
    --results_dir results \
    --output_filename final_loss_regression_table.csv

echo "Phase 4 Evaluation Complete. Results written to results/final_loss_regression_table.csv"