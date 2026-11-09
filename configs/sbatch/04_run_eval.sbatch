#!/bin/bash
#SBATCH --job-name=tfa_p4_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=sched_mit_psfc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail

mkdir -p logs results

source ~/.zshrc || source ~/.bashrc
conda activate tfa_exp

echo "=== Phase 4: Evaluating All Checkpoints and Running Regressions ==="
python scripts/04_evaluate_and_regress.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --checkpoints_dir checkpoints/aft_runs \
    --data_dir data \
    --results_dir results \
    --output_filename final_loss_regression_table.csv

echo "Phase 4 Evaluation Complete. Results written to results/final_loss_regression_table.csv"