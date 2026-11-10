#!/usr/bin/env bash
#SBATCH --job-name=tfa_p2_profile
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
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

echo "=== Phase 2 Step 3: Sweep All Layers for Similarity ==="
python scripts/02c_sweep_layers.py \
    --ref_model_path checkpoints/llama3_1_8b_sft_merged \
    --data_dir data \
    --num_samples 50 \
    --output_file results/layer_similarity.txt

# --- Parse the best layers from the sweep ---
# The file is sorted: 1st data row (line 2) is most orthogonal, last row is most similar.
ORTHO_LAYER=$(sed -n '2p' results/layer_similarity.txt | cut -d',' -f1)
SIM_LAYER=$(tail -n 1 results/layer_similarity.txt | cut -d',' -f1)

echo ""
echo "=========================================================="
echo "Sweep Results:"
echo "  Most Orthogonal Layer: L${ORTHO_LAYER}"
echo "  Most Similar Layer:    L${SIM_LAYER}"
echo "=========================================================="
echo ""

# Array of layers to test: Baseline (16) + Similar + Orthogonal
LAYERS_TO_TEST=("16" "$SIM_LAYER" "$ORTHO_LAYER")
# Deduplicate and sort numerically
UNIQUE_LAYERS=($(echo "${LAYERS_TO_TEST[@]}" | tr ' ' '\n' | sort -nu | tr '\n' ' '))

for LAYER in "${UNIQUE_LAYERS[@]}"; do
    echo "=== Phase 2 Step 4: Extract Subspaces for Layer $LAYER ==="
    python scripts/02d_extract_subspaces.py \
        --ref_model_path checkpoints/llama3_1_8b_sft_merged \
        --data_dir data \
        --results_dir results \
        --target_layer "$LAYER" \
        --rank_k 4 \
        --output_file "data/spectral_subspaces_L${LAYER}.pt"

    echo "=== Phase 2 Step 5: Analyze Similarity for Layer $LAYER ==="
    python scripts/02e_domain_similarity.py \
        --ref_model_path checkpoints/llama3_1_8b_sft_merged \
        --data_dir data \
        --target_layer "$LAYER" \
        --rank_k 4 \
        --omega_size 100 \
        --cache_file "data/hidden_states_L${LAYER}.pt" | tee "results/similarity_analysis_L${LAYER}.txt"
    echo ""
done

echo "Phase 2 Complete. Generated subspaces for layers: ${UNIQUE_LAYERS[*]}"