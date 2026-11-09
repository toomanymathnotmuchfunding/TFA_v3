"""
scripts/02c_extract_subspaces.py

Phase 2 Step 3: Extract Subspace Projection Directions.
Extracts token hidden activations at a mid-to-late transformer layer (layer 16)
from pi_ref for Domain A and Domain B prompts.

Computes:
1. v_shared: Common/symmetric direction where both domains project with positive sign.
2. v_diff: Contrastive/asymmetric direction separating Domain A from Domain B.

Exports vectors formatted for rank-1 lora_A injection into data/domain_directions.pt.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract domain subspace directions.")
    parser.add_argument(
        "--ref_model_path",
        type=str,
        default="checkpoints/llama3_1_8b_sft_merged",
        help="Path to merged reference model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing domain JSONL files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save subspace summary metrics",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=16,
        help="Layer index to extract hidden state activations from (0-indexed)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=300,
        help="Number of prompts per domain to compute stable centroids",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/domain_directions.pt",
        help="Output path for the serialized directions dictionary",
    )
    return parser.parse_args()


def extract_prompt_activations(model, tokenizer, jsonl_path, target_layer, max_samples):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f][:max_samples]

    activations = []
    print(f"Extracting layer {target_layer} activations from {jsonl_path} (N={len(prompts)})...")

    for p in tqdm(prompts, desc=f"Layer {target_layer} probing"):
        prompt_str = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        enc = tokenizer(prompt_str, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(enc, output_hidden_states=True)
            # Hidden state at the final token of the target layer
            # Shape: (d_model,)
            h_last = outputs.hidden_states[target_layer][0, -1, :].detach().float()
            activations.append(h_last.cpu())

    return torch.stack(activations)  # (N, d_model)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    path_A = os.path.join(args.data_dir, "domain_A_pool.jsonl")
    path_B = os.path.join(args.data_dir, "domain_B_ood.jsonl")

    if not os.path.exists(path_A) or not os.path.exists(path_B):
        raise FileNotFoundError("Domain JSONL pools missing. Run 02a_extract_data.py first.")

    print(f"Loading reference model from {args.ref_model_path} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Extract activation matrices: H_A and H_B
    H_A = extract_prompt_activations(model, tokenizer, path_A, args.target_layer, args.max_samples)
    H_B = extract_prompt_activations(model, tokenizer, path_B, args.target_layer, args.max_samples)

    # Compute domain centroids
    mu_A = H_A.mean(dim=0)
    mu_B = H_B.mean(dim=0)

    # Global centering to isolate cross-domain directions
    mu_global = (mu_A + mu_B) / 2.0
    delta_A = mu_A - mu_global
    delta_B = mu_B - mu_global

    u_A = delta_A / torch.norm(delta_A)
    u_B = delta_B / torch.norm(delta_B)

    # 1. High-similarity shared axis
    v_shared_raw = u_A + u_B
    v_shared = v_shared_raw / torch.norm(v_shared_raw)

    # 2. Contrastive difference axis
    v_diff_raw = u_A - u_B
    v_diff = v_diff_raw / torch.norm(v_diff_raw)

    # Orthogonalize v_diff against v_shared to guarantee perpendicular subspaces
    proj = torch.dot(v_diff, v_shared)
    v_diff = v_diff - proj * v_shared
    v_diff = v_diff / torch.norm(v_diff)

    # Geometric alignment metrics
    cos_sim_shared_A = torch.dot(u_A, v_shared).item()
    cos_sim_shared_B = torch.dot(u_B, v_shared).item()
    cos_sim_diff_A = torch.dot(u_A, v_diff).item()
    cos_sim_diff_B = torch.dot(u_B, v_diff).item()
    dot_shared_diff = torch.dot(v_shared, v_diff).item()

    # Empirical mean projections across all prompt representations
    mean_proj_A_shared = (H_A @ v_shared).mean().item()
    mean_proj_B_shared = (H_B @ v_shared).mean().item()
    mean_proj_A_diff = (H_A @ v_diff).mean().item()
    mean_proj_B_diff = (H_B @ v_diff).mean().item()

    metrics = {
        "target_layer": args.target_layer,
        "d_model": int(v_shared.shape[0]),
        "cos_sim_shared_A": cos_sim_shared_A,
        "cos_sim_shared_B": cos_sim_shared_B,
        "cos_sim_diff_A": cos_sim_diff_A,
        "cos_sim_diff_B": cos_sim_diff_B,
        "dot_shared_diff": dot_shared_diff,
        "mean_proj_A_shared": mean_proj_A_shared,
        "mean_proj_B_shared": mean_proj_B_shared,
        "mean_proj_A_diff": mean_proj_A_diff,
        "mean_proj_B_diff": mean_proj_B_diff,
    }

    print("\n--- Subspace Geometry Verification ---")
    print(f"Mutual Orthogonality (v_shared · v_diff): {dot_shared_diff:.6e} (Target: ~0.0)")
    print(f"Shared Axis Alignment     -> cos(A, v_shared): {cos_sim_shared_A:+.4f} | cos(B, v_shared): {cos_sim_shared_B:+.4f}")
    print(f"Difference Axis Alignment -> cos(A, v_diff):   {cos_sim_diff_A:+.4f} | cos(B, v_diff):   {cos_sim_diff_B:+.4f}")
    print(f"Prompt Mean Projection along v_shared -> Domain A: {mean_proj_A_shared:+.4f} | Domain B: {mean_proj_B_shared:+.4f}")
    print(f"Prompt Mean Projection along v_diff   -> Domain A: {mean_proj_A_diff:+.4f} | Domain B: {mean_proj_B_diff:+.4f}")

    # Export vectors as bfloat16 tensors for direct injection
    payload = {
        "v_shared": v_shared.to(torch.bfloat16),
        "v_diff": v_diff.to(torch.bfloat16),
        "metadata": metrics,
    }

    torch.save(payload, args.output_file)
    print(f"\nSaved subspace vectors to {args.output_file}")

    metrics_path = os.path.join(args.results_dir, "subspace_geometry.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved geometric summary to {metrics_path}")


if __name__ == "__main__":
    main()