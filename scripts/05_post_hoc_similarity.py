"""
scripts/05_post_hoc_similarity.py
Evaluates the geometric alignment of H_A and H_B inside the trained LoRA subspace.
"""
import os
import json
import argparse
import torch
import safetensors.torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", default="checkpoints/aft_runs")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--target_layer", type=int, default=16)
    p.add_argument("--cache_file", default="data/hidden_states_L16.pt")
    return p.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.cache_file):
        raise FileNotFoundError(f"Missing {args.cache_file}. Run 02d first.")
        
    data = torch.load(args.cache_file, weights_only=True)
    H_A, H_B = data["H_A"], data["H_B"] # (N, 4096)
    
    ckpts = [d for d in os.listdir(args.checkpoints_dir) if os.path.isdir(os.path.join(args.checkpoints_dir, d))]
    
    print(f"{'Run Name':<30} | {'CosSim(A,B) in Update Subspace':<30}")
    print("-" * 65)

    for ckpt in sorted(ckpts):
        adapter_path = os.path.join(args.checkpoints_dir, ckpt)
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        
        if not os.path.exists(safetensors_path):
            continue # Skip FFT or failed runs
            
        state_dict = safetensors.torch.load_file(safetensors_path)
        
        # Find the A and B matrices for the target layer up_proj
        A_key = [k for k in state_dict if f"layers.{args.target_layer}" in k and "lora_A" in k and "up_proj" in k]
        B_key = [k for k in state_dict if f"layers.{args.target_layer}" in k and "lora_B" in k and "up_proj" in k]
        
        if not A_key or not B_key:
            continue
            
        A = state_dict[A_key[0]].float() # Shape (4, 4096)
        B = state_dict[B_key[0]].float() # Shape (14336, 4)
        
        # The effective weight update is B @ A
        delta_W = B @ A # Shape (14336, 4096)
        
        # We want the input-space directions (column space)
        # Since A is rank 4, V_top from SVD will just be the row space of A.
        # We can just project H directly using A.T (since A is orthonormal-ish)
        V_top = A / (A.norm(dim=1, keepdim=True) + 1e-12) # Normalize rows
        
        # Project H_A and H_B onto the learned subspace
        H_A_proj = H_A @ V_top.T # Shape (N, 4)
        H_B_proj = H_B @ V_top.T # Shape (N, 4)
        
        mu_A = H_A_proj.mean(dim=0)
        mu_B = H_B_proj.mean(dim=0)
        
        cos_sim = torch.nn.functional.cosine_similarity(mu_A, mu_B, dim=0).item()
        
        # We can also compute the norm (magnitude of the shift in this subspace)
        mag_A = mu_A.norm().item()
        mag_B = mu_B.norm().item()
        
        print(f"{ckpt:<30} | {cos_sim:+.4f} (mag: {mag_A:.2f} / {mag_B:.2f})")

if __name__ == "__main__":
    main()