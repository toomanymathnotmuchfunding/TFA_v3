"""
scripts/02c_sweep_layers.py

Scouting script: Sweeps all layers of the reference model to compute the 
cosine similarity between Domain A (Omega) and Domain B mean shifts.
Outputs a sorted CSV used by 02_run_profiling.sbatch to pick the most 
similar and most orthogonal layers for the OOD transfer experiment.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep all layers for A/B similarity.")
    parser.add_argument("--ref_model_path", type=str, default="checkpoints/llama3_1_8b_sft_merged")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_samples", type=int, default=50, help="Samples per domain (50 is enough)")
    parser.add_argument("--output_file", type=str, default="results/layer_similarity.txt")
    return parser.parse_args()


def load_jsonl(path, limit):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for i, line in enumerate(f) if i < limit]


class MultiLayerHook:
    """Hooks the v_proj input for every layer simultaneously."""
    def __init__(self, model):
        self.hidden = {}
        self.handles = []
        num_layers = len(model.model.layers)
        for i in range(num_layers):
            layer_module = model.model.layers[i].self_attn.v_proj
            self.handles.append(
                layer_module.register_forward_hook(
                    lambda m, i, o, idx=i: self.hidden.__setitem__(idx, i[0].detach())
                )
            )
            
    def remove(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def get_deltas(model, tok, hook, items, desc):
    """Returns a tensor of shape (N, num_layers, hidden_dim)."""
    all_deltas = []
    num_layers = len(model.model.layers)
    
    for item in tqdm(items, desc=desc, leave=False):
        def get_rep(response):
            prompt_str = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{item['prompt']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            full_str = prompt_str + response + "<|eot_id|>"
            
            enc_prompt = tok(prompt_str, return_tensors="pt").input_ids.to(model.device)
            enc_full = tok(full_str, return_tensors="pt").input_ids.to(model.device)
            prompt_len = enc_prompt.shape[1]
            
            model(enc_full)
            
            # Extract and mean-pool response tokens for every layer
            layer_means = []
            for i in range(num_layers):
                resp_hidden = hook.hidden[i][:, prompt_len:, :]
                if resp_hidden.shape[1] == 0:
                    resp_hidden = hook.hidden[i][:, -1:, :]
                layer_means.append(resp_hidden.mean(dim=1).squeeze(0))
                
            return torch.stack(layer_means) # (num_layers, hidden_dim)
            
        h_w = get_rep(item["chosen"])
        h_l = get_rep(item["rejected"])
        all_deltas.append((h_l - h_w).cpu().float())
        
    return torch.stack(all_deltas) # (N, num_layers, hidden_dim)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    items_A = load_jsonl(os.path.join(args.data_dir, "omega_train.jsonl"), args.num_samples)
    items_B = load_jsonl(os.path.join(args.data_dir, "domain_B_ood.jsonl"), args.num_samples)

    print(f"Loading reference model from {args.ref_model_path}...")
    tok = AutoTokenizer.from_pretrained(args.ref_model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()
    hook = MultiLayerHook(model)

    print(f"Extracting Δh for Domain A (N={len(items_A)})...")
    H_A = get_deltas(model, tok, hook, items_A, desc="Domain A")
    
    print(f"Extracting Δh for Domain B (N={len(items_B)})...")
    H_B = get_deltas(model, tok, hook, items_B, desc="Domain B")
    
    hook.remove()

    # Mean shift per layer
    mu_A = H_A.mean(dim=0) # (num_layers, hidden_dim)
    mu_B = H_B.mean(dim=0) # (num_layers, hidden_dim)

    results = []
    num_layers = mu_A.shape[0]
    
    for layer in range(num_layers):
        cos_sim = torch.nn.functional.cosine_similarity(mu_A[layer], mu_B[layer], dim=0).item()
        results.append((layer, cos_sim))
        
    # Sort ascending: most orthogonal first, most similar last
    results.sort(key=lambda x: x[1])

    # Print table
    print(f"\n{'Layer':<6} | {'Cosine Similarity (A vs B)':<25}")
    print("-" * 35)
    for layer, sim in results:
        print(f"L{layer:<5} | {sim:+.4f}")

    # Save to CSV for the bash script
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("layer,cosine_similarity\n")
        for layer, sim in results:
            f.write(f"{layer},{sim}\n")

    print(f"\nSaved layer sweep to {args.output_file}")
    print(f"  Most Orthogonal: L{results[0][0]} (sim: {results[0][1]:+.4f})")
    print(f"  Most Similar:    L{results[-1][0]} (sim: {results[-1][1]:+.4f})")


if __name__ == "__main__":
    main()