"""
scripts/02c_extract_subspaces.py

Phase 2 Step 3: Spectral Decomposition of Representation Covariance R_Omega.
Captures activations at target_layer (v_proj input) for prompt-response pairs.
Computes:
  - Delta h(x) = h(rejected) - h(chosen)
  - Truncated SVD on H_A (N x d) to extract:
      1. V_topk: Top-k eigenvectors of R_Omega (k x d_in)
      2. v_shared: Leading principal component across domains (d_in,)
      3. v_diff: Leading differential component between Domain A and Domain B (d_in,)
Saves results dictionary to --output_file (data/spectral_subspaces.pt).
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract spectral subspaces for R_Omega")
    parser.add_argument("--ref_model_path", type=str, default="checkpoints/llama3_1_8b_sft_merged")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--omega_size", type=int, default=100)
    parser.add_argument("--target_layer", type=int, default=16)
    parser.add_argument("--rank_k", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="data/spectral_subspaces.pt")
    return parser.parse_args()


class RepresentationExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.captured_hidden = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        layer_module = self.model.model.layers[self.target_layer].self_attn.v_proj

        def hook_fn(module, input, output):
            # input[0] shape: (batch_size, seq_len, hidden_dim)
            self.captured_hidden = input[0].detach()

        self._hook_handle = layer_module.register_forward_hook(hook_fn)

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()


@torch.inference_mode()
def extract_delta_h(model, tokenizer, extractor, prompt, chosen, rejected):
    def get_rep(response):
        prompt_str = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        full_str = prompt_str + response + "<|eot_id|>"

        enc_prompt = tokenizer(prompt_str, return_tensors="pt").input_ids.to(model.device)
        enc_full = tokenizer(full_str, return_tensors="pt").input_ids.to(model.device)
        prompt_len = enc_prompt.shape[1]

        model(enc_full)
        resp_hidden = extractor.captured_hidden[:, prompt_len:, :]
        if resp_hidden.shape[1] == 0:
            resp_hidden = extractor.captured_hidden[:, -1:, :]
        mean_hidden = resp_hidden.mean(dim=1).squeeze(0)  # (hidden_dim,)
        return mean_hidden.cpu().float()

    h_w = get_rep(chosen)
    h_l = get_rep(rejected)
    dh = h_l - h_w

    if not torch.isfinite(dh).all():
        dh = torch.nan_to_num(dh, nan=0.0, posinf=1.0, neginf=-1.0)
    return dh


def load_jsonl(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    omega_path = os.path.join(args.data_dir, "omega_train.jsonl")
    domain_b_path = os.path.join(args.data_dir, "domain_B_ood.jsonl")

    if not os.path.exists(omega_path) or not os.path.exists(domain_b_path):
        raise FileNotFoundError("Missing omega_train.jsonl or domain_B_ood.jsonl in data_dir.")

    omega_data = load_jsonl(omega_path, limit=args.omega_size)
    domain_b_data = load_jsonl(domain_b_path, limit=len(omega_data))

    print(f"Loading reference model from {args.ref_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    extractor = RepresentationExtractor(model, args.target_layer)

    print(f"Extracting Delta h over Omega (Domain A, N={len(omega_data)})...")
    deltas_A = []
    for item in tqdm(omega_data, desc="Extracting Omega Reps"):
        dh = extract_delta_h(model, tokenizer, extractor, item["prompt"], item["chosen"], item["rejected"])
        deltas_A.append(dh)
    H_A = torch.stack(deltas_A, dim=0).float()  # (N_A, d)

    print(f"Extracting Delta h over Domain B sample (N={len(domain_b_data)})...")
    deltas_B = []
    for item in tqdm(domain_b_data, desc="Extracting Domain B Reps"):
        dh = extract_delta_h(model, tokenizer, extractor, item["prompt"], item["chosen"], item["rejected"])
        deltas_B.append(dh)
    H_B = torch.stack(deltas_B, dim=0).float()  # (N_B, d)

    extractor.remove()

    print("Running SVD on representation matrix...")

    # H_A is (N, d). SVD returns Vh with shape (min(N, d), d) = (100, 4096)
    U, S, Vh = torch.linalg.svd(H_A, full_matrices=False)
    
    N_A = H_A.shape[0]
    eigenvalues = (S ** 2) / N_A

    # Top-k eigenvectors of R_Omega: shape (k, d)
    V_topk = Vh[:args.rank_k, :].contiguous()

    # Leading shared direction: 1D vector of shape (d,)
    v_shared = Vh[0, :].contiguous()
    v_shared = v_shared / (torch.norm(v_shared) + 1e-12)

    # Leading difference direction: mean shift projected orthogonal to v_shared
    mu_A = H_A.mean(dim=0)
    mu_B = H_B.mean(dim=0)
    v_diff_raw = mu_A - mu_B
    proj = torch.dot(v_diff_raw, v_shared) * v_shared
    v_diff = v_diff_raw - proj
    v_diff_norm = torch.norm(v_diff).item()

    if v_diff_norm <= 1e-6:
        raise RuntimeError(
            "Differential direction is degenerate: ||mu_A - mu_B|| orthogonal to "
            f"v_shared is {v_diff_norm:.2e}. Domain A and Domain B share the same "
            "leading direction in this layer, so a 'differential' subspace is not "
            "identifiable. Either (a) change --target_layer, (b) increase --rank_k "
            "and inspect V_topk, or (c) report that the _diff arms are undefined "
            "for this configuration."
        )
    v_diff = v_diff / v_diff_norm

    # Guarantee all outputs are clean 1D/2D float CPU tensors
    subspace_payload = {
        "v_shared": v_shared.cpu().float(),
        "v_diff": v_diff.cpu().float(),
        "V_topk": V_topk.cpu().float(),
        "eigenvalues": eigenvalues[:args.rank_k].cpu().float(),
        "target_layer": int(args.target_layer),
        "rank_k": int(args.rank_k),
    }

    torch.save(subspace_payload, args.output_file)
    print(f"\nSuccessfully extracted spectral subspaces!")
    print(f"  v_shared norm: {torch.norm(v_shared):.4f} | shape: {list(v_shared.shape)}")
    print(f"  v_diff norm:   {torch.norm(v_diff):.4f} | shape: {list(v_diff.shape)}")
    print(f"  V_topk shape:  {list(V_topk.shape)}")
    print(f"  Saved to:      {args.output_file}")


if __name__ == "__main__":
    main()