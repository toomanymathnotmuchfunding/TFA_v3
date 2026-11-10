"""
scripts/02b_profile_b0.py

Computes baseline bias under the frozen merged reference model:
    b0(x) = log pi_ref(y_l | x) - log pi_ref(y_w | x)

Performs greedy state selection on Domain A:
  - Slices the top omega_size prompts with highest b0 into data/omega_train.jsonl
  - Saves the remaining Domain A prompts to data/domain_A_heldout.jsonl
  - Annotates Domain B with b0 and saves to data/domain_B_ood.jsonl
"""

import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Extract spectral subspaces for R_Omega")
    parser.add_argument("--ref_model_path", type=str, required=True, help="Path to reference merged model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset jsonl files")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing profiling results")
    parser.add_argument("--target_layer", type=int, default=16, help="Target transformer layer for representation extraction")
    parser.add_argument("--rank_k", type=int, default=4, help="Rank of spectral decomposition subspace")
    parser.add_argument("--output_file", type=str, default="data/spectral_subspaces.pt", help="Path to save decomposed subspaces")
    return parser.parse_args()


def compute_sequence_logp(model, tokenizer, prompt: str, response: str) -> float:
    prompt_str = (
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    full_str = prompt_str + response + "<|eot_id|>"

    enc_prompt = tokenizer(prompt_str, return_tensors="pt").input_ids.to(model.device)
    enc_full = tokenizer(full_str, return_tensors="pt").input_ids.to(model.device)

    prompt_len = enc_prompt.shape[1]
    labels = enc_full.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(enc_full, labels=labels)
        resp_token_count = (labels != -100).sum().item()
        total_logp = -outputs.loss.item() * resp_token_count

    return total_logp / max(1, resp_token_count)


def score_domain(model, tokenizer, input_path: str, desc: str):
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    scored = []
    for item in tqdm(records, desc=desc):
        logp_w = compute_sequence_logp(model, tokenizer, item["prompt"], item["chosen"])
        logp_l = compute_sequence_logp(model, tokenizer, item["prompt"], item["rejected"])
        b0 = logp_l - logp_w

        row = dict(item)
        row["logp_chosen"] = logp_w
        row["logp_rejected"] = logp_l
        row["b0"] = float(b0)
        scored.append(row)

    return scored


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    path_A = os.path.join(args.data_dir, "domain_A_pool.jsonl")
    path_B = os.path.join(args.data_dir, "domain_B_ood.jsonl")

    if not os.path.exists(path_A) or not os.path.exists(path_B):
        raise FileNotFoundError(f"Missing input pools in {args.data_dir}. Run 02a_extract_data.py first.")

    print(f"Loading reference model: {args.ref_model_path} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    print("Scoring baseline bias b0 on Domain A...")
    scored_A = score_domain(model, tokenizer, path_A, desc="Domain A Profiling")

    print("Scoring baseline bias b0 on Domain B...")
    scored_B = score_domain(model, tokenizer, path_B, desc="Domain B Profiling")

    # Greedy sort on Domain A descending by initial misalignment b0
    scored_A_sorted = sorted(scored_A, key=lambda x: x["b0"], reverse=True)
    omega_train = scored_A_sorted[: args.omega_size]
    domain_A_heldout = scored_A_sorted[args.omega_size :]

    out_omega = os.path.join(args.data_dir, "omega_train.jsonl")
    out_heldout = os.path.join(args.data_dir, "domain_A_heldout.jsonl")
    out_b_ood = os.path.join(args.data_dir, "domain_B_ood.jsonl")

    with open(out_omega, "w", encoding="utf-8") as f:
        for r in omega_train:
            f.write(json.dumps(r) + "\n")

    with open(out_heldout, "w", encoding="utf-8") as f:
        for r in domain_A_heldout:
            f.write(json.dumps(r) + "\n")

    with open(out_b_ood, "w", encoding="utf-8") as f:
        for r in scored_B:
            f.write(json.dumps(r) + "\n")

    # Export distribution diagnostics
    summary = [
        {
            "Split": "Omega (Train, Top b0)",
            "Count": len(omega_train),
            "Mean_b0": pd.Series([x["b0"] for x in omega_train]).mean(),
            "Std_b0": pd.Series([x["b0"] for x in omega_train]).std(),
            "Min_b0": pd.Series([x["b0"] for x in omega_train]).min(),
            "Max_b0": pd.Series([x["b0"] for x in omega_train]).max(),
        },
        {
            "Split": "Domain A Heldout",
            "Count": len(domain_A_heldout),
            "Mean_b0": pd.Series([x["b0"] for x in domain_A_heldout]).mean(),
            "Std_b0": pd.Series([x["b0"] for x in domain_A_heldout]).std(),
            "Min_b0": pd.Series([x["b0"] for x in domain_A_heldout]).min(),
            "Max_b0": pd.Series([x["b0"] for x in domain_A_heldout]).max(),
        },
        {
            "Split": "Domain B OOD",
            "Count": len(scored_B),
            "Mean_b0": pd.Series([x["b0"] for x in scored_B]).mean(),
            "Std_b0": pd.Series([x["b0"] for x in scored_B]).std(),
            "Min_b0": pd.Series([x["b0"] for x in scored_B]).min(),
            "Max_b0": pd.Series([x["b0"] for x in scored_B]).max(),
        },
    ]

    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(args.results_dir, "initial_b0_stats.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved b0 metrics to {summary_path}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()