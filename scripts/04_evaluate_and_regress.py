"""
scripts/04_evaluate_and_regress.py

Phase 4: Evaluation and Statistical Regression Analysis across Experimental Arms.
Evaluates the 12 fine-tuned model checkpoints across three evaluation splits:
  1. Omega (In-distribution training prompts, N=100)
  2. Domain A Held-out (In-distribution test prompts, N~900)
  3. Domain B OOD (Out-of-distribution test prompts, N~1000)

Calculates:
  - Post-training bias: b1(x) = log pi_theta(y_l | x) - log pi_theta(y_w | x)
  - Principal empirical quadratic loss: L = (1 / |S|) * sum_{x in S} (b1(x))^2
  - OLS regression slope: b1(x) = alpha + beta_slope * b0(x) (measuring shrinkage/rotation)

Exports results to a structured CSV file in results/.
"""

import os
import glob
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and run regression analysis.")
    parser.add_argument(
        "--ref_model_path",
        type=str,
        default="checkpoints/llama3_1_8b_sft_merged",
        help="Path to merged reference model checkpoint",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints/aft_runs",
        help="Root directory containing the trained AFT checkpoint folders",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the split JSONL files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save the final regression table",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="final_loss_regression_table.csv",
        help="Filename for the exported CSV summary table",
    )
    return parser.parse_args()


@torch.inference_mode()
def compute_sequence_logp(model, tokenizer, prompt, response):
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

    outputs = model(enc_full, labels=labels)
    resp_token_count = (labels != -100).sum().item()
    
    # Total unnormalized sequence log-probability
    total_logp = -outputs.loss.item() * resp_token_count
    return total_logp


def evaluate_split(model, tokenizer, split_data, desc="Evaluating"):
    b0_vals = []
    b1_vals = []

    for item in tqdm(split_data, desc=desc, leave=False):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        logp_w = compute_sequence_logp(model, tokenizer, prompt, chosen)
        logp_l = compute_sequence_logp(model, tokenizer, prompt, rejected)
        b1 = logp_l - logp_w

        b1_vals.append(b1)
        
        # Safely extract b0 if pre-annotated
        if "b0" in item and item["b0"] is not None:
            b0_vals.append(float(item["b0"]))
        else:
            b0_vals.append(np.nan)

    b1_arr = np.array(b1_vals, dtype=np.float64)
    b0_arr = np.array(b0_vals, dtype=np.float64)

    # Quadratic principal loss
    loss = float(np.mean(b1_arr ** 2))
    mean_b1 = float(np.mean(b1_arr))
    std_b1 = float(np.std(b1_arr))

    # Linear regression slope: b1 ~ alpha + slope * b0 (only on valid pairs)
    valid_mask = ~np.isnan(b0_arr)
    if valid_mask.sum() > 2 and np.var(b0_arr[valid_mask]) > 1e-8:
        slope, intercept, r_val, p_val, std_err = linregress(b0_arr[valid_mask], b1_arr[valid_mask])
        r2 = float(r_val ** 2)
    else:
        slope, r2 = 0.0, 0.0

    return {
        "loss": loss,
        "mean_b1": mean_b1,
        "std_b1": std_b1,
        "slope": float(slope),
        "r2": r2,
    }


def load_model_checkpoint(checkpoint_path, ref_model_path):
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        base = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    splits = {
        "Omega": os.path.join(args.data_dir, "omega_train.jsonl"),
        "Domain_A_Heldout": os.path.join(args.data_dir, "domain_A_heldout.jsonl"),
        "Domain_B_OOD": os.path.join(args.data_dir, "domain_B_ood.jsonl"),
    }

    loaded_splits = {}
    for name, path in splits.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required data split at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            loaded_splits[name] = [json.loads(line) for line in f]

    print(f"Loading tokenizer from: {args.ref_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint_folders = sorted(
        [
            d for d in glob.glob(os.path.join(args.checkpoints_dir, "*"))
            if os.path.isdir(d) and not os.path.basename(d).startswith(".")
        ]
    )

    if not checkpoint_folders:
        raise FileNotFoundError(f"No checkpoint subdirectories found in: {args.checkpoints_dir}")

    print(f"Discovered {len(checkpoint_folders)} checkpoint directories for evaluation.")
    records = []

    for ckp_dir in checkpoint_folders:
        run_name = os.path.basename(ckp_dir)
        print(f"\n==========================================")
        print(f"Evaluating Arm: {run_name}")
        print(f"==========================================")

        model = load_model_checkpoint(ckp_dir, args.ref_model_path)
        row = {"run_name": run_name}

        for split_name, split_data in loaded_splits.items():
            metrics = evaluate_split(model, tokenizer, split_data, desc=f"{run_name} [{split_name}]")
            row[f"L_{split_name}"] = metrics["loss"]
            row[f"mean_b1_{split_name}"] = metrics["mean_b1"]
            row[f"std_b1_{split_name}"] = metrics["std_b1"]
            row[f"slope_{split_name}"] = metrics["slope"]
            row[f"r2_{split_name}"] = metrics["r2"]

        records.append(row)

        del model
        torch.cuda.empty_cache()

    df_results = pd.DataFrame(records)
    out_csv = os.path.join(args.results_dir, args.output_filename)
    df_results.to_csv(out_csv, index=False)

    print("\n\n" + "=" * 90)
    print("FINAL EXPERIMENTAL LOSS & GENERALIZATION REGRESSION TABLE")
    print("=" * 90)
    display_cols = ["run_name", "L_Omega", "L_Domain_A_Heldout", "L_Domain_B_OOD", "slope_Domain_B_OOD"]
    print(df_results[display_cols].to_string(index=False))
    print("=" * 90)
    print(f"Saved complete table to {out_csv}")


if __name__ == "__main__":
    main()