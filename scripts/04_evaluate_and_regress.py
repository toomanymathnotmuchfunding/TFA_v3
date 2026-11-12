"""
scripts/04_evaluate_and_regress.py
"""

import os
import glob
import json
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import linregress
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and run regression analysis.")
    parser.add_argument("--ref_model_path", type=str, default="checkpoints/llama3_1_8b_sft_merged")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/aft_runs")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_filename", type=str, default="final_loss_regression_table.csv")
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
    # MEAN per-token log-prob (must match 02b_profile_b0.py)
    return -outputs.loss.item()


def evaluate_split(model, tokenizer, split_data, desc="Evaluating"):
    b0_vals, b1_vals = [], []

    for item in tqdm(split_data, desc=desc, leave=False):
        prompt, chosen, rejected = item["prompt"], item["chosen"], item["rejected"]
        logp_w = compute_sequence_logp(model, tokenizer, prompt, chosen)
        logp_l = compute_sequence_logp(model, tokenizer, prompt, rejected)
        b1_vals.append(logp_l - logp_w)
        b0_vals.append(float(item["b0"]) if "b0" in item and item["b0"] is not None else np.nan)

    b1_arr = np.array(b1_vals, dtype=np.float64)
    b0_arr = np.array(b0_vals, dtype=np.float64)

    loss = float(np.mean(b1_arr ** 2))
    mean_b1, std_b1 = float(np.mean(b1_arr)), float(np.std(b1_arr))

    valid_mask = ~np.isnan(b0_arr)
    n_valid = int(valid_mask.sum())
    mean_b0 = float(np.nanmean(b0_arr))
    std_b0 = float(np.nanstd(b0_arr))

    if n_valid < max(10, len(split_data) // 2):
        raise RuntimeError(f"{desc}: only {n_valid}/{len(split_data)} samples have b0.")

    if np.var(b0_arr[valid_mask]) > 1e-8:
        slope, _, r_val, _, _ = linregress(b0_arr[valid_mask], b1_arr[valid_mask])
        r2 = float(r_val ** 2)
    else:
        slope, r2 = 0.0, 0.0

    return {
        "loss": loss,
        "mean_b1": mean_b1,
        "std_b1": std_b1,
        "mean_b0": mean_b0,
        "std_b0": std_b0,
        "slope": float(slope),
        "r2": r2,
        "n_b0_valid": n_valid,
        "n_total": len(split_data),
    }, b0_arr, b1_arr


def load_model_checkpoint(checkpoint_path, ref_model_path):
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        base = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
    model.eval()
    return model


def generate_plots(df, output_dir, scatter_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    # ---- Plot 1: b_0 distribution across splits (log-scale x) ----
    # This shows WHY Omega is the "hard" training set.
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    for label, col, color in [
        ("Omega (Train, Top b0)", "mean_b0_Omega", "#d62728"),
        ("Domain A Heldout", "mean_b0_Domain_A_Heldout", "#1f77b4"),
        ("Domain B OOD", "mean_b0_Domain_B_OOD", "#2ca02c"),
    ]:
        if col in df.columns:
            vals = df[col].dropna().values
            ax.axvline(vals.mean(), color=color, linewidth=2, label=label, linestyle="--")
    plt.axvline(0.0, color="black", linewidth=1, linestyle=":")
    plt.xlabel("Mean per-token b0 (reference bias)")
    plt.title("Baseline b0 by split — Omega is deliberately the misaligned tail")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "b0_distribution.png"), dpi=300)
    plt.close()

    # ---- Plot 2: Loss across splits (log scale y) ----
    plt.figure(figsize=(14, 6))
    df_melted = df.melt(
        id_vars=["run_name"],
        value_vars=["loss_Omega", "loss_Domain_A_Heldout", "loss_Domain_B_OOD"],
        var_name="Split",
        value_name="Loss",
    )
    df_melted["Split"] = df_melted["Split"].map({
        "loss_Omega": "Omega (Train)",
        "loss_Domain_A_Heldout": "Domain A (Heldout)",
        "loss_Domain_B_OOD": "Domain B (OOD)",
    })
    sns.barplot(data=df_melted, x="run_name", y="Loss", hue="Split", palette="viridis")
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.title("Quadratic Loss across arms and splits (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_comparison.png"), dpi=300)
    plt.close()

    # ---- Plot 3: OOD shrinkage slopes ----
    plt.figure(figsize=(14, 5))
    sns.barplot(data=df, x="run_name", y="slope_Domain_B_OOD", color="steelblue")
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="No shrinkage (slope=1)")
    plt.axhline(0.0, color="green", linestyle=":", linewidth=1.5, label="Full shrinkage (slope=0)")
    plt.xticks(rotation=45, ha="right")
    plt.title("OOD shrinkage: slope of b1 ~ b0 on Domain B")
    plt.ylabel("Slope (closer to 0 = more shrinkage)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ood_slope.png"), dpi=300)
    plt.close()

    # ---- Plot 4: Per-arm b0 vs b1 scatter on OOD ----
    # This is the "money plot" — visually shows whether the arm shrinks, rotates, or ignores OOD.
    available = sorted(glob.glob(os.path.join(scatter_dir, "*_Domain_B_OOD.npz")))
    if available:
        n_arms = len(available)
        n_cols = min(4, n_arms)
        n_rows = (n_arms + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

        # Global bounds for consistent axes
        all_b0, all_b1 = [], []
        for path in available:
            d = np.load(path)
            mask = ~np.isnan(d["b0"])
            all_b0.append(d["b0"][mask])
            all_b1.append(d["b1"][mask])
        all_b0 = np.concatenate(all_b0)
        all_b1 = np.concatenate(all_b1)
        lo = float(min(all_b0.min(), all_b1.min())) - 0.5
        hi = float(max(all_b0.max(), all_b1.max())) + 0.5

        for i, path in enumerate(available):
            ax = axes[i // n_cols, i % n_cols]
            name = os.path.basename(path).replace("_Domain_B_OOD.npz", "")
            d = np.load(path)
            mask = ~np.isnan(d["b0"])
            ax.scatter(d["b0"][mask], d["b1"][mask], alpha=0.35, s=10, color="steelblue")
            # Identity line (no change) and flat line (full shrinkage)
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="No change")
            ax.axhline(0.0, color="green", linestyle=":", linewidth=1, label="Full shrinkage")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("b0 (reference)")
            ax.set_ylabel("b1 (trained)")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)

        # Hide unused subplots
        for j in range(n_arms, n_rows * n_cols):
            axes[j // n_cols, j % n_cols].axis("off")

        fig.suptitle("OOD: b1 vs b0 per arm — where does the trained model land?", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ood_b0_vs_b1.png"), dpi=300)
        plt.close()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    scatter_dir = os.path.join(args.results_dir, "scatter_data")
    os.makedirs(scatter_dir, exist_ok=True)

    splits = {
        "Omega": os.path.join(args.data_dir, "omega_train.jsonl"),
        "Domain_A_Heldout": os.path.join(args.data_dir, "domain_A_heldout.jsonl"),
        "Domain_B_OOD": os.path.join(args.data_dir, "domain_B_ood.jsonl"),
    }
    loaded_splits = {}
    for name, path in splits.items():
        with open(path, "r", encoding="utf-8") as f:
            loaded_splits[name] = [json.loads(line) for line in f]

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint_folders = sorted(
        d for d in glob.glob(os.path.join(args.checkpoints_dir, "*")) if os.path.isdir(d)
    )
    records = []

    for ckp_dir in checkpoint_folders:
        run_name = os.path.basename(ckp_dir)
        print(f"\n=== Evaluating Arm: {run_name} ===")

        model = load_model_checkpoint(ckp_dir, args.ref_model_path)
        row = {"run_name": run_name}

        for split_name, split_data in loaded_splits.items():
            metrics, b0_arr, b1_arr = evaluate_split(
                model, tokenizer, split_data, desc=f"{run_name} [{split_name}]"
            )
            for k, v in metrics.items():
                row[f"{k}_{split_name}"] = v

            # Save per-sample (b0, b1) for later scatter plots
            np.savez(
                os.path.join(scatter_dir, f"{run_name}_{split_name}.npz"),
                b0=b0_arr,
                b1=b1_arr,
            )

        records.append(row)
        del model
        torch.cuda.empty_cache()

    df_results = pd.DataFrame(records)
    out_csv = os.path.join(args.results_dir, args.output_filename)
    df_results.to_csv(out_csv, index=False)

    print(f"\nSaved complete table to {out_csv}")
    print("Generating plots...")
    generate_plots(df_results, os.path.join(args.results_dir, "plots"), scatter_dir)
    print(f"Plots saved to {args.results_dir}/plots/")
    print(f"Per-sample scatter data saved to {scatter_dir}/")


if __name__ == "__main__":
    main()