"""
scripts/02a_extract_data.py

Phase 2 Step 1: Extract and Match Domain Data from BeaverTails.
- Domain A: Financial & Property Crime / Theft
- Domain B: Drugs, Weapons & Banned Substances

Constructs (prompt, chosen, rejected) triplets:
- chosen (y_w): safe response (is_safe == True)
- rejected (y_l): unsafe response under target category (is_safe == False)
"""

import os
import json
import argparse
from collections import defaultdict
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and split BeaverTails domain pools.")
    parser.add_argument("--dataset_name", type=str, default="PKU-Alignment/BeaverTails")
    parser.add_argument("--dataset_split", type=str, default="330k_train")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--domain_a_category",
        type=str,
        default="financial_crime,property_crime,theft",
    )
    parser.add_argument(
        "--domain_b_category",
        type=str,
        default="drug_abuse,weapons,banned_substance",
    )
    parser.add_argument("--max_pairs_per_domain", type=int, default=2500)
    parser.add_argument("--omega_size", type=int, default=100)
    return parser.parse_args()


def extract_matched_pairs(dataset, target_category, max_pairs):
    print(f"Pass 1: Identifying unsafe completions for '{target_category}'...")
    unsafe_by_prompt = defaultdict(list)
    for row in dataset:
        if not row["is_safe"] and row.get("category", {}).get(target_category, False):
            p = row["prompt"].strip()
            unsafe_by_prompt[p].append(row["response"].strip())

    target_prompts = set(unsafe_by_prompt.keys())
    print(f"Found {len(target_prompts)} unique candidate prompts. Pass 2: Finding safe completions...")

    safe_by_prompt = defaultdict(list)
    for row in dataset:
        p = row["prompt"].strip()
        if p in target_prompts and row["is_safe"]:
            safe_by_prompt[p].append(row["response"].strip())

    pairs = []
    for p in target_prompts:
        if safe_by_prompt[p] and unsafe_by_prompt[p]:
            pairs.append({
                "prompt": p,
                "chosen": safe_by_prompt[p][0],
                "rejected": unsafe_by_prompt[p][0]
            })
            if len(pairs) >= max_pairs:
                break

    return pairs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    print(f"\n--- Extracting Domain A ('{args.domain_a_category}') ---")
    domain_a_pairs = extract_matched_pairs(ds, args.domain_a_category, args.max_pairs_per_domain)
    print(f"Extracted {len(domain_a_pairs)} paired triplets for Domain A.")

    print(f"\n--- Extracting Domain B ('{args.domain_b_category}') ---")
    domain_b_pairs = extract_matched_pairs(ds, args.domain_b_category, args.max_pairs_per_domain)
    print(f"Extracted {len(domain_b_pairs)} paired triplets for Domain B.")

    # Partition Domain A: Omega candidate train set, pool, and held-out set
    omega_train = domain_a_pairs[:args.omega_size]
    domain_a_pool = domain_a_pairs[args.omega_size:]
    domain_a_heldout = domain_a_pool[:len(domain_a_pool) // 2]

    def write_jsonl(filename, data):
        path = os.path.join(args.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} items to {path}")

    write_jsonl("omega_train.jsonl", omega_train)
    write_jsonl("domain_A_pool.jsonl", domain_a_pool)
    write_jsonl("domain_A_heldout.jsonl", domain_a_heldout)
    write_jsonl("domain_B_ood.jsonl", domain_b_pairs)


if __name__ == "__main__":
    main()