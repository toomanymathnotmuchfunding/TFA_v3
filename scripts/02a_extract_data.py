"""
scripts/02a_extract_data.py

Phase 2 Step 1: Extract and Match Domain Data from BeaverTails.
- Domain A: Financial & Property Crime / Theft
- Domain B: Drugs, Weapons & Banned Substances
"""

import os
import json
import argparse
import random
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for response sampling.")
    return parser.parse_args()

def _collect_unsafe_by_prompt(dataset, keys):
    """Return prompt -> [unsafe responses] for rows flagged under any of `keys`."""
    unsafe_by_prompt = defaultdict(list)
    for row in dataset:
        if row["is_safe"]:
            continue
        row_cats = row.get("category", {})
        if any(bool(row_cats.get(k, False)) for k in keys):
            unsafe_by_prompt[row["prompt"].strip()].append(row["response"].strip())
    return unsafe_by_prompt


def extract_matched_pairs(dataset, categories_csv, max_pairs, seed=42):
    rng = random.Random(seed)

    # BeaverTails stores the full comma-containing string as ONE category key,
    # e.g. "financial_crime,property_crime,theft". Try exact match first.
    whole_key = categories_csv.strip()
    unsafe_by_prompt = _collect_unsafe_by_prompt(dataset, {whole_key})

    if not unsafe_by_prompt:
        # Fallback: some future schema might split on commas. Try that form too.
        split_keys = {c.strip() for c in categories_csv.split(",") if c.strip()}
        if split_keys != {whole_key}:
            print(f"  Exact key '{whole_key}' found nothing; trying split form {sorted(split_keys)}...")
            unsafe_by_prompt = _collect_unsafe_by_prompt(dataset, split_keys)

    if not unsafe_by_prompt:
        raise RuntimeError(
            f"No rows matched category key '{whole_key}' (or split form). "
            f"Confirm the key exists in dataset['category'].keys()."
        )

    target_prompts = set(unsafe_by_prompt.keys())
    safe_by_prompt = defaultdict(list)
    for row in dataset:
        p = row["prompt"].strip()
        if p in target_prompts and row["is_safe"]:
            safe_by_prompt[p].append(row["response"].strip())

    pairs = []
    for p in sorted(target_prompts):
        if safe_by_prompt[p] and unsafe_by_prompt[p]:
            pairs.append({
                "prompt": p,
                "chosen": rng.choice(safe_by_prompt[p]),
                "rejected": rng.choice(unsafe_by_prompt[p]),
            })
            if len(pairs) >= max_pairs:
                break
    return pairs

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    print(f"\n--- Extracting Domain A ---")
    domain_a_pool = extract_matched_pairs(ds, args.domain_a_category, args.max_pairs_per_domain, seed=args.seed)
    domain_a_prompts = {p["prompt"] for p in domain_a_pool}
    print(f"Extracted {len(domain_a_pool)} candidate pairs for Domain A pool.")
    assert len(domain_a_pool) > 0, "FATAL: Domain A extraction resulted in 0 pairs. Verify category matching."

    print(f"\n--- Extracting Domain B ---")
    domain_b_pairs = extract_matched_pairs(ds, args.domain_b_category, args.max_pairs_per_domain, seed=args.seed)
    domain_b_pairs = [p for p in domain_b_pairs if p["prompt"] not in domain_a_prompts]
    print(f"Extracted {len(domain_b_pairs)} candidate pairs for Domain B.")
    assert len(domain_b_pairs) > 0, "FATAL: Domain B extraction resulted in 0 pairs. Verify category matching or decontamination."

    # 02a produces raw candidate pools; 02b profiles b0 and partitions Omega authoritative splits
    path_a = os.path.join(args.output_dir, "domain_A_pool.jsonl")
    path_b = os.path.join(args.output_dir, "domain_B_ood.jsonl")

    with open(path_a, "w", encoding="utf-8") as f:
        for r in domain_a_pool:
            f.write(json.dumps(r) + "\n")

    with open(path_b, "w", encoding="utf-8") as f:
        for r in domain_b_pairs:
            f.write(json.dumps(r) + "\n")

    print(f"Saved raw domain pools to {path_a} and {path_b}.")


if __name__ == "__main__":
    main()