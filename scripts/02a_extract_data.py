"""
scripts/02a_extract_data.py

Extract and Match Domain Data from BeaverTails into (prompt, chosen, rejected) triplets.
- Domain A: financial_crime,property_crime,theft
- Domain B: drug_abuse,weapons,banned_substance
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


def extract_pairs(dataset, target_category, max_pairs):
    # Group responses by prompt
    prompt_groups = defaultdict(lambda: {"safe": [], "unsafe": []})
    
    for row in dataset:
        cat_dict = row.get("category", {})
        # Check if the row belongs to the target category
        if cat_dict.get(target_category, False):
            p = row["prompt"].strip()
            resp = row["response"].strip()
            if row["is_safe"]:
                prompt_groups[p]["safe"].append(resp)
            else:
                prompt_groups[p]["unsafe"].append(resp)

    # Form (prompt, chosen, rejected) pairs
    pairs = []
    for p, resps in prompt_groups.items():
        if resps["safe"] and resps["unsafe"]:
            # Pick first safe as chosen and first unsafe as rejected
            pairs.append({
                "prompt": p,
                "chosen": resps["safe"][0],
                "rejected": resps["unsafe"][0]
            })
            if len(pairs) >= max_pairs:
                break
                
    return pairs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    print(f"Extracting Domain A pairs ('{args.domain_a_category}')...")
    domain_a_pairs = extract_pairs(ds, args.domain_a_category, args.max_pairs_per_domain)
    print(f"Extracted {len(domain_a_pairs)} pairs for Domain A.")

    print(f"Extracting Domain B pairs ('{args.domain_b_category}')...")
    domain_b_pairs = extract_pairs(ds, args.domain_b_category, args.max_pairs_per_domain)
    print(f"Extracted {len(domain_b_pairs)} pairs for Domain B.")

    # Split Domain A into pool and held-out evaluation set
    # Reserving the first omega_size for training candidate selection
    omega_train = domain_a_pairs[:args.omega_size]
    domain_a_pool = domain_a_pairs[args.omega_size:]
    domain_a_heldout = domain_a_pool[:len(domain_a_pool) // 2]

    # Save to data directory
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