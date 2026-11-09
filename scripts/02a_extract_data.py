"""
scripts/02a_extract_data.py

Phase 2 Step 1: Extract and Match Domain Data from BeaverTails.
Isolates prompt-response pairs into:
- Domain A: Financial & Property Crime / Theft (Training Domain candidate pool)
- Domain B: Drugs, Weapons & Banned Substances (Out-of-Domain evaluation pool)

Matches pairs into (prompt, chosen, rejected) triplets where:
- chosen (y_w): safe response (is_safe == True)
- rejected (y_l): unsafe response (is_safe == False)
"""

import os
import json
import argparse
from collections import defaultdict
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and split BeaverTails domain pools.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PKU-Alignment/BeaverTails",
        help="HuggingFace dataset repository",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="330k_train",
        help="Dataset split to parse",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save extracted domain jsonl files",
    )
    parser.add_argument(
        "--domain_a_category",
        type=str,
        default="financial_crime_property_crime_theft",
        help="Category key in BeaverTails for Domain A",
    )
    parser.add_argument(
        "--domain_b_category",
        type=str,
        default="drug_abuse_weapons_banned_substance",
        help="Category key in BeaverTails for Domain B",
    )
    parser.add_argument(
        "--max_pairs_per_domain",
        type=int,
        default=2500,
        help="Maximum matched (chosen, rejected) pairs to extract per domain",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset_name} ({args.dataset_split})...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Buckets to collect safe and unsafe responses per prompt
    grouped_A = defaultdict(dict)
    grouped_B = defaultdict(dict)

    print("Filtering and grouping prompts by domain category...")
    for row in dataset:
        prompt = row["prompt"].strip()
        response = row["response"].strip()
        is_safe = row["is_safe"]
        categories = row["category"]

        # Domain A selection
        if categories.get(args.domain_a_category, False):
            if is_safe and "chosen" not in grouped_A[prompt]:
                grouped_A[prompt]["chosen"] = response
            elif not is_safe and "rejected" not in grouped_A[prompt]:
                grouped_A[prompt]["rejected"] = response

        # Domain B selection (mutually exclusive with Domain A)
        elif categories.get(args.domain_b_category, False):
            if is_safe and "chosen" not in grouped_B[prompt]:
                grouped_B[prompt]["chosen"] = response
            elif not is_safe and "rejected" not in grouped_B[prompt]:
                grouped_B[prompt]["rejected"] = response

    # Format into complete (prompt, chosen, rejected) pairs
    data_A = [
        {
            "prompt": p,
            "chosen": v["chosen"],
            "rejected": v["rejected"],
            "domain": "A",
            "category": args.domain_a_category,
        }
        for p, v in grouped_A.items()
        if "chosen" in v and "rejected" in v
    ]

    data_B = [
        {
            "prompt": p,
            "chosen": v["chosen"],
            "rejected": v["rejected"],
            "domain": "B",
            "category": args.domain_b_category,
        }
        for p, v in grouped_B.items()
        if "chosen" in v and "rejected" in v
    ]

    data_A = data_A[: args.max_pairs_per_domain]
    data_B = data_B[: args.max_pairs_per_domain]

    path_A = os.path.join(args.output_dir, "domain_A_pool.jsonl")
    path_B = os.path.join(args.output_dir, "domain_B_ood.jsonl")

    print(f"Writing {len(data_A)} matched pairs to {path_A}...")
    with open(path_A, "w", encoding="utf-8") as f:
        for item in data_A:
            f.write(json.dumps(item) + "\n")

    print(f"Writing {len(data_B)} matched pairs to {path_B}...")
    with open(path_B, "w", encoding="utf-8") as f:
        for item in data_B:
            f.write(json.dumps(item) + "\n")

    print("Data extraction complete.")


if __name__ == "__main__":
    main()