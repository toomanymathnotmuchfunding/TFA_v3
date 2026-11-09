"""
scripts/01b_merge_sft.py

Phase 1 Step 2: Merge Conversational SFT Adapter into Base Weights.
Permanently combines the LoRA weights trained in 01_conversational_sft.py into 
meta-llama/Llama-3.1-8B to produce the standalone, non-PEFT reference model pi_ref.
This model acts as the fixed origin for baseline bias (b0) and the reference policy 
for subsequent alignment (DPO) runs.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge conversational LoRA into base model.")
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model ID or local path to base model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="checkpoints/llama3_1_8b_sft",
        help="Path to the trained LoRA adapter checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/llama3_1_8b_sft_merged",
        help="Destination directory for the standalone merged reference model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model: {args.base_model_id} in bfloat16 on CPU...")
    # Load on CPU to avoid unnecessary GPU VRAM overhead during merging
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Attaching LoRA adapter from: {args.adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging adapter weights into base model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged reference model to: {args.output_dir}...")
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print(f"Copying and saving tokenizer to: {args.output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Reference model materialization complete.")
    print(f"pi_ref is ready at: {args.output_dir}")


if __name__ == "__main__":
    main()