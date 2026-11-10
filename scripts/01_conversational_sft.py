"""
scripts/01_conversational_sft.py

Phase 1: Conversational Supervised Fine-Tuning (SFT)
Trains a standard LoRA adapter on meta-llama/Llama-3.1-8B using a benign 
conversational instruction corpus (ultrachat_200k).
Establishes stable chat-template adherence without imparting safety/refusal bias.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer
from peft import LoraConfig
import inspect
from trl import SFTConfig   # subclass of TrainingArguments in TRL ≥ 0.12


def parse_args():
    parser = argparse.ArgumentParser(description="Conversational SFT for Base Llama-3.1-8B")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model checkpoint identifier",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="HuggingFace instruction dataset",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train_sft[:20000]",
        help="Slice of dataset to train on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/llama3_1_8b_sft",
        help="Destination directory for trained SFT adapter",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence token length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW peak learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Fallback: Llama-3 base models lack chat_template in tokenizer_config.json
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_split})...")
    raw_dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("Applying chat template formatting...")
    formatted_dataset = raw_dataset.map(
        format_chat,
        remove_columns=raw_dataset.column_names,
        desc="Formatting chat turns",
    )

    print(f"Loading base model: {args.model_id} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # Standard conversational adapter: r=16 across all projection & MLP layers
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -------- Version-adaptive SFT config --------
    config_fields = inspect.signature(SFTConfig.__init__).parameters
    config_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.num_train_epochs,
        logging_steps=25,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    # `max_length` (new) vs `max_seq_length` (old)
    if "max_length" in config_fields:
        config_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in config_fields:
        config_kwargs["max_seq_length"] = args.max_seq_length
    if "dataset_text_field" in config_fields:
        config_kwargs["dataset_text_field"] = "text"

    training_args = SFTConfig(**config_kwargs)

    # -------- Version-adaptive SFTTrainer kwargs --------
    trainer_fields = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = dict(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        args=training_args,
    )
    # `processing_class` (new) vs `tokenizer` (old)
    if "processing_class" in trainer_fields:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    # Old TRL still wants these on the trainer, not the config
    if "max_seq_length" in trainer_fields and "max_length" not in config_kwargs:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    if "dataset_text_field" in trainer_fields and "dataset_text_field" not in config_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(**trainer_kwargs)

    print("Beginning conversational SFT...")
    trainer.train()

    print(f"Saving SFT adapter checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Conversational SFT complete.")


if __name__ == "__main__":
    main()