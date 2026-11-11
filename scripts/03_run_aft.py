"""
scripts/03_run_aft.py

Phase 3: Alignment Fine-Tuning (AFT) via Custom Margin-DPO on Omega.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import inspect
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run AFT Margin-DPO on Omega.")
    parser.add_argument("--ref_model_path", type=str, default="checkpoints/llama3_1_8b_sft_merged")
    parser.add_argument("--data_path", type=str, default="data/omega_train.jsonl")
    parser.add_argument("--directions_path", type=str, default="data/spectral_subspaces.pt")
    parser.add_argument(
        "--arch",
        type=str,
        choices=["frozen_shared", "frozen_diff", "train_shared", "train_diff", "lora4", "fft"],
        required=True,
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["zero", "overcorrect"],
        required=True,
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--target_layer", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


class MarginDPODataCollator(DPODataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "margin" in features[0]:
            batch["margin"] = torch.tensor([f["margin"] for f in features], dtype=torch.float32)
        return batch


class RobustMarginDPOTrainer(DPOTrainer):
    def __init__(self, *args, frozen_a=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_a = frozen_a

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if (n in decay_parameters and p.requires_grad and not (self.frozen_a and "lora_A" in n))
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad and not (self.frozen_a and "lora_A" in n))
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        beta = getattr(self, "beta", self.args.beta)

        margins = getattr(self, "_current_batch_margins", None)
        if margins is None:
            losses = -F.logsigmoid(beta * logits)
        else:
            margins = torch.as_tensor(margins, device=logits.device, dtype=logits.dtype).view_as(logits)
            losses = -F.logsigmoid(beta * logits - margins)

        chosen_rewards = beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = beta * (rejected_logps - ref_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._current_batch_margins = inputs.get("margin", None)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def format_dataset(jsonl_path, target_type, beta, omega):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f]

    formatted = []
    multiplier = 1.0 if target_type == "zero" else (1.0 + omega)

    for item in raw_items:
        assert "b0" in item, "FATAL: 'b0' missing from training sample. Run 02b_profile_b0.py first."
        
        # Provide raw text strings. TRL will handle the tokenization and masks.
        prompt_str = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{item['prompt']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        chosen_str = item["chosen"] + "<|eot_id|>"
        rejected_str = item["rejected"] + "<|eot_id|>"

        b0_val = float(item["b0"])
        margin = beta * multiplier * max(0.0, b0_val)

        formatted.append({
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str,
            "margin": margin,
        })

    return Dataset.from_list(formatted)


def inject_and_configure_lora(model, arch, directions_path, target_layer):
    if arch == "fft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    if arch == "lora4":
        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, peft_config)

    # 1D Subspaces (frozen_shared, frozen_diff, train_shared, train_diff)
    peft_config = LoraConfig(
        r=1,
        lora_alpha=1,
        target_modules=["v_proj"],
        layers_to_transform=[target_layer],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    subspace_data = torch.load(directions_path, map_location="cpu", weights_only=True)
    v_vec = subspace_data["v_shared"] if "shared" in arch else subspace_data["v_diff"]
    v_vec = v_vec.float()

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and str(target_layer) in name:
            target_param = module.lora_A["default"].weight
            with torch.no_grad():
                target_param.data.copy_(v_vec.to(target_param.dtype).unsqueeze(0))

            if "frozen" in arch:
                target_param.requires_grad = False
                print(f"FROZEN enforcement: requires_grad=False on {name}.lora_A")
            else:
                target_param.requires_grad = True
                print(f"TRAINABLE: requires_grad=True on {name}.lora_A")

    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    lr = args.learning_rate
    if lr is None:
        lr = 1e-5 if args.arch == "fft" else 5e-4

    print(f"Preparing Margin-DPO: Arch = {args.arch} | Target = {args.target} (omega = {args.omega})")
    print(f"Learning Rate: {lr} | Output Directory: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = format_dataset(args.data_path, args.target, args.beta, args.omega)
    margins = [x["margin"] for x in train_dataset]
    print(f"Loaded {len(train_dataset)} pairs. Margin stats: Mean={sum(margins)/len(margins):.4f}, Min={min(margins):.4f}, Max={max(margins):.4f}")

    print("Loading base policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.config.use_cache = False

    # Configure structural parameter subspace
    model = inject_and_configure_lora(model, args.arch, args.directions_path, args.target_layer)

    # For FFT, precompute ref logps with a temporary ref model, then free it.
    ref_model = None
    if args.arch == "fft":
        print("Instantiating temporary reference model for precomputation...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")

    optim_name = "adamw_bnb_8bit" if args.arch == "fft" else "adamw_torch"

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="no",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        beta=args.beta,
        max_length=512,
        max_prompt_length=256,
        optim=optim_name,
        precompute_ref_log_probs=True,
    )

    data_collator = MarginDPODataCollator(pad_token_id=tokenizer.pad_token_id)
    is_frozen_arm = "frozen" in args.arch

    # Inspect the *parent* class signature — RobustMarginDPOTrainer.__init__
    # uses (*args, **kwargs), so neither tokenizer nor processing_class appears
    # in its own signature.
    _trainer_fields = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in _trainer_fields:
        _tok_kw = "processing_class"
    elif "tokenizer" in _trainer_fields:
        _tok_kw = "tokenizer"
    else:
        raise RuntimeError(
            "DPOTrainer.__init__ accepts neither 'processing_class' nor 'tokenizer'; "
            "check the installed TRL version."
        )
    trainer = RobustMarginDPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args,
        frozen_a=is_frozen_arm,
        **{_tok_kw: tokenizer},
    )

    print("\n--- Verifying Active Trainable Parameters ---")
    active_b = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            print(f"  {name:60} | requires_grad={param.requires_grad} | shape={list(param.shape)}")
            if is_frozen_arm and "lora_A" in name:
                assert not param.requires_grad, f"FATAL: {name} has requires_grad=True in a frozen-A arm!"
            if "lora_B" in name and param.requires_grad:
                active_b = True

    if args.arch != "fft":
        assert active_b, "FATAL: No active trainable lora_B adapter found!"

    print("Starting alignment fine-tuning...")
    trainer.train()

    print(f"Saving aligned checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("AFT run complete.")


if __name__ == "__main__":
    main()