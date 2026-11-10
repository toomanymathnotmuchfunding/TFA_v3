"""
scripts/03_run_aft.py

Phase 3: Alignment Fine-Tuning (AFT) via Custom Margin-DPO on Omega.
Executes any of the 12 experimental conditions:
  Architectures:
    1. frozen_shared: Rank-1 LoRA, A initialized to v_shared and frozen, B trainable
    2. frozen_diff:   Rank-1 LoRA, A initialized to v_diff and frozen, B trainable
    3. train_shared:  Rank-1 LoRA, A initialized to v_shared, both A and B trainable
    4. train_diff:    Rank-1 LoRA, A initialized to v_diff, both A and B trainable
    5. lora4:         Standard Rank-4 LoRA across projection layers, all trainable
    6. fft:           Full Fine-Tuning, unconstrained parameter update
  Target Policies:
    1. zero:        M(x) = beta * max(0, b0(x))
    2. overcorrect: M(x) = beta * (1 + omega) * max(0, b0(x))  (default omega = 1.0)
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run AFT Margin-DPO on Omega.")
    parser.add_argument(
        "--ref_model_path",
        type=str,
        default="checkpoints/llama3_1_8b_sft_merged",
        help="Path to merged reference model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/omega_train.jsonl",
        help="Path to Omega training data containing cached b0",
    )
    parser.add_argument(
        "--directions_path",
        type=str,
        default="data/spectral_subspaces.pt",
        help="Path to precomputed subspace projection vectors",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["frozen_shared", "frozen_diff", "train_shared", "train_diff", "lora4", "fft"],
        required=True,
        help="Structural bias space architecture",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["zero", "overcorrect"],
        required=True,
        help="Target policy: zero (standard target) or overcorrect",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="Overshoot parameter omega for over-corrected target (M = beta * (1 + omega) * b0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO KL-regularization penalty coefficient",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=16,
        help="Layer index for targeted 1D subspace injection",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs on Omega",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
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
        default=None,
        help="Learning rate (defaults to 5e-4 for LoRA, 1e-5 for FFT)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for aligned model checkpoint",
    )
    return parser.parse_args()


class RobustMarginDPOTrainer(DPOTrainer):
    """
    Subclasses TRL's DPOTrainer to apply dynamic sample-level margin offsets M(x)
    and guarantee frozen-A isolation from the optimizer.
    """
    def __init__(self, *args, frozen_a=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_a = frozen_a

    def create_optimizer(self):
        """
        Guarantees that lora_A parameters are never passed to AdamW/SGD,
        preventing optimizer state allocation or weight updates.
        """
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
        """
        Computes custom margin DPO loss:
          L = -log sigma( beta * (pi_logratios - ref_logratios) - M(x) )
        """
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        # Pull dynamic margin tensor stored during compute_loss / forward pass
        margins = getattr(self, "_current_batch_margins", None)
        if margins is None:
            # Fallback to standard DPO loss if no custom margin is present
            losses = -F.logsigmoid(self.beta * logits)
        else:
            margins = torch.as_tensor(margins, device=logits.device, dtype=logits.dtype).view_as(logits)
            losses = -F.logsigmoid(self.beta * logits - margins)

        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Cache batch margin tensor for dpo_loss calculation
        if "margin" in inputs:
            self._current_batch_margins = inputs["margin"]
        else:
            self._current_batch_margins = None

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
        prompt_str = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{item['prompt']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        chosen_str = item["chosen"] + "<|eot_id|>"
        rejected_str = item["rejected"] + "<|eot_id|>"

        b0_val = float(item.get("b0", 0.0))
        # Dynamic margin: M(x) = beta * multiplier * max(0, b0(x))
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

    subspace_data = torch.load(directions_path, map_location="cpu")
    v_vec = subspace_data["v_shared"] if "shared" in arch else subspace_data["v_diff"]
    v_vec = v_vec.float()

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and str(target_layer) in name:
            with torch.no_grad():
                module.lora_A["default"].weight.copy_(v_vec.unsqueeze(0))

            if "frozen" in arch:
                module.lora_A["default"].weight.requires_grad = False
                def make_zero_hook():
                    return lambda grad: torch.zeros_like(grad)
                module.lora_A["default"].weight.register_hook(make_zero_hook())
                print(f"FROZEN enforcement: Hook registered & requires_grad=False on {name}.lora_A")
            else:
                module.lora_A["default"].weight.requires_grad = True
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
    print(f"Loading reference model from: {args.ref_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = format_dataset(args.data_path, args.target, args.beta, args.omega)
    print(f"Constructed {len(train_dataset)} training pairs with dynamic margins.")

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        beta=args.beta,
        max_length=768,
        max_prompt_length=384,
        model_init_kwargs=None,
    )

    is_frozen_arm = "frozen" in args.arch
    trainer = RobustMarginDPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_args,
        frozen_a=is_frozen_arm,
    )

    print("\n--- Verifying Active Trainable Parameters ---")
    for name, param in model.named_parameters():
        if "lora_" in name:
            print(f"  {name:60} | requires_grad={param.requires_grad} | shape={list(param.shape)}")
            if is_frozen_arm and "lora_A" in name:
                assert not param.requires_grad, f"FATAL: {name} has requires_grad=True in a frozen-A arm!"

    print("Starting alignment fine-tuning...")
    trainer.train()

    print(f"Saving aligned checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("AFT run complete.")


if __name__ == "__main__":
    main()