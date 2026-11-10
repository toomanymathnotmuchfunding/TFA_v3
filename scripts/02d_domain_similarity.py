"""
scripts/02d_domain_similarity.py

Optional cross-domain analysis of the Δh subspaces. Loads cached H_A/H_B
(data/hidden_states.pt, produced by 02c if patched, or by this script on
first run). Reports three views of "how similar are Domain A and Domain B":

  1. Cross-cosine matrix C[i,j] = cos(V_A[i], V_B[j]) between the two
     domains' top-k principal directions (k×k).
  2. Variance explained: fraction of one domain's total Δh variance captured
     by the other domain's top-k directions.
  3. Mean projections: mu_A·v and mu_B·v for v in {V_topk, v_shared, v_diff}.
     Positive on both → shared shift. Opposite signs → direction discriminates.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_model_path", default="checkpoints/llama3_1_8b_sft_merged")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--target_layer", type=int, default=16)
    p.add_argument("--rank_k", type=int, default=4)
    p.add_argument("--omega_size", type=int, default=100)
    p.add_argument("--cache_file", default="data/hidden_states.pt")
    return p.parse_args()


class Hook:
    def __init__(self, model, layer_idx):
        self.h = None
        self.handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(
            lambda m, i, o: setattr(self, "h", i[0].detach())
        )
    def remove(self):
        self.handle.remove()


@torch.inference_mode()
def delta_h_one(model, tok, hook, prompt, chosen, rejected):
    def rep(resp):
        pre = (f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
               f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        plen = tok(pre, return_tensors="pt").input_ids.shape[1]
        full = tok(pre + resp + "<|eot_id|>", return_tensors="pt").input_ids.to(model.device)
        model(full)
        h = hook.h[:, plen:, :]
        if h.shape[1] == 0:
            h = hook.h[:, -1:, :]
        return h.mean(dim=1).squeeze(0).cpu().float()
    return rep(rejected) - rep(chosen)


def load_jsonl(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main():
    args = parse_args()

    if os.path.exists(args.cache_file):
        c = torch.load(args.cache_file, weights_only=True)
        H_A, H_B = c["H_A"], c["H_B"]
        print(f"Loaded cached H_A={tuple(H_A.shape)}, H_B={tuple(H_B.shape)}")
    else:
        print(f"No cache at {args.cache_file}; extracting Δh (one-time ~15 min)...")
        omega = load_jsonl(os.path.join(args.data_dir, "omega_train.jsonl"), limit=args.omega_size)
        dom_b = load_jsonl(os.path.join(args.data_dir, "domain_B_ood.jsonl"), limit=len(omega))
        tok = AutoTokenizer.from_pretrained(args.ref_model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="cuda",
        )
        model.eval()
        hook = Hook(model, args.target_layer)
        H_A = torch.stack([delta_h_one(model, tok, hook, it["prompt"], it["chosen"], it["rejected"])
                           for it in tqdm(omega, desc="H_A")]).float()
        H_B = torch.stack([delta_h_one(model, tok, hook, it["prompt"], it["chosen"], it["rejected"])
                           for it in tqdm(dom_b, desc="H_B")]).float()
        hook.remove()
        torch.save({"H_A": H_A, "H_B": H_B, "target_layer": args.target_layer}, args.cache_file)
        print(f"Cached to {args.cache_file}")

    k = args.rank_k
    _, _, Vh_A = torch.linalg.svd(H_A, full_matrices=False)
    _, _, Vh_B = torch.linalg.svd(H_B, full_matrices=False)

    # ---------- 1. Cross-cosine matrix ----------
    print(f"\n=== 1. Cross-cosine: A's top-{k} principal axes vs B's top-{k} ===")
    C = (Vh_A[:k] @ Vh_B[:k].T).cpu()
    print("        " + "".join(f"  B[{j}]  " for j in range(k)))
    for i in range(k):
        print(f"A[{i}]   " + "".join(f" {C[i, j].item():+7.4f} " for j in range(k)))
    print("  |C[i,j]| ≈ 1 → A's i-th axis aligns with B's j-th (sign: flip → -1)")

    # ---------- 2. Variance explained ----------
    print(f"\n=== 2. Variance explained (energy overlap) ===")
    tot_A = (H_A ** 2).sum().item()
    tot_B = (H_B ** 2).sum().item()
    for i in range(k):
        v = Vh_A[i]
        print(f"  A[{i}] dir:   B-data captures {( (H_B @ v).pow(2).sum().item() / tot_B):6.2%}"
              f" of B's total  |  self: {( (H_A @ v).pow(2).sum().item() / tot_A):6.2%}")
    for j in range(k):
        v = Vh_B[j]
        print(f"  B[{j}] dir:   A-data captures {( (H_A @ v).pow(2).sum().item() / tot_A):6.2%}"
              f" of A's total")

    # ---------- 3. Mean projections ----------
    print(f"\n=== 3. Mean Δh projections ===")
    mu_A = H_A.mean(dim=0)
    mu_B = H_B.mean(dim=0)
    nA, nB = mu_A.norm().item(), mu_B.norm().item()
    print(f"  ‖mu_A‖ = {nA:.3f}   ‖mu_B‖ = {nB:.3f}")
    print(f"  {'direction':<12} {'mu_A·v':>10} {'mu_B·v':>10} {'A-frac':>8} {'B-frac':>8}")

    def row(label, v):
        pa = (mu_A @ v).item()
        pb = (mu_B @ v).item()
        print(f"  {label:<12} {pa:>10.3f} {pb:>10.3f} "
              f"{pa / nA:>8.2%} {pb / nB:>8.2%}")

    for i in range(k):
        row(f"V_topk[{i}]", Vh_A[i])

    v_shared = Vh_A[0] / Vh_A[0].norm()
    v_diff_raw = mu_A - mu_B
    v_diff = v_diff_raw - (v_diff_raw @ v_shared) * v_shared
    v_diff = v_diff / v_diff.norm()
    row("v_diff", v_diff)
    print("  A-frac = mu·v / ‖mu‖ → fraction of each domain's mean magnitude on this axis")
    print("  Same sign on both → shared shift. Opposite sign → discriminates domains.")


if __name__ == "__main__":
    main()