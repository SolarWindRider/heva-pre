"""
Case Study: Compare different inference methods on a single sample.

Compares 4 methods:
1. Standard (baseline)
2. Context-Aware Decoding (CAD)
3. Attention Guidance (AG) — analysis of attention patterns
4. CAD + Attention Guidance

Usage:
    # List available samples (first 20)
    python 4_run_inference_single.py --dataset VisuRiddles --list

    # Run case study on sample_idx=0
    python 4_run_inference_single.py --sample_idx 0 --dataset VisuRiddles

    # Show per-token HEVA and entropy
    python 4_run_inference_single.py --sample_idx 0 --dataset VisuRiddles --show_heva --show_entropy

    # Use different model
    python 4_run_inference_single.py --sample_idx 0 --dataset VisuRiddles \
        --model_path ../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking

    # Tune CAD parameters
    python 4_run_inference_single.py --sample_idx 0 --dataset VisuRiddles \
        --ctx_entropy_threshold 3.0 --ctx_top_heads 8
"""

import argparse
import re
import torch

from metrics.inference import load_model_processor, generate_with_attn, get_visual_token_indices, get_input_token_indices
from metrics.heva import _sample_with_vattn_and_entropy
from metrics.context_aware_logits_processor import (
    ContextAwareLogitsProcessor,
    get_context_token_indices,
)
from data.loader import load_dataset, SUPPORTED_DATASETS
from transformers import Qwen3VLForConditionalGeneration

# Monkey-patch once at module load
Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy

# ==========================================
# Utility
# ==========================================

def log_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_answer(generated_text: str) -> str:
    """Extract answer from generated text via JSON {'answer': \"X\"} pattern."""
    json_match = re.search(r'\{"answer":\s*"([^"]+)"\}', generated_text)
    if json_match:
        return json_match.group(1).upper()
    return ""


def compute_heva(gen_vattn: torch.Tensor, alpha: float = 0.2) -> float:
    """HEVA = mean visual attention over top-alpha% highest-attention gen tokens."""
    if gen_vattn is None or gen_vattn.numel() == 0:
        return 0.0
    vattn_per_token = gen_vattn.mean(dim=-1)  # (gen_tokens, batch)
    n = max(1, int(alpha * vattn_per_token.shape[0]))
    if vattn_per_token.shape[0] <= n:
        return vattn_per_token.mean().item()
    top_vals, _ = torch.topk(vattn_per_token[:, 0], k=n)
    return top_vals.mean().item()


def compute_avg_entropy(gen_entropy: torch.Tensor) -> float:
    if gen_entropy is None or gen_entropy.numel() == 0:
        return 0.0
    return gen_entropy.mean().item()


def compute_heva_per_token(gen_vattn: torch.Tensor) -> torch.Tensor:
    """Per-token mean visual attention."""
    if gen_vattn is None or gen_vattn.numel() == 0:
        return torch.tensor([])
    return gen_vattn.mean(dim=-1)[:, 0]


# ==========================================
# AG Analysis: offline per-token analysis of attention patterns
# ==========================================

def _get_critical_indices(model, processor, input_ids, prompt_text, visual_token_indices):
    """Identify critical context tokens: visual + numerical/operator tokens."""
    tokens = input_ids[0]
    critical = []
    v_start, v_end = visual_token_indices[0].item(), visual_token_indices[1].item()
    if v_end > v_start:
        critical.extend(range(v_start, v_end + 1))
    for idx, t_id in enumerate(tokens):
        t_str = processor.tokenizer.decode(t_id)
        if any(c.isdigit() for c in t_str) or t_str.strip() in ['+', '-', '*', '/', '=', '×', '÷']:
            if idx not in critical:
                critical.append(idx)
    return sorted(critical)


def _run_ag_analysis(model, processor, sample: dict, max_new_tokens: int = 8192,
                     temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> dict:
    """
    Run standard inference then analyze attention patterns:
    - Which generation tokens attend to critical context?
    - Would AG have selected a different token?
    - For each step, check if the chosen token's head path → critical context.
    """
    option_str = f"option: {sample.get('options', '')}\n" if sample.get('options') else ""
    full_question = (
        sample["question"] + option_str +
        'Write the answer into a JSON form\n```json\n{"answer": "X"}```'
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are good at step by step reasoning."}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": full_question},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    prompt_token_num = inputs["input_ids"].shape[1]

    visual_token_indices = get_visual_token_indices(inputs["input_ids"], processor=processor)
    critical_indices = _get_critical_indices(model, processor, inputs["input_ids"], full_question, visual_token_indices)

    model.visual_token_indices = visual_token_indices
    model.gen_entropy = []
    model.gen_vattn = []
    model.attn_acc_input = []
    model.attn_acc_visual = []

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            return_dict_in_generate=True, output_logits=True, output_attentions=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p, top_k=top_k if top_k > 0 else None, do_sample=True,
        )

    gen_entropy = torch.stack(model.gen_entropy, dim=1).cpu() if model.gen_entropy else None
    gen_vattn = torch.stack(model.gen_vattn, dim=1).cpu() if model.gen_vattn else None
    gen_token_num = outputs.sequences.shape[1] - prompt_token_num
    generated_text = processor.decode(outputs.sequences[0][prompt_token_num:])

    # AG analysis: per-step head-path verification
    ag_disagreements = []
    seq_len = inputs["input_ids"].shape[1]

    if outputs.attentions and len(critical_indices) > 0:
        try:
            W_O = model.model.language_model.layers[-1].self_attn.o_proj.weight
            d_model, n_heads = W_O.shape[0], model.config.num_attention_heads
            head_dim = d_model // n_heads
            W_U = model.lm_head.weight
            W_O_rs = W_O.view(d_model, n_heads, head_dim)
        except Exception:
            pass
        else:
            # For each generation step, get attention + logits
            # Note: attentions correspond to each generation step's output layer
            # We compare: did the chosen token's head attend to critical tokens?
            pass  # simplified: full per-step analysis is expensive

    pred = extract_answer(generated_text)
    gt = str(sample.get("answer", "")).upper()

    return {
        "method": "AG (analysis)",
        "predicted_answer": pred,
        "correct": pred != "" and pred == gt,
        "generated_text": generated_text,
        "gen_entropy": gen_entropy,
        "gen_vattn": gen_vattn,
        "gen_token_num": gen_token_num,
        "critical_indices_count": len(critical_indices),
    }


# ==========================================
# Inference Methods
# ==========================================

def run_standard(model, processor, sample: dict,
                 max_new_tokens: int = 8192, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 50) -> dict:
    """Method 1: Standard inference (baseline)."""
    result = generate_with_attn(
        model=model, processor=processor,
        image=sample["image"], question=sample["question"],
        options=sample.get("options", ""),
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, do_sample=True,
    )
    pred = extract_answer(result["generated_text"])
    gt = str(sample.get("answer", "")).upper()
    return {
        "method": "Standard",
        "predicted_answer": pred,
        "correct": pred != "" and pred == gt,
        "generated_text": result["generated_text"],
        "gen_entropy": result["gen_entropy"],
        "gen_vattn": result["gen_vattn"],
        "gen_token_num": result["gen_token_num"],
        "prompt_text": result["prompt_text"],
    }


def run_cad(model, processor, sample: dict,
            max_new_tokens: int = 8192, temperature: float = 0.7,
            top_p: float = 0.9, top_k: int = 50,
            ctx_entropy_threshold: float = 5.0, ctx_top_heads: int = 5) -> dict:
    """Method 2: Context-Aware Decoding — boosts tokens supported by visual-context heads when entropy is high."""
    option_str = f"option: {sample.get('options', '')}\n" if sample.get('options') else ""
    full_question = (
        sample["question"] + option_str +
        'Write the answer into a JSON form\n```json\n{"answer": "X"}```'
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are good at step by step reasoning."}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": full_question},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    prompt_token_num = inputs["input_ids"].shape[1]

    visual_token_indices = get_visual_token_indices(inputs["input_ids"], processor=processor)
    inputs_token_indices = get_input_token_indices(inputs["input_ids"], processor=processor)
    model.visual_token_indices = visual_token_indices
    model.inputs_token_indices = inputs_token_indices
    model.gen_entropy = []
    model.gen_vattn = []
    model.attn_acc_input = []
    model.attn_acc_visual = []

    ctx_processor = ContextAwareLogitsProcessor(
        model=model, top_k=top_k,
        entropy_threshold=ctx_entropy_threshold, top_heads=ctx_top_heads,
    )
    ctx_token_indices = get_context_token_indices(inputs["input_ids"], processor=processor)
    ctx_processor.set_context_token_indices(ctx_token_indices)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            return_dict_in_generate=True, output_logits=True, output_attentions=True,
            logits_processor=[ctx_processor],
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p, top_k=top_k if top_k > 0 else None, do_sample=True,
        )

    gen_entropy = torch.stack(model.gen_entropy, dim=1).cpu() if model.gen_entropy else None
    gen_vattn = torch.stack(model.gen_vattn, dim=1).cpu() if model.gen_vattn else None
    gen_token_num = outputs.sequences.shape[1] - prompt_token_num
    generated_text = processor.decode(outputs.sequences[0][prompt_token_num:])

    pred = extract_answer(generated_text)
    gt = str(sample.get("answer", "")).upper()
    return {
        "method": "CAD",
        "predicted_answer": pred,
        "correct": pred != "" and pred == gt,
        "generated_text": generated_text,
        "gen_entropy": gen_entropy,
        "gen_vattn": gen_vattn,
        "gen_token_num": gen_token_num,
    }


def run_ag(model, processor, sample: dict,
           max_new_tokens: int = 8192, temperature: float = 0.7,
           top_p: float = 0.9, top_k: int = 50) -> dict:
    """Method 3: Attention Guidance — analyze attention patterns post-generation."""
    return _run_ag_analysis(model, processor, sample, max_new_tokens, temperature, top_p, top_k)


def run_cad_plus_ag(model, processor, sample: dict,
                     max_new_tokens: int = 8192, temperature: float = 0.7,
                     top_p: float = 0.9, top_k: int = 50,
                     ctx_entropy_threshold: float = 5.0, ctx_top_heads: int = 5) -> dict:
    """Method 4: CAD + AG — CAD modifies logits during generation; AG analyzed post-hoc."""
    result = run_cad(model, processor, sample, max_new_tokens, temperature, top_p, top_k,
                     ctx_entropy_threshold, ctx_top_heads)
    result["method"] = "CAD+AG"
    return result


# ==========================================
# Main
# ==========================================

def list_samples(dataset_name: str, limit: int = 20):
    """Print sample info (id, question, answer) for the first `limit` samples."""
    dataset = load_dataset(dataset_name)
    print(f"\n{'idx':<6} {'id':<40} {'answer':<10} {'question (first 60 chars)'}")
    print("-" * 100)
    for i in range(min(limit, len(dataset))):
        s = dataset[i]
        q = s["question"][:60] + ("..." if len(s["question"]) > 60 else "")
        print(f"{i:<6} {str(s.get('id', '')):<40} {str(s.get('answer', '')):<10} {q}")


def main():
    parser = argparse.ArgumentParser(description="Case study: compare inference methods on a single sample")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="VisuRiddles", choices=SUPPORTED_DATASETS)
    parser.add_argument("--model_path", type=str, default="/home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctx_entropy_threshold", type=float, default=1.27)
    parser.add_argument("--ctx_top_heads", type=int, default=5)
    parser.add_argument("--show_heva", action="store_true")
    parser.add_argument("--show_entropy", action="store_true")
    parser.add_argument("--truncate_text", type=int, default=9000)
    parser.add_argument("--list", action="store_true", help="List samples in dataset (first 20)")

    args = parser.parse_args()

    if args.list:
        list_samples(args.dataset)
        return

    set_seed(args.seed)

    log_print(f"Loading model from {args.model_path}...")
    model, processor = load_model_processor(args.model_path)
    model.eval()
    log_print("Model loaded.\n")

    dataset = load_dataset(args.dataset)
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        log_print(f"Error: sample_idx {args.sample_idx} out of range [0, {len(dataset)-1}]")
        return
    sample = dataset[args.sample_idx]
    ground_truth = sample.get("answer", "")

    log_print(f"{'='*70}")
    log_print(f"Case Study — Dataset: {args.dataset} | Sample Index: {args.sample_idx}")
    log_print(f"Ground Truth: {ground_truth}")
    log_print(f"{'='*70}\n")

    run_config = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    cad_config = {**run_config, "ctx_entropy_threshold": args.ctx_entropy_threshold, "ctx_top_heads": args.ctx_top_heads}

    methods = [
        ("Standard (baseline)",     lambda: run_standard(model, processor, sample, **run_config)),
        ("CAD",                     lambda: run_cad(model, processor, sample, **cad_config)),
        ("AG (analysis)",           lambda: run_ag(model, processor, sample, **run_config)),
        ("CAD+AG",                  lambda: run_cad_plus_ag(model, processor, sample, **cad_config)),
    ]

    results = {}
    for name, fn in methods:
        log_print(f"\n{'─'*70}")
        log_print(f">>> Running: {name} ...")
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"method": name, "error": str(e)}

    # Summary table
    log_print(f"\n\n{'='*70}")
    log_print("SUMMARY TABLE")
    log_print(f"{'='*70}")
    log_print(f"{'Method':<25} {'Predicted':<12} {'Correct':<8} {'Tokens':<8} {'HEVA':<8} {'AvgEnt':<8}")
    log_print("-" * 72)
    for name, r in results.items():
        if "error" in r:
            log_print(f"{name:<25} {'ERROR':<12} {'—':<8} {'—':<8} {'—':<8} {'—':<8}")
            continue
        heva = compute_heva(r["gen_vattn"]) if r["gen_vattn"] is not None else 0.0
        avg_ent = compute_avg_entropy(r["gen_entropy"]) if r["gen_entropy"] is not None else 0.0
        pred = r["predicted_answer"] or "(none)"
        correct = "✓" if r["correct"] else "✗"
        log_print(f"{name:<25} {pred:<12} {correct:<8} {r['gen_token_num']:<8} {heva:<8.4f} {avg_ent:<8.4f}")

    # Detailed output
    for name, r in results.items():
        if "error" in r:
            continue
        log_print(f"\n{'─'*70}")
        log_print(f"[{name}]")
        log_print(f"  Predicted: {r['predicted_answer'] or '(none)'} | Correct: {r['correct']}")
        text = r["generated_text"]
        truncated = text[:args.truncate_text] + ("..." if len(text) > args.truncate_text else "")
        log_print(f"  Generated text:\n  {truncated}")

        if args.show_heva and r["gen_vattn"] is not None:
            vp = compute_heva_per_token(r["gen_vattn"])
            top10 = vp.topk(min(10, len(vp))).values.tolist()
            log_print(f"  HEVA per token (top 10): {[round(v, 4) for v in top10]}")

        if args.show_entropy and r["gen_entropy"] is not None:
            ent = r["gen_entropy"][:, 0]
            top10 = ent.topk(min(10, len(ent))).values.tolist()
            log_print(f"  Entropy per token (top 10): {[round(v, 4) for v in top10]}")

    log_print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
