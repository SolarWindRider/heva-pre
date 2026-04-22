import argparse
import json
import os
import re
import pickle
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics.heva import _sample_with_vattn_and_entropy, get_entropy, get_vattn
from metrics.context_aware_logits_processor import (
    ContextAwareLogitsProcessor,
    get_context_token_indices,
    get_visual_token_indices,
)
from data.loader import load_dataset, SUPPORTED_DATASETS

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Monkey-patch Qwen3VLForConditionalGeneration to capture attention/entropy during generation
Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy


# ==========================================
# Utility Functions
# ==========================================


def log_print(*args, **kwargs):
    """Print with immediate flush."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def set_seed(seed: int):
    """Set global random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_image_token_id(processor) -> int:
    """Get the image token ID from processor."""
    try:
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<|image_pad|>")
        ]
    except (ValueError, KeyError):
        image_token_id = 151643
    return image_token_id


def build_prompt(
    sample: Dict[str, Any], processor, prompt_template: str = None
) -> Tuple[List[Dict], str]:
    """
    Build messages for Qwen3-VL chat template.

    Args:
        sample: Data sample with 'question', 'options', 'image' keys
        processor: Qwen3-VL processor

    Returns:
        messages: List of message dicts for chat template
        full_question: The formatted question text
    """
    option_str = f"option: {sample['options']}\n" if sample.get("options") else ""

    if prompt_template is None:
        prompt_template = '{question}{options}Think carefully and write the answer into a JSON form\n```json\n{{"answer": "X"}}```'

    full_question = prompt_template.format(
        question=sample["question"], options=option_str
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": full_question},
            ],
        },
    ]
    return messages, full_question


# ==========================================
# Module 1: Critical Context Detection
# ==========================================


def get_critical_context_indices(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    prompt_text: str,
    visual_token_indices: Tuple[torch.Tensor, torch.Tensor],
) -> List[int]:
    """
    Identify critical context indices for the input.

    For VL models, we focus on:
    1. Visual token indices (image patches)
    2. Numerical tokens (digits)
    3. Operator tokens (+, -, *, /, =)

    Args:
        model: Qwen3VLForConditionalGeneration model
        processor: AutoProcessor instance
        input_ids: Input token IDs (batch, seq_len)
        prompt_text: Decoded prompt text
        visual_token_indices: (start, end) indices for visual tokens

    Returns:
        List of critical token indices
    """
    tokens = input_ids[0]
    critical_indices = []

    v_start, v_end = visual_token_indices[0].item(), visual_token_indices[1].item()
    if v_end > v_start:
        critical_indices.extend(range(v_start, v_end + 1))

    log_print(
        f"[Context] Critical indices ({len(critical_indices)}): visual=[{v_start}, {v_end}], text_nums={len(critical_indices) - max(0, v_end - v_start + 1)}"
    )
    return sorted(critical_indices)


# ==========================================
# Module 2: DLA Causal Path Computation
# ==========================================


def compute_head_path_simple(
    model: Qwen3VLForConditionalGeneration,
    last_z: torch.Tensor,
    W_U: torch.Tensor,
    W_O: torch.Tensor,
    token_id: int,
    d_model: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Simple per-layer head contribution computation.

    For a given output token, compute which attention head contributed most
    using: head_output = z[head] · W_O[head] · W_U[token]

    Args:
        model: The model
        last_z: z tensor from last layer (batch, seq, heads, d_head)
        W_U: Unembedding matrix (vocab, d_model)
        W_O: Output projection weights reshaped to (d_model, n_heads, d_head)
        token_id: Target token ID
        d_model: Model dimension

    Returns:
        Dict mapping layer_idx -> {"head": head_idx, "score": contribution}
    """
    n_heads = last_z.shape[2]
    head_dim = last_z.shape[3]

    if W_U.shape[0] == d_model:
        W_U_target = W_U[:, token_id].squeeze()
    else:
        W_U_target = W_U[token_id, :].squeeze()

    z = last_z[0, -1, :, :]
    W_O_reshaped = W_O.view(d_model, n_heads, head_dim)

    head_outputs = torch.einsum("hd,mhd->hm", z, W_O_reshaped)
    head_contributions = head_outputs @ W_U_target

    max_head_idx = torch.argmax(head_contributions).item()
    max_score = head_contributions[max_head_idx].item()

    return {
        model.model.language_model.config.num_hidden_layers
        - 1: {"head": max_head_idx, "score": max_score}
    }


# ==========================================
# Module 3: Path Verification
# ==========================================


def verify_attention_focus(
    attention_weights: torch.Tensor,
    head_path: Dict[int, Dict[str, Any]],
    critical_indices: List[int],
    seq_len: int,
    top_k_attn: int = 5,
) -> bool:
    """
    Check if attention heads in path focus on critical context indices.

    Args:
        attention_weights: Attention weights tensor (batch, heads, query, key)
        head_path: Dict of layer -> {"head": head_idx, "score": ...}
        critical_indices: List of critical token indices
        seq_len: Sequence length
        top_k_attn: Number of top attended tokens to check

    Returns:
        True if >= 30% of layers' heads attend to critical tokens
    """
    if not critical_indices or attention_weights is None:
        return False

    attn = attention_weights[-1]
    attn_last_query = attn[0, :, -1, :]

    hit_count = 0
    total_layers_checked = len(head_path)

    for layer, info in head_path.items():
        head = info["head"]
        attn_pattern = attn_last_query[head]

        k = min(top_k_attn, seq_len)
        _, top_attn_indices = torch.topk(attn_pattern, k=k)
        top_attn_indices_list = top_attn_indices.tolist()

        if any(idx in top_attn_indices_list for idx in critical_indices):
            hit_count += 1

    hit_ratio = hit_count / total_layers_checked if total_layers_checked > 0 else 0
    return hit_ratio >= 0.3


# ==========================================
# Module 4: Attention-Guided Token Selection
# ==========================================


def attention_guided_token_selection(
    model: Qwen3VLForConditionalGeneration,
    last_z: torch.Tensor,
    attention_weights: Tuple[torch.Tensor, ...],
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    critical_indices: List[int],
    top_k_vocab: int = 10,
    top_k_attn: int = 5,
) -> List[int]:
    """
    Select tokens based on attention path verification.

    Args:
        model: The model
        last_z: Last layer's z tensor
        attention_weights: Tuple of attention tensors per layer
        logits: Raw logits (batch, vocab)
        token_ids: Candidate token IDs (top_k_vocab,)
        critical_indices: List of critical context indices
        top_k_vocab: Number of top vocab candidates
        top_k_attn: Top-k attention positions to check

    Returns:
        List of valid token IDs that pass path verification
    """
    seq_len = last_z.shape[1] if last_z is not None else 1

    try:
        last_layer = model.model.language_model.layers[-1]
        W_O = last_layer.self_attn.o_proj.weight
        d_model = W_O.shape[0]
        n_heads = model.model.language_model.config.num_attention_heads
        head_dim = d_model // n_heads
    except Exception:
        return token_ids.tolist()

    try:
        W_U = model.lm_head.weight
    except Exception:
        return token_ids.tolist()

    W_O_reshaped = W_O.view(d_model, n_heads, head_dim)

    valid_candidates = []
    valid_logits = []

    for i in range(min(top_k_vocab, len(token_ids))):
        tok_id = token_ids[i].item()
        base_logit = logits[0, tok_id].item()

        path = compute_head_path_simple(
            model, last_z, W_U, W_O_reshaped, tok_id, d_model
        )
        is_valid = verify_attention_focus(
            attention_weights, path, critical_indices, seq_len, top_k_attn
        )

        if is_valid:
            valid_candidates.append(tok_id)
            valid_logits.append(base_logit)

    return valid_candidates if valid_candidates else token_ids[:1].tolist()


# ==========================================
# Module 5: Inference with Attention Capture
# ==========================================


def generate_with_attention_guidance(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    max_new_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    use_context_aware: bool = False,
    ctx_entropy_threshold: float = 5.0,
    ctx_top_heads: int = 5,
    use_attention_guidance: bool = False,
    dla_entropy_threshold: float = None,
    prompt_template: str = None,
) -> Dict[str, Any]:
    """
    Run inference with optional attention-guided generation.

    Args:
        model: Qwen3VLForConditionalGeneration model
        processor: AutoProcessor instance
        sample: Data sample with 'question', 'options', 'image', 'answer'
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling
        do_sample: Whether to sample
        use_context_aware: Use context-aware logits processor
        ctx_entropy_threshold: Entropy threshold for context-aware mode
        ctx_top_heads: Number of context heads for context-aware mode
        use_attention_guidance: Use attention-guided token selection
        prompt_template: Custom prompt template

    Returns:
        Dict with generation results including attention metrics
    """
    messages, full_question = build_prompt(sample, processor, prompt_template)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    prompt_token_num = inputs["input_ids"].shape[1]

    visual_token_indices = get_visual_token_indices(
        inputs["input_ids"], processor=processor
    )
    inputs_token_indices = (
        torch.zeros_like(visual_token_indices[0]),
        visual_token_indices[1],
    )

    model.visual_token_indices = visual_token_indices
    model.inputs_token_indices = inputs_token_indices
    model.gen_entropy = []
    model.gen_vattn = []
    model.attn_acc_input = []
    model.attn_acc_visual = []
    model.gen_zs = (
        []
    )  # per-layer z for DLA trace (num_layers, batch, seq, heads, d_head)

    # [NEW] Attention-guided generation: set critical context indices and guidance flag on model
    # Critical indices = visual token indices (image patches) in the prompt
    # This mirrors the HEVA focus: which tokens attend to visual context?
    if use_attention_guidance:
        # visual_token_indices is already computed above: (start, end) tensor
        v_start = visual_token_indices[0].item()
        v_end = visual_token_indices[1].item()
        critical_indices = list(range(v_start, v_end + 1))
        model.critical_indices = critical_indices
        model.use_attention_guidance = True
        model.attn_guidance_top_k = top_k
        model.attn_guidance_topk_attn = 5
        model.dla_entropy_threshold = dla_entropy_threshold
        log_print(
            f"[Attention Guidance] Critical indices: visual=[{v_start}, {v_end}], total={len(critical_indices)}"
        )
    else:
        model.use_attention_guidance = False
        model.critical_indices = []

    logits_processor = None
    if use_context_aware:
        ctx_processor = ContextAwareLogitsProcessor(
            model=model,
            top_k=top_k,
            entropy_threshold=ctx_entropy_threshold,
            top_heads=ctx_top_heads,
        )
        ctx_token_indices = get_context_token_indices(
            inputs["input_ids"], processor=processor
        )
        ctx_processor.set_context_token_indices(ctx_token_indices)
        logits_processor = [ctx_processor]

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p,
        "top_k": top_k if top_k > 0 else None,
        "do_sample": do_sample,
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_logits=True,
            output_attentions=True,
            logits_processor=logits_processor,
            **gen_config,
        )

    gen_token_num = outputs.sequences.shape[1] - prompt_token_num

    if model.gen_entropy:
        gen_entropy = torch.stack(model.gen_entropy, dim=1).cpu()
        gen_vattn = torch.stack(model.gen_vattn, dim=1).cpu()
        attn_acc_input = torch.stack(model.attn_acc_input, dim=1).cpu()
        attn_acc_visual = torch.stack(model.attn_acc_visual, dim=1).cpu()
        gen_zs = [z.cpu() for z in model.gen_zs if z is not None]
    else:
        gen_entropy = gen_vattn = attn_acc_input = attn_acc_visual = gen_zs = None

    generated_text = processor.decode(outputs.sequences[0][prompt_token_num:])

    answer_pred = ""
    json_match = re.search(r'\{\s*"answer"\s*:\s*"([^"]+)"\s*\}', generated_text)
    if json_match:
        answer_pred = json_match.group(1).upper()

    return {
        "prompt_text": processor.decode(inputs["input_ids"][0]),
        "generated_text": generated_text,
        "predicted_answer": answer_pred,
        "correct": answer_pred != ""
        and answer_pred in str(sample.get("answer", "")).upper(),
        "prompt_token_num": prompt_token_num,
        "gen_token_num": gen_token_num,
        "gen_entropy": gen_entropy,
        "gen_vattn": gen_vattn,
        "attn_acc_input": attn_acc_input,
        "attn_acc_visual": attn_acc_visual,
        "gen_zs": gen_zs,  # list of (num_layers, batch, seq, heads, d_head) per token
        "sequences": outputs.sequences,
    }


# ==========================================
# Module 6: Batch Inference Runner
# ==========================================


def run_inference(
    model_path: str,
    dataset_name: str,
    sample_indices: List[int],
    output_dir: str,
    max_new_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    use_context_aware: bool = False,
    ctx_entropy_threshold: float = 5.0,
    ctx_top_heads: int = 5,
    use_attention_guidance: bool = False,
    dla_entropy_threshold: float = None,
    prompt_template: str = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run batch inference on dataset.

    Args:
        model_path: Path to Qwen3-VL model
        dataset_name: Name of dataset to load
        sample_indices: List of sample indices to process
        output_dir: Output directory
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling
        do_sample: Whether to sample
        use_context_aware: Enable context-aware decoding
        ctx_entropy_threshold: Entropy threshold for CAD
        ctx_top_heads: Number of context heads for CAD
        use_attention_guidance: Enable attention-guided selection
        prompt_template: Custom prompt template
        shuffle: Shuffle dataset before processing
        seed: Random seed

    Returns:
        (results, errors) lists
    """
    set_seed(seed)

    os.makedirs(f"{output_dir}/pkls", exist_ok=True)

    log_print(f"Loading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    log_print(f"Model loaded: {model_path}")

    dataset = load_dataset(dataset_name)
    log_print(f"Dataset '{dataset_name}' loaded: {len(dataset)} samples")

    if shuffle:
        import random

        random.seed(seed)
        sample_indices = sample_indices.copy()
        random.shuffle(sample_indices)

    results = []
    errors = []

    for idx in tqdm(sample_indices, desc="Inference"):
        try:
            sample = dataset[idx]

            result = generate_with_attention_guidance(
                model=model,
                processor=processor,
                sample=sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                use_context_aware=use_context_aware,
                ctx_entropy_threshold=ctx_entropy_threshold,
                ctx_top_heads=ctx_top_heads,
                use_attention_guidance=use_attention_guidance,
                dla_entropy_threshold=dla_entropy_threshold,
                prompt_template=prompt_template,
            )

            sample_id = (
                str(sample.get("id", idx))
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )
            gen_entropy_path = os.path.join(
                output_dir, "pkls", f"{idx}_gen_entropy.pkl"
            )
            gen_vattn_path = os.path.join(output_dir, "pkls", f"{idx}_gen_vattn.pkl")
            gen_zs_path = os.path.join(output_dir, "pkls", f"{idx}_gen_zs.pkl")

            if result["gen_entropy"] is not None:
                with open(gen_entropy_path, "wb") as f:
                    pickle.dump(result["gen_entropy"], f)
            if result["gen_vattn"] is not None:
                with open(gen_vattn_path, "wb") as f:
                    pickle.dump(result["gen_vattn"], f)
            if result["gen_zs"] is not None:
                with open(gen_zs_path, "wb") as f:
                    pickle.dump(result["gen_zs"], f)

            avg_vattn = (
                result["gen_vattn"].mean().item()
                if result["gen_vattn"] is not None
                else 0.0
            )

            meta = {
                "idx": idx,
                "sample_id": sample_id,
                "image_path": sample.get("image_path", sample.get("image", "")),
                "ground_truth": sample.get("answer", ""),
                "predicted_answer": result["predicted_answer"],
                "correct": result["correct"],
                "prompt_token_num": result["prompt_token_num"],
                "gen_token_num": result["gen_token_num"],
                "avg_vattn": avg_vattn,
                "gen_entropy_path": gen_entropy_path,
                "gen_vattn_path": gen_vattn_path,
                "gen_zs_path": gen_zs_path,
                "use_context_aware": use_context_aware,
                "use_attention_guidance": use_attention_guidance,
                "dla_entropy_threshold": dla_entropy_threshold,
            }

            meta_path = os.path.join(output_dir, f"{sample_id}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            results.append(
                {
                    "idx": idx,
                    "sample_id": sample_id,
                    "meta_path": meta_path,
                }
            )

            log_print(
                f"[{sample_id}] pred={result['predicted_answer']} | gt={sample.get('answer', '')} | correct={result['correct']} | vattn={avg_vattn:.3f}"
            )

        except Exception as e:
            errors.append({"idx": idx, "error": str(e)})
            log_print(f"Error at idx {idx}: {e}")
            import traceback

            traceback.print_exc()

    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(
            {
                "num_samples": len(results),
                "num_errors": len(errors),
                "results": results,
                "errors": errors,
            },
            f,
            indent=2,
        )

    log_print(f"\nCompleted: {len(results)} results, {len(errors)} errors")
    log_print(f"Output: {output_dir}")

    return results, errors


# ==========================================
# Entry Point
# ==========================================


def main():
    from data.loader import SUPPORTED_DATASETS

    parser = argparse.ArgumentParser(
        description="HEVA Attention-Guided Generation for Qwen3-VL"
    )

    # Experiment
    parser.add_argument(
        "--exp_name", type=str, default="trace_exp", help="Experiment name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="VisuRiddles",
        choices=SUPPORTED_DATASETS,
        help="Dataset name",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples (-1 for all)"
    )
    parser.add_argument(
        "--shuffle", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Instruct",
        help="Model path",
    )

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument(
        "--do_sample", type=str, default="true", choices=["true", "false"]
    )

    # Context-Aware Decoding
    parser.add_argument(
        "--use_context_aware", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument("--ctx_entropy_threshold", type=float, default=5.0)
    parser.add_argument("--ctx_top_heads", type=int, default=5)

    # Attention Guidance
    parser.add_argument(
        "--use_attention_guidance", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument(
        "--dla_entropy_threshold",
        type=float,
        default=None,
        help="DLA only applied when entropy > threshold",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--prompt_template", type=str, default=None, help="Custom prompt template"
    )

    args = parser.parse_args()

    do_sample = args.do_sample.lower() == "true"
    use_context_aware = args.use_context_aware.lower() == "true"
    use_attention_guidance = args.use_attention_guidance.lower() == "true"
    shuffle = args.shuffle.lower() == "true"

    output_dir = os.path.join(args.output_dir, args.exp_name, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "model_name": args.model_path.split("/")[-1],
        "model_path": args.model_path,
        "num_samples": args.num_samples,
        "shuffle": shuffle,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": do_sample,
        "use_context_aware": use_context_aware,
        "ctx_entropy_threshold": args.ctx_entropy_threshold,
        "ctx_top_heads": args.ctx_top_heads,
        "use_attention_guidance": use_attention_guidance,
        "dla_entropy_threshold": args.dla_entropy_threshold,
        "prompt_template": args.prompt_template,
    }
    config_path = os.path.join(output_dir, "exp_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log_print(f"Config saved to: {config_path}")

    dataset = load_dataset(args.dataset)
    if args.num_samples == -1:
        end_idx = len(dataset)
    else:
        end_idx = min(args.num_samples, len(dataset))
    sample_indices = list(range(0, end_idx))

    log_print(f"Starting inference: {len(sample_indices)} samples")
    log_print(f"Model: {args.model_path}")
    log_print(
        f"Context-Aware: {use_context_aware}, Attention-Guidance: {use_attention_guidance}"
    )

    run_inference(
        model_path=args.model_path,
        dataset_name=args.dataset,
        sample_indices=sample_indices,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=do_sample,
        use_context_aware=use_context_aware,
        ctx_entropy_threshold=args.ctx_entropy_threshold,
        ctx_top_heads=args.ctx_top_heads,
        use_attention_guidance=use_attention_guidance,
        dla_entropy_threshold=args.dla_entropy_threshold,
        prompt_template=args.prompt_template,
        shuffle=shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
