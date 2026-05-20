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
        prompt_template = '{question}  {options} \nThink carefully and write the answer into a JSON form\n```json\n{{"answer": "X"}}```'

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
    samples: List[Dict[str, Any]],
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
) -> List[Dict[str, Any]]:
    batch_size = len(samples)
    messages_list = []
    for sample in samples:
        msgs, _ = build_prompt(sample, processor, prompt_template)
        messages_list.append(msgs)

    all_inputs = []
    prompt_token_nums = []
    for msgs in messages_list:
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)
        all_inputs.append(inputs)
        prompt_token_nums.append(inputs["input_ids"].shape[1])

    max_prompt_len = max(inp["input_ids"].shape[1] for inp in all_inputs)
    batch_input_ids_list = []
    batch_attention_mask_list = []
    for inp in all_inputs:
        pad_len = max_prompt_len - inp["input_ids"].shape[1]
        # Left-padding is REQUIRED for batched causal LM generation.
        # Right-padding uses padding-position hidden states for next-token
        # prediction (logits[:, -1, :]), producing corrupted/shorter-sequence output.
        pad_token_id = processor.tokenizer.pad_token_id or 0
        if pad_len > 0:
            batch_input_ids_list.append(torch.cat([torch.full((1, pad_len), pad_token_id, dtype=inp["input_ids"].dtype), inp["input_ids"]], dim=1))
            batch_attention_mask_list.append(torch.cat([torch.zeros(1, pad_len, dtype=inp["attention_mask"].dtype), inp["attention_mask"]], dim=1))
        else:
            batch_input_ids_list.append(inp["input_ids"])
            batch_attention_mask_list.append(inp["attention_mask"])

    batch_input_ids = torch.cat(batch_input_ids_list, dim=0).to(model.device)
    batch_attention_mask = torch.cat(batch_attention_mask_list, dim=0).to(model.device)
    prompt_token_num = max_prompt_len

    all_pixel_values = [inp["pixel_values"] for inp in all_inputs]
    all_image_grid_thw = torch.cat([inp["image_grid_thw"] for inp in all_inputs], dim=0).to(model.device)
    batch_pixel_values = torch.cat(all_pixel_values, dim=0).to(model.device)

    visual_indices_list = [get_visual_token_indices(batch_input_ids[i:i+1], processor=processor) for i in range(len(all_inputs))]
    visual_start = torch.stack([v[0] for v in visual_indices_list])
    visual_end = torch.stack([v[1] for v in visual_indices_list])
    model.visual_token_indices = (visual_start, visual_end)
    model.inputs_token_indices = (torch.zeros_like(visual_start), visual_end)
    if hasattr(model, 'gen_entropy') and model.gen_entropy:
        model.gen_entropy.clear()
    model.gen_entropy = []
    model.gen_vattn = []
    model.attn_acc_input = []
    model.attn_acc_visual = []
    model.gen_zs = []

    if use_attention_guidance:
        v_start = visual_start[0].item()
        v_end = visual_end[0].item()
        model.critical_indices = list(range(v_start, v_end + 1))
        model.use_attention_guidance = True
        model.attn_guidance_top_k = top_k
        model.attn_guidance_topk_attn = 5
        model.dla_entropy_threshold = dla_entropy_threshold

    logits_processor = None
    if use_context_aware:
        ctx_processor = ContextAwareLogitsProcessor(model=model, top_k=top_k, entropy_threshold=ctx_entropy_threshold, top_heads=ctx_top_heads)
        ctx_token_indices = get_context_token_indices(batch_input_ids, processor=processor)
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
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            pixel_values=batch_pixel_values,
            image_grid_thw=all_image_grid_thw,
            return_dict_in_generate=True,
            output_logits=True,
            output_attentions=True,
            logits_processor=logits_processor,
            **gen_config,
        )

    gen_token_num = outputs.sequences.shape[1] - prompt_token_num

    actual_batch_size = len(samples)
    if model.gen_entropy:
        gen_entropy = torch.stack(model.gen_entropy, dim=0).cpu()
        gen_vattn = torch.stack(model.gen_vattn, dim=0).cpu()
        attn_acc_input = torch.stack(model.attn_acc_input, dim=0).cpu()
        attn_acc_visual = torch.stack(model.attn_acc_visual, dim=0).cpu()
        gen_zs = [z.cpu() for z in model.gen_zs if z is not None]
        if gen_entropy.shape[1] < actual_batch_size:
            raise IndexError(f"gen_entropy dim=1 is {gen_entropy.shape[1]} but actual_batch_size={actual_batch_size}")
    else:
        gen_entropy = gen_vattn = attn_acc_input = attn_acc_visual = gen_zs = None

    results = []
    for b in range(actual_batch_size):
        generated_text = processor.decode(outputs.sequences[b][prompt_token_num:])
        answer_pred = ""
        answer_format = samples[b].get("answer_format", "mcq")

        if answer_format == "open_vqa":
            json_match = re.search(r'\{\s*"answer"\s*:\s*"([^"]+)"\s*\}', generated_text)
            if json_match:
                answer_pred = json_match.group(1).upper()
            else:
                match = re.search(r'(?:the answer is|answer:)\s*([a-z]+)', generated_text, re.I)
                if match:
                    answer_pred = match.group(1).upper()
                else:
                    words = re.findall(r'\b[a-z]{2,}\b', generated_text)
                    answer_pred = max(words, key=len).upper() if words else ""
            gt = str(samples[b].get("answer", "")).upper()
            correct = answer_pred != "" and answer_pred == gt
        else:
            json_match = re.search(r'\{\s*"answer"\s*:\s*"([^"]+)"\s*\}', generated_text)
            if json_match:
                answer_pred = json_match.group(1).upper()
            correct = answer_pred != "" and answer_pred in str(samples[b].get("answer", "")).upper()

        prompt_tokens = all_inputs[b]["input_ids"][0].tolist()
        gen_tokens = outputs.sequences[b][prompt_token_num:].tolist()
        results.append({
            "prompt_text": processor.decode(all_inputs[b]["input_ids"][0]),
            "generated_text": generated_text,
            "predicted_answer": answer_pred,
            "correct": correct,
            "prompt_token_num": prompt_token_nums[b],
            "gen_token_num": gen_token_num,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "gen_entropy": gen_entropy[:, b] if gen_entropy is not None else None,
            "gen_vattn": gen_vattn[:, b] if gen_vattn is not None else None,
            "attn_acc_input": attn_acc_input[:, b] if attn_acc_input is not None else None,
            "attn_acc_visual": attn_acc_visual[:, b] if attn_acc_visual is not None else None,
            "gen_zs": gen_zs,
        })

    return results


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
    batch_size: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    set_seed(seed)

    os.makedirs(f"{output_dir}/pkls", exist_ok=True)

    log_print(f"Loading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).cuda()
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

    num_batches = (len(sample_indices) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(sample_indices), batch_size), desc="Batch Inference", total=num_batches):
        batch_end = min(batch_start + batch_size, len(sample_indices))
        batch_indices = sample_indices[batch_start:batch_end]
        batch_samples = [dataset[idx] for idx in batch_indices]

        try:
            batch_results = generate_with_attention_guidance(
                model=model,
                processor=processor,
                samples=batch_samples,
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

            for i, (idx, result) in enumerate(zip(batch_indices, batch_results)):
                sample = batch_samples[i]
                sample_id = (
                    str(sample.get("id", idx))
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(":", "_")
                )
                gen_entropy_path = os.path.join(output_dir, "pkls", f"{idx}_gen_entropy.pkl")
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

                tokens = {
                    "prompt_text": processor.decode(result["prompt_tokens"]),
                    "gen_text": processor.decode(result["gen_tokens"]),
                    "prompt_tokens": processor.batch_decode(result["prompt_tokens"]),
                    "gen_tokens": processor.batch_decode(result["gen_tokens"]),
                }
                tokens_path = os.path.join(output_dir, "pkls", f"{idx}_tokens.json")
                with open(tokens_path, "w", encoding="utf-8") as f:
                    json.dump(tokens, f, ensure_ascii=False, indent=2)

                avg_vattn = result["gen_vattn"].mean().item() if result["gen_vattn"] is not None else 0.0

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
                    "tokens_path": tokens_path,
                    "use_context_aware": use_context_aware,
                    "use_attention_guidance": use_attention_guidance,
                    "dla_entropy_threshold": dla_entropy_threshold,
                }

                meta_path = os.path.join(output_dir, f"{sample_id}_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                results.append({"idx": idx, "sample_id": sample_id, "meta_path": meta_path})

                log_print(
                    f"[{sample_id}] pred={result['predicted_answer']} | gt={sample.get('answer', '')} | correct={result['correct']} | vattn={avg_vattn:.3f}"
                )

        except Exception as e:
            for idx in batch_indices:
                errors.append({"idx": idx, "error": str(e)})
                log_print(f"Error at batch idx {idx}: {e}")
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
        help="Dataset name(s), comma-separated (e.g., 'VisuRiddles,RAVEN,MARVEL')",
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

    # Batch processing
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference (default: 1)"
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

    datasets = [d.strip() for d in args.dataset.split(",")]
    log_print(f"Datasets: {datasets}")

    for dataset_name in datasets:
        output_dir = os.path.join(args.output_dir, args.exp_name, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        config = {
            "exp_name": args.exp_name,
            "dataset": dataset_name,
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
            "batch_size": args.batch_size,
            "prompt_template": args.prompt_template,
        }
        config_path = os.path.join(output_dir, "exp_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log_print(f"Config saved to: {config_path}")

        dataset = load_dataset(dataset_name)
        if args.num_samples == -1:
            end_idx = len(dataset)
        else:
            end_idx = min(args.num_samples, len(dataset))
        sample_indices = list(range(0, end_idx))

        log_print(f"Starting inference: {len(sample_indices)} samples, batch_size={args.batch_size}")
        log_print(f"Model: {args.model_path}")
        log_print(
            f"Context-Aware: {use_context_aware}, Attention-Guidance: {use_attention_guidance}"
        )

        run_inference(
            model_path=args.model_path,
            dataset_name=dataset_name,
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
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
