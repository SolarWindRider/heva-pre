"""
HEVA 模型推理模块
支持 Qwen3-VL 推理并捕获 attention 和 logits
支持 NVIDIA GPU 和华为 NPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Dict, Any, Tuple, Optional
import torch.nn.functional as F
from metrics.heva import compute_heva


# 立即刷新输出的打印函数
def log_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def load_model_processor(model_path: str):
    """
    Args:
        model_path: 模型路径
        device: 设备，默认为自动检测
        num_gpus: 推理使用的加速卡数量，默认为1
        heva_device: HEVA 计算设备，默认为 None (与 model 相同)
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )

    # 设置为评估模式
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    log_print(f"Load model from {model_path} successfully.")

    return model, processor


def get_visual_token_indices(input_ids: torch.Tensor, processor: AutoProcessor) -> torch.Tensor:
    """
    获取视觉 token 的索引

    在 Qwen3-VL 中:
    - image tokens 使用特殊的 token_id 表示
    - 需要根据 token_id 范围来确定视觉 token 位置

    Returns:
        视觉 token 索引的 tensor
    """
    # Qwen3-VL 的 image token ID 范围
    # 查看 tokenizer 中的特殊 token
    # 这里我们通过检查 input_ids 中是否有大量连续的 image token 来判断

    # 获取 image token 的 token_id
    # 在 Qwen3-VL 中，图像 token 通常在特定范围内
    # 这里使用一个更通用的方法：通过 processor 获取 image token

    # 尝试从 processor 获取 image token id
    try:
        # Qwen3-VL 使用的 image token
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<|image_pad|>")
        ]
    except (ValueError, KeyError):
        # 如果找不到，使用默认值
        # Qwen3-VL-2B 通常使用 151643 作为 image pad
        image_token_id = 151643

    # 找出所有 image token 的位置
    visual_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

    return visual_indices


def get_gen_token_indices(input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """
    获取生成 token 的索引

    Args:
        input_ids: 输入的 token IDs
        prompt_length: prompt 的长度（不包括图像 token）

    Returns:
        生成 token 索引
    """
    # 生成 token 从 prompt_length 开始
    gen_indices = torch.arange(prompt_length, len(input_ids))
    return gen_indices


def generate_with_heva(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image: str,
    question: str,
    options: str,
    max_new_tokens: int,
    alpha_values: list[float],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
) -> Dict[str, Any]:
    """
    带 attention 的生成 - 使用 KV Cache 优化版本

    核心思路：
    1. 使用 HuggingFace generate() + use_cache=True 大幅减少显存
    2. 生成后用单独的前向传播获取 attention（分段计算避免OOM）

    Args:
        image: 图像路径
        question: 问题文本
        options: 选项文本
        max_new_tokens: 最大生成长度
        temperature: 温度
        top_p: top-p 采样
        top_k: top-k 采样
        do_sample: 是否采样

    Returns:
        {
            "generated_text": str,
            "generated_ids": torch.Tensor,
            "logits": torch.Tensor,  # (gen_len, vocab_size)
            "gen_to_visual_attentions": list,  # 新增：只保存 gen->visual 的attention
            "input_ids": torch.Tensor,
            "visual_token_indices": torch.Tensor,
            "prompt_length": int,
            "heva": float,
            "prompt": str,
        }
    """
    # 构建prompt
    option_str = f"option: {options}\n" if options else ""
    full_question = question + option_str + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are good at step by step reasoning."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": full_question},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    prompt_token_num = inputs["input_ids"].shape[1]

    # 获取视觉 token 索引
    visual_token_indices = get_visual_token_indices(inputs["input_ids"][0])

    # 准备 generation config
    generation_config = {
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
            **generation_config,
        )
    gen_token_num = outputs.sequences.shape[1] - prompt_token_num

    gen_entropies = []

    for logits in outputs.logits:  # outputs.logits是(leng[tuple], batch, vocab_size)
        gen_logits = logits[:, prompt_token_num:]  # (gen_len, vocab_size)
        gen_log_probs = F.log_softmax(gen_logits, dim=-1)
        entropy = -(gen_log_probs.exp() * gen_log_probs).sum(dim=-1)  # (B,)
        gen_entropies.append(entropy)

    gen_entropy = torch.stack(gen_entropies, dim=0)  # (T, B)


    # 选取 top-alpha 高熵 tokens
    top_alpha_token_indices_dic = {}
    for alpha in alpha_values:
        top_alpha_token_indices_dic[f"heva_{alpha:.2f}"] = torch.topk(gen_entropy, max(1, int(gen_entropy.shape[0] * alpha)), dim=0).indices  # (k, B)
        top_alpha_token_indices = torch.topk(gen_entropy, max(1, int(gen_entropy.shape[0] * alpha)), dim=0).indices  # (k, B)
        heva_result = compute_heva(
            logits=outputs.logits[0],  # (seq_len, vocab_size)
            attentions=[attn[0] for attn in outputs.attentions],  # List of (num_heads, seq_len, seq_len)
            visual_token_indices=visual_token_indices,
            gen_token_indices=get_gen_token_indices(inputs["input_ids"][0], prompt_token_num),
            alpha=alpha,
            use_last_layer_only=True,
        )
        heva_value = heva_result["heva"]
        log_print(f"Alpha: {alpha:.2f}, HEVA: {heva_value:.4f}")


    return {
        "prompt_text": processor.decode(inputs["input_ids"][0]),
        "generated_text": processor.decode(outputs.sequences[0][prompt_token_num:]),
        "prompt_token_num": prompt_token_num,
        "gen_token_num": gen_token_num,
        "heva": heva_value,
    }
