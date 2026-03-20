"""
HEVA 模型推理模块
支持 Qwen3-VL 推理并捕获 attention 和 logits
支持 NVIDIA GPU 和华为 NPU
"""

import torch
from typing import Dict, Any
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from metrics.heva import _sample_with_vattn_and_entropy

Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy  # 替换原有的 _sample 方法，以便在生成过程中获取注意力权重和熵值


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

    model = Qwen3VLForConditionalGeneration.from_pretrained(
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
    visual_indices = visual_indices[0], visual_indices[-1]
    return visual_indices


def generate_with_attn(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: str,
    question: str,
    options: str,
    max_new_tokens: int,
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
    inputs = inputs.to(model.device)
    prompt_token_num = inputs["input_ids"].shape[1]

    # 获取视觉 token 索引
    visual_token_indices = get_visual_token_indices(inputs["input_ids"][0], processor=processor)
    model.visual_token_indices = visual_token_indices
    model.gen_entropy = []
    model.gen_vattn = []
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
            return_dict_in_generate=True,  # 必须为True才可以计算entropy和vattn
            output_logits=True,  # 必须为True才可以计算entropy和vattn
            output_attentions=True,  # 必须为True才可以计算entropy和vattn
            **generation_config,
        )
    gen_token_num = outputs.sequences.shape[1] - prompt_token_num

    model.gen_entropy = torch.stack(model.gen_entropy, dim=1).detach().cpu()  # (gen_tokens_num, batch_size)
    model.gen_vattn = torch.stack(model.gen_vattn, dim=1).detach().cpu()  # (gen_tokens_num, batch_size, visual_token_num)

    return {
        "prompt_text": processor.decode(inputs["input_ids"][0]),
        "generated_text": processor.decode(outputs.sequences[0][prompt_token_num:]),
        "prompt_token_num": prompt_token_num,
        "gen_token_num": gen_token_num,
        "gen_entropy": model.gen_entropy,
        "gen_vattn": model.gen_vattn,
    }
