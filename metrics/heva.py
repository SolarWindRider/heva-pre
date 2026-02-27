"""
HEVA (High-Entropy Visual Attention) 计算模块

根据 doc.md 中的定义实现:
1. 计算生成 token 的 entropy
2. 选取 top-alpha 高熵 token
3. 计算这些 token 对视觉 token 的平均注意力
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算每个 token 的 entropy

    Args:
        logits: (seq_len, vocab_size) 或 (batch, seq_len, vocab_size)

    Returns:
        entropy: (seq_len,) 或 (batch, seq_len)
    """
    if logits.dim() == 3:
        # (batch, seq_len, vocab_size) -> (batch, seq_len)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    elif logits.dim() == 2:
        # (seq_len, vocab_size) -> (seq_len,)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    else:
        raise ValueError(f"Unexpected logits dimension: {logits.dim()}")

    return entropy


def compute_heva(
    logits: torch.Tensor,
    attentions: Tuple[torch.Tensor],
    visual_token_indices: torch.Tensor,
    gen_token_indices: torch.Tensor,
    alpha: float = 0.2,
) -> Dict[str, Any]:
    """
    计算 HEVA 指标

    Args:
        logits: (seq_len, vocab_size) 模型输出的 logits
        attentions: (num_layers, batch, num_heads, seq_len, seq_len) attention 元组
        visual_token_indices: 视觉 token 的索引
        gen_token_indices: 生成 token 的索引
        alpha: 高熵 token 的比例

    Returns:
        {
            "heva": float,
            "high_entropy_tokens": torch.Tensor,
            "visual_attentions": list,
            "entropies": torch.Tensor
        }
    """
    # 1. 计算所有 token 的 entropy
    entropies = compute_entropy(logits)  # (seq_len,)

    # 2. 只对生成 token 计算 entropy
    if len(gen_token_indices) == 0:
        return {
            "heva": 0.0,
            "high_entropy_tokens": torch.tensor([], dtype=torch.long),
            "visual_attentions": [],
            "entropies": entropies
        }

    entropy_gen = entropies[gen_token_indices]  # (gen_len,)

    # 3. 选取 top-alpha 高熵 token
    k = max(1, int(len(entropy_gen) * alpha))
    if k >= len(entropy_gen):
        k = len(entropy_gen) - 1

    _, top_indices = torch.topk(entropy_gen, k)
    selected_tokens = gen_token_indices[top_indices]  # 高熵 token 的绝对索引

    # 4. 获取最后一层的 attention
    # attentions 是 Tuple[ (batch, num_heads, seq_len, seq_len) ]
    attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
    num_heads = attn.shape[0]

    # 5. 对每个高熵 token 计算视觉注意力
    visual_attentions = []

    for t in selected_tokens:
        # attn[:, t, :] 的形状: (num_heads, seq_len)
        attn_t = attn[:, t, :]  # (num_heads, seq_len)

        # 对视觉 token 求和
        if len(visual_token_indices) > 0:
            visual_attn_sum = attn_t[:, visual_token_indices].sum(dim=-1)  # (num_heads,)
        else:
            visual_attn_sum = torch.zeros(num_heads, device=attn.device)

        # 对 head 求平均
        v_t = visual_attn_sum.mean().item()  # scalar
        visual_attentions.append(v_t)

    # 6. 计算 HEVA
    if len(visual_attentions) > 0:
        heva = np.mean(visual_attentions)
    else:
        heva = 0.0

    return {
        "heva": heva,
        "high_entropy_tokens": selected_tokens,
        "visual_attentions": visual_attentions,
        "entropies": entropies,
        "selected_entropy_values": entropy_gen[top_indices].cpu().numpy()
    }


def compute_heva_from_result(result: Dict[str, Any], alpha: float = 0.2) -> Dict[str, Any]:
    """
    从推理结果计算 HEVA

    Args:
        result: generate_with_attention 返回的结果
        alpha: 高熵 token 比例

    Returns:
        HEVA 计算结果
    """
    # 获取参数
    logits = result["logits"]
    attentions = result["attentions"]
    visual_token_indices = result["visual_token_indices"]

    # 计算生成 token 索引
    prompt_length = result["prompt_length"]
    input_ids = result["input_ids"]

    # 生成 token 从 prompt_length 开始，到序列结束
    gen_token_indices = torch.arange(prompt_length, len(input_ids), device=input_ids.device)

    return compute_heva(
        logits=logits,
        attentions=attentions,
        visual_token_indices=visual_token_indices,
        gen_token_indices=gen_token_indices,
        alpha=alpha
    )


def compute_token_visual_attention(
    token_idx: int,
    attentions: Tuple[torch.Tensor],
    visual_token_indices: torch.Tensor,
) -> float:
    """
    计算单个 token 对视觉 token 的注意力

    Args:
        token_idx: token 索引
        attentions: attention 元组
        visual_token_indices: 视觉 token 索引

    Returns:
        视觉注意力值
    """
    # 最后一层的 attention
    attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
    num_heads = attn.shape[0]

    # 获取该 token 的 attention
    attn_t = attn[:, token_idx, :]  # (num_heads, seq_len)

    # 对视觉 token 求和
    if len(visual_token_indices) > 0:
        visual_attn_sum = attn_t[:, visual_token_indices].sum(dim=-1).mean().item()
    else:
        visual_attn_sum = 0.0

    return visual_attn_sum


def validate_attention_normalization(attentions: Tuple[torch.Tensor], sample_idx: int = 0) -> bool:
    """
    验证 attention 行和为 1 (检查点1)

    Args:
        attentions: attention 元组
        sample_idx: 样本索引

    Returns:
        是否归一化
    """
    attn = attentions[-1][sample_idx]  # (num_heads, seq_len, seq_len)

    # 对最后一个维度求和
    row_sums = attn.sum(dim=-1)  # (num_heads, seq_len)

    # 检查是否接近 1
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


if __name__ == "__main__":
    # 测试
    print("HEVA module loaded successfully")
    print("Functions available:")
    print("  - compute_entropy")
    print("  - compute_heva")
    print("  - compute_heva_from_result")
    print("  - compute_token_visual_attention")
    print("  - validate_attention_normalization")
