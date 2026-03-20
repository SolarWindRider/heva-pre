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





def compute_heva(
    logits: torch.Tensor,
    attentions: Tuple[torch.Tensor],
    visual_token_indices: torch.Tensor,
    gen_token_indices: torch.Tensor,
    alpha: float = 0.2,
    use_last_layer_only: bool = True,
) -> Dict[str, Any]:
    """
    计算 HEVA 指标

    Args:
        logits: (seq_len, vocab_size) 模型输出的 logits
        attentions: (num_layers, batch, num_heads, seq_len, seq_len) attention 元组
        visual_token_indices: 视觉 token 的索引
        gen_token_indices: 生成 token 的索引
        alpha: 高熵 token 的比例
        use_last_layer_only: 是否只使用最后一层 attention（节省显存）

    Returns:
        {
            "heva": float,
            "high_entropy_tokens": torch.Tensor,
            "visual_attentions": list,
            "entropies": torch.Tensor
        }
    """


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

    # 4. 获取最后一层的 attention (使用 bfloat16 减少显存)
    # attentions 是 Tuple[ (batch, num_heads, seq_len, seq_len) ]
    attn = attentions[-1][0]  # (num_heads, seq_len, seq_len) - 保持原 dtype
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]

    # 5. 批量计算所有高熵 token 对视觉 token 的注意力（避免循环）
    if len(visual_token_indices) > 0:
        # 将 visual_token_indices 转为掩码
        visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn.device)
        visual_mask[visual_token_indices] = True

        # 批量获取所有 selected_tokens 的注意力
        # attn[:, selected_tokens, :] 形状: (num_heads, k, seq_len)
        attn_selected = attn[:, selected_tokens, :]  # (num_heads, k, seq_len)

        # 对视觉 token 求和 (使用掩码)
        # 需要将 visual_mask 扩展到兼容维度
        visual_mask_expanded = visual_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len)
        attn_selected_masked = attn_selected.masked_fill(~visual_mask_expanded, 0)

        # 对 seq_len 维度求和 -> (num_heads, k)
        visual_attn_sum = attn_selected_masked.sum(dim=-1)

        # 对 head 求平均 -> (k,)
        v_t = visual_attn_sum.mean(dim=0)  # (k,)

        visual_attentions = v_t.tolist()
    else:
        visual_attentions = [0.0] * len(selected_tokens)

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
        "selected_entropy_values": entropy_gen[top_indices].float().cpu().numpy()
    }



