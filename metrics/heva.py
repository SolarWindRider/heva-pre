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
import sys

# 立即刷新输出的打印函数
def log_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算每个 token 的 entropy

    Args:
        logits: (seq_len, vocab_size) 或 (batch, seq_len, vocab_size)

    Returns:
        entropy: (seq_len,) 或 (batch, seq_len)
    """
    # 转换为 float 以避免 bfloat16 问题
    logits = logits.to(torch.float32)

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


def compute_entropy_chunked(
    logits: torch.Tensor,
    chunk_size: int = 256,
    move_to_cpu: bool = True,
) -> torch.Tensor:
    """
    分块计算 entropy (时间换空间，适用于显存不足的情况)

    Args:
        logits: (seq_len, vocab_size)
        chunk_size: 每次处理的 token 数量
        move_to_cpu: 是否将结果移到CPU以释放显存 (默认True)

    Returns:
        entropy: (seq_len,)
    """
    seq_len = logits.shape[0]
    entropies = []

    # 保持原始设备
    target_device = logits.device if not move_to_cpu else 'cpu'

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logits = logits[start:end, :].to(torch.float32)

        probs = F.softmax(chunk_logits, dim=-1)
        chunk_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        # 根据设置决定是否移到CPU
        if move_to_cpu:
            entropies.append(chunk_entropy.cpu())
        else:
            entropies.append(chunk_entropy.to(target_device))
        del chunk_logits, probs, chunk_entropy

    return torch.cat(entropies, dim=0)


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


def compute_heva_from_result(result: Dict[str, Any], alpha: float = 0.2) -> Dict[str, Any]:
    """
    从推理结果计算 HEVA

    Args:
        result: generate_with_attention 返回的结果
        alpha: 高熵 token 比例

    Returns:
        HEVA 计算结果
    """
    import config
    log_print(f"[HEVA] config imported, CHUNK_SIZE={config.CHUNK_SIZE}, USE_LAST_LAYER_ONLY={config.USE_LAST_LAYER_ONLY}")

    # 获取参数
    logits = result["logits"]
    attentions = result["attentions"]
    visual_token_indices = result["visual_token_indices"]

    # 打印调试信息
    log_print(f"[HEVA] logits: {logits.shape if logits is not None else None}")
    log_print(f"[HEVA] attentions: {len(attentions) if attentions else None} layers")
    log_print(f"[HEVA] visual_token_indices: len={len(visual_token_indices) if visual_token_indices is not None else 0}")

    # 计算生成 token 索引
    prompt_length = result["prompt_length"]
    input_ids = result["input_ids"]

    log_print(f"[HEVA] prompt_length: {prompt_length}")
    log_print(f"[HEVA] input_ids length: {len(input_ids) if hasattr(input_ids, '__len__') else 'N/A'}")

    # 获取序列长度 (优先使用 logits 的长度，因为它是完整序列)
    seq_len = logits.shape[0] if hasattr(logits, 'shape') and logits is not None else (
        input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
    )
    log_print(f"[HEVA] seq_len (determined): {seq_len}")
    log_print(f"[HEVA] logits device: {logits.device}")
    log_print(f"[HEVA] attentions[0] device: {attentions[0].device if attentions else 'None'}")

    # 生成 token 从 prompt_length 开始，到序列结束
    # 确保 prompt_length 不超过 seq_len
    if prompt_length < seq_len:
        gen_token_indices = torch.arange(prompt_length, seq_len, device=logits.device if logits is not None else 'cpu')
    else:
        gen_token_indices = torch.tensor([], dtype=torch.long, device=logits.device if logits is not None else 'cpu')

    log_print(f"[HEVA] gen_token_indices device: {gen_token_indices.device}, len={len(gen_token_indices)}")

    # 检查 visual_token_indices 是否在有效范围内
    log_print(f"[HEVA] visual_token_indices device: {visual_token_indices.device if visual_token_indices is not None else 'None'}")
    if visual_token_indices is not None and len(visual_token_indices) > 0:
        max_visual_idx = visual_token_indices.max().item() if len(visual_token_indices) > 0 else 0
        log_print(f"[HEVA] max visual_token_idx: {max_visual_idx}, seq_len: {seq_len}")
        if max_visual_idx >= seq_len:
            log_print(f"[ERROR] visual_token_indices contains indices >= seq_len! This will cause all-zero attention.")

    log_print(f"[HEVA] config.CHUNK_SIZE: {config.CHUNK_SIZE}, config.USE_LAST_LAYER_ONLY: {config.USE_LAST_LAYER_ONLY}")

    # 如果序列很长，使用分块计算
    if seq_len > config.CHUNK_SIZE and config.USE_LAST_LAYER_ONLY:
        log_print(f"[HEVA] Calling compute_heva_chunked (seq_len={seq_len} > CHUNK_SIZE={config.CHUNK_SIZE})")
        return compute_heva_chunked(
            logits=logits,
            attentions=attentions,
            visual_token_indices=visual_token_indices,
            gen_token_indices=gen_token_indices,
            alpha=alpha,
            chunk_size=config.CHUNK_SIZE,
            move_entropy_to_cpu=False  # Keep on NPU device for async computation
        )

    return compute_heva(
        logits=logits,
        attentions=attentions,
        visual_token_indices=visual_token_indices,
        gen_token_indices=gen_token_indices,
        alpha=alpha,
        use_last_layer_only=config.USE_LAST_LAYER_ONLY
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
    if attentions is None or len(attentions) == 0:
        return False

    attn = attentions[-1][sample_idx]  # (num_heads, seq_len, seq_len)

    # 对最后一个维度求和
    row_sums = attn.sum(dim=-1)  # (num_heads, seq_len)

    # 检查是否接近 1
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def compute_heva_chunked(
    logits: torch.Tensor,
    attentions: Tuple[torch.Tensor],
    visual_token_indices: torch.Tensor,
    gen_token_indices: torch.Tensor,
    alpha: float = 0.2,
    chunk_size: int = 512,
    use_chunked_entropy: bool = True,
    move_entropy_to_cpu: bool = True,
) -> Dict[str, Any]:
    """
    分块计算 HEVA（适用于超长序列，节省显存）

    当序列很长时，将高熵 token 分批处理，每批处理 chunk_size 个 token，
    避免一次性加载所有 attention 数据导致显存溢出。

    Args:
        logits: (seq_len, vocab_size) 模型输出的 logits
        attentions: (num_layers, batch, num_heads, seq_len, seq_len) attention 元组
        visual_token_indices: 视觉 token 的索引
        gen_token_indices: 生成 token 的索引
        alpha: 高熵 token 的比例
        chunk_size: 每批处理的 token 数量
        use_chunked_entropy: 是否使用分块计算 entropy (时间换空间)

    Returns:
        HEVA 计算结果
    """
    log_print(f"[HEVA chunked] Entering function...")
    log_print(f"[HEVA chunked] logits device: {logits.device}")
    log_print(f"[HEVA chunked] attentions[0] device: {attentions[0].device if attentions else 'None'}")
    log_print(f"[HEVA chunked] visual_token_indices device: {visual_token_indices.device}")
    log_print(f"[HEVA chunked] gen_token_indices device: {gen_token_indices.device}")
    log_print(f"[HEVA chunked] use_chunked_entropy: {use_chunked_entropy}")

    # 1. 计算所有 token 的 entropy
    if use_chunked_entropy:
        # 使用分块版本节省显存 (时间换空间)
        log_print(f"[HEVA chunked] Calling compute_entropy_chunked with move_to_cpu={move_entropy_to_cpu}...")
        entropies = compute_entropy_chunked(logits, chunk_size=chunk_size, move_to_cpu=move_entropy_to_cpu)
        log_print(f"[HEVA chunked] compute_entropy_chunked done, entropies device: {entropies.device}")
    else:
        entropies = compute_entropy(logits)

    # 计算完熵后立即释放logits内存 (NPU memory optimization)
    del logits

    if len(gen_token_indices) == 0:
        return {
            "heva": 0.0,
            "high_entropy_tokens": torch.tensor([], dtype=torch.long),
            "visual_attentions": [],
            "entropies": entropies
        }

    entropy_gen = entropies[gen_token_indices]

    # 2. 选取 top-alpha 高熵 token
    k = max(1, int(len(entropy_gen) * alpha))
    if k >= len(entropy_gen):
        k = len(entropy_gen) - 1

    _, top_indices = torch.topk(entropy_gen, k)
    selected_tokens = gen_token_indices[top_indices]

    # 3. 获取最后一层 attention
    attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]

    # 4. 分块计算
    visual_attentions = []

    if len(visual_token_indices) > 0:
        visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn.device)
        visual_mask[visual_token_indices] = True

        # 分批处理
        for start in range(0, len(selected_tokens), chunk_size):
            end = min(start + chunk_size, len(selected_tokens))
            chunk_tokens = selected_tokens[start:end]

            # 获取当前 chunk 的 attention
            attn_chunk = attn[:, chunk_tokens, :]  # (num_heads, chunk_size, seq_len)

            # 扩展掩码
            visual_mask_expanded = visual_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len)
            attn_chunk_masked = attn_chunk.masked_fill(~visual_mask_expanded, 0)

            # 求和
            visual_attn_sum = attn_chunk_masked.sum(dim=-1)  # (num_heads, chunk_size)
            v_t = visual_attn_sum.mean(dim=0)  # (chunk_size,)

            visual_attentions.extend(v_t.tolist())

            # 释放中间变量
            del attn_chunk, attn_chunk_masked, visual_attn_sum
    else:
        visual_attentions = [0.0] * len(selected_tokens)

    # 5. 计算 HEVA
    heva = np.mean(visual_attentions) if visual_attentions else 0.0

    # 释放attentions内存 (NPU memory optimization)
    del attentions

    return {
        "heva": heva,
        "high_entropy_tokens": selected_tokens,
        "visual_attentions": visual_attentions,
        "entropies": entropies,
        "selected_entropy_values": entropy_gen[top_indices].float().cpu().numpy()
    }


if __name__ == "__main__":
    # 测试
    log_print("HEVA module loaded successfully")
    log_print("Functions available:")
    log_print("  - compute_entropy")
    log_print("  - compute_heva")
    log_print("  - compute_heva_chunked (分块计算，适用于超长序列)")
    log_print("  - compute_heva_from_result")
    log_print("  - compute_token_visual_attention")
    log_print("  - validate_attention_normalization")
