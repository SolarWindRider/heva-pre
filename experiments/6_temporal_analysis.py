"""
时间动态分析脚本

分析 HEVA 和 entropy 在生成过程中的时间动态
"""

import argparse
import json
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset
from models.inference import load_model
from metrics.heva import compute_entropy, compute_token_visual_attention


def run_temporal_analysis(
    model,
    dataset,
    sample_indices,
    output_path,
    alpha=0.2,
    max_new_tokens=128
):
    """
    运行时间动态分析

    Args:
        model: 推理模型
        dataset: 数据集
        sample_indices: 样本索引
        output_path: 输出路径
        alpha: 高熵 token 比例
        max_new_tokens: 最大生成长度
    """
    all_entropies = []
    all_visual_attentions = []
    all_high_entropy_attentions = []
    results = []

    for idx in tqdm(sample_indices, desc="Running temporal analysis"):
        try:
            sample = dataset[idx]
            question_with_options = sample['question'] + "\n" + sample['options']

            # 运行推理
            result = model.generate_with_attention(
                image=sample['image'],
                question=question_with_options,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )

            # 提取数据
            logits = result['logits']
            attentions = result['attentions']
            visual_token_indices = result['visual_token_indices']
            prompt_length = result['prompt_length']
            input_ids = result['input_ids']

            # 计算 entropy
            entropies = compute_entropy(logits)  # (seq_len,)

            # 生成 token 的 entropy
            gen_token_indices = torch.arange(prompt_length, len(input_ids))

            # 计算每个生成 token 的视觉注意力
            visual_attns = []
            for t in gen_token_indices:
                v_attn = compute_token_visual_attention(t, attentions, visual_token_indices)
                visual_attns.append(v_attn)

            visual_attns = np.array(visual_attns)
            entropies_gen = entropies[prompt_length:].cpu().numpy()

            # 选取高熵 token
            k = max(1, int(len(entropies_gen) * alpha))
            top_k_indices = np.argsort(entropies_gen)[-k:]

            high_entropy_attns = visual_attns[top_k_indices]

            # 保存
            all_entropies.append(entropies_gen)
            all_visual_attentions.append(visual_attns)
            all_high_entropy_attentions.append(high_entropy_attns)

            results.append({
                'sample_id': sample['id'],
                'idx': idx,
                'entropies': entropies_gen,
                'visual_attentions': visual_attns,
                'high_entropy_attentions': high_entropy_attns,
                'num_gen_tokens': len(entropies_gen),
            })

        except Exception as e:
            print(f"Error at idx {idx}: {e}")

    # 计算对齐的平均曲线
    max_len = max(len(e) for e in all_entropies)

    # Pad 到相同长度
    entropies_padded = []
    visual_attns_padded = []

    for ent, va in zip(all_entropies, all_visual_attentions):
        if len(ent) < max_len:
            ent = np.pad(ent, (0, max_len - len(ent)), mode='constant', constant_values=np.nan)
            va = np.pad(va, (0, max_len - len(va)), mode='constant', constant_values=np.nan)
        entropies_padded.append(ent)
        visual_attns_padded.append(va)

    entropies_array = np.array(entropies_padded)
    visual_attns_array = np.array(visual_attns_padded)

    # 计算对齐后的平均值
    mean_entropy = np.nanmean(entropies_array, axis=0)
    mean_visual_attn = np.nanmean(visual_attns_array, axis=0)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'mean_entropy': mean_entropy,
            'mean_visual_attention': mean_visual_attn,
            'all_entropies': all_entropies,
            'all_visual_attentions': all_visual_attentions,
            'all_high_entropy_attentions': all_high_entropy_attentions,
        }, f)

    print(f"Saved temporal analysis to {output_path}")

    # 计算相关性
    # 对齐到答案 token 的位置
    # 简化：取最后一个 token 的 entropy 和 visual attention
    last_entropy = np.array([r['entropies'][-1] for r in results if len(r['entropies']) > 0])
    last_visual_attn = np.array([r['visual_attentions'][-1] for r in results if len(r['visual_attentions']) > 0])

    if len(last_entropy) > 1:
        corr = np.corrcoef(last_entropy, last_visual_attn)[0, 1]
        print(f"Correlation between last token entropy and visual attention: {corr:.4f}")

    return results


def main():
    import config
    parser = argparse.ArgumentParser(description='Run temporal analysis')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of samples')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--save_path', type=str, default=os.path.join(config.RESULTS_DIR, 'temporal_analysis.pkl'),
                        help='Output path')
    parser.add_argument('--alpha', type=float, default=0.2, help='High entropy ratio')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens')

    args = parser.parse_args()

    # 加载模型和数据
    print("Loading model...")
    model = load_model()

    print("Loading dataset...")
    dataset = load_dataset()

    # 确定样本索引
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    sample_indices = list(range(args.start_idx, end_idx))

    print(f"Processing samples {args.start_idx} to {end_idx}")

    # 运行分析
    run_temporal_analysis(
        model, dataset, sample_indices,
        args.save_path, args.alpha, args.max_new_tokens
    )


if __name__ == "__main__":
    main()
