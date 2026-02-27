"""
因果敏感性实验

对高 HEVA 和低 HEVA 样本进行 masked 实验
"""

import argparse
import json
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.nn.functional import kl_div, softmax, log_softmax
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset
from models.inference import load_model
from metrics.heva import compute_heva_from_result, validate_attention_normalization, compute_entropy


def compute_kl_divergence(logits1, logits2):
    """
    计算两个分布之间的 KL 散度

    Args:
        logits1: 第一个分布的 logits
        logits2: 第二个分布的 logits

    Returns:
        KL 散度
    """
    log_probs1 = log_softmax(logits1, dim=-1)
    probs2 = softmax(logits2, dim=-1)

    kl = kl_div(log_probs1, probs2, reduction='batchmean')

    return kl.item()


def run_causal_exp(
    model,
    dataset,
    heva_results,
    output_path,
    max_new_tokens=128,
    top_percent=0.2,
    bottom_percent=0.2
):
    """
    运行因果敏感性实验

    Args:
        model: 推理模型
        dataset: 数据集
        heva_results: 包含 HEVA 值的结果
        output_path: 输出路径
        max_new_tokens: 最大生成长度
        top_percent: 高 HEVA 样本比例
        bottom_percent: 低 HEVA 样本比例

    Returns:
        实验结果
    """
    # 排序选取 top 和 bottom
    sorted_results = sorted(heva_results, key=lambda x: x['heva'], reverse=True)

    n = len(sorted_results)
    top_n = max(1, int(n * top_percent))
    bottom_n = max(1, int(n * bottom_percent))

    top_samples = sorted_results[:top_n]
    bottom_samples = sorted_results[-bottom_n:]

    print(f"Top {top_n} samples (high HEVA): {[r['sample_id'] for r in top_samples]}")
    print(f"Bottom {bottom_n} samples (low HEVA): {[r['sample_id'] for r in bottom_samples]}")

    results = []

    # 对 top 样本运行实验
    for sample_info in tqdm(top_samples, desc="Running top samples"):
        idx = sample_info['idx']

        try:
            sample = dataset[idx]
            question_with_options = sample['question'] + "\n" + sample['options']

            # 原图推理
            result_original = model.generate_with_attention(
                image=sample['image'],
                question=question_with_options,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )

            # masked 推理（用空白图）
            result_masked = model.generate_with_attention(
                image=Image.new('RGB', (448, 448), color='black'),  # 零图像
                question=question_with_options,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )

            # 计算 KL 散度
            if result_original['logits'] is not None and result_masked['logits'] is not None:
                # 只比较生成部分
                prompt_len = result_original['prompt_length']
                logits_orig = result_original['logits'][prompt_len:]
                logits_mask = result_masked['logits'][prompt_len:]

                # 对齐长度
                min_len = min(len(logits_orig), len(logits_mask))
                logits_orig = logits_orig[:min_len]
                logits_mask = logits_mask[:min_len]

                kl = compute_kl_divergence(logits_orig, logits_mask)
            else:
                kl = None

            results.append({
                'sample_id': sample_info['sample_id'],
                'idx': idx,
                'group': 'high_heva',
                'heva': sample_info['heva'],
                'original_correct': sample_info.get('correct', False),
                'masked_correct': None,  # 需要单独计算
                'kl_divergence': kl,
            })

        except Exception as e:
            print(f"Error at idx {idx}: {e}")

    # 对 bottom 样本运行实验
    for sample_info in tqdm(bottom_samples, desc="Running bottom samples"):
        idx = sample_info['idx']

        try:
            sample = dataset[idx]
            question_with_options = sample['question'] + "\n" + sample['options']

            # 原图推理
            result_original = model.generate_with_attention(
                image=sample['image'],
                question=question_with_options,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )

            # masked 推理
            result_masked = model.generate_with_attention(
                image=Image.new('RGB', (448, 448), color='black'),
                question=question_with_options,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )

            # 计算 KL 散度
            if result_original['logits'] is not None and result_masked['logits'] is not None:
                prompt_len = result_original['prompt_length']
                logits_orig = result_original['logits'][prompt_len:]
                logits_mask = result_masked['logits'][prompt_len:]

                min_len = min(len(logits_orig), len(logits_mask))
                logits_orig = logits_orig[:min_len]
                logits_mask = logits_mask[:min_len]

                kl = compute_kl_divergence(logits_orig, logits_mask)
            else:
                kl = None

            results.append({
                'sample_id': sample_info['sample_id'],
                'idx': idx,
                'group': 'low_heva',
                'heva': sample_info['heva'],
                'original_correct': sample_info.get('correct', False),
                'masked_correct': None,
                'kl_divergence': kl,
            })

        except Exception as e:
            print(f"Error at idx {idx}: {e}")

    # 统计
    high_heva_results = [r for r in results if r['group'] == 'high_heva']
    low_heva_results = [r for r in results if r['group'] == 'low_heva']

    print("\n=== Causal Experiment Results ===")
    print(f"High HEVA group:")
    if high_heva_results:
        kls = [r['kl_divergence'] for r in high_heva_results if r['kl_divergence'] is not None]
        if kls:
            print(f"  Mean KL: {np.mean(kls):.4f}")

    print(f"Low HEVA group:")
    if low_heva_results:
        kls = [r['kl_divergence'] for r in low_heva_results if r['kl_divergence'] is not None]
        if kls:
            print(f"  Mean KL: {np.mean(kls):.4f}")

    # 比较
    if high_heva_results and low_heva_results:
        high_kls = [r['kl_divergence'] for r in high_heva_results if r['kl_divergence'] is not None]
        low_kls = [r['kl_divergence'] for r in low_heva_results if r['kl_divergence'] is not None]

        if high_kls and low_kls:
            cliff = compute_cliff_delta(np.array(high_kls), np.array(low_kls))
            print(f"\nCliff's delta: {cliff:.4f}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'top_samples': top_samples,
            'bottom_samples': bottom_samples,
        }, f)

    print(f"\nSaved causal results to {output_path}")

    return results


def compute_cliff_delta(group1, group2):
    """Cliff's delta"""
    n1, n2 = len(group1), len(group2)
    dominance = 0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    return dominance / (n1 * n2)


def main():
    import config
    parser = argparse.ArgumentParser(description='Run causal experiment')
    parser.add_argument('--inference_results', type=str,
                        default=os.path.join(config.RESULTS_DIR, 'inference_results.pkl'),
                        help='Inference results path')
    parser.add_argument('--save_path', type=str, default=os.path.join(config.RESULTS_DIR, 'causal_exp.pkl'),
                        help='Output path')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens')
    parser.add_argument('--top_percent', type=float, default=0.2, help='Top HEVA percent')
    parser.add_argument('--bottom_percent', type=float, default=0.2, help='Bottom HEVA percent')

    args = parser.parse_args()

    # 加载推理结果
    print("Loading inference results...")
    with open(args.inference_results, 'rb') as f:
        inference_data = pickle.load(f)

    heva_results = inference_data['results']

    # 加载模型和数据
    print("Loading model...")
    model = load_model()

    print("Loading dataset...")
    dataset = load_dataset()

    # 运行因果实验
    run_causal_exp(
        model, dataset, heva_results,
        args.save_path, args.max_new_tokens,
        args.top_percent, args.bottom_percent
    )


if __name__ == "__main__":
    main()
