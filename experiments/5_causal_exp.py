"""
因果敏感性实验

对高 HEVA 和低 HEVA 样本进行 masked 实验

新格式：
- results/{exp_name}/{dataset}/
  - {sample_id}_meta.json (人类可读元数据)
  - {sample_id}_tensor.pkl (tensor数据)
"""

import argparse
import json
import os
import torch
import pickle
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.nn.functional import kl_div, softmax, log_softmax
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset, SUPPORTED_DATASETS
from models.inference import load_model
from metrics.heva import compute_heva_from_result, validate_attention_normalization, compute_entropy


# 异步保存tensor的线程池
_executor = ThreadPoolExecutor(max_workers=4)


def save_tensor_async(tensor_data, filepath):
    """异步保存tensor数据"""
    def _save():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(tensor_data, f)
    _executor.submit(_save)


def set_seed(seed: int):
    """设置全局随机种子以保证可复现性"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


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


def load_inference_results(exp_name, dataset):
    """加载推理结果"""
    import config

    # 尝试从新格式加载
    index_path = os.path.join(config.RESULTS_DIR, exp_name, dataset, 'index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        return index_data.get('results', [])

    # 尝试从旧格式加载
    old_path = os.path.join(config.RESULTS_DIR, f'{dataset}_results.pkl')
    if os.path.exists(old_path):
        with open(old_path, 'rb') as f:
            data = pickle.load(f)
        return data.get('results', [])

    raise FileNotFoundError(f"No inference results found for {dataset}")


def run_causal_exp(
    model,
    dataset,
    heva_results,
    output_dir,
    max_new_tokens=8192,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    image_size=448,
    top_percent=0.2,
    bottom_percent=0.2,
):
    """
    运行因果敏感性实验

    Args:
        model: 推理模型
        dataset: 数据集
        heva_results: 包含 HEVA 值的结果
        output_dir: 输出目录
        max_new_tokens: 最大生成长度
        top_percent: 高 HEVA 样本比例
        bottom_percent: 低 HEVA 样本比例

    Returns:
        实验结果
    """
    # 排序选取 top 和 bottom
    sorted_results = sorted(heva_results, key=lambda x: x.get('heva', 0), reverse=True)

    n = len(sorted_results)
    top_n = max(1, int(n * top_percent))
    bottom_n = max(1, int(n * bottom_percent))

    top_samples = sorted_results[:top_n]
    bottom_samples = sorted_results[-bottom_n:]

    print(f"Top {top_n} samples (high HEVA)")
    print(f"Bottom {bottom_n} samples (low HEVA)")

    os.makedirs(output_dir, exist_ok=True)

    results = []

    # 对 top 样本运行实验
    for sample_info in tqdm(top_samples, desc="Running top samples"):
        idx = sample_info.get('idx')
        if idx is None:
            continue

        try:
            sample = dataset[idx]

            # 原图推理
            result_original = model.generate_with_attention(
                image=sample['image'],
                question=sample['question'],
                options=sample['options'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=image_size,
            )

            # masked 推理（用空白图）
            result_masked = model.generate_with_attention(
                image=Image.new('RGB', (448, 448), color='black'),
                question=sample['question'],
                options=sample['options'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=image_size,
            )

            # 计算 KL 散度
            if result_original.get('logits') is not None and result_masked.get('logits') is not None:
                prompt_len = result_original['prompt_length']
                logits_orig = result_original['logits'][prompt_len:]
                logits_mask = result_masked['logits'][prompt_len:]

                min_len = min(len(logits_orig), len(logits_mask))
                logits_orig = logits_orig[:min_len]
                logits_mask = logits_mask[:min_len]

                kl = compute_kl_divergence(logits_orig, logits_mask)
            else:
                kl = None

            sample_id = str(sample['id']).replace('/', '_').replace('\\', '_').replace(':', '_')

            results.append({
                'sample_id': sample_id,
                'idx': idx,
                'group': 'high_heva',
                'heva': sample_info.get('heva', 0),
                'kl_divergence': kl,
            })

        except Exception as e:
            print(f"Error at idx {idx}: {e}")

    # 对 bottom 样本运行实验
    for sample_info in tqdm(bottom_samples, desc="Running bottom samples"):
        idx = sample_info.get('idx')
        if idx is None:
            continue

        try:
            sample = dataset[idx]

            # 原图推理
            result_original = model.generate_with_attention(
                image=sample['image'],
                question=sample['question'],
                options=sample['options'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=image_size,
            )

            # masked 推理
            result_masked = model.generate_with_attention(
                image=Image.new('RGB', (448, 448), color='black'),
                question=sample['question'],
                options=sample['options'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=image_size,
            )

            # 计算 KL 散度
            if result_original.get('logits') is not None and result_masked.get('logits') is not None:
                prompt_len = result_original['prompt_length']
                logits_orig = result_original['logits'][prompt_len:]
                logits_mask = result_masked['logits'][prompt_len:]

                min_len = min(len(logits_orig), len(logits_mask))
                logits_orig = logits_orig[:min_len]
                logits_mask = logits_mask[:min_len]

                kl = compute_kl_divergence(logits_orig, logits_mask)
            else:
                kl = None

            sample_id = str(sample['id']).replace('/', '_').replace('\\', '_').replace(':', '_')

            results.append({
                'sample_id': sample_id,
                'idx': idx,
                'group': 'low_heva',
                'heva': sample_info.get('heva', 0),
                'kl_divergence': kl,
            })

        except Exception as e:
            print(f"Error at idx {idx}: {e}")

    # 保存索引文件
    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'results': results,
            'top_samples': top_samples,
            'bottom_samples': bottom_samples,
        }, f, indent=2)

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

    print(f"\nSaved causal results to {output_dir}")

    return results


def main():
    import config

    parser = argparse.ArgumentParser(description='Run causal experiment')

    # 实验配置
    parser.add_argument('--exp_name', type=str, default=config.DEFAULT_EXP_NAME,
                       help='Experiment name')
    parser.add_argument('--dataset', type=str, default='VisuRiddles',
                       choices=SUPPORTED_DATASETS, help='Dataset name')

    # 采样配置
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # 模型超参数
    parser.add_argument('--max_new_tokens', type=int, default=8192, help='Max new tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k')
    parser.add_argument('--image_size', type=int, default=448, help='Image size')

    # 因果实验参数
    parser.add_argument('--top_percent', type=float, default=0.2, help='Top HEVA percent')
    parser.add_argument('--bottom_percent', type=float, default=0.2, help='Bottom HEVA percent')

    # 模型配置
    parser.add_argument('--model_path', type=str, default=None, help='Model path')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置输出目录
    args.output_dir = os.path.join(config.RESULTS_DIR, args.exp_name, args.dataset, "causal_exp")

    # 保存实验配置
    model_path = args.model_path if args.model_path else config.MODEL_DIR
    exp_config = {
        'exp_name': args.exp_name,
        'dataset': args.dataset,
        'model_name': model_path.split("/")[-1],
        'model_path': model_path,
        'top_percent': args.top_percent,
        'bottom_percent': args.bottom_percent,
        'seed': args.seed,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'image_size': args.image_size,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {config_path}")

    # 加载推理结果
    print(f"Loading inference results from {args.exp_name}/{args.dataset}...")
    heva_results = load_inference_results(args.exp_name, args.dataset)
    print(f"Loaded {len(heva_results)} inference results")

    # 加载模型和数据
    print("Loading model...")
    model = load_model(model_path=args.model_path)

    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Dataset size: {len(dataset)}")

    print(f"Output directory: {args.output_dir}")

    # 运行因果实验
    run_causal_exp(
        model, dataset, heva_results,
        args.output_dir,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.image_size,
        args.top_percent,
        args.bottom_percent,
    )


if __name__ == "__main__":
    main()
