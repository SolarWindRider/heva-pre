"""
时间动态分析脚本

分析 HEVA 和 entropy 在生成过程中的时间动态

新格式：
- results/{exp_name}/{dataset}/
  - {sample_id}_meta.json (人类可读元数据)
  - {sample_id}_tensor.pkl (tensor数据)
"""

import argparse
import json
import os
import pickle
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset, SUPPORTED_DATASETS
from models.inference import load_model
from metrics.heva import compute_entropy, compute_token_visual_attention


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


def run_temporal_analysis(
    model,
    dataset,
    sample_indices,
    output_dir,
    alpha=0.2,
    max_new_tokens=8192,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    image_size=448,
):
    """
    运行时间动态分析

    Args:
        model: 推理模型
        dataset: 数据集
        sample_indices: 样本索引
        output_dir: 输出目录
        alpha: 高熵 token 比例
        max_new_tokens: 最大生成长度
    """
    os.makedirs(output_dir, exist_ok=True)

    all_entropies = []
    all_visual_attentions = []
    all_high_entropy_attentions = []
    results = []

    for idx in tqdm(sample_indices, desc="Running temporal analysis"):
        try:
            sample = dataset[idx]

            # 运行推理
            result = model.generate_with_attention(
                image=sample['image'],
                question=sample['question'],
                options=sample['options'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=image_size,
            )

            # 检查结果是否有效
            if result is None or result.get('logits') is None:
                print(f"Warning: Failed to get valid result for idx {idx}, skipping...")
                continue

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

            sample_id = str(sample['id']).replace('/', '_').replace('\\', '_').replace(':', '_')

            results.append({
                'sample_id': sample_id,
                'idx': idx,
                'entropies': entropies_gen,
                'visual_attentions': visual_attns,
                'high_entropy_attentions': high_entropy_attns,
                'num_gen_tokens': len(entropies_gen),
            })

        except Exception as e:
            print(f"Error at idx {idx}: {e}")

    # 计算对齐的平均曲线
    max_len = max(len(e) for e in all_entropies) if all_entropies else 0

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

    # 保存索引文件
    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'results': results,
            'mean_entropy': mean_entropy.tolist() if len(mean_entropy) > 0 else [],
            'mean_visual_attention': mean_visual_attn.tolist() if len(mean_visual_attn) > 0 else [],
        }, f, indent=2)

    print(f"Saved temporal analysis to {output_dir}")

    # 计算相关性
    last_entropy = np.array([r['entropies'][-1] for r in results if len(r['entropies']) > 0])
    last_visual_attn = np.array([r['visual_attentions'][-1] for r in results if len(r['visual_attentions']) > 0])

    if len(last_entropy) > 1:
        corr = np.corrcoef(last_entropy, last_visual_attn)[0, 1]
        print(f"Correlation between last token entropy and visual attention: {corr:.4f}")

    return results


def main():
    import config

    parser = argparse.ArgumentParser(description='Run temporal analysis')

    # 实验配置
    parser.add_argument('--exp_name', type=str, default=config.DEFAULT_EXP_NAME,
                       help='Experiment name')
    parser.add_argument('--dataset', type=str, default='VisuRiddles',
                       choices=SUPPORTED_DATASETS, help='Dataset name')

    # 采样配置
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples (-1 for all)')
    parser.add_argument('--shuffle', type=str, default='false', choices=['true', 'false'], help='Shuffle dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # 模型超参数
    parser.add_argument('--max_new_tokens', type=int, default=8192, help='Max new tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k')
    parser.add_argument('--image_size', type=int, default=448, help='Image size')

    # 分析参数
    parser.add_argument('--alpha', type=float, default=0.2, help='High entropy ratio')

    # 模型配置
    parser.add_argument('--model_path', type=str, default=None, help='Model path')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置输出目录
    args.output_dir = os.path.join(config.RESULTS_DIR, args.exp_name, args.dataset, "temporal_analysis")

    # 保存实验配置
    model_path = args.model_path if args.model_path else config.MODEL_DIR
    exp_config = {
        'exp_name': args.exp_name,
        'dataset': args.dataset,
        'model_name': model_path.split("/")[-1],
        'model_path': model_path,
        'num_samples': args.num_samples,
        'shuffle': args.shuffle,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'alpha': args.alpha,
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

    # 加载模型和数据
    print("Loading model...")
    model = load_model(model_path=args.model_path)

    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Dataset size: {len(dataset)}")

    # 确定样本索引
    if args.num_samples == -1:
        end_idx = len(dataset)
    else:
        end_idx = min(args.num_samples, len(dataset))
    sample_indices = list(range(0, end_idx))

    if args.shuffle.lower() == 'true':
        import random
        random.seed(42)
        random.shuffle(sample_indices)

    print(f"Processing samples 0 to {end_idx} ({len(sample_indices)} samples)")
    print(f"Output directory: {args.output_dir}")

    # 运行分析
    run_temporal_analysis(
        model, dataset, sample_indices,
        args.output_dir,
        args.alpha,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.image_size,
    )


if __name__ == "__main__":
    main()
