"""
扰动实验

验证 HEVA(original) > HEVA(perturbed)

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
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset, SUPPORTED_DATASETS
from data.perturbations import get_perturbation_functions
from models.inference import load_model
from metrics.heva import compute_heva_from_result, validate_attention_normalization


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


def run_perturbation_exp(
    model,
    dataset,
    sample_indices,
    output_dir,
    perturbation_name,
    max_new_tokens=8192,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    image_size=448,
):
    """
    运行扰动实验

    Args:
        model: 推理模型
        dataset: 数据集
        sample_indices: 样本索引
        output_dir: 输出目录
        perturbation_name: 扰动名称
        max_new_tokens: 最大生成长度
        temperature: 温度
        top_p: top-p 采样
        top_k: top-k 采样
        image_size: 图像大小
    """
    perturbations = get_perturbation_functions()
    perturb_func = perturbations.get(perturbation_name)

    if perturb_func is None:
        raise ValueError(f"Unknown perturbation: {perturbation_name}. Available: {list(perturbations.keys())}")

    os.makedirs(output_dir, exist_ok=True)

    results = []
    errors = []

    for idx in tqdm(sample_indices, desc=f"Running {perturbation_name}"):
        try:
            sample = dataset[idx]

            # 应用扰动
            perturbed_image = perturb_func(sample['image'])

            # 运行推理
            result = model.generate_with_attention(
                image=perturbed_image,
                question=sample['question'],
                options=sample['options'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=image_size,
            )

            # 检查结果是否有效
            if result is None:
                print(f"Warning: Failed to get valid result for idx {idx}, skipping...")
                continue

            # 验证 attention 归一化
            is_normalized = False
            if result.get('attentions') is not None:
                is_normalized = validate_attention_normalization(result['attentions'])
            else:
                print(f"Warning: No attention data for idx {idx}")

            # 计算 HEVA
            try:
                heva_result = compute_heva_from_result(result, alpha=0.2)
                result['heva'] = heva_result['heva']
            except Exception as e:
                print(f"Warning: Failed to compute HEVA for idx {idx}: {e}")
                result['heva'] = 0.0

            # 检查生成的答案 - 使用 JSON 格式提取
            generated_text = result['generated_text']
            answer_pred = ""

            json_match = re.search(r'\{"answer":\s*"([^"]+)"\}', generated_text)
            if json_match:
                answer_pred = json_match.group(1).upper()

            sample_id = str(sample['id']).replace('/', '_').replace('\\', '_').replace(':', '_')

            # 计算 gen_token_num
            gen_token_num = len(result['generated_ids']) - result['prompt_length']

            # 保存元数据
            meta = {
                'sample_id': sample_id,
                'idx': idx,
                'condition': perturbation_name,
                'question': sample['question'],
                'options': sample['options'],
                'prompt': result.get('prompt', ''),
                'ground_truth': sample['answer'],
                'predicted_answer': answer_pred,
                'generated_text': generated_text,
                'correct': answer_pred in sample['answer'],
                'heva': float(result['heva']),
                'attention_validated': is_normalized,
                'prompt_token_num': result['prompt_length'],
                'gen_token_num': gen_token_num,
            }

            meta_path = os.path.join(output_dir, f"{sample_id}_meta.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            results.append({
                'sample_id': sample_id,
                'meta_path': meta_path,
            })

        except Exception as e:
            errors.append({'idx': idx, 'error': str(e)})
            print(f"Error at idx {idx}: {e}")

    # 保存索引文件
    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'perturbation': perturbation_name,
            'results': results,
            'errors': errors,
            'num_samples': len(results),
            'num_errors': len(errors),
        }, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_dir}")
    print(f"Errors: {len(errors)}")

    return results, errors


def main():
    import config

    parser = argparse.ArgumentParser(description='Run perturbation experiment')

    # 实验配置
    parser.add_argument('--exp_name', type=str, default=config.DEFAULT_EXP_NAME,
                       help='Experiment name')
    parser.add_argument('--dataset', type=str, default='VisuRiddles',
                       choices=SUPPORTED_DATASETS, help='Dataset name')

    # 扰动配置
    parser.add_argument('--perturbation', type=str, default='shuffle_patches',
                       choices=['shuffle_patches', 'gaussian_blur', 'zero_image', 'gaussian_noise', 'shuffle_pixels'],
                       help='Perturbation type')

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

    # 模型配置
    parser.add_argument('--model_path', type=str, default=None, help='Model path')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置输出目录
    args.output_dir = os.path.join(config.RESULTS_DIR, args.exp_name, args.dataset, f"perturbation_{args.perturbation}")

    # 保存实验配置
    model_path = args.model_path if args.model_path else config.MODEL_DIR
    exp_config = {
        'exp_name': args.exp_name,
        'dataset': args.dataset,
        'perturbation': args.perturbation,
        'model_name': model_path.split("/")[-1],
        'model_path': model_path,
        'num_samples': args.num_samples,
        'shuffle': args.shuffle,
        'batch_size': args.batch_size,
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

    # 运行扰动实验
    results, errors = run_perturbation_exp(
        model, dataset, sample_indices,
        args.output_dir,
        args.perturbation,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.image_size,
    )

    # 统计准确率
    if results:
        correct = sum(1 for r in results if r.get('correct', False))
        print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")


if __name__ == "__main__":
    main()
