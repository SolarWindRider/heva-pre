"""
扰动实验

验证 HEVA(original) > HEVA(perturbed)
"""

import argparse
import json
import os
import torch
import pickle
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset
from data.perturbations import get_perturbation_functions
from models.inference import load_model
from metrics.heva import compute_heva_from_result, validate_attention_normalization


def run_perturbation_exp(
    model,
    dataset,
    sample_indices,
    perturbation_name,
    output_path,
    max_new_tokens=128
):
    """
    运行扰动实验

    Args:
        model: 推理模型
        dataset: 数据集
        sample_indices: 样本索引
        perturbation_name: 扰动名称
        output_path: 输出路径
        max_new_tokens: 最大生成长度
    """
    perturbations = get_perturbation_functions()
    perturb_func = perturbations.get(perturbation_name)

    if perturb_func is None:
        raise ValueError(f"Unknown perturbation: {perturbation_name}. Available: {list(perturbations.keys())}")

    results = []
    errors = []

    for idx in tqdm(sample_indices, desc=f"Running {perturbation_name}"):
        try:
            sample = dataset[idx]
            question_with_options = sample['question'] + "\n" + sample['options']

            # 应用扰动
            perturbed_image = perturb_func(sample['image'])

            # 运行推理
            result = model.generate_with_attention(
                image=perturbed_image,
                question=question_with_options,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
            )

            # 验证 attention 归一化
            is_normalized = validate_attention_normalization(result['attentions'])

            # 计算 HEVA
            heva_result = compute_heva_from_result(result, alpha=0.2)

            # 检查生成的答案
            generated_text = result['generated_text']
            answer_pred = None
            for opt in ['A', 'B', 'C', 'D']:
                if opt in generated_text.upper()[:10]:
                    answer_pred = opt
                    break

            results.append({
                'sample_id': sample['id'],
                'idx': idx,
                'condition': perturbation_name,
                'ground_truth': sample['answer'],
                'predicted_answer': answer_pred,
                'correct': answer_pred == sample['answer'],
                'heva': heva_result['heva'],
                'attention_validated': is_normalized,
            })

        except Exception as e:
            errors.append({'idx': idx, 'error': str(e)})
            print(f"Error at idx {idx}: {e}")

    return results, errors


def main():
    parser = argparse.ArgumentParser(description='Run perturbation experiment')
    parser.add_argument('--perturbation', type=str, default='shuffle_patches',
                        choices=['shuffle_patches', 'gaussian_blur', 'zero_image', 'gaussian_noise', 'shuffle_pixels'],
                        help='Perturbation type')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--save_path', type=str, default=None, help='Output path')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens')

    args = parser.parse_args()

    # 设置默认输出路径
    if args.save_path is None:
        import config
        args.save_path = os.path.join(config.RESULTS_DIR, f'perturbation_{args.perturbation}.pkl')

    # 加载模型和数据
    print("Loading model...")
    model = load_model()

    print("Loading dataset...")
    dataset = load_dataset()

    # 确定样本索引
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    sample_indices = list(range(args.start_idx, end_idx))

    print(f"Running perturbation: {args.perturbation}")
    print(f"Processing samples {args.start_idx} to {end_idx}")

    # 运行扰动实验
    results, errors = run_perturbation_exp(
        model, dataset, sample_indices,
        args.perturbation, args.save_path,
        args.max_new_tokens
    )

    # 保存结果
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    with open(args.save_path, 'wb') as f:
        pickle.dump({
            'perturbation': args.perturbation,
            'results': results,
            'errors': errors
        }, f)

    print(f"Saved {len(results)} results to {args.save_path}")

    # 统计准确率
    correct = sum(1 for r in results if r['correct'])
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"Mean HEVA: {sum(r['heva'] for r in results)/len(results):.4f}")


if __name__ == "__main__":
    main()
