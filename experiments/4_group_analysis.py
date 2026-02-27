"""
分组分析实验

将样本分为 visual-critical 和 language-guessable 两组
比较 HEVA 差异
"""

import argparse
import json
import os
import re
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset
from models.inference import load_model
from metrics.heva import compute_heva_from_result, validate_attention_normalization


# 定义 visual-critical 关键词
VISUAL_KEYWORDS = [
    'color', 'left', 'right', 'count', 'position', 'number',
    '颜色', '左', '右', '数量', '位置', '数字',
    '多少', '几个', '哪个', '几个', '颜色'
]


def classify_sample(question: str) -> str:
    """
    分类样本

    Args:
        question: 问题文本

    Returns:
        'visual-critical' 或 'language-guessable'
    """
    question_lower = question.lower()

    for keyword in VISUAL_KEYWORDS:
        if keyword in question_lower:
            return 'visual-critical'

    return 'language-guessable'


def run_group_analysis(
    model,
    dataset,
    sample_indices,
    output_path,
    max_new_tokens=128
):
    """
    运行分组分析

    Args:
        model: 推理模型
        dataset: 数据集
        sample_indices: 样本索引
        output_path: 输出路径
        max_new_tokens: 最大生成长度
    """
    results = []
    errors = []

    for idx in tqdm(sample_indices, desc="Running group analysis"):
        try:
            sample = dataset[idx]
            question_with_options = sample['question'] + "\n" + sample['options']

            # 分类
            group = classify_sample(sample['question'])

            # 运行推理
            result = model.generate_with_attention(
                image=sample['image'],
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
                'group': group,
                'question': sample['question'],
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
    import config
    parser = argparse.ArgumentParser(description='Run group analysis experiment')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--save_path', type=str, default=os.path.join(config.RESULTS_DIR, 'group_analysis.pkl'),
                        help='Output path')
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
    results, errors = run_group_analysis(
        model, dataset, sample_indices,
        args.save_path, args.max_new_tokens
    )

    # 保存结果
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    import pickle
    with open(args.save_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'errors': errors
        }, f)

    print(f"Saved {len(results)} results to {args.save_path}")

    # 统计分组
    visual_samples = [r for r in results if r['group'] == 'visual-critical']
    language_samples = [r for r in results if r['group'] == 'language-guessable']

    print(f"\nGroup Statistics:")
    print(f"  Visual-Critical: {len(visual_samples)}")
    print(f"  Language-Guessable: {len(language_samples)}")

    if len(visual_samples) > 0:
        mean_visual = sum(r['heva'] for r in visual_samples) / len(visual_samples)
        print(f"  Mean HEVA (Visual): {mean_visual:.4f}")

    if len(language_samples) > 0:
        mean_language = sum(r['heva'] for r in language_samples) / len(language_samples)
        print(f"  Mean HEVA (Language): {mean_language:.4f}")


if __name__ == "__main__":
    main()
