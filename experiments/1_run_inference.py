"""
运行推理并缓存结果

usage: python run_inference.py [--num_samples N] [--start_idx START] [--save_path PATH]
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
from models.inference import load_model
from metrics.heva import compute_heva_from_result, validate_attention_normalization


def run_inference(model, dataset, sample_indices, output_path, max_new_tokens=128):
    """
    运行推理并缓存结果

    Args:
        model: 推理模型
        dataset: 数据集
        sample_indices: 样本索引列表
        output_path: 输出路径
        max_new_tokens: 最大生成长度
    """
    results = []
    errors = []

    for idx in tqdm(sample_indices, desc="Running inference"):
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

            # 验证 attention 归一化
            is_normalized = validate_attention_normalization(result['attentions'])
            result['attention_validated'] = is_normalized

            # 计算 HEVA
            heva_result = compute_heva_from_result(result, alpha=0.2)
            result['heva'] = heva_result['heva']

            # 检查生成的答案
            generated_text = result['generated_text']
            # 提取选项字母 (A, B, C, D)
            answer_pred = None
            for opt in ['A', 'B', 'C', 'D']:
                if opt in generated_text.upper()[:10]:
                    answer_pred = opt
                    break

            results.append({
                'sample_id': sample['id'],
                'idx': idx,
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'generated_text': generated_text,
                'predicted_answer': answer_pred,
                'correct': answer_pred == sample['answer'],
                'heva': result['heva'],
                'attention_validated': is_normalized,
                # 保存关键数据用于后续分析
                'logits': result['logits'].cpu() if result['logits'] is not None else None,
                'visual_token_indices': result['visual_token_indices'].cpu(),
                'prompt_length': result['prompt_length'],
                'input_ids': result['input_ids'].cpu(),
                'entropies': heva_result['entropies'].cpu().numpy(),
                'num_gen_tokens': len(result['input_ids']) - result['prompt_length'],
            })

        except Exception as e:
            errors.append({'idx': idx, 'error': str(e)})
            print(f"Error at idx {idx}: {e}")

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'errors': errors
        }, f)

    print(f"Saved {len(results)} results to {output_path}")
    print(f"Errors: {len(errors)}")

    return results, errors


def main():
    import config
    parser = argparse.ArgumentParser(description='Run inference on VisuRiddles dataset')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to process')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--save_path', type=str, default=os.path.join(config.RESULTS_DIR, 'inference_results.pkl'),
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

    print(f"Processing samples {args.start_idx} to {end_idx} ({len(sample_indices)} samples)")

    # 运行推理
    run_inference(model, dataset, sample_indices, args.save_path, args.max_new_tokens)


if __name__ == "__main__":
    main()
