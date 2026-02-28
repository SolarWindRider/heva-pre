"""
运行推理并缓存结果

新格式：
- results/{exp_name}/{dataset}/
  - {sample_id}_meta.json (人类可读元数据)

usage: python 1_run_inference.py [--exp_name EXP_NAME] [--dataset DATASET] [--num_samples N]
"""

import argparse
import json
import os
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset
from models.inference import load_model
from metrics.heva import compute_heva_from_result, validate_attention_normalization


def run_inference(model, dataset, sample_indices, output_dir,
                  max_new_tokens=8192, temperature=0.7, top_p=0.9, top_k=50, do_sample=True, image_size=448):
    """
    运行推理并缓存结果

    Args:
        model: 推理模型
        dataset: 数据集
        sample_indices: 样本索引列表
        output_dir: 输出目录 (results/{dataset}/)
        max_new_tokens: 最大生成长度
        temperature: 温度
        top_p: top-p 采样
        top_k: top-k 采样
        do_sample: 是否采样
        image_size: 图像处理大小
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    errors = []

    for idx in tqdm(sample_indices, desc="Running inference"):
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
                do_sample=do_sample,
                image_size=image_size,
            )

            # 检查结果是否有效
            if result is None:
                print(f"Warning: Failed to get valid result for idx {idx}, skipping...")
                continue

            # 验证 attention 归一化 (如果 attention 可用)
            is_normalized = False
            if result.get('attentions') is not None:
                is_normalized = validate_attention_normalization(result['attentions'])
                result['attention_validated'] = is_normalized
            else:
                result['attention_validated'] = False
                print(f"Warning: No attention data for idx {idx}, proceeding without attention validation")

            # 计算 HEVA (多个 alpha 值: 10% 到 100%)
            try:
                heva_values = {}
                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    heva_result = compute_heva_from_result(result, alpha=alpha)
                    heva_values[f'heva_{int(alpha*100)}'] = float(heva_result['heva'])
                result['heva_dict'] = heva_values
            except Exception as e:
                print(f"Warning: Failed to compute HEVA for idx {idx}: {e}")
                result['heva_dict'] = {f'heva_{int(a*100)}': 0.0 for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

            # 检查生成的答案
            generated_text = result['generated_text']
            # 提取选项字母 (A, B, C, D)
            answer_pred = ""

            # 优先从JSON格式提取答案: {"answer": "X"}
            import re
            json_match = re.search(r'\{"answer":\s*"([^"]+)"\}', generated_text)
            if json_match:
                answer_pred = json_match.group(1).upper()
            else:
                # 如果没有JSON格式，则认为回答有误，保持空字符
                pass

            sample_id = str(sample['id'])
            # 清理sample_id中的非法文件名字符
            sample_id = sample_id.replace('/', '_').replace('\\', '_').replace(':', '_')

            # 准备元数据 (人类可读)
            # gen_token_num = 完整生成序列长度 - prompt长度
            gen_token_num = len(result['generated_ids']) - result['prompt_length']
            meta = {
                'sample_id': sample_id,
                'idx': idx,
                'image_path': sample.get('image_path', ''),
                'question': sample['question'],
                'options': sample['options'],
                'raw_question': result.get('raw_question', ''),  # 问题+选项+JSON格式指令
                'prompt': result.get('prompt', ''),  # 喂给模型的完整prompt (含特殊token)
                'ground_truth': sample['answer'],
                'predicted_answer': answer_pred,
                'generated_text': generated_text,
                'correct': answer_pred in sample['answer'],
                'heva': result.get('heva_dict', {}),  # HEVA 字典: {heva_10: x, heva_20: x, ..., heva_100: x}
                'attention_validated': is_normalized,
                'prompt_token_num': result['prompt_length'],
                'gen_token_num': gen_token_num,
            }

            # 保存元数据为json
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
            'results': results,
            'errors': errors,
            'num_samples': len(results),
            'num_errors': len(errors),
        }, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_dir}")
    print(f"Errors: {len(errors)}")
    print(f"Index saved to: {index_path}")

    return results, errors


def set_seed(seed: int):
    """设置全局随机种子以保证可复现性"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # PyTorch 2.0+ 需要设置以下选项来保证确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def main():
    import config
    from data.loader import SUPPORTED_DATASETS
    parser = argparse.ArgumentParser(description='Run inference on multimodal dataset')

    # 实验配置
    parser.add_argument('--exp_name', type=str, default=config.DEFAULT_EXP_NAME,
                       help='Experiment name (default: exp001)')
    parser.add_argument('--dataset', type=str, default='VisuRiddles',
                       choices=SUPPORTED_DATASETS, help='Dataset name')

    # 采样配置
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to process (-1 for all)')
    parser.add_argument('--shuffle', type=str, default='false', choices=['true', 'false'], help='Shuffle dataset before processing')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # 模型超参数
    parser.add_argument('--max_new_tokens', type=int, default=8192, help='Max new tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')

    # 图像处理
    parser.add_argument('--image_size', type=int, default=448, help='Image size for processing')
    parser.add_argument('--do_sample', action='store_true', default=True, help='Enable sampling')

    # 输出配置
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')

    # 模型配置
    parser.add_argument('--model_path', type=str, default=None, help='Model path (default: from config.py)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置默认输出目录: results/{exp_name}/{dataset}/
    if args.output_dir is None:
        args.output_dir = os.path.join(config.RESULTS_DIR, args.exp_name, args.dataset)

    # 创建实验目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存实验配置到 config.json
    # 使用自定义模型路径或默认路径
    model_path = args.model_path if args.model_path else config.MODEL_DIR
    exp_config = {
        'exp_name': args.exp_name,
        'dataset': args.dataset,
        'model_name': model_path.split("/")[-1],
        'model_path': model_path,
        'num_gpus': args.num_gpus,
        'num_samples': args.num_samples,
        'shuffle': args.shuffle,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'image_size': args.image_size,
        'do_sample': args.do_sample,
    }
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {config_path}")

    # 加载模型和数据
    print("Loading model...")
    model = load_model(model_path=args.model_path, num_gpus=args.num_gpus)

    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Dataset size: {len(dataset)}")

    # 确定样本索引
    # -1 表示处理全部样本
    if args.num_samples == -1:
        end_idx = len(dataset)
    else:
        end_idx = min(args.num_samples, len(dataset))
    sample_indices = list(range(0, end_idx))

    # 可选：shuffle 数据集
    if args.shuffle.lower() == 'true':
        import random
        random.seed(42)  # 固定随机种子保证可复现
        random.shuffle(sample_indices)

    print(f"Processing samples 0 to {end_idx} ({len(sample_indices)} samples)")
    print(f"Output directory: {args.output_dir}")
    print(f"Hyperparameters: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, num_gpus={args.num_gpus}")

    # 运行推理
    run_inference(
        model, dataset, sample_indices, args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
