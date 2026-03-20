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
from metrics.inference import load_model_processor, generate_with_heva
from data.loader import load_dataset
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_inference(
    model_path,
    dataset,
    sample_indices,
    output_dir,
    max_new_tokens=8192,
    alpha_values=[0.2],
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
):
    """
    运行推理并缓存结果 (增量计算HEVA，无OOM问题)

    Args:
        model_path: 模型路径
        dataset: 数据集
        sample_indices: 样本索引列表
        output_dir: 输出目录 (results/{dataset}/)
        max_new_tokens: 最大生成长度
        temperature: 温度
        top_p: top-p 采样
        top_k: top-k 采样
        do_sample: 是否采样
    """
    os.makedirs(output_dir, exist_ok=True)
    model, processor = load_model_processor(model_path)
    results = []
    errors = []

    for idx in tqdm(sample_indices, desc="Running inference"):
        try:
            sample = dataset[idx]

            # 运行推理（HEVA在生成过程中同步计算，不会OOM）
            result = generate_with_heva(
                model=model,
                processor=processor,
                image=sample["image"],
                question=sample["question"],
                options=sample["options"],
                max_new_tokens=max_new_tokens,
                alpha_values=alpha_values,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )

            # 检查结果是否有效
            if result is None:
                print(f"Warning: Failed to get valid result for idx {idx}, skipping...")
                continue

            heva_value = result.get("heva", {"error": "HEVA computation failed"})  # 直接获取 HEVA 值

            # 检查生成的答案
            generated_text = result["generated_text"]
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

            sample_id = str(sample["id"])
            # 清理sample_id中的非法文件名字符
            sample_id = sample_id.replace("/", "_").replace("\\", "_").replace(":", "_")

            # 准备元数据 (人类可读)
            meta = {
                "idx": idx,
                "sample_id": sample_id,
                "image_path": sample["image_path"],
                "ground_truth": sample["answer"],
                "predicted_answer": answer_pred,
                "correct": answer_pred != "" and answer_pred in sample["answer"].upper(),
                "prompt_token_num": result["prompt_token_num"],
                "gen_token_num": result["gen_token_num"],
                "heva": heva_value,
                "prompt": result["prompt_text"],
                "generated_text": generated_text,
            }

            # 保存元数据为json
            meta_path = os.path.join(output_dir, f"{sample_id}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            results.append(
                {
                    "idx": idx,
                    "sample_id": sample_id,
                    "meta_path": meta_path,
                }
            )

            # 打印进度
            print(f"[{sample_id}] Saved to {meta_path}, HEVA={heva_value}")

        except Exception as e:
            errors.append({"idx": idx, "error": str(e)})
            print(f"Error at idx {idx}: {e}")
            import traceback

            traceback.print_exc()

    # 保存索引文件
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(
            {
                "num_samples": len(results),
                "num_errors": len(errors),
                "results": results,
                "errors": errors,
            },
            f,
            indent=2,
        )

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
    from data.loader import SUPPORTED_DATASETS

    parser = argparse.ArgumentParser(description="Run inference on multimodal dataset")

    # 实验配置
    parser.add_argument("--exp_name", type=str, default="exp001", help="Experiment name (default: exp001)")
    parser.add_argument("--dataset", type=str, default="VisuRiddles", choices=SUPPORTED_DATASETS, help="Dataset name")

    # 采样配置
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument("--shuffle", type=str, default="false", choices=["true", "false"], help="Shuffle dataset before processing")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # 模型超参数
    parser.add_argument("--max_new_tokens", type=int, default=17, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")

    # 图像处理 (使用模型默认)
    parser.add_argument("--do_sample", action="store_true", default=True, help="Enable sampling")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking", help="Model path")
    parser.add_argument("--alpha_values", type=str, default="0.2", help='Comma-separated alpha values for HEVA (e.g., "0.1,0.2,0.3")')

    args = parser.parse_args()

    args.alpha_values = sorted([float(x) for x in args.alpha_values.split(",")], reverse=True)  # 从大到小排序，优先计算高alpha

    # 设置随机种子
    set_seed(args.seed)

    args.output_dir = os.path.join(args.output_dir, args.exp_name, args.dataset)

    # 创建实验目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存实验配置到 config.json
    # 使用自定义模型路径或默认路径
    model_path = args.model_path
    exp_config = {
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "model_name": model_path.split("/")[-1],
        "model_path": model_path,
        "num_gpus": args.num_gpus,
        "heva_device": args.heva_device,
        "num_samples": args.num_samples,
        "shuffle": args.shuffle,
        # "batch_size": args.batch_size,
        "batch_size": 1,  # 固定为1，逐样本计算HEVA，避免OOM
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "alpha_values": args.alpha_values,
    }
    config_path = os.path.join(args.output_dir, "exp_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {config_path}")

    # 加载数据
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
    if args.shuffle.lower() == "true":
        import random

        random.seed(42)  # 固定随机种子保证可复现
        random.shuffle(sample_indices)

    print(f"Processing samples 0 to {end_idx} ({len(sample_indices)} samples)")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Hyperparameters: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, num_gpus={args.num_gpus}"
    )

    # 运行推理
    run_inference(
        model_path,
        dataset,
        sample_indices,
        args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
    )


if __name__ == "__main__":
    main()
