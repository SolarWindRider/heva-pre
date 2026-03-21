import json
import os
from pathlib import Path
import numpy as np
from scipy import stats


# 统计ACC
def calculate_acc(exp_dir):
    """
    统计实验结果中各个benchmark的准确率

    Args:
        exp_dir: 实验路径，如 ./results/exp001
    """
    exp_path = Path(exp_dir)

    # 遍历所有benchmark文件夹
    results = {}
    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name

        # 查找所有含有meta的json文件
        correct_count = 0
        total_count = 0

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_count += 1
                    if data["correct"] is True:
                        correct_count += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to read {json_file}: {e}")
                continue

        if total_count > 0:
            acc = correct_count / total_count
            results[bench_name] = {"correct": correct_count, "total": total_count, "acc": acc}

    # 打印结果
    print(f"\n{'Benchmark':<20} {'Correct':<10} {'Total':<10} {'ACC':<10}")
    print("-" * 50)
    for bench_name, stats_dict in sorted(results.items()):
        print(f"{bench_name:<20} {stats_dict['correct']:<10} {stats_dict['total']:<10} {stats_dict['acc']:.4f}")

    # 打印总计
    total_correct = sum(s["correct"] for s in results.values())
    total_samples = sum(s["total"] for s in results.values())
    if total_samples > 0:
        overall_acc = total_correct / total_samples
        print("-" * 50)
        print(f"{'Overall':<20} {total_correct:<10} {total_samples:<10} {overall_acc:.4f}")

    return results


# 统计回答长度与正确性的相关性
def analyze_response_length_correlation(exp_dir):
    """
    分析回答长度与正确性之间的相关性

    Args:
        exp_dir: 实验路径，如 ./results/exp001
    """
    exp_path = Path(exp_dir)

    # 收集所有数据
    all_lengths = []
    all_correct = []

    bench_results = {}

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name
        lengths = []
        correct_list = []

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "gen_token_num" in data and "correct" in data:
                        length = data["gen_token_num"]
                        correct = 1 if data["correct"] is True else 0
                        lengths.append(length)
                        correct_list.append(correct)
                        all_lengths.append(length)
                        all_correct.append(correct)
            except (json.JSONDecodeError, IOError):
                continue

        if lengths:
            bench_results[bench_name] = {
                "lengths": lengths,
                "correct_list": correct_list,
                "avg_length": np.mean(lengths),
                "correct_rate": np.mean(correct_list),
            }

    # 转换为numpy数组
    all_lengths = np.array(all_lengths)
    all_correct = np.array(all_correct)

    # 计算整体相关性
    correlation, p_value = stats.pearsonr(all_lengths, all_correct)

    print(f"\n{'='*60}")
    print("回答长度与正确性相关性分析")
    print(f"{'='*60}")
    print(f"\n整体相关性:")
    print(f"  皮尔逊相关系数: {correlation:.4f}")
    print(f"  P值: {p_value:.4e}")

    # 按benchmark分组统计
    print(f"\n{'Benchmark':<20} {'Avg Length':<15} {'Correct Rate':<15}")
    print("-" * 50)
    for bench_name, stats_dict in sorted(bench_results.items()):
        print(f"{bench_name:<20} {stats_dict['avg_length']:<15.1f} {stats_dict['correct_rate']:<15.4f}")

    # 计算每个benchmark的相关性
    print(f"\n各Benchmark相关性:")
    print(f"{'Benchmark':<20} {'Correlation':<15} {'P-value':<15}")
    print("-" * 50)
    for bench_name, stats_dict in sorted(bench_results.items()):
        lengths = np.array(stats_dict["lengths"])
        correct = np.array(stats_dict["correct_list"])
        if len(lengths) > 2:
            corr, pval = stats.pearsonr(lengths, correct)
            print(f"{bench_name:<20} {corr:<15.4f} {pval:<15.4e}")
        else:
            print(f"{bench_name:<20} {'N/A':<15} {'N/A':<15}")

    # 按长度区间统计正确率
    print(f"\n按回答长度区间统计:")
    bins = [0, 1000, 3000, 5000, 10000, float("inf")]
    labels = ["<1K", "1K-3K", "3K-5K", "5K-10K", ">10K"]

    # 先打印整体汇总
    for i in range(len(bins) - 1):
        mask = (all_lengths >= bins[i]) & (all_lengths < bins[i + 1])
        if mask.sum() > 0:
            rate = all_correct[mask].mean()
            count = mask.sum()
            print(f"  {labels[i]:<10}: 正确率={rate:.4f}, 样本数={count}")

    # 再打印每个benchmark的分布
    print(f"\n各Benchmark按长度区间统计:")

    # 构建表头
    header = f"{'Benchmark':<20}"
    for label in labels:
        header += f" {label:<12}"
    print(header)
    print("-" * (20 + 12 * len(labels)))

    for bench_name in sorted(bench_results.keys()):
        stats_dict = bench_results[bench_name]
        lengths = np.array(stats_dict["lengths"])
        correct = np.array(stats_dict["correct_list"])

        row = f"{bench_name:<20}"
        for i in range(len(bins) - 1):
            mask = (lengths >= bins[i]) & (lengths < bins[i + 1])
            if mask.sum() > 0:
                rate = correct[mask].mean()
                count = mask.sum()
                row += f" {rate:.2f}({count}) "
            else:
                row += " -(-)    "
        print(row)

    return {
        "correlation": correlation,
        "p_value": p_value,
        "bench_results": bench_results,
        "all_lengths": all_lengths,
        "all_correct": all_correct,
    }


if __name__ == "__main__":
    exp_dir = "./results/exp001"

    # # 统计ACC
    # calculate_acc(exp_dir)

    # # 分析回答长度与正确性相关性
    # analyze_response_length_correlation(exp_dir)
