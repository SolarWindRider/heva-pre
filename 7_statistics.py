"""
对1_run_inference.py结果的统计分析脚本
"""

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


# 分析token熵与视觉注意力的相关性
def analyze_entropy_vattn_correlation(exp_dir):
    """
    分析每个样本中token熵与视觉注意力(vattn)的相关性

    设计方案：
    1. 对每个样本，加载gen_entropy.pkl和gen_vattn.pkl
    2. 计算每个token的熵与视觉注意力的Pearson相关系数
    3. 汇总所有样本的相关性分布
    4. 分别分析正确和错误样本的相关性差异

    Args:
        exp_dir: 实验路径，如 ./results/exp001
    """
    import pickle

    exp_path = Path(exp_dir)

    # 收集数据
    all_correlations = []
    all_correct_correlations = []
    all_incorrect_correlations = []

    bench_results = {}

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name
        correlations = []

        pkls_dir = bench_dir / "pkls"
        if not pkls_dir.exists():
            continue

        # 遍历所有meta文件来获取样本信息
        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # 获取pkl文件路径
                    entropy_path = data.get("gen_entropy_path")
                    vattn_path = data.get("gen_vattn_path")
                    correct = data.get("correct", False)

                    if not entropy_path or not vattn_path:
                        continue

                    # 转换为绝对路径
                    entropy_path = Path(entropy_path)
                    vattn_path = Path(vattn_path)

                    if not entropy_path.exists() or not vattn_path.exists():
                        continue

                    # 加载数据
                    with open(entropy_path, "rb") as f:
                        entropy = pickle.load(f)
                        entropy = entropy.squeeze().numpy()

                    with open(vattn_path, "rb") as f:
                        vattn = pickle.load(f)
                        vattn = vattn.float().squeeze().numpy()

                    # 确保长度一致
                    assert len(entropy) == len(vattn), f"Length mismatch: entropy({len(entropy)}) vs vattn({len(vattn)}) in {json_file}"

                    # 计算相关系数
                    if len(entropy) > 10 and entropy.std() > 0 and vattn.std() > 0:
                        corr, _ = stats.pearsonr(entropy, vattn)
                        if not np.isnan(corr):
                            correlations.append(corr)
                            all_correlations.append(corr)

                            if correct:
                                all_correct_correlations.append(corr)
                            else:
                                all_incorrect_correlations.append(corr)

            except Exception as e:
                continue

        if correlations:
            bench_results[bench_name] = {
                "correlations": correlations,
                "mean_corr": np.mean(correlations),
                "median_corr": np.median(correlations),
                "std_corr": np.std(correlations),
            }

    # 转换为numpy数组
    all_correlations = np.array(all_correlations)
    all_correct_correlations = np.array(all_correct_correlations)
    all_incorrect_correlations = np.array(all_incorrect_correlations)

    print(f"\n{'='*60}")
    print("Token熵与视觉注意力(vattn)相关性分析")
    print(f"{'='*60}")

    # 整体统计
    print(f"\n整体统计 (N={len(all_correlations)}):")
    print(f"  均值: {np.mean(all_correlations):.4f}")
    print(f"  中位数: {np.median(all_correlations):.4f}")
    print(f"  标准差: {np.std(all_correlations):.4f}")
    print(f"  范围: [{np.min(all_correlations):.4f}, {np.max(all_correlations):.4f}]")

    # 按正确性分组统计
    print(f"\n按回答正确性分组:")
    print(f"  正确回答 (N={len(all_correct_correlations)}):")
    print(f"    均值: {np.mean(all_correct_correlations):.4f}")
    print(f"    中位数: {np.median(all_correct_correlations):.4f}")
    print(f"  错误回答 (N={len(all_incorrect_correlations)}):")
    print(f"    均值: {np.mean(all_incorrect_correlations):.4f}")
    print(f"    中位数: {np.median(all_incorrect_correlations):.4f}")

    # 统计显著性检验
    if len(all_correct_correlations) > 0 and len(all_incorrect_correlations) > 0:
        t_stat, p_value = stats.ttest_ind(all_correct_correlations, all_incorrect_correlations)
        print(f"\n  T检验: t={t_stat:.4f}, p={p_value:.4e}")
        if len(all_correct_correlations) > 10 and len(all_incorrect_correlations) > 10:
            mw_stat, mw_p = stats.mannwhitneyu(all_correct_correlations, all_incorrect_correlations)
            print(f"  Mann-Whitney U检验: U={mw_stat:.1f}, p={mw_p:.4e}")

    # 按benchmark统计
    print(f"\n各Benchmark统计:")
    print(f"{'Benchmark':<20} {'N':<8} {'Mean':<10} {'Median':<10} {'Std':<10}")
    print("-" * 58)
    for bench_name, stats_dict in sorted(bench_results.items()):
        print(
            f"{bench_name:<20} {len(stats_dict['correlations']):<8} "
            f"{stats_dict['mean_corr']:<10.4f} {stats_dict['median_corr']:<10.4f} "
            f"{stats_dict['std_corr']:<10.4f}"
        )

    # 相关性分布
    print(f"\n相关性分布:")
    bins = [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]
    labels = ["<-0.5", "-0.5~-0.2", "-0.2~0", "0~0.2", "0.2~0.5", ">0.5"]
    for i in range(len(bins) - 1):
        mask = (all_correlations >= bins[i]) & (all_correlations < bins[i + 1])
        count = mask.sum()
        pct = count / len(all_correlations) * 100
        print(f"  {labels[i]:<12}: {count:>4} ({pct:>5.1f}%)")

    return {
        "all_correlations": all_correlations,
        "correct_correlations": all_correct_correlations,
        "incorrect_correlations": all_incorrect_correlations,
        "bench_results": bench_results,
    }


# 分析高熵token是否具有更高的视觉注意力
def analyze_high_entropy_vattn(exp_dir, top_percent=0.2):
    """
    分析高熵token是否具有更高的视觉注意力

    设计方案：
    1. 对每个样本，选取熵值最高的top_percent% tokens
    2. 计算这些高熵token的平均视觉注意力
    3. 计算剩余token的平均视觉注意力
    4. 比较两者的差异

    Args:
        exp_dir: 实验路径
        top_percent: 高熵token的比例，默认20%
    """
    import pickle

    exp_path = Path(exp_dir)

    # 收集数据
    all_vattn_high_entropy = []  # 高熵token的平均视觉注意力
    all_vattn_low_entropy = []  # 低熵token的平均视觉注意力
    all_avg_entropy = []  # 样本平均熵

    bench_results = {}

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name
        vattn_high_list = []
        vattn_low_list = []
        avg_entropy_list = []

        pkls_dir = bench_dir / "pkls"
        if not pkls_dir.exists():
            continue

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    entropy_path = data.get("gen_entropy_path")
                    vattn_path = data.get("gen_vattn_path")
                    correct = data.get("correct", False)

                    if not entropy_path or not vattn_path:
                        continue

                    entropy_path = Path(entropy_path)
                    vattn_path = Path(vattn_path)

                    if not entropy_path.exists() or not vattn_path.exists():
                        continue

                    # 加载数据
                    with open(entropy_path, "rb") as f:
                        entropy = pickle.load(f)
                        entropy = entropy.squeeze().numpy()

                    with open(vattn_path, "rb") as f:
                        vattn = pickle.load(f)
                        vattn = vattn.float().squeeze().numpy()

                    assert len(entropy) == len(vattn), f"Length mismatch: entropy({len(entropy)}) vs vattn({len(vattn)}) in {json_file}"

                    if len(entropy) < 50:
                        continue

                    # 取熵值最高和最低的top_percent% token
                    n_high = max(1, int(len(entropy) * top_percent))
                    high_entropy_indices = np.argsort(entropy)[-n_high:]
                    low_entropy_indices = np.argsort(entropy)[:-n_high]

                    vattn_high = vattn[high_entropy_indices].mean()
                    vattn_low = vattn[low_entropy_indices].mean()

                    vattn_high_list.append(vattn_high)
                    vattn_low_list.append(vattn_low)
                    avg_entropy_list.append(entropy.mean())

                    all_vattn_high_entropy.append(vattn_high)
                    all_vattn_low_entropy.append(vattn_low)
                    all_avg_entropy.append(entropy.mean())

            except Exception as e:
                continue

        if vattn_high_list:
            bench_results[bench_name] = {
                "vattn_high": vattn_high_list,
                "vattn_low": vattn_low_list,
                "avg_entropy": avg_entropy_list,
                "diff": np.mean(vattn_high_list) - np.mean(vattn_low_list),
            }

    all_vattn_high_entropy = np.array(all_vattn_high_entropy)
    all_vattn_low_entropy = np.array(all_vattn_low_entropy)
    all_avg_entropy = np.array(all_avg_entropy)

    print(f"\n{'='*60}")
    print(f"高熵Token (top {top_percent*100:.0f}%) 视觉注意力分析")
    print(f"{'='*60}")

    # 整体统计
    print(f"\n整体统计 (N={len(all_vattn_high_entropy)}):")
    print(f"  高熵token平均视觉注意力: {np.mean(all_vattn_high_entropy):.6f}")
    print(f"  低熵token平均视觉注意力: {np.mean(all_vattn_low_entropy):.6f}")
    print(f"  差值 (高-低): {np.mean(all_vattn_high_entropy) - np.mean(all_vattn_low_entropy):.6f}")
    print(f"  比值 (高/低): {np.mean(all_vattn_high_entropy) / np.mean(all_vattn_low_entropy):.4f}")

    # 统计检验
    t_stat, p_value = stats.ttest_rel(all_vattn_high_entropy, all_vattn_low_entropy)
    print(f"\n配对t检验: t={t_stat:.4f}, p={p_value:.4e}")

    # 计算样本级别的相关性：平均熵 vs 平均视觉注意力
    corr_entropy_vattn, p_val = stats.pearsonr(all_avg_entropy, all_vattn_high_entropy)
    print(f"\n样本平均熵 vs 高熵token平均视觉注意力:")
    print(f"  相关系数: {corr_entropy_vattn:.4f}")
    print(f"  P值: {p_val:.4e}")

    # 按benchmark统计
    print(f"\n各Benchmark统计:")
    print(f"{'Benchmark':<20} {'High-视觉注意力':<12} {'Low-视觉注意力':<12} {'Diff':<12} {'Ratio':<10}")
    print("-" * 66)
    for bench_name, stats_dict in sorted(bench_results.items()):
        high_mean = np.mean(stats_dict["vattn_high"])
        low_mean = np.mean(stats_dict["vattn_low"])
        diff = high_mean - low_mean
        ratio = high_mean / low_mean if low_mean > 0 else 0
        print(f"{bench_name:<20} {high_mean:<12.6f} {low_mean:<12.6f} {diff:<12.6f} {ratio:<10.4f}")

    # 分析：验证HEVA假设
    print(f"\n" + "=" * 60)
    print("结论分析")
    print("=" * 60)
    diff_mean = np.mean(all_vattn_high_entropy) - np.mean(all_vattn_low_entropy)
    if diff_mean > 0 and p_value < 0.05:
        print(f"✓ 高熵token的视觉注意力显著更高 (差值: {diff_mean:.6f})")
        print(f"  这支持HEVA的核心假设：高熵token更依赖视觉信息")
    else:
        print(f"✗ 高熵token的视觉注意力无显著差异 (差值: {diff_mean:.6f})")

    return {
        "vattn_high_entropy": all_vattn_high_entropy,
        "vattn_low_entropy": all_vattn_low_entropy,
        "avg_entropy": all_avg_entropy,
        "bench_results": bench_results,
    }


# 分析高熵token的视觉注意力与回答正确性的关系
def analyze_vattn_correctness(exp_dir, top_percent=0.2):
    """
    分析高熵token的视觉注意力与回答正确性的关系

    设计方案：
    1. 对每个样本计算HEVA (高熵token的平均视觉注意力)
    2. 按HEVA高低将样本分组
    3. 统计各组的正确率
    4. 分析高HEVA是否对应更高的正确率

    Args:
        exp_dir: 实验路径
        top_percent: 高熵token比例
    """
    import pickle

    exp_path = Path(exp_dir)

    # 收集数据：视觉注意力, 正确性, 熵
    all_vattn = []  # 高熵token的视觉注意力
    all_correct = []  # 是否正确
    all_avg_entropy = []  # 平均熵
    all_vattn_ratio = []  # 高熵token 视觉注意力 / 低熵token 视觉注意力

    bench_results = {}

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name
        heva_list = []
        correct_list = []
        avg_entropy_list = []
        heva_ratio_list = []

        pkls_dir = bench_dir / "pkls"
        if not pkls_dir.exists():
            continue

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    entropy_path = data.get("gen_entropy_path")
                    vattn_path = data.get("gen_vattn_path")
                    correct = data.get("correct", False)

                    if not entropy_path or not vattn_path:
                        continue

                    entropy_path = Path(entropy_path)
                    vattn_path = Path(vattn_path)

                    if not entropy_path.exists() or not vattn_path.exists():
                        continue

                    # 加载数据
                    with open(entropy_path, "rb") as f:
                        entropy = pickle.load(f)
                        entropy = entropy.squeeze().numpy()

                    with open(vattn_path, "rb") as f:
                        vattn = pickle.load(f)
                        vattn = vattn.float().squeeze().numpy()

                    min_len = min(len(entropy), len(vattn))
                    entropy = entropy[:min_len]
                    vattn = vattn[:min_len]

                    if len(entropy) < 50:
                        continue

                    # 计算HEVA
                    n_high = max(1, int(len(entropy) * top_percent))
                    high_entropy_indices = np.argsort(entropy)[-n_high:]
                    low_entropy_indices = np.argsort(entropy)[:-n_high]

                    heva = vattn[high_entropy_indices].mean()  # 即HEVA
                    vattn_low = vattn[low_entropy_indices].mean()
                    heva_ratio = heva / vattn_low if vattn_low > 0 else 0

                    heva_list.append(heva)
                    correct_list.append(1 if correct else 0)
                    avg_entropy_list.append(entropy.mean())
                    heva_ratio_list.append(heva_ratio)

                    all_vattn.append(heva)
                    all_correct.append(1 if correct else 0)
                    all_avg_entropy.append(entropy.mean())
                    all_vattn_ratio.append(heva_ratio)

            except Exception as e:
                continue

        if heva_list:
            bench_results[bench_name] = {
                "heva": heva_list,
                "correct": correct_list,
                "avg_entropy": avg_entropy_list,
                "heva_ratio": heva_ratio_list,
            }

    all_vattn = np.array(all_vattn)
    all_correct = np.array(all_correct)
    all_avg_entropy = np.array(all_avg_entropy)
    all_vattn_ratio = np.array(all_vattn_ratio)

    print(f"\n{'='*60}")
    print(f"高熵Token HEVA与回答正确性关系分析")
    print(f"{'='*60}")

    # 1. 计算HEVA与正确性的相关性
    corr_heva_correct, p_val1 = stats.pearsonr(all_vattn, all_correct)
    corr_ratio_correct, p_val2 = stats.pearsonr(all_vattn_ratio, all_correct)

    print(f"\n1. HEVA与正确性相关性:")
    print(f"   HEVA vs 正确性: r={corr_heva_correct:.4f}, p={p_val1:.4e}")
    print(f"   HEVA比值 vs 正确性: r={corr_ratio_correct:.4f}, p={p_val2:.4e}")

    # 2. 按HEVA高低分组统计正确率
    print(f"\n2. 按HEVA高低分组统计正确率:")
    heva_median = np.median(all_vattn)

    low_heva_mask = all_vattn < heva_median
    high_heva_mask = all_vattn >= heva_median

    low_heva_acc = all_correct[low_heva_mask].mean()
    high_heva_acc = all_correct[high_heva_mask].mean()

    print(f"   低HEVA组 (N={low_heva_mask.sum()}): 正确率={low_heva_acc:.4f}")
    print(f"   高HEVA组 (N={high_heva_mask.sum()}): 正确率={high_heva_acc:.4f}")

    # 3. 按HEVA四分位数分组
    print(f"\n3. 按HEVA四分位数分组:")
    q25 = np.percentile(all_vattn, 25)
    q50 = np.percentile(all_vattn, 50)
    q75 = np.percentile(all_vattn, 75)

    bins = [0, q25, q50, q75, float("inf")]
    labels = ["Q1(最低)", "Q2", "Q3", "Q4(最高)"]

    print(f"   分位点: Q25={q25:.6f}, Q50={q50:.6f}, Q75={q75:.6f}")
    print(f"\n   {'分组':<12} {'正确率':<10} {'样本数':<10}")
    print(f"   {'-'*32}")
    for i in range(len(bins) - 1):
        mask = (all_vattn >= bins[i]) & (all_vattn < bins[i + 1])
        if mask.sum() > 0:
            acc = all_correct[mask].mean()
            count = mask.sum()
            print(f"   {labels[i]:<12} {acc:<10.4f} {count:<10}")

    # 4. 按HEVA比值分组 (高熵token HEVA / 低熵token HEVA)
    print(f"\n4. 按HEVA比值分组 (高/低):")
    ratio_median = np.median(all_vattn_ratio)

    low_ratio_mask = all_vattn_ratio < ratio_median
    high_ratio_mask = all_vattn_ratio >= ratio_median

    low_ratio_acc = all_correct[low_ratio_mask].mean()
    high_ratio_acc = all_correct[high_ratio_mask].mean()

    print(f"   低比值组 (N={low_ratio_mask.sum()}): 正确率={low_ratio_acc:.4f}")
    print(f"   高比值组 (N={high_ratio_mask.sum()}): 正确率={high_ratio_acc:.4f}")

    # 5. 按benchmark分析
    print(f"\n5. 各Benchmark: HEVA与正确性相关性:")
    print(f"   {'Benchmark':<20} {'Corr':<12} {'P-value':<12}")
    print(f"   {'-'*44}")
    for bench_name in sorted(bench_results.keys()):
        data = bench_results[bench_name]
        heva = np.array(data["heva"])
        correct = np.array(data["correct"])

        if len(heva) > 10 and heva.std() > 0:
            corr, pval = stats.pearsonr(heva, correct)
            print(f"   {bench_name:<20} {corr:<12.4f} {pval:<12.4e}")

    # 6. 综合分析
    print(f"\n" + "=" * 60)
    print("结论分析")
    print("=" * 60)

    if high_heva_acc > low_heva_acc and p_val1 < 0.05:
        print(f"✓ 高HEVA样本正确率更高 ({high_heva_acc:.2%} vs {low_heva_acc:.2%})")
    elif high_heva_acc < low_heva_acc and p_val1 < 0.05:
        print(f"✗ 低HEVA样本正确率更高 ({low_heva_acc:.2%} vs {high_heva_acc:.2%})")
    else:
        print(f"○ HEVA与正确性无显著关联")

    return {
        "vattn": all_vattn,
        "correct": all_correct,
        "vattn_ratio": all_vattn_ratio,
        "bench_results": bench_results,
    }


# 验证假设：高熵token的视觉注意力更高 -> 更容易回答正确
def verify_heva_hypothesis(exp_dir, top_percent=0.2):
    """
    验证假设：高熵token的HEVA更高更容易回答正确

    设计：
    1. 对每个样本计算HEVA（高熵token的平均视觉注意力）
    2. 比较正确vs错误样本的HEVA分布
    3. 使用多种统计检验验证差异显著性
    4. 展示HEVA阈值对正确率的预测能力

    Args:
        exp_dir: 实验路径
        top_percent: 高熵token比例
    """
    import pickle

    exp_path = Path(exp_dir)

    # 收集数据
    correct_heva = []  # 正确样本的HEVA
    incorrect_heva = []  # 错误样本的HEVA

    bench_correct_heva = {}
    bench_incorrect_heva = {}

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name
        c_heva = []
        ic_heva = []

        pkls_dir = bench_dir / "pkls"
        if not pkls_dir.exists():
            continue

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    entropy_path = data.get("gen_entropy_path")
                    vattn_path = data.get("gen_vattn_path")
                    correct = data.get("correct", False)

                    if not entropy_path or not vattn_path:
                        continue

                    entropy_path = Path(entropy_path)
                    vattn_path = Path(vattn_path)

                    if not entropy_path.exists() or not vattn_path.exists():
                        continue

                    # 加载数据
                    with open(entropy_path, "rb") as f:
                        entropy = pickle.load(f)
                        entropy = entropy.squeeze().numpy()

                    with open(vattn_path, "rb") as f:
                        vattn = pickle.load(f)
                        vattn = vattn.float().squeeze().numpy()

                    assert len(entropy) == len(vattn), f"Length mismatch: entropy({len(entropy)}) vs vattn({len(vattn)}) in {json_file}"
                    if len(entropy) < 50:
                        continue

                    # 计算样本级HEVA
                    n_high = max(1, int(len(entropy) * top_percent))
                    high_entropy_indices = np.argsort(entropy)[-n_high:]
                    heva = vattn[high_entropy_indices].mean()

                    if correct:
                        correct_heva.append(heva)
                        c_heva.append(heva)
                    else:
                        incorrect_heva.append(heva)
                        ic_heva.append(heva)

            except Exception as e:
                continue

        if c_heva or ic_heva:
            bench_correct_heva[bench_name] = c_heva
            bench_incorrect_heva[bench_name] = ic_heva

    correct_heva = np.array(correct_heva)
    incorrect_heva = np.array(incorrect_heva)

    print(f"\n{'='*60}")
    print("验证假设：推理结果的HEVA更高 -> 更容易回答正确")
    print(f"{'='*60}")

    # 1. 基本统计
    print(f"\n1. 基本统计:")
    print(f"   正确样本 (N={len(correct_heva)}): HEVA均值={correct_heva.mean():.6f}, 中位数={np.median(correct_heva):.6f}")
    print(f"   错误样本 (N={len(incorrect_heva)}): HEVA均值={incorrect_heva.mean():.6f}, 中位数={np.median(incorrect_heva):.6f}")
    print(f"   差异: 均值差={correct_heva.mean() - incorrect_heva.mean():.6f}")

    # 2. 统计检验
    print(f"\n2. 统计检验:")
    # t检验
    t_stat, t_pval = stats.ttest_ind(correct_heva, incorrect_heva)
    print(f"   独立样本t检验: t={t_stat:.4f}, p={t_pval:.4e}")

    # Mann-Whitney U检验 (非参数)
    u_stat, u_pval = stats.mannwhitneyu(correct_heva, incorrect_heva, alternative="greater")
    print(f"   Mann-Whitney U检验 (单侧): U={u_stat:.1f}, p={u_pval:.4e}")

    # 效应量 (Cohen's d)
    pooled_std = np.sqrt((correct_heva.std() ** 2 + incorrect_heva.std() ** 2) / 2)
    cohens_d = (correct_heva.mean() - incorrect_heva.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"   效应量 (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) >= 0.8:
        effect_size = "大效应"
    elif abs(cohens_d) >= 0.5:
        effect_size = "中等效应"
    elif abs(cohens_d) >= 0.2:
        effect_size = "小效应"
    else:
        effect_size = "微小效应"
    print(f"   效应量解释: {effect_size}")

    # 3. 各Benchmark对比
    print(f"\n3. 各Benchmark对比:")
    print(f"   {'Benchmark':<18} {'正确HEVA':<14} {'错误HEVA':<14} {'差值':<12} {'p-value':<12}")
    print(f"   {'-'*70}")
    for bench_name in sorted(bench_correct_heva.keys()):
        c_heva = np.array(bench_correct_heva[bench_name])
        ic_heva = np.array(bench_incorrect_heva[bench_name])

        if len(c_heva) > 2 and len(ic_heva) > 2 and c_heva.std() > 0 and ic_heva.std() > 0:
            _, pval = stats.ttest_ind(c_heva, ic_heva)
            diff = c_heva.mean() - ic_heva.mean()
            print(f"   {bench_name:<18} {c_heva.mean():<14.6f} {ic_heva.mean():<14.6f} {diff:<12.6f} {pval:<12.4e}")
        else:
            print(f"   {bench_name:<18} {c_heva.mean():<14.6f} {ic_heva.mean():<14.6f} {'N/A':<12} {'N/A':<12}")

    # 4. HEVA阈值预测能力
    print(f"\n4. HEVA阈值预测能力:")

    # 找最佳阈值
    thresholds = np.percentile(np.concatenate([correct_heva, incorrect_heva]), [10, 20, 30, 40, 50, 60, 70, 80, 90])

    print(f"   {'阈值':<12} {'>阈值正确率':<14} {'<阈值正确率':<14} {'提升':<10}")
    print(f"   {'-'*50}")

    best_thresh = 0
    best_improvement = 0

    for thresh in thresholds:
        high_acc = (correct_heva >= thresh).mean() if len(correct_heva) > 0 else 0
        low_acc = (incorrect_heva >= thresh).mean() if len(incorrect_heva) > 0 else 0

        # 总体正确率
        all_heva = np.concatenate([correct_heva, incorrect_heva])
        all_correct = np.concatenate([np.ones(len(correct_heva)), np.zeros(len(incorrect_heva))])

        acc_above = all_correct[all_heva >= thresh].mean()
        acc_below = all_correct[all_heva < thresh].mean()
        improvement = acc_above - acc_below

        print(f"   {thresh:<12.6f} {acc_above:<14.4f} {acc_below:<14.4f} {improvement:<10.4f}")

        if improvement > best_improvement:
            best_improvement = improvement
            best_thresh = thresh

    # 5. 结论
    print(f"\n" + "=" * 60)
    print("验证结论")
    print("=" * 60)

    if t_pval < 0.05 and correct_heva.mean() > incorrect_heva.mean():
        print(f"✓ 假设验证成功！")
        print(f"  - 正确样本的HEVA显著更高 ({correct_heva.mean():.6f} vs {incorrect_heva.mean():.6f})")
        print(f"  - 统计检验显著 (p={t_pval:.4e})")
        print(f"  - 效应量为{effect_size} (d={cohens_d:.4f})")
        print(f"  - 最佳阈值: {best_thresh:.6f}, 可提升正确率 {best_improvement:.2%}")
    else:
        print(f"✗ 假设未通过验证")

    return {
        "correct_heva": correct_heva,
        "incorrect_heva": incorrect_heva,
        "cohens_d": cohens_d,
        "t_pval": t_pval,
    }


# 分析Instruct模型 vs Thinking模型中高熵token的特征差异
def analyze_entropy_token_patterns(exp_dir, top_percent=0.2):
    """
    分析高熵token的特征：
    1. 高熵token在回答中的位置分布
    2. 高熵token是否对应思考/推理token
    3. 高熵token的熵值分布

    目的：理解Instruct模型中高熵token的含义
    """
    import pickle

    exp_path = Path(exp_dir)

    # 收集数据
    all_entropy_positions = []  # 高熵token的位置分布 (相对位置 0-1)
    all_entropy_values = []  # 高熵token的熵值
    all_heva = []  # HEVA值

    all_low_entropy_positions = []  # 低熵token的位置
    all_low_entropy_values = []  # 低熵token的熵值

    bench_stats = {}

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name
        positions = []
        entropies = []
        heva_list = []

        pkls_dir = bench_dir / "pkls"
        if not pkls_dir.exists():
            continue

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    entropy_path = data.get("gen_entropy_path")
                    vattn_path = data.get("gen_vattn_path")

                    if not entropy_path or not vattn_path:
                        continue

                    entropy_path = Path(entropy_path)
                    vattn_path = Path(vattn_path)

                    if not entropy_path.exists() or not vattn_path.exists():
                        continue

                    with open(entropy_path, "rb") as f:
                        entropy = pickle.load(f)
                        entropy = entropy.squeeze().numpy()

                    with open(vattn_path, "rb") as f:
                        vattn = pickle.load(f)
                        vattn = vattn.float().squeeze().numpy()

                    min_len = min(len(entropy), len(vattn))
                    entropy = entropy[:min_len]
                    vattn = vattn[:min_len]

                    if len(entropy) < 50:
                        continue

                    n_tokens = len(entropy)
                    n_high = max(1, int(n_tokens * top_percent))

                    # 高熵token索引
                    high_entropy_indices = np.argsort(entropy)[-n_high:]
                    low_entropy_indices = np.argsort(entropy)[:-n_high]

                    # 相对位置 (0-1)
                    high_positions = high_entropy_indices / n_tokens
                    low_positions = low_entropy_indices / n_tokens

                    positions.extend(high_positions)
                    entropies.extend(entropy[high_entropy_indices])

                    all_entropy_positions.extend(high_positions)
                    all_entropy_values.extend(entropy[high_entropy_indices])
                    all_low_entropy_positions.extend(low_positions)
                    all_low_entropy_values.extend(entropy[low_entropy_indices])

                    # HEVA
                    heva = vattn[high_entropy_indices].mean()
                    heva_list.append(heva)
                    all_heva.append(heva)

            except Exception as e:
                continue

        if positions:
            bench_stats[bench_name] = {
                "positions": positions,
                "entropies": entropies,
                "heva": heva_list,
            }

    all_entropy_positions = np.array(all_entropy_positions)
    all_entropy_values = np.array(all_entropy_values)
    all_low_entropy_positions = np.array(all_low_entropy_positions)
    all_low_entropy_values = np.array(all_low_entropy_values)
    all_heva = np.array(all_heva)

    print(f"\n{'='*60}")
    print("高熵Token特征分析 (理解Instruct模型)")
    print(f"{'='*60}")

    # 1. 高熵token的位置分布
    print(f"\n1. 高熵Token位置分布 (0=开头, 1=结尾):")
    print(f"   均值: {all_entropy_positions.mean():.4f}")
    print(f"   中位数: {np.median(all_entropy_positions):.4f}")
    print(f"   标准差: {all_entropy_positions.std():.4f}")

    # 按位置区间统计
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ["开头0-25%", "25-50%", "50-75%", "结尾75-100%"]
    print(f"\n   按位置区间分布:")
    for i in range(len(bins) - 1):
        mask = (all_entropy_positions >= bins[i]) & (all_entropy_positions < bins[i + 1])
        pct = mask.sum() / len(all_entropy_positions) * 100
        print(f"   {labels[i]}: {pct:.1f}%")

    # 对比低熵token位置
    print(f"\n2. 高熵 vs 低熵Token位置对比:")
    print(f"   高熵token位置均值: {all_entropy_positions.mean():.4f}")
    print(f"   低熵token位置均值: {all_low_entropy_positions.mean():.4f}")

    # t检验
    t_pos, p_pos = stats.ttest_ind(all_entropy_positions, all_low_entropy_positions)
    print(f"   位置差异t检验: t={t_pos:.4f}, p={p_pos:.4e}")

    # 3. 熵值分布
    print(f"\n3. 熵值分布:")
    print(f"   高熵token熵值: 均值={all_entropy_values.mean():.4f}, 范围=[{all_entropy_values.min():.4f}, {all_entropy_values.max():.4f}]")
    print(
        f"   低熵token熵值: 均值={all_low_entropy_values.mean():.4f}, 范围=[{all_low_entropy_values.min():.4f}, {all_low_entropy_values.max():.4f}]"
    )

    # 4. 理解Instruct模型高熵token的含义
    print(f"\n" + "=" * 60)
    print("Instruct模型高熵token含义推断")
    print("=" * 60)

    # 分析开头部分高熵token的比例
    head_ratio = (all_entropy_positions < 0.3).mean()
    tail_ratio = (all_entropy_positions > 0.7).mean()

    print(f"   开头30%位置的高熵token占比: {head_ratio:.1%}")
    print(f"   结尾30%位置的高熵token占比: {tail_ratio:.1%}")

    # 对比思考token的可能性
    # 如果高熵token集中在开头，可能是系统prompt/指令
    # 如果高熵token分散在整个回答，可能是内容不确定性高
    if head_ratio > 0.4:
        meaning = "可能包含大量系统指令/格式token，这些token本身就具有高不确定性"
    elif tail_ratio > 0.4:
        meaning = "可能主要是回答结尾的开放式生成，具有较高的词汇不确定性"
    else:
        meaning = "高熵token分散分布，可能表示内容生成的不确定性"

    print(f"\n   推断: {meaning}")

    return {
        "positions": all_entropy_positions,
        "entropies": all_entropy_values,
        "heva": all_heva,
        "bench_stats": bench_stats,
    }


# 分析高熵token对应的具体文本内容
def analyze_high_entropy_text(exp_dir, top_percent=0.2, num_samples=5):
    """
    分析高熵token对应的具体文本内容，理解其含义
    """
    from transformers import AutoTokenizer
    import pickle

    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained("../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking", trust_remote_code=True)
    except:
        print("Warning: Cannot load tokenizer, skipping text analysis")
        return

    exp_path = Path(exp_dir)

    print(f"\n{'='*60}")
    print("高熵Token文本内容分析")
    print(f"{'='*60}")

    sample_count = 0

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name

        pkls_dir = bench_dir / "pkls"
        if not pkls_dir.exists():
            continue

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                generated_text = data.get("generated_text", "")
                if not generated_text:
                    continue

                entropy_path = data.get("gen_entropy_path")
                if not entropy_path:
                    continue

                entropy_path = Path(entropy_path)
                if not entropy_path.exists():
                    continue

                # 分词
                tokens = tokenizer.encode(generated_text, add_special_tokens=False)

                # 加载熵
                with open(entropy_path, "rb") as f:
                    entropy = pickle.load(f)
                entropy = entropy.squeeze().numpy()

                min_len = min(len(entropy), len(tokens))
                entropy = entropy[:min_len]
                tokens = tokens[:min_len]

                if len(entropy) < 50:
                    continue

                # 找高熵token
                n_high = max(1, int(len(entropy) * top_percent))
                high_entropy_indices = sorted(np.argsort(entropy)[-n_high:])

                # 按位置分组显示
                if sample_count < num_samples:
                    print(f"\n--- {bench_name} (样本 {sample_count+1}) ---")
                    print(f"总token数: {len(entropy)}, 高熵token数: {len(high_entropy_indices)}")
                    print(f"高熵token位置: 均值={np.mean(high_entropy_indices)/len(entropy):.3f}")

                    # 显示开头的高熵token
                    print(f"\n开头部分高熵token (前10个):")
                    for idx in high_entropy_indices[:10]:
                        if idx < len(tokens):
                            token_text = tokenizer.decode([tokens[idx]])
                            print(f'  [{idx:4d}] pos={idx/len(entropy):.3f}, entropy={entropy[idx]:.4f}, text="{token_text}"')

                    # 显示中间的高熵token
                    mid_start = len(high_entropy_indices) // 3
                    mid_end = 2 * len(high_entropy_indices) // 3
                    print(f"\n中间部分高熵token (中间10个):")
                    for idx in high_entropy_indices[mid_start : mid_start + 10]:
                        if idx < len(tokens):
                            token_text = tokenizer.decode([tokens[idx]])
                            print(f'  [{idx:4d}] pos={idx/len(entropy):.3f}, entropy={entropy[idx]:.4f}, text="{token_text}"')

                    # 显示结尾的高熵token
                    print(f"\n结尾部分高熵token (最后10个):")
                    for idx in high_entropy_indices[-10:]:
                        if idx < len(tokens):
                            token_text = tokenizer.decode([tokens[idx]])
                            print(f'  [{idx:4d}] pos={idx/len(entropy):.3f}, entropy={entropy[idx]:.4f}, text="{token_text}"')

                    sample_count += 1

            except Exception as e:
                continue

        if sample_count >= num_samples:
            break

    # 统计高频高熵token
    print(f"\n" + "=" * 60)
    print("高频高熵Token统计")
    print(f"=" * 60)

    # 收集多个样本的高熵token
    all_high_tokens = []
    all_high_token_texts = []

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        pkls_dir = bench_dir / "pkls"
        if not pkls_dir.exists():
            continue

        for json_file in bench_dir.glob("*_meta.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                generated_text = data.get("generated_text", "")
                if not generated_text:
                    continue

                entropy_path = data.get("gen_entropy_path")
                if not entropy_path:
                    continue

                entropy_path = Path(entropy_path)
                if not entropy_path.exists():
                    continue

                tokens = tokenizer.encode(generated_text, add_special_tokens=False)

                with open(entropy_path, "rb") as f:
                    entropy = pickle.load(f)
                entropy = entropy.squeeze().numpy()

                min_len = min(len(entropy), len(tokens))
                entropy = entropy[:min_len]
                tokens = tokens[:min_len]

                if len(entropy) < 50:
                    continue

                n_high = max(1, int(len(entropy) * top_percent))
                high_entropy_indices = np.argsort(entropy)[-n_high:]

                for idx in high_entropy_indices:
                    if idx < len(tokens):
                        all_high_tokens.append(tokens[idx])
                        all_high_token_texts.append(tokenizer.decode([tokens[idx]]))

            except Exception as e:
                continue

    # 统计最常见的高熵token
    from collections import Counter

    token_counts = Counter(all_high_token_texts)
    print(f"\n最常见的高熵token (top 30):")
    for text, count in token_counts.most_common(30):
        print(f'  "{text}": {count}')

    return {}


# 对比Thinking vs Instruct模型的高熵token模式
def compare_thinking_vs_instruct(thinking_exp_dir, instruct_exp_dir, top_percent=0.2):
    """
    对比Thinking和Instruct模型的高熵token模式差异
    """
    import pickle

    print(f"\n{'='*60}")
    print("Thinking vs Instruct 模型高熵Token模式对比")
    print(f"{'='*60}")

    # 收集两个实验的数据
    def collect_data(exp_dir):
        exp_path = Path(exp_dir)
        sample_hevas = []
        sample_positions = []
        sample_entropies = []

        for bench_dir in sorted(exp_path.iterdir()):
            if not bench_dir.is_dir():
                continue

            pkls_dir = bench_dir / "pkls"
            if not pkls_dir.exists():
                continue

            for json_file in bench_dir.glob("*_meta.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    entropy_path = data.get("gen_entropy_path")
                    vattn_path = data.get("gen_vattn_path")

                    if not entropy_path or not vattn_path:
                        continue

                    entropy_path = Path(entropy_path)
                    vattn_path = Path(vattn_path)

                    if not entropy_path.exists() or not vattn_path.exists():
                        continue

                    with open(entropy_path, "rb") as f:
                        entropy = pickle.load(f)
                        entropy = entropy.squeeze().numpy()

                    with open(vattn_path, "rb") as f:
                        vattn = pickle.load(f)
                        vattn = vattn.float().squeeze().numpy()

                    min_len = min(len(entropy), len(vattn))
                    entropy = entropy[:min_len]
                    vattn = vattn[:min_len]

                    if len(entropy) < 50:
                        continue

                    n_tokens = len(entropy)
                    n_high = max(1, int(n_tokens * top_percent))
                    high_entropy_indices = np.argsort(entropy)[-n_high:]

                    # 高熵token的平均位置
                    avg_pos = high_entropy_indices.mean() / n_tokens
                    sample_positions.append(avg_pos)

                    # HEVA
                    sample_hevas.append(vattn[high_entropy_indices].mean())

                    # 平均熵
                    sample_entropies.append(entropy[high_entropy_indices].mean())

                except Exception as e:
                    continue

        return np.array(sample_hevas), np.array(sample_positions), np.array(sample_entropies)

    thinking_hevas, thinking_positions, thinking_entropies = collect_data(thinking_exp_dir)
    instruct_hevas, instruct_positions, instruct_entropies = collect_data(instruct_exp_dir)

    print(f"\n1. HEVA对比:")
    print(f"   Thinking模型: 均值={thinking_hevas.mean():.6f}, 中位数={np.median(thinking_hevas):.6f}")
    print(f"   Instruct模型: 均值={instruct_hevas.mean():.6f}, 中位数={np.median(instruct_hevas):.6f}")

    print(f"\n2. 高熵Token位置对比:")
    print(f"   Thinking模型: 均值={thinking_positions.mean():.4f}, 中位数={np.median(thinking_positions):.4f}")
    print(f"   Instruct模型: 均值={instruct_positions.mean():.4f}, 中位数={np.median(instruct_positions):.4f}")

    print(f"\n3. 高熵Token熵值对比:")
    print(f"   Thinking模型: 均值={thinking_entropies.mean():.4f}")
    print(f"   Instruct模型: 均值={instruct_entropies.mean():.4f}")

    print(f"\n4. 位置分布详细对比:")
    print(f"   {'模型':<15} {'开头30%':<12} {'中间40%':<12} {'结尾30%':<12}")
    print(f"   {'-'*51}")

    for name, positions in [("Thinking", thinking_positions), ("Instruct", instruct_positions)]:
        head = (positions < 0.3).mean() * 100
        mid = ((positions >= 0.3) & (positions < 0.7)).mean() * 100
        tail = (positions >= 0.7).mean() * 100
        print(f"   {name:<15} {head:<12.1f} {mid:<12.1f} {tail:<12.1f}")

    print(f"\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    if thinking_positions.mean() < instruct_positions.mean():
        print(
            f"   Thinking模型: 高熵token更集中在回答{'' if thinking_positions.mean() < 0.3 else '中间'}偏{'' if thinking_positions.mean() < 0.3 else '后'}位置"
        )
        print(f"   → 这些位置可能对应模型'思考/推理'过程")
    else:
        print(f"   Instruct模型: 高熵token位置分布与Thinking不同")

    return {
        "thinking": {"hevas": thinking_hevas, "positions": thinking_positions, "entropies": thinking_entropies},
        "instruct": {"hevas": instruct_hevas, "positions": instruct_positions, "entropies": instruct_entropies},
    }


if __name__ == "__main__":
    exp_dir = "./results/exp001"

    # 统计ACC
    calculate_acc(exp_dir)
    calculate_acc(exp_dir[:-1] + "2")

    # 分析回答长度与正确性相关性
    analyze_response_length_correlation(exp_dir)
    analyze_response_length_correlation(exp_dir[:-1] + "2")

    # 分析token熵与视觉注意力相关性
    analyze_entropy_vattn_correlation(exp_dir)
    analyze_entropy_vattn_correlation(exp_dir[:-1] + "2")

    # 分析高熵token是否有更高的视觉注意力
    analyze_high_entropy_vattn(exp_dir)
    analyze_high_entropy_vattn(exp_dir[:-1] + "2")

    # 分析HEVA与回答正确性的关系
    analyze_vattn_correctness(exp_dir)
    analyze_vattn_correctness(exp_dir[:-1] + "2")

    # 验证假设：高熵token的HEVA更高 -> 更容易回答正确
    verify_heva_hypothesis(exp_dir)
    verify_heva_hypothesis(exp_dir[:-1] + "2")

    # 分析高熵token的特征（理解Instruct模型）
    analyze_entropy_token_patterns(exp_dir)
    analyze_entropy_token_patterns(exp_dir[:-1] + "2")

    # 分析高熵token对应的文本内容
    analyze_high_entropy_text(exp_dir, top_percent=0.2, num_samples=5)
    analyze_high_entropy_text(exp_dir[:-1] + "2", top_percent=0.2, num_samples=5)

    # 对比Thinking vs Instruct模型
    compare_thinking_vs_instruct("./results/exp001", "./results/exp002")
