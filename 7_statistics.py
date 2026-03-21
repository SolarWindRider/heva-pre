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
def analyze_entropy_heva_correlation(exp_dir):
    """
    分析每个样本中token熵与视觉注意力(HEVA)的相关性

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
                    min_len = min(len(entropy), len(vattn))
                    entropy = entropy[:min_len]
                    vattn = vattn[:min_len]

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
    print("Token熵与视觉注意力(HEVA)相关性分析")
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
        print(f"{bench_name:<20} {len(stats_dict['correlations']):<8} "
              f"{stats_dict['mean_corr']:<10.4f} {stats_dict['median_corr']:<10.4f} "
              f"{stats_dict['std_corr']:<10.4f}")

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
def analyze_high_entropy_heva(exp_dir, top_percent=0.2):
    """
    分析高熵token是否具有更高的视觉注意力

    设计方案：
    1. 对每个样本，选取熵值最高的top_percent% tokens
    2. 计算这些高熵token的平均视觉注意力 (即HEVA)
    3. 计算剩余token的平均视觉注意力
    4. 比较两者的差异

    Args:
        exp_dir: 实验路径
        top_percent: 高熵token的比例，默认20%
    """
    import pickle

    exp_path = Path(exp_dir)

    # 收集数据
    all_heva_high_entropy = []  # 高熵token的平均视觉注意力
    all_heva_low_entropy = []   # 低熵token的平均视觉注意力
    all_avg_entropy = []         # 样本平均熵

    bench_results = {}

    for bench_dir in sorted(exp_path.iterdir()):
        if not bench_dir.is_dir():
            continue

        bench_name = bench_dir.name
        heva_high_list = []
        heva_low_list = []
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

                    min_len = min(len(entropy), len(vattn))
                    entropy = entropy[:min_len]
                    vattn = vattn[:min_len]

                    if len(entropy) < 50:
                        continue

                    # 计算HEVA：取熵值最高的top_percent% token
                    n_high = max(1, int(len(entropy) * top_percent))
                    high_entropy_indices = np.argsort(entropy)[-n_high:]
                    low_entropy_indices = np.argsort(entropy)[:-n_high]

                    heva_high = vattn[high_entropy_indices].mean()
                    heva_low = vattn[low_entropy_indices].mean()

                    heva_high_list.append(heva_high)
                    heva_low_list.append(heva_low)
                    avg_entropy_list.append(entropy.mean())

                    all_heva_high_entropy.append(heva_high)
                    all_heva_low_entropy.append(heva_low)
                    all_avg_entropy.append(entropy.mean())

            except Exception as e:
                continue

        if heva_high_list:
            bench_results[bench_name] = {
                "heva_high": heva_high_list,
                "heva_low": heva_low_list,
                "avg_entropy": avg_entropy_list,
                "diff": np.mean(heva_high_list) - np.mean(heva_low_list),
            }

    all_heva_high_entropy = np.array(all_heva_high_entropy)
    all_heva_low_entropy = np.array(all_heva_low_entropy)
    all_avg_entropy = np.array(all_avg_entropy)

    print(f"\n{'='*60}")
    print(f"高熵Token (top {top_percent*100:.0f}%) 视觉注意力分析")
    print(f"{'='*60}")

    # 整体统计
    print(f"\n整体统计 (N={len(all_heva_high_entropy)}):")
    print(f"  高熵token平均HEVA: {np.mean(all_heva_high_entropy):.6f}")
    print(f"  低熵token平均HEVA: {np.mean(all_heva_low_entropy):.6f}")
    print(f"  差值 (高-低): {np.mean(all_heva_high_entropy) - np.mean(all_heva_low_entropy):.6f}")
    print(f"  比值 (高/低): {np.mean(all_heva_high_entropy) / np.mean(all_heva_low_entropy):.4f}")

    # 统计检验
    t_stat, p_value = stats.ttest_rel(all_heva_high_entropy, all_heva_low_entropy)
    print(f"\n配对t检验: t={t_stat:.4f}, p={p_value:.4e}")

    # 计算样本级别的相关性：平均熵 vs HEVA
    corr_entropy_heva, p_val = stats.pearsonr(all_avg_entropy, all_heva_high_entropy)
    print(f"\n样本平均熵 vs 高熵token平均HEVA:")
    print(f"  相关系数: {corr_entropy_heva:.4f}")
    print(f"  P值: {p_val:.4e}")

    # 按benchmark统计
    print(f"\n各Benchmark统计:")
    print(f"{'Benchmark':<20} {'High-HEVA':<12} {'Low-HEVA':<12} {'Diff':<12} {'Ratio':<10}")
    print("-" * 66)
    for bench_name, stats_dict in sorted(bench_results.items()):
        high_mean = np.mean(stats_dict["heva_high"])
        low_mean = np.mean(stats_dict["heva_low"])
        diff = high_mean - low_mean
        ratio = high_mean / low_mean if low_mean > 0 else 0
        print(f"{bench_name:<20} {high_mean:<12.6f} {low_mean:<12.6f} {diff:<12.6f} {ratio:<10.4f}")

    # 分析：验证HEVA假设
    print(f"\n" + "="*60)
    print("结论分析")
    print("="*60)
    diff_mean = np.mean(all_heva_high_entropy) - np.mean(all_heva_low_entropy)
    if diff_mean > 0 and p_value < 0.05:
        print(f"✓ 高熵token的视觉注意力显著更高 (差值: {diff_mean:.6f})")
        print(f"  这支持HEVA的核心假设：高熵token更依赖视觉信息")
    else:
        print(f"✗ 高熵token的视觉注意力无显著差异 (差值: {diff_mean:.6f})")

    return {
        "heva_high_entropy": all_heva_high_entropy,
        "heva_low_entropy": all_heva_low_entropy,
        "avg_entropy": all_avg_entropy,
        "bench_results": bench_results,
    }


# 分析高熵token的HEVA与回答正确性的关系
def analyze_heva_correctness(exp_dir, top_percent=0.2):
    """
    分析高熵token的HEVA与回答正确性的关系

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

    # 收集数据：HEVA, 正确性, 熵
    all_heva = []  # 高熵token的平均视觉注意力 (即HEVA)
    all_correct = []  # 是否正确
    all_avg_entropy = []  # 平均熵
    all_heva_ratio = []  # 高熵token HEVA / 低熵token HEVA

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
                    heva_low = vattn[low_entropy_indices].mean()
                    heva_ratio = heva / heva_low if heva_low > 0 else 0

                    heva_list.append(heva)
                    correct_list.append(1 if correct else 0)
                    avg_entropy_list.append(entropy.mean())
                    heva_ratio_list.append(heva_ratio)

                    all_heva.append(heva)
                    all_correct.append(1 if correct else 0)
                    all_avg_entropy.append(entropy.mean())
                    all_heva_ratio.append(heva_ratio)

            except Exception as e:
                continue

        if heva_list:
            bench_results[bench_name] = {
                "heva": heva_list,
                "correct": correct_list,
                "avg_entropy": avg_entropy_list,
                "heva_ratio": heva_ratio_list,
            }

    all_heva = np.array(all_heva)
    all_correct = np.array(all_correct)
    all_avg_entropy = np.array(all_avg_entropy)
    all_heva_ratio = np.array(all_heva_ratio)

    print(f"\n{'='*60}")
    print(f"高熵Token HEVA与回答正确性关系分析")
    print(f"{'='*60}")

    # 1. 计算HEVA与正确性的相关性
    corr_heva_correct, p_val1 = stats.pearsonr(all_heva, all_correct)
    corr_ratio_correct, p_val2 = stats.pearsonr(all_heva_ratio, all_correct)

    print(f"\n1. HEVA与正确性相关性:")
    print(f"   HEVA vs 正确性: r={corr_heva_correct:.4f}, p={p_val1:.4e}")
    print(f"   HEVA比值 vs 正确性: r={corr_ratio_correct:.4f}, p={p_val2:.4e}")

    # 2. 按HEVA高低分组统计正确率
    print(f"\n2. 按HEVA高低分组统计正确率:")
    heva_median = np.median(all_heva)

    low_heva_mask = all_heva < heva_median
    high_heva_mask = all_heva >= heva_median

    low_heva_acc = all_correct[low_heva_mask].mean()
    high_heva_acc = all_correct[high_heva_mask].mean()

    print(f"   低HEVA组 (N={low_heva_mask.sum()}): 正确率={low_heva_acc:.4f}")
    print(f"   高HEVA组 (N={high_heva_mask.sum()}): 正确率={high_heva_acc:.4f}")

    # 3. 按HEVA四分位数分组
    print(f"\n3. 按HEVA四分位数分组:")
    q25 = np.percentile(all_heva, 25)
    q50 = np.percentile(all_heva, 50)
    q75 = np.percentile(all_heva, 75)

    bins = [0, q25, q50, q75, float('inf')]
    labels = ['Q1(最低)', 'Q2', 'Q3', 'Q4(最高)']

    print(f"   分位点: Q25={q25:.6f}, Q50={q50:.6f}, Q75={q75:.6f}")
    print(f"\n   {'分组':<12} {'正确率':<10} {'样本数':<10}")
    print(f"   {'-'*32}")
    for i in range(len(bins) - 1):
        mask = (all_heva >= bins[i]) & (all_heva < bins[i + 1])
        if mask.sum() > 0:
            acc = all_correct[mask].mean()
            count = mask.sum()
            print(f"   {labels[i]:<12} {acc:<10.4f} {count:<10}")

    # 4. 按HEVA比值分组 (高熵token HEVA / 低熵token HEVA)
    print(f"\n4. 按HEVA比值分组 (高/低):")
    ratio_median = np.median(all_heva_ratio)

    low_ratio_mask = all_heva_ratio < ratio_median
    high_ratio_mask = all_heva_ratio >= ratio_median

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
    print(f"\n" + "="*60)
    print("结论分析")
    print("="*60)

    if high_heva_acc > low_heva_acc and p_val1 < 0.05:
        print(f"✓ 高HEVA样本正确率更高 ({high_heva_acc:.2%} vs {low_heva_acc:.2%})")
    elif high_heva_acc < low_heva_acc and p_val1 < 0.05:
        print(f"✗ 低HEVA样本正确率更高 ({low_heva_acc:.2%} vs {high_heva_acc:.2%})")
    else:
        print(f"○ HEVA与正确性无显著关联")

    return {
        "heva": all_heva,
        "correct": all_correct,
        "heva_ratio": all_heva_ratio,
        "bench_results": bench_results,
    }


if __name__ == "__main__":
    exp_dir = "./results/exp001"

    # 统计ACC
    # calculate_acc(exp_dir)

    # 分析回答长度与正确性相关性
    # analyze_response_length_correlation(exp_dir)

    # 分析token熵与视觉注意力相关性
    analyze_entropy_heva_correlation(exp_dir)

    # 分析高熵token是否有更高的视觉注意力
    analyze_high_entropy_heva(exp_dir)

    # 分析HEVA与回答正确性的关系
    analyze_heva_correctness(exp_dir)
