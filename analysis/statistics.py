"""
统计分析模块

实现:
1. t-test
2. Cohen's d
3. Pearson 相关系数
4. Bootstrap 置信区间
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Any


def compute_ttest(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    """
    独立样本 t 检验

    Args:
        group1: 第一组数据
        group2: 第二组数据

    Returns:
        {
            "t_statistic": float,
            "p_value": float,
            "mean1": float,
            "mean2": float,
            "std1": float,
            "std2": float
        }
    """
    t_stat, p_val = stats.ttest_ind(group1, group2)

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "mean1": np.mean(group1),
        "mean2": np.mean(group2),
        "std1": np.std(group1, ddof=1),
        "std2": np.std(group2, ddof=1)
    }


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    计算 Cohen's d 效应量

    Args:
        group1: 第一组数据
        group2: 第二组数据

    Returns:
        Cohen's d 值
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_pearson(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Pearson 相关系数

    Args:
        x: x 数据
        y: y 数据

    Returns:
        {
            "r": float,
            "p_value": float
        }
    """
    r, p_val = stats.pearsonr(x, y)

    return {
        "r": r,
        "p_value": p_val
    }


def compute_spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Spearman 相关系数

    Args:
        x: x 数据
        y: y 数据

    Returns:
        {
            "rho": float,
            "p_value": float
        }
    """
    rho, p_val = stats.spearmanr(x, y)

    return {
        "rho": rho,
        "p_value": p_val
    }


def compute_bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95
) -> Dict[str, Any]:
    """
    Bootstrap 置信区间

    Args:
        data: 数据数组
        statistic_func: 统计函数（默认为 mean）
        n_bootstrap: Bootstrap 次数
        ci: 置信水平

    Returns:
        {
            "statistic": float,
            "ci_lower": float,
            "ci_upper": float
        }
    """
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)

    return {
        "statistic": statistic_func(data),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_std": np.std(bootstrap_stats)
    }


def compute_cliff_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cliff's Delta 效应量（非参数）

    Args:
        group1: 第一组数据
        group2: 第二组数据

    Returns:
        Cliff's delta 值 (-1 到 1)
    """
    n1, n2 = len(group1), len(group2)

    # 计算 dominance
    dominance = 0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1

    return dominance / (n1 * n2)


def compute_wilcoxon(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    """
    Wilcoxon 符号秩检验（配对样本）

    Args:
        group1: 第一组数据
        group2: 第二组数据

    Returns:
        {
            "statistic": float,
            "p_value": float
        }
    """
    stat, p_val = stats.wilcoxon(group1, group2)

    return {
        "statistic": stat,
        "p_value": p_val
    }


def compute_mann_whitney(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    """
    Mann-Whitney U 检验（非参数）

    Args:
        group1: 第一组数据
        group2: 第二组数据

    Returns:
        {
            "U_statistic": float,
            "p_value": float
        }
    """
    stat, p_val = stats.mannwhitneyu(group1, group2)

    return {
        "U_statistic": stat,
        "p_value": p_val
    }


def compute_summary_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    汇总统计

    Args:
        data: 数据数组

    Returns:
        {
            "mean": float,
            "std": float,
            "median": float,
            "min": float,
            "max": float,
            "q25": float,
            "q75": float
        }
    """
    return {
        "mean": np.mean(data),
        "std": np.std(data, ddof=1),
        "median": np.median(data),
        "min": np.min(data),
        "max": np.max(data),
        "q25": np.percentile(data, 25),
        "q75": np.percentile(data, 75)
    }


if __name__ == "__main__":
    # 测试
    group1 = np.random.randn(100) + 1
    group2 = np.random.randn(100)

    print("t-test:", compute_ttest(group1, group2))
    print("Cohen's d:", compute_cohens_d(group1, group2))
    print("Cliff delta:", compute_cliff_delta(group1, group2))
    print("Summary:", compute_summary_statistics(group1))
