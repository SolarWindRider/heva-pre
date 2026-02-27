"""
可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Tuple
import pandas as pd


def set_style():
    """设置绘图样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_heva_perturbation(
    results: List[Dict[str, Any]],
    save_path: str = None
):
    """
    绘制 HEVA vs 扰动类型的图

    Args:
        results: 结果列表，每个包含 condition 和 heva
        save_path: 保存路径
    """
    set_style()

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))

    # 绘制箱线图
    sns.boxplot(data=df, x='condition', y='heva')

    # 添加数据点
    sns.stripplot(data=df, x='condition', y='heva', color='black', alpha=0.3, size=4)

    plt.xlabel('Perturbation Type')
    plt.ylabel('HEVA')
    plt.title('HEVA vs Image Perturbation')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_group_comparison(
    group1_heva: np.ndarray,
    group2_heva: np.ndarray,
    group1_name: str = "Visual-Critical",
    group2_name: str = "Language-Guessable",
    save_path: str = None
):
    """
    绘制两组 HEVA 对比图

    Args:
        group1_heva: 第一组 HEVA 值
        group2_heva: 第二组 HEVA 值
        group1_name: 第一组名称
        group2_name: 第二组名称
        save_path: 保存路径
    """
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 柱状图
    means = [np.mean(group1_heva), np.mean(group2_heva)]
    stds = [np.std(group1_heva), np.std(group2_heva)]

    axes[0].bar([group1_name, group2_name], means, yerr=stds, capsize=5, alpha=0.7)
    axes[0].set_ylabel('HEVA')
    axes[0].set_title('Mean HEVA Comparison')

    # 添加数值标签
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom')

    # 核密度图
    sns.kdeplot(group1_heva, ax=axes[1], label=group1_name, fill=True, alpha=0.5)
    sns.kdeplot(group2_heva, ax=axes[1], label=group2_name, fill=True, alpha=0.5)
    axes[1].set_xlabel('HEVA')
    axes[1].set_ylabel('Density')
    axes[1].set_title('HEVA Distribution')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_temporal_dynamics(
    time_steps: List[int],
    entropies: np.ndarray,
    visual_attentions: np.ndarray,
    save_path: str = None
):
    """
    绘制时间动态曲线

    Args:
        time_steps: 时间步
        entropies: 熵值 (samples, time_steps)
        visual_attentions: 视觉注意力 (samples, time_steps)
        save_path: 保存路径
    """
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制平均曲线
    mean_entropy = np.mean(entropies, axis=0)
    std_entropy = np.std(entropies, axis=0)

    mean_va = np.mean(visual_attentions, axis=0)
    std_va = np.std(visual_attentions, axis=0)

    # 熵曲线
    axes[0].plot(time_steps, mean_entropy, 'b-', linewidth=2)
    axes[0].fill_between(time_steps, mean_entropy - std_entropy, mean_entropy + std_entropy, alpha=0.3)
    axes[0].set_xlabel('Generation Step')
    axes[0].set_ylabel('Entropy')
    axes[0].set_title('Entropy over Generation')

    # 视觉注意力曲线
    axes[1].plot(time_steps, mean_va, 'r-', linewidth=2)
    axes[1].fill_between(time_steps, mean_va - std_va, mean_va + std_va, alpha=0.3, color='red')
    axes[1].set_xlabel('Generation Step')
    axes[1].set_ylabel('Visual Attention')
    axes[1].set_title('Visual Attention over Generation')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_attention_heatmap(
    attention: np.ndarray,
    token_labels: List[str] = None,
    save_path: str = None
):
    """
    绘制 attention 热力图

    Args:
        attention: attention 矩阵 (seq_len, seq_len)
        token_labels: token 标签
        save_path: 保存路径
    """
    set_style()

    plt.figure(figsize=(12, 10))

    sns.heatmap(attention, cmap='viridis', square=True)

    if token_labels:
        plt.xticks(np.arange(len(token_labels)) + 0.5, token_labels, rotation=90, fontsize=8)
        plt.yticks(np.arange(len(token_labels)) + 0.5, token_labels, fontsize=8)

    plt.title('Attention Heatmap')
    plt.xlabel('Key')
    plt.ylabel('Query')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_correlation(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    save_path: str = None
):
    """
    绘制相关性散点图

    Args:
        x: x 数据
        y: y 数据
        xlabel: x 轴标签
        ylabel: y 轴标签
        save_path: 保存路径
    """
    set_style()

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, alpha=0.5)

    # 添加趋势线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(x), p(np.sort(x)), 'r--', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{xlabel} vs {ylabel}')

    # 计算相关系数
    r = np.corrcoef(x, y)[0, 1]
    plt.text(0.05, 0.95, f'r = {r:.3f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_heva_distribution(
    heva_values: np.ndarray,
    title: str = "HEVA Distribution",
    save_path: str = None
):
    """
    绘制 HEVA 分布图

    Args:
        heva_values: HEVA 值数组
        title: 标题
        save_path: 保存路径
    """
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(heva_values, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(heva_values), color='red', linestyle='--', label=f'Mean: {np.mean(heva_values):.3f}')
    axes[0].set_xlabel('HEVA')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].legend()

    # 箱线图
    axes[1].boxplot(heva_values, vert=True)
    axes[1].set_ylabel('HEVA')
    axes[1].set_title(f'{title} - Boxplot')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # 测试
    print("Visualization module loaded successfully")
    print("Functions available:")
    print("  - plot_heva_perturbation")
    print("  - plot_group_comparison")
    print("  - plot_temporal_dynamics")
    print("  - plot_attention_heatmap")
    print("  - plot_correlation")
    print("  - plot_heva_distribution")
