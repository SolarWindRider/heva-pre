"""
统计分析脚本

对实验结果进行统计分析并生成表格和图表
"""

import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.statistics import (
    compute_ttest, compute_cohens_d, compute_cliff_delta,
    compute_pearson, compute_summary_statistics, compute_bootstrap_ci
)
from analysis.plots import (
    plot_heva_perturbation, plot_group_comparison,
    plot_heva_distribution, plot_correlation
)


def load_inference_results(path):
    """加载推理结果"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['results']


def analyze_perturbation_results(results):
    """分析扰动结果"""
    print("\n" + "="*50)
    print("Perturbation Analysis")
    print("="*50)

    df = pd.DataFrame(results)

    # 按条件分组统计
    grouped = df.groupby('condition').agg({
        'heva': ['mean', 'std', 'count'],
        'correct': ['mean']
    })

    print("\nHEVA by Condition:")
    print(grouped)

    # 与 original 比较
    original_heva = df[df['condition'] == 'original']['heva'].values

    conditions = df['condition'].unique()
    for cond in conditions:
        if cond == 'original':
            continue
        cond_heva = df[df['condition'] == cond]['heva'].values

        if len(original_heva) > 0 and len(cond_heva) > 0:
            # t-test
            ttest = compute_ttest(original_heva, cond_heva)
            print(f"\n{cond} vs original:")
            print(f"  t-statistic: {ttest['t_statistic']:.4f}")
            print(f"  p-value: {ttest['p_value']:.4f}")
            print(f"  Cohen's d: {compute_cohens_d(original_heva, cond_heva):.4f}")


def analyze_group_results(results):
    """分析分组结果"""
    print("\n" + "="*50)
    print("Group Analysis")
    print("="*50)

    df = pd.DataFrame(results)

    visual_samples = df[df['group'] == 'visual-critical']
    language_samples = df[df['group'] == 'language-guessable']

    print(f"\nVisual-Critical: {len(visual_samples)} samples")
    print(f"Language-Guessable: {len(language_samples)} samples")

    if len(visual_samples) > 0 and len(language_samples) > 0:
        visual_heva = visual_samples['heva'].values
        language_heva = language_samples['heva'].values

        # t-test
        ttest = compute_ttest(visual_heva, language_heva)
        print(f"\nt-test:")
        print(f"  t-statistic: {ttest['t_statistic']:.4f}")
        print(f"  p-value: {ttest['p_value']:.4f}")
        print(f"  Mean difference: {ttest['mean1'] - ttest['mean2']:.4f}")

        # Cohen's d
        cohens_d = compute_cohens_d(visual_heva, language_heva)
        print(f"\nCohen's d: {cohens_d:.4f}")

        # Cliff's delta
        cliff = compute_cliff_delta(visual_heva, language_heva)
        print(f"Cliff's delta: {cliff:.4f}")

        # Bootstrap CI
        print("\nBootstrap 95% CI:")
        ci_visual = compute_bootstrap_ci(visual_heva)
        ci_language = compute_bootstrap_ci(language_heva)
        print(f"  Visual-Critical: [{ci_visual['ci_lower']:.4f}, {ci_visual['ci_upper']:.4f}]")
        print(f"  Language-Guessable: [{ci_language['ci_lower']:.4f}, {ci_language['ci_upper']:.4f}]")

        # 准确率比较
        visual_acc = visual_samples['correct'].mean()
        language_acc = language_samples['correct'].mean()
        print(f"\nAccuracy:")
        print(f"  Visual-Critical: {visual_acc:.2%}")
        print(f"  Language-Guessable: {language_acc:.2%}")

    return {
        'visual_heva': visual_heva if len(visual_samples) > 0 else np.array([]),
        'language_heva': language_heva if len(language_samples) > 0 else np.array([])
    }


def generate_plots(results, output_dir):
    """生成图表"""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    # HEVA 分布
    plot_heva_distribution(
        df['heva'].values,
        title="HEVA Distribution",
        save_path=os.path.join(output_dir, 'heva_distribution.png')
    )

    # 分组对比
    if 'group' in df.columns:
        visual_heva = df[df['group'] == 'visual-critical']['heva'].values
        language_heva = df[df['group'] == 'language-guessable']['heva'].values

        if len(visual_heva) > 0 and len(language_heva) > 0:
            plot_group_comparison(
                visual_heva, language_heva,
                group1_name="Visual-Critical",
                group2_name="Language-Guessable",
                save_path=os.path.join(output_dir, 'group_comparison.png')
            )

    # 扰动对比
    if 'condition' in df.columns:
        plot_heva_perturbation(
            results,
            save_path=os.path.join(output_dir, 'perturbation_comparison.png')
        )


def main():
    import config
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results_dir', type=str, default=config.RESULTS_DIR,
                        help='Results directory')
    parser.add_argument('--output_dir', type=str, default=config.PLOTS_DIR,
                        help='Output directory for plots')

    args = parser.parse_args()

    # 分析推理结果
    inference_path = os.path.join(args.results_dir, 'inference_results.pkl')
    if os.path.exists(inference_path):
        print("Analyzing inference results...")
        results = load_inference_results(inference_path)
        df = pd.DataFrame(results)

        print("\n" + "="*50)
        print("Inference Results Summary")
        print("="*50)
        print(f"Total samples: {len(results)}")
        print(f"Mean HEVA: {df['heva'].mean():.4f}")
        print(f"Std HEVA: {df['heva'].std():.4f}")
        print(f"Accuracy: {df['correct'].mean():.2%}")

        # HEVA 分布统计
        print("\nHEVA Statistics:")
        stats = compute_summary_statistics(df['heva'].values)
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")

        # 生成图表
        generate_plots(results, args.output_dir)

    # 分析分组结果
    group_path = os.path.join(args.results_dir, 'group_analysis.pkl')
    if os.path.exists(group_path):
        with open(group_path, 'rb') as f:
            group_data = pickle.load(f)
        analyze_group_results(group_data['results'])

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
