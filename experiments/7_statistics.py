"""
统计分析脚本

对实验结果进行统计分析并生成表格和图表
支持多个数据集的独立分析
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


# 数据集列表
SUPPORTED_DATASETS = ["VisuRiddles", "RAVEN", "MARVEL", "LogicVista", "PuzzleVQA", "AlgoPuzzleVQA"]


def load_inference_results(dataset_dir):
    """
    加载推理结果

    支持两种格式:
    1. 新格式: results/{dataset}/index.json + {sample_id}_meta.json
    2. 旧格式: results/inference_{dataset}.pkl (兼容)
    """
    index_path = os.path.join(dataset_dir, 'index.json')

    # 尝试新格式
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)

        results = []
        for item in index_data.get('results', []):
            meta_path = item['meta_path']
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                results.append(meta)
            else:
                print(f"Warning: Meta file not found: {meta_path}")

        return results, index_data.get('errors', [])

    # 尝试旧格式兼容
    pkl_path = dataset_dir.replace('results/', 'results/inference_') + '.pkl'
    if not dataset_dir.endswith('.pkl'):
        pkl_path = dataset_dir + '.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data['results'], data.get('errors', [])

    raise FileNotFoundError(f"Cannot find results in {dataset_dir}")


def analyze_single_dataset(results, dataset_name):
    """分析单个数据集的结果"""
    print("\n" + "="*60)
    print(f"Dataset: {dataset_name}")
    print("="*60)

    # 新格式: results是metadata列表
    df = pd.DataFrame(results)

    stats = {
        'dataset': dataset_name,
        'num_samples': len(results),
    }

    print(f"\nTotal samples: {len(results)}")

    # 计算准确率 - 兼容新旧格式
    if 'correct' in df.columns:
        # 新格式: correct是布尔值
        accuracy = df['correct'].mean()
    elif 'accuracy' in df.columns:
        # 旧格式
        accuracy = df['accuracy'].mean()
    else:
        accuracy = 0.0
    stats['accuracy'] = accuracy
    print(f"Accuracy: {accuracy:.2%}")

    # HEVA 统计 - 支持新格式 (字典) 和旧格式 (数值)
    # 新格式: heva 是字典 {heva_10: x, heva_20: x, ..., heva_100: x}
    if 'heva' in df.columns and len(df['heva']) > 0:
        first_heva = df['heva'].iloc[0]
        if isinstance(first_heva, dict):
            # 新格式: 提取 heva_20 作为默认统计值
            df['heva_numeric'] = df['heva'].apply(lambda x: x.get('heva_20', x.get('heva_100', 0.0)) if isinstance(x, dict) else x)
            heva_col = df['heva_numeric']
            # 保存完整的 heva_dict 统计
            for alpha_key in ['heva_10', 'heva_20', 'heva_30', 'heva_40', 'heva_50', 'heva_60', 'heva_70', 'heva_80', 'heva_90', 'heva_100']:
                alpha_values = df['heva'].apply(lambda x: x.get(alpha_key, 0.0) if isinstance(x, dict) else 0.0)
                stats[f'{alpha_key}_mean'] = alpha_values.mean()
                stats[f'{alpha_key}_std'] = alpha_values.std()
        else:
            # 旧格式: 已经是数值
            heva_col = df['heva']
    else:
        heva_col = pd.Series([0.0] * len(df))

    mean_heva = heva_col.mean()
    std_heva = heva_col.std()
    stats['mean_heva'] = mean_heva
    stats['std_heva'] = std_heva

    print(f"\nHEVA Statistics (alpha=20%):")
    print(f"  Mean: {mean_heva:.4f}")
    print(f"  Std: {std_heva:.4f}")

    # 详细统计
    heva_values = heva_col.values
    detailed_stats = compute_summary_statistics(heva_values)
    for k, v in detailed_stats.items():
        stats[f'heva_{k}'] = v
        print(f"  {k}: {v:.4f}")

    return stats


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

    # 处理 heva 列 - 支持字典格式
    if 'heva' in df.columns and len(df['heva']) > 0:
        first_heva = df['heva'].iloc[0]
        if isinstance(first_heva, dict):
            df['heva_numeric'] = df['heva'].apply(lambda x: x.get('heva_20', x.get('heva_100', 0.0)) if isinstance(x, dict) else x)

    # 与 original 比较
    heva_key = 'heva_numeric' if 'heva_numeric' in df.columns else 'heva'
    original_heva = df[df['condition'] == 'original'][heva_key].values

    conditions = df['condition'].unique()
    for cond in conditions:
        if cond == 'original':
            continue
        cond_heva = df[df['condition'] == cond][heva_key].values

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
        # 处理 heva 列 - 支持字典格式
        if 'heva' in df.columns and len(df['heva']) > 0:
            first_heva = df['heva'].iloc[0]
            if isinstance(first_heva, dict):
                df['heva_numeric'] = df['heva'].apply(lambda x: x.get('heva_20', x.get('heva_100', 0.0)) if isinstance(x, dict) else x)
                visual_heva = visual_samples['heva_numeric'].values
                language_heva = language_samples['heva_numeric'].values
            else:
                visual_heva = visual_samples['heva'].values
                language_heva = language_samples['heva'].values
        else:
            visual_heva = np.array([])
            language_heva = np.array([])

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


def generate_plots(results, output_dir, dataset_name):
    """生成图表"""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    # 处理 heva 列 - 支持字典格式
    if 'heva' in df.columns and len(df['heva']) > 0:
        first_heva = df['heva'].iloc[0]
        if isinstance(first_heva, dict):
            # 新格式: 提取数值
            df['heva_numeric'] = df['heva'].apply(lambda x: x.get('heva_20', x.get('heva_100', 0.0)) if isinstance(x, dict) else x)
            heva_values = df['heva_numeric'].values
        else:
            heva_values = df['heva'].values
    else:
        heva_values = np.array([])

    # HEVA 分布
    plot_heva_distribution(
        heva_values,
        title=f"HEVA Distribution - {dataset_name}",
        save_path=os.path.join(output_dir, f'heva_distribution_{dataset_name}.png')
    )

    # 分组对比
    if 'group' in df.columns:
        if 'heva_numeric' in df.columns:
            visual_heva = df[df['group'] == 'visual-critical']['heva_numeric'].values
            language_heva = df[df['group'] == 'language-guessable']['heva_numeric'].values
        else:
            visual_heva = df[df['group'] == 'visual-critical']['heva'].values
            language_heva = df[df['group'] == 'language-guessable']['heva'].values

        if len(visual_heva) > 0 and len(language_heva) > 0:
            plot_group_comparison(
                visual_heva, language_heva,
                group1_name="Visual-Critical",
                group2_name="Language-Guessable",
                save_path=os.path.join(output_dir, f'group_comparison_{dataset_name}.png')
            )

    # 扰动对比
    if 'condition' in df.columns:
        plot_heva_perturbation(
            results,
            save_path=os.path.join(output_dir, f'perturbation_comparison_{dataset_name}.png')
        )


def main():
    import config
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--exp_name', type=str, default=config.DEFAULT_EXP_NAME,
                       help='Experiment name (default: exp001)')
    parser.add_argument('--results_dir', type=str, default=config.RESULTS_DIR,
                        help='Results directory')
    parser.add_argument('--output_dir', type=str, default=config.PLOTS_DIR,
                        help='Output directory for plots')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to analyze (e.g., VisuRiddles)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all available datasets')

    args = parser.parse_args()

    # 实验结果目录: results/{exp_name}/
    exp_dir = os.path.join(args.results_dir, args.exp_name)

    all_stats = []

    # 分析指定数据集或所有数据集
    datasets_to_analyze = []

    if args.all:
        datasets_to_analyze = SUPPORTED_DATASETS
    elif args.dataset:
        datasets_to_analyze = [args.dataset]
    else:
        # 自动检测已有的结果文件夹 (新格式: results/{exp_name}/{dataset}/)
        for ds in SUPPORTED_DATASETS:
            dataset_dir = os.path.join(exp_dir, ds)
            index_path = os.path.join(dataset_dir, 'index.json')
            # 兼容旧格式
            old_path = os.path.join(args.results_dir, f'inference_{ds}.pkl')
            if os.path.exists(index_path) or os.path.exists(old_path):
                datasets_to_analyze.append(ds)

    for dataset_name in datasets_to_analyze:
        # 新格式目录: results/{exp_name}/{dataset}/
        dataset_dir = os.path.join(exp_dir, dataset_name)
        index_path = os.path.join(dataset_dir, 'index.json')
        # 兼容旧格式
        old_path = os.path.join(args.results_dir, f'inference_{dataset_name}.pkl')

        # 先尝试新格式，再尝试旧格式
        if not os.path.exists(index_path) and os.path.exists(old_path):
            # 旧格式兼容：results/inference_{dataset}.pkl -> 映射到新目录
            dataset_dir = os.path.join(args.results_dir, dataset_name)
            index_path = os.path.join(dataset_dir, 'index.json')

        result_exists = os.path.exists(index_path) or os.path.exists(old_path)

        if result_exists:
            print(f"\n{'='*60}")
            print(f"Analyzing: {dataset_name}")
            print('='*60)

            try:
                results, errors = load_inference_results(dataset_dir)
                print(f"Loaded {len(results)} results, {len(errors)} errors")
                stats = analyze_single_dataset(results, dataset_name)
                all_stats.append(stats)

                # 生成图表
                generate_plots(results, args.output_dir, dataset_name)

                # 保存单个数据集的统计结果
                stats_path = os.path.join(dataset_dir, 'stats.json')
                with open(stats_path, 'w') as f:
                    # 转换 numpy 类型为 Python 类型
                    stats_json = {}
                    for k, v in stats.items():
                        if isinstance(v, (np.integer, np.floating)):
                            stats_json[k] = float(v)
                        else:
                            stats_json[k] = v
                    json.dump(stats_json, f, indent=2)
                print(f"Stats saved to: {stats_path}")

            except Exception as e:
                print(f"Error analyzing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Result not found for: {dataset_name}")

    # 保存汇总统计
    if all_stats:
        summary_path = os.path.join(args.results_dir, 'all_datasets_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_path}")

        # 打印汇总表格
        print(f"\n{'='*60}")
        print("ALL DATASETS SUMMARY")
        print('='*60)

        summary_df = pd.DataFrame(all_stats)
        print(summary_df.to_string(index=False))

    # 分析分组结果
    group_path = os.path.join(args.results_dir, 'group_analysis.pkl')
    if os.path.exists(group_path):
        with open(group_path, 'rb') as f:
            group_data = pickle.load(f)
        analyze_group_results(group_data['results'])

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
