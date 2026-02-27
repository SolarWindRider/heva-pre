# HEVA 实验项目

根据 doc.md 规范实现的 HEVA (High-Entropy Visual Attention) 指标验证实验。

## 项目结构

```
heva-pre/
├── doc.md                    # 技术定义文档
├── config.py                  # 配置文件
├── data/
│   ├── loader.py             # VisuRiddles 数据加载器
│   └── perturbations.py      # 图像扰动函数
├── models/
│   └── inference.py          # Qwen3-VL 推理模块
├── metrics/
│   ├── __init__.py
│   └── heva.py               # HEVA 计算模块
├── analysis/
│   ├── statistics.py         # 统计分析模块
│   └── plots.py              # 可视化模块
├── experiments/
│   ├── 1_run_inference.py    # 运行推理
│   ├── 3_perturbation_exp.py # 扰动实验
│   ├── 4_group_analysis.py   # 分组分析
│   ├── 5_causal_exp.py       # 因果敏感性实验
│   ├── 6_temporal_analysis.py # 时间动态分析
│   └── 7_statistics.py       # 统计分析脚本
├── results/                   # 实验结果目录
└── plots/                     # 图表目录
```

## 快速开始

### 运行方式

```bash
# 激活环境
conda activate PyTorch-2.1.0

# 切换到项目目录
cd /home/ma-user/work/heva-pre

# 1. 运行推理
python experiments/1_run_inference.py --num_samples 50

# 2. 运行扰动实验
python experiments/3_perturbation_exp.py --perturbation shuffle_patches --num_samples 20

# 3. 运行分组分析
python experiments/4_group_analysis.py --num_samples 100

# 4. 运行因果敏感性实验
python experiments/5_causal_exp.py

# 5. 运行时间动态分析
python experiments/6_temporal_analysis.py --num_samples 30

# 6. 统计分析
python experiments/7_statistics.py
```

## 依赖

- torch
- transformers
- numpy
- scipy
- sklearn
- matplotlib
- seaborn
- pandas
- pillow
- tqdm

## HEVA 定义

HEVA (High-Entropy Visual Attention) 定义为高熵生成 token 在最后一层解码器中对视觉 token 的平均注意力质量。

```
HEVA = (1/|S|) * sum(v_t for t in S)

其中:
- S = top-alpha% 高熵生成 token 集合
- v_t = token t 对所有视觉 token 的注意力之和
```

## 验证检查点

根据 doc.md，需要验证：

1. **Attention 归一化**: attention 行和为 1
2. **视觉 token 索引正确**: 打印 input_ids 确保是图像 embedding 区间
3. **HEVA 数值范围**:
   - 完全语言 shortcut: 0.01 ~ 0.05
   - 视觉依赖样本: 0.1 ~ 0.3
   - 极端视觉依赖: > 0.4
