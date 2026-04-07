# HEVA 实验脚本总览

## 研究问题

**HEVA（High-Entropy Visual Attention）** 验证一个核心假设：

> 模型在生成高不确定性（高熵）的 token 时，是否真正在"看图"（attend to 视觉 token）？

- 如果 HEVA 高 → 模型生成高熵 token 时大量 attending 到视觉 token → **视觉依赖**
- 如果 HEVA 低 → 模型靠语言先验而非图像信息来生成 → **语言捷径**

## 两个实验脚本的关系

```
2_run_inference_heva_force.py
    └── 目的：验证 HEVA 指标与正确率的相关性
    └── 范式：离线分析（只捕获数据，事后计算 HEVA）
    └── 核心问题：HEVA 高的样本是否回答更正确？

3_run_inference_trace.py
    ├── 目的：验证"选择 token 时是否真的在追视觉路径"
    ├── 范式：在线引导（采样时做 DLA 路径验证）
    └── 核心问题：模型的采样决策是否真的有视觉因果路径？

           ┌─ 2_run_inference_heva_force.py ──→ 验证 HEVA 指标有效性
研究流程 ──┤
           └─ 3_run_inference_trace.py ──────→ 验证因果路径假设
                                                           │
                                                   如果假设成立
                                                   → 可用 CAD 在高熵时
                                                     过滤保留 top_k//2 视觉上下文支持候选
```

## 关键概念

### gen_entropy（熵）
衡量该 token 位置的不确定性。熵高意味着模型对下一个 token 不是很确定，可能需要依赖上下文（图像）。

### gen_vattn（视觉注意力）
该 token 对 `<|image_pad|>` token 的注意力权重之和。直接反映"这个 token 有多在看图"。

### gen_zs（所有层 z）
每步生成时捕获所有层的 z（attention 原始输出，o_proj 之前）。用于离线反向路径重算，验证 DLA 路径的正确性。

### DLA 反向路径
从输出 token 的 logit 开始，通过 `z · W_O · W_V` 逐层反向，找出每层中贡献最大的 head。这回答了"最终选择这个 token 的功劳主要来自哪些层和 head"。

### Critical Indices
想验证"是否在看图"时，`critical_indices` = 视觉 token 位置（`range(v_start, v_end+1)`）。参考代码中做数字推理时，`critical_indices` = 数字/运算符 token 位置。

## 脚本文件映射

| 脚本 | 核心 Monkey-Patch | 主要捕获 |
|------|-----------------|---------|
| `2_run_inference_heva_force.py` | `_sample` → `_sample_with_vattn_and_entropy` | `gen_entropy`, `gen_vattn` |
| `3_run_inference_trace.py` | 同上 + attention forward 捕获所有层 z | `gen_entropy`, `gen_vattn`, `gen_zs` |
| `metrics/heva.py` | 包含 `_sample_with_vattn_and_entropy` 的实现 | — |
| `metrics/inference.py` | 被 `2_run_inference_heva_force.py` 使用 | `generate_with_attn`（简化版捕获） |
| `metrics/context_aware_logits_processor.py` | 两者共用 | CAD 的 logits processor |

## HEVA 分析脚本

`7_statistics.py` 对两个实验的输出都适用：

```python
# HEVA 计算
HEVA = mean(vattn[entropy > percentile(entropy, 80)])

# 按 HEVA 分组看正确率
HEVA_high_correct_rate > HEVA_low_correct_rate → 假设成立
```
