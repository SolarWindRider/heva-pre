# `2_run_inference_heva_force.py` — HEVA 指标验证实验

## 实验目的

验证 **HEVA（High-Entropy Visual Attention）指标**与回答正确性之间的相关性。

核心假设：高熵 token 的视觉注意力（vattn）越高，模型回答越可能正确。

## 方法概述

本脚本采用**离线统计分析**范式：生成时只捕获数据，生成后分析。

```
生成阶段：捕获 gen_entropy + gen_vattn
    ↓
保存到 pkl + meta json
    ↓
7_statistics.py 做分析
```

## 核心机制

### Monkey-Patch：`Qwen3VLForConditionalGeneration._sample`

在 `metrics/inference.py` 中：

```python
Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy
```

替换了 transformers 内部的 `_sample` 方法，在**每个生成步骤**中插入捕获逻辑。

### 捕获的数据

| 字段 | 形状 | 含义 |
|------|------|------|
| `gen_entropy` | `(gen_tokens, batch)` | 每步的 Shannon 熵 |
| `gen_vattn` | `(gen_tokens, batch)` | 每步对视觉 token 的注意力之和 |
| `attn_acc_visual` | `(gen_tokens, batch)` | top-k atten 中视觉 token 占比 |
| `attn_acc_input` | `(gen_tokens, batch)` | top-k atten 中 input token 占比 |

### HEVA 计算公式（事后分析）

```
HEVA = mean(vattn[entropy > percentile(entropy, 80)])
```

即：**熵值 top-20% 的 token，对视觉 token 的平均注意力**。

### Context-Aware Decoding（可选）

在 `metrics/context_aware_logits_processor.py` 中实现。当生成 token 熵值高于阈值时，对 **top-k 候选进行过滤**：保留 context_support 最高的 top_k // 2 个候选，其余设为 -inf（不 boost 分数）。

```python
ContextAwareLogitsProcessor(
    model=model,
    top_k=50,
    entropy_threshold=1.27,   # 触发阈值
    top_heads=5,              # 选用的上下文 head 数
)
```

## 输出结构

```
results/{exp_name}/{dataset}/
├── exp_config.json              # 实验配置
├── index.json                  # 结果索引
├── {sample_id}_meta.json       # 每样本元数据
│   ├── ground_truth
│   ├── predicted_answer
│   ├── correct
│   ├── avg_vattn               # 平均视觉注意力
│   └── generated_text
└── pkls/
    ├── {idx}_gen_entropy.pkl  # 熵值序列
    └── {idx}_gen_vattn.pkl     # 视觉注意力序列
```

## 运行方式

```bash
# 标准运行（无 CAD）
python 2_run_inference_heva_force.py \
    --exp_name exp001 \
    --dataset VisuRiddles \
    --num_samples 100

# 开启 Context-Aware Decoding
python 2_run_inference_heva_force.py \
    --exp_name exp_cad \
    --dataset RAVEN \
    --use_context_aware \
    --ctx_entropy_threshold 1.27 \
    --ctx_top_heads 5 \
    --num_samples 100
```

## 与 `3_run_inference_trace.py` 的区别

| 维度 | 本脚本 | `3_run_inference_trace.py` |
|------|--------|---------------------------|
| **范式** | 离线分析（生成时只捕获） | 在线引导（生成时做 DLA 路径验证过滤） |
| **DLA 路径** | 不计算 | 完整反向路径（所有层） |
| **候选过滤** | 无 | 对每个候选 token 验证路径后才采样 |
| **捕获内容** | `gen_entropy`, `gen_vattn` | + `gen_zs`（所有层 z）|
| **适用场景** | HEVA 指标分析、CAD 效果对比 | 验证"注意力路径指向视觉"与采样选择的关系 |
