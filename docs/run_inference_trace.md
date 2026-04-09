# `3_run_inference_trace.py` — DLA Trace 引导生成实验

## 实验目的

在生成过程中**实时验证 DLA（Direct Logits Attribution）反向路径**：每个候选 token 的注意力是否真正指向视觉 token？只从通过验证的候选中采样。

核心问题：模型在选择高熵 token 时，是否真的在"看图"？还是只是走了语言先验捷径？

## 方法概述

本脚本采用**在线引导**范式（区别于 `2_run_inference_heva_force.py` 的离线分析）：

```
生成阶段：每步采样前
    ├─ 捕获 gen_zs（所有层 z） + gen_entropy + gen_vattn
    ├─ 对候选 token 计算 DLA 反向路径
    ├─ 验证路径是否聚焦视觉 token
    └─ 只从通过验证的候选中采样
         ↓
    保存完整 trace 数据
         ↓
    7_statistics.py 做分析
```

## 核心技术：DLA 反向路径计算

### 什么是 z？

在 Attention 计算中，`z` 是 attention output 在 `o_proj` **之前**的原始值：

```
Attention 计算流程（单层）：
    Q = q_norm(q_proj(hidden_states))      # (batch, seq, heads, d_head)
    K = k_norm(k_proj(hidden_states))       # (batch, seq, heads, d_head)
    V = v_proj(hidden_states)              # (batch, seq, heads, d_head)

    attn_weights = softmax(Q @ K.T / sqrt(d_head))
    attn_output = attn_weights @ V          # (batch, seq, heads, d_head)
                                            # ← 这就是 z（捕获位置）

    attn_output = attn_output.reshape(..., -1)
    output = o_proj(attn_output)             # 投影到 d_model
```

**捕获所有层的 z**（不仅是最后一层），是为了从输出 token 逐层反向追溯到输入。

### 反向路径的计算

`compute_dla_path_for_token()` 对每个候选 token 逐层反向计算"哪个 head 贡献最大"。

#### 单层内部的计算

```
Step 1: z @ W_O = 各 head 的输出向量
    z: (heads, d_head) — attention 原始输出（o_proj 之前）
    W_O: (heads, d_head, d_model)
    → head_outputs = z @ W_O: (heads, d_model)

Step 2: 计算每 head 的贡献分数（dot product）
    contribution[head] = head_outputs[head] · target_vector  # scalar
    → 几何意义：head 输出向量与目标 logit 向量的对齐程度

Step 3: 选贡献最大的 head
    max_head = argmax(contribution)
    记录：path[layer] = {"head": max_head, "score": contribution[max_head]}
```

#### 层与层之间的传递

```
Attention 单层计算顺序（残差结构）：
    输入 → QKV投影 → RoPE → Attention → z → W_O → +残差 → LayerNorm → 下一层输入

追溯时：
    输出 ← W_U[token_id]
            ↑
    W_O @ z ← 该层选 head
            ↑
    通过残差连接，z 来自上一层的 V 输出
    所以往前追溯 = 追踪 V 路径

因此传递公式为：
    target_vector = W_O[head] @ target_vector

以 4 层模型为例，完整追溯路径：

```
初始：target_vector = W_U[token_id]  ← 该 token 的 logit 向量
        ↓
Layer 3 (最后一层)：
    z[3] @ W_O[3] → contributions → 选 head=7
    target_vector = W_O[3,7] @ target_vector
        ↓
Layer 2：
    z[2] @ W_O[2] → contributions → 选 head=3
    target_vector = W_O[2,3] @ target_vector
        ↓
Layer 1：
    z[1] @ W_O[1] → contributions → 选 head=12
    target_vector = W_O[1,12] @ target_vector
        ↓
Layer 0：
    z[0] @ W_O[0] → contributions → 选 head=5
    target_vector = W_O[0,5] @ target_vector

最终输出：
    path = {
        0: {"head": 5,  "score": s0},
        1: {"head": 12, "score": s1},
        2: {"head": 3,  "score": s2},
        3: {"head": 7,  "score": s3},
    }
```

**为什么用 `W_O[head]` 而不是 `W_V @ W_O[head]`？**

`W_V` 在 GQA（Grouped Query Attention）结构中作用于 KV head 维度（融合后），而 `z` 是所有 Q head 的混合输出，两者维度不兼容无法直接分解。因此使用 `W_O[head]`（输出投影）来做层间追溯——它直接对应每个 head 的输出方向，信息流必经路径。

**关键数据**：
- 用 `z`（attention 原始输出）和 `W_O`/`W_V`/`W_U` 矩阵计算
- 不依赖 softmax 后的 attention weights

### 验证路径是否聚焦视觉 token

`verify_attention_focus_on_path()` 检查 path 中各 head **实际 atten 到了哪些 token**：

```
attentions[-1] = 最后一层的 attention 权重
  shape: (batch, heads, query_seq, key_seq)
  这是 softmax 后的实际注意力权重

attn_last = attentions[-1][b, :, -1, :]
  即：最后一个 query（当前生成 token）对所有 key 的注意力
  shape: (heads, key_len)

For layer i in path:
    head = path[i]["head"]
    attn_pattern = attn_last[head]  # (key_len,) — 该 head 的实际注意力分布
    取 top-5 atten 位置

    检查：top-5 中有任意落在 critical_indices（视觉 token 范围）吗？
    有 → 该层"命中"

命中比例 = 命中的层数 / 总层数
通过验证 = 命中比例 >= 30%
```

**两个函数用到了不同的数据**：
| 函数 | 数据来源 |
|------|---------|
| `compute_dla_path_for_token` | `z` + `W_O` + `W_V` + `W_U` 矩阵（因果结构） |
| `verify_attention_focus_on_path` | `outputs.attentions`（softmax 后的实际注意力权重） |

也就是说：先从**数学上**追溯哪个 head 的因果贡献大，再从**实际 atten 行为**上验证这个 head 是否真的在看视觉 token。两者都通过才认为该候选 token 是"视觉依赖的"。

### 采样过滤

```python
if valid_candidates:
    # 从通过验证的候选中重新归一化分布并采样
    normalized_probs = softmax(valid_logits)
    next_tokens = multinomial(normalized_probs, 1)
else:
    # 没有候选通过验证，回退到原始 top-1
    next_tokens = top_indices[0]
```

## Monkey-Patch 机制

在 `metrics/heva.py` 中有两层 patch：

### 第一层：`_sample` → `_sample_with_vattn_and_entropy`

替换 transformers 生成循环的采样函数，在每步插入捕获 + 验证逻辑。

### 第二层：`Qwen3VLTextAttention.forward`

替换 attention 的 forward，在其中捕获 `z`（按层索引存储到 `model._all_layers_z`）：

```python
# 在 attention forward 中：
self._z_by_layer[self.layer_idx] = attn_output.clone()

# 捕获后立即清空，为下一层做准备（通过闭包引用 model._all_layers_z）
```

在 `_sample` 的每步循环结束后，从 `model._all_layers_z` 收集所有层的 z：

```python
all_z = torch.stack([model._all_layers_z[i] for i in range(num_layers)], dim=0)
# shape: (num_layers, batch, seq, heads, d_head)
model.gen_zs.append(all_z.cpu())
model._all_layers_z.clear()
```

## 关键代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| z 捕获 patch | `metrics/heva.py` | `patched_forward()` |
| gen_zs 收集 | `metrics/heva.py` | `_sample_with_vattn_and_entropy` 循环内 |
| DLA 路径计算 | `metrics/heva.py` | `compute_dla_path_for_token()` |
| 路径验证 | `metrics/heva.py` | `verify_attention_focus_on_path()` |
| 候选过滤注入 | `metrics/heva.py` | `next_token_scores` 处理后 |
| critical_indices 注入 | `3_run_inference_trace.py` | `generate_with_attention_guidance()` |

## 输出结构

```
results/{exp_name}/{dataset}/
├── exp_config.json
├── index.json
├── {sample_id}_meta.json
│   ├── ground_truth
│   ├── predicted_answer
│   ├── correct
│   ├── avg_vattn
│   ├── gen_entropy_path
│   ├── gen_vattn_path
│   └── gen_zs_path              # ← 新增：所有层 z 的完整轨迹
└── pkls/
    ├── {idx}_gen_entropy.pkl
    ├── {idx}_gen_vattn.pkl
    └── {idx}_gen_zs.pkl         # ← 新增：list of (num_layers, batch, seq, heads, d_head)
```

## 运行方式

```bash
# 标准推理（仅捕获，不做 DLA 引导）
python 3_run_inference_trace.py \
    --exp_name exp001 \
    --dataset VisuRiddles \
    --num_samples 50

# 开启 DLA Trace 引导生成
python 3_run_inference_trace.py \
    --exp_name exp_trace \
    --dataset VisuRiddles \
    --use_attention_guidance true \
    --num_samples 50

# 叠加 Context-Aware Decoding + DLA Trace 引导
python 3_run_inference_trace.py \
    --exp_name exp_cad_trace \
    --dataset RAVEN \
    --use_context_aware true \
    --use_attention_guidance true \
    --ctx_entropy_threshold 5.0 \
    --num_samples 100
```

## 四种运行模式详解

本脚本有两个独立机制，共四种组合：

| 模式 | `--use_attention_guidance` | `--use_context_aware` | 描述 |
|------|:---:|:---:|------|
| **M0: 标准推理** | `false` | `false` | 只捕获 trace，不做干预 |
| **M1: 纯 DLA 引导** | `true` | `false` | DLA 路径验证过滤候选 |
| **M2: 纯 CAD** | `false` | `true` | 高熵时过滤为 top_k//2 候选 |
| **M3: DLA + CAD** | `true` | `true` | CAD 过滤 + DLA 过滤叠加 |

### M0: 标准推理（默认）

```bash
python 3_run_inference_trace.py \
    --exp_name exp001 --dataset VisuRiddles --num_samples 50
```

只做数据捕获，采样完全不做干预。用于采集 baseline 数据，供事后分析 HEVA 分布。

### M1: 纯 DLA 引导

```bash
python 3_run_inference_trace.py \
    --exp_name exp_trace --dataset VisuRiddles \
    --use_attention_guidance true --num_samples 50
```

**每步**的采样流程：

```
next_token_scores = logits_processor(...)
        ↓
取 top-10 候选
        ↓
对每个候选 token：
    [阶段一] compute_dla_path_for_token(all_zs, model, token_id)
        → 逐层反向追溯：W_U ← z·W_O·W_V
        → 每层选贡献最大的 head
        → 得到 path = {layer: {"head": h, "score": s}}
        ↓
    [阶段二] verify_attention_focus_on_path(attentions, path, critical_indices)
        → 检查 path 中各 head 的 top-5 实际 atten 是否命中视觉 token
        → 命中率 >= 30% → is_valid = True
        ↓
所有候选都验证完后：
    幸存候选（is_valid=True）→ 重新归一化 → 采样
```

**两阶段缺一不可**：
- 阶段一只回答"哪个 head 贡献大"（因果结构，用 `z` + 矩阵计算）
- 阶段二才回答"该 head 是否真的在看视觉 token"（实际行为，用 softmax atten 权重验证）

**没有 CAD**：`logits_processor` 只做标准处理，熵值不参与任何决策。

**特点**：
- DLA 过滤是**确定性的**（只看 attention 结构）
- 即使熵很低，DLA 也会执行（每步都做）
- 适合验证"DLA 路径验证"单独对生成的影响

### M2: 纯 CAD（Context-Aware Decoding）

```bash
python 3_run_inference_trace.py \
    --exp_name exp_cad --dataset RAVEN \
    --use_context_aware true --ctx_entropy_threshold 5.0 --num_samples 100
```

**每步**的采样流程：

```
next_token_scores = logits_processor([CAD_processor, ...])
        ↓
CAD_processor.__call__(input_ids, scores) 内部：
        ↓
Step 1: 计算熵 entropy = H(scores)
        ↓
        if entropy < ctx_entropy_threshold:
            → 直接返回原始 scores，不干预
        ↓
Step 2: entropy >= threshold，进入 CAD 处理：
        ↓
Step 3: 识别 context_heads（哪些 head 最 atten 到视觉 token）
        → 从 outputs.attentions 最后一层，取各 head 对视觉 token 的 atten 之和
        → 选 top-h 个 head（默认 5 个）
        ↓
Step 4: 取 top-k 候选（默认 20 个）
        → topk_ids = topk(scores[0], k=20)
        ↓
Step 5: 对每个候选计算 context support
        → 对每个 context_head，计算：
            support[token] += z[head] · W_O[head] · W_U[token]
        → 最终 support = 各 head 贡献之和 / head数
        ↓
Step 6: 只保留 top-k 中 context_support 最高的 top_k // 2 个候选
        → 其余候选的 score 直接设为 -inf（过滤，不是 boost）
        ↓
返回修改后的 scores（大部分 token 被设为 -inf）
        ↓
标准采样（multinomial）：从幸存候选中采样
```

**关键澄清**：
- **不是给 logits 加分**，而是把不合格的候选直接设为 `-inf` 排除掉
- 幸存候选数 = `top_k // 2`（即默认 10 个）
- 采样只在这 top-k//2 个幸存者中概率采样

**没有 DLA**：`use_attention_guidance=false` 时，不进入 DLA 验证分支。

**特点**：
- CAD 是**过滤性**的（确定性排除不合格候选）
- 只在高熵时触发（低熵时模型很确定，不干预）
- CAD 事后会记录在 meta.json 的 `use_context_aware` 字段

### M3: DLA + CAD 叠加

```bash
python 3_run_inference_trace.py \
    --exp_name exp_cad_trace --dataset RAVEN \
    --use_context_aware true --use_attention_guidance true \
    --ctx_entropy_threshold 5.0 --num_samples 100
```

**每步的完整执行流程（伪代码级）**：

```python
# heva.py:236 — logits_processor 调用（CAD 介入）
next_token_scores = logits_processor(input_ids, next_token_logits)
#   ↓
#   CAD_processor.__call__(input_ids, scores) 内部：
#   ├── 计算 entropy = H(scores)  （第 250 行）
#   ├── if entropy < threshold: 直接返回 scores，不干预
#   └── if entropy >= threshold:
#       ├── 取 top-k 候选 token（默认 20）
#       ├── 计算每个候选的 context_support
#       ├── 保留 top-k//2 高 support 的（10个），其余设为 -inf
#       └── 返回处理后的 scores
#
# heva.py:241-285 — Attention Guidance 介入（每步执行，与 entropy 无关）
if use_attention_guidance and gen_zs:
    # 在 CAD 处理后的 scores 上取 top-10
    _, top_indices = torch.topk(next_token_scores, k=10)

    for i in range(len(top_indices)):
        tok_id = top_indices[i]
        # 阶段一：DLA 反向路径计算
        path = compute_dla_path_for_token(all_zs_current, model, tok_id)
        #   逐层反向：W_U ← z·W_O，每层选贡献最大的 head
        #
        # 阶段二：验证路径是否聚焦视觉 token
        is_valid = verify_attention_focus_on_path(
            outputs.attentions, path, critical_indices, top_k_attn=5
        )
        #   检查 path 中各 head 的 top-5 atten 是否命中视觉 token
        #   命中率 >= 30% → is_valid = True

        if is_valid:
            valid_candidates.append(tok_id)
            valid_logits.append(next_token_scores[tok_id])

    if valid_candidates:
        # 从通过验证的候选中重新归一化采样
        probs = softmax(valid_logits)
        next_tokens = multinomial(probs, 1)
        has_attn_guidance_override = True

# 如果 attention guidance 没有产生幸存者，回退到 top-1
if not has_attn_guidance_override:
    next_tokens = top_indices[0]
```

**关键执行顺序与数据依赖**：

| 步骤 | 代码位置 | 输入 | 输出 | 触发条件 |
|------|---------|------|------|---------|
| 1. CAD 处理 | `heva.py:236` | 原始 `next_token_logits` | 部分 token 设为 `-inf` | entropy >= threshold |
| 2. DLA 取候选 | `heva.py:248` | CAD 处理后的 scores | top-10 候选 | 始终（与 entropy 无关）|
| 3. DLA 路径计算 | `heva.py:265` | `gen_zs[-1]`（所有层 z）| path dict | 始终 |
| 4. 路径验证 | `heva.py:266` | `outputs.attentions` | `is_valid` bool | 始终 |
| 5. 重新采样 | `heva.py:275-277` | valid_candidates | `next_tokens` | 存在幸存者 |

**数据流图**：

```
原始 logits (vocab,)
       │
       ▼
[CAD] entropy >= threshold?
       │是                    │否
       ▼                     ▼
  top-k → support 排序     直接返回原始 scores
  保留 top-k//2
  其余设为 -inf
       │
       ▼
  处理后的 scores (vocab,)
       │
       ▼
[AG] torch.topk(scores, k=10)
  → top_indices (10,)
       │
       ▼
  for each tok_id in top_indices:
       │
       ├─→ compute_dla_path_for_token()
       │    用 gen_zs[-1] + W_U + W_O
       │    输出: path = {layer: {"head": h, "score": s}}
       │
       └─→ verify_attention_focus_on_path()
            用 outputs.attentions（softmax 权重）
            检查各 head 的 top-5 atten 是否命中 visual tokens
            输出: is_valid (bool)
       │
       ▼
  过滤: 只保留 is_valid=True 的候选
       │
       ▼
  幸存者 → softmax → multinomial → next_token
```

**注意**：CAD 和 DLA 都是**过滤**机制，不是 boost。CAD 按 context_support 过滤，DLA 按视觉注意力路径验证过滤。两者顺序执行，DLA 在 CAD 的结果上继续过滤。

### 关键区别总结

| 维度 | M1 纯 DLA | M2 纯 CAD | M3 DLA + CAD |
|------|----------|----------|-------------|
| **触发条件** | 每步执行 | 只在高熵时触发 | CAD 按熵触发 + DLA 每步执行 |
| **干预方式** | 过滤候选（路径验证） | 过滤候选（context_support 排序） | CAD 过滤 + DLA 过滤叠加 |
| **critical_indices** | 用于 DLA 验证 | 用于识别 context_heads | 用于两者 |
| **幸存候选数** | 动态（通过验证的比例） | `top_k // 2` 个 | CAD 幸存者再经 DLA 过滤 |
| **典型研究问题** | "只看路径是否够吗？" | "过滤到 top_k//2 有效吗？" | "两层过滤是否叠加有效？" |

### 研究问题的对应关系

```
M0 (标准推理)
    → 采集 baseline，分析 HEVA 分布
    → 问题：HEVA 高的样本是否回答更正确？

M1 (纯 DLA)
    → 控制采样路径，验证"路径验证"单独的作用
    → 问题：强制走视觉路径是否提升正确率？

M2 (纯 CAD)
    → 高熵时过滤为 top_k//2 幸存候选
    → 问题：CAD 能提升 HEVA 低的样本的正确率吗？

M3 (DLA + CAD)
    → CAD 过滤 + DLA 过滤叠加
    → 问题：两层干预是否比单层更有效？
```

## 与 `2_run_inference_heva_force.py` 的区别

| 维度 | `2_run_inference_heva_force.py` | `3_run_inference_trace.py` |
|------|--------------------------------|---------------------------|
| **范式** | 离线分析（只捕获数据） | 在线引导（捕获 + 实时过滤） |
| **DLA 路径** | 不计算 | 完整反向路径（所有层） |
| **候选过滤** | 无 | 对候选 token 验证后才采样 |
| **z 捕获** | 只捕获最后一层（`_last_z`） | **所有层**（`_all_layers_z`） |
| **gen_zs 输出** | 无 | 有（用于离线 DLA 分析） |
| **适用场景** | HEVA 指标验证、CAD 对比 | 验证"注意力路径 → 采样选择"的因果关系 |
| **研究问题** | "高熵 token 是否在看图？" | "模型选择 token 时是否真的在追视觉路径？" |

## 完整 Trace 的含义

每个生成 token 的完整 trace 包含：

```
gen_entropy[t]         — Shannon 熵（不确定性）
gen_vattn[t]           — 对视觉 token 的注意力之和
gen_zs[t]              — (num_layers, batch, seq, heads, d_head)
                         所有层的 z，用于离线反向路径重算
attn_acc_visual[t]     — top-k atten 中视觉 token 占比
attn_acc_input[t]      — top-k atten 中 input token 占比
valid_candidates[t]    — 通过 DLA 路径验证的候选 token（如果开启引导）
```

这使得事后可以**重新分析任意 token 的 DLA 路径**，验证在线决策是否正确。
