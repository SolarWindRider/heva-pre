# HEVA 技术定义文档（实现级规范）

---

# 1. 基本符号定义

我们考虑自回归多模态模型：

[
\pi_\theta(y \mid I, q)
]

输入拼接顺序（Qwen3-VL 实际结构）：

```
[System tokens]
[User text tokens]
[Image tokens]
[Assistant prefix]
[Generated tokens]
```

设：

* 总序列长度：( L )
* 视觉 token 索引集合：( \mathcal{V} \subset {1,...,L} )
* 生成 token 索引集合：( \mathcal{T} \subset {1,...,L} )

注意：

视觉 token 在 input_ids 中是连续的一段 embedding。

---

# 2. Logits 与 Entropy 定义

对于生成步骤 ( t \in \mathcal{T} )：

模型输出：

[
z_t \in \mathbb{R}^{|\mathcal{Vocab}|}
]

概率分布：

[
p_t = \text{softmax}(z_t)
]

Token entropy 定义为：

[
H_t = - \sum_{i=1}^{|\mathcal{Vocab}|} p_t(i)\log p_t(i)
]

---

# 3. 高熵 Token 集合 S

设生成 token 总数为 ( T )。

定义比例参数：

[
\alpha \in (0,1)
]

默认：

[
\alpha = 0.2
]

定义：

[
S =
\text{Top-}\alpha T \text{ tokens ranked by } H_t
]

实现要求：

* 必须基于生成 token 排序
* 不包含 prompt token
* 不包含 image token

---

# 4. Attention 张量结构

在 Qwen3-VL 中：

当设置：

```python
output_attentions=True
```

得到：

```python
attentions: Tuple[num_layers]
```

每层：

[
A^{(l)} \in \mathbb{R}^{B \times H \times L \times L}
]

其中：

* B = batch size
* H = head 数
* L = 序列总长度

我们只使用最后一层：

[
A^{(L)}
]

---

# 5. 单 Token 的视觉注意力度量

对于生成 token index ( t \in \mathcal{T} )：

提取：

[
A^{(L)}[:, :, t, :]
]

形状：

[
(H, L)
]

定义该 token 的视觉注意力为：

[
v_t
===

\frac{1}{H}
\sum_{h=1}^{H}
\sum_{k \in \mathcal{V}}
A^{(L,h)}_{t,k}
]

解释：

* 先对视觉 token 求和
* 再对 head 求平均

得到：

[
v_t \in [0,1]
]

（因为 attention 行已归一化）

---

# 6. HEVA 定义

最终 HEVA 为：

[
\boxed{
HEVA
====

\frac{1}{|S|}
\sum_{t \in S}
v_t
}
]

---

# 7. 重要性质

## 性质 1

[
0 \le HEVA \le 1
]

---

## 性质 2

若模型完全忽略图像：

[
A_{t,k} \approx 0
\quad \forall k \in \mathcal{V}
]

则：

[
HEVA \approx 0
]

---

## 性质 3

若视觉 token 在高熵阶段承担消歧作用：

[
HEVA \uparrow
]

---

# 8. 实现级伪代码

```python
def compute_heva(logits, attentions, image_token_indices, gen_token_indices, alpha=0.2):

    # 1. compute entropy
    probs = softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    # 2. select generation entropy
    entropy_gen = entropy[gen_token_indices]

    # 3. select top alpha tokens
    k = int(len(entropy_gen) * alpha)
    top_indices = torch.topk(entropy_gen, k).indices
    selected_tokens = gen_token_indices[top_indices]

    # 4. last layer attention
    attn = attentions[-1][0]  # (H, L, L)

    heva_values = []

    for t in selected_tokens:
        attn_t = attn[:, t, :]  # (H, L)
        visual_attn = attn_t[:, image_token_indices].sum(dim=-1)
        v_t = visual_attn.mean()
        heva_values.append(v_t)

    return torch.stack(heva_values).mean().item()
```

---

# 9. 实现时必须满足的检查

### 检查 1：attention 行和为 1

验证：

```python
attn_t.sum(dim=-1) ≈ 1
```

---

### 检查 2：视觉 token index 正确

必须打印：

```python
print(input_ids[image_token_indices])
```

确保确实是 image embedding 区间。

---

### 检查 3：HEVA 数值范围

正常情况下：

* 完全语言 shortcut：0.01 ~ 0.05
* 视觉依赖样本：0.1 ~ 0.3
* 极端视觉依赖：> 0.4

若全是 0.8 以上，说明 index 错了。

---

# 10. 实验中使用的 HEVA 变体

你在预实验阶段只允许使用：

[
HEVA_{light}
]

不能：

* 引入 no-image
* 引入 KL
* 引入 hidden state 分析

先验证单一指标。

---

# 11. 论文中必须写的精确定义

论文里必须写成：

> HEVA is defined as the average visual attention mass assigned by high-entropy generated tokens in the final decoder layer.

---

# HEVA 预实验开发文档

（NeurIPS 验证阶段专用）

---

# 1. 项目目标

验证 HEVA（High-Entropy Visual Attention）是否满足：

1. 对视觉扰动敏感
2. 区分视觉关键样本
3. 预测视觉 token 因果敏感性
4. 在决策阶段出现峰值

不涉及训练，不涉及 RL。

---

# 2. 技术栈

## 模型

* `Qwen3-VL-2B-Instruct`
* HuggingFace Transformers
* 输出需开启：

  * `output_attentions=True`
  * `output_hidden_states=False`

来源：
Qwen
Qwen3-VL-2B-Instruct

---

## 数据集

VisuRiddles

需要：

* 图像路径
* 问题文本
* 标准答案

---

## 依赖

* torch
* transformers
* numpy
* scipy
* sklearn
* matplotlib
* pandas

---

# 3. 实验整体流程架构

```
data/
models/
metrics/
experiments/
analysis/
plots/
```

---

# 4. 核心模块设计

---

# 4.1 Data Loader 模块

### 输入

* 图像路径
* 问题
* 答案

### 输出

```python
{
  "image": PIL.Image,
  "question": str,
  "answer": str
}
```

---

# 4.2 Inference + Attention Capture 模块

必须支持：

```python
model.generate(
    ...
    output_attentions=True,
    return_dict_in_generate=True
)
```

需要提取：

* logits
* attentions (last layer)
* generated tokens

---

### 注意

Qwen3-VL attention 结构：

```
attentions:
tuple(num_layers)
  each: (batch, num_heads, seq_len, seq_len)
```

我们只用：

```
attentions[-1]
```

---

# 4.3 视觉 Token 索引定位

必须实现函数：

```python
def get_visual_token_indices(input_ids, tokenizer):
```

方法：

* 找到 image special tokens
* 或根据 processor 记录的 image embedding 位置

输出：

```
List[int]
```

---

# 4.4 Entropy 计算模块

对生成的每个 token：

```python
p = softmax(logits_t)
entropy = -sum(p * log(p))
```

返回：

```
List[entropy]
```

---

# 4.5 HEVA 计算模块

步骤：

1. 选 top 20% entropy token index
2. 对每个 token t：

   * 取 attention[t, :, :]
   * 对视觉 token 维度求和
   * 对 head 取平均
3. 再对 token 取平均

输出：

```
float HEVA
```

---

# 5. 具体实验开发任务

---

# 实验 1：视觉扰动一致性

## 目标

验证：

HEVA(original) > HEVA(perturbed)

---

## 需要实现

### 图像扰动函数

```python
def shuffle_patches(image, patch_size=32)
def gaussian_blur(image)
def zero_image(image)
```

---

## Pipeline

对每个样本：

```
original
shuffle
blur
zero
```

保存：

```
{
  sample_id,
  condition,
  HEVA,
  accuracy
}
```

---

## 输出文件

`results_perturbation.csv`

---

# 实验 2：视觉 vs 语言样本区分

---

## 样本自动分类

规则：

如果问题包含：

* color
* left/right
* count
* position
* number

标记为 visual-critical

否则为 language-guessable

---

## 统计

计算：

```
mean HEVA_visual
mean HEVA_language
t-test
Cohen_d
```

输出：

`results_group_comparison.json`

---

# 实验 3：因果敏感性（最重要）

---

## 样本选择

按 HEVA 排序：

* top 20%
* bottom 20%

---

## 实验

对两组分别：

1. 原图
2. mask 所有视觉 token（直接删除 image embedding）

计算：

```
KL(original || masked)
accuracy_drop
```

---

## KL 计算

对完整生成分布求平均：

```
mean KL over tokens
```

---

## 输出

`results_causal.csv`

---

# 实验 4：时间动态分析

---

对每个样本：

保存：

```
time_step
entropy
visual_attention
```

生成：

* 平均曲线
* 对齐答案 token 的峰值图

---

# 6. 统计分析模块

---

必须统一实现：

```python
def compute_ttest(group1, group2)
def compute_cohens_d(group1, group2)
def compute_pearson(x, y)
def compute_bootstrap_ci(data)
```

---

所有统计必须输出：

```
mean
std
p-value
effect_size
confidence_interval
```

---

# 7. 可视化模块

---

生成图：

1. HEVA vs perturbation
2. HEVA 分组对比柱状图
3. KL divergence 对比
4. 时间动态曲线
5. attention heatmap 示例

---

# 8. 运行顺序

```
1_run_inference.py
2_compute_heva.py
3_perturbation_exp.py
4_group_analysis.py
5_causal_exp.py
6_temporal_analysis.py
7_statistics.py
8_plot_all.py
```

---

# 9. 关键实验阈值（论文目标）

成功标准：

| 实验       | 判定                |
| ---------- | ------------------- |
| 扰动一致性 | p < 0.01            |
| 分组区分   | Cohen d > 0.8       |
| 因果 KL    | Cliff delta > 0.6   |
| 时间峰值   | entropy & HEVA 同步 |

---

# 10. 最终输出结构

```
tables/
figures/
appendix_stats/
```

生成：

* Table 1: Perturbation results
* Table 2: Group comparison
* Table 3: Causal masking
* Figure 1: HEVA sensitivity
* Figure 2: Temporal dynamics
* Figure 3: Attention heatmaps

---

# 11. 计算资源建议

* GPU 24GB 足够
* batch size = 1（需要 attention）
* 保存中间结果（避免重复生成）

---

# 12. 重要工程注意事项

1. 一定缓存 logits 和 attentions
2. 不要重复 forward
3. 视觉 token index 必须验证正确
4. entropy 必须基于真实 softmax
5. KL 必须同一生成长度

---

# 13. 预期论文结论逻辑

若全部实验成立：

可以得出：

> HEVA is statistically significant, perturbation-consistent, and causally predictive of visual dependency.

---



所在操作系统为linux

python环境必须通过  conda activate PyTorch-2.1.0  启动



模型路径为 /home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Instruct


数据集路径为 /home/ma-user/work/datas/VisuRiddles
