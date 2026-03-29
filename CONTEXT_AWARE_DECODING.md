# ContextAwareLogitsProcessor 实现文档

## 概述

本文档描述了 `ContextAwareLogitsProcessor` 的实现。这是一个 **Context-Aware Decoding** 的 LogitsProcessor，当模型对下一个 token 不确定（高熵）时，引入"上下文关注型 attention heads"对候选 token 的支持度来辅助决策。

**核心思想**（来自 doc2.md）：
```
P(token) + λ × ContextEvidence(token)
```

这里的 "Context" 可以是任意你想让模型在不确定时参考的上下文元素。在本项目中，这些上下文元素恰好是**视觉 token**，但方法论是通用的。

---

## 整体架构

```
logits
    ↓
logits_processor (ContextAwareLogitsProcessor)  ← 插入位置
    ↓
logits_warper (top-k / top-p)
    ↓
softmax → sampling
```

**关键点**：必须在 top-k 截断之前修改 logits，否则无法重新排序 token。

---

## 核心模块

### 1. compute_entropy()

**文件**: [context_aware_logits_processor.py:20-30](metrics/context_aware_logits_processor.py#L20-L30)

计算概率分布的熵值：

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    return entropy  # (batch,)
```

- **输入**: `(batch, vocab)` logits
- **输出**: `(batch,)` 每个样本的熵值
- **原理**: 均匀分布 → 高熵（不确定），尖峰分布 → 低熵（确定）

### 2. get_context_token_indices()

**文件**: [context_aware_logits_processor.py:210-280](metrics/context_aware_logits_processor.py#L210-L280)

获取上下文 token 的索引范围，支持三种模式：

```python
def get_context_token_indices(input_ids, processor, image_token_indices=None):
    # 1. 用户指定 → 使用用户指定范围
    if image_token_indices is not None:
        return image_token_indices

    # 2. 自动检测 image tokens
    # ...

    # 3. 无 image tokens → 使用所有 prompt tokens（排除 padding）
    return get_all_prompt_token_indices(input_ids, processor)
```

**优先级**：
1. 用户指定 `image_token_indices` → 使用用户指定的范围
2. 自动检测 image tokens → 找到 `<|image_pad|>` 的位置
3. 无 image tokens → 返回所有非 padding token 的范围

### 3. select_context_heads()

**文件**: [context_aware_logits_processor.py:35-75](metrics/context_aware_logits_processor.py#L35-L75)

从 attention patterns 中识别最关注目标上下文元素（如视觉 token）的 attention heads：

```python
def select_context_heads(attentions, context_token_indices, top_h=5):
    attn = attentions[-1]  # (batch, heads, query, key)
    last_token_attn = attn[:, :, -1, :]  # 最后一个 query 对所有 key 的注意力

    # 对每个 head，计算其对 context token 位置的平均注意力
    for b in batch:
        attn_to_ctx = last_token_attn[b, :, ctx_start:ctx_end + 1].sum(dim=-1)

    # 选 top-h 个 heads
    top_heads = torch.topk(avg_attn_to_ctx, k=top_h).indices.tolist()
    return [(layer_idx, h) for h in top_heads]
```

**核心逻辑**：
1. 取最后一层 attention（语义信息最丰富）
2. 对最后一个生成 token，计算其对 context token 位置的注意力总和
3. 选取得分最高的 heads 作为 "context heads"

> **注意**：在当前的视觉推理场景中，context token = 视觉 token。但这是因为任务需要，而非方法本身的限制。

### 4. compute_token_support_from_attentions()

**文件**: [context_aware_logits_processor.py:80-125](metrics/context_aware_logits_processor.py#L80-L125)

计算候选 token 被 context heads 支持的程度：

```python
def compute_token_support_from_attentions(model, attentions, cache, token_ids, context_heads, context_token_indices):
    # 对每个候选 token
    for token_id in token_ids:
        total_support = 0.0
        for (layer_idx, head_idx) in context_heads:
            # 获取该 head 对 context tokens 的平均注意力
            head_attn_to_ctx = attn[0, head_idx, -1, ctx_start:ctx_end + 1].mean()
            total_support += head_attn_to_ctx.item()
        supports.append(total_support / len(context_heads))
```

**核心逻辑**：context heads 对 context tokens 的平均注意力作为"上下文依赖强度"的代理。

### 5. ContextAwareLogitsProcessor

**文件**: [context_aware_logits_processor.py:130-200](metrics/context_aware_logits_processor.py#L130-L200)

主类，实现 `LogitsProcessor` 接口：

**核心逻辑**：当 entropy > threshold 时，在 top-k 候选中选 context head 最支持的那个。

```python
class ContextAwareLogitsProcessor(LogitsProcessor):
    def __init__(self, model, top_k=20, entropy_threshold=5.0, top_heads=5):
        self.model = model
        self.top_k = top_k                      # 候选 token 数量
        self.entropy_threshold = entropy_threshold  # 触发阈值
        self.top_heads = top_heads               # 使用的 context heads 数量

    def __call__(self, input_ids, scores):
        # 1. 计算熵，低熵则不干预
        entropy = compute_entropy(scores)
        if entropy[0] < self.entropy_threshold:
            return scores

        # 2. 获取 context heads
        context_heads = self._get_context_heads()

        # 3. 取 top-k 候选
        topk_scores, topk_ids = torch.topk(scores[0], k=self.top_k)

        # 4. 计算每个候选的 context support
        supports = self._compute_support(topk_ids)

        # 5. 选择策略：
        #    - 如果有明显 winner（support ratio > 1.5）：直接选中
        #    - 否则 blend context support 与原分布
        ...
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 20 | 候选 token 数量 |
| `entropy_threshold` | 5.0 | 触发阈值 |
| `top_heads` | 5 | 使用的 context heads 数量 |

### 6. ContextAwareModelWrapper

**文件**: [context_aware_logits_processor.py:290-340](metrics/context_aware_logits_processor.py#L290-L340)

用于注册 forward hooks 捕获 attention 的辅助类：

```python
class ContextAwareModelWrapper:
    def _register_hooks(self):
        def get_attention_hook(module, input, output):
            if hasattr(output, 'attentions'):
                self._attentions = output.attentions

        # 在每层注册 hook
        for layer in self.model.model.layers:
            self._hooks.append(layer.register_forward_hook(get_attention_hook))
```

**用法**：
```python
with ContextAwareModelWrapper(model, processor) as wrapper:
    outputs = model.generate(...)
    attentions = wrapper.get_attentions()
```

---

## 与 generate() 的集成

### 方式1: 通过 monkey-patched _sample（推荐）

当前 HEVA 项目使用的方式，`heva.py` monkey-patch 了 `_sample` 方法：

```python
from metrics.heva import _sample_with_vattn_and_entropy
Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy
```

在 `_sample_with_vattn_and_entropy` 中：
1. 每个生成步骤都会调用 `logits_processor`
2. `model.gen_entropy`, `model.gen_vattn` 被填充
3. `model._last_attentions` 可被 LogitsProcessor 访问

### 方式2: 通过 ContextAwareModelWrapper

注册 forward hooks 捕获 attention：

```python
processor = ContextAwareLogitsProcessor(model=model, ...)
processor.set_context_token_indices(context_token_indices)

with ContextAwareModelWrapper(model, processor) as wrapper:
    outputs = model.generate(
        **inputs,
        logits_processor=[processor],
        output_attentions=True,
    )
```

---

## 使用示例

### 完整使用流程

```python
from metrics.context_aware_logits_processor import (
    ContextAwareLogitsProcessor,
    create_context_aware_processor,
)

# 1. 加载模型（需要 eager attention）
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # 必须！
)

# 2. 准备输入
inputs = processor.apply_chat_template(messages, return_tensors="pt")
inputs = inputs.to(model.device)

# 3. 获取 context token indices（这里是视觉 token，但可以泛化）
context_token_indices = get_context_token_indices(inputs["input_ids"], processor)

# 4. 创建 LogitsProcessor
ctx_processor = create_context_aware_processor(
    model=model,
    processor=processor,
    top_k=20,
    entropy_threshold=5.0,
    lambda_=0.5,
    top_heads=5,
)
ctx_processor.set_context_token_indices(context_token_indices)

# 5. 生成
outputs = model.generate(
    **inputs,
    logits_processor=[ctx_processor],  # ← 插入这里
    max_new_tokens=100,
    do_sample=True,
    top_k=20,
    output_attentions=True,  # 必须！
)
```

---

## 测试说明

**文件**: [devp3.py](devp3.py)

功能测试包含 5 个测试用例：

| 测试 | 内容 |
|------|------|
| `test_import` | 验证模块导入正确 |
| `test_compute_entropy` | 验证熵计算：均匀分布 > 尖峰分布 |
| `test_select_context_heads` | 验证 context heads 检测：模拟 attention，验证选中正确的 heads |
| `test_logits_processor_forward` | 验证 `__call__` 方法：高熵时干预，低熵时直接返回 |
| `test_end_to_end_with_model` | 端到端测试：与 Qwen3-VL 模型集成，验证生成正常 |

### 运行测试

```bash
cd /home/ma-user/work/heva-pre
python devp3.py
```

---

## 已知问题与限制

1. **Context token ID**: 不同模型版本使用不同的 token ID，需要从 tokenizer 动态获取

2. **Batch size**: 当前实现主要支持 batch=1 的情况

3. **Attention 获取**: 需要 `output_attentions=True` 或通过 hooks 捕获

4. **性能**: 每次生成步骤都会调用 `__call__`，有轻微额外开销

---

## 方法论总结

**Context-Aware Decoding**：当模型在生成时处于"不确定"状态（高熵），它倾向于依赖语言先验而非实际输入的上下文信息。这个 LogitsProcessor 通过 context heads 的证据来辅助选择 token。

**选择策略**：
- 当 entropy > threshold 时，对 top-k 候选 token 计算 context support
- 如果 context head 对某个 token 的支持远高于其他候选（ratio > 1.5），直接选中它
- 否则，将 context support 与原分布 blend 后再选

**本质**：不是调整概率，而是**用 context 证据替代/融合模型判断**。

---

## 参考文献

- doc2.md: 原始设计文档
- HuggingFace Transformers: `LogitsProcessor` 接口
