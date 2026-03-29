很好，这一步我们直接把你的想法**工程化成一个标准、可复用、可插入 `generate()` 的插件**。我会给你一份**可以照着实现的开发文档（接近内部设计文档级别）**。

目标是：

> ✅ 在 HuggingFace Transformers 的 `generate()` 中
> 插入一个 **LogitsProcessor**
> 实现：
> 🔥「高 entropy 时，用 *看图 head* 支持的 token 来重排 top-k」

---

# 🧩 一、整体架构设计

---

## 🎯 插件职责

你的插件做三件事：

```text
logits → (判断 entropy) → (如果高 entropy)
       → top-k → 计算 image-head support
       → 调整 logits → 返回
```

---

## 📦 插件形式

你要实现：

```python
class VisualGroundedLogitsProcessor(LogitsProcessor):
```

并插入：

```python
model.generate(
    ...,
    logits_processor=[VisualGroundedLogitsProcessor(...)]
)
```

---

# 🔄 二、generate() 调用流程（你插入的位置）

---

在 `generate()` 内部：

```text
logits
 ↓
logits_processor   ← ✅ 你在这里
 ↓
logits_warper (top-k / top-p)
 ↓
softmax
 ↓
sampling
```

---

## ❗关键点

👉 **你必须在 top-k 之前修改 logits**

否则：

* token 已被截断
* 你无法重新排序

---

# 🏗 三、核心模块设计

---

# 1️⃣ HEVA / entropy 计算模块

---

## 输入

```python
scores: (batch, vocab)
```

---

## 实现

```python
def compute_entropy(scores):
    probs = torch.softmax(scores, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    return entropy  # (batch,)
```

---

# 2️⃣ Image-head 识别模块（提前完成）

---

## 🔥 推荐设计：在 forward hook 中缓存

你不能在 processor 里重新算 attention（太慢）

---

## 做法：Monkey patch forward

```python
def attach_attention_hooks(model):

    def forward_hook(module, input, output):
        if hasattr(output, "attentions") and output.attentions is not None:
            model._last_attentions = output.attentions

    model._orig_forward = model.forward

    def new_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        outputs = model._orig_forward(*args, **kwargs)

        model._last_attentions = outputs.attentions
        return outputs

    model.forward = new_forward
```

---

## 🔥 Image-head 选择函数

```python
def select_image_heads(model, image_token_positions, top_h=5):
    """
    根据 attention 强度选最关注 image token 的 heads
    """

    attn = model._last_attentions[-1]  # last layer
    # shape: [batch, heads, query, key]

    attn_to_image = attn[:, :, -1, image_token_positions].sum(dim=-1)
    # (batch, heads)

    top_heads = torch.topk(attn_to_image[0], k=top_h).indices

    return [(model.cfg.n_layers - 1, int(h)) for h in top_heads]
```

---

# 3️⃣ Token 支持度计算模块（核心）

---

## 输入

```python
topk_token_ids: (k,)
image_heads: List[(layer, head)]
```

---

## 实现

```python
def compute_support(model, cache, token_ids, image_heads):
    dtype = model.W_U.dtype
    d_model = model.cfg.d_model

    supports = []

    for token_id in token_ids:

        # 取 unembed 向量
        if model.W_U.shape[0] == d_model:
            W_U_token = model.W_U[:, token_id]
        else:
            W_U_token = model.W_U[token_id, :]

        total = 0.0

        for (layer, head) in image_heads:
            z = cache["z", layer][0, -1, head, :]
            W_O = model.W_O[layer][head]

            head_out = z @ W_O
            total += (head_out @ W_U_token).item()

        supports.append(total)

    return torch.tensor(supports, device=token_ids.device)
```

---

# 🚀 四、最终 LogitsProcessor 实现

---

## 完整代码

```python
from transformers import LogitsProcessor
import torch

class VisualGroundedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        model,
        cache,
        image_token_positions,
        top_k=20,
        entropy_threshold=5.0,
        lambda_=0.5,
        top_heads=5,
    ):
        self.model = model
        self.cache = cache
        self.image_token_positions = image_token_positions
        self.top_k = top_k
        self.entropy_threshold = entropy_threshold
        self.lambda_ = lambda_
        self.top_heads = top_heads

    def __call__(self, input_ids, scores):
        """
        scores: (batch, vocab)
        """

        # 1️⃣ entropy
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        # 只处理 batch=1（先简化）
        if entropy[0] < self.entropy_threshold:
            return scores

        # 2️⃣ 选 image heads
        image_heads = select_image_heads(
            self.model,
            self.image_token_positions,
            self.top_heads
        )

        # 3️⃣ top-k
        topk_scores, topk_ids = torch.topk(scores[0], k=self.top_k)

        # 4️⃣ support
        supports = compute_support(
            self.model,
            self.cache,
            topk_ids,
            image_heads
        )

        # normalize
        supports = (supports - supports.mean()) / (supports.std() + 1e-6)

        # 5️⃣ 调整 logits
        adjusted = topk_scores + self.lambda_ * supports

        # 写回
        new_scores = scores.clone()
        new_scores[0, topk_ids] = adjusted

        return new_scores
```

---

# ⚠️ 五、关键工程问题（必须解决）

---

## ❗1. cache 同步问题（最重要）

你的 `cache` 必须是：

```text
当前 step 的 cache
```

---

### ❌ 错误方式

```python
processor(cache=旧cache)
```

---

### ✅ 正确方式

你需要：

👉 在每一步 forward 后更新 cache

---

## ✔ 推荐方案

你可以：

```python
model._last_cache = cache
```

然后 processor 里用：

```python
cache = model._last_cache
```

---

---

## ❗2. generate 不会自动给你 cache！

👉 解决：

用 `return_dict_in_generate=True`

或者：

👉 **自己写 decode loop（推荐）**

---

# 🧪 六、测试方法

---

## baseline

```python
model.generate(prompt)
```

---

## 你的方法

```python
processor = VisualGroundedLogitsProcessor(
    model=model,
    cache=model._last_cache,
    image_token_positions=[0,1,2],
)

model.generate(
    prompt,
    logits_processor=[processor],
    do_sample=True,
    top_k=20
)
```

---

# 📈 七、建议实验

---

## 对比：

* 原始 top-k
* entropy only
* HEVA only
* 🔥 Visual grounded（你这个）

---

## 指标：

* accuracy
* hallucination rate
* image-grounding score

---

# 🧠 八、方法本质（你可以写论文）

---

## 你的方法：

```text
P(token) + λ × VisualEvidence(token)
```

---

## VisualEvidence 来自：

```text
“真正看图的 attention heads”
```

---

👉 这是：

> 🔥 **attention-grounded decoding**

---

# 🧾 九、一句话总结

> 实现的是一个在高不确定性时，让“看图的注意力头”参与决策的 LogitsProcessor，本质是在 decoding 阶段引入 grounded reasoning 信号。

---

