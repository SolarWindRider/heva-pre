# Context-Aware Decoding (CAD) Method Specification

## Overview

Context-Aware Decoding (CAD) is a logits processor that boosts tokens supported by "context heads" — attention heads that focus on context tokens (e.g., visual tokens in multimodal tasks). CAD activates only when the model is uncertain (entropy above a threshold). When triggered, it filters the model's top-k candidates to retain only those with high context-head support, then sets scores for dropped candidates to `-inf`.

CAD is defined by the doc2.md design formula:

```
P(token) + lambda * ContextEvidence(token)
```

where ContextEvidence is computed using attention heads that most attend to context tokens.

---

## Algorithm

```
INPUT: input_ids, scores (batch, vocab), model, processor
OUTPUT: modified scores

1. Compute entropy for each sample in batch:
   entropy[b] = -(probs * log(probs + 1e-9)).sum(dim=-1)

2. For each sample b where entropy[b] >= entropy_threshold:
   a. Detect or use provided context token indices (start, end)
   b. Get top-k candidate token IDs by logit score
   c. Select top_h context heads from last layer attention:
      - For last generated token, sum attention to context positions
      - Average over batch
      - Return top_h heads as (layer_idx, head_idx) tuples
   d. Compute context support for each candidate token:
      - Retrieve z from model.model.language_model.layers[-1].self_attn._last_z
      - For each context head (only head_idx matters):
        contribution = z[0, -1, head_idx, :] @ W_O[head_idx] @ W_U[token_id]
      - Support(token) = sum(contributions) / num_context_heads
   e. Keep top-k//2 candidates with highest support
   f. Set scores for dropped candidates to -inf
   g. Store keep_indices in model._cad_top_k[b] for DLA intersection

3. Return modified scores
```

---

## Code Walkthrough

### File: `metrics/context_aware_logits_processor.py`

#### `compute_entropy()` — Lines 23-35

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    return entropy
```

Computes Shannon entropy across the vocabulary dimension. Input: `(batch, vocab)`. Output: `(batch,)`. Uses `+1e-9` to prevent log(0).

---

#### `select_context_heads()` — Lines 38-82

```python
def select_context_heads(attentions: tuple, context_token_indices: tuple, top_h: int = 5) -> list:
```

Identifies attention heads that most attend to context tokens.

**Parameters:**
- `attentions`: Tuple of attention tensors per layer, each `(batch, heads, query, key)`. Uses only `attentions[-1]` (last layer).
- `context_token_indices`: `(start_indices, end_indices)`, each shape `(batch,)`
- `top_h`: Number of top heads to select (default 5)

**Returns:** List of `(layer_idx, head_idx)` tuples for the last layer only.

**Logic (lines 56-76):**
1. Extract `last_token_attn = attn[:, :, -1, :]` — attention from last generated token to all previous positions, shape `(batch, heads, seq_len)`.
2. For each batch element, sum attention weights over the context token range `[ctx_start, ctx_end + 1]`.
3. Average across batch to get `(heads,)` score vector.
4. Select top-k heads via `torch.topk`.

**Returns:** `[(len(attentions) - 1, h) for h in top_heads]` — all from the last layer (layer index is always `len(attentions) - 1`).

---

#### `compute_token_support_from_attentions()` — Lines 85-173

```python
def compute_token_support_from_attentions(
    model,
    attentions: tuple,
    token_ids: torch.Tensor,
    context_heads: list,
    context_token_indices: tuple,
) -> torch.Tensor:
```

Computes per-token context support using `z @ W_O @ W_U`.

**Parameters:**
- `model`: Qwen3VLForConditionalGeneration instance
- `attentions`: Ignored — z comes from `model._last_z` cached on the model
- `token_ids`: Candidate token IDs, shape `(k,)`
- `context_heads`: List of `(layer_idx, head_idx)` tuples
- `context_token_indices`: `(start, end)` for context tokens (unused in this function)

**Returns:** `torch.Tensor` of shape `(k,)` — support score per candidate token.

**Key implementation details:**

1. **z retrieval (lines 117-123):**
   ```python
   last_layer = model.model.language_model.layers[-1]
   z = getattr(last_layer.self_attn, "_last_z", None)
   ```
   `z` shape: `(batch, seq, heads, d_head)` = `(1, 1, 16, 128)` for 2B model.
   `attn_implementation="eager"` causes `eager_attention_forward` to transpose `(batch, heads, seq, d_head)` to `(batch, seq, heads, d_head)` before storing in `_last_z`.

2. **W_O reshape (lines 131-138):**
   ```python
   W_O = last_layer.self_attn.o_proj.weight  # (d_model, n_heads * head_dim)
   W_O = W_O.view(n_heads_cfg, head_dim, d_model)  # (n_heads, head_dim, d_model)
   ```
   For 2B: `(2048, 2048)` reshaped to `(16, 128, 2048)`.

3. **Per-token computation (lines 151-171):** For each `token_id`:
   - Extract `head_z = z[0, -1, head_idx, :]` — shape `(d_head,)`, the z vector for the last position and this head.
   - Compute `head_output = head_z @ head_W_O` — shape `(d_model,)` — this is `z_head @ W_O`.
   - Compute `contribution = head_output @ W_U[token_id]` — scalar dot product.
   - Sum contributions across all context heads, divide by `len(context_heads)`.

---

#### `ContextAwareLogitsProcessor` class — Lines 176-288

```python
class ContextAwareLogitsProcessor(LogitsProcessor):
```

Logits processor that selects tokens based on context-head support when model is uncertain.

**Constructor (lines 187-211):**
```python
def __init__(
    self,
    model,
    top_k: int = 20,
    entropy_threshold: float = 5.0,
    top_heads: int = 5,
):
    self.model = model
    self.top_k = top_k
    self.entropy_threshold = entropy_threshold
    self.top_heads = top_heads
    self._last_attentions = None
    self._last_hidden_states = None
    self._context_token_indices = None
```

- `top_k`: Number of top tokens to consider (default 20).
- `entropy_threshold`: Entropy threshold to trigger CAD (default 5.0).
- `top_heads`: Number of top context heads to identify (default 5).

**`set_context_token_indices()` — Lines 213-215:**
```python
def set_context_token_indices(self, indices: tuple):
    self._context_token_indices = indices
```

Stores the `(start_indices, end_indices)` tuple for context token positions.

**`_get_context_heads()` — Lines 217-221:**
```python
def _get_context_heads(self) -> list:
    if not hasattr(self.model, "_last_attentions") or self.model._last_attentions is None:
        return []
    return select_context_heads(self.model._last_attentions, self._context_token_indices, self.top_heads)
```

Retrieves context heads from `model._last_attentions`, which is populated by the monkey-patch in `inference.py` line 12.

**`_compute_support()` — Lines 223-234:**
```python
def _compute_support(self, token_ids: torch.Tensor) -> torch.Tensor:
    if not hasattr(self.model, "_last_attentions") or self.model._last_attentions is None:
        return torch.zeros(len(token_ids), device=token_ids.device)
    return compute_token_support_from_attentions(
        self.model,
        self.model._last_attentions,
        token_ids,
        self._get_context_heads(),
        self._context_token_indices,
    )
```

**`__call__()` — Lines 236-288:**

Main entry point. Applies CAD for each sample in batch where `entropy >= entropy_threshold`.

```python
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
```

**Logic per sample (lines 256-281):**

1. Skip if `entropy[b] < self.entropy_threshold` (line 257-258).
2. Get context token range for this sample (lines 260-263).
3. Get top-k token IDs by logit score (lines 265-266):
   ```python
   k = min(self.top_k, scores.shape[-1])
   _, topk_ids = torch.topk(scores[b], k=k)
   ```
4. Compute context support for each candidate (line 268):
   ```python
   supports = self._compute_support(topk_ids)
   ```
5. Determine threshold to keep top-k//2 (lines 269-272):
   ```python
   keep_count = max(1, k // 2)
   sorted_supports, _ = torch.sort(supports, descending=True)
   threshold = sorted_supports[keep_count - 1].item()
   keep_mask = (supports >= threshold)
   ```
6. Build drop mask and set `scores[b, drop_mask] = -inf` (lines 274-281):
   ```python
   drop_mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
   drop_mask[topk_ids] = False
   drop_mask[topk_ids[~keep_mask]] = True
   scores[b, drop_mask] = -float("inf")
   ```
7. Store CAD candidates for DLA intersection (lines 283-286):
   ```python
   if not hasattr(self.model, "_cad_top_k"):
       self.model._cad_top_k = {}
   self.model._cad_top_k[b] = set(keep_indices)
   ```

---

#### `ContextAwareModelWrapper` class — Lines 291-362

```python
class ContextAwareModelWrapper:
```

Attaches forward hooks to capture attention tensors and z vectors.

**`_register_hooks()` — Lines 305-335:**

Registers two types of hooks:

1. **Attention capture hook (lines 310-314):**
   ```python
   def get_attention_hook(module, input, output):
       if hasattr(output, "attentions") and output.attentions is not None:
           self._attentions = output.attentions
       if hasattr(output, "hidden_states") and output.hidden_states is not None:
           self._hidden_states = output.hidden_states
   ```
   Attached to each layer in `model.model.language_model.layers`.

2. **z capture hook (lines 316-319):**
   ```python
   def capture_z_hook(module, input, output):
       self._last_z = input[0].clone()
   ```
   Attached to `last_layer.self_attn.o_proj` — captures the input to the output projection, which is z before `W_O`.

**Context manager (`__enter__`/`__exit__`) — Lines 343-349:** Automatically registers hooks on enter and removes on exit.

---

#### `get_context_token_indices()` — Lines 364-405

```python
def get_context_token_indices(
    input_ids: torch.Tensor,
    processor,
    image_token_indices: tuple = None,
) -> tuple:
```

Determines which tokens are "context tokens" (e.g., visual tokens).

**Priority (lines 386-405):**
1. If `image_token_indices` is provided by caller — use it directly.
2. Auto-detect using `get_visual_token_indices(input_ids, processor)` from `inference.py`.
3. If no image tokens found — fall back to `get_input_token_indices(input_ids, processor)` (all non-padding tokens).

---

#### `get_image_token_id()` — Lines 408-425

```python
def get_image_token_id(processor) -> int:
```

Gets the `<|image_pad|>` token ID from the processor's tokenizer. First looks in `additional_special_tokens_ids`, falls back to `151643`.

---

## Hooks and Model Integration

### Monkey-Patch in `metrics/inference.py` (Line 12)

```python
Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy
```

Immediately on import of `metrics/inference.py`, the `_sample` method of `Qwen3VLForConditionalGeneration` is replaced with `_sample_with_vattn_and_entropy` from `metrics/heva.py`. This replacement captures `gen_entropy`, `gen_vattn`, `gen_zs`, and also stores `_last_attentions` on the model instance.

### How Attention and z are Captured

The monkey-patch `_sample_with_vattn_and_entropy` in `heva.py` wraps the model's forward pass. At each generation step:

1. `model.forward()` is called with `output_attentions=True`.
2. The returned `outputs.attentions` is stored as `model._last_attentions` (tuple of per-layer attention tensors).
3. For the last layer, a forward hook on `o_proj` captures `input[0]` as `model._last_z` — the pre-projection attention output z.

### Flow in `3_run_inference_trace.py`

1. Model is loaded with `attn_implementation="eager"` (required for attention capture).
2. `ContextAwareLogitsProcessor` is instantiated with the model.
3. At each generation step, `_sample_with_vattn_and_entropy` calls the model forward and populates `model._last_attentions` and `model._last_z`.
4. `ContextAwareLogitsProcessor.__call__()` reads from `model._last_attentions` to identify context heads and compute support.

---

## Interaction with DLA

CAD and DLA (Direct Logits Attribution) operate as a two-stage filter in `metrics/heva.py` lines 324-330.

### DLA Filter

DLA (lines 306-322 in heva.py) computes causal path scores for each top-k candidate token using `compute_dla_path_for_token()`. It keeps top-k//2 candidates with highest path scores, stored as `dla_candidates` (a Python set).

### CAD Intersection (Lines 324-330)

```python
if use_context_aware and hasattr(self, "_cad_top_k") and self._cad_top_k is not None:
    cad_candidates = self._cad_top_k.get(b, set())
    final_candidates = list(dla_candidates & cad_candidates) if cad_candidates else list(dla_candidates)
else:
    final_candidates = list(dla_candidates)
if not final_candidates:
    final_candidates = list(dla_candidates)
```

**Logic:**
1. If `use_context_aware` is True and `model._cad_top_k` exists, get CAD's keep set for this batch element.
2. Compute set intersection: `dla_candidates & cad_candidates`.
3. If the intersection is non-empty: sample uniformly from the intersected set.
4. If the intersection is empty: fall back to DLA's candidates only (line 330).
5. Sampling is done via multinomial over the final candidate logits (lines 333-334).

**Key point:** CAD is **subtractive**. It only removes tokens — it never boosts scores. The intersection means a token must pass both DLA's causal path filter AND CAD's context support filter to survive.

---

## Configuration

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Qwen3VLForConditionalGeneration | required | Model instance |
| `top_k` | int | 20 | Number of top tokens to consider |
| `entropy_threshold` | float | 5.0 | Entropy threshold to trigger CAD |
| `top_heads` | int | 5 | Number of top context heads to select |

### Setting Context Token Indices

Before using the processor, context token indices must be set:

```python
processor = ContextAwareLogitsProcessor(model)
processor.set_context_token_indices((start_indices, end_indices))
```

Where `start_indices` and `end_indices` are tensors of shape `(batch_size,)` as returned by `get_visual_token_indices()` or `get_context_token_indices()`.

### Usage in Generation

```python
from metrics.context_aware_logits_processor import ContextAwareLogitsProcessor

cad_processor = ContextAwareLogitsProcessor(model, top_k=20, entropy_threshold=5.0, top_heads=5)
cad_processor.set_context_token_indices(visual_token_indices)

outputs = model.generate(
    **inputs,
    logits_processor=[cad_processor],
    ...
)
```

### Key Stored Attributes

| Attribute | Location | Purpose |
|-----------|----------|---------|
| `model._last_attentions` | Set by `_sample_with_vattn_and_entropy` in heva.py | Tuple of per-layer attention tensors |
| `model._last_z` | Set by forward hook on last layer's o_proj | Pre-W_O attention output for last layer |
| `model._cad_top_k` | Set by `ContextAwareLogitsProcessor.__call__()` line 285 | Dict mapping batch index to set of kept token IDs |
| `model.visual_token_indices` | Set in inference.py line 199 | `(start, end)` tensor for visual tokens |

---

## File Locations Summary

| Component | File | Key Lines |
|-----------|------|-----------|
| `compute_entropy` | `metrics/context_aware_logits_processor.py` | 23-35 |
| `select_context_heads` | `metrics/context_aware_logits_processor.py` | 38-82 |
| `compute_token_support_from_attentions` | `metrics/context_aware_logits_processor.py` | 85-173 |
| `ContextAwareLogitsProcessor` | `metrics/context_aware_logits_processor.py` | 176-288 |
| `ContextAwareModelWrapper` | `metrics/context_aware_logits_processor.py` | 291-362 |
| `get_context_token_indices` | `metrics/context_aware_logits_processor.py` | 364-405 |
| `get_image_token_id` | `metrics/context_aware_logits_processor.py` | 408-425 |
| Monkey-patch | `metrics/inference.py` | 12 |
| `get_visual_token_indices` | `metrics/inference.py` | 47-82 |
| `get_input_token_indices` | `metrics/inference.py` | 84-120 |
| `_sample_with_vattn_and_entropy` | `metrics/heva.py` | — |
| DLA intersection with CAD | `metrics/heva.py` | 324-330 |