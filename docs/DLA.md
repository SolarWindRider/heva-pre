# Direct Logits Attribution (DLA)

## Overview

Direct Logits Attribution (DLA) is a visual error correction mechanism for Qwen3-VL. At each generation step, when the model ignores visual tokens (producing a language-shortcut prediction), DLA identifies alternative token candidates whose causal path through the residual stream shows stronger visual grounding, and resamples from those candidates.

DLA is one of two complementary mechanisms in the HEVA pipeline. It is gated by `use_attention_guidance`. The other mechanism, Context-Aware Decoding (CAD), is gated by `use_context_aware`. They can be used independently or in combination (DLA + CAD).

## Algorithm (pseudo-code)

```
INPUT: next_token_logits, all_zs (per-layer z tensors), model, b (batch index)
OUTPUT: resampled next token for batch element b

1. If dla_entropy_threshold is set and step_entropy < threshold:
      skip DLA, return argmax(logits)  # DLA not triggered for low-entropy steps

2. candidate_scores = []
3. For each top-k token by logit score:
       tok_id = top_indices[i]
       path_dict, avg_dot = compute_dla_path_for_token(all_zs, model, tok_id, b)
       candidate_scores.append((tok_id, avg_dot))

4. Sort candidate_scores descending by avg_dot
5. keep_count = max(1, len(candidate_scores) // 2)
6. dla_candidates = {tok_id for tok_id, _ in candidate_scores[:keep_count]}

7. If use_context_aware and _cad_top_k exists:
       final_candidates = dla_candidates ∩ cad_candidates
       if empty: fall back to dla_candidates
   Else:
       final_candidates = dla_candidates

8. Sample next token uniformly from final_candidates (multinomial over softmax of original logits restricted to final_candidates)
```

## The z-capture monkey-patch

**File:** `metrics/inference.py` line 12

```python
Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy
```

On import, `inference.py` replaces `Qwen3VLForConditionalGeneration._sample` with the custom `_sample_with_vattn_and_entropy` function defined in `metrics/heva.py`. This is the single injection point for all attention capture and DLA logic. The replacement is permanent for the life of the process.

## Per-step z collection

**File:** `metrics/heva.py` lines 240-273

At each generation step, `_sample_with_vattn_and_entropy` collects z tensors from all layers:

```python
num_layers = self.model.language_model.config.num_hidden_layers
all_z_list = []
for layer_idx in range(num_layers):
    z = self._model_all_z_ref.get(layer_idx)
    if z is not None:
        all_z_list.append(z)
    else:
        first_z = next((v for v in self._model_all_z_ref.values()), None)
        if first_z is not None:
            all_z_list.append(torch.zeros_like(first_z))
        else:
            all_z_list.append(None)
```

`self._model_all_z_ref` is populated by the monkey-patch on `Qwen3VLTextAttention.forward` (lines 136-182). After collection, the z tensors are stacked into shape `(num_layers, batch, seq, heads, d_head)` and appended to `self.gen_zs`:

```python
all_z = torch.stack(padded, dim=0)
self.gen_zs.append(all_z)
```

At the end of each generation step, `_model_all_z_ref` is cleared:

```python
# metrics/heva.py line 405
self._model_all_z_ref.clear()
```

This prevents z from the previous step contaminating the next step.

## The main DLA selection loop

**File:** `metrics/heva.py` lines 280-343

The loop iterates over each batch element. For each element `b`:

1. **Entropy gate** (lines 299-304): If `dla_entropy_threshold` is set and `step_entropy < dla_entropy_threshold`, DLA is skipped and the model keeps its argmax prediction.

2. **Top-k candidates by logit** (lines 307-311): The top `top_k_vocab` tokens are selected by their raw logit scores using `torch.topk`.

3. **Per-candidate DLA scoring** (lines 313-318): For each candidate token, `compute_dla_path_for_token` returns a path dict and an `avg_dot` score.

4. **Ranking and filtering** (lines 319-322): Candidates are sorted by `avg_dot` descending, and the top half (`keep_count = max(1, len(candidates) // 2)`) are kept as `dla_candidates`.

5. **CAD intersection** (lines 324-330): If `use_context_aware` is True and `self._cad_top_k` is available, `final_candidates = dla_candidates ∩ cad_candidates`. If the intersection is non-empty, sample from the intersection. If the intersection is empty, fall back to `dla_candidates` (DLA's ranked list). Only when `dla_candidates` itself is empty does the model retain its original prediction.

6. **Multinomial sampling** (lines 332-336): Valid logits are collected for `final_candidates`, softmax-normalized, and sampled from using `torch.multinomial`:

```python
valid_logits = [next_token_logits[b, tok_id].item() for tok_id in final_candidates]
valid_logits_tensor = torch.tensor(valid_logits, device=next_token_logits.device)
cat_probs = F.softmax(valid_logits_tensor, dim=-1)
next_tokens[b] = int(final_candidates[torch.multinomial(cat_probs, 1).item()])
```

The DLA override mechanism works via `has_attn_guidance_override` (heva.py:337). When `final_candidates` is non-empty for batch element `b`, `has_attn_guidance_override[b] = True`. This flag signals to the generation loop to use the DLA-selected multinomial sample (lines 332-336) instead of the default softmax-multinomial sampling (lines 380-381). No `dla_top1` vs `model_top1` comparison is performed — any non-empty `final_candidates` set triggers DLA sampling.

## compute_dla_path_for_token

**File:** `metrics/heva.py` lines 519-591

This function computes the backward causal path for a candidate token. It mirrors the logic of transformer_lens's `get_backward_causal_path_for_token`.

### Signature

```python
def compute_dla_path_for_token(all_zs, model, token_id, b=0):
```

### Input handling

```python
if isinstance(all_zs, list):
    all_zs = torch.stack(all_zs, dim=1)  # (num_layers, gen_tokens, batch, seq, heads, d_head)
    last_zs = all_zs[:, -1, b, -1, :, :]  # (num_layers, heads, d_head)
else:
    # Single tensor: (num_layers, batch, heads, seq, d_head)
    last_zs = all_zs[:, b, :, -1, :]  # (num_layers, heads, d_head)
```

For a list of per-step z tensors, it stacks them along the token dimension and extracts the last generated token (`-1` in the gen_tokens dimension) for batch element `b`. For a single tensor (already stacked), it extracts the last position (`-1` in the seq dimension).

### Backward path computation

The loop iterates from the last layer backward to the first:

```python
for layer_idx in reversed(range(num_layers)):
    z = last_zs[layer_idx]  # (heads, d_head)
    W_O = _get_layer_W_O(model, layer_idx)  # (heads, d_head, d_model)

    head_outputs = torch.einsum("hd,hdm->hm", z, W_O)  # (heads, d_model)

    head_dots = head_outputs @ target_vector  # (heads,)

    max_head = torch.argmax(head_dots).item()
    max_dot = head_dots[max_head].item()
    path[layer_idx] = {"head": max_head, "score": max_dot}
    layer_dots.append(max_dot)

    target_vector = head_outputs.mean(dim=0)  # residual propagation, no normalization
```

1. `z @ W_O` gives the per-head output contribution to the residual stream.
2. The dot product with `target_vector` (initialized from `W_U[token_id]`) scores how much each head's output aligns with the gradient direction for that token.
3. The head with maximum dot product is recorded for that layer.
4. The residual target for the previous layer is the mean of all head outputs (not normalized, to preserve magnitude).

### W_U extraction

```python
W_U = model.lm_head.weight  # (vocab, d_model)
if W_U.shape[0] == model.model.language_model.config.hidden_size:
    target_vector = W_U[:, token_id].detach()
else:
    target_vector = W_U[token_id, :].detach()
```

The code handles two possible weight layouts for `lm_head`.

### Return

```python
avg_dot = float(sum(layer_dots) / len(layer_dots))
return {k: path[k] for k in sorted(path.keys())}, avg_dot
```

Returns a dict mapping layer index to `{"head": head_idx, "score": dot_value}` and the average dot product across all layers.

## Why raw_dot not cosine

**File:** `metrics/heva.py` lines 575-578

The code comment explains the design decision:

```python
# Raw dot product (no normalization): head_output · target.
# Magnitude of head_output ||h|| IS the signal — heads that contribute
# more to the residual (e.g. by attending to visual tokens) have
# larger ||h||. Cosine threw this away, leaving only noise (~0.03).
head_dots = head_outputs @ target_vector  # (heads,)
```

Cosine similarity normalizes out the magnitude of `head_outputs`, discarding the very signal that distinguishes visually-attending heads. Raw dot product preserves this magnitude as the score. Heads that attend strongly to visual tokens produce larger `head_outputs` vectors, yielding larger dot products with the token gradient.

## The 4 NV Settings

The experiment scripts define four configuration settings (NV0 through NV4) for the Qwen3-VL-2B models:

| Setting | `--use_attention_guidance` | `--use_context_aware` | `--dla_entropy_threshold` | Description |
|--------|:---:|:---:|:---:|---|
| **NV0** | `false` | `false` | — | Standard generation, capture only |
| **NV1** | `true` | `false` | `-10` (all tokens) | DLA-only, every token evaluated |
| **NV2** | `false` | `true` | — | CAD-only, high-entropy triggers filtering |
| **NV3** | `true` | `false` | `1.3` | DLA-only, high-entropy tokens only |
| **NV4** | `true` | `true` | `1.3` | DLA + CAD both filter, intersection is final |

### NV0 (Standard / capture-only)

```bash
# NV0-2B.sh
python 3_run_inference_trace.py ... --use_attention_guidance false --use_context_aware false
```

No intervention. DLA and CAD are both disabled. Used for baseline capture.

### NV1 (DLA on all tokens)

```bash
# NV1-2b.sh
python 3_run_inference_trace.py ... --use_attention_guidance true --dla_entropy_threshold -10 --use_context_aware false
```

`dla_entropy_threshold = -10` is effectively negative infinity, meaning DLA evaluates every token regardless of entropy.

### NV2 (CAD only)

```bash
# NV-V-2b.sh
python 3_run_inference_trace.py ... --use_attention_guidance false --use_context_aware true --ctx_entropy_threshold 1.3
```

CAD evaluates every step. DLA is disabled. Only CAD filtering applies.

### NV3 (DLA on high-entropy tokens)

```bash
# NV3-2b.sh
python 3_run_inference_trace.py ... --use_attention_guidance true --dla_entropy_threshold 1.3 --use_context_aware false
```

DLA only fires for steps where `entropy >= 1.3`. Low-entropy tokens skip DLA.

### NV4 (DLA + CAD)

```bash
# NV4-2b.sh
python 3_run_inference_trace.py ... --use_attention_guidance true --dla_entropy_threshold 1.3 --use_context_aware true --ctx_entropy_threshold 1.3
```

DLA and CAD both filter independently. The final candidate set is their intersection. If empty, DLA does not fire and the model decides via normal sampling.

## CLI Configuration

**File:** `3_run_inference_trace.py` lines 715-731

```python
# Context-Aware Decoding
parser.add_argument("--use_context_aware", type=str, default="false", choices=["true", "false"])
parser.add_argument("--ctx_entropy_threshold", type=float, default=5.0)
parser.add_argument("--ctx_top_heads", type=int, default=5)

# Attention Guidance
parser.add_argument("--use_attention_guidance", type=str, default="false", choices=["true", "false"])
parser.add_argument(
    "--dla_entropy_threshold",
    type=float,
    default=None,
    help="DLA only applied when entropy > threshold",
)
```

Boolean arguments are passed as strings (`"true"` / `"false"`) and converted:

```python
use_context_aware = args.use_context_aware.lower() == "true"
use_attention_guidance = args.use_attention_guidance.lower() == "true"
```

Model attributes set before generation:

```python
model.use_attention_guidance = True
model.attn_guidance_top_k = top_k          # default 40
model.attn_guidance_topk_attn = 5
model.dla_entropy_threshold = dla_entropy_threshold
model.use_context_aware = True
```

## Empirical findings

The DLA mechanism was validated across 12 datasets (VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA, AI2D, RealWorldQA, MMMU, MMMU_Pro, MathVista, MathVision) on both Qwen3-VL-2B-Instruct and Qwen3-VL-2B-Thinking models.

DLA fires whenever its selection produces a non-empty `final_candidates` set (heva.py:337), which is essentially every high-entropy step. There is no separate gating condition that compares visual attention hit_ratio of model's top-1 vs DLA's top-1.

The raw dot product scoring was found to produce meaningful rankings where cosine similarity produced only noise (correlation ~0.03). The magnitude of the head output vector in the residual stream is the discriminative signal, not the angle.

---

## Known Issues (as of 2026-06-03)

### Negative Accuracy Impact on 2B Models

**Empirical finding**: DLA at threshold 1.3 produces **-4.66% accuracy** on Qwen3-VL-2B-Thinking and **-5.50%** on Qwen3-VL-2B-Instruct (averaged over 6 datasets × 100 samples). The full diagnosis is in `docs/DIAGNOSIS.md`.

**Root causes identified**:

1. **Threshold 1.3 fires too often** (~21% of generation steps for both models). For a 1500-token generation, DLA triggers 300+ times, each adding small perturbations that accumulate.

2. **Vicious cycle**: DLA samples non-argmax structural tokens ("→", "So", "Then"). Context gets longer, model entropy rises, DLA fires again. In worst cases, model hits 8192 token limit without reaching answer (e.g., AI2D #2: base 893 tokens → DLA 8192 tokens).

3. **raw_dot signal is confidence, not visual attention**: `||head_output||` correlates with model confidence on the token, not with actual visual attention. DLA's top-20 by raw_dot is essentially the same set as the model's top-20 by logit. DLA resampling adds noise to the model's already-strong language prior.

4. **Task-dependent effect**: DLA helps on direct visual lookup (AI2D, RealWorldQA: +1% to +3%) but hurts on multi-step reasoning (LogicVista, MARVEL: -9% to -17%).

### Mitigations

- **Higher threshold** (recommended first test): 2.0 (5.7% trigger rate) or 2.5 (~2.5% trigger rate). Test scripts: `NV3-2b-thr2.0.sh`, `NV3-2b-thr2.5.sh`.
- **Visual grounding verification** (recommended long-term fix): re-implement `verify_attention_focus_on_path` to check attention pattern of DLA's top heads, only override when DLA actually has visual evidence the model lacks.
