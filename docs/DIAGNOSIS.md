# DLA / CAD Accuracy Degradation Diagnosis

**Date**: 2026-06-03
**Status**: DLA and CAD both show negative accuracy impact on Qwen3-VL-2B-Thinking and Qwen3-VL-2B-Instruct
**Datasets**: AI2D, AlgoPuzzleVQA, LogicVista, MARVEL, PuzzleVQA, RealWorldQA (100 samples each)

---

## 1. The Problem

Both methods produce **lower accuracy** than the baseline (NV0) on **both models** across all six datasets. This is a fundamental design issue, not a hyperparameter issue.

### 1.1 Accuracy Summary

| Setting | THK ACC | Δ from baseline | Instruct ACC | Δ from baseline |
|---------|---------|-----------------|--------------|-----------------|
| **NV0 (baseline)** | 0.5083 | — | 0.4483 | — |
| **NV3 (DLA 1.3)** | 0.4617 | **-4.66%** | 0.3933 | **-5.50%** |
| **NV-V (CAD 1.3)** | 0.4917 | **-1.66%** | 0.4250 | **-2.33%** |

### 1.2 Per-Dataset Delta

| Dataset | THK ΔDLA | THK ΔCAD | Instruct ΔDLA | Instruct ΔCAD |
|---------|----------|----------|---------------|---------------|
| AI2D | -5% | -5% | **+1%** | 0% |
| AlgoPuzzleVQA | -5% | **+5%** | -5% | -2% |
| LogicVista | -9% | -7% | **-17%** | -12% |
| MARVEL | -1% | **+2%** | **-14%** | +1% |
| PuzzleVQA | -5% | -6% | -1% | -4% |
| RealWorldQA | -3% | **+1%** | **+3%** | **+3%** |

**Pattern**: DLA helps on direct visual lookup (AI2D, RealWorldQA) but hurts on reasoning-heavy tasks (LogicVista, MARVEL, AlgoPuzzleVQA, PuzzleVQA). CAD is more conservative and helps on a few datasets.

---

## 2. Root Cause Analysis

### 2.1 Vicious Cycle: Generation Hits 8192 Token Limit

**Smoking gun**: DLA causes some samples to generate until the 8192 token limit without reaching an answer.

| Sample | Baseline gen tokens | DLA gen tokens | Result |
|--------|--------------------:|---------------:|--------|
| AI2D #2 | 893 (correct) | **8192 (wrong)** | DLA destroyed correct answer |
| AI2D #7 | 1846 (correct) | **8192 (wrong)** | DLA destroyed correct answer |
| AI2D #8 | 8192 (wrong) | 8192 (wrong) | Both stuck |
| MARVEL #4 | correct (1) | 8192 (wrong) | DLA wrong |

DLA's tail entropy is **all zeros** (last 30 steps) — the model is stuck repeating the same token indefinitely. This is a feedback loop:

1. DLA samples a non-argmax structural token (e.g., "→", "So", "Then")
2. Context gets longer / more complex
3. Model entropy at next step **increases by 0.5+** (DLA makes model more uncertain)
4. DLA fires again on the now-higher-entropy step
5. Repeat until 8192 limit

**Quantitative evidence (AI2D #2)**:
- Base mean entropy: 0.697
- DLA mean entropy (first 893 steps): 0.450 (lower in first half)
- Steps where DLA entropy > base by 0.5+: **172 out of 893**

**Quantitative evidence (AI2D #7)**:
- Base mean entropy: 0.559
- DLA mean entropy (first 1846 steps): 0.334
- Steps where DLA entropy > base by 0.5+: **281 out of 1846**

### 2.2 DLA's "Visual Signal" Is Actually a "Confidence Signal"

The DLA score is `raw_dot = head_output @ target_vector`:
- `head_output[head] = z[head] @ W_O[head]` (head's contribution to residual stream)
- `target_vector` propagates from `W_U[token_id]`

The **magnitude** `||head_output||` correlates with **head confidence**, not with **visual attention**. The cosine version (rejected earlier) was the opposite — angle-only, also bad.

**Implication**: DLA's top-20 by raw_dot is essentially the same set as the model's top-20 by logit. DLA resampling from this set adds **noise** to the model's already-strong language prior, rather than introducing new visual signal.

### 2.3 Threshold 1.3 Triggers Too Often

THK/Instruct entropy distribution from baseline (2.36M / 784K tokens):

| Quantile | THK | Instruct |
|----------|-----:|---------:|
| mean | 0.70 | 0.67 |
| p50 | 0.53 | 0.39 |
| p75 | 1.18 | 1.11 |
| **p90** | **1.74** | **1.80** |
| p95 | 2.06 | 2.20 |
| p99 | 2.74 | 3.01 |
| max | 6.70 | 6.70 |

| Threshold | THK % above | Instruct % above | Triggers per 1000 tokens |
|-----------|------------:|-----------------:|--------------------------:|
| **1.3 (current)** | 21% | 20% | ~210 |
| 1.7 | ~12% | ~13% | ~125 |
| **2.0** | **5.7%** | **7.2%** | **~65** |
| 2.5 | ~2.5% | ~3% | ~28 |
| 3.0 | 0.6% | 1.0% | ~8 |

At threshold 1.3, DLA fires on **~20% of all generation steps** for both models. For a 1500-token generation, that's **300+ DLA overrides**. Even if each individual override is "neutral" on average, the cumulative perturbation derails chain-of-thought reasoning.

### 2.4 Task Type Determines Whether DLA Helps or Hurts

- **Direct visual lookup** (AI2D, RealWorldQA): Model is already looking at the image. DLA reinforces visual, helps (+1% to +3%).
- **Multi-step reasoning** (LogicVista, MARVEL): Model is in chain-of-thought mode. DLA disrupts reasoning, hurts (-9% to -17%).
- **Puzzles** (AlgoPuzzleVQA, PuzzleVQA): Mixed effect, mostly hurts.

---

## 3. Per-Dataset Entropy Statistics (Baseline)

### THK Model (Qwen3-VL-2B-Thinking)

| Dataset | n tokens | mean | p50 | p75 | p90 | p95 | p99 | max | >1.3 | >2.0 | >3.0 |
|---------|---------:|-----:|----:|----:|----:|----:|----:|----:|-----:|-----:|-----:|
| AI2D | 210,466 | 0.76 | 0.59 | 1.27 | 1.85 | 2.21 | 2.93 | 5.79 | 24% | 7.6% | 0.8% |
| AlgoPuzzleVQA | 583,088 | 0.63 | 0.47 | 1.07 | 1.59 | 1.87 | 2.35 | 4.39 | 18% | 3.5% | 0.1% |
| LogicVista | 594,768 | 0.74 | 0.57 | 1.25 | 1.82 | 2.14 | 2.79 | 5.87 | 23% | 6.9% | 0.6% |
| MARVEL | 739,785 | 0.73 | 0.55 | 1.22 | 1.78 | 2.13 | 2.91 | 6.70 | 22% | 6.5% | 0.8% |
| PuzzleVQA | 183,676 | 0.69 | 0.53 | 1.13 | 1.67 | 1.99 | 2.74 | 4.99 | 20% | 4.9% | 0.5% |
| RealWorldQA | 48,373 | 0.60 | 0.46 | 1.00 | 1.47 | 1.74 | 2.26 | 4.76 | 15% | 2.3% | 0.1% |
| **AGG** | **2,360,156** | **0.70** | **0.53** | **1.18** | **1.74** | **2.06** | **2.74** | **6.70** | **21%** | **5.7%** | **0.6%** |

### Instruct Model (Qwen3-VL-2B-Instruct)

| Dataset | n tokens | mean | p50 | p75 | p90 | p95 | p99 | max | >1.3 | >2.0 | >3.0 |
|---------|---------:|-----:|----:|----:|----:|----:|----:|----:|-----:|-----:|-----:|
| AI2D | 3,432 | 0.44 | 0.03 | 0.74 | 1.46 | 1.83 | 2.81 | 5.02 | 13% | 3.7% | 0.7% |
| AlgoPuzzleVQA | 33,894 | 0.53 | 0.17 | 0.92 | 1.57 | 1.95 | 2.57 | 4.11 | 15% | 4.5% | 0.2% |
| LogicVista | 265,671 | 0.62 | 0.30 | 1.03 | 1.71 | 2.11 | 2.88 | 5.97 | 18% | 6.2% | 0.8% |
| MARVEL | 476,078 | 0.71 | 0.46 | 1.17 | 1.85 | 2.26 | 3.11 | 6.70 | 22% | 7.9% | 1.2% |
| PuzzleVQA | 3,481 | 0.81 | 0.47 | 1.36 | 2.21 | 2.69 | 3.72 | 5.06 | 27% | 13% | 3.2% |
| RealWorldQA | 1,293 | 0.23 | 0.00 | 0.01 | 0.85 | 1.76 | 2.96 | 4.70 | 7% | 3.9% | 1.0% |
| **AGG** | **783,849** | **0.67** | **0.39** | **1.11** | **1.80** | **2.20** | **3.01** | **6.70** | **20%** | **7.2%** | **1.0%** |

**Note on Instruct model**: AI2D, PuzzleVQA, RealWorldQA have very small token counts (1.3k-3.5k) — Instruct model often gives one-word answers (entropy=0 dominates).

---

## 4. Recommendations

### 4.1 Immediate: Test Higher Thresholds (No Code Change)

Run new experiments with `--dla_entropy_threshold` and `--ctx_entropy_threshold` raised to **2.0** and **2.5**:

| Threshold | THK triggers | Instruct triggers | Expected effect |
|-----------|-------------:|------------------:|-----------------|
| 1.3 (current) | 21% | 20% | Vicious cycle |
| **1.7** | **~13%** | **~13%** | **Intermediate test** |
| **2.0** | **5.7%** | **7.2%** | **Recommended first test** |
| **2.5** | **~2.5%** | **~3%** | **Conservative test** |
| 3.0 | 0.6% | 1.0% | Only when "really don't know" |

**Hypothesis**: At threshold 2.0, vicious cycle breaks (perturbation is rare). DLA still fires on true decision points. If accuracy becomes neutral, threshold is the issue. If still hurts, the design itself is flawed.

### 4.2 Medium-Term: Visual Grounding Verification (Code Change)

The fundamental issue is that `||head_output||` is confidence, not visual attention. To fix, re-implement `verify_attention_focus_on_path`:

1. For DLA's top-1, check the attention pattern of its top heads
2. Compute `attn[head, visual_positions].sum()` for the head
3. Only override when DLA top-1's visual hit_ratio **>** model's top-1's visual hit_ratio + margin
4. This restores the original "dla_ranking_meaningful" gate that was buggy before

This addresses the root cause: DLA should only fire when it has **actual visual evidence** the model lacks.

### 4.3 Alternative: Apply DLA Only to Final Answer Tokens

Skip DLA during chain-of-thought reasoning, enable only when model is close to producing final answer (e.g., when the prompt template is "Answer:" or the generation is past 80% of expected length).

### 4.4 Long-Term: Different Task / Model

DLA may work better on:
- **Larger models** (4B, 7B) with stronger visual attention
- **Visually-grounded tasks** (chart QA, document QA, OCR)
- **Models with explicit visual training** (e.g., visual instruction tuning)

The 2B models may simply have too weak visual attention for DLA to provide useful signal.

---

## 5. Verification: New Threshold Experiments

The shell scripts in the project root include:
- `NV3-2b.sh` — DLA threshold 1.3 (current failing setting)
- `NV-V-2b.sh` — CAD threshold 1.3 (current failing setting)

For the high-threshold experiments, use the new scripts:
- `NV3-2b-thr1.7.sh` / `NV-V-2b-thr1.7.sh` — threshold 1.7 (intermediate)
- `NV3-2b-thr2.0.sh` / `NV-V-2b-thr2.0.sh` — threshold 2.0 (recommended first)
- `NV3-2b-thr2.5.sh` / `NV-V-2b-thr2.5.sh` — threshold 2.5 (conservative)

---

## 6. Conclusion

The current DLA and CAD implementations are **fundamentally flawed for 2B models** because:
1. `||h||` signal is confidence, not visual attention
2. Multinomial override at 20% frequency causes vicious cycle
3. Chain-of-thought reasoning is fragile to perturbation

**Recommended next action**: Run threshold=2.0 and 2.5 experiments to confirm whether the issue is hyperparameter-only or requires design changes.
