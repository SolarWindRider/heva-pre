# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HEVA (High-Entropy Visual Attention) is a research project validating an attention-based metric for measuring visual dependence in multimodal language models (Qwen3-VL). The metric measures how much high-entropy generation tokens attend to visual tokens during inference.

**HEVA Formula:**
```
HEVA = (1/|S|) * Σ v_t
where:
- S = top-alpha% (default 20%) highest entropy generation tokens
- v_t = sum of attention from token t to all visual tokens, averaged over heads
```

## Architecture

Pipeline flow:
```
data/loader.py         → Load datasets (VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA)
data/perturbations.py → Image perturbations (shuffle_patches, gaussian_blur, zero_image, etc.)
    ↓
metrics/inference.py  → Qwen3-VL inference with attention capture
    ↓
metrics/heva.py       → Monkey-patches Qwen3VLForConditionalGeneration._sample to capture entropy and visual attention
    ↓
analysis/             → Statistics and visualization
```

**Key architectural decision**: The model is monkey-patched at `Qwen3VLForConditionalGeneration._sample` via `_sample_with_vattn_and_entropy` in `metrics/heva.py` to capture per-token entropy and visual attention during generation. This avoids storing full attention tensors.

## Environment

```bash
conda activate PyTorch-2.1.0
```

## Common Commands

```bash
# Run inference with attention capture (outputs to results/{exp_name}/{dataset}/)
python 1_run_inference.py --exp_name exp001 --dataset VisuRiddles --num_samples 50

# Compute HEVA from inference results
python 2_run_inference_heva.py

# Run statistical analysis
python 7_statistics.py
```

## Supported Datasets

`data/loader.py` supports: VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA

## Paths

- Model: `/home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Instruct`
- Data root: `/home/ma-user/work/datas`
- Results: `./results/`

## Key Implementation Details

### Model Requirements
- Uses HuggingFace Transformers Qwen3-VL model
- Must use `eager` attention (not SDPA/flash attention): `attn_implementation="eager"`
- Requires `output_attentions=True`, `output_logits=True`, `return_dict_in_generate=True`

### Visual Token Identification
- Visual token ID: `151643` (`<|image_pad|>`) in Qwen3-VL
- `get_visual_token_indices()` in `metrics/inference.py` finds continuous image token range

### Memory Optimization (in generate_with_attn)
- Uses `use_cache=True` (KV cache) for reduced memory
- Only captures `gen_entropy` and `gen_vattn` per token (not full attention tensors)
- `gen_entropy`: (gen_tokens_num, batch_size)
- `gen_vattn`: (gen_tokens_num, batch_size, visual_token_num)

### Monkey-Patching Mechanism
- Line 12 in `metrics/inference.py`: `Qwen3VLForConditionalGeneration._sample = _sample_with_vattn_and_entropy`
- This replaces the generation method to capture attention and entropy at each step
- Results stored in model instance attributes: `model.gen_entropy`, `model.gen_vattn`, `model.attn_acc_input`, `model.attn_acc_visual`

### Expected HEVA Ranges
- Language shortcut samples: 0.01 ~ 0.05
- Visual-dependent samples: 0.1 ~ 0.3
- Extreme visual dependency: > 0.4

### Validation Checks (from doc.md)
1. **Attention normalization**: `attn_t.sum(dim=-1) ≈ 1`
2. **Visual token index correct**: print `input_ids[image_token_indices]` to verify image embedding range
3. **HEVA numerical range**: If all > 0.8, visual token indices are likely incorrect

## Success Criteria (from doc.md)

| Experiment | Criterion |
|------------|-----------|
| Perturbation consistency | p < 0.01 |
| Group distinction | Cohen d > 0.8 |
| Causal KL | Cliff delta > 0.6 |
| Temporal peak | entropy & HEVA synchronized |

## Dependencies

- torch, transformers, numpy, scipy, sklearn, pandas
- matplotlib, seaborn, pillow, tqdm
