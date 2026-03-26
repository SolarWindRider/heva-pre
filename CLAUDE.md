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

```
data/loader.py         → Load datasets (VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA)
data/perturbations.py  → Image perturbations (shuffle_patches, gaussian_blur, zero_image, etc.)
    ↓
models/inference.py    → Qwen3-VL inference with attention capture
    ↓
metrics/heva.py        → Monkey-patches Qwen3VLForConditionalGeneration._sample to capture entropy and visual attention during generation
    ↓
analysis/              → Statistics and visualization
    ↓
experiment scripts     → Run experiments (1_run_inference.py, 2_run_inference_heva.py, 7_statistics.py)
```

**Key architectural decision**: The model is monkey-patched at `Qwen3VLForConditionalGeneration._sample` via `_sample_with_vattn_and_entropy` in `metrics/heva.py` to capture per-token entropy and visual attention during generation. This avoids storing full attention tensors.

## Environment

```bash
conda activate PyTorch-2.1.0
```

## Common Commands

```bash
# Run inference with attention capture
python 1_run_inference.py --num_samples 50

# Compute HEVA from inference results
python 2_run_inference_heva.py

# Run statistical analysis
python 7_statistics.py
```

## Key Implementation Details

### Model Requirements
- Uses HuggingFace Transformers Qwen3-VL model
- Must use `eager` attention (not SDPA/flash attention)
- Requires `output_attentions=True` and `output_logits=True`

### Visual Token Identification
- Visual token ID: `151643` (`<|image_pad|>`) in Qwen3-VL
- `get_visual_token_indices()` in `models/inference.py` finds continuous image token range

### Memory Optimization (in generate_with_attn)
- Uses `use_cache=True` (KV cache) for reduced memory
- Only captures `gen_entropy` and `gen_vattn` per token (not full attention tensors)
- `gen_entropy`: (gen_tokens_num, batch_size)
- `gen_vattn`: (gen_tokens_num, batch_size, visual_token_num)

### Expected HEVA Ranges
- Language shortcut samples: 0.01 ~ 0.05
- Visual-dependent samples: 0.1 ~ 0.3
- Extreme visual dependency: > 0.4

### Validation Checks
If HEVA values are all > 0.8, visual token indices are likely incorrect.

## Supported Datasets

`data/loader.py` supports: VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA

## Paths
- Model: `/home/ma-user/work/Downloads/Models/Qwen/Qwen3-VL-2B-Instruct`
- Data root: `/home/ma-user/work/datas`
- Results: `./results/`

## Dependencies
- torch, transformers, numpy, scipy, sklearn, pandas
- matplotlib, seaborn, pillow, tqdm
