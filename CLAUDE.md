# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HEVA (High-Entropy Visual Attention) is a research project implementing and validating an attention-based metric for measuring visual dependence in multimodal language models (Qwen3-VL). The metric measures how much high-entropy generation tokens attend to visual tokens.

## Architecture

The project follows a modular pipeline architecture:

```
data/loader.py        → Load VisuRiddles/RAVEN datasets
    ↓
models/inference.py   → Qwen3-VL inference with attention capture
    ↓
metrics/heva.py      → HEVA calculation from attention weights
    ↓
analysis/            → Statistics and visualization
    ↓
experiments/         → Experiment scripts (1-7)
```

**Key architectural decision**: HEVA uses only the last decoder layer attention (configurable via `USE_LAST_LAYER_ONLY` in config.py) to minimize GPU memory.

## Common Commands

```bash
# Activate conda environment (REQUIRED)
conda activate PyTorch-2.1.0

# Run inference (generates attention weights)
python experiments/1_run_inference.py --num_samples 50

# Run perturbation experiments
python experiments/3_perturbation_exp.py --perturbation shuffle_patches --num_samples 20

# Run group analysis
python experiments/4_group_analysis.py --num_samples 100

# Run causal sensitivity experiment
python experiments/5_causal_exp.py

# Run temporal analysis
python experiments/6_temporal_analysis.py --num_samples 30

# Run statistics
python experiments/7_statistics.py
```

## Key Implementation Details

### HEVA Formula
```
HEVA = (1/|S|) * Σ v_t

where:
- S = top-alpha% (default 20%) highest entropy generation tokens
- v_t = sum of attention from token t to all visual tokens, averaged over heads
```

### Model Requirements
- Uses HuggingFace Transformers Qwen3-VL model
- Must use `eager` attention (not SDPA/flash attention)
- Requires `output_attentions=True` and `output_scores=True`

### Token Identification
- Visual token indices: identified by special image pad token (151643 in Qwen3-VL)
- Prompt length: determined by finding `<|im_end|>` token (151645)
- Only generation tokens (after prompt) are used for HEVA calculation

### Memory Optimization (config.py)
- `USE_LAST_LAYER_ONLY = True`: Only stores last attention layer
- `CHUNK_SIZE = 512`: Chunk size for long sequence processing
- `CLEAR_CACHE_AFTER_COMPUTE = True`: Clear GPU cache after HEVA calculation

### Expected HEVA Ranges
- Language shortcut samples: 0.01 ~ 0.05
- Visual-dependent samples: 0.1 ~ 0.3
- Extreme visual dependency: > 0.4

### Validation Checks (Important for Debugging)

When implementing or modifying HEVA calculations, verify:

1. **Attention normalization**: `attn_t.sum(dim=-1) ≈ 1` for each token
2. **Visual token indices**: Print `input_ids[image_token_indices]` to confirm image embedding range
3. **HEVA range**: If values are all > 0.8, token indices are likely incorrect

## Data Format

Input JSON format:
```json
{
  "id": "sample_id",
  "image": "image_filename.jpg",
  "question": "question text",
  "options": "A. ...\nB. ...\nC. ...\nD. ...",
  "answer": "A"
}
```

## Paths (from config.py)

- Model: `../Downloads/Models/Qwen/Qwen3-VL-2B-Thinking` (relative to project root)
- Dataset: `../datas/VisuRiddles`
- Results: `./results/`
- Plots: `./plots/`
- Tables: `./tables/`

## Available Experiments

- `1_run_inference.py` - Run inference with attention capture
- `3_perturbation_exp.py` - Visual perturbation consistency
- `4_group_analysis.py` - Visual vs language sample comparison
- `5_causal_exp.py` - Causal sensitivity analysis
- `6_temporal_analysis.py` - Temporal dynamics
- `7_statistics.py` - Statistical analysis

Note: Experiments 2 and 8 are not implemented.

## Dependencies

- torch, transformers
- numpy, scipy, sklearn, pandas
- matplotlib, seaborn
- pillow, tqdm
