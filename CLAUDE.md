# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HEVA (High-Entropy Visual Attention) is a research project that implements and validates an attention-based metric for measuring visual dependence in multimodal language models (Qwen3-VL). The metric measures how much high-entropy generation tokens attend to visual tokens.

## Project Structure

```
heva-pre/
├── config.py                    # Configuration (paths, memory optimization settings)
├── doc.md                       # Technical specification document
├── README.md                    # Quick start guide
├── data/
│   ├── loader.py               # Dataset loading (VisuRiddles, RAVEN, etc.)
│   └── perturbations.py        # Image perturbation functions
├── models/
│   └── inference.py            # Qwen3-VL model inference with attention capture
├── metrics/
│   └── heva.py                 # HEVA calculation implementation
├── analysis/
│   ├── statistics.py            # Statistical analysis
│   └── plots.py                 # Visualization
└── experiments/                 # Experiment scripts (1-7)
```

## Common Commands

```bash
# Activate conda environment
conda activate PyTorch-2.1.0

# Run inference on dataset
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

### HEVA Metric (metrics/heva.py)

The HEVA metric measures visual attention from high-entropy generation tokens:
- Calculate entropy for each generated token: H_t = -sum(p_t * log(p_t))
- Select top-alpha% tokens by entropy (default alpha=0.2)
- Calculate attention from these tokens to visual tokens

Key functions:
- `compute_entropy()` - Calculate token entropy from logits
- `compute_heva()` - Main HEVA calculation
- `compute_heva_chunked()` - Memory-efficient version for long sequences
- `compute_heva_from_result()` - Process inference results

### Model Inference (models/inference.py)

Uses Qwen3-VL model with:
- `generate_with_attention()` - Main inference method that captures attention weights
- Uses `output_attentions=True` and `output_scores=True` in generate() for single-pass inference
- Requires `eager` attention implementation (not SDPA/flash attention)

### Memory Optimization (config.py)

Key settings to reduce GPU memory:
- `USE_LAST_LAYER_ONLY = True` - Only store last attention layer
- `CHUNK_SIZE = 512` - Chunk size for long sequence processing
- `CLEAR_CACHE_AFTER_COMPUTE = True` - Clear GPU cache after HEVA calculation

### Data Format

Input JSON format expected:
```json
{
  "id": "sample_id",
  "image": "image_filename.jpg",
  "question": "question text",
  "options": "A. ...\nB. ...\nC. ...\nD. ...",
  "answer": "A"
}
```

## Dependencies

- torch
- transformers
- numpy, scipy, sklearn
- matplotlib, seaborn, pandas
- pillow, tqdm

## Architecture Notes

- The model uses HuggingFace Transformers Qwen3-VL implementation
- Attention weights are captured during autoregressive generation
- HEVA only uses the last layer attention (configured in config.py)
- Visual token indices are identified by special image pad token (151643 in Qwen3-VL)
- Prompt length is determined by finding the `<|im_end|>` token (151645)
