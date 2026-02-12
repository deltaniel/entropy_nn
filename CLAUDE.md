# CLAUDE.md

**Meta-instruction**: Keep this file minimal (< 100 lines). Include only universally necessary context. Remove anything discoverable through code search. Never add task-specific instructions.

## Project Overview

`entropy_nn` is a research project exploring entropy-based compression to reduce memory usage during neural network inference. The core idea: store weights/activations in compressed form and decode on-demand (layer-by-layer) rather than fully decompressing before computation.

**Goal**: Understand how on-demand entropy decoding affects peak memory, bandwidth, latency, and accuracy using small testbed models.

**Current focus**: Quantization → entropy measurement → compression analysis on MNIST MLP.

## Architecture

**Core pattern**: Analysis functions return structured data (dataclasses), printing is separate.

```
src/entropy_nn/
├── models/           # Neural architectures (MLP)
├── compression/      # Quantization, entropy, baselines (returns data structures)
├── data/            # Dataset loaders
├── training/        # Training loops (returns metrics dicts)
├── experiments/     # Config (YAML) and results (JSON) management
└── cli/             # Entry points
```

**CLI**: `train-mnist-mlp`, `eval-mnist-mlp` (support both YAML config + JSON results OR plain CLI args)

## Key Commands

```bash
# Development
pip install -e .[dev]
ruff format && ruff check

# Experiments with config files
train-mnist-mlp --config configs/train_baseline.yaml
eval-mnist-mlp --config configs/eval_8bit.yaml

# CLI args (with or without config)
train-mnist-mlp --num-epochs 20 --lr 0.01
train-mnist-mlp --config configs/train_baseline.yaml --num-epochs 20  # override
```

## Conventions

- Python 3.13+, type hints everywhere: `from __future__ import annotations`
- Ruff: 120 char lines, py313 target
- Data structures over printing: functions return dataclasses, separate `print_*()` for display
- YAML for input configs, JSON for output results
- No tests exist yet (pytest configured in pyproject.toml, testpaths=["tests"])

## Research Context

**Next priorities**:
1. Differentiable entropy estimator for training-time regularization
2. Decode-on-the-fly inference pipeline (compress → decode per layer → measure peak memory)
3. Activation compression during training

See README.md for usage examples and detailed workflow documentation.
