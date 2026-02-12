# CLAUDE.md

**Meta-instruction**: Keep this file minimal (< 100 lines). Include only universally necessary context. Remove anything discoverable through code search. Never add task-specific instructions.

## Project Overview

`entropy_nn` is a solo research project exploring entropy-based compression to reduce **resource consumption** (memory, energy, compute) during neural network training and inference. The core idea: store weights/activations in compressed form and decode on-demand (layer-by-layer) to reduce memory bandwidth and energy usage.

**Problem**: Memory access dominates energy consumption in neural networks (~100× more than arithmetic). Neural network training/inference is resource-intensive and energy-expensive.

**Goal**: Quantify how on-demand entropy decoding affects peak memory, energy, bandwidth, latency, and accuracy using testbed models.

**Current focus**: Quantization → entropy measurement → compression analysis on MNIST MLP. Next: decode-on-fly inference with memory/energy profiling.

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

**Core Innovation**: Treat entropy as a first-class execution metric. Weights/activations stay compressed until computation time, trading decode compute for memory bandwidth (and thus energy).

**Why This Matters**: Memory access costs ~100× more energy than arithmetic. Reducing memory bandwidth directly reduces energy consumption and enables training/inference on resource-constrained hardware.

**Next priorities**:
1. Decode-on-the-fly inference pipeline (compress → decode per layer → measure peak memory + energy)
2. Energy break-even analysis (when does decode energy < saved memory energy?)
3. Activation compression during training (novel, high-impact)

**Literature Gap**: Most work focuses on storage compression or theoretical ratios. We target runtime energy/memory reduction with empirical validation on real hardware.

See [docs/research_direction.md](docs/research_direction.md) for detailed motivation, related work, and implementation roadmap. See README.md for usage examples and workflow documentation.
