# entropy_nn

Research project exploring entropy-based compression for reducing memory usage during neural network execution.

## Quick Start

```bash
# Install
pip install -e .[dev]

# Train a model with experiment tracking
train-mnist-mlp --config configs/train_baseline.yaml

# Evaluate with 8-bit quantization analysis
eval-mnist-mlp --config configs/eval_8bit.yaml

# Or use CLI args directly
train-mnist-mlp --name quick_test --num-epochs 5
eval-mnist-mlp --checkpoint checkpoints/mnist_mlp/model_state.pt --bits 4 --zlib
```

## Features

The project includes a comprehensive experiment tracking system with:

- **YAML configs** for reproducible experiments
- **JSON results** with structured data for analysis
- **Flexible CLI**: supports both config files and plain arguments
- **Data-first design**: analysis functions return structures, printing is separate

### Key Features

1. **Configuration Management**
   - Define experiments in YAML files (see `configs/`)
   - Override any config value via CLI args
   - Full config saved with results for reproducibility

2. **Structured Results**
   - Per-epoch training metrics (loss, accuracy, time)
   - Entropy analysis per-layer and aggregate
   - Zlib compression baseline measurements
   - Activation entropy probes
   - All saved as JSON with timestamps

3. **Flexible Workflows**
   - Use config files for standard experiments
   - Use CLI args for quick iterations
   - Mix both: config file + selective overrides

## Experiment Tracking Workflow

### 1. Training with Configs

```bash
# Use predefined config
train-mnist-mlp --config configs/train_baseline.yaml --name my_run

# Override specific values
train-mnist-mlp --config configs/train_baseline.yaml --num-epochs 20 --lr 0.01
```

**Output:**
- `outputs/my_run_config.yaml` — Full configuration
- `outputs/my_run_results.json` — Training metrics
- `checkpoints/mnist_mlp/model_state.pt` — Model checkpoint

### 2. Evaluation at Multiple Bit-Widths

```bash
# 8-bit analysis
eval-mnist-mlp --config configs/eval_8bit.yaml --name my_run_8bit

# 4-bit analysis
eval-mnist-mlp --config configs/eval_4bit.yaml --name my_run_4bit

# Quick eval with CLI args
eval-mnist-mlp \
  --checkpoint checkpoints/mnist_mlp/model_state.pt \
  --bits 2 \
  --zlib \
  --act-probe \
  --name my_run_2bit
```

**Output:**
- `outputs/my_run_8bit_results.json` — Entropy, compression, accuracy stats
- `outputs/my_run_8bit_config.yaml` — Eval configuration

### 3. Analyzing Results

Results are saved as structured JSON:

```python
import json
from pathlib import Path

# Load results
results = json.load(open("outputs/my_run_results.json"))

# Access training metrics
print(f"Final accuracy: {results['training']['final_test_acc']}")
print(f"Per-epoch loss: {results['training']['test_loss']}")

# Access compression analysis
for layer in results['evaluation']['compression']['per_layer_entropy']:
    print(f"{layer['name']}: {layer['entropy_bits']:.2f} bits/sym")
```

## Project Structure

```
entropy_nn/
├── configs/                  # YAML experiment configs
│   ├── train_baseline.yaml
│   ├── eval_8bit.yaml
│   └── eval_4bit.yaml
├── src/entropy_nn/
│   ├── models/              # Neural network architectures
│   ├── compression/         # Quantization, entropy, baselines
│   ├── data/                # Dataset loaders
│   ├── training/            # Training loops
│   ├── experiments/         # Config and results management
│   └── cli/                 # Command-line interfaces
├── outputs/                 # Experiment results (JSON + YAML)
├── checkpoints/             # Model checkpoints
└── data/                    # Downloaded datasets
```

## Configuration Schema

See [CLAUDE.md](CLAUDE.md) for full documentation of:
- Training config options
- Evaluation config options
- Results JSON schema
- Design patterns

## CLI Options

Commands can be used with or without config files:

```bash
# With config file (saves structured results)
train-mnist-mlp --config configs/train_baseline.yaml

# With plain CLI args (also saves structured results when --name is provided)
train-mnist-mlp --num-epochs 10 --lr 0.001 --name my_experiment

# Mixed: config + overrides
train-mnist-mlp --config configs/train_baseline.yaml --num-epochs 20
```

## Development

```bash
# Format code
ruff format

# Lint
ruff check

# Run tests
pytest
```

## Research Context

- **Goal**: Reduce memory usage during inference via on-demand entropy decoding
- **Approach**: Quantize → measure entropy → compress → decode layer-by-layer
- **Metrics**: Entropy (bits/symbol), compression ratio, accuracy, memory footprint

See [CLAUDE.md](CLAUDE.md) for detailed project overview and research directions.
