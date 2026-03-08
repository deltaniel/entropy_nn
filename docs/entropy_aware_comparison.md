# EntropyAware MLP Comparison Tool

## Overview

The `compare-entropy-aware` CLI tool compares the performance of baseline FP32 MLP models against EntropyAwareMLP models that use on-the-fly entropy decoding with different coders (ANS and Huffman).

## Implementation Summary

### What Was Implemented

1. **Extended Profiling Module** ([profiling.py](../src/entropy_nn/experiments/profiling.py))
   - Added `MultiModelComparison` dataclass for storing comparison results
   - Added `print_multi_model_comparison()` function for formatted output tables
   - Includes relative performance metrics and success criteria checks

2. **New CLI Tool** ([compare_entropy_aware.py](../src/entropy_nn/cli/compare_entropy_aware.py))
   - Loads trained baseline MLP checkpoint
   - Creates EntropyAwareMLP variants with different coders
   - Profiles memory and latency for each model
   - Optionally evaluates accuracy on test set
   - Generates JSON results and comparison tables

3. **Fixed Huffman Decoder Bug** ([coders.py](../src/entropy_nn/compression/coders.py))
   - Fixed stack-based decoding order for Huffman coder (line 94)
   - **Note**: Huffman coder still has issues and needs further debugging

### Key Features

- **Multiple Coder Support**: Compare ANS and Huffman entropy coders
- **Comprehensive Profiling**:
  - Peak memory usage (allocated MB)
  - Inference latency (mean ± std in ms)
  - Model accuracy (optional)
  - Compression ratio
- **Relative Metrics**: Automatic calculation of performance deltas vs baseline
- **Success Criteria**: Based on research goals (Phase 1):
  - Memory reduction ≥40%
  - Latency overhead <2×
  - Accuracy ≥98% of baseline

## Usage

### Basic Usage

```bash
compare-entropy-aware \
  --checkpoint checkpoints/mnist_mlp_experiment/model_state.pt \
  --bits 8 \
  --coders ans huffman
```

### Quick Test (skip accuracy evaluation)

```bash
compare-entropy-aware \
  --checkpoint checkpoints/mnist_mlp_experiment/model_state.pt \
  --bits 8 \
  --coders ans \
  --skip-accuracy \
  --num-iterations 20
```

### All Options

```bash
compare-entropy-aware \
  --checkpoint PATH              # Required: path to trained MLP checkpoint
  --bits {2,3,4,5,6,7,8,16}     # Quantization bits (default: 8)
  --device {cuda,cpu,auto}       # Device (default: auto)
  --coders ans huffman           # Which coders to test (default: both)
  --num-iterations 100           # Latency profiling iterations (default: 100)
  --warmup 10                    # Warmup iterations (default: 10)
  --batch-size 256               # Batch size for profiling (default: 256)
  --skip-accuracy                # Skip accuracy eval (faster)
  --skip-baseline                # Skip baseline profiling
  --output-dir results           # Output directory (default: ./results)
  --name my_experiment           # Experiment name
```

## Example Output

```
============================================================
EntropyAwareMLP Comparison: test_ans_only
============================================================
Checkpoint:     checkpoints\mnist_mlp_experiment\model_state.pt
Device:         cuda
Quantization:   8 bits
Coders:         ans
Profiling:      20 iterations, 5 warmup
============================================================

[1/5] Loading checkpoint...
  Loaded 235146 parameters

[2/5] Loading test data...
  Test set: 10000 samples
  Profile batch size: 256

[3/5] Profiling baseline MLP (FP32)...
  Memory:   11.30 MB
  Latency:  0.252 ± 0.005 ms

[4/5] Profiling EntropyAwareMLP models (8-bit)...
  [1/1] Testing ANS coder...
    Creating compressed model...
    Compression: 5.41x (0.90 MB -> 0.17 MB)
    Profiling...
    Memory:   11.43 MB
    Latency:  6.578 ± 0.539 ms

[5/5] Results:

========================================================================================================================
MULTI-MODEL COMPARISON: test_ans_only
========================================================================================================================
Model                     Bits   Coder      Memory (MB)  Latency (ms)   Accuracy (%) Comp Ratio
------------------------------------------------------------------------------------------------------------------------
Baseline (FP32)           FP32   -          11.30        0.252 ± 0.005  N/A          1.00x
EntropyAware (ANS)        8      ANS        11.43        6.578 ± 0.539  N/A          5.41x

------------------------------------------------------------------------------------------------------------------------
RELATIVE TO BASELINE:
------------------------------------------------------------------------------------------------------------------------
Model                     Bits   Coder      Mem D (%)    Latency D (%)    Acc D (%)    Status
------------------------------------------------------------------------------------------------------------------------
EntropyAware (ANS)        8      ANS        +1.2         +2514.9          N/A          [FAIL]
========================================================================================================================
```

## Current Findings

### Baseline vs EntropyAware (ANS, 8-bit)

**Compression:**
- Model size: **5.41× smaller** (0.90 MB → 0.17 MB)

**Memory (Peak Allocated):**
- Baseline: 11.30 MB
- EntropyAware: 11.43 MB (+1.2%)
- **Note**: Peak memory includes activations + decode buffers, not just model weights

**Latency (Inference Time):**
- Baseline: 0.252 ms
- EntropyAware: 6.578 ms (~26× slower)
- Decode overhead dominates performance

### Why Memory Doesn't Decrease?

The peak memory metric measures total allocated memory during inference, which includes:
1. Input batch and activations
2. Decode buffers (for on-the-fly weight decompression)
3. Temporary tensors

The actual **stored model size** is 5.41× smaller, but during inference we need buffers to decode weights, which offsets the savings.

### Known Issues

1. **Huffman Coder Bug**: Huffman decoder fails with "Ran out of bits" error
   - Likely issue with stack-based encoding/decoding order
   - Needs further investigation with constriction library documentation

2. **Memory Measurement**: Current profiling measures peak allocated memory
   - Should add separate metric for "stored model size" vs "runtime peak memory"
   - Consider measuring bandwidth savings separately

3. **Latency Overhead**: 26× slowdown is significant
   - Decode overhead is the bottleneck
   - Future optimization: batch decoding, caching, parallel decode

## Next Steps

### Immediate Fixes
- [ ] Debug and fix Huffman coder
- [ ] Add separate "stored model size" metric
- [ ] Test with different quantization bits (4-bit, 6-bit)

### Optimizations
- [ ] Profile decode time breakdown (quantize vs entropy decode)
- [ ] Implement decode caching strategies
- [ ] Batch weight decoding across layers
- [ ] Measure actual memory bandwidth savings

### Extended Evaluation
- [ ] Run with accuracy evaluation (`--no-skip-accuracy`)
- [ ] Compare multiple bit widths (2, 4, 6, 8, 16)
- [ ] Profile on CPU vs GPU
- [ ] Measure energy consumption (if hardware supports it)

## Related Files

- CLI: [src/entropy_nn/cli/compare_entropy_aware.py](../src/entropy_nn/cli/compare_entropy_aware.py)
- Profiling: [src/entropy_nn/experiments/profiling.py](../src/entropy_nn/experiments/profiling.py)
- EntropyAwareMLP: [src/entropy_nn/models/mlp.py](../src/entropy_nn/models/mlp.py)
- Coders: [src/entropy_nn/compression/coders.py](../src/entropy_nn/compression/coders.py)
