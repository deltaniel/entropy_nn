# Huffman Coder Fix Summary

## Problem

The Huffman coder was failing with "Ran out of bits in compressed data" error during decoding.

## Root Cause

**StackCoder** in the constriction library requires **reusing the same instance** for both encoding and decoding. Creating a new `StackCoder(compressed_data)` instance for decoding doesn't work correctly.

### Investigation

Test results showed:
- ✅ Encoding → decode from **same** StackCoder instance: Works perfectly
- ❌ Encoding → decode from **new** StackCoder instance: Fails after 2 symbols

This is incompatible with our use case where we:
1. Encode weights during model compression
2. Save compressed data
3. Decode later in a different context (can't keep the same StackCoder instance)

## Solution

Switched from **StackCoder** to **QueueEncoder/QueueDecoder**:

```python
# OLD (StackCoder - doesn't work for separate encoding/decoding)
encoder = constriction.symbol.StackCoder()
for sym in reversed(data):  # Stack requires reverse order
    encoder.encode_symbol(sym, codebook)
payload, _ = encoder.get_compressed()

decoder = constriction.symbol.StackCoder(payload)  # ❌ Doesn't work!

# NEW (Queue-based - works correctly)
encoder = constriction.symbol.QueueEncoder()
for sym in data:  # Queue uses forward order
    encoder.encode_symbol(sym, codebook)
payload, _ = encoder.get_compressed()

decoder = constriction.symbol.QueueDecoder(payload)  # ✅ Works!
```

### Code Changes

**File**: [src/entropy_nn/compression/coders.py](../src/entropy_nn/compression/coders.py)

**Encoding** (lines 60-65):
- Changed: `StackCoder()` → `QueueEncoder()`
- Changed: `for sym in s_np[::-1]` → `for sym in s_np` (no reversal needed)

**Decoding** (lines 88-94):
- Changed: `StackCoder(encoded.payload)` → `QueueDecoder(encoded.payload)`

## Performance Results

Comparison on MNIST MLP (8-bit quantization, 235K parameters):

| Model | Memory (MB) | Latency (ms) | Accuracy (%) | Compression |
|-------|------------|--------------|--------------|-------------|
| **Baseline (FP32)** | 11.30 | 0.329 ± 0.062 | 98.07% | 1.00× |
| **EntropyAware (ANS)** | 11.43 | 6.627 ± 0.290 | 98.09% | **5.41×** |
| **EntropyAware (Huffman)** | 11.43 | 40.929 ± 5.792 | 98.09% | **5.38×** |

### Relative to Baseline

| Model | Memory Δ | Latency Δ | Accuracy Δ | Status |
|-------|----------|-----------|------------|--------|
| **ANS** | +1.2% | +1915% (~20× slower) | +0.02% | ❌ FAIL |
| **Huffman** | +1.2% | +12,344% (~124× slower) | +0.02% | ❌ FAIL |

## Key Findings

### ✅ Advantages
1. **Both coders work correctly** now
2. **Excellent compression**: ~5.4× reduction in model size
3. **Accuracy preserved**: 98.09% (same as baseline 98.07%)
4. **Memory footprint**: Minimal increase (+1.2%) during inference

### ⚠️ Challenges
1. **Huffman is 6× slower than ANS** (40.9 ms vs 6.6 ms per batch)
2. **Both are much slower than baseline** due to decode overhead
3. **Memory metric misleading**: Measures peak runtime memory (includes activations + buffers), not stored model size

### Why Huffman is Slower

1. **Symbol-by-symbol processing**: Python loop for each symbol
2. **No batch optimization**: Unlike ANS which can process arrays
3. **Library implementation**: constriction's Huffman may not be optimized for Python

## Recommendations

### For Inference Speed
- **Use ANS coder**: 6× faster than Huffman with same compression ratio
- **Optimize decode path**: Consider caching, batch decoding, or C++ extension

### For Research
- **Huffman is still valuable** for comparing different entropy coding methods
- **Both coders useful** for understanding compression vs latency trade-offs

### Next Steps
1. ✅ **Fixed**: Huffman coder works correctly
2. 🔄 **Optimize**: Profile decode overhead breakdown
3. 🔄 **Measure**: Add "stored model size" metric (separate from runtime peak memory)
4. 🔄 **Experiment**: Test with different quantization bits (4-bit, 6-bit)
5. 🔄 **Explore**: Batch weight decoding across layers

## Usage

```bash
# Compare both coders
compare-entropy-aware \
  --checkpoint checkpoints/mnist_mlp_experiment/model_state.pt \
  --bits 8 \
  --coders ans huffman

# Use only ANS (faster)
compare-entropy-aware \
  --checkpoint checkpoints/mnist_mlp_experiment/model_state.pt \
  --bits 8 \
  --coders ans
```

## Related Files

- Fixed code: [src/entropy_nn/compression/coders.py](../src/entropy_nn/compression/coders.py)
- CLI tool: [src/entropy_nn/cli/compare_entropy_aware.py](../src/entropy_nn/cli/compare_entropy_aware.py)
- Comparison guide: [docs/entropy_aware_comparison.md](entropy_aware_comparison.md)
