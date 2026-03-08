# EntropyAwareMLP Timing Analysis

## Debug Timing Feature

Added comprehensive timing instrumentation to [EntropyAwareMLP](../src/entropy_nn/models/mlp.py) to identify latency bottlenecks.

### Usage

```python
from entropy_nn.models.mlp import EntropyAwareMLP

# Create model
model = EntropyAwareMLP.from_trained_mlp(baseline, bits=8, coder="ans")

# Enable timing
model.enable_debug_timing()

# Run inference
for batch in data_loader:
    output = model(batch)

# View results
model.print_timing_stats()

# Get raw stats
stats = model.get_timing_stats()

# Reset for new measurement
model.reset_timing_stats()
```

## Bottleneck Analysis Results

Profiling on MNIST MLP (8-bit, batch size 256, 10 iterations):

### ANS Coder

| Operation | Mean (ms) | % of Total | Notes |
|-----------|-----------|------------|-------|
| **Layer 0 Weight Decode** | 4.543 | 23.3% | **Largest bottleneck** |
| **Layer 0 Total Decode** | 4.956 | 25.4% | Weight + bias + buffers |
| Layer 1 Weight Decode | 0.818 | 4.2% | Smaller layer |
| Layer 1 Total Decode | 1.069 | 5.5% | |
| Layer 2 Weight Decode | 0.091 | 0.5% | Smallest layer |
| Layer 2 Total Decode | 0.287 | 1.5% | |
| **Total Decode Time** | - | **~32.4%** | Sum of all decode ops |
| **Forward (compute)** | 0.295 | 1.5% | All 3 layers |
| **Forward Total** | 6.637 | 34.0% | End-to-end batch time |

### Huffman Coder

| Operation | Mean (ms) | % of Total | Notes |
|-----------|-----------|------------|-------|
| **Layer 0 Weight Decode** | 36.059 | 27.6% | **7.9× slower than ANS!** |
| **Layer 0 Total Decode** | 36.834 | 28.2% | |
| Layer 1 Weight Decode | 5.475 | 4.2% | 6.7× slower than ANS |
| Layer 1 Total Decode | 5.876 | 4.5% | |
| Layer 2 Weight Decode | 0.318 | 0.2% | 3.5× slower than ANS |
| Layer 2 Total Decode | 0.551 | 0.4% | |
| **Total Decode Time** | - | **~33.0%** | Dominates runtime |
| **Forward (compute)** | 0.495 | 0.4% | Similar to ANS |
| **Forward Total** | 43.791 | 33.5% | 6.6× slower than ANS |

## Key Findings

### 🔴 Critical Bottleneck: Weight Decoding

**Layer 0 weight decoding dominates performance:**
- ANS: 4.5 ms (23% of time)
- Huffman: 36.1 ms (28% of time)

This is the largest layer (784→256 = 200,704 params) and requires decoding ~200K symbols per forward pass.

### 📊 Decode vs Compute Breakdown

| Component | ANS | Huffman | Notes |
|-----------|-----|---------|-------|
| **Decode** | 119.3 ms (61%) | 854.9 ms (65%) | Entropy decode + dequantize |
| **Forward** | 3.0 ms (2%) | 5.0 ms (0.4%) | Actual matrix multiply |
| **Overhead** | 72.9 ms (37%) | 447.7 ms (34%) | Buffer mgmt, etc. |
| **Total** | 195.2 ms | 1307.6 ms | Per 10 batches |

### 🐌 Why Huffman is 6.6× Slower

Huffman decode time breakdown per 10 batches:
- Total decode: 854.9 ms (vs ANS: 119.3 ms)
- **7.2× slower decoding**

Reasons:
1. **Symbol-by-symbol Python loop** for each weight
2. **No batch processing** (unlike ANS which decodes arrays)
3. **Python overhead** calling `decode_symbol()` ~230K times per batch

### ⚡ Actual Compute is Fast

Matrix operations (forward pass) are **negligible**:
- ANS: 0.295 ms (1.5% of time)
- Huffman: 0.495 ms (0.4% of time)

The neural network computation itself is not the bottleneck!

### 📦 Buffer Operations are Cheap

- Weight copy (CPU→GPU): 0.267-0.493 ms per layer
- Bias operations: 0.047-0.181 ms per layer
- Buffer transfers: 0.002-0.004 ms (negligible)

## Optimization Opportunities

### 🎯 High Impact

1. **Optimize ANS decode for layer 0** (4.5 ms → target <1 ms)
   - Currently 23% of runtime
   - Use C++ extension or optimized library
   - Consider GPU-based decode

2. **Batch decode across multiple forward passes**
   - Amortize decode cost over N batches
   - Cache decoded weights temporarily
   - Trade-off: memory vs latency

3. **Layer-specific caching**
   - Cache layer 0 weights (changes infrequently?)
   - Only decode when needed
   - Adaptive caching based on memory budget

### 🎨 Medium Impact

4. **Profile ANS library performance**
   - Is constriction optimal for our use case?
   - Try alternative implementations
   - Consider custom ANS decoder for neural nets

5. **Fuse decode + dequantize**
   - Currently two separate operations
   - Combine for better cache locality

6. **Parallel decode**
   - Decode multiple layers in parallel
   - Use threading or async

### 🔬 Low Impact (Huffman-specific)

7. **Switch Huffman to vectorized implementation**
   - Replace Python loop with batch decode
   - Likely requires custom implementation
   - Or just use ANS (already 6.6× faster!)

## Recommendations

### For Production
✅ **Use ANS coder** - 6.6× faster than Huffman

🔄 **Profile decode optimization impact**:
- If we reduce layer 0 decode from 4.5ms to 1ms
- Total batch time: 6.6ms → ~3ms
- Would be ~10× slower than baseline (vs current 20×)

### For Research
📊 **This timing tool is valuable** for:
- Comparing different entropy coders
- Testing decode optimizations
- Understanding resource trade-offs
- Profiling custom implementations

🎯 **Next experiments**:
- Test different quantization bits (4-bit, 6-bit)
- Compare on larger models (ResNet, Transformer)
- Measure on CPU vs GPU
- Profile with different batch sizes

## How to Use Timing Debug

### Quick Test

```bash
# Run the test script
python test_timing_debug.py
```

### In Your Code

```python
import torch
from entropy_nn.models.mlp import EntropyAwareMLP

# Create model
model = EntropyAwareMLP.from_trained_mlp(...)
model = model.to(device)

# Enable timing
model.enable_debug_timing()

# Run inference
for i in range(100):
    output = model(batch)

# Print detailed breakdown
model.print_timing_stats()

# Get stats programmatically
stats = model.get_timing_stats()
for op, timing in stats.items():
    print(f"{op}: {timing['mean_ms']:.2f} ms")
```

### Integration with Profiling Tool

```python
from entropy_nn.experiments.profiling import profile_inference_complete

# Enable timing on model
model.enable_debug_timing()

# Run profiling
profile = profile_inference_complete(model, batch, device)

# Get detailed breakdown
model.print_timing_stats()
```

## Related Files

- Implementation: [src/entropy_nn/models/mlp.py](../src/entropy_nn/models/mlp.py)
- Test script: [test_timing_debug.py](../test_timing_debug.py)
- Comparison CLI: [src/entropy_nn/cli/compare_entropy_aware.py](../src/entropy_nn/cli/compare_entropy_aware.py)
