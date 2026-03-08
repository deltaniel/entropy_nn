"""Analyze memory usage breakdown for EntropyAwareMLP."""

import torch
from entropy_nn.models import MLP, MNIST_DIMS
from entropy_nn.models.mlp import EntropyAwareMLP

# Load baseline model
print("Loading checkpoint...")
ckpt = torch.load("checkpoints/mnist_mlp_experiment/model_state.pt", map_location="cpu", weights_only=True)
baseline = MLP(dims=MNIST_DIMS, flatten_input=True)
baseline.load_state_dict(ckpt["model_state"], strict=True)

# Create EntropyAware model
print("Creating EntropyAwareMLP (ANS, 8-bit)...")
entropy_model = EntropyAwareMLP.from_trained_mlp(baseline, bits=8, coder="ans")

# Analyze storage size
print("\n" + "="*80)
print("STORAGE SIZE ANALYSIS")
print("="*80)

# Baseline model size
baseline_params = sum(p.numel() for p in baseline.parameters())
baseline_bytes = sum(p.numel() * p.element_size() for p in baseline.parameters())
baseline_mb = baseline_bytes / (1024 * 1024)

print(f"\nBaseline FP32 Model:")
print(f"  Parameters:  {baseline_params:,}")
print(f"  Storage:     {baseline_mb:.3f} MB")
print(f"  Per param:   {baseline_bytes / baseline_params:.1f} bytes")

# EntropyAware compressed size
comp_stats = entropy_model.get_compression_stats()
print(f"\nEntropyAware Compressed Model:")
print(f"  Parameters:  {int(comp_stats['total_params']):,}")
print(f"  Storage:     {comp_stats['encoded_mb']:.3f} MB")
print(f"  Per param:   {comp_stats['encoded_bytes'] / comp_stats['total_params']:.2f} bytes")
print(f"  Compression: {comp_stats['compression_ratio']:.2f}x")

print(f"\n  Saved: {baseline_mb - comp_stats['encoded_mb']:.3f} MB on disk/network")

# Analyze runtime memory
print("\n" + "="*80)
print("RUNTIME MEMORY ANALYSIS")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create sample batch
batch_size = 256
sample_input = torch.randn(batch_size, 28, 28).to(device)

# Baseline runtime memory
baseline = baseline.to(device)
print(f"\nBaseline Model on {device}:")

# Model parameters
model_mem = sum(p.numel() * p.element_size() for p in baseline.parameters())
print(f"  Model params:     {model_mem / (1024**2):.3f} MB")

# Buffers (none for baseline)
print(f"  Buffers:          0.000 MB")

# Estimate activation memory (rough)
# Layer 0: batch × 784 → batch × 256
# Layer 1: batch × 256 → batch × 128
# Layer 2: batch × 128 → batch × 10
activation_mem = (
    batch_size * 784 * 4 +  # Input
    batch_size * 256 * 4 +  # After layer 0
    batch_size * 128 * 4 +  # After layer 1
    batch_size * 10 * 4     # Output
)
print(f"  Est. activations: {activation_mem / (1024**2):.3f} MB (for batch_size={batch_size})")
print(f"  TOTAL (est):      {(model_mem + activation_mem) / (1024**2):.3f} MB")

# EntropyAware runtime memory
entropy_model = entropy_model.to(device)
print(f"\nEntropyAware Model on {device}:")

# Compressed data (stored on CPU in numpy arrays)
compressed_mem = comp_stats['encoded_bytes']
print(f"  Compressed data:  {compressed_mem / (1024**2):.3f} MB (CPU)")

# Decode buffers
buffer_mem = (
    entropy_model.weight_buffer.numel() * entropy_model.weight_buffer.element_size() +
    entropy_model.bias_buffer.numel() * entropy_model.bias_buffer.element_size()
)
print(f"  Decode buffers:   {buffer_mem / (1024**2):.3f} MB (GPU)")
print(f"    - weight_buffer: {entropy_model.weight_buffer.numel():,} elements")
print(f"    - bias_buffer:   {entropy_model.bias_buffer.numel():,} elements")

# Activations (same as baseline)
print(f"  Est. activations: {activation_mem / (1024**2):.3f} MB (same as baseline)")

# Temporary decode tensors (hard to measure, but significant)
largest_layer = max(l.in_features * l.out_features for l in entropy_model.compressed_layers)
temp_mem = largest_layer * 4  # FP32 for temporary decode
print(f"  Temp decode mem:  {temp_mem / (1024**2):.3f} MB (peak)")

total_runtime = compressed_mem + buffer_mem + activation_mem + temp_mem
print(f"  TOTAL (est):      {total_runtime / (1024**2):.3f} MB")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
baseline_total = (model_mem + activation_mem) / (1024**2)
entropy_total = total_runtime / (1024**2)

print(f"\nStorage (on disk):")
print(f"  Baseline:     {baseline_mb:.3f} MB")
print(f"  EntropyAware: {comp_stats['encoded_mb']:.3f} MB")
print(f"  Savings:      {baseline_mb - comp_stats['encoded_mb']:.3f} MB ({comp_stats['compression_ratio']:.2f}x smaller)")

print(f"\nRuntime Memory (estimated):")
print(f"  Baseline:     {baseline_total:.3f} MB")
print(f"  EntropyAware: {entropy_total:.3f} MB")
print(f"  Difference:   {entropy_total - baseline_total:+.3f} MB ({(entropy_total/baseline_total - 1)*100:+.1f}%)")

print("\n" + "="*80)
print("WHY NO MEMORY SAVINGS?")
print("="*80)
print("""
Runtime memory breakdown:

Baseline FP32:
  + Weights on GPU (0.90 MB)
  + Activations (0.72 MB)
  = ~1.6 MB total

EntropyAware:
  + Compressed weights (0.17 MB) - saved!
  - Decode buffers (0.90 MB) - same size as original weights!
  - Temporary decode tensors (0.78 MB) - extra overhead!
  + Activations (0.72 MB) - same
  = ~2.6 MB total (MORE than baseline!)

The problem: We maintain decode buffers sized to hold the LARGEST layer's
uncompressed weights. During inference, we need working memory to decode.

This is why peak memory increases slightly despite 5.4x compression!
""")

print("\n" + "="*80)
print("MEMORY SAVINGS OPPORTUNITIES")
print("="*80)
print("""
1. ELIMINATE PERSISTENT BUFFERS (reduces 0.90 MB)
   - Don't pre-allocate weight_buffer/bias_buffer
   - Decode directly to temporary tensors each time
   - Trade-off: slightly slower (allocation overhead)

2. STREAM DECODING (reduces ~0.78 MB)
   - Decode in chunks rather than full layers
   - Only materialize what's needed for computation
   - More complex implementation

3. IN-PLACE DECODE (reduces ~0.78 MB)
   - Decode directly into computation graph
   - Avoid intermediate tensors
   - Requires custom CUDA kernels

4. COMPRESS ACTIVATIONS TOO (reduces 0.72 MB)
   - Main memory consumer for large batches!
   - Store activations compressed between layers
   - Significant complexity increase

Current implementation optimizes for SPEED (pre-allocated buffers),
not MEMORY. For memory-constrained scenarios, need different approach.
""")
