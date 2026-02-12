# Key Concepts Explained

**Purpose**: Accessible explanations of core concepts for understanding the entropy-first execution research.

**Focus**: Resource efficiency - reducing memory, energy, and computational requirements of neural networks.

---

## Neural Network Basics (Quick Refresher)

### What's a Neural Network?
A function with **learnable parameters** (weights and biases) that maps inputs to outputs.

**Example**: Handwritten digit classifier
```
Input: 28×28 pixel image (784 numbers)
   ↓
Hidden layers: Feature extraction and transformation
   ↓
Output: 10 probabilities (one per digit 0-9)
```

### What Are "Weights"?
The learned parameters of the network - the numbers that get adjusted during training.

**Analogy**: Recipe ingredients and their amounts. The recipe (architecture) is fixed, but the amounts (weights) are learned from data.

**Size**:
- Our MNIST MLP: ~1.2M weights
- Each weight: 32-bit float by default (4 bytes)
- Total: ~4.8 MB uncompressed

---

## Quantization (Reducing Number Precision)

### The Idea
Instead of storing high-precision floating-point numbers, use lower-precision integers.

**Analogy**:
- **32-bit float**: Like measuring with a ruler marked in millimeters (3.14159265...)
- **8-bit integer**: Like measuring with a ruler marked in centimeters (3)
- Still useful, just less precise

### How It Works
```
Original weight: 3.14159265 (32 bits)
              ↓
Find scale factor: max_value / 127
              ↓
Quantized: round(3.14159265 / scale) → 42 (8 bits)
              ↓
Reconstruct: 42 × scale ≈ 3.14
```

**Result**: 4× memory reduction (32 bits → 8 bits), with small accuracy loss

### Trade-offs
- ✅ **Pros**: Less memory, faster transfer, sometimes faster compute
- ❌ **Cons**: Precision loss, potential accuracy drop

**Typical Results**:
- 8-bit: <1% accuracy loss (widely used)
- 4-bit: 1-3% accuracy loss (active research)
- 2-bit: 5-10% accuracy loss (challenging)

---

## Entropy (How Random Are Your Numbers?)

### What Is Entropy?
A measure from **information theory** that tells you: "On average, how many bits do you REALLY need to represent each value?"

**Formula** (don't worry about memorizing):
```
H = -Σ p(x) × log₂(p(x))
```
Where `p(x)` = probability of seeing value `x`

### Intuitive Understanding

**Low Entropy** (Lots of repetition):
```
Data: [5, 5, 5, 5, 6, 5, 5, 5]
Most common value: 5 appears 7/8 times
Entropy: ~0.54 bits per value
Compresses well!
```

**High Entropy** (Very random):
```
Data: [1, 7, 3, 9, 2, 8, 4, 6]
All values unique
Entropy: 3.0 bits per value
Doesn't compress much
```

### Why It Matters for Neural Networks

After quantization, weight distributions often become "spiky" - lots of the same values:

```
Before quantization: [0.142, 0.156, 0.149, 0.151, ...]  ← All different
After 8-bit quantization: [2, 2, 2, 2, 3, 2, 2, ...]   ← Lots of 2's!
```

**Discovery**: "We used 8 bits per weight, but entropy is only 3.5 bits per weight"

**Implication**: Theoretical 2.3× compression opportunity (8 ÷ 3.5)

---

## Compression (Making Data Smaller)

### Lossless Compression
Store the exact same data using fewer bits. Can perfectly reconstruct original.

**Types**:

1. **Run-Length Encoding** (simplest)
   ```
   Original: [5, 5, 5, 5, 6, 6, 2, 2, 2]
   Compressed: [(5,4), (6,2), (2,3)]  ← "5 appears 4 times, then 6 appears 2 times..."
   ```

2. **Huffman Coding** (classic)
   - Assign short codes to frequent values, long codes to rare values
   - Example: '5' → `0`, '6' → `10`, '2' → `11`
   - Achieves compression close to entropy limit

3. **ANS (Asymmetric Numeral Systems)** (modern)
   - Similar to Huffman but better compression
   - Used in Zstandard, JPEG XL
   - More complex to implement

4. **Zlib/gzip** (general purpose)
   - Combines dictionary compression with Huffman
   - Good baseline for comparison

### Lossy Compression
Accept some information loss for better compression.

**In our context**: Quantization is lossy (reduces precision), entropy coding is lossless (perfectly reconstructs quantized values)

---

## The Big Idea: Decode-on-the-Fly Execution

### Traditional Approach
```
1. Load model from disk → Decompress → Store all weights in RAM
2. Run inference: weights already in RAM (fast access)
3. Memory bandwidth: Fetch all weights from DRAM for each forward pass
```

**Problem**:
- All weights must fit in RAM simultaneously
- High memory bandwidth consumption (major energy cost)
- Memory access costs 100× more energy than arithmetic

### Our Proposed Approach
```
1. Store compressed weights in RAM (2-4× smaller)
2. For each layer:
   - Decode only that layer's weights
   - Compute layer output
   - Discard decompressed weights
   - Move to next layer
```

**Benefits**:
- Peak RAM = only 1-2 layers decompressed at a time
- 2-4× less memory bandwidth → 50-75% less energy for memory access
- Smaller models fit on cheaper/lower-power hardware

**Cost**: Must decode during inference (added compute)

**Trade-off Analysis**:
```
Energy saved = (Reduced memory accesses) × (Energy per access)
Energy added = (Decode operations) × (Energy per op)

Win if: Energy_saved > Energy_added

Since memory access costs ~100× more than compute, we win if:
  Compression ratio > ~1.5-2× (achievable with entropy coding!)
```

---

## Why Memory Matters: Energy & Resources

### The Energy Bottleneck

**Memory access dominates energy consumption in neural networks**:
```
Operation          | Energy Cost | Relative
-------------------|-------------|----------
DRAM access (32b)  | ~200 pJ     | 100×
Cache access       | ~10 pJ      | 5×
32-bit multiply    | ~3 pJ       | 1.5×
8-bit multiply     | ~0.2 pJ     | 1×
```

**Implication**: Loading data from memory costs 100-1000× more energy than computing with it!

**Example**: Running inference on a 4GB model
- Memory transfers: ~1 billion accesses × 200 pJ = 200 mJ per inference
- Arithmetic: ~10 billion ops × 0.2 pJ = 2 mJ per inference
- **Memory dominates**: 99% of energy goes to memory access!

### Training Bottleneck

**What limits training?**
1. **Model parameters** (weights)
2. **Activations** (intermediate values, needed for backpropagation)
3. **Optimizer states** (momentum, variance in Adam)

**Memory Breakdown** (typical):
```
Total = 1× weights + 3-5× activations + 2× optimizer states
      = 1× + 3-5× + 2× = 6-8× model size
```

**Example**: A 4GB model needs 24-32GB RAM to train!

**Energy Impact**:
- Larger memory → More power draw
- More memory traffic → More energy per training step
- Batch size limited by memory → More training steps needed

### Inference Bottleneck

**Edge/Mobile Deployment**:
- Limited RAM (1-4GB typical)
- Battery constraints (memory access drains battery)
- No cloud connectivity needed (privacy, latency)

**Datacenter Inference**:
- Millions of queries per day
- Memory bandwidth = major energy cost
- 2× memory reduction = ~50% energy savings

### Our Opportunity

If we can compress:
- **Weights**: 2-4× smaller → 50-75% less memory bandwidth → 50-75% less energy
- **Activations**: 4-8× smaller → Larger batch sizes OR lower memory/energy
- **Hardware**: Fit on smaller/cheaper devices with lower power requirements

---

## Training-Time Compression (The Novel Part)

### Current State
- Weight compression: Well-explored (quantization, pruning)
- Activation compression: Under-explored!

### Why Activations?
During training, must store activations from forward pass for backward pass:

```
Forward:  Input → Layer1 → [save act1] → Layer2 → [save act2] → ... → Loss
                    ↓                      ↓
Backward: Gradient needs act1          Gradient needs act2
```

**Problem**: Activations can be 3-5× larger than weights!

### Proposed Solution
```
Forward:  Layer → activation → [quantize + compress] → save compressed
Backward: [decompress] → use for gradient → discard decompressed
```

**Potential Impact**:
- 4-8× reduction in activation memory
- → 4-8× larger batch sizes on same GPU
- → Faster training (larger batches → fewer updates)

---

## Practical Implications

### 1. Energy Reduction (Primary Goal)

**Current State**: Training large models is energy-intensive
- GPT-3 training: ~1,287 MWh (equivalent to 120 US homes for a year)
- Inference at scale: Millions of queries × memory bandwidth = massive energy

**With Compression**:
```
2× compression → ~50% memory bandwidth reduction
               → ~50% reduction in memory-related energy
               → For inference-heavy applications, this is ~40-50% total energy
```

**Impact**:
- Lower operational costs (energy bills)
- Reduced carbon footprint
- Longer battery life for edge devices

### 2. Hardware Requirements

**Current**: Memory-constrained
- Training: Need expensive high-RAM GPUs (80GB A100s)
- Inference: Edge devices can't run models > 1-2GB

**With Compression**: Democratized access
- Training: 2-4× compression → train on consumer GPUs
- Inference: Run larger models on phones, embedded systems
- Longer hardware lifetime (don't need to upgrade for memory)

**Example**:
```
Model: 4GB weights
Current: Need 24-32GB GPU for training (4-8× model size)
With 2× compression: Could train on 12-16GB GPU (consumer-grade)
```

### 3. Research Accessibility

**Current Barrier**: Many researchers lack access to large compute clusters

**With Compression**:
- Train competitive models on academic/personal hardware
- Faster iteration (train locally, no queue for cluster)
- Lower cost per experiment → more experiments feasible

**Example**: Train a research model that currently needs $10k in compute for $2-3k

### 4. Environmental Impact

**Direct**:
- Lower energy consumption per training run / inference query
- Reduced cooling requirements (less power = less heat)

**Indirect**:
- Less pressure to upgrade hardware → less e-waste
- Smaller models fit on existing hardware longer
- Reduced manufacturing energy for new hardware

---

## Success Criteria

### What Would Make This Successful?

**Minimum Viable**:
- Memory savings: ≥30% peak memory reduction
- Speed: <2× inference time (decode overhead acceptable)
- Accuracy: ≥95% of uncompressed model

**Strong Result**:
- Memory savings: ≥50% peak memory reduction
- Speed: <1.5× inference time
- Activation compression working (proof of concept)

**Game-Changer**:
- Memory savings: ≥60% peak memory reduction
- Speed: ~1× inference time (decode free or HW-accelerated)
- Training with compressed activations: 4-8× larger batches

---

## Common Questions

### Q: Why not just buy more RAM or better hardware?
**A**: This is about efficiency and accessibility:
1. **Energy**: More RAM = more power consumption (ongoing cost + environmental impact)
2. **Access**: Not everyone can afford $80k GPU clusters
3. **Scalability**: Models are growing faster than RAM capacity
4. **Sustainability**: Forced hardware upgrades create e-waste

The question isn't "Can we afford it?" but "Can we do better?"

### Q: Isn't quantization already solving this?
**A**: Quantization helps, but leaves opportunity on the table:
1. Most implementations: 8-bit quantization stored as 8-bit integers (no compression)
2. Our observation: 8-bit quantized weights have ~3.5-bit entropy → wasted opportunity
3. Training-time activation compression: Still under-explored
4. **Our contribution**: Exploit the gap between quantization bitwidth and actual entropy

### Q: Won't decompression be too slow?
**A**: That's the key research question!
- **Hypothesis**: If memory bandwidth-limited, decode compute < memory savings
- **Energy angle**: Decode uses ~0.2 pJ/op, memory access uses ~200 pJ → 1000× difference
- **Empirical**: We need to measure actual trade-offs on real hardware

The bet: Memory is so expensive (energy-wise) that even significant decode overhead is worth it.

### Q: Has anyone done this before?
**A**: Partially, but gaps remain:
- **Storage compression**: Well-explored (compress model for download)
- **Sparse models**: Decode-on-fly exists (DeepSZ, Smart-DNN+)
- **Dense entropy-coded execution**: Some recent work (2024-2025), but limited empirical validation
- **Training-time activation compression**: Very under-explored
- **Energy focus**: Most papers report compression ratios, not energy/memory bandwidth

**Our angle**: Treat entropy as first-class execution metric with empirical energy/memory validation

---

## Further Reading

### Accessible Introductions
- **Quantization**: [Hugging Face Quantization Guide](https://huggingface.co/docs/optimum/concept_guides/quantization)
- **Entropy**: Khan Academy "Information Theory" series
- **Compression**: [Gzip vs Bzip2 vs Zstd](https://catchchallenger.first-world.info/wiki/Quick_Benchmark:_Gzip_vs_Bzip2_vs_LZMA_vs_XZ_vs_LZ4_vs_LZO)

### Technical Deep Dives
- Cover & Thomas, "Elements of Information Theory" (Chapter 2: Entropy)
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)

### Related Research
- See [research_direction.md](research_direction.md) for recent papers and literature review

---

**Questions?** See [research_direction.md](research_direction.md) for detailed research context and implementation plans.
