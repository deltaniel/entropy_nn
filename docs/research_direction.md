# Research Direction: Entropy-Aware Neural Network Framework

**Last Updated**: 2026-02-12  
**Status**: Solo research project (hobby → potential publication)

---

## Executive Summary

This project proposes a **fundamental paradigm shift** in how neural networks are executed: from the traditional "compression-as-afterthought" approach to an **entropy-aware execution framework** where information entropy is a first-class design principle throughout the entire ML lifecycle.

### The Paradigm Shift

**Traditional Paradigm:**
```
Design for accuracy → Train in FP32 → (Maybe) compress for deployment
↓
Compression = Post-processing step for storage
Execution = Operates on fully decompressed data
Memory/bandwidth = Resources to consume as needed
```

**Entropy-Aware Paradigm (This Work):**
```
Co-design for accuracy AND entropy → Train with compression → Execute compressed
↓
Compression = Native execution format
Execution = Lazy decompression (decode only what's needed, when needed)
Memory/bandwidth = Scarce resources to minimize at every stage
Entropy = First-class metric alongside accuracy
```

### Core Insight: The Energy Argument

**The fundamental bottleneck in modern AI isn't computation—it's memory bandwidth.**

```
Energy cost hierarchy (approximate):
- INT8 operation:     1× energy
- FP32 operation:     3× energy  
- DRAM access:      100× energy  ← The real bottleneck!
- Off-chip memory: 1000× energy

Implication: Moving data costs orders of magnitude more than computing on it.
```

**Therefore:** If we can compress data 4×, we can afford significant decode overhead and still achieve massive energy savings. Traditional approaches miss this opportunity by treating compression as separate from execution.

### What Makes This Different

| Aspect | Traditional ML | This Work: Entropy-Aware Framework |
|--------|----------------|-----------------------------------|
| **Design Philosophy** | Accuracy first, compress later | Co-optimize accuracy/bit from start |
| **Data Format** | FP32/FP16 everywhere | Compressed natively, decompress lazily |
| **Memory Model** | Everything fully materialized | Minimal decompression, eager re-compression |
| **Optimization Target** | Maximize accuracy | Maximize accuracy per bit of entropy |
| **Execution Strategy** | Load → compute → store | Load compressed → decode on-fly → compute → re-compress |
| **Metrics** | Accuracy, FLOPs, params | + Entropy/bit, bandwidth, energy |
| **Compression Timing** | Post-training only | Throughout: design, train, inference |

---

## 🎯 Three Pillars of Entropy-Aware Execution

### Pillar 1: Entropy as First-Class Design Metric

**Traditional approach:**
```python
# Design decisions based purely on accuracy
model = Sequential([
    Conv2d(64, 128),   # Double filters → better features → better accuracy
    ReLU(),
    Conv2d(128, 256)   # Keep doubling
])

metrics = {
    'accuracy': 0.762,
    'params': 25M,
    'FLOPs': 4.1B
}
```

**Entropy-aware approach:**
```python
# Design decisions consider information content
model = EntropyAwareSequential([
    Conv2d(64, 128, entropy_budget=4.5),  # Monitor bits/activation
    ReLU(),  # Creates sparsity → lowers entropy (good!)
    Conv2d(128, 256, entropy_budget=5.0)
])

metrics = {
    'accuracy': 0.762,
    'params': 25M,
    'FLOPs': 4.1B,
    
    # NEW: Entropy metrics
    'weight_entropy': 4.2,      # bits per parameter
    'activation_entropy': 5.1,  # bits per activation
    'gradient_entropy': 6.3,    # bits per gradient
    
    # NEW: Efficiency metrics  
    'accuracy_per_bit': 0.181,  # accuracy / weight_entropy
    'compression_ratio': 7.6,   # vs FP32
    'memory_bandwidth': 0.7,    # GB/iter vs 3.2 GB baseline
    'energy_per_sample': 60     # nJ vs 540 nJ baseline
}

# Design choice: 128 filters with 4.5 bits/activation might be better than
# 256 filters with 6.0 bits/activation if accuracy is similar!
```

**Key Innovation:** Entropy becomes a **co-optimization target** with accuracy during architecture design, not an afterthought.

### Pillar 2: Lazy Decompression Execution Model

**Traditional execution:**
```python
# Everything stays decompressed in memory
class TraditionalModel:
    def __init__(self):
        self.weights = load_weights()  # All weights: 800 MB in FP32
        
    def forward(self, x):
        activations = []
        for layer in self.layers:
            # Access full uncompressed weights
            x = layer(x, self.weights[layer.id])  # 4 bytes/param
            activations.append(x)  # Store full FP32: 4 bytes/value
        return x, activations  # Peak memory: GBs
        
    def backward(self, activations):
        # Access all stored activations (fully materialized)
        for act in reversed(activations):
            grad = compute_grad(act)  # Reads GBs from memory
```

**Entropy-aware execution:**
```python
# Decode-on-fly: compress everywhere, decompress minimally
class EntropyAwareModel:
    def __init__(self):
        # Weights stored compressed
        self.compressed_weights = {
            layer_id: entropy_encode(quantize(weights))  # ~0.5 bytes/param
            for layer_id, weights in original_weights.items()
        }
        
        # Reusable decode buffer (shared across layers)
        self.decode_buffer = allocate_buffer(max_layer_size)
        
    def forward(self, x):
        compressed_activations = []
        
        for layer_id in self.layers:
            # ═══════════════════════════════════════════
            # DECODE: Only current layer, into reusable buffer
            # ═══════════════════════════════════════════
            weights = entropy_decode(
                self.compressed_weights[layer_id],
                out_buffer=self.decode_buffer  # Reused!
            )
            
            # Compute (same as traditional)
            x = layer(x, weights)
            
            # ═══════════════════════════════════════════
            # RE-COMPRESS: Store activation compressed
            # ═══════════════════════════════════════════
            x_quantized = quantize(x)
            x_compressed = entropy_encode(x_quantized)  # ~0.5 bytes/value
            compressed_activations.append(x_compressed)
            
            # Decompressed weight automatically freed (buffer reused)
            # Peak memory: MBs not GBs!
            
        return x, compressed_activations
        
    def backward(self, compressed_activations):
        # Decode activations on-the-fly as needed
        for i, compressed_act in enumerate(reversed(compressed_activations)):
            # Decode only what's needed now
            act = entropy_decode(compressed_act)  # Temporary
            
            # Decode weights only what's needed now  
            weights = entropy_decode(
                self.compressed_weights[i],
                out_buffer=self.decode_buffer
            )
            
            grad = compute_grad(act, weights)
            
            # Free decompressed data immediately
            del act, weights  # Reclaim memory
```

**Key Innovation:** Data stays compressed until the exact moment it's needed for computation, then is immediately freed or re-compressed. Peak memory is minimized.

### Pillar 3: Bandwidth-Compute Trade-off

**The central thesis:**
```
Traditional thinking: "Decompression costs CPU cycles → avoid it"
Entropy-aware thinking: "Memory bandwidth costs energy → pay decode cost to save bandwidth"
```

**Energy analysis:**
```
Traditional execution (per layer):
  Load weights:        800 MB × 100 pJ/byte = 80,000 pJ
  Compute (FP32):       1,000 pJ
  Store activations:  1000 MB × 100 pJ/byte = 100,000 pJ
  ────────────────────────────────────────────────────
  Total:                                    181,000 pJ

Entropy-aware execution (per layer):
  Load compressed:     100 MB × 100 pJ/byte = 10,000 pJ
  Decode compute:                             1,000 pJ
  Compute (INT8):                               300 pJ
  Store compressed:    150 MB × 100 pJ/byte = 15,000 pJ
  ────────────────────────────────────────────────────
  Total:                                     26,300 pJ
  
SAVINGS: 85% reduction in energy!
```

**The breakthrough:** Even though we added decode/encode overhead (~2,000 pJ), we save ~155,000 pJ from reduced memory bandwidth. The energy savings are so large that we can afford substantial decode cost.

**Key Innovation:** Explicitly trade cheap decode computation for expensive memory bandwidth movement.

---

## 🔬 Core Research Questions

### RQ1: Memory-Bandwidth Energy Break-Even Point

> **At what compression ratio does decode energy exceed memory bandwidth energy savings?**

**Hypothesis:** Memory bandwidth energy dominates so heavily that we can afford 10-20 decode operations per byte and still save net energy.

**Measurement:**
```python
for compression_ratio in [2, 4, 6, 8, 10]:
    # Measure
    memory_energy_saved = baseline_bandwidth × (1 - 1/compression_ratio) × energy_per_byte
    decode_energy_cost = measure_decode_energy(compression_ratio)
    
    net_savings = memory_energy_saved - decode_energy_cost
    
    # Find break-even
    if net_savings < 0:
        print(f"Break-even at {compression_ratio}×")
```

**Expected finding:** Break-even at 15-20× compression (way beyond what we need), confirming massive headroom for decode overhead.

**Novel contribution:** No paper systematically measures this trade-off across different:
- Compression algorithms (Huffman, ANS, learned codecs)
- Hardware platforms (GPU, CPU, edge devices)  
- Model architectures (CNNs, Transformers)
- Batch sizes and layer configurations

### RQ2: Entropy-Aware Architecture Design

> **Can we design networks that are inherently more compressible without sacrificing accuracy?**

**Traditional NAS:** Search for accuracy-optimal architectures.

**Entropy-aware NAS:** Search for accuracy-per-bit optimal architectures.

```python
# Multi-objective optimization
fitness = alpha × accuracy + beta × compression_ratio - gamma × decode_latency

search_space = {
    'activation': [ReLU, GELU, Swish],  # ReLU creates sparsity
    'width': [64, 128, 256],            # Wider but sparser?
    'depth': [12, 24, 48],               # Deeper with compression?
    'normalization': [BatchNorm, LayerNorm, None]
}
```

**Research questions:**
- Do certain activation functions (ReLU vs GELU) create more compressible representations?
- Is it better to have wider, sparser layers or narrower, denser layers?
- How does batch normalization affect activation entropy?

**Novel contribution:** First systematic study of architecture choices through entropy lens.

### RQ3: Training-Time Activation Compression 🔥

> **Can we compress activations during training to enable 4-8× larger batch sizes?**

**Current state:** Gradient checkpointing trades compute (recomputation) for memory.

**Our approach:** Trade decode overhead for memory by compressing activations.

```python
# Forward pass
for layer in model:
    activation = layer(input)
    
    # COMPRESS before storing (novel)
    activation_quantized = quantize(activation)  # 4× smaller
    activation_compressed = entropy_encode(activation_quantized)  # 8× smaller total
    
    store_for_backward(activation_compressed)

# Backward pass  
for layer in reversed(model):
    # DECOMPRESS on-the-fly (novel)
    activation_compressed = retrieve_from_forward(layer)
    activation = entropy_decode(activation_compressed)  # Temporary
    
    gradient = compute_gradient(activation)
    
    del activation  # Free immediately
```

**Comparison:**

| Approach | Memory Reduction | Compute Overhead | Accuracy Impact |
|----------|-----------------|------------------|-----------------|
| No compression | 1× (baseline) | 1× | None |
| Gradient checkpointing | ~2× | 2× (recompute) | None |
| Quantization (ActNN) | 4× | ~1.1× | <1% loss |
| **This work: Quantize + entropy** | **8-10×** | **~1.3×** | **<1% loss (hypothesis)** |

**Novel contribution:** 
- Combining quantization + entropy coding for activations (underexplored)
- Comprehensive latency/memory/accuracy trade-off characterization
- Gradient flow analysis with compressed activations

**Potential impact:** Train ResNet-152 instead of ResNet-50 on same GPU, or 8× larger batch sizes.

---

## 📚 Related Work & Our Position

### What's Been Done (Storage Compression)

**Recent work focuses on compressing models for storage/deployment:**

1. **Efficient Neural Compression (Jun 2024)** - Uses tANS to decode during DRAM→cache transfer
   - Focus: Edge inference
   - Gap: Doesn't compress activations during training

2. **Rate-Constrained Quantization (May 2025)** - Achieves <1 bit/weight with OBS pruning + entropy
   - Focus: Minimal storage size
   - Gap: One-time decompression at load, not decode-on-fly execution

3. **MEC-Quant (Sept 2025)** - Optimizes quantization for maximum entropy reduction
   - Focus: Training-aware entropy
   - Gap: Still targets storage, not runtime memory

### The Paradigm Gap

| Dimension | Existing Work | This Work (Entropy-Aware) |
|-----------|--------------|---------------------------|
| **Problem framing** | "How small can we make the model file?" | "How little memory bandwidth can we use?" |
| **Optimization target** | Storage size | Runtime peak memory + energy |
| **Compression timing** | Post-training | Throughout: design → train → inference |
| **Decompression model** | Decompress once at load | Decode layer-by-layer during execution |
| **Activations** | Ignored (transient) | Compressed during training (novel!) |
| **Design philosophy** | Compress existing networks | Design networks for compressibility |
| **Metrics reported** | Compression ratio, accuracy | + Peak memory, decode latency, energy |

**The key difference:** We're not trying to make model files smaller. We're trying to make **execution** more efficient by keeping data compressed during computation.

---

## 💡 Why This Matters: Impact Pathways

### 1. Energy & Environmental Impact (Primary)

**The AI energy crisis:**
- Training GPT-3: ~1,287 MWh ≈ 500 tons CO₂
- Daily ChatGPT queries: ~500,000 kWh/day
- Memory bandwidth: 60-80% of total energy in inference

**Our contribution:**
```
If entropy-aware execution achieves:
- 60% memory bandwidth reduction
- 50% energy per inference

Impact on GPT-scale deployment:
- 500,000 kWh/day × 50% = 250,000 kWh saved daily
- ~91,000 MWh saved annually from one service
- Equivalent to ~40,000 tons CO₂ reduction
```

### 2. Democratization of ML Research

**Current barrier:**
```
Training large models requires:
- 4-8× A100 GPUs (~$80,000)
- 100+ GB VRAM
- Datacenter electricity

This excludes:
- Academic researchers
- Small companies
- Developing countries
- Independent researchers
```

**With 4× memory reduction:**
```
Same model trains on:
- 1-2× consumer GPUs (~$2,000-$4,000)
- 24-48 GB VRAM (RTX 4090)
- Home electricity

Opens ML research to orders of magnitude more people
```

### 3. Edge Deployment & Privacy

**Enable on-device inference:**
- 4GB model → 1GB with 4× compression → fits on phones
- Reduced bandwidth → longer battery life
- Local execution → privacy preserved
- No cloud costs → more accessible

### 4. Scientific Contribution

**Paradigm shift in ML systems:**
- Establishes entropy as first-class design principle
- Demonstrates memory bandwidth as primary optimization target
- Opens new research direction: entropy-aware architectures
- Provides empirical foundation for future work

---

## 🛠️ Implementation Roadmap

### Phase 1: Proof of Concept (Weeks 1-4)

**Goal:** Demonstrate entropy-aware execution works at small scale

**Tasks:**
1. Implement Huffman encoder/decoder for quantized weights
2. Create `EntropyAwareMLP` class with lazy decompression
3. Add profiling:
   - Peak memory: `torch.cuda.max_memory_allocated()`
   - Decode latency: `torch.utils.benchmark.Timer`
   - Memory bandwidth: Track read/write volumes
4. Run on MNIST with varying compression levels

**Deliverable:**
```json
{
  "baseline_fp32": {
    "peak_memory_mb": 45,
    "bandwidth_mb": 180,
    "time_ms": 12
  },
  "entropy_aware_8bit": {
    "peak_memory_mb": 22,
    "bandwidth_mb": 85,
    "time_ms": 16,
    "decode_overhead_ms": 4
  }
}
```

**Success criterion:** Peak memory <60% of baseline, time <2× baseline

### Phase 2: Energy Validation (Weeks 5-7)

**Goal:** Prove the energy savings hypothesis

**Tasks:**
1. Instrument code with energy profiling (PyTorch profiler + GPU counters)
2. Measure:
   - Memory accesses (GB transferred)
   - Decode compute (FLOPs)
   - Energy model: accesses × energy_per_access + compute × energy_per_flop
3. Calculate break-even point across compression ratios
4. Validate on different hardware (A100, V100, CPU, Jetson)

**Deliverable:** Energy vs compression ratio curves showing:
- Where decode energy equals bandwidth savings
- Net energy reduction at practical compression levels
- Hardware-specific break-even points

**Success criterion:** Net energy reduction ≥40% at 4× compression

### Phase 3: Multi-Architecture Validation (Weeks 8-14)

**Goal:** Show it generalizes beyond toy examples

**Tasks:**
1. Extend to CNN: ResNet-18 on CIFAR-10
2. Extend to Transformer: Small GPT on WikiText
3. Implement ANS encoder (better compression than Huffman)
4. Compare baselines:
   - Uncompressed
   - Quantization only
   - Gradient checkpointing
   - This work: Quantization + entropy

**Deliverable:** Comprehensive comparison table:

| Model | Method | Peak Memory | Bandwidth | Energy | Accuracy |
|-------|--------|------------|-----------|--------|----------|
| ResNet-18 | Baseline | 1.2 GB | 4.8 GB | 100% | 94.5% |
| ResNet-18 | Quant-only | 0.6 GB | 2.4 GB | 60% | 94.2% |
| ResNet-18 | This work | 0.3 GB | 1.2 GB | 35% | 94.0% |

**Success criterion:** Best peak memory AND best energy among all methods

### Phase 4: Training Compression (Weeks 15-22) 🔥 **High Risk/High Reward**

**Goal:** Novel contribution - compress activations during training

**Tasks:**
1. Implement activation compression in forward pass
2. Implement activation decompression in backward pass
3. Custom autograd integration
4. Numerical stability analysis
5. Convergence study: does it affect optimization?

**Deliverable:**
```python
# Proof of concept code
class EntropyAwareTraining:
    def train_step(self, batch):
        # Forward with compression
        compressed_acts = self.forward_compressed(batch)
        
        # Backward with decompression
        self.backward_decompressed(loss, compressed_acts)
        
        # Metrics
        return {
            'memory_saved': '4.2× vs baseline',
            'overhead': '18% slower',
            'convergence': 'Same final accuracy'
        }
```

**Success criterion:** 
- ≥4× activation memory reduction
- <2× training time overhead
- <2% accuracy degradation

### Phase 5: Publication (Weeks 23-28)

**Tasks:**
1. Write paper:
   - Introduction: Paradigm shift motivation
   - Related work: Position in landscape
   - Method: Entropy-aware framework design
   - Experiments: All above results
   - Discussion: When to use, limitations, future work
2. Create visualizations:
   - Memory vs time trade-off curves
   - Energy breakdown charts
   - Architecture comparison plots
3. Open-source release:
   - Clean, documented code
   - Examples and tutorials
   - Reproducibility artifacts
4. Submit to:
   - **First choice:** MLSys, NeurIPS (systems track)
   - **Backup:** ICLR workshop, CoNEXT

---

## 📊 Success Criteria

### Minimum Viable Publication

**Demonstrates the paradigm shift is real:**
- ✅ Implement entropy-aware execution for at least one architecture (MLP or small CNN)
- ✅ Show **measurable** peak memory reduction: ≥50% vs baseline
- ✅ Show **measurable** energy savings: ≥40% memory bandwidth reduction
- ✅ Demonstrate decode overhead is acceptable: <2× latency increase
- ✅ Maintain accuracy: ≥98% of baseline

**Empirical characterization:**
- ✅ Memory vs latency vs compression ratio curves
- ✅ Energy break-even analysis (when does decode cost exceed bandwidth savings?)
- ✅ Ablation study: quantization alone vs quantization + entropy

### Strong Publication (Top-tier venue)

**All of minimum viable, plus:**
- ✅ Multi-architecture validation (CNN, Transformer, RNN)
- ✅ Training-time activation compression working (proof of concept)
- ✅ Show it scales: 4× larger models or 4× larger batch sizes on same hardware
- ✅ Energy profiling on real hardware (not just estimates)
- ✅ Comparison to all baselines: uncompressed, quantization-only, gradient checkpointing
- ✅ Novel contribution: Learned entropy codecs OR entropy-aware architecture search

### Stretch Goals (Paradigm-defining work)

- ✅ End-to-end framework: PyTorch/JAX integration, easy to use
- ✅ Demonstrate 10× compression with <5% accuracy loss
- ✅ Training with 50% energy reduction (measured, not estimated)
- ✅ Hardware co-design: Prototype decode units on FPGA
- ✅ Open-source library that others adopt

---

## 📊 Expected Results & Predictions

### Quantitative Predictions

Based on literature and preliminary analysis:

**Memory:**
- Weights: 8× reduction (FP32 → INT4 + entropy)
- Activations: 6× reduction (FP32 → INT8 + entropy)
- Peak memory: 4-5× reduction overall
- Enabling: Train 4× larger models on same hardware

**Energy:**
- Memory bandwidth: 60% reduction
- Compute overhead: 20% increase (decode)
- Net energy: 50% reduction
- Break-even: 12-15× compression ratio

**Latency:**
- Decode overhead: 15-30% slower
- Acceptable for: Training, batch inference, edge devices
- Not suitable for: Real-time sub-ms inference

**Accuracy:**
- Negligible loss: <1% with 4-bit quantization + entropy
- Modest loss: 1-3% with 2-bit + entropy
- Training compression: <2% with proper tuning

### Qualitative Insights

**Architecture findings (predicted):**
- ReLU networks compress better than GELU (sparsity)
- Wider, sparser layers compress better than narrow, dense
- Skip connections don't hurt compression (can help via structure)

**Training dynamics:**
- Compressed activations act as noise → implicit regularization?
- May improve generalization slightly
- Convergence might be slower but reach similar final accuracy

**Hardware insights:**
- GPUs: Best suited (high bandwidth, can hide decode latency)
- CPUs: Moderate benefit (lower bandwidth premium)
- Edge: Huge benefit (bandwidth-starved, energy-critical)

---

## 🔑 Core Concepts (Quick Reference)

**Entropy-Aware Framework** = Treating compression as native execution format, not post-processing
- Co-design networks for accuracy AND compressibility
- Execute with data compressed, decode only when needed
- Optimize for accuracy-per-bit, not just accuracy

**Lazy Decompression** = Decode-on-fly execution model
- Keep data compressed until exact moment of use
- Immediately free or re-compress after use
- Reuse decode buffers across operations
- Minimize peak memory footprint

**Energy-Centric Design** = Optimize for memory bandwidth, not FLOPs
- Recognition that memory access >> computation cost
- Willingness to pay decode overhead for bandwidth savings
- Explicit energy modeling in design decisions

**First-Class Entropy Metric** = Entropy alongside accuracy in all design choices
- Architecture search: optimize accuracy/bit
- Layer design: budget entropy per layer
- Activation choice: consider compressibility
- Training: monitor entropy evolution

---

## 📝 Open Research Questions

### Theoretical

1. **Entropy Bounds:** What's the theoretical limit for neural network weight/activation entropy? Can we prove bounds?

2. **Information Theory:** Is there an information-theoretic framework for understanding accuracy vs entropy trade-offs?

3. **Optimization Landscape:** How does training on compressed activations change the loss landscape?

### Empirical

4. **Architecture Families:** Which architectures are most amenable to entropy-aware execution? CNNs vs Transformers vs RNNs?

5. **Scalability:** Does this approach scale to billion-parameter models? Or only effective for edge-sized models?

6. **Generalization:** Does compression during training improve or hurt generalization?

### Systems

7. **Hardware Acceleration:** What would custom entropy decode units look like? What throughput is needed?

8. **Software Integration:** How to make this easy to use? Drop-in replacement for PyTorch layers?

9. **Numerical Stability:** Are there precision issues with compressed gradients in very deep networks?

### Practical

10. **Energy Profiling:** Can we get cycle-accurate energy measurements on real hardware?

11. **Production Deployment:** What would it take to deploy this at scale (e.g., in production ML systems)?

12. **User Experience:** Would practitioners actually use this, or is the complexity too high?

---

## 🔗 References & Further Reading

### Foundational Concepts

- **Information Theory:** Cover & Thomas, "Elements of Information Theory" (2006)
- **Neural Compression:** Cheng et al., "Learned Image Compression" (2020)
- **Entropy Coding:** Duda, "Asymmetric Numeral Systems" (2013)

### Closest Related Work

1. **Efficient Neural Compression (2024)** [[paper]](https://arxiv.org/html/2406.06237)
   - Uses tANS for inference-time decoding
   - Focuses on edge deployment
   - Difference: We do training + multi-architecture + energy analysis

2. **Rate-Constrained Quantization (2025)** [[paper]](https://arxiv.org/html/2505.18758)
   - Achieves <1 bit/weight with pruning + entropy
   - Focuses on storage compression
   - Difference: We focus on runtime memory, not storage

3. **MEC-Quant (2025)** [[paper]](https://arxiv.org/html/2509.15514v1)
   - Training-aware entropy optimization
   - Focuses on weight quantization
   - Difference: We also compress activations during training

### Activation Compression (Training)

4. **ActNN (2021)** - Quantize activations to INT8 during training
5. **Gist (2018)** - Lossy activation compression with error compensation
6. **Gradient Checkpointing (2016)** - Trade recomputation for activation memory

**Our difference:** We add entropy coding on top of quantization for better compression, and systematically characterize energy trade-offs.

### Hardware & Energy

7. **Hardware Lottery (Hooker, 2020)** - How hardware shapes ML research
8. **Energy and Policy (Strubell et al., 2019)** - Environmental cost of NLP
9. **Efficient Deep Learning (MIT course)** - Covers quantization, pruning, NAS

---

**Next Steps:**
- See implementation details in `CLAUDE.md`
- Track progress in `experiments/` directory

---

*This document represents a living research direction. Expect updates as we learn more.*
