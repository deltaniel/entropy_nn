from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MemoryProfile:
    """Memory profiling results."""

    peak_memory_bytes: int
    peak_memory_mb: float
    peak_memory_allocated_bytes: int
    peak_memory_allocated_mb: float
    device: str


@dataclass
class LatencyProfile:
    """Latency profiling results."""

    total_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    num_iterations: int


@dataclass
class InferenceProfile:
    """Complete inference profiling results."""

    memory: MemoryProfile
    latency: LatencyProfile
    accuracy: float | None = None
    bandwidth_mb: float | None = None


def get_memory_stats(device: torch.device | str) -> dict[str, int]:
    """
    Get current memory statistics.

    Args:
        device: Device to profile

    Returns:
        Dictionary with memory stats
    """
    device = torch.device(device)

    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
        }
    else:
        # CPU memory tracking is less precise
        # We can use tracemalloc for more accurate CPU profiling if needed
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}


def reset_memory_stats(device: torch.device | str) -> None:
    """
    Reset memory statistics.

    Args:
        device: Device to reset
    """
    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    gc.collect()


@torch.no_grad()
def profile_inference_memory(
    model: nn.Module,
    input_batch: torch.Tensor,
    device: torch.device | str,
    warmup: int = 3,
) -> MemoryProfile:
    """
    Profile peak memory usage during inference.

    Args:
        model: Model to profile
        input_batch: Input batch tensor
        device: Device to run on
        warmup: Number of warmup iterations

    Returns:
        MemoryProfile with peak memory statistics
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    input_batch = input_batch.to(device)

    # Warmup iterations
    for _ in range(warmup):
        _ = model(input_batch)

    # Reset memory stats before measurement
    reset_memory_stats(device)

    # Profile memory
    _ = model(input_batch)

    # Get peak memory
    stats = get_memory_stats(device)

    if device.type == "cuda":
        peak_allocated = stats["max_allocated"]
        peak_reserved = stats["reserved"]
    else:
        # For CPU, estimate based on model size
        # This is a rough estimate - actual memory usage may vary
        peak_allocated = sum(p.numel() * p.element_size() for p in model.parameters())
        peak_reserved = peak_allocated

    return MemoryProfile(
        peak_memory_bytes=peak_reserved,
        peak_memory_mb=peak_reserved / (1024 * 1024),
        peak_memory_allocated_bytes=peak_allocated,
        peak_memory_allocated_mb=peak_allocated / (1024 * 1024),
        device=str(device),
    )


@torch.no_grad()
def profile_inference_latency(
    model: nn.Module,
    input_batch: torch.Tensor,
    device: torch.device | str,
    num_iterations: int = 100,
    warmup: int = 10,
) -> LatencyProfile:
    """
    Profile inference latency.

    Args:
        model: Model to profile
        input_batch: Input batch tensor
        device: Device to run on
        num_iterations: Number of timed iterations
        warmup: Number of warmup iterations

    Returns:
        LatencyProfile with timing statistics
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    input_batch = input_batch.to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(input_batch)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model(input_batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = variance**0.5

    return LatencyProfile(
        total_time_ms=sum(times),
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        num_iterations=num_iterations,
    )


@torch.no_grad()
def profile_inference_complete(
    model: nn.Module,
    input_batch: torch.Tensor,
    device: torch.device | str,
    num_iterations: int = 100,
    warmup: int = 10,
) -> InferenceProfile:
    """
    Complete inference profiling (memory + latency).

    Args:
        model: Model to profile
        input_batch: Input batch tensor
        device: Device to run on
        num_iterations: Number of latency iterations
        warmup: Number of warmup iterations

    Returns:
        InferenceProfile with all metrics
    """
    memory_profile = profile_inference_memory(model, input_batch, device, warmup=warmup)
    latency_profile = profile_inference_latency(model, input_batch, device, num_iterations, warmup)

    return InferenceProfile(memory=memory_profile, latency=latency_profile)


@torch.no_grad()
def profile_model_accuracy(model: nn.Module, test_loader, device: torch.device | str) -> float:
    """
    Profile model accuracy on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Accuracy as fraction [0, 1]
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return correct / total if total > 0 else 0.0


@dataclass
class ComparativeProfile:
    """Comparative profiling results between two models."""

    baseline: InferenceProfile
    compressed: InferenceProfile
    compression_ratio: float
    memory_reduction: float  # Fraction [0, 1]
    latency_overhead: float  # Fraction (>0 means slower)
    accuracy_delta: float | None = None


def profile_comparative(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    input_batch: torch.Tensor,
    device: torch.device | str,
    test_loader=None,
    num_iterations: int = 100,
) -> ComparativeProfile:
    """
    Compare baseline vs compressed model performance.

    Args:
        baseline_model: Original uncompressed model
        compressed_model: Compressed model
        input_batch: Input batch for profiling
        device: Device to run on
        test_loader: Optional test loader for accuracy comparison
        num_iterations: Number of latency iterations

    Returns:
        ComparativeProfile with comparison metrics
    """
    # Profile baseline
    baseline_profile = profile_inference_complete(baseline_model, input_batch, device, num_iterations)

    # Profile compressed
    compressed_profile = profile_inference_complete(compressed_model, input_batch, device, num_iterations)

    # Calculate metrics
    memory_reduction = (
        1.0
        - compressed_profile.memory.peak_memory_allocated_bytes / baseline_profile.memory.peak_memory_allocated_bytes
        if baseline_profile.memory.peak_memory_allocated_bytes > 0
        else 0.0
    )

    latency_overhead = (
        compressed_profile.latency.mean_time_ms / baseline_profile.latency.mean_time_ms - 1.0
        if baseline_profile.latency.mean_time_ms > 0
        else 0.0
    )

    # Estimate compression ratio for compressed model
    if hasattr(compressed_model, "get_compression_stats"):
        stats = compressed_model.get_compression_stats()
        compression_ratio = stats["compression_ratio"]
    else:
        compression_ratio = 1.0

    # Accuracy comparison if test loader provided
    accuracy_delta = None
    if test_loader is not None:
        baseline_acc = profile_model_accuracy(baseline_model, test_loader, device)
        compressed_acc = profile_model_accuracy(compressed_model, test_loader, device)
        baseline_profile.accuracy = baseline_acc
        compressed_profile.accuracy = compressed_acc
        accuracy_delta = compressed_acc - baseline_acc

    return ComparativeProfile(
        baseline=baseline_profile,
        compressed=compressed_profile,
        compression_ratio=compression_ratio,
        memory_reduction=memory_reduction,
        latency_overhead=latency_overhead,
        accuracy_delta=accuracy_delta,
    )


def print_memory_profile(profile: MemoryProfile, label: str = "Memory") -> None:
    """Pretty-print memory profile."""
    print(f"\n[{label}] Device: {profile.device}")
    print(f"  Peak allocated: {profile.peak_memory_allocated_mb:.2f} MB")
    print(f"  Peak reserved:  {profile.peak_memory_mb:.2f} MB")


def print_latency_profile(profile: LatencyProfile, label: str = "Latency") -> None:
    """Pretty-print latency profile."""
    print(f"\n[{label}] {profile.num_iterations} iterations")
    print(f"  Mean: {profile.mean_time_ms:.3f} ms")
    print(f"  Std:  {profile.std_time_ms:.3f} ms")
    print(f"  Total: {profile.total_time_ms:.1f} ms")


def print_inference_profile(profile: InferenceProfile, label: str = "Inference") -> None:
    """Pretty-print complete inference profile."""
    print(f"\n[{label} Profile]")
    print(f"  Memory (allocated): {profile.memory.peak_memory_allocated_mb:.2f} MB")
    print(f"  Latency (mean):     {profile.latency.mean_time_ms:.3f} ms")
    if profile.accuracy is not None:
        print(f"  Accuracy:           {profile.accuracy * 100:.2f}%")


def print_comparative_profile(profile: ComparativeProfile) -> None:
    """Pretty-print comparative profile."""
    print("\n" + "=" * 60)
    print("COMPARATIVE PROFILE: Baseline vs Compressed")
    print("=" * 60)

    print("\nBASELINE:")
    print_inference_profile(profile.baseline, "Baseline")

    print("\nCOMPRESSED:")
    print_inference_profile(profile.compressed, "Compressed")

    print("\n" + "-" * 60)
    print("COMPARISON METRICS:")
    print("-" * 60)
    print(f"  Compression ratio:  {profile.compression_ratio:.2f}x")
    print(f"  Memory reduction:   {profile.memory_reduction * 100:.1f}%")
    print(f"  Latency overhead:   {profile.latency_overhead * 100:+.1f}%")

    if profile.accuracy_delta is not None:
        print(f"  Accuracy delta:     {profile.accuracy_delta * 100:+.2f}%")

    # Success criteria from research doc
    print("\n" + "-" * 60)
    print("SUCCESS CRITERIA (Phase 1 - Proof of Concept):")
    print("-" * 60)
    memory_ok = profile.memory_reduction >= 0.40  # ≥40% reduction (target is 50%)
    latency_ok = profile.latency_overhead < 1.0  # <2× slower
    accuracy_ok = profile.accuracy_delta is None or profile.accuracy_delta >= -0.02  # ≥98% of baseline

    print(f"  {'[OK]' if memory_ok else '[FAIL]'} Peak memory <60% of baseline:     {memory_ok}")
    print(f"  {'[OK]' if latency_ok else '[FAIL]'} Time <2× baseline:                {latency_ok}")
    if profile.accuracy_delta is not None:
        print(f"  {'[OK]' if accuracy_ok else '[FAIL]'} Accuracy ≥98% of baseline:        {accuracy_ok}")

    all_ok = memory_ok and latency_ok and (accuracy_ok if profile.accuracy_delta is not None else True)
    print(f"\n  Overall: {'SUCCESS' if all_ok else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)


@dataclass
class MultiModelComparison:
    """Comparison results for multiple model variants."""

    model_name: str
    bits: int  # 0 for FP32 baseline
    coder_type: str | None  # "ans", "huffman", or None for baseline
    profile: InferenceProfile
    compression_stats: dict[str, float] | None = None


def print_multi_model_comparison(
    baseline_name: str,
    comparisons: list[MultiModelComparison],
) -> None:
    """
    Pretty-print comparison table for multiple model variants.

    Args:
        baseline_name: Name of the baseline model
        comparisons: List of comparison results (baseline should be first)
    """
    print("\n" + "=" * 120)
    print(f"MULTI-MODEL COMPARISON: {baseline_name}")
    print("=" * 120)

    # Header
    print(
        f"{'Model':<25} {'Bits':<6} {'Coder':<10} "
        f"{'Memory (MB)':<12} {'Latency (ms)':<14} {'Accuracy (%)':<12} {'Comp Ratio':<12}"
    )
    print("-" * 120)

    # Find baseline for relative metrics
    baseline = comparisons[0] if comparisons else None

    # Data rows
    for comp in comparisons:
        model_str = comp.model_name
        bits_str = str(comp.bits) if comp.bits > 0 else "FP32"
        coder_str = comp.coder_type.upper() if comp.coder_type else "-"
        mem_str = f"{comp.profile.memory.peak_memory_allocated_mb:.2f}"
        latency_str = f"{comp.profile.latency.mean_time_ms:.3f} ± {comp.profile.latency.std_time_ms:.3f}"
        acc_str = f"{comp.profile.accuracy * 100:.2f}" if comp.profile.accuracy is not None else "N/A"

        if comp.compression_stats and comp.compression_stats.get("compression_ratio", 0) > 0:
            comp_str = f"{comp.compression_stats['compression_ratio']:.2f}x"
        else:
            comp_str = "1.00x"

        print(f"{model_str:<25} {bits_str:<6} {coder_str:<10} {mem_str:<12} {latency_str:<14} {acc_str:<12} {comp_str:<12}")

    # Relative comparisons
    if baseline and len(comparisons) > 1:
        print("\n" + "-" * 120)
        print("RELATIVE TO BASELINE:")
        print("-" * 120)
        print(
            f"{'Model':<25} {'Bits':<6} {'Coder':<10} "
            f"{'Mem D (%)':<12} {'Latency D (%)':<16} {'Acc D (%)':<12} {'Status':<15}"
        )
        print("-" * 120)

        for comp in comparisons[1:]:
            model_str = comp.model_name
            bits_str = str(comp.bits)
            coder_str = comp.coder_type.upper() if comp.coder_type else "-"

            # Memory delta
            mem_delta = (
                (comp.profile.memory.peak_memory_allocated_mb / baseline.profile.memory.peak_memory_allocated_mb - 1.0)
                * 100
                if baseline.profile.memory.peak_memory_allocated_mb > 0
                else 0.0
            )
            mem_delta_str = f"{mem_delta:+.1f}"

            # Latency delta
            lat_delta = (
                (comp.profile.latency.mean_time_ms / baseline.profile.latency.mean_time_ms - 1.0) * 100
                if baseline.profile.latency.mean_time_ms > 0
                else 0.0
            )
            lat_delta_str = f"{lat_delta:+.1f}"

            # Accuracy delta
            if comp.profile.accuracy is not None and baseline.profile.accuracy is not None:
                acc_delta = (comp.profile.accuracy - baseline.profile.accuracy) * 100
                acc_delta_str = f"{acc_delta:+.2f}"
            else:
                acc_delta_str = "N/A"

            # Success check
            memory_ok = mem_delta <= -40  # ≥40% reduction
            latency_ok = lat_delta < 100  # <2× slower
            if comp.profile.accuracy is not None and baseline.profile.accuracy is not None:
                accuracy_ok = (comp.profile.accuracy / baseline.profile.accuracy) >= 0.98
            else:
                accuracy_ok = True

            status = "[PASS]" if (memory_ok and latency_ok and accuracy_ok) else "[FAIL]"

            print(
                f"{model_str:<25} {bits_str:<6} {coder_str:<10} "
                f"{mem_delta_str:<12} {lat_delta_str:<16} {acc_delta_str:<12} {status:<15}"
            )

    print("=" * 120)
