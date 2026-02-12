from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from entropy_nn.compression.quantize import uniform_quantize_symmetric


def tensor_entropy_from_ints(x_int: torch.Tensor, num_symbols: int) -> float:
    """
    x_int: integer tensor with values in [0, num_symbols-1]
    Returns entropy in bits/symbol.
    """
    counts = torch.bincount(x_int.flatten().cpu(), minlength=num_symbols).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum().item())


@dataclass
class LayerEntropyStats:
    """Entropy statistics for a single layer."""

    name: str
    entropy_bits: float
    compression_ratio: float
    scale: float
    num_params: int


@dataclass
class EntropyAnalysisResults:
    """Complete entropy analysis results."""

    bits: int
    per_layer: list[LayerEntropyStats]
    mean_entropy_bits: float
    mean_compression_ratio: float
    total_params: int


@torch.no_grad()
def measure_weight_entropy(model: nn.Module, bits: int = 8) -> EntropyAnalysisResults:
    """
    Measure entropy of quantized weights.

    Args:
        model: neural network model
        bits: quantization bitwidth

    Returns:
        EntropyAnalysisResults with per-layer and aggregate statistics
    """
    total_elems = 0
    total_bits_lb = 0.0
    layer_stats = []

    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue

        q, scale, qmax = uniform_quantize_symmetric(p.data, bits=bits)
        q_sym = (q + qmax).to(torch.int64)
        H = tensor_entropy_from_ints(q_sym, num_symbols=(2 * qmax + 1))

        n = p.numel()
        total_elems += n
        total_bits_lb += H * n

        ratio = (bits / H) if H > 0 else float("inf")
        layer_stats.append(
            LayerEntropyStats(
                name=name,
                entropy_bits=H,
                compression_ratio=ratio,
                scale=scale,
                num_params=n,
            )
        )

    mean_entropy = total_bits_lb / total_elems if total_elems > 0 else 0.0
    mean_ratio = (bits / mean_entropy) if mean_entropy > 0 else float("inf")

    return EntropyAnalysisResults(
        bits=bits,
        per_layer=layer_stats,
        mean_entropy_bits=mean_entropy,
        mean_compression_ratio=mean_ratio,
        total_params=total_elems,
    )


def print_entropy_analysis(results: EntropyAnalysisResults) -> None:
    """Pretty-print entropy analysis results."""
    print(f"\n[Entropy Analysis] {results.bits}-bit symmetric quantization:")

    for stats in results.per_layer:
        print(
            f"{stats.name:12s} | H={stats.entropy_bits:6.3f} bits/sym | "
            f"best_ratio={stats.compression_ratio:5.2f}x | "
            f"scale={stats.scale:.6g} | n={stats.num_params}"
        )

    print(
        f"{'TOTAL':12s} | H={results.mean_entropy_bits:6.3f} bits/sym | "
        f"best_ratio={results.mean_compression_ratio:5.2f}x"
    )
