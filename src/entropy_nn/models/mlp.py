from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn

from entropy_nn.compression.coders import EncodedTensor, decode_tensor, encode_tensor
from entropy_nn.compression.quantize import uniform_quantize_symmetric

MNIST_DIMS = [28 * 28, 256, 128, 10]


class MLP(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        *,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
        flatten_input: bool = False,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must have at least input and output dimension")

        self.flatten_input = flatten_input
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.fcs = nn.ModuleList(layers)

        self.foo = 123

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_input:
            x = x.flatten(1)

        for fc in self.fcs[:-1]:
            x = self.activation(fc(x))
            if self.dropout is not None:
                x = self.dropout(x)

        self.foo

        x = self.fcs[-1](x)
        return x


@dataclass
class CompressedLayer:
    """Compressed weights and bias for a single layer."""

    weight: EncodedTensor
    bias: EncodedTensor | None
    in_features: int
    out_features: int


class EntropyAwareMLP(nn.Module):
    """
    MLP with entropy-aware execution: weights stored compressed, decoded layer-by-layer.

    Supports debug timing to identify bottlenecks. Enable with model.enable_debug_timing().
    """

    def __init__(
        self,
        compressed_layers: list[CompressedLayer],
        *,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
        flatten_input: bool = False,
    ) -> None:
        super().__init__()
        self.compressed_layers = compressed_layers
        self.flatten_input = flatten_input
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        max_weight_size = max(layer.in_features * layer.out_features for layer in compressed_layers)
        max_bias_size = max(layer.out_features for layer in compressed_layers)

        self.weight_buffer = torch.zeros(max_weight_size, dtype=torch.float32)
        self.bias_buffer = torch.zeros(max_bias_size, dtype=torch.float32)

        # Debug timing
        self._debug_timing = False
        self._timing_stats: dict[str, list[float]] = defaultdict(list)
        self._forward_count = 0

    # ----------------------------
    # Construction from trained MLP
    # ----------------------------
    @classmethod
    @torch.no_grad()
    def from_trained_mlp(
        cls,
        model: MLP,
        *,
        bits: int = 8,
        coder: Literal["ans", "huffman"] = "ans",
        activation: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> EntropyAwareMLP:
        """
        Create EntropyAwareMLP from a trained standard MLP.

        Args:
            model: trained MLP to compress
            bits: quantization bits
            coder: "ans" or "huffman"
            activation: activation fn (defaults to model's activation)
            dropout: dropout rate (defaults to 0.0 for inference)
        """

        compressed_layers: list[CompressedLayer] = []

        for fc in model.fcs:
            q_w, scale_w, qmax_w = uniform_quantize_symmetric(fc.weight.data, bits=bits)
            w_enc = encode_tensor(q_w, scale_w, qmax_w, coder_type=coder)

            b_enc = None
            if fc.bias is not None:
                q_b, scale_b, qmax_b = uniform_quantize_symmetric(fc.bias.data, bits=bits)
                b_enc = encode_tensor(q_b, scale_b, qmax_b, coder_type=coder)

            compressed_layers.append(
                CompressedLayer(
                    weight=w_enc,
                    bias=b_enc,
                    in_features=fc.in_features,
                    out_features=fc.out_features,
                )
            )

        return cls(
            compressed_layers,
            activation=activation if activation is not None else model.activation,
            dropout=dropout,
            flatten_input=model.flatten_input,
        )

    # ----------------------------
    # Debug timing
    # ----------------------------
    def enable_debug_timing(self) -> None:
        """Enable debug timing to profile bottlenecks."""
        self._debug_timing = True
        self._timing_stats.clear()
        self._forward_count = 0

    def disable_debug_timing(self) -> None:
        """Disable debug timing."""
        self._debug_timing = False

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        """
        Get timing statistics.

        Returns:
            Dictionary with timing info for each operation:
            - mean: average time in milliseconds
            - total: total time in milliseconds
            - count: number of times called
        """
        stats = {}
        for key, times in self._timing_stats.items():
            if times:
                stats[key] = {
                    "mean_ms": sum(times) / len(times) * 1000,
                    "total_ms": sum(times) * 1000,
                    "count": len(times),
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }
        return stats

    def print_timing_stats(self) -> None:
        """Pretty-print timing statistics."""
        stats = self.get_timing_stats()
        if not stats:
            print("No timing data collected. Call enable_debug_timing() first.")
            return

        print(f"\n{'='*80}")
        print(f"EntropyAwareMLP Timing Statistics ({self._forward_count} forward passes)")
        print(f"{'='*80}")
        print(f"{'Operation':<30} {'Mean (ms)':<12} {'Total (ms)':<12} {'Count':<8} {'% of Total':<12}")
        print(f"{'-'*80}")

        # Calculate total time
        total_time = sum(s["total_ms"] for s in stats.values())

        # Sort by total time descending
        for key in sorted(stats.keys(), key=lambda k: stats[k]["total_ms"], reverse=True):
            s = stats[key]
            pct = (s["total_ms"] / total_time * 100) if total_time > 0 else 0
            print(f"{key:<30} {s['mean_ms']:>11.3f} {s['total_ms']:>11.1f} {s['count']:>7} {pct:>11.1f}%")

        print(f"{'-'*80}")
        print(f"{'TOTAL':<30} {'':<12} {total_time:>11.1f}")
        print(f"{'='*80}\n")

    def reset_timing_stats(self) -> None:
        """Reset timing statistics."""
        self._timing_stats.clear()
        self._forward_count = 0

    # ----------------------------
    # Lazy decode per layer
    # ----------------------------
    def decode_layer_weights(self, layer_idx: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor | None]:
        layer = self.compressed_layers[layer_idx]

        # Buffer transfer timing
        if self._debug_timing:
            t0 = time.perf_counter()

        if str(self.weight_buffer.device) != device:
            self.weight_buffer = self.weight_buffer.to(device)
            self.bias_buffer = self.bias_buffer.to(device)

        if self._debug_timing:
            self._timing_stats[f"layer{layer_idx}_buffer_transfer"].append(time.perf_counter() - t0)

        # Weight decode timing
        if self._debug_timing:
            t0 = time.perf_counter()

        weight_full = decode_tensor(layer.weight)  # CPU tensor

        if self._debug_timing:
            self._timing_stats[f"layer{layer_idx}_weight_decode"].append(time.perf_counter() - t0)

        # Weight copy timing
        if self._debug_timing:
            t0 = time.perf_counter()

        weight_size = layer.out_features * layer.in_features
        self.weight_buffer[:weight_size] = weight_full.reshape(-1).to(device)
        weight_view = self.weight_buffer[:weight_size].reshape(layer.out_features, layer.in_features)

        if self._debug_timing:
            self._timing_stats[f"layer{layer_idx}_weight_copy"].append(time.perf_counter() - t0)

        # Bias decode and copy timing
        bias_view = None
        if layer.bias is not None:
            if self._debug_timing:
                t0 = time.perf_counter()

            bias_full = decode_tensor(layer.bias)

            if self._debug_timing:
                self._timing_stats[f"layer{layer_idx}_bias_decode"].append(time.perf_counter() - t0)
                t0 = time.perf_counter()

            self.bias_buffer[: layer.out_features] = bias_full.reshape(-1).to(device)
            bias_view = self.bias_buffer[: layer.out_features]

            if self._debug_timing:
                self._timing_stats[f"layer{layer_idx}_bias_copy"].append(time.perf_counter() - t0)

        return weight_view, bias_view

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._debug_timing:
            t_forward_start = time.perf_counter()
            self._forward_count += 1

        if self.flatten_input:
            x = x.flatten(1)

        device = x.device

        # Hidden layers
        for i in range(len(self.compressed_layers) - 1):
            # Decode timing (includes all decode operations)
            if self._debug_timing:
                t0 = time.perf_counter()

            weight, bias = self.decode_layer_weights(i, device=str(device))

            if self._debug_timing:
                self._timing_stats[f"layer{i}_total_decode"].append(time.perf_counter() - t0)

            # Linear + activation timing
            if self._debug_timing:
                t0 = time.perf_counter()

            x = torch.nn.functional.linear(x, weight, bias)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)

            if self._debug_timing:
                self._timing_stats[f"layer{i}_forward"].append(time.perf_counter() - t0)

        # Output layer
        last_idx = len(self.compressed_layers) - 1

        if self._debug_timing:
            t0 = time.perf_counter()

        weight, bias = self.decode_layer_weights(last_idx, device=str(device))

        if self._debug_timing:
            self._timing_stats[f"layer{last_idx}_total_decode"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()

        x = torch.nn.functional.linear(x, weight, bias)

        if self._debug_timing:
            self._timing_stats[f"layer{last_idx}_forward"].append(time.perf_counter() - t0)
            self._timing_stats["forward_total"].append(time.perf_counter() - t_forward_start)

        return x

    def get_compression_stats(self) -> dict[str, float]:
        total_params = 0

        total_encoded_bytes = 0
        for layer in self.compressed_layers:
            total_encoded_bytes += layer.weight.payload.nbytes
            total_params += layer.in_features * layer.out_features
            if layer.bias is not None:
                total_encoded_bytes += layer.bias.payload.nbytes
                total_params += layer.out_features

        original_bytes = total_params * 4
        return {
            "total_params": float(total_params),
            "original_bytes": float(original_bytes),
            "encoded_bytes": float(total_encoded_bytes),
            "compression_ratio": float(original_bytes / total_encoded_bytes) if total_encoded_bytes > 0 else 0.0,
            "original_mb": float(original_bytes / (1024 * 1024)),
            "encoded_mb": float(total_encoded_bytes / (1024 * 1024)),
        }
