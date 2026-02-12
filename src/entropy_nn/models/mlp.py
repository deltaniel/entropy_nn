from __future__ import annotations

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_input:
            x = x.flatten(1)

        for fc in self.fcs[:-1]:
            x = self.activation(fc(x))
            if self.dropout is not None:
                x = self.dropout(x)

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

    Supports:
      - coder_kind="ans": ANS stack coder (fast, near-optimal) :contentReference[oaicite:2]{index=2}
      - coder_kind="huffman": Huffman symbol codebook (may be faster decode, but API depends on binding)
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
    # Lazy decode per layer
    # ----------------------------
    def decode_layer_weights(self, layer_idx: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor | None]:
        layer = self.compressed_layers[layer_idx]

        if str(self.weight_buffer.device) != device:
            self.weight_buffer = self.weight_buffer.to(device)
            self.bias_buffer = self.bias_buffer.to(device)

        weight_full = decode_tensor(layer.weight)  # CPU tensor
        weight_size = layer.out_features * layer.in_features
        self.weight_buffer[:weight_size] = weight_full.reshape(-1).to(device)
        weight_view = self.weight_buffer[:weight_size].reshape(layer.out_features, layer.in_features)

        bias_view = None
        if layer.bias is not None:
            bias_full = decode_tensor(layer.bias)
            self.bias_buffer[: layer.out_features] = bias_full.reshape(-1).to(device)
            bias_view = self.bias_buffer[: layer.out_features]

        return weight_view, bias_view

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_input:
            x = x.flatten(1)

        device = x.device
        for i in range(len(self.compressed_layers) - 1):
            weight, bias = self.decode_layer_weights(i, device=str(device))
            x = torch.nn.functional.linear(x, weight, bias)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)

        weight, bias = self.decode_layer_weights(len(self.compressed_layers) - 1, device=str(device))
        x = torch.nn.functional.linear(x, weight, bias)
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
