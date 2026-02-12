from __future__ import annotations


import torch
import torch.nn as nn


def tensor_entropy_from_ints(x_int: torch.Tensor, num_symbols: int) -> float:
    """
    x_int: integer tensor with values in [0, num_symbols-1]
    Returns entropy in bits/symbol.
    """
    counts = torch.bincount(x_int.flatten().cpu(), minlength=num_symbols).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum().item())


def uniform_quantize_symmetric(x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float, int]:
    """
    Symmetric per-tensor quantization: x = scale * q, q in [-qmax, qmax]
    Returns:
        q: quantized integer tensor
        scale: scaling factor
        qmax: maximum quantized integer
    """
    qmax = (2 ** (bits - 1)) - 1
    max_val = x.abs().max().item()
    scale = max_val / qmax if max_val > 0 else 1.0
    q = torch.clamp((x / scale).round(), -qmax, qmax).to(torch.int16)
    return q, scale, qmax


@torch.no_grad()
def report_weight_entropy(model: nn.Module, bits: int = 8) -> None:
    for name, p in model.named_parameters():
        if p.ndim == 2:
            q, scale, qmax = uniform_quantize_symmetric(p.data, bits=bits)
            q_sym = (q + qmax).to(torch.int64)
            H = tensor_entropy_from_ints(q_sym, num_symbols=(2 * qmax + 1))
            print(f"{name:10s} | {bits}-bit quant | entropy {H:.3f} bits/symbol | scale {scale:.6g}")
