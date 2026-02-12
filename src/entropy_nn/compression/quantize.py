from __future__ import annotations

import torch


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


def pack_bits_u16(values: torch.Tensor, bits: int) -> bytes:
    """
    Pack non-negative integer symbols into a compact byte stream using exactly `bits` bits/value.

    values: 1D tensor of dtype int64/int32/int16 on CPU, values must fit in [0, 2**bits - 1]
    bits: number of bits per value (1..16)

    Returns: packed bytes (little-endian within the bitstream).
    """
    if bits < 1 or bits > 16:
        raise ValueError(f"bits must be in [1,16], got {bits}")

    v = values.reshape(-1).to(torch.int64).cpu()
    if v.numel() == 0:
        return b""

    maxv = (1 << bits) - 1
    if v.min().item() < 0 or v.max().item() > maxv:
        raise ValueError(f"values out of range for bits={bits}: min={v.min().item()} max={v.max().item()}")

    out = bytearray()
    acc = 0
    acc_bits = 0

    for x in v.tolist():
        acc |= (x & maxv) << acc_bits
        acc_bits += bits

        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

    if acc_bits > 0:
        out.append(acc & 0xFF)

    return bytes(out)


def quantized_symbols_for_packing(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, int]:
    """
    Quantize tensor symmetrically to signed integers, then convert to unsigned symbols for packing.

    Returns:
      sym: int64 tensor in [0, 2**bits - 1] (safe for pack_bits_u16)
      scale: float
      qmax: int
    """
    q, scale, qmax = uniform_quantize_symmetric(x, bits=bits)
    # q in [-qmax, qmax] ; map to [0, 2*qmax]
    sym = (q + qmax).to(torch.int64)

    # For symmetric quantization, number of used symbols is (2*qmax+1), which fits in 2**bits.
    # For bits=16, qmax=32767 => 65535 symbols, still fine.
    return sym, scale, qmax
