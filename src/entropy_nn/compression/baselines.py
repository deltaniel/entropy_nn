from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from entropy_nn.compression.quantize import (
    pack_bits_u16,
    quantized_symbols_for_packing,
)


@dataclass
class ZlibCompressionResults:
    """Results from zlib compression baseline."""

    bits: int
    level: int
    total_raw_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    per_layer: list[dict[str, str | int | float]]


@torch.no_grad()
def weight_zlib_compress(model: nn.Module, bits: int, level: int = 9) -> ZlibCompressionResults:
    """
    Compresses quantized weights with zlib as a practical baseline.

    RAW bytes are now properly bit-packed (bits/value), so ratios are meaningful across bitwidths.

    Returns:
        ZlibCompressionResults with per-layer and total statistics
    """
    total_raw = 0
    total_comp = 0
    per_layer_results = []

    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue

        sym, _, _ = quantized_symbols_for_packing(p.data, bits=bits)
        raw = pack_bits_u16(sym, bits=bits)
        comp = zlib.compress(raw, level=level)

        total_raw += len(raw)
        total_comp += len(comp)

        ratio = (len(raw) / len(comp)) if len(comp) > 0 else float("inf")
        per_layer_results.append(
            {
                "name": name,
                "raw_bytes": len(raw),
                "compressed_bytes": len(comp),
                "ratio": ratio,
            }
        )

    ratio_total = total_raw / total_comp if total_comp > 0 else float("inf")

    return ZlibCompressionResults(
        bits=bits,
        level=level,
        total_raw_bytes=total_raw,
        total_compressed_bytes=total_comp,
        compression_ratio=ratio_total,
        per_layer=per_layer_results,
    )


def print_zlib_compression(results: ZlibCompressionResults) -> None:
    """Pretty-print zlib compression results."""
    print(f"\n[Baseline] zlib compression (level={results.level}) on BIT-PACKED quantized weights:")

    for layer in results.per_layer:
        print(
            f"{layer['name']:12s} | "
            f"raw={layer['raw_bytes'] / 1024:8.2f} KB | "
            f"comp={layer['compressed_bytes'] / 1024:8.2f} KB | "
            f"ratio={layer['ratio']:5.2f}x"
        )

    print(
        f"{'TOTAL':12s} | "
        f"raw={results.total_raw_bytes / 1024:8.2f} KB | "
        f"comp={results.total_compressed_bytes / 1024:8.2f} KB | "
        f"ratio={results.compression_ratio:5.2f}x"
    )
    print("Note: raw size is true bit-packed size; comp includes zlib header/overhead.")
