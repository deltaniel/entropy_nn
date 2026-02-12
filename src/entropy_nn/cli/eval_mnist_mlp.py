from __future__ import annotations

import argparse
import zlib
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from entropy_nn.data.mnist import make_loaders
from entropy_nn.losses.entropy import (
    tensor_entropy_from_ints,
    uniform_quantize_symmetric,
    report_weight_entropy,
)
from entropy_nn.models.mlp import MLP
from entropy_nn.training.loops import evaluate

MNIST_DIMS = [28 * 28, 256, 128, 10]


def _load_checkpoint(ckpt_path: Path, map_location: str | torch.device = "cpu") -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt

    if isinstance(ckpt, dict):
        return {"model_state": ckpt}

    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")


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



@torch.no_grad()
def _weight_entropy_and_ratios(model: nn.Module, bits: int) -> None:
    """
    Prints per-layer entropy, theoretical ratio (bits/H), and totals.
    """
    print("\n[Derived] Entropy + theoretical best ratio (bits / entropy):")

    total_elems = 0
    total_bits_lb = 0.0

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
        print(
            f"{name:12s} | H={H:6.3f} bits/sym | best_ratio={ratio:5.2f}x | "
            f"scale={scale:.6g} | n={n}"
        )

    if total_elems > 0:
        H_total = total_bits_lb / total_elems
        ratio_total = (bits / H_total) if H_total > 0 else float("inf")
        print(f"{'TOTAL':12s} | H={H_total:6.3f} bits/sym | best_ratio={ratio_total:5.2f}x")


@torch.no_grad()
def _weight_zlib_compress(model: nn.Module, bits: int, level: int = 9) -> None:
    """
    Compresses quantized weights with zlib as a practical baseline.

    RAW bytes are now properly bit-packed (bits/value), so ratios are meaningful across bitwidths.
    """
    print("\n[Baseline] zlib compression on BIT-PACKED quantized weights (bytes):")

    total_raw = 0
    total_comp = 0

    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue

        sym, _, _ = quantized_symbols_for_packing(p.data, bits=bits)
        raw = pack_bits_u16(sym, bits=bits)
        comp = zlib.compress(raw, level=level)

        total_raw += len(raw)
        total_comp += len(comp)

        ratio = (len(raw) / len(comp)) if len(comp) > 0 else float("inf")
        print(
            f"{name:12s} | raw={len(raw)/1024:8.2f} KB | comp={len(comp)/1024:8.2f} KB | ratio={ratio:5.2f}x"
        )

    if total_raw > 0:
        ratio_total = total_raw / total_comp if total_comp > 0 else float("inf")
        print(
            f"{'TOTAL':12s} | raw={total_raw/1024:8.2f} KB | comp={total_comp/1024:8.2f} KB | ratio={ratio_total:5.2f}x"
        )
        print("Note: raw size is true bit-packed size; comp includes zlib header/overhead.")


@torch.no_grad()
def _activation_entropy_probe(
    model: MLP,
    test_loader,
    device: torch.device,
    bits: int,
) -> None:
    """
    Captures one batch, measures entropy of ReLU outputs after first two linear layers.
    Assumes MLP has a Sequential/ModuleList 'fcs' or individual layers; adapts to your model.
    """
    print("\n[Probe] Activation entropy on one batch (ReLU outputs):")

    x, _ = next(iter(test_loader))
    x = x.to(device, non_blocking=True)

    acts: dict[str, torch.Tensor] = {}

    def save_act(name: str):
        def _hook(_m, _inp, out):
            acts[name] = out.detach()
        return _hook

    hooks = []
    # Your printed names are fcs.0.weight etc, so assume fcs[0], fcs[1]
    if hasattr(model, "fcs"):
        hooks.append(model.fcs[0].register_forward_hook(save_act("fcs0_out")))
        hooks.append(model.fcs[1].register_forward_hook(save_act("fcs1_out")))
    else:
        # fallback for older style
        if hasattr(model, "fc1"):
            hooks.append(model.fc1.register_forward_hook(save_act("fc1_out")))
        if hasattr(model, "fc2"):
            hooks.append(model.fc2.register_forward_hook(save_act("fc2_out")))

    _ = model(x)

    for h in hooks:
        h.remove()

    if not acts:
        print("Could not capture activations (no recognized layers).")
        return

    for name, a in acts.items():
        a = F.relu(a)
        zero_frac = float((a == 0).float().mean().item())

        q, scale, qmax = uniform_quantize_symmetric(a, bits=bits)
        q_sym = (q + qmax).to(torch.int64)
        H = tensor_entropy_from_ints(q_sym, num_symbols=(2 * qmax + 1))

        print(
            f"{name:10s} | H={H:6.3f} bits/sym | zero_frac={zero_frac*100:5.1f}% | scale={scale:.6g} | shape={tuple(a.shape)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a checkpoint and report entropy/compression stats (MNIST MLP).")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/mnist_mlp/model_state.pt"))
    parser.add_argument("--bits", type=int, default=8, choices=[2, 3, 4, 5, 6, 7, 8, 16])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-eval", action="store_true", help="Skip MNIST evaluation; only report entropy.")
    parser.add_argument("--zlib", action="store_true", help="Also run zlib compression baseline on quantized weights.")
    parser.add_argument("--zlib-level", type=int, default=9, choices=list(range(1, 10)))
    parser.add_argument("--act-probe", action="store_true", help="Also probe activation entropy on one batch.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    ckpt = _load_checkpoint(args.ckpt, map_location="cpu")

    model = MLP(dims=MNIST_DIMS, flatten_input=True).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if "config" in ckpt:
        cfg = ckpt["config"]
        try:
            print("Saved config:", asdict(cfg))
        except Exception:
            print("Saved config:", cfg)

    print(f"\nWeight entropy report ({args.bits}-bit symmetric quantization):")
    report_weight_entropy(model, bits=args.bits)

    _weight_entropy_and_ratios(model, bits=args.bits)

    if args.zlib:
        _weight_zlib_compress(model, bits=args.bits, level=args.zlib_level)

    # Optional: sanity-check accuracy and support activation probing
    test_loader = None
    if (not args.no_eval) or args.act_probe:
        _, test_loader = make_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    if not args.no_eval and test_loader is not None:
        m = evaluate(model, test_loader, device)
        print(f"\nCheckpoint eval | [Test] loss {m.loss:.4f} acc {m.acc * 100:.2f}%")

    if args.act_probe and test_loader is not None:
        _activation_entropy_probe(model, test_loader, device, bits=args.bits)

    print("\nDone.")


if __name__ == "__main__":
    main()
