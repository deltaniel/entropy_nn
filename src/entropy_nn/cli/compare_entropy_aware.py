from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch

from entropy_nn.data.mnist import make_loaders
from entropy_nn.experiments.profiling import (
    MultiModelComparison,
    print_multi_model_comparison,
    profile_inference_complete,
    profile_model_accuracy,
)
from entropy_nn.models import MLP, MNIST_DIMS
from entropy_nn.models.mlp import EntropyAwareMLP


def _load_checkpoint(ckpt_path: Path, map_location: str | torch.device = "cpu") -> dict:
    """Load checkpoint with backward compatibility."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt

    if isinstance(ckpt, dict):
        return {"model_state": ckpt}

    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline MLP vs EntropyAwareMLP with ANS/Huffman coders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained baseline MLP checkpoint")
    parser.add_argument(
        "--bits", type=int, default=8, choices=[2, 3, 4, 5, 6, 7, 8, 16], help="Quantization bits for entropy models"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Data root directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for profiling and accuracy eval")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"), help="Output directory for results")
    parser.add_argument("--name", type=str, default="entropy_comparison", help="Experiment name")
    parser.add_argument(
        "--coders",
        type=str,
        nargs="+",
        default=["ans", "huffman"],
        choices=["ans", "huffman"],
        help="Which entropy coders to test",
    )
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of latency profiling iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations for profiling")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip full accuracy evaluation (faster)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline profiling (only test entropy models)")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"EntropyAwareMLP Comparison: {args.name}")
    print(f"{'='*60}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Device:         {device}")
    print(f"Quantization:   {args.bits} bits")
    print(f"Coders:         {', '.join(args.coders)}")
    print(f"Profiling:      {args.num_iterations} iterations, {args.warmup} warmup")
    print(f"{'='*60}\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline checkpoint
    print("[1/5] Loading checkpoint...")
    ckpt = _load_checkpoint(args.checkpoint, map_location="cpu")
    baseline_model = MLP(dims=MNIST_DIMS, flatten_input=True)

    # Load state dict
    baseline_model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"  Loaded {sum(p.numel() for p in baseline_model.parameters())} parameters")

    # Load test data
    print("\n[2/5] Loading test data...")
    _, test_loader = make_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Test set: {len(test_loader.dataset)} samples")

    # Get a sample batch for profiling
    sample_batch, _ = next(iter(test_loader))
    print(f"  Profile batch size: {sample_batch.shape[0]}")

    # Results storage
    comparisons: list[MultiModelComparison] = []

    # Profile baseline model
    if not args.skip_baseline:
        print("\n[3/5] Profiling baseline MLP (FP32)...")
        baseline_model = baseline_model.to(device)

        baseline_profile = profile_inference_complete(
            baseline_model,
            sample_batch,
            device,
            num_iterations=args.num_iterations,
            warmup=args.warmup,
        )

        if not args.skip_accuracy:
            print("  Evaluating accuracy...")
            baseline_acc = profile_model_accuracy(baseline_model, test_loader, device)
            baseline_profile.accuracy = baseline_acc

        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        baseline_mb = (baseline_params * 4) / (1024 * 1024)  # FP32 = 4 bytes

        print(f"  Memory:   {baseline_profile.memory.peak_memory_allocated_mb:.2f} MB")
        print(f"  Latency:  {baseline_profile.latency.mean_time_ms:.3f} ± {baseline_profile.latency.std_time_ms:.3f} ms")
        if baseline_profile.accuracy is not None:
            print(f"  Accuracy: {baseline_profile.accuracy * 100:.2f}%")

        comparisons.append(
            MultiModelComparison(
                model_name="Baseline (FP32)",
                bits=0,
                coder_type=None,
                profile=baseline_profile,
                compression_stats={
                    "total_params": float(baseline_params),
                    "original_mb": baseline_mb,
                    "compression_ratio": 1.0,
                },
            )
        )

        # Move back to CPU to free GPU memory
        baseline_model = baseline_model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Profile EntropyAware models
    print(f"\n[4/5] Profiling EntropyAwareMLP models ({args.bits}-bit)...")

    for i, coder in enumerate(args.coders):
        coder_typed: Literal["ans", "huffman"] = coder  # type: ignore
        print(f"\n  [{i+1}/{len(args.coders)}] Testing {coder.upper()} coder...")

        # Create compressed model (do this on CPU first)
        print("    Creating compressed model...")
        entropy_model = EntropyAwareMLP.from_trained_mlp(
            baseline_model if args.skip_baseline else baseline_model.cpu(),
            bits=args.bits,
            coder=coder_typed,
            dropout=0.0,
        )

        # Get compression stats
        comp_stats = entropy_model.get_compression_stats()
        print(
            f"    Compression: {comp_stats['compression_ratio']:.2f}x "
            f"({comp_stats['original_mb']:.2f} MB -> {comp_stats['encoded_mb']:.2f} MB)"
        )

        # Move to device for profiling
        entropy_model = entropy_model.to(device)

        # Profile
        print("    Profiling...")
        entropy_profile = profile_inference_complete(
            entropy_model,
            sample_batch,
            device,
            num_iterations=args.num_iterations,
            warmup=args.warmup,
        )

        if not args.skip_accuracy:
            print("    Evaluating accuracy...")
            entropy_acc = profile_model_accuracy(entropy_model, test_loader, device)
            entropy_profile.accuracy = entropy_acc

        print(f"    Memory:   {entropy_profile.memory.peak_memory_allocated_mb:.2f} MB")
        print(f"    Latency:  {entropy_profile.latency.mean_time_ms:.3f} ± {entropy_profile.latency.std_time_ms:.3f} ms")
        if entropy_profile.accuracy is not None:
            print(f"    Accuracy: {entropy_profile.accuracy * 100:.2f}%")

        comparisons.append(
            MultiModelComparison(
                model_name=f"EntropyAware ({coder.upper()})",
                bits=args.bits,
                coder_type=coder,
                profile=entropy_profile,
                compression_stats=comp_stats,
            )
        )

        # Clean up
        entropy_model = entropy_model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print comparison table
    print("\n[5/5] Results:")
    print_multi_model_comparison(args.name, comparisons)

    # Save results to JSON
    results_dict = {
        "name": args.name,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "bits": args.bits,
        "coders": args.coders,
        "num_iterations": args.num_iterations,
        "warmup": args.warmup,
        "comparisons": [
            {
                "model_name": c.model_name,
                "bits": c.bits,
                "coder_type": c.coder_type,
                "memory_mb": c.profile.memory.peak_memory_allocated_mb,
                "latency_mean_ms": c.profile.latency.mean_time_ms,
                "latency_std_ms": c.profile.latency.std_time_ms,
                "accuracy": c.profile.accuracy,
                "compression_stats": c.compression_stats,
            }
            for c in comparisons
        ],
    }

    results_path = args.output_dir / f"{args.name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nSaved results: {results_path}")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
