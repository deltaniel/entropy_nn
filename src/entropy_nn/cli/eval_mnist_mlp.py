from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from entropy_nn.compression.baselines import print_zlib_compression, weight_zlib_compress
from entropy_nn.compression.entropy import (
    measure_weight_entropy,
    print_entropy_analysis,
    tensor_entropy_from_ints,
)
from entropy_nn.compression.quantize import uniform_quantize_symmetric
from entropy_nn.data.mnist import make_loaders
from entropy_nn.experiments.config import EvalConfig, ExperimentConfig
from entropy_nn.experiments.results import (
    ActivationProbeResults,
    CompressionResults,
    EvaluationResults,
    ExperimentResults,
    LayerEntropyStats,
)
from entropy_nn.models import MLP, MNIST_DIMS
from entropy_nn.training.loops import evaluate


def _load_checkpoint(ckpt_path: Path, map_location: str | torch.device = "cpu") -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt

    if isinstance(ckpt, dict):
        return {"model_state": ckpt}

    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")


@torch.no_grad()
def _activation_entropy_probe(
    model: MLP,
    test_loader,
    device: torch.device,
    bits: int,
) -> ActivationProbeResults:
    """
    Captures one batch, measures entropy of ReLU outputs after first two linear layers.
    """
    x, _ = next(iter(test_loader))
    x = x.to(device, non_blocking=True)

    acts: dict[str, torch.Tensor] = {}

    def save_act(name: str):
        def _hook(_m, _inp, out):
            acts[name] = out.detach()

        return _hook

    hooks = []
    if hasattr(model, "fcs"):
        hooks.append(model.fcs[0].register_forward_hook(save_act("fcs0_out")))
        hooks.append(model.fcs[1].register_forward_hook(save_act("fcs1_out")))
    else:
        if hasattr(model, "fc1"):
            hooks.append(model.fc1.register_forward_hook(save_act("fc1_out")))
        if hasattr(model, "fc2"):
            hooks.append(model.fc2.register_forward_hook(save_act("fc2_out")))

    _ = model(x)

    for h in hooks:
        h.remove()

    layer_results = []
    for name, a in acts.items():
        a = F.relu(a)
        zero_frac = float((a == 0).float().mean().item())

        q, scale, qmax = uniform_quantize_symmetric(a, bits=bits)
        q_sym = (q + qmax).to(torch.int64)
        H = tensor_entropy_from_ints(q_sym, num_symbols=(2 * qmax + 1))

        layer_results.append(
            {
                "name": name,
                "entropy_bits": H,
                "zero_fraction": zero_frac,
                "scale": scale,
                "shape": list(a.shape),
            }
        )

    return ActivationProbeResults(bits=bits, layers=layer_results)


def print_activation_probe(results: ActivationProbeResults) -> None:
    """Pretty-print activation probe results."""
    print(f"\n[Probe] Activation entropy on one batch (ReLU outputs, {results.bits}-bit quant):")
    for layer in results.layers:
        print(
            f"{layer['name']:10s} | H={layer['entropy_bits']:6.3f} bits/sym | "
            f"zero_frac={layer['zero_fraction'] * 100:5.1f}% | "
            f"scale={layer['scale']:.6g} | shape={tuple(layer['shape'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MNIST MLP checkpoint with experiment tracking.")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    parser.add_argument("--name", type=str, help="Experiment name (overrides config)")

    # Eval args (override config)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path")
    parser.add_argument("--bits", type=int, choices=[2, 3, 4, 5, 6, 7, 8, 16], help="Quantization bits")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu/auto)")
    parser.add_argument("--data-root", type=Path, help="Data root directory")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-workers", type=int, help="Number of workers")
    parser.add_argument("--skip-eval", action="store_true", help="Skip accuracy evaluation")
    parser.add_argument("--zlib", action="store_true", help="Run zlib compression baseline")
    parser.add_argument("--zlib-level", type=int, choices=list(range(1, 10)), help="Zlib compression level")
    parser.add_argument("--act-probe", action="store_true", help="Probe activation entropy")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = ExperimentConfig(
            name="mnist_mlp_eval",
            eval=EvalConfig(),
        )

    # Override config with command-line args
    if args.name:
        config.name = args.name
    if args.output_dir:
        config.output_dir = args.output_dir

    # Ensure eval config exists
    if config.eval is None:
        config.eval = EvalConfig()

    eval_cfg = config.eval

    # Override eval config with command-line args
    if args.checkpoint is not None:
        eval_cfg.checkpoint = args.checkpoint
    if args.bits is not None:
        eval_cfg.bits = args.bits
    if args.device is not None:
        eval_cfg.device = args.device
    if args.data_root is not None:
        eval_cfg.data_root = args.data_root
    if args.batch_size is not None:
        eval_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        eval_cfg.num_workers = args.num_workers
    if args.skip_eval:
        eval_cfg.skip_eval = True
    if args.zlib:
        eval_cfg.run_zlib = True
    if args.zlib_level is not None:
        eval_cfg.zlib_level = args.zlib_level
    if args.act_probe:
        eval_cfg.run_activation_probe = True

    # Set device
    if eval_cfg.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = eval_cfg.device
    device = torch.device(device_str)

    print(f"\n=== Evaluation: {config.name} ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {eval_cfg.checkpoint}")
    print(f"Quantization: {eval_cfg.bits} bits")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = _load_checkpoint(eval_cfg.checkpoint, map_location="cpu")

    # Create and load model
    model = MLP(dims=MNIST_DIMS, flatten_input=True).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if "config" in ckpt:
        print("Checkpoint contains saved config")

    # Measure weight entropy
    entropy_results = measure_weight_entropy(model, bits=eval_cfg.bits)
    print_entropy_analysis(entropy_results)

    # Build compression results
    compression = CompressionResults(
        bits=eval_cfg.bits,
        per_layer_entropy=[
            LayerEntropyStats(
                name=stats.name,
                entropy_bits=stats.entropy_bits,
                compression_ratio=stats.compression_ratio,
                scale=stats.scale,
                num_params=stats.num_params,
            )
            for stats in entropy_results.per_layer
        ],
        mean_entropy_bits=entropy_results.mean_entropy_bits,
        mean_compression_ratio=entropy_results.mean_compression_ratio,
    )

    # Zlib baseline
    if eval_cfg.run_zlib:
        zlib_results = weight_zlib_compress(model, bits=eval_cfg.bits, level=eval_cfg.zlib_level)
        print_zlib_compression(zlib_results)

        compression.zlib_enabled = True
        compression.zlib_level = zlib_results.level
        compression.total_raw_bytes = zlib_results.total_raw_bytes
        compression.total_compressed_bytes = zlib_results.total_compressed_bytes
        compression.zlib_compression_ratio = zlib_results.compression_ratio

    # Initialize evaluation results
    eval_results = EvaluationResults(compression=compression)

    # Model evaluation
    test_loader = None
    if (not eval_cfg.skip_eval) or eval_cfg.run_activation_probe:
        _, test_loader = make_loaders(
            data_root=eval_cfg.data_root,
            batch_size=eval_cfg.batch_size,
            num_workers=eval_cfg.num_workers,
        )

    if not eval_cfg.skip_eval and test_loader is not None:
        m = evaluate(model, test_loader, device)
        eval_results.test_loss = m.loss
        eval_results.test_acc = m.acc
        print(f"\n[Checkpoint Evaluation] | Test loss {m.loss:.4f} | Test acc {m.acc * 100:.2f}%")

    # Activation probe
    if eval_cfg.run_activation_probe and test_loader is not None:
        act_results = _activation_entropy_probe(model, test_loader, device, bits=eval_cfg.bits)
        print_activation_probe(act_results)
        eval_results.activation_probe = act_results

    # Compile and save results
    experiment_results = ExperimentResults(
        name=config.name,
        config=config.to_dict(),
        evaluation=eval_results,
    )

    results_path = config.output_dir / f"{config.name}_results.json"
    experiment_results.to_json(results_path)
    print(f"\nSaved results: {results_path}")

    # Save config
    config_path = config.output_dir / f"{config.name}_config.yaml"
    config.to_yaml(config_path)
    print(f"Saved config: {config_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
