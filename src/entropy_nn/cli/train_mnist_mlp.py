from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from entropy_nn.data.mnist import make_loaders
from entropy_nn.experiments.config import ExperimentConfig, TrainConfig
from entropy_nn.experiments.results import ExperimentResults, TrainingResults
from entropy_nn.models import MLP
from entropy_nn.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST with experiment tracking.")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    parser.add_argument("--name", type=str, help="Experiment name (overrides config)")

    # Training args (override config)
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-workers", type=int, help="Number of data loading workers")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--data-root", type=Path, help="Data root directory")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu/auto)")
    parser.add_argument("--checkpoint-dir", type=Path, help="Checkpoint directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = ExperimentConfig(
            name="mnist_mlp_baseline",
            train=TrainConfig(),
        )

    # Override config with command-line args
    if args.name:
        config.name = args.name
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Ensure train config exists
    if config.train is None:
        config.train = TrainConfig()

    train_cfg = config.train

    # Override train config with command-line args
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.num_epochs is not None:
        train_cfg.num_epochs = args.num_epochs
    if args.data_root is not None:
        train_cfg.data_root = args.data_root
    if args.device is not None:
        train_cfg.device = args.device

    # Set device
    if train_cfg.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = train_cfg.device
    device = torch.device(device_str)

    # Set random seed
    torch.manual_seed(train_cfg.seed)
    torch.backends.cudnn.benchmark = True

    print(f"\n=== Experiment: {config.name} ===")
    print(f"Device: {device}")
    print(f"Config: lr={train_cfg.lr}, batch_size={train_cfg.batch_size}, epochs={train_cfg.num_epochs}")

    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = config.output_dir / f"{config.name}_config.yaml"
    config.to_yaml(config_path)
    print(f"Saved config: {config_path}")

    # Load data
    train_loader, test_loader = make_loaders(
        data_root=train_cfg.data_root,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    # Create model
    model = MLP(dims=train_cfg.dims, dropout=train_cfg.dropout, flatten_input=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    # Train
    t_start = time.time()
    epoch_results = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        device,
        num_epochs=train_cfg.num_epochs,
    )
    total_time = time.time() - t_start

    # Save checkpoint
    ckpt_path = config.checkpoint_dir / "model_state.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint: {ckpt_path}")

    # Compile results
    training_results = TrainingResults(
        epochs=epoch_results["epochs"],
        train_loss=epoch_results["train_loss"],
        train_acc=epoch_results["train_acc"],
        test_loss=epoch_results["test_loss"],
        test_acc=epoch_results["test_acc"],
        epoch_time_s=epoch_results["epoch_time_s"],
        final_train_loss=epoch_results["train_loss"][-1],
        final_train_acc=epoch_results["train_acc"][-1],
        final_test_loss=epoch_results["test_loss"][-1],
        final_test_acc=epoch_results["test_acc"][-1],
        total_time_s=total_time,
        checkpoint_path=str(ckpt_path),
    )

    experiment_results = ExperimentResults(
        name=config.name,
        config=config.to_dict(),
        training=training_results,
    )

    # Save results
    results_path = config.output_dir / f"{config.name}_results.json"
    experiment_results.to_json(results_path)
    print(f"Saved results: {results_path}")

    print("\n=== Training Complete ===")
    print(f"Final test accuracy: {training_results.final_test_acc * 100:.2f}%")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
