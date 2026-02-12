from __future__ import annotations

import argparse
from pathlib import Path

import torch

from entropy_nn.data.mnist import make_loaders
from entropy_nn.models.mlp import MLP
from entropy_nn.training import train

CKPT_DIR = Path("checkpoints") / "mnist_mlp"
MNIST_DIMS = [28 * 28, 256, 128, 10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    print(f"Device: {device}")

    train_loader, test_loader = make_loaders(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = MLP(dims=MNIST_DIMS, flatten_input=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, test_loader, optimizer, device, num_epochs=args.num_epochs)

    # Save checkpoint
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / "model_state.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
