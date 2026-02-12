from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from entropy_nn.utils.metrics import Metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validation", leave=False)

    for x, y in pbar:
        x: torch.Tensor = x.to(device, non_blocking=True)
        y: torch.Tensor = y.to(device, non_blocking=True)

        logits: torch.Tensor = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

        avg_loss = total_loss / total
        avg_acc = correct / total
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc * 100:.2f}%")

    loss = total_loss / total
    acc = correct / total
    return Metrics(loss=loss, acc=acc)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> Metrics:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for x, y in pbar:
        x: torch.Tensor = x.to(device, non_blocking=True)
        y: torch.Tensor = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits: torch.Tensor = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

        avg_loss = total_loss / total
        avg_acc = correct / total

        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc * 100:.2f}%")

    loss = total_loss / total
    acc = correct / total
    return Metrics(loss=loss, acc=acc)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
) -> dict[str, list]:
    """
    Run the full training loop with per-epoch evaluation.

    Returns:
        Dictionary with per-epoch metrics:
        - epochs: list of epoch numbers
        - train_loss, train_acc: training metrics
        - test_loss, test_acc: test metrics
        - epoch_time_s: time per epoch
    """
    m = evaluate(model, test_loader, device)
    print(f"Init | [Test] loss {m.loss:.4f} acc {m.acc * 100:.2f}%")

    # Initialize result tracking
    results = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time_s": [],
    }

    outer_pbar = tqdm(range(1, num_epochs + 1), desc="Training")

    for epoch in outer_pbar:
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch, total_epochs=num_epochs)
        test_m = evaluate(model, test_loader, device)
        dt = time.time() - t0

        # Record results
        results["epochs"].append(epoch)
        results["train_loss"].append(train_m.loss)
        results["train_acc"].append(train_m.acc)
        results["test_loss"].append(test_m.loss)
        results["test_acc"].append(test_m.acc)
        results["epoch_time_s"].append(dt)

        tqdm.write(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"[Train] loss {train_m.loss:.4f} acc {train_m.acc * 100:.2f}% | "
            f"[Test] loss {test_m.loss:.4f} acc {test_m.acc * 100:.2f}% | "
            f"{dt:.1f}s"
        )

    return results
