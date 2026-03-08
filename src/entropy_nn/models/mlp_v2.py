from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from torch.profiler import profile, ProfilerActivity
from entropy_nn.models.interaction_energy_mlp import InteractionEnergyMLP
from entropy_nn.models.stat_affine_mlp import StatAffineMLP

MNIST_DIMS = [28 * 28, 256, 128, 10]


@torch.no_grad()
def profile_one_pass(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device, non_blocking=True)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total" if device=="cuda" else "self_cpu_time_total",
                                    row_limit=15))


@torch.no_grad()
def benchmark_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_warmup_batches: int = 10,
    num_timed_batches: int = 100,
) -> dict[str, float]:
    model.eval()

    n_batches = len(loader)
    warmup = min(num_warmup_batches, n_batches)
    timed = min(num_timed_batches, max(0, n_batches - warmup))
    if timed == 0:
        raise ValueError(f"Not enough batches in loader (len={n_batches}) for timing.")

    it = iter(loader)

    # Warm-up
    for _ in range(warmup):
        x, _ = next(it)
        x = x.to(device, non_blocking=True)
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_ms = []
        for _ in range(timed):
            x, _ = next(it)
            x = x.to(device, non_blocking=True)

            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))  # milliseconds
        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
    else:
        times_ms = []
        for _ in range(timed):
            x, _ = next(it)
            x = x.to(device)
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1e3)
        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))

    batch_size = loader.batch_size or 1
    return {
        "mean_batch_ms": mean_ms,
        "std_batch_ms": std_ms,
        "mean_example_ms": mean_ms / batch_size,
    }

@torch.no_grad()
def benchmark_gpu_memory(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_warmup_batches: int = 10,
    num_measured_batches: int = 50,
) -> dict:
    if device != "cuda":
        return {}

    model.eval()
    n_batches = len(loader)
    warmup = min(num_warmup_batches, n_batches)
    timed = min(num_measured_batches, max(0, n_batches - warmup))
    if timed == 0:
        raise ValueError(f"Not enough batches in loader (len={n_batches}) for timing.")
    it = iter(loader)

    # Warm-up (to stabilize allocator)
    for _ in range(warmup):
        x, _ = next(it)
        x = x.to(device, non_blocking=True)
        _ = model(x)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Measure a fixed number of batches
    for _ in range(timed):
        x, _ = next(it)
        x = x.to(device, non_blocking=True)
        _ = model(x)

    torch.cuda.synchronize()

    return {
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        "allocated_mb_end": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb_end": torch.cuda.memory_reserved() / (1024**2),
    }

def run_benchmarks(name: str, model: nn.Module, test_loader: DataLoader, cfg: Config) -> None:
    model = model.to(cfg.device)

    t = benchmark_inference_time(model, test_loader, cfg.device)
    print(f"{name} inference time: {t}")

    if cfg.device == "cuda":
        m = benchmark_gpu_memory(model, test_loader, cfg.device)
        print(f"{name} GPU memory: {m}")



class MLP(nn.Module):
    def __init__(self, activation: nn.Module) -> None:
        super().__init__()

        self.fcs = nn.ModuleList([
            nn.Linear(28 * 28, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10),
        ])
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)

        for i, fc in enumerate(self.fcs[:-1]):
            x = self.activation(fc(x))

        x = self.fcs[-1](x)
        return x


class CompactMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fcs = nn.ModuleList([
            nn.Linear(28 * 28, 256),
            nn.Linear(128, 128),
            nn.Linear(64, 10),
        ])
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)

        for i, fc in enumerate(self.fcs[:-1]):
            x = self.activation(fc(x))
            # remove the 50% smallest activations
            k = x.shape[1] // 2
            x = torch.topk(x, k, dim=1, sorted=True).values

        x = self.fcs[-1](x)
        return x

# ----------------------------
# Training / Eval
# ----------------------------

@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (optional; can slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.numel()

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train_one_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, cfg: Config) -> None:
    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * y.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.numel()

            pbar.set_postfix(loss=running_loss / total, acc=correct / total)

        test_loss, test_acc = evaluate(model, test_loader, cfg.device)
        print(f"  test: loss={test_loss:.4f} acc={test_acc:.4f}")


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda")
    )

    print("Training baseline MLP")
    baseline = MLP(activation=nn.ReLU())
    train_one_model(baseline, train_loader, test_loader, cfg)
    run_benchmarks("Baseline", baseline, test_loader, cfg)
    # profile_one_pass(baseline.to(cfg.device), test_loader, cfg.device)

    print("\nTraining GELU MLP")
    gelu_mlp = MLP(activation=nn.GELU())
    train_one_model(gelu_mlp, train_loader, test_loader, cfg)
    run_benchmarks("GELU", gelu_mlp, test_loader, cfg)

    print("\nTraining Swish MLP")
    swish_mlp = MLP(activation=nn.SiLU())
    train_one_model(swish_mlp, train_loader, test_loader, cfg)
    run_benchmarks("Swish", swish_mlp, test_loader, cfg)

    print("\nTraining Mish MLP")
    mish_mlp = MLP(activation=nn.Mish())
    train_one_model(mish_mlp, train_loader, test_loader, cfg)
    run_benchmarks("Mish", mish_mlp, test_loader, cfg)

    # print("\nTraining CompactMLP (top-k values)")
    # compact = CompactMLP()
    # train_one_model(compact, train_loader, test_loader, cfg)
    # run_benchmarks("Compact", compact, test_loader, cfg)
    # profile_one_pass(compact.to(cfg.device), test_loader, cfg.device)

if __name__ == "__main__":
    main()
