from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    dims: list[int] = field(default_factory=lambda: [28 * 28, 256, 128, 10])
    dropout: float = 0.0

    # Optimization
    lr: float = 1e-3
    num_epochs: int = 10
    batch_size: int = 128

    # Data
    data_root: Path = field(default_factory=lambda: Path("data"))
    num_workers: int = 2

    # Hardware
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, handling Path objects."""
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """Load from dictionary."""
        d = d.copy()
        if "data_root" in d:
            d["data_root"] = Path(d["data_root"])
        return cls(**d)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Checkpoint
    checkpoint: Path = field(default_factory=lambda: Path("checkpoints/mnist_mlp/model_state.pt"))

    # Quantization analysis
    bits: int = 8

    # Data
    data_root: Path = field(default_factory=lambda: Path("data"))
    batch_size: int = 256
    num_workers: int = 2

    # Hardware
    device: str = "auto"

    # Analysis options
    skip_eval: bool = False
    run_zlib: bool = False
    zlib_level: int = 9
    run_activation_probe: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, handling Path objects."""
        d = asdict(self)
        d["checkpoint"] = str(self.checkpoint)
        d["data_root"] = str(self.data_root)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalConfig:
        """Load from dictionary."""
        d = d.copy()
        if "checkpoint" in d:
            d["checkpoint"] = Path(d["checkpoint"])
        if "data_root" in d:
            d["data_root"] = Path(d["data_root"])
        return cls(**d)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Experiment metadata
    name: str = "mnist_mlp_baseline"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/mnist_mlp"))

    # Training config (optional for eval-only experiments)
    train: TrainConfig | None = None

    # Evaluation config (optional for train-only experiments)
    eval: EvalConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
        }
        if self.train is not None:
            d["train"] = self.train.to_dict()
        if self.eval is not None:
            d["eval"] = self.eval.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Load from dictionary."""
        d = d.copy()
        if "output_dir" in d:
            d["output_dir"] = Path(d["output_dir"])
        if "checkpoint_dir" in d:
            d["checkpoint_dir"] = Path(d["checkpoint_dir"])
        if "train" in d and d["train"] is not None:
            d["train"] = TrainConfig.from_dict(d["train"])
        if "eval" in d and d["eval"] is not None:
            d["eval"] = EvalConfig.from_dict(d["eval"])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
