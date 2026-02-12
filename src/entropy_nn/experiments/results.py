from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TrainingResults:
    """Results from a training run."""

    # Per-epoch metrics
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)
    epoch_time_s: list[float] = field(default_factory=list)

    # Final metrics
    final_train_loss: float = 0.0
    final_train_acc: float = 0.0
    final_test_loss: float = 0.0
    final_test_acc: float = 0.0

    # Training metadata
    total_time_s: float = 0.0
    checkpoint_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingResults:
        """Load from dictionary."""
        return cls(**d)


@dataclass
class LayerEntropyStats:
    """Entropy statistics for a single layer."""

    name: str
    entropy_bits: float
    compression_ratio: float
    scale: float
    num_params: int


@dataclass
class CompressionResults:
    """Compression analysis results."""

    # Entropy-based analysis
    bits: int
    per_layer_entropy: list[LayerEntropyStats] = field(default_factory=list)
    mean_entropy_bits: float = 0.0
    mean_compression_ratio: float = 0.0

    # Zlib baseline (optional)
    zlib_enabled: bool = False
    zlib_level: int = 9
    total_raw_bytes: int = 0
    total_compressed_bytes: int = 0
    zlib_compression_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompressionResults:
        """Load from dictionary."""
        d = d.copy()
        if "per_layer_entropy" in d:
            d["per_layer_entropy"] = [LayerEntropyStats(**layer) for layer in d["per_layer_entropy"]]
        return cls(**d)


@dataclass
class ActivationProbeResults:
    """Activation entropy probe results."""

    bits: int
    layers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActivationProbeResults:
        """Load from dictionary."""
        return cls(**d)


@dataclass
class EvaluationResults:
    """Results from an evaluation run."""

    # Model performance
    test_loss: float = 0.0
    test_acc: float = 0.0

    # Compression analysis
    compression: CompressionResults | None = None

    # Activation probe (optional)
    activation_probe: ActivationProbeResults | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
        }
        if self.compression is not None:
            d["compression"] = self.compression.to_dict()
        if self.activation_probe is not None:
            d["activation_probe"] = self.activation_probe.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvaluationResults:
        """Load from dictionary."""
        d = d.copy()
        if "compression" in d and d["compression"] is not None:
            d["compression"] = CompressionResults.from_dict(d["compression"])
        if "activation_probe" in d and d["activation_probe"] is not None:
            d["activation_probe"] = ActivationProbeResults.from_dict(d["activation_probe"])
        return cls(**d)


@dataclass
class ExperimentResults:
    """Complete experiment results."""

    # Metadata
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    # Results
    training: TrainingResults | None = None
    evaluation: EvaluationResults | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "name": self.name,
            "timestamp": self.timestamp,
            "config": self.config,
        }
        if self.training is not None:
            d["training"] = self.training.to_dict()
        if self.evaluation is not None:
            d["evaluation"] = self.evaluation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentResults:
        """Load from dictionary."""
        d = d.copy()
        if "training" in d and d["training"] is not None:
            d["training"] = TrainingResults.from_dict(d["training"])
        if "evaluation" in d and d["evaluation"] is not None:
            d["evaluation"] = EvaluationResults.from_dict(d["evaluation"])
        return cls(**d)

    def to_json(self, path: Path | str, indent: int = 2) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_json(cls, path: Path | str) -> ExperimentResults:
        """Load results from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
