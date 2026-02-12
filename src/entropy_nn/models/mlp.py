from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        *,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
        flatten_input: bool = False,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must have at least input and output dimension")

        self.flatten_input = flatten_input
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.fcs = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_input:
            x = x.flatten(1)

        for fc in self.fcs[:-1]:
            x = self.activation(fc(x))
            if self.dropout is not None:
                x = self.dropout(x)

        x = self.fcs[-1](x)

        return x
