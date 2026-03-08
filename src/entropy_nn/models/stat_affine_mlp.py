import torch
import torch.nn as nn

class StatAffine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Linear(2, dim)
        self.beta = nn.Linear(2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5

        stats = torch.cat([mu, std], dim=1)
        g = self.gamma(stats)
        b = self.beta(stats)

        return g * x + b
    

class StatAffineMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fcs = nn.ModuleList([
            nn.Linear(28 * 28, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10),
        ])

        # Interaction-energy activations per hidden layer dimension
        self.activations = nn.ModuleList([
            StatAffine(dim=256),
            StatAffine(dim=128),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)

        for fc, act in zip(self.fcs[:-1], self.activations):
            x = fc(x)
            x = act(x)

        x = self.fcs[-1](x)
        return x
