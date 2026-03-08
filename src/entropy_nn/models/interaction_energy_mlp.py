import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionEnergy(nn.Module):
    """
    Interaction-energy activation over features within each sample.

    Given x in R^{B x D}, outputs:
        y_i = x_i - sum_j w_{ij} * phi(x_i - x_j)

    where:
      - w_{ij} are learned pairwise interaction weights (low-rank for efficiency)
      - phi is an odd, smooth function (tanh by default) so interactions depend on differences

    This is O(B * D^2) per layer if implemented naively; here we use a low-rank
    factorization to keep it O(B * D * r) memory and O(B * D^2) compute is avoided.

    NOTE: This is a *global* activation across the feature dimension, so it is
    not as cheap as ReLU. Keep dims modest or choose small rank.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        phi: str = "tanh",
        strength: float = 1.0,
        learn_strength: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.eps = eps

        # Low-rank parameterization: W ≈ A B^T, where A,B ∈ R^{D x r}
        self.A = nn.Parameter(torch.randn(dim, rank) * (1.0 / (dim**0.5)))
        self.B = nn.Parameter(torch.randn(dim, rank) * (1.0 / (dim**0.5)))

        if learn_strength:
            self.log_strength = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(strength + eps)))))
        else:
            self.register_buffer("log_strength", torch.tensor(float(torch.log(torch.tensor(strength + eps)))))
        self.phi = phi

    def _phi(self, d: torch.Tensor) -> torch.Tensor:
        # d has shape [B, D, D] (differences)
        if self.phi == "tanh":
            return torch.tanh(d)
        if self.phi == "softsign":
            return d / (1.0 + d.abs())
        if self.phi == "arctan":
            return torch.atan(d)
        if self.phi == "sign":  # non-smooth (subgrad); included for experimentation
            return d.sign()
        raise ValueError(f"Unknown phi='{self.phi}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        returns: [B, D]
        """
        B, D = x.shape
        if D != self.dim:
            raise ValueError(f"InteractionEnergy expected dim={self.dim}, got {D}")

        # Compute pairwise differences: d_{ij} = x_i - x_j  => shape [B, D, D]
        # This is the heaviest step; for D=256 it's usually manageable on GPU.
        d = x.unsqueeze(2) - x.unsqueeze(1)  # [B, D, D]
        phi_d = self._phi(d)                 # [B, D, D]

        # Low-rank interactions:
        # We want: interaction_i = sum_j W_{ij} * phi(x_i - x_j)
        # with W ≈ A B^T => W_{ij} = sum_k A_{ik} B_{jk}
        #
        # Then interaction_i = sum_k A_{ik} * sum_j B_{jk} * phi_{ij}
        #
        # Compute S_{k,i} = sum_j B_{jk} * phi_{ij}  for each k
        # phi_d: [B, D, D], B: [D, r]
        # -> tmp: [B, D, r] where tmp[:, i, k] = sum_j phi[:, i, j] * B[j, k]
        tmp = torch.einsum("bij,jr->bir", phi_d, self.B)

        # interaction[:, i] = sum_k tmp[:, i, k] * A[i, k]
        interaction = (tmp * self.A.unsqueeze(0)).sum(dim=2)  # [B, D]

        strength = torch.exp(self.log_strength).clamp(min=0.0)
        y = x - strength * interaction
        return y


class InteractionEnergyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fcs = nn.ModuleList([
            nn.Linear(28 * 28, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10),
        ])

        # Interaction-energy activations per hidden layer dimension
        self.activations = nn.ModuleList([
            InteractionEnergy(dim=256, rank=16, phi="tanh", strength=0.5, learn_strength=True),
            InteractionEnergy(dim=128, rank=16, phi="tanh", strength=0.5, learn_strength=True),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)

        for fc, act in zip(self.fcs[:-1], self.activations):
            x = fc(x)
            x = act(x)

        x = self.fcs[-1](x)
        return x