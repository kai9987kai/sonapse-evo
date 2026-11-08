from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence
import numpy as np
import torch


@dataclass(frozen=True)
class NeuralPort:
    port_id: str
    indices: Tuple[int, ...]
    region_id: int
    name: str = ""

    @property
    def width(self) -> int:
        return len(self.indices)

    def extract(self, rates: torch.Tensor) -> torch.Tensor:
        """Extract activity r_S from full rate vector r."""
        idx = torch.as_tensor(self.indices, dtype=torch.long, device=rates.device)
        return rates[..., idx]

    def project(self, local_current: torch.Tensor, total_neurons: int) -> torch.Tensor:
        """Scatter local current to full N-dimensional neuron space (P_T)."""
        out = torch.zeros((*local_current.shape[:-1], total_neurons), dtype=local_current.dtype, device=local_current.device)
        idx = torch.as_tensor(self.indices, dtype=torch.long, device=local_current.device)
        out[..., idx] = local_current
        return out


def create_uniform_ports(
    total_neurons: int,
    num_regions: int,
    port_width: int = 32,
    ports_per_region: int = 2,
    seed: int = 42,
) -> Tuple[NeuralPort, ...]:
    rng = np.random.default_rng(seed)
    nodes_per_region = total_neurons // num_regions
    ports = []

    for r in range(num_regions):
        start = r * nodes_per_region
        end = start + nodes_per_region if r < num_regions - 1 else total_neurons
        region_nodes = np.arange(start, end)

        for p in range(ports_per_region):
            p_start = start + p * port_width
            if p_start + port_width <= end:
                chosen = np.arange(p_start, p_start + port_width)
            else:
                chosen = rng.choice(region_nodes, size=min(port_width, len(region_nodes)), replace=False)
            chosen.sort()
            port = NeuralPort(
                port_id=f"R{r}_P{p}",
                indices=tuple(int(x) for x in chosen),
                region_id=r,
                name=f"Region-{r}-Port-{p}",
            )
            ports.append(port)

    return tuple(ports)
