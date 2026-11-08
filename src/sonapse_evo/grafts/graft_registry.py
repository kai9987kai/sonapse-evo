from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn

from .low_rank_bridge import LowRankBridge
from .recurrent_microcircuit import RecurrentMicrocircuit
from .biological_graft import BiologicalMotifGraft
from .donor_adapter import DonorAdapter
from ..contracts.graft_manifest import GraftSpec


class GraftRegistry(nn.Module):
    """Central registry and execution bus for all active neural grafts."""

    def __init__(self, total_neurons: int, global_cap: float = 0.10):
        super().__init__()
        self.total_neurons = total_neurons
        self.global_cap = float(global_cap)
        self.grafts = nn.ModuleDict()
        self.specs: Dict[str, GraftSpec] = {}

    def register_graft(self, graft: nn.Module, spec: GraftSpec) -> None:
        if spec.graft_id in self.grafts:
            raise KeyError(f"Graft with ID '{spec.graft_id}' already registered")
        self.grafts[spec.graft_id] = graft
        self.specs[spec.graft_id] = spec

    def unregister_graft(self, graft_id: str) -> None:
        if graft_id in self.grafts:
            del self.grafts[graft_id]
            del self.specs[graft_id]

    def clear(self) -> None:
        self.grafts.clear()
        self.specs.clear()

    def step_ramps(self) -> None:
        for graft in self.grafts.values():
            if hasattr(graft, "step_ramp"):
                graft.step_ramp()

    def reset_states(self) -> None:
        for graft in self.grafts.values():
            if hasattr(graft, "reset_state"):
                graft.reset_state()

    def compute_total_current(
        self,
        full_rates: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        donor_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(self.grafts) == 0:
            return torch.zeros_like(full_rates)

        total_current = torch.zeros_like(full_rates)
        for gid, graft in self.grafts.items():
            if isinstance(graft, DonorAdapter):
                i_g = graft(full_rates, donor_representation=donor_features, context=context)
            else:
                i_g = graft(full_rates, context=context)
            total_current = total_current + i_g

        # Enforce global infinity-norm cap: ||I||_inf <= B
        clamped = torch.clamp(total_current, -self.global_cap, self.global_cap)
        return clamped

    def count_parameters(self) -> int:
        return sum(sum(p.numel() for p in g.parameters()) for g in self.grafts.values())

    def get_summary(self) -> List[Dict[str, Any]]:
        summary = []
        for gid, spec in self.specs.items():
            graft = self.grafts[gid]
            alpha = getattr(graft, "alpha", 0.0)
            summary.append({
                "graft_id": gid,
                "graft_type": spec.graft_type,
                "rank": spec.rank,
                "current_gain": alpha,
                "max_gain": spec.max_gain,
                "parameters": sum(p.numel() for p in graft.parameters()),
            })
        return summary
