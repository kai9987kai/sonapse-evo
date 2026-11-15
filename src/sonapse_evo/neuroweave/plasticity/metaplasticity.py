"""Meta-plasticity: the plasticity of plasticity.

Sets the per-synapse learning-rate scale ``syn.lr_scale`` (which every rule
passes through in ``apply_weight_update``) from three bounded factors:

1. Activity history (BCM-like sliding threshold on the postsynaptic neuron)

       s_act_j = clip(exp(-k * (rate_ema_j - target) / target), 0.25, 4)

   persistently over-active neurons become harder to potentiate, quiet ones easier.
2. Consolidation protection (cascade / synaptic-intelligence flavour)

       c_ij  <- (1 - rho) c_ij + rho * tanh(utility_ij / mean_utility)
       s_con  = 1 / (1 + strength * c_ij)

   synapses that have been consistently useful resist further change.
3. Sign agreement of successive weight changes (delta-bar-delta)

       s_dbd <- clip(s_dbd * (1.05 if sign(dw_t) == sign(dw_{t-1}) else 0.7), 0.2, 3)

``lr_scale = clip(s_act[post] * s_con * s_dbd, lr_min, lr_max)``.  All three
factors start at exactly 1, so an untrained network is unaffected.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from ..contracts import Mechanism, Stage, StepContext


class MetaPlasticity(Mechanism):
    mechanism_id = "meta_plasticity"
    stage = Stage.LEARN

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"activity_gain": 1.0, "target_rate": 0.20, "consolidation_rate": 0.01, "consolidation_strength": 3.0,
                "lr_min": 0.1, "lr_max": 4.0, "every": 4}

    def on_attach(self) -> None:
        self._alloc()

    def on_topology_change(self) -> None:
        self._alloc()

    def _alloc(self) -> None:
        e = self.substrate.syn.n_edges
        if getattr(self, "dbd", None) is None or self.dbd.numel() != e:
            self.dbd = torch.ones(e)
            self.prev_delta = self.substrate.syn.w_delta.clone()
            self.prev_change = torch.zeros(e)

    def end_step(self, ctx: StepContext) -> None:
        p = self.params
        if not ctx.learning or ctx.step % int(p["every"]) != 0:
            return
        sub, syn = self.substrate, self.substrate.syn
        self._alloc()
        _, post = sub.edge_index()
        change = syn.w_delta - self.prev_delta
        moved = change != 0
        agree = torch.sign(change) == torch.sign(self.prev_change)
        factor = torch.where(agree, torch.full_like(self.dbd, 1.05), torch.full_like(self.dbd, 0.7))
        self.dbd = torch.where(moved & (self.prev_change != 0), torch.clamp(self.dbd * factor, 0.2, 3.0), self.dbd)
        self.prev_change = torch.where(moved, change, self.prev_change)
        self.prev_delta = syn.w_delta.clone()
        mean_utility = float(syn.utility.mean())
        if mean_utility > 0:
            target = torch.tanh(syn.utility / mean_utility)
            syn.consolidation = (1 - p["consolidation_rate"]) * syn.consolidation + p["consolidation_rate"] * target
        s_act = torch.clamp(torch.exp(-p["activity_gain"] * (sub.cell.rate_ema - p["target_rate"]) / p["target_rate"]), 0.25, 4.0)
        s_con = 1.0 / (1.0 + p["consolidation_strength"] * syn.consolidation)
        syn.lr_scale = torch.clamp(s_act[post] * s_con * self.dbd, p["lr_min"], p["lr_max"])

    def state_dict(self) -> Dict[str, Any]:
        return {"dbd": self.dbd.clone(), "prev_delta": self.prev_delta.clone(), "prev_change": self.prev_change.clone()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name in ("dbd", "prev_delta", "prev_change"):
            if name in state:
                setattr(self, name, torch.as_tensor(state[name], dtype=torch.float32).clone())

    def telemetry(self) -> Dict[str, Any]:
        syn = self.substrate.syn
        return {"lr_scale_mean": float(syn.lr_scale.mean()), "lr_scale_min": float(syn.lr_scale.min()),
                "lr_scale_max": float(syn.lr_scale.max()), "consolidation_mean": float(syn.consolidation.mean()),
                "delta_bar_delta_mean": float(self.dbd.mean())}
