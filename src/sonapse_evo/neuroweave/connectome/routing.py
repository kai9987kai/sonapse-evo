"""Multi-region routing: region-pair gains over the connectome's pathways.

A gain matrix ``G`` (regions x regions) scales transmission by source and target region:

    edge_gain[e] = G[region(pre_e), region(post_e)]

``G`` starts at exactly 1 everywhere, so enabling the mechanism changes nothing until a gain is set
(``set_gain`` / ``set_pathway``) or adaptation is switched on.  Named pathways are derived from region
roles: ``sensory_to_association``, ``association_to_memory``, ``memory_to_action``, ``feedback``,
``predictive_feedback`` (association -> sensory) and ``cross_region`` (every off-diagonal pair).

Optional adaptation (``adapt=True``): reward-modulated strengthening of the pathways that carried more
traffic than average,   G += eta * reward * (traffic - mean traffic) / (|traffic|_max),  clipped to [0.5, 1.5].
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..contracts import Mechanism, Stage, StepContext
from .regions import ROLE_ORDER


class MultiRegionRouting(Mechanism):
    mechanism_id = "multi_region_routing"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"adapt": False, "eta": 0.01, "gain_min": 0.5, "gain_max": 1.5, "gains": {}, "pathways": {}}

    def on_attach(self) -> None:
        sub = self.substrate
        self.regions = [int(g) for g in torch.unique(sub.region_ids).tolist()]
        self.index = torch.zeros(max(self.regions) + 1, dtype=torch.long)
        self.index[torch.as_tensor(self.regions)] = torch.arange(len(self.regions))
        layout = getattr(sub, "layout", None)
        if layout is not None and layout.num_regions == len(self.regions):
            self.roles = list(layout.roles)
        else:                                            # legacy three-region convention
            self.roles = ["sensory"] + ["association"] * max(0, len(self.regions) - 2) + ["motor"]
            self.roles = self.roles[: len(self.regions)]
        r = len(self.regions)
        self.gain = torch.ones((r, r))
        self.traffic = torch.zeros((r, r))
        for key, value in self.params["gains"].items():
            a, b = (int(x) for x in str(key).split("->"))
            self.set_gain(a, b, value)
        for name, value in self.params["pathways"].items():
            self.set_pathway(name, value)

    def pathway_pairs(self, name: str) -> List[Tuple[int, int]]:
        by_role = lambda role: [i for i, r in enumerate(self.roles) if r == role]
        if name == "sensory_to_association":
            return [(a, b) for a in by_role("sensory") for b in by_role("association")]
        if name == "association_to_memory":
            return [(a, b) for a in by_role("association") for b in by_role("memory")]
        if name == "memory_to_action":
            return [(a, b) for a in by_role("memory") + by_role("association") for b in by_role("motor")]
        if name == "predictive_feedback":
            return [(a, b) for a in by_role("association") for b in by_role("sensory")]
        if name == "feedback":
            order = [ROLE_ORDER[r] for r in self.roles]
            return [(a, b) for a in range(len(order)) for b in range(len(order)) if order[a] > order[b]]
        if name == "cross_region":
            return [(a, b) for a in range(len(self.roles)) for b in range(len(self.roles)) if a != b]
        raise KeyError(f"unknown pathway '{name}'")

    def set_gain(self, src_region: int, dst_region: int, value: float) -> None:
        p = self.params
        self.gain[int(src_region), int(dst_region)] = float(np.clip(value, p["gain_min"], p["gain_max"]))

    def set_pathway(self, name: str, value: float) -> None:
        for a, b in self.pathway_pairs(name):
            self.set_gain(a, b, value)

    def _edge_regions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sub = self.substrate
        pre, post = sub.edge_index()
        return self.index[sub.region_ids[pre]], self.index[sub.region_ids[post]]

    def begin_step(self, ctx: StepContext) -> None:
        if bool((self.gain != 1.0).any()):
            a, b = self._edge_regions()
            ctx.scale_edge_gain(self.gain[a, b])

    def end_step(self, ctx: StepContext) -> None:
        signal = self.substrate.syn.last_signal
        a, b = self._edge_regions()
        if signal.numel() != a.numel():
            return
        r = len(self.regions)
        flow = torch.zeros(r * r).scatter_add_(0, a * r + b, signal.abs()).reshape(r, r)
        self.traffic = 0.9 * self.traffic + 0.1 * flow
        if self.params["adapt"] and ctx.learning and ctx.signals.reward != 0.0:
            off = ~torch.eye(r, dtype=torch.bool)
            scale = float(self.traffic[off].max()) if r > 1 else 0.0
            if scale > 0:
                delta = self.params["eta"] * float(np.clip(ctx.signals.reward, -1, 1)) * (self.traffic - self.traffic[off].mean()) / scale
                self.gain = torch.where(off, torch.clamp(self.gain + delta, self.params["gain_min"], self.params["gain_max"]), self.gain)

    def state_dict(self) -> Dict[str, Any]:
        return {"gain": self.gain.clone(), "traffic": self.traffic.clone()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gain = torch.as_tensor(state["gain"], dtype=torch.float32).clone()
        self.traffic = torch.as_tensor(state.get("traffic", self.traffic), dtype=torch.float32).clone()

    def telemetry(self) -> Dict[str, Any]:
        r = len(self.regions)
        off = ~torch.eye(r, dtype=torch.bool)
        return {"regions": r, "roles": list(self.roles), "gain_min": float(self.gain.min()), "gain_max": float(self.gain.max()),
                "non_unit_gains": int((self.gain != 1.0).sum()), "gain": self.gain.tolist(), "traffic": self.traffic.tolist(),
                "inter_region_traffic": float(self.traffic[off].sum()) if r > 1 else 0.0,
                "intra_region_traffic": float(torch.diagonal(self.traffic).sum()), "adaptive": bool(self.params["adapt"])}
