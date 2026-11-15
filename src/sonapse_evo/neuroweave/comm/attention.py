"""Connectivity-context attention.

For every neuron the eight most active of its strongest incoming connections
are attended with a softmax over ``2.2 * |w_ij r_j|`` and the attended
presynaptic rate

    att_i = sum_j softmax_j(2.2 |w_ij r_j|) r_j

is delivered to the apical dendrite (``apical += gain * att``).  The candidate
set per neuron is the ``fan_in`` largest-|w| incoming edges, so the cost is
O(N * fan_in) regardless of the total edge count.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from ..contracts import Mechanism, Stage, StepContext


class ConnectivityAttention(Mechanism):
    mechanism_id = "connectivity_attention"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"fan_in": 24, "top_k": 8, "temperature": 2.2, "gain": 0.22}

    def on_attach(self) -> None:
        self.attention = torch.zeros(self.substrate.total_neurons)
        self._rebuild()

    def on_topology_change(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        sub = self.substrate
        n, fan_in = sub.total_neurons, int(self.params["fan_in"])
        pre, post = sub.edge_index()
        w = sub.effective_weights().detach().abs()
        order = torch.argsort(w, descending=True)                 # strongest edges first
        order = order[torch.argsort(post[order], stable=True)]    # grouped by target, strength order kept
        sorted_post = post[order]
        counts = torch.bincount(sorted_post, minlength=n)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(order)) - starts[sorted_post]
        keep = rank < fan_in
        self.table = torch.full((n, fan_in), -1, dtype=torch.long)     # edge ids, -1 = empty slot
        self.table[sorted_post[keep], rank[keep]] = order[keep]
        self.valid = self.table >= 0
        self.safe = torch.clamp(self.table, min=0)
        self.src = pre[self.safe]

    def begin_step(self, ctx: StepContext) -> None:
        sub = self.substrate
        r = sub.r.detach()
        w = sub.effective_weights().detach()
        src_rate = r[self.src]
        score = (w[self.safe] * src_rate).abs() * float(self.params["temperature"])
        score = torch.where(self.valid, score, torch.full_like(score, -1e9))
        k = min(int(self.params["top_k"]), score.shape[1])
        top, idx = torch.topk(score, k, dim=1)
        weights = torch.softmax(top, dim=1) * (top > -1e8)
        self.attention = (weights * torch.gather(src_rate, 1, idx)).sum(dim=1)
        ctx.apical_in += float(self.params["gain"]) * self.attention
        ctx.scratch["attention"] = self.attention

    def reset(self) -> None:
        self.attention = torch.zeros(self.substrate.total_neurons)

    def state_dict(self) -> Dict[str, Any]:
        return {"table": self.table.clone(), "attention": self.attention.clone()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        table = torch.as_tensor(state.get("table", self.table), dtype=torch.long)
        pre, _ = self.substrate.edge_index()
        if table.shape == self.table.shape and int(table.max()) < pre.numel():   # only if it fits the current edge set
            self.table = table.clone()
            self.valid = self.table >= 0
            self.safe = torch.clamp(self.table, min=0)
            self.src = pre[self.safe]
        if "attention" in state:
            self.attention = torch.as_tensor(state["attention"], dtype=torch.float32).clone()

    def telemetry(self) -> Dict[str, Any]:
        return {"mean_attention": float(self.attention.abs().mean()), "max_attention": float(self.attention.abs().max())}
