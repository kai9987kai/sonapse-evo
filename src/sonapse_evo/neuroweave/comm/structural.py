"""Structural encoder: graph-derived fingerprints per neuron.

Sixteen features per neuron (adapted from NEUROWEAVE-LAB ``computeStructural``):
in/out degree, mean in/out |weight|, reciprocity, two-hop reach, neighbour
out-degree, source sign, mean incoming delay and myelin, a triangle (shared
target) ratio, plus four slowly varying state features (myelin, utility,
calcium, trace).  The first eleven are static given the graph and are
recomputed only when the topology changes.

Effect on computation: the *context message* delivered to the context soma and
distal dendrite weights each presynaptic source by

    c_w[src] = 1 + 0.35 * in_degree[src] / max_in_degree + 0.20 * reciprocity[src]

so that well-connected, reciprocally embedded sources contribute more context.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from ..contracts import Mechanism, Stage, StepContext

FINGERPRINT_FEATURES: Tuple[str, ...] = (
    "in_degree", "out_degree", "mean_in_weight", "mean_out_weight", "reciprocity", "two_hop", "neighbour_out_degree",
    "source_sign", "mean_in_delay", "mean_in_myelin", "triangle_ratio", "myelin", "utility", "calcium",
    "excitatory", "trace",
)


def compute_fingerprints(pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int,
                         delay: np.ndarray = None, myelin: np.ndarray = None) -> np.ndarray:
    """Static part of the fingerprint: an ``(n, 11)`` float32 array.  O(E) plus one sparse product."""
    pre, post = np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    live = np.abs(w) > 0
    pre, post, w = pre[live], post[live], w[live]
    delay = np.zeros(len(w)) if delay is None else np.asarray(delay, dtype=np.float64)[live]
    myelin = np.full(len(w), 0.5) if myelin is None else np.asarray(myelin, dtype=np.float64)[live]
    in_deg = np.bincount(post, minlength=n).astype(np.float64)
    out_deg = np.bincount(pre, minlength=n).astype(np.float64)
    safe_in, safe_out = np.maximum(in_deg, 1.0), np.maximum(out_deg, 1.0)
    mean_in_w = np.bincount(post, weights=np.abs(w), minlength=n) / safe_in
    mean_out_w = np.bincount(pre, weights=np.abs(w), minlength=n) / safe_out
    keys = pre * n + post
    reciprocal = np.isin(post * n + pre, keys).astype(np.float64)
    reciprocity = np.bincount(post, weights=reciprocal, minlength=n) / safe_in
    two_hop = np.bincount(post, weights=in_deg[pre], minlength=n) / (safe_in * n)
    neighbour = np.bincount(post, weights=out_deg[pre], minlength=n) / (safe_in * n)
    sign = np.where(np.arange(n) % 5 == 0, -1.0, 1.0)
    mean_delay = np.bincount(post, weights=delay, minlength=n) / safe_in / 3.0
    mean_myelin = np.bincount(post, weights=myelin, minlength=n) / safe_in
    adj = sp.csr_matrix((np.ones(len(pre)), (pre, post)), shape=(n, n))
    adj.data[:] = 1.0
    shared = (adj @ adj.T).multiply(adj).sum(axis=0)          # sources of i that share a target with i
    triangles = np.asarray(shared).ravel() / np.maximum(in_deg * out_deg, 1.0)
    feats = np.stack([in_deg / n, out_deg / n, mean_in_w, mean_out_w, reciprocity, two_hop, neighbour, sign,
                      mean_delay, mean_myelin, triangles], axis=1)
    return feats.astype(np.float32)


class StructuralEncoder(Mechanism):
    mechanism_id = "structural_encoder"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"degree_weight": 0.35, "reciprocity_weight": 0.20}

    def on_attach(self) -> None:
        self._rebuild()

    def on_topology_change(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        sub = self.substrate
        pre, post = sub.edge_index()
        w = sub.effective_weights().detach()
        static = compute_fingerprints(pre.numpy(), post.numpy(), w.numpy(), sub.total_neurons,
                                      sub.syn.delay.numpy(), sub.syn.myelin.numpy())
        self.static = torch.as_tensor(static)
        in_deg = self.static[:, 0]
        self.degree_norm = in_deg / torch.clamp(in_deg.max(), min=1e-9)
        self.context_weight = (1.0 + self.params["degree_weight"] * self.degree_norm
                               + self.params["reciprocity_weight"] * self.static[:, 4])

    def fingerprints(self) -> torch.Tensor:
        """Full ``(N, 16)`` fingerprint including the current slow state features."""
        cell = self.substrate.cell
        dynamic = torch.stack([cell.myelin, cell.utility, cell.calcium, self.static[:, 7].clamp(min=0.0) * 2 - 1, cell.trace], dim=1)
        return torch.cat([self.static, dynamic], dim=1)

    def begin_step(self, ctx: StepContext) -> None:
        ctx.context_weight = self.context_weight if ctx.context_weight is None else ctx.context_weight * self.context_weight

    def state_dict(self) -> Dict[str, Any]:
        return {"static": self.static.clone(), "context_weight": self.context_weight.clone(), "degree_norm": self.degree_norm.clone()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("static") is not None and len(state["static"]) == self.substrate.total_neurons:
            self.static = torch.as_tensor(state["static"], dtype=torch.float32).clone()
            self.context_weight = torch.as_tensor(state["context_weight"], dtype=torch.float32).clone()
            self.degree_norm = torch.as_tensor(state["degree_norm"], dtype=torch.float32).clone()

    def telemetry(self) -> Dict[str, Any]:
        s = self.static
        return {"mean_in_degree": float(s[:, 0].mean() * self.substrate.total_neurons),
                "mean_reciprocity": float(s[:, 4].mean()), "mean_triangle_ratio": float(s[:, 10].mean()),
                "context_weight_max": float(self.context_weight.max())}
