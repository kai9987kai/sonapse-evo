"""Connectivity-only transmitter hypotheses (heuristic, WireType-inspired).

A six-class hypothesis per neuron is scored from graph features alone
(normalised in/out degree, weights, reciprocity, two-hop reach, neighbour
degree, triangle ratio), turned into probabilities with a sharpened softmax and
reported with a confidence.  The scoring weights are hand-set, **not** trained
on any transmitter annotation; the output is a hypothesis layer for experiments.

It never relabels the excitatory / inhibitory source sign.  Its only effect on
computation is a small bounded release gain for the axon terminal

    transmitter_gain = clip(0.94 + 0.12 * confidence, 0.94, 1.06).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ..comm.structural import compute_fingerprints
from ..contracts import Mechanism, Stage, StepContext

TRANSMITTER_CLASSES: Tuple[str, ...] = ("glutamate-like", "GABA-like", "acetylcholine-like", "dopamine-like",
                                        "serotonin-like", "octopamine-like")


class TransmitterInference(Mechanism):
    mechanism_id = "transmitter_inference"
    stage = Stage.MODULATE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"sharpness": 3.0, "apply_gain": True}

    def on_attach(self) -> None:
        self._rebuild()

    def on_topology_change(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        sub = self.substrate
        pre, post = sub.edge_index()
        f = torch.as_tensor(compute_fingerprints(pre.numpy(), post.numpy(), sub.effective_weights().detach().numpy(), sub.total_neurons))

        def norm(x: torch.Tensor) -> torch.Tensor:
            return x / torch.clamp(x.max(), min=1e-9)

        indeg, outdeg, in_w, out_w = norm(f[:, 0]), norm(f[:, 1]), norm(f[:, 2]), norm(f[:, 3])
        recip, two_hop, neigh, tri = f[:, 4], norm(f[:, 5]), norm(f[:, 6]), norm(f[:, 10])
        raw = torch.stack([
            0.18 + 0.62 * outdeg + 0.12 * out_w + 0.08 * recip,
            0.18 + 0.62 * indeg + 0.12 * two_hop + 0.08 * (1 - recip),
            0.18 + 0.34 * recip + 0.28 * tri + 0.20 * two_hop + 0.08 * neigh,
            0.18 + 0.35 * outdeg + 0.28 * neigh + 0.16 * recip + 0.08 * out_w,
            0.18 + 0.38 * indeg + 0.28 * two_hop + 0.16 * neigh,
            0.18 + 0.35 * outdeg + 0.25 * tri + 0.20 * out_w + 0.08 * (1 - recip),
        ], dim=1)
        self.probabilities = torch.softmax((raw - raw.max(dim=1, keepdim=True).values) * float(self.params["sharpness"]), dim=1)
        self.confidence, self.prediction = self.probabilities.max(dim=1)
        self.gain = torch.clamp(0.94 + 0.12 * self.confidence, 0.94, 1.06)

    def begin_step(self, ctx: StepContext) -> None:
        if self.params["apply_gain"]:
            ctx.scale_transmitter_gain(self.gain)

    def state_dict(self) -> Dict[str, Any]:
        return {"probabilities": self.probabilities.clone(), "gain": self.gain.clone()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        probabilities = state.get("probabilities")
        if probabilities is not None and len(probabilities) == self.substrate.total_neurons:
            self.probabilities = torch.as_tensor(probabilities, dtype=torch.float32).clone()
            self.confidence, self.prediction = self.probabilities.max(dim=1)
            self.gain = torch.as_tensor(state["gain"], dtype=torch.float32).clone()

    def label(self, neuron: int) -> str:
        return TRANSMITTER_CLASSES[int(self.prediction[neuron])]

    def telemetry(self) -> Dict[str, Any]:
        counts = torch.bincount(self.prediction, minlength=len(TRANSMITTER_CLASSES)).tolist()
        return {"class_counts": dict(zip(TRANSMITTER_CLASSES, counts)), "mean_confidence": float(self.confidence.mean()),
                "gain_min": float(self.gain.min()), "gain_max": float(self.gain.max()),
                "note": "heuristic connectivity-only hypothesis; not a measured annotation"}
