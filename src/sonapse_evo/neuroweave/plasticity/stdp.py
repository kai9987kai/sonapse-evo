"""Pair-based spike-timing-dependent plasticity on the sparse edge list.

Exponentially decaying spike traces per synapse

    x_pre  <- clip(0.92 x_pre  + s_pre,  0, 1.5)
    x_post <- clip(0.92 x_post + s_post, 0, 1.5)

``rule="causal"`` (default, classical pair STDP: pre-before-post potentiates)
    d|w| = gate * (A_plus * s_post * x_pre  -  A_minus * s_pre * x_post)
``rule="neuroweave"`` (the timing convention used in NEUROWEAVE-LAB 0.4.0)
    d|w| = gate * (A_plus * s_pre * x_post  -  A_minus * s_post * x_pre)

``gate = clip(0.75 + 0.25 ACh + 0.15 NA - 0.2 eCB, 0.45, 1.35)`` when affective
control is enabled, else 1.  Spike events come from the membrane model; if
``membrane_spiking`` is disabled a rate-onset event (rate above twice the
population mean plus 0.05) stands in so the rule degrades gracefully.  The
source sign never changes and magnitudes stay bounded.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..contracts import Mechanism, Stage, StepContext


class PairSTDP(Mechanism):
    mechanism_id = "pair_stdp"
    stage = Stage.LEARN

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"rule": "causal", "a_plus": 0.004, "a_minus": 0.0045, "trace_decay": 0.92, "trace_max": 1.5}

    def on_attach(self) -> None:
        self.updates = 0
        self.last_potentiation = 0.0
        self.last_depression = 0.0

    def spikes(self) -> torch.Tensor:
        sub = self.substrate
        if self.enabled("membrane_spiking"):
            return sub.cell.spike
        r = sub.r.detach()
        return (r > 2.0 * r.mean() + 0.05).to(torch.float32)

    def end_step(self, ctx: StepContext) -> None:
        sub, syn, p = self.substrate, self.substrate.syn, self.params
        pre, post = sub.edge_index()
        spikes = self.spikes()
        s_pre, s_post = spikes[pre], spikes[post]
        if ctx.learning:
            gate = 1.0
            if self.enabled("affective_control"):
                m = ctx.signals.modulators
                gate = float(np.clip(0.75 + 0.25 * m.acetylcholine + 0.15 * m.noradrenaline - 0.2 * m.endocannabinoid, 0.45, 1.35))
            gate *= float(ctx.signals.plasticity_gain)
            if p["rule"] == "neuroweave":
                pot, dep = p["a_plus"] * s_pre * syn.post_trace, p["a_minus"] * s_post * syn.pre_trace
            else:
                pot, dep = p["a_plus"] * s_post * syn.pre_trace, p["a_minus"] * s_pre * syn.post_trace
            sub.apply_weight_update(syn.sign * gate * (pot - dep))
            self.last_potentiation, self.last_depression = float(pot.sum()), float(dep.sum())
            self.updates += 1
        syn.pre_trace = torch.clamp(p["trace_decay"] * syn.pre_trace + s_pre, 0.0, p["trace_max"])
        syn.post_trace = torch.clamp(p["trace_decay"] * syn.post_trace + s_post, 0.0, p["trace_max"])

    def state_dict(self) -> Dict[str, Any]:
        return {"updates": int(self.updates)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.updates = int(state.get("updates", 0))

    def telemetry(self) -> Dict[str, Any]:
        syn = self.substrate.syn
        return {"rule": self.params["rule"], "updates": int(self.updates), "potentiation": self.last_potentiation,
                "depression": self.last_depression, "pre_trace_max": float(syn.pre_trace.max()) if syn.pre_trace.numel() else 0.0,
                "post_trace_max": float(syn.post_trace.max()) if syn.post_trace.numel() else 0.0,
                "event_rate": float(self.spikes().mean())}
