"""Reward-gated three-factor plasticity on the sparse edge list (O(E) per step).

Local traces (always updated while learning is on)

    coincidence_ij = r_i(t-1) * (r_j(t) - rate_ema_j)          pre x centred post
    trace_ij       = 0.92 trace_ij + 0.08 coincidence_ij       fast
    eligibility_ij = lambda e_ij + (1 - lambda) coincidence_ij slow (tau_eligibility)
    utility_ij     = 0.995 utility_ij + 0.005 |r_i r_j|

Third factor.  Two interchangeable modes:

``mode="neuroweave"`` (default; adapted from NEUROWEAVE-LAB)
    third = clip((reward + 0.2 * dopamine) * affect_gain * plasticity_gain, -1, 1)
    d|w|  = eta * third * (trace + 0.2 * eligibility)
    No reward and neutral dopamine  ->  exactly no weight change.

``mode="sonapse"`` (sparse port of the existing dense ``NeuromodulatedPlasticityEngine``)
    M     = (0.7 * DA + 0.3 * ACh) * NA_gain,  DA := reward, ACh := uncertainty   (the existing multiplier)
    dw    = eta * M * e_ij - decay * dw_ij - kappa * (r_j - target) * dw_ij
    followed by the existing homeostatic fan-in bound  sum_i |dw_ij| <= fan_in_bound.

In both modes the change goes through ``WeaveSubstrate.apply_weight_update``,
which keeps each synapse's sign fixed (Dale-style) and its magnitude bounded.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ...substrate.plasticity import NeuromodulatoryState
from ..contracts import Mechanism, SignalKind, Stage, StepContext


class RewardGatedPlasticity(Mechanism):
    mechanism_id = "reward_gated_plasticity"
    stage = Stage.LEARN

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"mode": "neuroweave", "eta": 0.02, "tau_eligibility": 20.0, "dopamine_gain": 0.2,
                "decay": 0.0005, "kappa": 0.001, "target_rate": 0.20, "fan_in_bound": 2.0}

    def on_attach(self) -> None:
        self.lambda_e = float(np.exp(-1.0 / max(1.0, self.params["tau_eligibility"])))
        self.updates = 0
        self.last_third = 0.0
        self.last_mean_delta = 0.0
        self.prev_rates = self.substrate.r.detach().clone()

    def reset(self) -> None:
        self.prev_rates = self.substrate.r.detach().clone()

    def third_factor(self, ctx: StepContext) -> float:
        sig, p = ctx.signals, self.params
        if p["mode"] == "sonapse":
            legacy = NeuromodulatoryState(dopamine_rpe=float(sig.reward), acetylcholine_unc=float(sig.uncertainty),
                                          serotonin_risk=float(sig.modulators.serotonin),
                                          noradrenaline_gain=1.0 + float(sig.modulators.noradrenaline))
            return legacy.compute_plasticity_multiplier() * float(sig.plasticity_gain)
        gain = 1.0
        if self.enabled("affective_control"):
            a, m = sig.affect, sig.modulators
            gain = float(np.clip(0.8 + 0.25 * m.noradrenaline + 0.2 * a.curiosity - 0.15 * a.stress, 0.5, 1.4))
        return float(np.clip((sig.reward + p["dopamine_gain"] * sig.modulators.dopamine) * gain * sig.plasticity_gain, -1.0, 1.0))

    def end_step(self, ctx: StepContext) -> None:
        sub, syn, p = self.substrate, self.substrate.syn, self.params
        pre, post = sub.edge_index()
        r = sub.r.detach()
        coincidence = self.prev_rates[pre] * (r[post] - sub.cell.rate_ema[post])
        self.prev_rates = r.clone()
        if not ctx.learning:
            return
        syn.trace = 0.92 * syn.trace + 0.08 * coincidence
        syn.eligibility = self.lambda_e * syn.eligibility + (1.0 - self.lambda_e) * coincidence
        syn.utility = 0.995 * syn.utility + 0.005 * (self.prev_rates[pre] * r[post]).abs()
        syn.age = syn.age + 1
        third = self.third_factor(ctx)
        if not np.isfinite(third):                     # never let a corrupted signal reach the weights
            third = 0.0
        self.last_third = third
        if p["mode"] == "sonapse":
            dw = p["eta"] * third * syn.eligibility * syn.sign \
                - p["decay"] * syn.w_delta - p["kappa"] * (r[post] - p["target_rate"]) * syn.w_delta
            applied = sub.apply_weight_update(dw)
            fan_in = torch.zeros(sub.total_neurons).scatter_add_(0, post, syn.w_delta.abs())
            scale = torch.clamp(fan_in / p["fan_in_bound"], min=1.0)
            syn.w_delta = syn.w_delta / scale[post]
        else:
            if third == 0.0:
                return
            applied = sub.apply_weight_update(syn.sign * p["eta"] * third * (syn.trace + 0.2 * syn.eligibility))
        self.updates += 1
        self.last_mean_delta = float(applied.abs().mean())
        if self.updates % 50 == 0:
            sub.events.publish("plasticity.reward_gated", SignalKind.LEARNING, ctx.step,
                               {"third_factor": third, "mean_delta": self.last_mean_delta}, self.mechanism_id)

    def state_dict(self) -> Dict[str, Any]:
        return {"updates": int(self.updates), "prev_rates": self.prev_rates.clone()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.updates = int(state.get("updates", 0))
        if "prev_rates" in state:
            self.prev_rates = torch.as_tensor(state["prev_rates"], dtype=torch.float32).clone()

    def telemetry(self) -> Dict[str, Any]:
        syn = self.substrate.syn
        return {"mode": self.params["mode"], "updates": int(self.updates), "third_factor": self.last_third,
                "mean_applied_delta": self.last_mean_delta, "mean_eligibility": float(syn.eligibility.abs().mean()),
                "mean_trace": float(syn.trace.abs().mean())}
