"""Six neuromodulator-like fields with heterogeneous, regionally biased receptor profiles.

Global fields (adapted from NEUROWEAVE-LAB; ``a`` = mean rate, ``p`` = predicted mean rate)

    DA   <- clip(0.965 DA  + 0.035 reward, -1, 1)
    5HT  <- clip(0.995 5HT + 0.005 (0.5 + a), 0, 1)
    OA   <- clip(0.92  OA  + 0.08 |a - p|, 0, 1)
    ACh  <- clip(0.90  ACh + 0.10 max(uncertainty, 0.15), 0, 1)

(noradrenaline and the endocannabinoid-like field are driven by the affective
controller when it is enabled and otherwise decay to rest).

Local fields.  Every neuron has eight receptor sensitivities (D1, D2, OA, alpha,
M1, CB1, 5HT1A, 5HT2A) drawn from a seeded distribution with a regional bias.
The local field is a low-pass of the *deviation* of each modulator from its
neutral value times the matching receptor mix, so a neutral global state gives
a local field of exactly 0 and no effect anywhere.

Effects on computation (each can be switched off through ``params`` for ablation)

    gating soma drive     x (1 + 0.15 dDA - 0.08 d5HT) x (1 + 0.08 lNA - 0.06 lCB + 0.04 lOA + 0.02 lDA + 0.03 l5HT)
    output gain           x clip(1 + 0.15 lNA, 0.8, 1.3)                (arousal / gain)
    sensory weighting     x clip(1 + 0.30 lACh, 0.7, 1.5)               (attention)
    plasticity gain       x clip(1 + 0.5 dACh + 0.3 NA - 0.2 eCB + 0.25 |DA|, 0.5, 2)
    consolidation gate    = clip(0.5 + 0.5 5HT + 0.3 dACh, 0, 1.5)      (published for sleep / memory)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..contracts import Mechanism, ModulatorState, Stage, StepContext

RECEPTORS: Tuple[str, ...] = ("d1", "d2", "octopamine", "alpha", "m1", "cb1", "serotonin1a", "serotonin2a")
_NEUTRAL = ModulatorState()
# regional receptor bias by position along the population (0 = sensory entry, 1 = motor outflow)
_REGION_BIAS = {"d1": (0.0, 0.25), "d2": (0.0, 0.10), "octopamine": (0.10, 0.0), "alpha": (0.10, 0.10),
                "m1": (0.25, 0.0), "cb1": (0.05, 0.05), "serotonin1a": (0.05, 0.15), "serotonin2a": (0.10, 0.05)}


class NeuromodulatoryField(Mechanism):
    mechanism_id = "neuromodulatory_field"
    stage = Stage.MODULATE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"gating": True, "gain": True, "sensory": True, "plasticity": True,
                "clamp_values": {}}       # name -> fixed value (intervention / do-operator on a modulator)

    def on_attach(self) -> None:
        sub = self.substrate
        n = sub.total_neurons
        rng = np.random.default_rng(sub.weave.seed + 4099)
        regions = sub.region_ids.to(torch.float32)
        position = (regions / max(1.0, float(regions.max()))).numpy()
        self.receptors: Dict[str, torch.Tensor] = {}
        for name in RECEPTORS:
            lo, hi = _REGION_BIAS[name]
            values = 0.5 + 0.5 * rng.random(n) + lo * (1.0 - position) + hi * position
            self.receptors[name] = torch.as_tensor(values, dtype=torch.float32)
        self.local = {name: torch.zeros(n) for name in ModulatorState.NAMES}
        self.steps = 0

    def reset(self) -> None:
        self.local = {name: torch.zeros(self.substrate.total_neurons) for name in ModulatorState.NAMES}

    def receptor_matrix(self) -> torch.Tensor:
        return torch.stack([self.receptors[name] for name in RECEPTORS])

    def begin_step(self, ctx: StepContext) -> None:
        sig, p = ctx.signals, self.params
        m = sig.modulators
        activity = float(self.substrate.r.detach().abs().mean())
        m.dopamine = 0.965 * m.dopamine + 0.035 * float(np.clip(sig.reward, -1.0, 1.0))
        m.serotonin = 0.995 * m.serotonin + 0.005 * (0.5 + activity)
        m.octopamine = 0.92 * m.octopamine + 0.08 * abs(activity - sig.prediction) if sig.prediction else 0.92 * m.octopamine
        m.acetylcholine = 0.90 * m.acetylcholine + 0.10 * max(float(sig.uncertainty), 0.15)
        if not self.enabled("affective_control"):
            m.noradrenaline *= 0.9
            m.endocannabinoid *= 0.96
        for name, value in p["clamp_values"].items():
            setattr(m, name, float(value))
        m.clamp()

        d = {name: getattr(m, name) - getattr(_NEUTRAL, name) for name in ModulatorState.NAMES}
        rc, lf = self.receptors, self.local
        lf["dopamine"] = 0.94 * lf["dopamine"] + 0.06 * d["dopamine"] * (rc["d1"] - 0.5 * rc["d2"])
        lf["serotonin"] = 0.97 * lf["serotonin"] + 0.03 * d["serotonin"] * (0.6 * rc["serotonin1a"] + 0.4 * rc["serotonin2a"])
        lf["octopamine"] = 0.92 * lf["octopamine"] + 0.08 * d["octopamine"] * rc["octopamine"]
        lf["noradrenaline"] = 0.92 * lf["noradrenaline"] + 0.08 * d["noradrenaline"] * rc["alpha"]
        lf["acetylcholine"] = 0.90 * lf["acetylcholine"] + 0.10 * d["acetylcholine"] * rc["m1"]
        lf["endocannabinoid"] = 0.96 * lf["endocannabinoid"] + 0.04 * d["endocannabinoid"] * rc["cb1"]

        if p["gating"]:
            global_gate = 1.0 + 0.15 * d["dopamine"] - 0.08 * d["serotonin"]
            local_gate = (1.0 + 0.08 * lf["noradrenaline"] - 0.06 * lf["endocannabinoid"] + 0.04 * lf["octopamine"]
                          + 0.02 * lf["dopamine"] + 0.03 * lf["serotonin"])
            if self.enabled("multi_soma"):
                ctx.scale_gating_drive(global_gate * local_gate)        # through the gating soma population
            else:
                ctx.scale_gain(torch.clamp(global_gate * local_gate, 0.8, 1.25))   # same modulation as an output gain
        if p["gain"] and float(lf["noradrenaline"].abs().max()) > 0:
            ctx.scale_gain(torch.clamp(1.0 + 0.15 * lf["noradrenaline"], 0.8, 1.3))
        if p["sensory"] and float(lf["acetylcholine"].abs().max()) > 0:
            ctx.scale_sensory_gain(torch.clamp(1.0 + 0.30 * lf["acetylcholine"], 0.7, 1.5))
        if p["plasticity"]:
            sig.plasticity_gain = float(np.clip(1.0 + 0.5 * d["acetylcholine"] + 0.3 * m.noradrenaline
                                                - 0.2 * m.endocannabinoid + 0.25 * abs(m.dopamine), 0.5, 2.0))
        sig.attention_gain = float(np.clip(1.0 + 0.5 * d["acetylcholine"], 0.5, 1.5))
        sig.extra["consolidation_gate"] = float(np.clip(0.5 + 0.5 * m.serotonin + 0.3 * d["acetylcholine"], 0.0, 1.5))
        self.steps += 1

    def regional_levels(self) -> Dict[str, Dict[str, float]]:
        """Mean local field per region (regional modulation observatory)."""
        out: Dict[str, Dict[str, float]] = {}
        for region in torch.unique(self.substrate.region_ids).tolist():
            idx = self.substrate.region_ids == region
            out[str(region)] = {name: float(field[idx].mean()) for name, field in self.local.items()}
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"receptors": {k: v.clone() for k, v in self.receptors.items()},
                "local": {k: v.clone() for k, v in self.local.items()}, "steps": int(self.steps)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name, value in state.get("receptors", {}).items():
            self.receptors[name] = torch.as_tensor(value, dtype=torch.float32).clone()
        for name, value in state.get("local", {}).items():
            self.local[name] = torch.as_tensor(value, dtype=torch.float32).clone()
        self.steps = int(state.get("steps", 0))

    def telemetry(self) -> Dict[str, Any]:
        return {"levels": self.substrate.signals.modulators.as_dict(),
                "local_mean": {k: float(v.mean()) for k, v in self.local.items()},
                "receptor_mean": {k: float(v.mean()) for k, v in self.receptors.items()},
                "regional": self.regional_levels(),
                "plasticity_gain": float(self.substrate.signals.plasticity_gain),
                "consolidation_gate": float(self.substrate.signals.extra.get("consolidation_gate", 0.0))}
