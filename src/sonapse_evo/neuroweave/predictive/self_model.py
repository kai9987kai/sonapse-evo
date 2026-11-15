"""Action-grounded self-model (efference copy and comparator).

For every completed action the model compares

    intended action   (what the controller chose)
    issued command    (what was actually sent, after exploration noise / persistence)
    predicted outcome (forward model, or the environment's own kinematic prediction when it supplies one)
    observed outcome  (what actually happened)

and estimates

    error        e = |predicted - observed|,   scale = |predicted| + 0.25
    authorship     = clip(1 - e / scale, 0, 1)                    "did my command cause this?"
    actuator slip  = clip(|issued - intended| / (|intended| + 0.25), 0, 1)
    disturbance    = clip(e / scale - slip, 0, 1)                 unexplained, external
    reliability   <- 0.95 reliability + 0.05 authorship           (published as ``signals.control_reliability``)

These feed back into learning and decision making: credit for outcomes that were not self-caused is
discounted (``plasticity_gain *= floor + (1 - floor) * authorship``) and behavioural persistence is scaled
by control reliability.  When the environment provides no outcome vectors, the outcome is the change of the
first 16 observation dimensions and a linear forward model is learned online.

These are diagnostics of a control loop; they are not a claim of agency or self-awareness.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ..contracts import Mechanism, Stage, StepContext

KEY_DIM, ACTION_DIM = 16, 5


def _vec(value, size: int) -> np.ndarray:
    out = np.zeros(size)
    if value is not None:
        v = np.asarray(value, dtype=np.float64).ravel()[:size]
        out[: v.size] = v
    return out


class EfferenceSelfModel(Mechanism):
    mechanism_id = "efference_self_model"
    stage = Stage.LEARN

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"lr": 0.2, "credit_floor": 0.5, "couple_learning": True, "couple_persistence": True, "warmup": 20}

    def on_attach(self) -> None:
        self.forward = np.zeros((KEY_DIM, KEY_DIM + ACTION_DIM + 1))
        self.reliability = 1.0
        self.comparisons = 0
        self.reset()

    def reset(self) -> None:
        self.prev_key: Optional[np.ndarray] = None
        self.report = {"authorship": 1.0, "disturbance": 0.0, "actuator_slip": 0.0, "error": 0.0, "source": "none"}

    def begin_step(self, ctx: StepContext) -> None:
        p, sig = self.params, ctx.signals
        fb = sig.feedback
        key = _vec(ctx.observation.detach().numpy(), KEY_DIM)
        if fb.issued_action is not None and self.prev_key is not None:
            intended, issued = _vec(fb.intended_action, ACTION_DIM), _vec(fb.issued_action, ACTION_DIM)
            if fb.predicted_outcome is not None and fb.observed_outcome is not None:
                predicted = np.asarray(fb.predicted_outcome, dtype=np.float64).ravel()
                observed = np.asarray(fb.observed_outcome, dtype=np.float64).ravel()
                common = min(predicted.size, observed.size)
                predicted, observed = predicted[:common], observed[:common]
                source = "environment"
            else:
                observed = key - self.prev_key
                predicted = self.forward @ np.concatenate([self.prev_key, intended, [1.0]])
                source = "forward_model"
                if ctx.learning:
                    x = np.concatenate([self.prev_key, issued, [1.0]])
                    self.forward += p["lr"] * np.outer(observed - self.forward @ x, x) / (1.0 + float(x @ x))
            valid = (predicted.size > 0 and np.all(np.isfinite(predicted)) and np.all(np.isfinite(observed))
                     and np.all(np.isfinite(intended)) and np.all(np.isfinite(issued)))
            if not valid:                              # corrupted feedback is ignored, never learned from
                self.prev_key = key
                sig.control_reliability = float(self.reliability)
                ctx.predictions["self_model"] = dict(self.report, control_reliability=float(self.reliability), skipped=True)
                return
            error = float(np.linalg.norm(predicted - observed))
            scale = max(float(np.linalg.norm(predicted)), float(np.linalg.norm(observed))) + 0.25
            slip = float(np.clip(np.linalg.norm(issued - intended) / (np.linalg.norm(intended) + 0.25), 0.0, 1.0))
            authorship = float(np.clip(1.0 - error / scale, 0.0, 1.0))
            self.report = {"authorship": authorship, "disturbance": float(np.clip(error / scale - slip, 0.0, 1.0)),
                           "actuator_slip": slip, "error": error, "source": source}
            if ctx.learning:                           # the running estimate adapts only while learning
                self.reliability = 0.95 * self.reliability + 0.05 * authorship
                self.comparisons += 1
        self.prev_key = key
        sig.control_reliability = float(self.reliability)
        trusted = self.report["source"] == "environment" or self.comparisons >= int(p["warmup"])
        if p["couple_learning"] and trusted:
            sig.plasticity_gain = float(sig.plasticity_gain * (p["credit_floor"] + (1.0 - p["credit_floor"]) * self.report["authorship"]))
        if p["couple_persistence"] and trusted:
            sig.persistence = float(sig.persistence * self.reliability)
        ctx.predictions["self_model"] = dict(self.report, control_reliability=float(self.reliability))

    def state_dict(self) -> Dict[str, Any]:
        return {"forward": torch.as_tensor(self.forward), "reliability": float(self.reliability), "comparisons": int(self.comparisons),
                "prev_key": None if self.prev_key is None else torch.as_tensor(self.prev_key), "report": dict(self.report)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.forward = np.asarray(state["forward"], dtype=np.float64).copy()
        self.reliability = float(state.get("reliability", 1.0))
        self.comparisons = int(state.get("comparisons", 0))
        prev = state.get("prev_key")
        self.prev_key = None if prev is None else np.asarray(prev, dtype=np.float64).copy()
        if "report" in state:
            self.report = dict(state["report"])

    def telemetry(self) -> Dict[str, Any]:
        return dict(self.report, control_reliability=float(self.reliability), comparisons=int(self.comparisons))
