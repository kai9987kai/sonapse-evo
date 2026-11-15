"""Hierarchical prediction horizons with fast, medium and slow temporal traces.

Target        g_t = [obs_key_t (16), mean rate, mean |rate|]
Input         x_t = [g_t, ema_fast(g), ema_mid(g), ema_slow(g), 1]        (tau ~ 2, 8, 32 steps)
One predictor per horizon h in (1, 4, 16, 64):   g_hat_{t+h} = g_t + W_h x_t   (a learned correction on top of
persistence; W_h starts at zero, so an untrained predictor equals the baseline), trained by normalised LMS on
(x_{t-h}, g_t) pairs taken from a history buffer.

Skill_h = 1 - err_h / persistence_err_h compares each predictor with the "nothing changes" baseline.
The *active horizon* is the longest horizon that still beats persistence and fits under an
uncertainty- and surprise-dependent cap

    cap = max_h / (1 + 10 * uncertainty + 4 * surprise)

so the system looks further ahead in stable, well-predicted conditions and shortens its horizon when the
world model disagrees with itself or the environment turns volatile.

Recurrent predictive feedback: the predicted change over the active horizon can be projected to the apical
dendrites with ``feedback_gain``; it starts at **0** (disabled) and must be raised explicitly.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, Stage, StepContext

KEY_DIM = 16
TARGET_DIM = KEY_DIM + 2
_DECAYS = (0.5, 0.875, 0.96875)


class HierarchicalPrediction(Mechanism):
    mechanism_id = "hierarchical_prediction"
    stage = Stage.LEARN

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"horizons": [1, 4, 16, 64], "lr": 0.2, "min_skill": 0.02, "feedback_gain": 0.0, "fan": 4}

    def on_attach(self) -> None:
        p, sub = self.params, self.substrate
        self.horizons: Tuple[int, ...] = tuple(int(h) for h in p["horizons"])
        in_dim = TARGET_DIM * 4 + 1
        self.weights = {h: np.zeros((TARGET_DIM, in_dim)) for h in self.horizons}
        self.error = {h: 0.0 for h in self.horizons}
        self.baseline = {h: 0.0 for h in self.horizons}
        self.updates = 0
        rng = np.random.default_rng(sub.weave.seed + 70001)
        proj = np.zeros((sub.total_neurons, TARGET_DIM), dtype=np.float32)
        cols = rng.integers(0, TARGET_DIM, size=(sub.total_neurons, int(p["fan"])))
        np.put_along_axis(proj, cols, rng.choice([-1.0, 1.0], size=cols.shape) / np.sqrt(int(p["fan"])), axis=1)
        self.projection = torch.as_tensor(proj)
        self.reset()

    def reset(self) -> None:
        self.traces = [np.zeros(TARGET_DIM) for _ in _DECAYS]
        self.history: deque = deque(maxlen=max(self.horizons) + 1)
        self.predictions: Dict[int, np.ndarray] = {h: np.zeros(TARGET_DIM) for h in self.horizons}
        self.active_horizon = self.horizons[0]
        self.feedback = np.zeros(TARGET_DIM)

    def skill(self, horizon: int) -> float:
        base = self.baseline[horizon]
        return 0.0 if base <= 1e-12 else 1.0 - self.error[horizon] / base

    def begin_step(self, ctx: StepContext) -> None:
        gain = float(self.params["feedback_gain"])
        if gain != 0.0 and np.any(self.feedback):
            ctx.apical_in += gain * (self.projection @ torch.as_tensor(self.feedback, dtype=torch.float32))

    def end_step(self, ctx: StepContext) -> None:
        p, sig = self.params, ctx.signals
        r = self.substrate.r.detach()
        key = ctx.observation.detach().numpy()[:KEY_DIM].astype(np.float64)
        target = np.concatenate([np.pad(key, (0, KEY_DIM - key.size)), [float(r.mean()), float(r.abs().mean())]])
        self.traces = [d * t + (1 - d) * target for d, t in zip(_DECAYS, self.traces)]
        x = np.concatenate([target, *self.traces, [1.0]])
        self.history.append((x, target))
        for h in self.horizons:
            if len(self.history) > h:
                x_past, g_past = self.history[-1 - h]
                err = target - (g_past + self.weights[h] @ x_past)
                if ctx.learning:                       # skill statistics and weights adapt only while learning
                    self.error[h] = 0.97 * self.error[h] + 0.03 * float(err @ err)
                    self.baseline[h] = 0.97 * self.baseline[h] + 0.03 * float((target - g_past) @ (target - g_past))
                    self.weights[h] += p["lr"] * np.outer(err, x_past) / (1.0 + float(x_past @ x_past))
            self.predictions[h] = target + self.weights[h] @ x
        if ctx.learning:
            self.updates += 1
        cap = max(self.horizons) / (1.0 + 10.0 * max(0.0, sig.uncertainty) + 4.0 * max(0.0, sig.surprise))
        usable = [h for h in self.horizons if h <= cap and self.skill(h) > p["min_skill"]]
        self.active_horizon = max(usable) if usable else self.horizons[0]
        self.feedback = self.predictions[self.active_horizon] - target
        sig.extra["active_horizon"] = float(self.active_horizon)
        ctx.predictions["horizons"] = {"by_horizon": {h: v.copy() for h, v in self.predictions.items()},
                                       "active": self.active_horizon, "provenance": Provenance.INTERNAL_PREDICTION}

    def state_dict(self) -> Dict[str, Any]:
        return {"weights": {str(h): torch.as_tensor(w) for h, w in self.weights.items()},
                "error": {str(h): v for h, v in self.error.items()}, "baseline": {str(h): v for h, v in self.baseline.items()},
                "updates": int(self.updates), "traces": [torch.as_tensor(t) for t in self.traces],
                "history": [[torch.as_tensor(x), torch.as_tensor(g)] for x, g in self.history],
                "predictions": {str(h): torch.as_tensor(v) for h, v in self.predictions.items()},
                "active_horizon": int(self.active_horizon), "feedback": torch.as_tensor(self.feedback)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for h in self.horizons:
            if str(h) in state.get("weights", {}):
                self.weights[h] = np.asarray(state["weights"][str(h)], dtype=np.float64).copy()
                self.error[h] = float(state["error"][str(h)])
                self.baseline[h] = float(state["baseline"][str(h)])
        self.updates = int(state.get("updates", 0))
        if "traces" in state:
            self.traces = [np.asarray(t, dtype=np.float64).copy() for t in state["traces"]]
            self.history = deque([(np.asarray(x, dtype=np.float64).copy(), np.asarray(g, dtype=np.float64).copy()) for x, g in state["history"]],
                                 maxlen=max(self.horizons) + 1)
            self.predictions = {h: np.asarray(state["predictions"][str(h)], dtype=np.float64).copy() for h in self.horizons}
            self.active_horizon = int(state.get("active_horizon", self.horizons[0]))
            self.feedback = np.asarray(state["feedback"], dtype=np.float64).copy()

    def telemetry(self) -> Dict[str, Any]:
        return {"horizons": list(self.horizons), "active_horizon": int(self.active_horizon),
                "skill": {str(h): round(self.skill(h), 4) for h in self.horizons},
                "error": {str(h): float(self.error[h]) for h in self.horizons},
                "feedback_gain": float(self.params["feedback_gain"]), "updates": int(self.updates)}
