"""Cross-modal fusion on top of the sensory hub.

Sensory prediction   x_hat_t = A x_{t-1}            (A learned online by normalised LMS, initialised to identity)
Prediction error     e_c = x_c - x_hat_c            for observed channels
Reliability          v_c <- 0.95 v_c + 0.05 e_c^2 ;   precision pi_c = 1 / (v_c + eps)
                     weight_c = clip(pi_c / mean(pi), 0.25, 2)       (reliability-weighted cue combination)
Attention            weight_c *= 1 + attention * min(1, |e_c| / sqrt(v_c + eps))   (surprising channels are boosted)
Synchronisation      weight_c *= exp(-age_c / tau)                    (stale channels are discounted)
Association          C <- (1 - rho) C + rho * outer(x - mean, x - mean)            (bounded Hebbian, 16 x 16)
Missing channels     x_c = 0.5 * x_hat_c + 0.5 * (C[c, obs] . x_obs) / sum |C[c, obs]|, injected at
                     ``impute_gain`` and labelled INTERNAL_PREDICTION - an imputed value is never reported
                     as an observation.

The fused vector is delivered as a *correction* to the hub's raw injection:
``sensory_extra += gain * P @ ((weight - 1) * x_obs + impute_gain * x_imputed)``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, Stage, StepContext
from .channels import SENSORY_CHANNELS

_C = len(SENSORY_CHANNELS)


class CrossModalFusion(Mechanism):
    mechanism_id = "cross_modal_fusion"
    stage = Stage.SENSE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"lr": 0.2, "attention": 0.5, "tau": 8.0, "association_rate": 0.02, "impute_gain": 0.5,
                "weight_min": 0.25, "weight_max": 2.5}

    def on_attach(self) -> None:
        self.predictor = np.eye(_C)
        self.variance = np.full(_C, 0.01)
        self.association = np.zeros((_C, _C))
        self.mean = np.zeros(_C)
        self.updates = 0
        self.reset()

    def reset(self) -> None:
        self.previous = np.zeros(_C)
        self.has_previous = False
        self.weights = np.ones(_C)
        self.error = np.zeros(_C)
        self.fused = np.zeros(_C)
        self.fused_provenance: List[str] = [Provenance.SYNTHETIC_SIMULATION.value] * _C
        self.imputed_count = 0

    def impute(self, x: np.ndarray, missing: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        out = np.zeros(_C)
        observed = ~missing
        for c in np.nonzero(missing)[0]:
            row = self.association[c, observed]
            denom = np.abs(row).sum()
            cross = float(row @ (x[observed] - self.mean[observed]) / denom) + self.mean[c] if denom > 1e-9 else prediction[c]
            out[c] = float(np.clip(0.5 * prediction[c] + 0.5 * cross, -1.0, 1.0))
        return out

    def begin_step(self, ctx: StepContext) -> None:
        hub = self.peer("multimodal_sensory")
        if hub is None:
            return                                   # graceful degradation: nothing to fuse
        p = self.params
        x = np.asarray(ctx.scratch.get("sensory_vector", hub.current_vector()), dtype=np.float64)
        missing = np.asarray(ctx.scratch.get("sensory_missing", hub.missing_mask()), dtype=bool)
        observed = ~missing
        prediction = self.predictor @ self.previous if self.has_previous else x.copy()
        error = np.where(observed, x - prediction, 0.0)
        weights = np.ones(_C)
        if self.has_previous:
            if ctx.learning:
                self.variance = np.where(observed, 0.95 * self.variance + 0.05 * error ** 2, self.variance)
                self.predictor += p["lr"] * np.outer(error, self.previous) / (1.0 + float(self.previous @ self.previous))
                self.mean = 0.98 * self.mean + 0.02 * x
                centred = np.where(observed, x - self.mean, 0.0)
                self.association = np.clip((1 - p["association_rate"]) * self.association + p["association_rate"] * np.outer(centred, centred), -1.0, 1.0)
                self.updates += 1
            precision = 1.0 / (self.variance + 1e-6)
            weights = precision / precision[observed].mean() if observed.any() else weights
            weights = weights * (1.0 + p["attention"] * np.minimum(1.0, np.abs(error) / np.sqrt(self.variance + 1e-6)))
        age = np.maximum(0, hub.clock - hub.updated_at).astype(np.float64)
        reliability = np.nan_to_num(hub.reliability, nan=0.0, posinf=1.0, neginf=0.0)
        weights = np.clip(weights * np.exp(-np.minimum(age, 50.0) / p["tau"]) * reliability, p["weight_min"], p["weight_max"])
        imputed = self.impute(x, missing, prediction) if missing.any() and self.has_previous else np.zeros(_C)
        correction = np.where(observed, (weights - 1.0) * x, p["impute_gain"] * imputed)
        if np.any(correction):
            ctx.sensory_extra += float(hub.params["gain"]) * hub.project(correction)
        self.weights, self.error = weights, error
        self.fused = np.where(observed, weights * x, imputed)
        provenance = hub.current_provenance()
        self.fused_provenance = [Provenance.INTERNAL_PREDICTION.value if missing[i] else provenance[i] for i in range(_C)]
        self.imputed_count = int(missing.sum())
        self.previous, self.has_previous = np.where(observed, x, prediction), True
        ctx.signals.extra["sensory_prediction_error"] = float(np.abs(error[observed]).mean()) if observed.any() else 0.0
        ctx.signals.extra["sensory_uncertainty"] = float(np.sqrt(self.variance).mean())
        ctx.scratch["fused_sensory"] = self.fused
        ctx.scratch["fused_provenance"] = self.fused_provenance
        ctx.predictions["sensory"] = {"next": self.predictor @ self.previous, "provenance": Provenance.INTERNAL_PREDICTION}

    def state_dict(self) -> Dict[str, Any]:
        return {"predictor": torch.as_tensor(self.predictor), "variance": torch.as_tensor(self.variance),
                "association": torch.as_tensor(self.association), "mean": torch.as_tensor(self.mean), "updates": int(self.updates),
                "previous": torch.as_tensor(self.previous), "has_previous": bool(self.has_previous)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name in ("predictor", "variance", "association", "mean"):
            setattr(self, name, np.asarray(state[name], dtype=np.float64).copy())
        self.updates = int(state.get("updates", 0))
        if "previous" in state:
            self.previous = np.asarray(state["previous"], dtype=np.float64).copy()
            self.has_previous = bool(state.get("has_previous", False))

    def telemetry(self) -> Dict[str, Any]:
        return {"weights": {n: round(float(w), 4) for n, w in zip(SENSORY_CHANNELS, self.weights)},
                "prediction_error": {n: round(float(e), 4) for n, e in zip(SENSORY_CHANNELS, self.error)},
                "uncertainty": {n: round(float(np.sqrt(v)), 4) for n, v in zip(SENSORY_CHANNELS, self.variance)},
                "imputed_channels": int(self.imputed_count), "fused_provenance": list(self.fused_provenance),
                "association_norm": float(np.linalg.norm(self.association)), "updates": int(self.updates)}
