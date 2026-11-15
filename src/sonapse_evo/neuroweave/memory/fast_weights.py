"""Universal fast weights: a small reward-updated key/value association matrix.

Key      k = first K observation dims                         (K = 8)
Value    v = K pooled features of the substrate state
Read     f = M k ;   extra_in[i] += f[i mod K] * read_gain    (0.03)
Write    when the lagged reward R != 0 (learning on):
         M <- clip(0.999 M + eta |R| R * outer(v, k), -1, 1)  (eta = 0.0012)
A zero matrix contributes exactly nothing.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..contracts import Mechanism, Stage, StepContext
from ..predictive.world_model import pooled_features


class FastWeights(Mechanism):
    mechanism_id = "fast_weights"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"dim": 8, "read_gain": 0.03, "eta": 0.0012, "decay": 0.999}

    def on_attach(self) -> None:
        k = int(self.params["dim"])
        self.matrix = np.zeros((k, k))
        self.slot = torch.arange(self.substrate.total_neurons) % k
        self.updates = 0

    def _key(self, ctx: StepContext) -> np.ndarray:
        k = int(self.params["dim"])
        key = ctx.observation.detach().numpy()[:k].astype(np.float64)
        return np.pad(key, (0, k - key.size))

    def begin_step(self, ctx: StepContext) -> None:
        if not np.any(self.matrix):
            return
        values = torch.as_tensor(self.matrix @ self._key(ctx), dtype=torch.float32)
        ctx.extra_in += values[self.slot] * float(self.params["read_gain"])

    def end_step(self, ctx: StepContext) -> None:
        reward = float(np.clip(ctx.signals.reward, -1.0, 1.0))
        if not ctx.learning or reward == 0.0:
            return
        p = self.params
        pool = pooled_features(self.substrate)[: int(p["dim"])]
        self.matrix = np.clip(p["decay"] * self.matrix + p["eta"] * abs(reward) * reward * np.outer(pool, self._key(ctx)), -1.0, 1.0)
        self.updates += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"matrix": torch.as_tensor(self.matrix), "updates": int(self.updates)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.matrix = np.asarray(state["matrix"], dtype=np.float64).copy()
        self.updates = int(state.get("updates", 0))

    def telemetry(self) -> Dict[str, Any]:
        return {"updates": int(self.updates), "matrix_norm": float(np.linalg.norm(self.matrix)),
                "matrix_max": float(np.abs(self.matrix).max())}
