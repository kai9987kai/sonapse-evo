"""FlyHash-style sparse random projection (after Dasgupta, Stevens & Navlakha 2017).

A fixed sparse binary expansion matrix samples ``fan`` input dimensions per
hash unit; the ``k`` most active units win and form a sparse binary tag:

    a = M (x - mean(x)),   code = top_k(a)   (binary, exactly k ones)

Similar observations receive overlapping tags.  The tag is (a) delivered to
the apical dendrite as a centred current ``gain * (code[h(i)] - k / m)`` and
(b) exposed to memory mechanisms as a locality-sensitive key.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..contracts import Mechanism, Stage, StepContext


def flyhash_code(x: np.ndarray, projection: np.ndarray, k: int) -> np.ndarray:
    """Binary winner-take-all tag of ``x`` under a sparse binary ``projection`` (m x d)."""
    x = np.asarray(x, dtype=np.float64)
    activity = projection @ (x - x.mean())
    code = np.zeros(projection.shape[0], dtype=np.float32)
    code[np.argsort(-activity, kind="stable")[:k]] = 1.0
    return code


class FlyHashProjection(Mechanism):
    mechanism_id = "flyhash_projection"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"hash_dim": 64, "fan": 6, "winners": 8, "input_dims": 32, "gain": 0.20}

    def on_attach(self) -> None:
        p, sub = self.params, self.substrate
        d = min(int(p["input_dims"]), sub.sensory_dim)
        rng = np.random.default_rng(sub.weave.seed + 31337)
        proj = np.zeros((int(p["hash_dim"]), d), dtype=np.float32)
        for row in proj:
            row[rng.choice(d, size=min(int(p["fan"]), d), replace=False)] = 1.0
        self.projection = proj
        self.unit_of = torch.as_tensor(rng.permutation(sub.total_neurons) % int(p["hash_dim"]), dtype=torch.long)
        self.code = np.zeros(int(p["hash_dim"]), dtype=np.float32)

    def tag(self, observation) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)[: self.projection.shape[1]]
        return flyhash_code(x, self.projection, int(self.params["winners"]))

    def begin_step(self, ctx: StepContext) -> None:
        obs = ctx.observation.detach().numpy()
        if not np.any(obs[: self.projection.shape[1]]):
            self.code = np.zeros_like(self.code)          # no input: no tag, exactly neutral
            return
        self.code = self.tag(obs)
        centred = torch.as_tensor(self.code - self.code.mean())
        ctx.apical_in += float(self.params["gain"]) * centred[self.unit_of]
        ctx.scratch["flyhash_code"] = self.code

    def telemetry(self) -> Dict[str, Any]:
        return {"active_bits": int(self.code.sum()), "hash_dim": int(self.projection.shape[0])}
