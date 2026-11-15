"""Associative trace memory (adapted from NEUROWEAVE-LAB ``memory``).

Store   when the lagged reward exceeds ``reward_threshold`` on every ``store_every``-th step:
        episode = (key = first ``key_dim`` observation dims, state = rate vector); FIFO, ``capacity`` episodes.
Recall  best cosine match of the current key; if similarity > ``threshold``:
        extra_in += state * similarity * gain            (0.11 in the reference)
An empty store contributes exactly nothing.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, Stage, StepContext
from .stores import cosine


class AssociativeTraceMemory(Mechanism):
    mechanism_id = "associative_trace_memory"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"key_dim": 32, "capacity": 32, "threshold": 0.35, "gain": 0.11, "reward_threshold": 0.35, "store_every": 12}

    def on_attach(self) -> None:
        self.episodes: List[Dict[str, Any]] = []
        self.hits = 0
        self.last_similarity = 0.0

    def _key(self, ctx: StepContext) -> np.ndarray:
        return ctx.observation.detach().numpy()[: int(self.params["key_dim"])].astype(np.float32)

    def begin_step(self, ctx: StepContext) -> None:
        self.last_similarity = 0.0
        if not self.episodes:
            return
        key = self._key(ctx)
        sims = [cosine(key, ep["key"]) for ep in self.episodes]
        best = int(np.argmax(sims))
        self.last_similarity = float(sims[best])
        if sims[best] > self.params["threshold"]:
            ctx.extra_in += self.episodes[best]["state"] * float(sims[best] * self.params["gain"])
            ctx.memory["associative"] = {"similarity": float(sims[best]), "index": best,
                                         "provenance": Provenance.PRIOR_EXPERIENCE}
            self.hits += 1

    def end_step(self, ctx: StepContext) -> None:
        p = self.params
        if ctx.learning and ctx.signals.reward > p["reward_threshold"] and ctx.step % int(p["store_every"]) == 0:
            self.episodes.append({"key": self._key(ctx), "state": self.substrate.r.detach().clone(),
                                  "reward": float(ctx.signals.reward), "step": int(ctx.step)})
            self.episodes = self.episodes[-int(p["capacity"]):]

    def replay_states(self, n: int) -> List[torch.Tensor]:
        return [ep["state"] for ep in self.episodes[-n:]]

    def state_dict(self) -> Dict[str, Any]:
        return {"episodes": [{"key": torch.as_tensor(e["key"]), "state": e["state"].clone(), "reward": e["reward"],
                              "step": e["step"]} for e in self.episodes], "hits": int(self.hits)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.episodes = [{"key": np.asarray(e["key"], dtype=np.float32), "state": torch.as_tensor(e["state"], dtype=torch.float32).clone(),
                          "reward": float(e["reward"]), "step": int(e["step"])} for e in state.get("episodes", [])]
        self.hits = int(state.get("hits", 0))

    def telemetry(self) -> Dict[str, Any]:
        return {"episodes": len(self.episodes), "capacity": int(self.params["capacity"]), "hits": int(self.hits),
                "last_similarity": self.last_similarity}
