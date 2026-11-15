"""Predictive replay: surprise- and novelty-prioritised reactivation of stored states.

Buffer    (key, rate state, priority) with priority = surprise + |reward| + novelty, bounded.
Schedule  every ``period`` steps (or ``run_replay(n)``) the ``batch`` highest-priority items are queued.
Replay    one queued state per following step is re-injected as a weak apical current
          (``gain`` = 0.05) labelled ``PRIOR_EXPERIENCE`` - a replayed state is never an observation.
          Its priority decays by ``priority_decay`` so replay moves on to other items.
Other subsystems (sleep consolidation, the Micro-Collective) pull ``prioritised_batch(n)`` / ``replay_states(n)``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, SignalKind, Stage, StepContext
from .stores import cosine


class PredictiveReplay(Mechanism):
    mechanism_id = "predictive_replay"
    stage = Stage.STRUCTURE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"capacity": 128, "period": 64, "batch": 4, "gain": 0.05, "priority_decay": 0.5, "key_dim": 32,
                "min_priority": 0.05}

    def on_attach(self) -> None:
        self.items: List[Dict[str, Any]] = []
        self.queue: List[torch.Tensor] = []
        self.replayed = 0
        self.phases = 0

    def reset(self) -> None:
        self.queue = []

    def begin_step(self, ctx: StepContext) -> None:
        if self.queue:
            ctx.apical_in += float(self.params["gain"]) * self.queue.pop(0)
            ctx.memory["replay"] = {"provenance": Provenance.PRIOR_EXPERIENCE, "remaining": len(self.queue)}
            self.replayed += 1

    def end_step(self, ctx: StepContext) -> None:
        if not ctx.learning:
            return
        p = self.params
        key = ctx.observation.detach().numpy()[: int(p["key_dim"])].astype(np.float32)
        novelty = 1.0 - max([cosine(key, it["key"]) for it in self.items[-16:]], default=0.0)
        priority = float(ctx.signals.surprise) + abs(float(ctx.signals.reward)) + 0.25 * max(0.0, novelty)
        if priority >= p["min_priority"]:
            self.items.append({"key": key, "state": self.substrate.r.detach().clone(), "priority": priority, "step": int(ctx.step)})
            if len(self.items) > int(p["capacity"]):
                del self.items[min(range(len(self.items)), key=lambda i: (self.items[i]["priority"], self.items[i]["step"]))]
        if self.substrate.weave_steps % int(p["period"]) == 0:
            self.run_replay(int(p["batch"]), ctx.step)

    def prioritised_batch(self, n: int) -> List[Dict[str, Any]]:
        return sorted(self.items, key=lambda it: (-it["priority"], it["step"]))[:n]

    def replay_states(self, n: int) -> List[torch.Tensor]:
        return [it["state"] for it in self.prioritised_batch(n)]

    def run_replay(self, n: int, step: int = 0) -> int:
        batch = self.prioritised_batch(n)
        for item in batch:
            self.queue.append(item["state"])
            item["priority"] *= float(self.params["priority_decay"])
        if batch:
            self.phases += 1
            self.substrate.events.publish("memory.replay", SignalKind.MEMORY_RETRIEVAL, step,
                                          {"scheduled": len(batch), "provenance": Provenance.PRIOR_EXPERIENCE.value}, self.mechanism_id)
        return len(batch)

    def state_dict(self) -> Dict[str, Any]:
        return {"items": [{"key": torch.as_tensor(it["key"]), "state": it["state"].clone(), "priority": it["priority"],
                           "step": it["step"]} for it in self.items], "replayed": int(self.replayed), "phases": int(self.phases),
                "queue": [q.clone() for q in self.queue]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.items = [{"key": np.asarray(it["key"], dtype=np.float32), "state": torch.as_tensor(it["state"], dtype=torch.float32).clone(),
                       "priority": float(it["priority"]), "step": int(it["step"])} for it in state.get("items", [])]
        self.replayed, self.phases = int(state.get("replayed", 0)), int(state.get("phases", 0))
        self.queue = [torch.as_tensor(q, dtype=torch.float32).clone() for q in state.get("queue", [])]

    def telemetry(self) -> Dict[str, Any]:
        return {"buffered": len(self.items), "queued": len(self.queue), "replayed": int(self.replayed), "phases": int(self.phases),
                "mean_priority": float(np.mean([it["priority"] for it in self.items])) if self.items else 0.0}
