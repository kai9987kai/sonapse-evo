"""Micro-Collective attached to the connectome.

Each step the mechanism reads the current sixteen-channel sensory vector (from the sensory hub when it is
attached, otherwise from the first sixteen observation dimensions), optionally trains the collective on
it, and publishes its next-sensory prediction (labelled ``INTERNAL_PREDICTION``).

The prediction is injected into the connectome as predictive context on the apical dendrites:

    apical_in += context_gain * P @ y_hat ,   context_gain = max_context_gain * clip(skill, 0, 1)

``skill`` is measured on held-out transitions against the persistence baseline.  An untrained collective
has skill 0, so its contribution is exactly zero until validation shows that it predicts better than
"nothing changes".  The modifier mechanisms ``expert_routing``, ``collective_meta_learning`` and
``predictive_replay`` switch the corresponding features of the collective.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, Stage, StepContext
from .micro_collective import MicroCollective


class MicroCollectiveMechanism(Mechanism):
    mechanism_id = "micro_collective"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"dim": 16, "experts": 4, "top_k": 2, "lr": 0.02, "train": True, "max_context_gain": 0.25, "fan": 4,
                "skill_every": 16}

    def on_attach(self) -> None:
        sub, p = self.substrate, self.params
        self.collective = MicroCollective(dim=int(p["dim"]), n_experts=int(p["experts"]), top_k=int(p["top_k"]), seed=sub.weave.seed,
                                          lr=float(p["lr"]), expert_routing=self.enabled("expert_routing"),
                                          meta_learning=self.enabled("collective_meta_learning"),
                                          predictive_replay=self.enabled("predictive_replay"))
        rng = np.random.default_rng(sub.weave.seed + 424243)
        proj = np.zeros((sub.total_neurons, int(p["dim"])), dtype=np.float32)
        cols = rng.integers(0, int(p["dim"]), size=(sub.total_neurons, int(p["fan"])))
        np.put_along_axis(proj, cols, rng.choice([-1.0, 1.0], size=cols.shape) / np.sqrt(int(p["fan"])), axis=1)
        self.projection = torch.as_tensor(proj)
        self.prediction = np.zeros(int(p["dim"]))
        self.skill = 0.0
        self.context_gain = 0.0

    def reset(self) -> None:
        self.collective.previous = None           # do not learn a transition across an episode boundary

    def sensory_vector(self, ctx: StepContext) -> np.ndarray:
        d = int(self.params["dim"])
        vector = ctx.scratch.get("fused_sensory", ctx.scratch.get("sensory_vector"))
        if vector is None:
            obs = ctx.observation.detach().numpy()[:d]
            vector = np.pad(obs, (0, d - obs.size))
        vector = np.asarray(vector, dtype=np.float64).ravel()[:d]
        return np.pad(vector, (0, d - vector.size))

    def begin_step(self, ctx: StepContext) -> None:
        p, mc = self.params, self.collective
        mc.expert_routing = self.enabled("expert_routing")
        mc.meta_learning = self.enabled("collective_meta_learning")
        mc.predictive_replay = self.enabled("predictive_replay")
        x = self.sensory_vector(ctx)
        if ctx.learning and p["train"]:
            mc.observe(x)
        if self.substrate.weave_steps % int(p["skill_every"]) == 0:
            self.skill = mc.skill()
        self.prediction = mc.predict(x)
        self.context_gain = float(p["max_context_gain"]) * float(np.clip(self.skill, 0.0, 1.0))
        ctx.predictions["micro_collective"] = {"next_sensory": self.prediction.copy(), "provenance": Provenance.INTERNAL_PREDICTION,
                                               "skill": self.skill, "entropy": mc.entropy}
        if self.context_gain > 0.0:
            ctx.apical_in += self.context_gain * (self.projection @ torch.as_tensor(self.prediction, dtype=torch.float32))

    def receipt(self) -> Dict[str, Any]:
        return dict(self.collective.receipt(), context_gain=self.context_gain)

    def state_dict(self) -> Dict[str, Any]:
        return {"snapshot": self.collective.snapshot(), "skill": float(self.skill)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.collective.load_snapshot(state["snapshot"])
        self.skill = float(state.get("skill", 0.0))

    def telemetry(self) -> Dict[str, Any]:
        r = self.collective.receipt()
        return {"experts": r["experts"], "top_k": r["top_k"], "updates": r["updates"], "validation_loss": r["validation_loss"],
                "persistence_loss": r["persistence_baseline_loss"], "skill": self.skill, "context_gain": self.context_gain,
                "rollbacks": r["rollbacks"], "routing_entropy": r["routing_entropy"], "parameters": r["parameters"],
                "expert_routing": self.collective.expert_routing, "meta_learning": self.collective.meta_learning,
                "predictive_replay": self.collective.predictive_replay, "trained": r["trained"]}
