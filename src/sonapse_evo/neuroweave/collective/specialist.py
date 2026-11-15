"""Supermix-inspired specialist: a small, separately trained model grafted onto the connectome.

``SpecialistModel`` is a freshly initialised two-layer network with its own optimiser and replay buffer.
No external checkpoint exists or is assumed; nothing here is a port of another model's weights.  It learns
from live experience to predict the next sensory vector and the reward from the current sensory vector
and the pooled per-region rates of the connectome.

``SpecialistGraft`` is the explicit, removable interface between that model and the main controller:

* reads   pooled region rates and the sensory vector;
* writes  ``ctx.predictions["specialist"]`` (always) and an apical context current whose gain starts at
          exactly 0 and rises only while held-out loss beats the persistence baseline (bounded by ``max_gain``);
* transfers knowledge only through ``export_knowledge`` / ``import_knowledge``;
* can be evaluated on its own with ``contribution_report``.

Detaching the graft leaves the connectome exactly as it was: the specialist never writes substrate state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..contracts import Mechanism, Provenance, Stage, StepContext, rng_state, set_rng_state

_REGIONS = 10


class SpecialistModel:
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 32, seed: int = 0, lr: float = 0.01,
                 capacity: int = 512, batch: int = 16):
        gen = torch.Generator().manual_seed(int(seed))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))
        with torch.no_grad():
            for layer in (self.net[0], self.net[2]):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) * (1.0 / np.sqrt(layer.in_features)))
                layer.bias.zero_()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(int(seed) + 1)
        self.capacity, self.batch = int(capacity), int(batch)
        self.train_set: List[Tuple[np.ndarray, np.ndarray]] = []
        self.val_set: List[Tuple[np.ndarray, np.ndarray]] = []
        self.seen = self.steps = 0
        self.seed = int(seed)

    @property
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def observe(self, features: np.ndarray, target: np.ndarray) -> None:
        pair = (np.asarray(features, dtype=np.float32).copy(), np.asarray(target, dtype=np.float32).copy())
        self.seen += 1
        if self.seen % 5 == 0:
            self.val_set = (self.val_set + [pair])[-128:]
        else:
            self.train_set = (self.train_set + [pair])[-self.capacity:]

    def train_step(self) -> Optional[float]:
        if len(self.train_set) < self.batch:
            return None
        idx = self.rng.choice(len(self.train_set), size=self.batch, replace=False)
        x = torch.as_tensor(np.stack([self.train_set[i][0] for i in idx]))
        y = torch.as_tensor(np.stack([self.train_set[i][1] for i in idx]))
        with torch.enable_grad():               # the specialist trains itself even if the caller disabled autograd
            loss = torch.mean((self.net(x) - y) ** 2)
            self.opt.zero_grad()
            loss.backward()
        self.opt.step()
        self.steps += 1
        return float(loss.item())

    def predict(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(features, dtype=np.float32))).numpy()

    def evaluate(self) -> float:
        if not self.val_set:
            return float("inf")
        with torch.no_grad():
            x = torch.as_tensor(np.stack([p[0] for p in self.val_set]))
            y = torch.as_tensor(np.stack([p[1] for p in self.val_set]))
            return float(torch.mean((self.net(x) - y) ** 2))

    def snapshot(self) -> Dict[str, Any]:
        def pairs(items):
            return [[torch.as_tensor(x), torch.as_tensor(y)] for x, y in items]

        return {"weights": {k: v.clone() for k, v in self.net.state_dict().items()}, "seen": self.seen, "steps": self.steps,
                "optimiser": self.opt.state_dict(), "rng": rng_state(self.rng), "train_set": pairs(self.train_set),
                "val_set": pairs(self.val_set)}

    def load_snapshot(self, snap: Dict[str, Any]) -> None:
        self.net.load_state_dict({k: torch.as_tensor(v) for k, v in snap["weights"].items()})
        self.seen, self.steps = int(snap.get("seen", 0)), int(snap.get("steps", 0))
        if "optimiser" in snap:
            opt = snap["optimiser"]
            opt["state"] = {int(k): {kk: (torch.as_tensor(vv) if isinstance(vv, (torch.Tensor, list)) else vv) for kk, vv in v.items()}
                            for k, v in opt["state"].items()}
            self.opt.load_state_dict(opt)
        set_rng_state(self.rng, snap.get("rng"))
        for name in ("train_set", "val_set"):
            if name in snap:
                setattr(self, name, [(np.asarray(x, dtype=np.float32).copy(), np.asarray(y, dtype=np.float32).copy()) for x, y in snap[name]])

    def receipt(self) -> Dict[str, Any]:
        loss = self.evaluate()
        return {"parameters": self.parameter_count, "samples_seen": self.seen, "optimiser_steps": self.steps, "seed": self.seed,
                "validation_loss": loss if np.isfinite(loss) else None,
                "source": "freshly initialised specialist; no external checkpoint"}


class SpecialistGraft(Mechanism):
    mechanism_id = "specialist_graft"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"dim": 16, "hidden": 32, "lr": 0.01, "train_every": 2, "max_gain": 0.2, "fan": 4, "skill_every": 16, "train": True}

    def on_attach(self) -> None:
        sub, p = self.substrate, self.params
        d = int(p["dim"])
        self.model = SpecialistModel(d + _REGIONS, d + 1, hidden=int(p["hidden"]), seed=sub.weave.seed + 808, lr=float(p["lr"]))
        rng = np.random.default_rng(sub.weave.seed + 909)
        proj = np.zeros((sub.total_neurons, d), dtype=np.float32)
        cols = rng.integers(0, d, size=(sub.total_neurons, int(p["fan"])))
        np.put_along_axis(proj, cols, rng.choice([-1.0, 1.0], size=cols.shape) / np.sqrt(int(p["fan"])), axis=1)
        self.projection = torch.as_tensor(proj)
        self.regions = torch.unique(sub.region_ids).tolist()[:_REGIONS]
        self.gain = self.skill = 0.0
        self.persistence_loss = float("inf")
        self.injected = 0.0
        self.prediction = np.zeros(d + 1)
        self.reset()

    def reset(self) -> None:
        self.prev_features: Optional[np.ndarray] = None

    def _features(self, ctx: StepContext) -> Tuple[np.ndarray, np.ndarray]:
        d, sub = int(self.params["dim"]), self.substrate
        vector = ctx.scratch.get("fused_sensory", ctx.scratch.get("sensory_vector"))
        if vector is None:
            obs = ctx.observation.detach().numpy()[:d]
            vector = np.pad(obs, (0, d - obs.size))
        sensory = np.asarray(vector, dtype=np.float32).ravel()[:d]
        sensory = np.pad(sensory, (0, d - sensory.size))
        r = sub.r.detach()
        pooled = np.zeros(_REGIONS, dtype=np.float32)
        for i, g in enumerate(self.regions):
            pooled[i] = float(r[sub.region_ids == g].mean())
        return np.concatenate([sensory, pooled]), sensory

    def begin_step(self, ctx: StepContext) -> None:
        p = self.params
        features, sensory = self._features(ctx)
        if ctx.learning and p["train"] and self.prev_features is not None:
            target = np.concatenate([sensory, [float(np.clip(ctx.signals.reward, -1.0, 1.0))]]).astype(np.float32)
            self.model.observe(self.prev_features, target)
            if self.substrate.weave_steps % int(p["train_every"]) == 0:
                self.model.train_step()
        if self.substrate.weave_steps % int(p["skill_every"]) == 0 and self.model.val_set:
            d = int(p["dim"])
            self.persistence_loss = float(np.mean([(np.concatenate([f[:d], [0.0]]) - t) ** 2 for f, t in self.model.val_set]))
            loss = self.model.evaluate()
            self.skill = float(1.0 - loss / self.persistence_loss) if self.persistence_loss > 1e-12 else 0.0
        self.gain = float(p["max_gain"]) * float(np.clip(self.skill, 0.0, 1.0))
        self.prediction = self.model.predict(features)
        ctx.predictions["specialist"] = {"next_sensory": self.prediction[:-1].copy(), "reward": float(self.prediction[-1]),
                                         "provenance": Provenance.INTERNAL_PREDICTION, "skill": self.skill, "gain": self.gain}
        if self.gain > 0.0:
            current = self.gain * (self.projection @ torch.as_tensor(self.prediction[:-1], dtype=torch.float32))
            ctx.apical_in += current
            self.injected = 0.95 * self.injected + 0.05 * float(current.abs().mean())
        self.prev_features = features

    # -- explicit knowledge-transfer interface -------------------------------
    def export_knowledge(self) -> Dict[str, Any]:
        return {"format": "SONAPSE_SPECIALIST", "version": "1.0", "snapshot": self.model.snapshot(), "receipt": self.model.receipt()}

    def import_knowledge(self, knowledge: Dict[str, Any]) -> None:
        if knowledge.get("format") != "SONAPSE_SPECIALIST":
            raise ValueError("unknown specialist knowledge package")
        self.model.load_snapshot(knowledge["snapshot"])

    def contribution_report(self) -> Dict[str, Any]:
        loss = self.model.evaluate()
        return {"validation_loss": loss if np.isfinite(loss) else None,
                "persistence_loss": self.persistence_loss if np.isfinite(self.persistence_loss) else None,
                "skill_vs_persistence": self.skill, "gain": self.gain, "mean_injected_current": self.injected,
                "optimiser_steps": self.model.steps, "samples_seen": self.model.seen, "removable": True}

    def state_dict(self) -> Dict[str, Any]:
        return {"model": self.model.snapshot(), "skill": float(self.skill), "gain": float(self.gain),
                "persistence_loss": float(self.persistence_loss) if np.isfinite(self.persistence_loss) else None,
                "injected": float(self.injected),
                "prev_features": None if self.prev_features is None else torch.as_tensor(self.prev_features)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_snapshot(state["model"])
        self.skill, self.gain = float(state.get("skill", 0.0)), float(state.get("gain", 0.0))
        loss = state.get("persistence_loss")
        self.persistence_loss = float("inf") if loss is None else float(loss)
        self.injected = float(state.get("injected", 0.0))
        prev = state.get("prev_features")
        self.prev_features = None if prev is None else np.asarray(prev, dtype=np.float32).copy()

    def telemetry(self) -> Dict[str, Any]:
        return self.contribution_report()
