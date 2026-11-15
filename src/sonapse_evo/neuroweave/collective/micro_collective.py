"""Micro-Collective: a small, local mixture of experts that predicts the next sensory state.

    route      p = softmax(R x + b) * gate ;  keep the top-k experts, renormalise      (top-1 without expert routing)
    predict    y_hat = sum_k w_k tanh(W_k x + c_k)
    learn      online self-supervised delta rule on (x_{t-1} -> x_t) transitions drawn from a bounded replay
               buffer (novelty-prioritised when predictive replay is on); each routed expert updates on its share
    router     R_k += lr * 0.015 * utility * x,  utility = clip(0.25 - expert_loss / dim, -0.75, 0.25)
    balance    b_k += balance_rate * sign(mean_load - load_k)                           (load balancing)
    validate   every fifth transition is held out; the best validation state is snapshotted; if validation
               loss degrades by more than 25 percent the model rolls back and halves its learning rate

Freshly initialised and trained only on local experience.  It is not, and does not contain, any external
checkpoint.  ``add_expert`` grows the collective beyond its initial size without disturbing existing
predictions: the new expert starts with zero weights and a routing gate of 0 that ramps up over training.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MC_FORMAT = "SONAPSE_MICRO_COLLECTIVE"
MC_VERSION = "1.0.0"


class MicroCollective:
    def __init__(self, dim: int = 16, n_experts: int = 4, top_k: int = 2, seed: int = 42, lr: float = 0.002,
                 replay_capacity: int = 384, validation_capacity: int = 96, expert_routing: bool = True,
                 meta_learning: bool = True, predictive_replay: bool = True, balance_rate: float = 0.001, gate_ramp: int = 32):
        self.dim, self.top_k, self.seed, self.lr = int(dim), int(top_k), int(seed), float(lr)
        self.replay_capacity, self.validation_capacity = int(replay_capacity), int(validation_capacity)
        self.expert_routing, self.meta_learning, self.predictive_replay = bool(expert_routing), bool(meta_learning), bool(predictive_replay)
        self.balance_rate, self.gate_ramp = float(balance_rate), int(gate_ramp)
        rng = np.random.default_rng(self.seed + 12031)
        k = int(n_experts)
        self.W = (rng.random((k, self.dim, self.dim)) - 0.5) * 0.035
        self.c = np.zeros((k, self.dim))
        self.R = (rng.random((k, self.dim)) - 0.5) * 0.04
        self.b = np.zeros(k)
        self.gate = np.ones(k)
        self.load = np.zeros(k)
        self.replay: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.validation: List[Tuple[np.ndarray, np.ndarray]] = []
        self.previous: Optional[np.ndarray] = None
        self.transitions = self.updates = self.router_updates = self.eval_count = self.rollbacks = 0
        self.best_loss, self.last_loss, self.best = float("inf"), None, None
        self.entropy = 0.0
        self.consecutive_rollbacks = 0
        self.max_consecutive_rollbacks = 3
        self.lr_floor = 1e-4

    @property
    def n_experts(self) -> int:
        return int(self.W.shape[0])

    @property
    def parameter_count(self) -> int:
        return int(self.W.size + self.c.size + self.R.size + self.b.size)

    # ---------------------------------------------------------------- routing
    def route(self, x: np.ndarray, track_load: bool = True) -> List[Tuple[int, float]]:
        logits = self.R @ x + self.b
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        self.entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum() / np.log(max(2, self.n_experts)))
        gated = probs * self.gate
        k = self.top_k if self.expert_routing else 1
        order = [i for i in np.argsort(-gated, kind="stable")[:k] if gated[i] > 0]
        total = float(gated[order].sum()) or 1.0
        chosen = [(int(i), float(gated[i] / total)) for i in order]
        if track_load:
            for i, _ in chosen:
                self.load[i] += 1
        return chosen

    def predict(self, x: np.ndarray, track_load: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros(self.dim)
        for i, w in self.route(x, track_load):
            out += w * np.tanh(self.W[i] @ x + self.c[i])
        return out

    # --------------------------------------------------------------- learning
    def observe(self, x: np.ndarray, train: bool = True) -> None:
        x = np.asarray(x, dtype=np.float64).copy()
        if self.previous is not None:
            novelty = float(np.abs(x - self.previous).mean())
            self.transitions += 1
            if self.transitions % 5 == 0:
                self.validation.append((self.previous, x))
                self.validation = self.validation[-self.validation_capacity:]
            else:
                self.replay.append((self.previous, x, novelty))
                self.replay = self.replay[-self.replay_capacity:]
                if train and len(self.replay) >= 8 and len(self.replay) % 4 == 0:
                    self.train_batch(4)
        self.previous = x

    def train_batch(self, epochs: int = 2) -> None:
        if not self.replay:
            return
        pool = sorted(self.replay, key=lambda item: -item[2]) if self.predictive_replay else self.replay
        for _ in range(epochs):
            for q in range(min(12, len(pool))):
                x, y, _ = pool[(self.updates * 7 + q * 13) % len(pool)]
                for i, w in self.route(x):
                    pred = np.tanh(self.W[i] @ x + self.c[i])
                    err = y - pred
                    g = err * (1.0 - pred ** 2) * w
                    self.W[i] = np.clip(self.W[i] + self.lr * np.outer(g, x), -1.5, 1.5)
                    self.c[i] = np.clip(self.c[i] + self.lr * g, -1.0, 1.0)
                    if self.expert_routing:
                        utility = float(np.clip(0.25 - float(err @ err) / self.dim, -0.75, 0.25))
                        self.R[i] = np.clip(self.R[i] + self.lr * 0.015 * utility * x, -1.0, 1.0)
                        self.b[i] = float(np.clip(self.b[i] + self.lr * 0.01 * utility, -1.0, 1.0))
                        self.router_updates += 1
            self.updates += 1
        if self.expert_routing:
            self.b = np.clip(self.b + self.balance_rate * np.sign(self.load.mean() - self.load), -1.0, 1.0)
        self.load[:] = 0
        self.gate = np.minimum(1.0, self.gate + 1.0 / max(1, self.gate_ramp))
        if self.meta_learning and len(self.validation) >= 8 and self.updates % 4 == 0:
            self.validate()

    def evaluate(self) -> float:
        if not self.validation:
            return float("inf")
        return float(np.mean([(self.predict(x) - y) ** 2 for x, y in self.validation]))

    def persistence_baseline_loss(self) -> float:
        if not self.validation:
            return float("inf")
        return float(np.mean([(x - y) ** 2 for x, y in self.validation]))

    def skill(self) -> float:
        """1 - loss / persistence loss on the held-out set (<= 0: no better than 'nothing changes')."""
        loss, base = self.evaluate(), self.persistence_baseline_loss()
        if not np.isfinite(loss) or not np.isfinite(base) or base <= 1e-12:
            return 0.0
        return float(1.0 - loss / base)

    def _weights(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "c": self.c.copy(), "R": self.R.copy(), "b": self.b.copy(), "gate": self.gate.copy()}

    def _evaluate_with(self, weights: Dict[str, np.ndarray]) -> float:
        saved = self._weights()
        try:
            for name, value in weights.items():
                setattr(self, name, value)
            return self.evaluate()
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def validate(self) -> float:
        """Keep the best state, measured on the CURRENT validation window, and roll back to it on degradation.

        The best snapshot is re-scored on the same window as the current weights, so a shift in the input
        distribution never triggers rollbacks by itself; consecutive rollbacks are capped and stop at the lr floor.
        """
        loss = self.evaluate()
        self.last_loss, self.eval_count = loss, self.eval_count + 1
        comparable = self.best is not None and self.best["W"].shape == self.W.shape
        best_now = self._evaluate_with(self.best) if comparable else float("inf")
        if loss <= best_now:
            self.best_loss, self.best = loss, self._weights()
            self.consecutive_rollbacks = 0
        elif (self.eval_count > 3 and loss > best_now * 1.25 and self.lr > self.lr_floor
              and self.consecutive_rollbacks < self.max_consecutive_rollbacks):
            for name, value in self.best.items():
                setattr(self, name, value.copy())
            self.best_loss = best_now
            self.lr = max(self.lr_floor, self.lr * 0.5)
            self.rollbacks += 1
            self.consecutive_rollbacks += 1
        else:
            self.best_loss = best_now
            self.consecutive_rollbacks = 0
        return loss

    def add_expert(self) -> int:
        """Append a zero-initialised expert whose routing gate starts at 0 (predictions are unchanged)."""
        self.W = np.concatenate([self.W, np.zeros((1, self.dim, self.dim))])
        self.c = np.concatenate([self.c, np.zeros((1, self.dim))])
        self.R = np.concatenate([self.R, np.zeros((1, self.dim))])
        self.b = np.append(self.b, 0.0)
        self.gate = np.append(self.gate, 0.0)
        self.load = np.append(self.load, 0.0)
        self.best = None
        return self.n_experts - 1

    # ------------------------------------------------------------ persistence
    def snapshot(self) -> Dict[str, Any]:
        return {"format": MC_FORMAT, "version": MC_VERSION, "dimension": self.dim, "expertCount": self.n_experts, "topK": self.top_k,
                "seed": self.seed, "learningRate": self.lr, "expertRouting": self.expert_routing, "metaLearning": self.meta_learning,
                "predictiveReplay": self.predictive_replay, "experts": {"W": self.W.tolist(), "c": self.c.tolist()},
                "router": self.R.tolist(), "routerBias": self.b.tolist(), "gate": self.gate.tolist(),
                "counters": {"transitions": self.transitions, "updates": self.updates, "routerUpdates": self.router_updates,
                             "evalCount": self.eval_count, "rollbacks": self.rollbacks, "consecutiveRollbacks": self.consecutive_rollbacks},
                "validationLoss": self.last_loss, "bestValidationLoss": self.best_loss if np.isfinite(self.best_loss) else None,
                "replay": [[x.tolist(), y.tolist(), n] for x, y, n in self.replay],
                "validation": [[x.tolist(), y.tolist()] for x, y in self.validation],
                "previous": None if self.previous is None else self.previous.tolist(), "trained": self.updates > 0,
                "best": None if self.best is None else {k: v.tolist() for k, v in self.best.items()}}

    def load_snapshot(self, snap: Dict[str, Any]) -> None:
        if snap.get("format") != MC_FORMAT or str(snap.get("version", "")).split(".")[0] != MC_VERSION.split(".")[0]:
            raise ValueError("incompatible Micro-Collective snapshot")
        self.dim, self.top_k, self.lr = int(snap["dimension"]), int(snap["topK"]), float(snap["learningRate"])
        self.expert_routing, self.meta_learning = bool(snap["expertRouting"]), bool(snap["metaLearning"])
        self.predictive_replay = bool(snap["predictiveReplay"])
        self.W, self.c = np.asarray(snap["experts"]["W"], dtype=np.float64), np.asarray(snap["experts"]["c"], dtype=np.float64)
        self.R, self.b = np.asarray(snap["router"], dtype=np.float64), np.asarray(snap["routerBias"], dtype=np.float64)
        self.gate = np.asarray(snap.get("gate", np.ones(len(self.b))), dtype=np.float64)
        if self.W.shape != (len(self.b), self.dim, self.dim) or self.R.shape != (len(self.b), self.dim):
            raise ValueError("Micro-Collective snapshot has inconsistent dimensions")
        self.load = np.zeros(len(self.b))
        cnt = snap.get("counters", {})
        self.transitions, self.updates = int(cnt.get("transitions", 0)), int(cnt.get("updates", 0))
        self.router_updates, self.eval_count, self.rollbacks = int(cnt.get("routerUpdates", 0)), int(cnt.get("evalCount", 0)), int(cnt.get("rollbacks", 0))
        self.consecutive_rollbacks = int(cnt.get("consecutiveRollbacks", 0))
        self.last_loss = snap.get("validationLoss")
        self.best_loss = float("inf") if snap.get("bestValidationLoss") is None else float(snap["bestValidationLoss"])
        self.replay = [(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), float(n)) for x, y, n in snap.get("replay", [])]
        self.validation = [(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)) for x, y in snap.get("validation", [])]
        self.previous = None if snap.get("previous") is None else np.asarray(snap["previous"], dtype=np.float64)
        best = snap.get("best")
        self.best = None if best is None else {k: np.asarray(v, dtype=np.float64) for k, v in best.items()}

    def receipt(self) -> Dict[str, Any]:
        """Training receipt: what was actually done and measured (no projected numbers)."""
        loss, base = self.evaluate(), self.persistence_baseline_loss()
        return {"format": MC_FORMAT, "version": MC_VERSION, "seed": self.seed, "experts": self.n_experts, "top_k": self.top_k,
                "parameters": self.parameter_count, "transitions_seen": self.transitions, "updates": self.updates,
                "router_updates": self.router_updates, "rollbacks": self.rollbacks, "learning_rate": self.lr,
                "replay_size": len(self.replay), "validation_size": len(self.validation),
                "validation_loss": loss if np.isfinite(loss) else None,
                "persistence_baseline_loss": base if np.isfinite(base) else None,
                "best_validation_loss": self.best_loss if np.isfinite(self.best_loss) else None,
                "skill_vs_persistence": self.skill(), "routing_entropy": self.entropy, "trained": self.updates > 0,
                "source": "freshly initialised; trained only on local experience"}


class PersistencePredictor:
    """Baseline: the next sensory vector equals the current one."""

    parameter_count = 0

    def __init__(self):
        self.pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self.previous: Optional[np.ndarray] = None

    def observe(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).copy()
        if self.previous is not None:
            self.pairs.append((self.previous, x))
        self.previous = x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def evaluate(self, pairs=None) -> float:
        pairs = self.pairs if pairs is None else pairs
        return float(np.mean([(x - y) ** 2 for x, y in pairs])) if pairs else float("inf")


class DenseMLPPredictor:
    """Parameter-matched dense baseline: one tanh hidden layer trained online by SGD on the same stream."""

    def __init__(self, dim: int = 16, parameter_budget: int = 1156, seed: int = 42, lr: float = 0.002):
        self.dim, self.lr = int(dim), float(lr)
        self.hidden = max(1, (int(parameter_budget) - self.dim) // (2 * self.dim + 1))
        rng = np.random.default_rng(seed + 7)
        self.W1 = rng.normal(0, 0.1, (self.hidden, self.dim))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 0.1, (self.dim, self.hidden))
        self.b2 = np.zeros(self.dim)
        self.previous: Optional[np.ndarray] = None
        self.updates = 0

    @property
    def parameter_count(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.W2 @ np.tanh(self.W1 @ np.asarray(x, dtype=np.float64) + self.b1) + self.b2)

    def observe(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).copy()
        if self.previous is not None:
            h = np.tanh(self.W1 @ self.previous + self.b1)
            out = np.tanh(self.W2 @ h + self.b2)
            d_out = (x - out) * (1 - out ** 2)
            d_h = (self.W2.T @ d_out) * (1 - h ** 2)
            self.W2 += self.lr * np.outer(d_out, h)
            self.b2 += self.lr * d_out
            self.W1 += self.lr * np.outer(d_h, self.previous)
            self.b1 += self.lr * d_h
            self.updates += 1
        self.previous = x

    def evaluate(self, pairs) -> float:
        return float(np.mean([(self.predict(x) - y) ** 2 for x, y in pairs])) if pairs else float("inf")
