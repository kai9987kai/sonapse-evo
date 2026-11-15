"""Ensemble world model: sensory forecasting, environment modelling, uncertainty and counterfactuals.

State features   f_t = [pool_t (8), obs_key_t (16)]        (pool mirrors the NEUROWEAVE pooled read-out)
Forward members  y_hat_m = W_m [f_{t-1}, a_{t-1}, 1]  ->  [f_t, reward_t]        m = 1..M  (M = 3 by default)
Learning         normalised LMS:  W_m += lr_m * (y - y_hat_m) x^T / (1 + |x|^2), with a seeded bootstrap mask
                 so members stay diverse.
Uncertainty      disagreement = mean over outputs of the standard deviation across members; published as
                 ``signals.uncertainty`` (it drives uncertainty gating, acetylcholine and affect).
Prediction       ``signals.prediction`` = predicted next mean |rate|.
Inverse model    a_hat_{t-1} = V [f_{t-1}, f_t, 1]   (action prediction).

``counterfactual(actions)`` and ``rollout(actions)`` evaluate alternative actions on the *model* only; they
never touch the substrate or the environment, and everything they return is labelled as a prediction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, Stage, StepContext, rng_state, set_rng_state

POOL_DIM, KEY_DIM, ACTION_DIM = 8, 16, 5
FEATURE_DIM = POOL_DIM + KEY_DIM


def pooled_features(substrate) -> np.ndarray:
    """Eight pooled read-out features of the substrate state."""
    r, cell = substrate.r.detach(), substrate.cell
    return np.array([
        float(r.mean()), float(r.abs().mean()), float(r.std()), float(torch.tanh(cell.soma_fast).mean()),
        float(torch.tanh(cell.soma_slow).mean()), float(torch.tanh(cell.soma_context).mean()), float(cell.trace.mean()),
        substrate.last_sparse_count / max(1, substrate.total_neurons),
    ], dtype=np.float64)


def _as_action(action) -> np.ndarray:
    out = np.zeros(ACTION_DIM)
    if action is not None:
        a = np.asarray(action, dtype=np.float64).ravel()[:ACTION_DIM]
        out[: a.size] = a
    return out


class PredictiveWorldModel(Mechanism):
    mechanism_id = "predictive_world_model"
    stage = Stage.LEARN

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"members": 3, "lr": 0.15, "bootstrap": 0.8, "inverse_lr": 0.1}

    def on_attach(self) -> None:
        p = self.params
        rng = np.random.default_rng(self.substrate.weave.seed + 60013)
        in_dim, out_dim = FEATURE_DIM + ACTION_DIM + 1, FEATURE_DIM + 1
        self.members = [rng.normal(0.0, 0.02 * (1 + 0.07 * i), size=(out_dim, in_dim)) for i in range(max(2, int(p["members"])))]
        self.inverse = np.zeros((ACTION_DIM, 2 * FEATURE_DIM + 1))
        self.rng = np.random.default_rng(self.substrate.weave.seed + 60017)
        self.error = np.full(len(self.members), 0.2)
        self.inverse_error = 0.0
        self.updates = 0
        self.reset()

    def reset(self) -> None:
        self.prev_features: Optional[np.ndarray] = None
        self.features = np.zeros(FEATURE_DIM)
        self.last_action = np.zeros(ACTION_DIM)
        self.next_prediction = np.zeros(FEATURE_DIM + 1)
        self.disagreement = 0.0

    def _features(self, ctx: StepContext) -> np.ndarray:
        key = ctx.observation.detach().numpy()[:KEY_DIM].astype(np.float64)
        return np.concatenate([pooled_features(self.substrate), np.pad(key, (0, KEY_DIM - key.size))])

    def _predict(self, features: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = np.concatenate([features, action, [1.0]])
        return np.stack([np.clip(w @ x, -3.0, 3.0) for w in self.members])       # (M, out)

    def end_step(self, ctx: StepContext) -> None:
        p, sig = self.params, ctx.signals
        features = self._features(ctx)
        action = _as_action(sig.feedback.issued_action)
        if self.prev_features is not None:
            x = np.concatenate([self.prev_features, action, [1.0]])
            y = np.concatenate([features, [float(np.clip(sig.reward, -1.0, 1.0))]])
            if ctx.learning:
                norm = 1.0 + float(x @ x)
                for i, w in enumerate(self.members):
                    err = y - np.clip(w @ x, -3.0, 3.0)
                    self.error[i] = 0.98 * self.error[i] + 0.02 * float(np.abs(err).mean())
                    if self.rng.random() < p["bootstrap"]:
                        w += p["lr"] * (1 + 0.07 * i) * np.outer(err, x) / norm
                z = np.concatenate([self.prev_features, features, [1.0]])
                inv_err = action - self.inverse @ z
                self.inverse += p["inverse_lr"] * np.outer(inv_err, z) / (1.0 + float(z @ z))
                self.inverse_error = 0.98 * self.inverse_error + 0.02 * float(np.abs(inv_err).mean())
                self.updates += 1
        self.prev_features, self.features, self.last_action = features, features, action
        preds = self._predict(features, action)
        self.next_prediction = preds.mean(axis=0)
        self.disagreement = float(preds.std(axis=0).mean())
        sig.prediction = float(self.next_prediction[1])
        sig.uncertainty = float(min(1.0, self.disagreement))
        ctx.predictions["world_model"] = {
            "next_features": self.next_prediction[:FEATURE_DIM].copy(), "next_reward": float(self.next_prediction[-1]),
            "disagreement": self.disagreement, "provenance": Provenance.INTERNAL_PREDICTION,
        }

    # -- model-only queries (never touch substrate or environment) -----------
    def counterfactual(self, actions: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """Predicted consequence of each alternative action from the current state."""
        out = []
        for action in actions:
            preds = self._predict(self.features, _as_action(action))
            mean = preds.mean(axis=0)
            out.append({"action": _as_action(action).tolist(), "predicted_reward": float(mean[-1]),
                        "predicted_features": mean[:FEATURE_DIM].tolist(), "disagreement": float(preds.std(axis=0).mean()),
                        "provenance": Provenance.INTERNAL_PREDICTION.value})
        return out

    def rollout(self, actions: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """Open-loop multi-step imagination: short-, medium- and longer-term predicted consequences."""
        features, rewards, spread = self.features.copy(), [], []
        for action in actions:
            preds = self._predict(features, _as_action(action))
            mean = preds.mean(axis=0)
            features = mean[:FEATURE_DIM]
            rewards.append(float(mean[-1]))
            spread.append(float(preds.std(axis=0).mean()))
        return {"predicted_rewards": rewards, "disagreement": spread, "final_features": features.tolist(),
                "provenance": Provenance.INTERNAL_PREDICTION.value}

    def predict_action(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        return self.inverse @ np.concatenate([before, after, [1.0]])

    def state_dict(self) -> Dict[str, Any]:
        return {"members": [torch.as_tensor(w) for w in self.members], "inverse": torch.as_tensor(self.inverse),
                "error": self.error.tolist(), "inverse_error": float(self.inverse_error), "updates": int(self.updates),
                "rng": rng_state(self.rng), "features": torch.as_tensor(self.features), "last_action": torch.as_tensor(self.last_action),
                "prev_features": None if self.prev_features is None else torch.as_tensor(self.prev_features),
                "next_prediction": torch.as_tensor(self.next_prediction), "disagreement": float(self.disagreement)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.members = [np.asarray(w, dtype=np.float64).copy() for w in state["members"]]
        self.inverse = np.asarray(state["inverse"], dtype=np.float64).copy()
        self.error = np.asarray(state.get("error", self.error), dtype=np.float64)
        self.inverse_error = float(state.get("inverse_error", 0.0))
        self.updates = int(state.get("updates", 0))
        set_rng_state(self.rng, state.get("rng"))
        if "features" in state:
            self.features = np.asarray(state["features"], dtype=np.float64).copy()
            self.last_action = np.asarray(state["last_action"], dtype=np.float64).copy()
            self.next_prediction = np.asarray(state["next_prediction"], dtype=np.float64).copy()
            self.disagreement = float(state.get("disagreement", 0.0))
            prev = state.get("prev_features")
            self.prev_features = None if prev is None else np.asarray(prev, dtype=np.float64).copy()

    def telemetry(self) -> Dict[str, Any]:
        return {"members": len(self.members), "updates": int(self.updates), "member_error": [float(e) for e in self.error],
                "disagreement": self.disagreement, "predicted_activity": float(self.next_prediction[1]),
                "predicted_reward": float(self.next_prediction[-1]), "inverse_model_error": float(self.inverse_error)}
