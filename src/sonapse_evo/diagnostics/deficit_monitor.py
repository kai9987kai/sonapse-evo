from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np


class DeficitMonitor:
    """Detects persistent functional deficits without access to lesion ground truth."""

    def __init__(
        self,
        beta: float = 0.90,
        threshold: float = 0.65,
        persistence_steps: int = 5,
        weights: Optional[np.ndarray] = None,
    ):
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.persistence_steps = int(persistence_steps)

        # Default weights for [e_pred, e_policy, u_unc, c_rep, r_task, s_stab]
        if weights is None:
            self.weights = np.array([0.20, 0.25, 0.15, 0.15, 0.20, 0.05], dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)
            self.weights /= np.sum(self.weights)

        self.smoothed_d: Optional[np.ndarray] = None
        self.history: List[float] = []
        self.consecutive_crossings = 0

    def reset(self):
        self.smoothed_d = None
        self.history.clear()
        self.consecutive_crossings = 0

    def update(
        self,
        prediction_error: float,
        policy_error: float,
        uncertainty: float,
        representation_collapse: float,
        task_deficit: float,
        instability: float,
    ) -> float:
        """Updates diagnostic vector d_t and returns scalar deficit score D_t."""
        raw_d = np.array([
            np.clip(prediction_error, 0.0, 1.0),
            np.clip(policy_error, 0.0, 1.0),
            np.clip(uncertainty, 0.0, 1.0),
            np.clip(representation_collapse, 0.0, 1.0),
            np.clip(task_deficit, 0.0, 1.0),
            np.clip(instability, 0.0, 1.0),
        ], dtype=np.float32)

        if self.smoothed_d is None:
            self.smoothed_d = raw_d
        else:
            self.smoothed_d = self.beta * self.smoothed_d + (1.0 - self.beta) * raw_d

        d_score = float(np.dot(self.weights, self.smoothed_d))
        self.history.append(d_score)

        if d_score >= self.threshold:
            self.consecutive_crossings += 1
        else:
            self.consecutive_crossings = max(0, self.consecutive_crossings - 1)

        return d_score

    def is_repair_justified(self) -> bool:
        """Returns True only when the deficit is severe and persistent."""
        return self.consecutive_crossings >= self.persistence_steps

    def quick_no_growth_probe(self, baseline_loss: float, perturbed_loss: float) -> bool:
        """Checks if a simple no-growth adaptation solves the issue.
        
        If an inexpensive parameter tweak reduces loss by > 30%, no structural graft is needed.
        """
        if baseline_loss <= 1e-4:
            return True  # no repair needed
        relative_gain = (baseline_loss - perturbed_loss) / (baseline_loss + 1e-6)
        return relative_gain > 0.30
