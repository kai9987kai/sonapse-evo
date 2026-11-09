from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class OpportunitySignal:
    """Component signals for autonomous architecture discovery."""
    perf_plateau: float = 0.0        # High when progress stalls
    generalization_gap: float = 0.0  # High when OOD performance lags behind train
    capacity_pressure: float = 0.0   # High when representation saturates
    redundancy: float = 0.0          # High when many synapses are inactive
    interference: float = 0.0        # High when adaptation harms prior tasks


class DiscoveryOpportunityMonitor:
    """Monitors running network telemetry to detect architectural optimization opportunities.
    
    Uses an exponentially weighted moving average with standard bias correction:
        z_t = beta * z_{t-1} + (1 - beta) * x_t
        z_hat_t = z_t / (1 - beta^t)
        
    Triggers discovery mode only when persistent pressure exceeds threshold,
    or during explicit scheduled exploration. Healthy networks are not forced to grow.
    """

    def __init__(
        self,
        beta: float = 0.90,
        trigger_threshold: float = 0.45,
        persistence_steps: int = 3,
        scheduled_exploration: bool = False,
    ):
        self.beta = beta
        self.trigger_threshold = trigger_threshold
        self.persistence_steps = persistence_steps
        self.scheduled_exploration = scheduled_exploration

        # State tracking
        self.smoothed_signals = np.zeros(5, dtype=np.float32)
        self.step_count: int = 0
        self.recent_returns: List[float] = []
        self.consecutive_triggers: int = 0
        self.history: List[Dict[str, float]] = []

    def update(
        self,
        return_train: float,
        return_ood: float,
        jacobian_norm: float,
        density: float,
        prior_retention_loss: float = 0.0,
    ) -> bool:
        """Updates internal statistics with fresh episode metrics.
        
        Returns True if autonomous discovery should be triggered.
        """
        self.step_count += 1
        self.recent_returns.append(return_train)
        if len(self.recent_returns) > 10:
            self.recent_returns.pop(0)

        # 1. Performance plateau (stagnant return variance over last episodes)
        plateau = 0.0
        if len(self.recent_returns) >= 5:
            r_std = float(np.std(self.recent_returns))
            # Plateau signal high if std is small (< 1.0) while return is not optimal
            plateau = float(np.clip(1.0 - (r_std / 3.0), 0.0, 1.0))

        # 2. Generalization gap
        gap = float(np.clip((return_train - return_ood) / max(1.0, abs(return_train)), 0.0, 1.0))

        # 3. Capacity pressure (spectral norm approaching unstable limit or near zero)
        cap = float(np.clip(abs(jacobian_norm - 0.70) / 0.50, 0.0, 1.0))

        # 4. Redundancy (sparsity vs density balance)
        red = float(np.clip(1.0 - density / 0.15, 0.0, 1.0))

        # 5. Task interference (forgetting prior skills)
        interf = float(np.clip(prior_retention_loss / 10.0, 0.0, 1.0))

        raw = np.array([plateau, gap, cap, red, interf], dtype=np.float32)

        # Exponential smoothing: z_t = beta * z_{t-1} + (1 - beta) * raw
        self.smoothed_signals = self.beta * self.smoothed_signals + (1.0 - self.beta) * raw
        bias_correction = 1.0 - (self.beta ** self.step_count)
        corrected = self.smoothed_signals / max(1e-4, bias_correction)

        # Weighted aggregate opportunity score:
        # Plateau: 25%, Gap: 30%, Capacity: 20%, Redundancy: 15%, Interference: 10%
        weights = np.array([0.25, 0.30, 0.20, 0.15, 0.10], dtype=np.float32)
        score = float(np.dot(corrected, weights))

        # Persistent threshold trigger check
        is_triggered = score >= self.trigger_threshold
        if is_triggered:
            self.consecutive_triggers += 1
        else:
            self.consecutive_triggers = 0

        self.history.append({
            "score": score,
            "plateau": float(corrected[0]),
            "gap": float(corrected[1]),
            "capacity": float(corrected[2]),
            "redundancy": float(corrected[3]),
            "interference": float(corrected[4]),
            "triggered": is_triggered,
        })

        if self.scheduled_exploration:
            return True

        return self.consecutive_triggers >= self.persistence_steps

    @property
    def current_score(self) -> float:
        weights = np.array([0.25, 0.30, 0.20, 0.15, 0.10], dtype=np.float32)
        bias_correction = 1.0 - (self.beta ** max(1, self.step_count))
        corrected = self.smoothed_signals / max(1e-4, bias_correction)
        return float(np.dot(corrected, weights))
