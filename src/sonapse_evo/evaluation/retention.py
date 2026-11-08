from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class RetentionEvaluator:
    """Evaluates retention and catastrophic forgetting across historical tasks."""

    def __init__(self, max_allowed_forgetting: float = 0.15):
        self.max_allowed_forgetting = float(max_allowed_forgetting)

    def compute_forgetting(
        self,
        before_returns: List[float],
        after_returns: List[float],
    ) -> Tuple[float, bool]:
        """Computes F = 1/K sum_{k=1}^K [R_k^{before} - R_k^{after}]_+ and checks gate."""
        if len(before_returns) != len(after_returns) or len(before_returns) == 0:
            return 0.0, True

        drops = []
        for r_before, r_after in zip(before_returns, after_returns):
            drop = max(0.0, float(r_before - r_after))
            drops.append(drop)

        mean_forgetting = float(np.mean(drops))
        passed = mean_forgetting <= self.max_allowed_forgetting
        return mean_forgetting, passed
