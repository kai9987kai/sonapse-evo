from __future__ import annotations

import numpy as np
import torch


class StreamingCovarianceTracker:
    """Welford-based online streaming covariance tracker across neural activation streams."""

    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        self.count = 0
        self.mean = np.zeros(num_neurons, dtype=np.float64)
        self.M2 = np.zeros((num_neurons, num_neurons), dtype=np.float64)

    def update(self, rates: np.ndarray) -> None:
        """Updates mean and co-moment with an observation vector (num_neurons,)."""
        self.count += 1
        delta = rates.astype(np.float64) - self.mean
        self.mean += delta / self.count
        delta2 = rates.astype(np.float64) - self.mean
        self.M2 += np.outer(delta, delta2)

    def get_covariance(self) -> np.ndarray:
        """Returns the unbiased empirical sample covariance matrix."""
        if self.count < 2:
            return np.eye(self.num_neurons, dtype=np.float32) * 1e-4
        return (self.M2 / (self.count - 1)).astype(np.float32)

    def get_highest_covariance_ports(
        self,
        region_a_nodes: np.ndarray,
        region_b_nodes: np.ndarray,
    ) -> Tuple[int, int, float]:
        """Identifies node pair (u in A, v in B) with highest absolute covariance."""
        cov = self.get_covariance()
        sub_cov = np.abs(cov[np.ix_(region_a_nodes, region_b_nodes)])
        idx = np.unravel_index(np.argmax(sub_cov), sub_cov.shape)
        u = int(region_a_nodes[idx[0]])
        v = int(region_b_nodes[idx[1]])
        val = float(sub_cov[idx])
        return u, v, val
