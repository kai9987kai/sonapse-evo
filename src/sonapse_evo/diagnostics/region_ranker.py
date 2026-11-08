from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import torch

from ..substrate.topology_state import TopologyState


class RegionRanker:
    """Ranks anatomical neural regions based on deficit severity and activity distortion."""

    def __init__(self, topology: TopologyState):
        self.topology = topology

    def rank_regions(
        self,
        current_rates: np.ndarray,
        reference_rates: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """Returns sorted list of (region_id, deficit_score), highest deficit first."""
        scores = []
        for r in range(self.topology.num_regions):
            nodes = self.topology.get_region_nodes(r)
            r_curr = current_rates[nodes]

            if reference_rates is not None:
                r_ref = reference_rates[nodes]
                # Mean absolute difference in rate profile
                diff = float(np.mean(np.abs(r_curr - r_ref)))
            else:
                # In absence of reference, evaluate rate starvation or saturation
                under_active = np.mean(r_curr < 0.02)
                over_active = np.mean(r_curr > 0.90)
                diff = float(under_active + over_active)

            # Representation variance penalty
            var = float(np.var(r_curr))
            var_penalty = max(0.0, 0.05 - var)

            total_score = diff + 2.0 * var_penalty
            scores.append((r, total_score))

        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
