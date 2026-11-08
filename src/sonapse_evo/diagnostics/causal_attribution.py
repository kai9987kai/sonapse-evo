from __future__ import annotations

from typing import Callable, List, Tuple, Dict, Any
import numpy as np


class CausalAttributionEngine:
    """Evaluates the interventional causal treatment effect of candidate grafts.
    
    tau_hat = 1/m sum_{s=1}^m [R_s(do(g=on)) - R_s(do(g=off))]
    """

    def __init__(self, num_paired_rollouts: int = 4):
        self.num_paired_rollouts = int(num_paired_rollouts)

    def evaluate_interventional_effect(
        self,
        rollout_fn: Callable[[int, bool], float],
        base_seed: int = 1000,
    ) -> Tuple[float, List[float], List[float]]:
        """Executes paired rollouts under identical counterfactual random seeds.
        
        Args:
            rollout_fn: Function (seed, graft_enabled) -> scalar_return
            base_seed: Base integer seed for reproducibility
            
        Returns:
            (tau_hat, on_returns, off_returns)
        """
        on_returns = []
        off_returns = []

        for s in range(self.num_paired_rollouts):
            eval_seed = base_seed + s * 997

            # do(g = off)
            r_off = rollout_fn(eval_seed, False)
            off_returns.append(float(r_off))

            # do(g = on)
            r_on = rollout_fn(eval_seed, True)
            on_returns.append(float(r_on))

        diffs = np.array(on_returns) - np.array(off_returns)
        tau_hat = float(np.mean(diffs))
        return tau_hat, on_returns, off_returns
