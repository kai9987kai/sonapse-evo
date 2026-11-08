from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ..contracts.graft_manifest import GraftSpec
from ..contracts.receipts import EvaluationResult


class CandidateSelector:
    """Selects optimal neural graft candidate via multi-objective utility scoring and successive halving."""

    def __init__(
        self,
        param_budget: int = 2048,
        compute_budget: int = 4096,
        lambda_param: float = 0.10,
        lambda_compute: float = 0.05,
        lambda_forgetting: float = 0.30,
        lambda_stability: float = 0.20,
    ):
        self.param_budget = int(param_budget)
        self.compute_budget = int(compute_budget)
        self.lambda_p = float(lambda_param)
        self.lambda_c = float(lambda_compute)
        self.lambda_f = float(lambda_forgetting)
        self.lambda_s = float(lambda_stability)

    def calculate_utility(
        self,
        eval_result: EvaluationResult,
        param_count: int,
        compute_ops: int,
        forgetting: float,
        stability_penalty: float,
    ) -> float:
        """U(g) = Delta R_g - lambda_p (P_g/P_b) - lambda_c (C_g/C_b) - lambda_f F_g - lambda_s S_g"""
        delta_r = eval_result.causal_tau
        p_cost = self.lambda_p * (float(param_count) / float(self.param_budget))
        c_cost = self.lambda_c * (float(compute_ops) / float(self.compute_budget))
        f_cost = self.lambda_f * float(forgetting)
        s_cost = self.lambda_s * float(stability_penalty)

        utility = delta_r - p_cost - c_cost - f_cost - s_cost
        return float(utility)

    def rank_evaluated_candidates(
        self,
        candidate_evals: List[Tuple[GraftSpec, EvaluationResult, int]],
    ) -> List[Tuple[GraftSpec, EvaluationResult, float]]:
        """Ranks candidates by utility and enforces eligibility criteria."""
        ranked = []
        for spec, res, param_count in candidate_evals:
            if not res.is_eligible:
                continue

            # Estimate compute ops: 2 * rank * (|S| + |T|)
            compute_ops = 2 * spec.rank * (len(spec.source_neurons) + len(spec.target_neurons))
            forgetting = 0.0 if res.retention_passed else 0.5
            stability_pen = 0.0 if res.stability_passed else 0.5

            utility = self.calculate_utility(
                res, param_count, compute_ops, forgetting, stability_pen
            )
            ranked.append((spec, res, utility))

        # Sort descending by utility
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked
