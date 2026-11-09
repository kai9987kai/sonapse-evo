from __future__ import annotations

from typing import Dict, Any, Tuple
from ..contracts.receipts import EvaluationResult
from ..contracts.evidence import EvidenceTier, is_promotable_evidence


class PromotionGate:
    """Verifies that an evaluated candidate satisfies all rigorous deployment gates."""

    def __init__(
        self,
        min_recovery_ratio: float = 0.25,
        min_causal_tau: float = 0.10,
        max_jacobian_norm: float = 1.10,
        require_promotable_tier: bool = False,
    ):
        self.min_recovery = float(min_recovery_ratio)
        self.min_tau = float(min_causal_tau)
        self.max_j = float(max_jacobian_norm)
        self.require_promotable_tier = require_promotable_tier

    def evaluate_promotion(self, res: EvaluationResult) -> Tuple[bool, Dict[str, bool]]:
        """Evaluates whether all gates pass."""
        evidence_valid = is_promotable_evidence(
            res.evidence_kind,
            allow_test_tiers=not self.require_promotable_tier,
        )
        checks = {
            "recovery_gate": res.recovery_ratio >= self.min_recovery,
            "causal_tau_gate": res.causal_tau >= self.min_tau,
            "retention_gate": bool(res.retention_passed),
            "replay_gate": bool(res.replay_verified),
            "resource_gate": bool(res.resource_passed),
            "stability_gate": bool(res.stability_passed),
            "evidence_tier_gate": bool(evidence_valid),
        }
        all_passed = all(checks.values())
        return all_passed, checks
