import pytest

from sonapse_evo.contracts.evidence import EvidenceTier, is_promotable_evidence
from sonapse_evo.contracts.receipts import EvaluationResult
from sonapse_evo.evaluation.promotion import PromotionGate


def test_evidence_tier_classification():
    assert is_promotable_evidence(EvidenceTier.CONFIRMATORY_RESULT) is True
    assert is_promotable_evidence(EvidenceTier.PAIRED_EXPERIMENT) is True
    assert is_promotable_evidence(EvidenceTier.SIMULATED_ASSAY) is False
    assert is_promotable_evidence(EvidenceTier.SYNTHETIC_FIXTURE) is False
    assert is_promotable_evidence(EvidenceTier.SMOKE_TEST) is False

    # Permitted in test mode
    assert is_promotable_evidence(EvidenceTier.SYNTHETIC_FIXTURE, allow_test_tiers=True) is True


def test_promotion_gate_with_evidence_tier_enforcement():
    gate_strict = PromotionGate(require_promotable_tier=True)
    gate_permissive = PromotionGate(require_promotable_tier=False)

    # Fixture result
    res_synthetic = EvaluationResult(
        candidate_id="cand_fixture",
        baseline_returns=(10.0, 10.0),
        candidate_returns=(20.0, 20.0),
        causal_tau=0.25,
        recovery_ratio=0.85,
        retention_passed=True,
        replay_verified=True,
        resource_passed=True,
        stability_passed=True,
        evidence_kind=EvidenceTier.SYNTHETIC_FIXTURE.value,
    )

    passed_strict, checks_strict = gate_strict.evaluate_promotion(res_synthetic)
    assert passed_strict is False
    assert checks_strict["evidence_tier_gate"] is False

    passed_perm, checks_perm = gate_permissive.evaluate_promotion(res_synthetic)
    assert passed_perm is True
    assert checks_perm["evidence_tier_gate"] is True

    # Confirmatory result
    res_confirmatory = EvaluationResult(
        candidate_id="cand_confirmatory",
        baseline_returns=(10.0, 10.0),
        candidate_returns=(20.0, 20.0),
        causal_tau=0.25,
        recovery_ratio=0.85,
        retention_passed=True,
        replay_verified=True,
        resource_passed=True,
        stability_passed=True,
        evidence_kind=EvidenceTier.CONFIRMATORY_RESULT.value,
    )

    passed_strict_conf, checks_strict_conf = gate_strict.evaluate_promotion(res_confirmatory)
    assert passed_strict_conf is True
    assert checks_strict_conf["evidence_tier_gate"] is True
