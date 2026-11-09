from __future__ import annotations

from enum import Enum
from typing import Union


class EvidenceTier(str, Enum):
    """Hierarchical evidence taxonomy for scientific integrity.
    
    Tiers:
    - SYNTHETIC_FIXTURE: Handcrafted or mock evaluation metrics for unit testing.
    - SMOKE_TEST: Fast integration sanity checks (<5 seconds) verifying basic pipeline wiring.
    - SIMULATED_ASSAY: Unpaired stochastic episode rollouts in a simulated task.
    - PAIRED_EXPERIMENT: Rigorous paired-seed counterfactual trials with matched control comparisons.
    - CONFIRMATORY_RESULT: Multi-seed, out-of-distribution replicated evaluation with bootstrap CI.
    """
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    SMOKE_TEST = "SMOKE_TEST"
    SIMULATED_ASSAY = "SIMULATED_ASSAY"
    PAIRED_EXPERIMENT = "PAIRED_EXPERIMENT"
    CONFIRMATORY_RESULT = "CONFIRMATORY_RESULT"


# Production promotion requires at least PAIRED_EXPERIMENT
PROMOTABLE_TIERS = {
    EvidenceTier.PAIRED_EXPERIMENT.value,
    EvidenceTier.CONFIRMATORY_RESULT.value,
}

# Permitted tiers during integration testing and CI smoke runs
TEST_TIERS = {
    EvidenceTier.SYNTHETIC_FIXTURE.value,
    EvidenceTier.SMOKE_TEST.value,
    EvidenceTier.SIMULATED_ASSAY.value,
}


def is_promotable_evidence(tier: Union[EvidenceTier, str], allow_test_tiers: bool = False) -> bool:
    """Check whether an evidence tier satisfies the promotion gate requirements."""
    val = tier.value if isinstance(tier, EvidenceTier) else str(tier)
    if allow_test_tiers:
        return True
    return val in PROMOTABLE_TIERS
