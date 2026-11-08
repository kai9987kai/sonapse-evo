from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
import hashlib
import json
import time


@dataclass(frozen=True)
class EvaluationResult:
    candidate_id: str
    baseline_returns: Tuple[float, ...]
    candidate_returns: Tuple[float, ...]
    causal_tau: float
    recovery_ratio: float
    retention_passed: bool
    replay_verified: bool
    resource_passed: bool
    stability_passed: bool
    evidence_kind: str
    details: Dict[str, Any] = field(default_factory=dict)
    receipt_digest: str = ""

    def __post_init__(self):
        if not self.receipt_digest:
            payload = {
                "candidate_id": self.candidate_id,
                "baseline_returns": list(self.baseline_returns),
                "candidate_returns": list(self.candidate_returns),
                "causal_tau": self.causal_tau,
                "recovery_ratio": self.recovery_ratio,
                "retention_passed": self.retention_passed,
                "replay_verified": self.replay_verified,
                "resource_passed": self.resource_passed,
                "stability_passed": self.stability_passed,
                "evidence_kind": self.evidence_kind,
            }
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            object.__setattr__(self, "receipt_digest", digest)

    @property
    def is_eligible(self) -> bool:
        return (
            self.causal_tau > 0.0
            and self.retention_passed
            and self.replay_verified
            and self.resource_passed
            and self.stability_passed
        )


@dataclass(frozen=True)
class TransactionReceipt:
    transaction_id: str
    status: str  # "COMMITTED" or "ROLLED_BACK"
    graft_id: str
    parent_digest: str
    committed_digest: str
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
