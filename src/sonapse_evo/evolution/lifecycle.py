from __future__ import annotations

from enum import Enum
from typing import Dict, Any, List, Optional
import time


class CandidateState(str, Enum):
    PROPOSED = "PROPOSED"
    SCREENED = "SCREENED"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    ELIGIBLE = "ELIGIBLE"
    ACTIVE = "ACTIVE"
    MONITORING = "MONITORING"
    ROLLED_BACK = "ROLLED_BACK"
    ARCHIVED = "ARCHIVED"


class CandidateLifecycle:
    """Manages state transitions and lifecycle audits for neural graft candidates."""

    ALLOWED_TRANSITIONS = {
        CandidateState.PROPOSED: [CandidateState.SCREENED, CandidateState.ARCHIVED],
        CandidateState.SCREENED: [CandidateState.TRAINING, CandidateState.ARCHIVED],
        CandidateState.TRAINING: [CandidateState.EVALUATING, CandidateState.ARCHIVED],
        CandidateState.EVALUATING: [CandidateState.ELIGIBLE, CandidateState.ARCHIVED],
        CandidateState.ELIGIBLE: [CandidateState.ACTIVE, CandidateState.ARCHIVED],
        CandidateState.ACTIVE: [CandidateState.MONITORING, CandidateState.ROLLED_BACK],
        CandidateState.MONITORING: [CandidateState.ARCHIVED, CandidateState.ROLLED_BACK],
        CandidateState.ROLLED_BACK: [CandidateState.ARCHIVED],
        CandidateState.ARCHIVED: [],
    }

    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.state = CandidateState.PROPOSED
        self.transition_log: List[Dict[str, Any]] = [
            {"from": None, "to": CandidateState.PROPOSED, "timestamp": time.time(), "reason": "Created"}
        ]

    def transition_to(self, new_state: CandidateState, reason: str = "") -> None:
        allowed = self.ALLOWED_TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            raise ValueError(
                f"Illegal transition for {self.candidate_id}: {self.state} -> {new_state}"
            )
        old_state = self.state
        self.state = new_state
        self.transition_log.append({
            "from": old_state,
            "to": new_state,
            "timestamp": time.time(),
            "reason": reason,
        })
