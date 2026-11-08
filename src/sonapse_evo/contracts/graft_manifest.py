from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple, Dict, Any
import hashlib
import json

GraftType = Literal["low_rank", "recurrent", "biological", "donor_adapter"]


@dataclass(frozen=True)
class GraftSpec:
    graft_id: str
    graft_type: GraftType
    source_neurons: Tuple[int, ...]
    target_neurons: Tuple[int, ...]
    rank: int = 8
    max_gain: float = 0.1
    ramp_steps: int = 50
    parent_digest: str = ""
    recurrent_state_dim: int = 16
    donor_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, total_neurons: int) -> None:
        if not self.graft_id:
            raise ValueError("graft_id must not be empty")
        if len(self.source_neurons) == 0:
            raise ValueError("source_neurons must not be empty")
        if len(self.target_neurons) == 0:
            raise ValueError("target_neurons must not be empty")
        if any(idx < 0 or idx >= total_neurons for idx in self.source_neurons):
            raise IndexError(f"source_neurons out of bounds for total_neurons={total_neurons}")
        if any(idx < 0 or idx >= total_neurons for idx in self.target_neurons):
            raise IndexError(f"target_neurons out of bounds for total_neurons={total_neurons}")
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.max_gain <= 0:
            raise ValueError("max_gain must be > 0")
        if self.ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")

    def compute_spec_digest(self) -> str:
        payload = {
            "graft_id": self.graft_id,
            "graft_type": self.graft_type,
            "source_neurons": list(self.source_neurons),
            "target_neurons": list(self.target_neurons),
            "rank": self.rank,
            "max_gain": self.max_gain,
            "ramp_steps": self.ramp_steps,
            "parent_digest": self.parent_digest,
            "recurrent_state_dim": self.recurrent_state_dim,
            "donor_id": self.donor_id,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
