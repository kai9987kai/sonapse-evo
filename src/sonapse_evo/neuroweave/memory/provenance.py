"""Memory provenance: every stored item knows where it came from.

Rule enforced in code: only information that was actually *observed*
(``DIRECT_SENSORY`` or ``EXTERNAL_SUPPLIED``) can ever be marked ``verified``.
A prediction or a simulated item that later turns out to match reality is not
relabelled; ``verify_against`` creates a **new** observed record that points
back at the prediction it confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np

from ..contracts import Provenance


class ProvenanceError(ValueError):
    """Raised when non-observed information would be treated as a verified observation."""


@dataclass
class MemoryRecord:
    record_id: int
    key: np.ndarray
    value: np.ndarray
    provenance: Provenance
    confidence: float = 0.5
    reward: float = 0.0
    step: int = 0
    region: Optional[int] = None
    context: Optional[np.ndarray] = None
    verified: bool = False
    uses: int = 0
    priority: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.provenance = Provenance(self.provenance)
        self.key = np.asarray(self.key, dtype=np.float32).ravel()
        self.value = np.asarray(self.value, dtype=np.float32).ravel()
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.verified and not self.provenance.is_observation:
            raise ProvenanceError(f"a {self.provenance.value} record cannot be a verified observation")

    def mark_verified(self) -> None:
        if not self.provenance.is_observation:
            raise ProvenanceError(f"a {self.provenance.value} record cannot be marked verified")
        self.verified = True

    def to_dict(self) -> Dict[str, Any]:
        return {"record_id": int(self.record_id), "key": self.key.tolist(), "value": self.value.tolist(),
                "provenance": self.provenance.value, "confidence": float(self.confidence), "reward": float(self.reward),
                "step": int(self.step), "region": self.region, "verified": bool(self.verified), "uses": int(self.uses),
                "context": None if self.context is None else np.asarray(self.context).tolist(),
                "priority": float(self.priority), "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        ctx = data.get("context")
        return cls(record_id=int(data["record_id"]), key=np.asarray(data["key"]), value=np.asarray(data["value"]),
                   provenance=Provenance(data["provenance"]), confidence=float(data.get("confidence", 0.5)),
                   reward=float(data.get("reward", 0.0)), step=int(data.get("step", 0)), region=data.get("region"),
                   context=None if ctx is None else np.asarray(ctx, dtype=np.float32), verified=bool(data.get("verified", False)),
                   uses=int(data.get("uses", 0)), priority=float(data.get("priority", 0.0)), meta=dict(data.get("meta", {})))


def verify_against(record: MemoryRecord, observation: np.ndarray, tolerance: float, new_id: int,
                   step: int = 0) -> Optional[MemoryRecord]:
    """If ``observation`` confirms ``record`` (RMS error <= tolerance) return a NEW verified observed record."""
    observation = np.asarray(observation, dtype=np.float32).ravel()
    n = min(observation.size, record.value.size)
    if n == 0:
        return None
    rms = float(np.sqrt(np.mean((record.value[:n] - observation[:n]) ** 2)))
    if rms > tolerance:
        return None
    return MemoryRecord(record_id=new_id, key=record.key.copy(), value=observation.copy(), provenance=Provenance.DIRECT_SENSORY,
                        confidence=max(record.confidence, 0.6), reward=record.reward, step=step, region=record.region,
                        verified=True, meta={"confirmed_prediction": int(record.record_id), "rms_error": rms})


def provenance_counts(records: Iterable[MemoryRecord]) -> Dict[str, int]:
    counts = {p.value: 0 for p in Provenance}
    for record in records:
        counts[record.provenance.value] += 1
    return counts
