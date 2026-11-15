"""Audit ledger: why every neuron or connection was created, modified or removed."""
from __future__ import annotations

from typing import Any, Dict, List


class StructuralLedger:
    def __init__(self, capacity: int = 4000):
        self.capacity = int(capacity)
        self.entries: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

    def record(self, step: int, action: str, target: str, reason: str, **details: Any) -> Dict[str, Any]:
        entry = {"step": int(step), "action": action, "target": target, "reason": reason, **details}
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity:]
        self.counts[action] = self.counts.get(action, 0) + 1
        return entry

    def summary(self) -> Dict[str, Any]:
        reasons: Dict[str, int] = {}
        for entry in self.entries:
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
        return {"counts": dict(self.counts), "reasons": reasons, "retained_entries": len(self.entries)}

    def state_dict(self) -> Dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries], "counts": dict(self.counts)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.entries = [dict(e) for e in state.get("entries", [])]
        self.counts = {k: int(v) for k, v in state.get("counts", {}).items()}
