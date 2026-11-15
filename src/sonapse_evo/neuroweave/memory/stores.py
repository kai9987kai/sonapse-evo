"""Bounded memory stores with sparse distributed indexing.

``SparseIndex``   a fixed seeded random projection followed by top-k winners turns a key into a sparse
                  binary code; an inverted index (active bit -> record ids) returns candidates whose codes
                  overlap the query, so retrieval inspects a small candidate set instead of every record.
``MemoryStore``   capacity-bounded store with priority eviction, cosine retrieval over candidates,
                  contextual retrieval, confidence decay, value downscaling and forgetting.
``SemanticStore`` online prototypes (running centroids) formed by consolidating similar episodes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import Provenance
from .provenance import MemoryRecord


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(a[:n] @ b[:n] / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-9))


class SparseIndex:
    def __init__(self, key_dim: int, code_dim: int = 256, active: int = 12, seed: int = 0):
        self.key_dim, self.code_dim, self.active = int(key_dim), int(code_dim), int(active)
        self.projection = np.random.default_rng(seed).standard_normal((self.code_dim, self.key_dim)).astype(np.float32)
        self.buckets: Dict[int, set] = {}
        self.codes: Dict[int, Tuple[int, ...]] = {}

    def code(self, key: np.ndarray) -> Tuple[int, ...]:
        key = np.asarray(key, dtype=np.float32).ravel()[: self.key_dim]
        key = np.pad(key, (0, self.key_dim - key.size))
        return tuple(sorted(np.argsort(-(self.projection @ key), kind="stable")[: self.active].tolist()))

    def add(self, record_id: int, key: np.ndarray) -> None:
        bits = self.code(key)
        self.codes[record_id] = bits
        for bit in bits:
            self.buckets.setdefault(bit, set()).add(record_id)

    def remove(self, record_id: int) -> None:
        for bit in self.codes.pop(record_id, ()):
            self.buckets.get(bit, set()).discard(record_id)

    def candidates(self, key: np.ndarray, min_overlap: int = 1) -> List[Tuple[int, int]]:
        """``[(record_id, overlap)]`` sorted by decreasing code overlap."""
        counts: Dict[int, int] = {}
        for bit in self.code(key):
            for rid in self.buckets.get(bit, ()):
                counts[rid] = counts.get(rid, 0) + 1
        return sorted(((rid, c) for rid, c in counts.items() if c >= min_overlap), key=lambda t: (-t[1], t[0]))

    def overlap(self, key_a: np.ndarray, key_b: np.ndarray) -> float:
        return len(set(self.code(key_a)) & set(self.code(key_b))) / float(self.active)


class MemoryStore:
    def __init__(self, capacity: int, key_dim: int, seed: int = 0, name: str = "episodic",
                 accept: Optional[Tuple[Provenance, ...]] = None):
        self.capacity, self.key_dim, self.name, self.accept = int(capacity), int(key_dim), name, accept
        self.records: Dict[int, MemoryRecord] = {}
        self.index = SparseIndex(key_dim, seed=seed)
        self.evicted = 0

    def __len__(self) -> int:
        return len(self.records)

    def _priority(self, record: MemoryRecord, now: int) -> float:
        recency = 1.0 / (1.0 + 0.01 * max(0, now - record.step))
        return record.confidence * (1.0 + abs(record.reward)) * recency

    def add(self, record: MemoryRecord) -> Optional[int]:
        """Insert a record, evicting the lowest-priority one when full.  Returns the evicted id, if any."""
        if self.accept is not None and record.provenance not in self.accept:
            raise ValueError(f"store '{self.name}' does not accept {record.provenance.value} records")
        evicted = None
        if len(self.records) >= self.capacity:
            evicted = min(self.records.values(), key=lambda r: (self._priority(r, record.step), r.record_id)).record_id
            self.remove(evicted)
            self.evicted += 1
        self.records[record.record_id] = record
        self.index.add(record.record_id, record.key)
        return evicted

    def remove(self, record_id: int) -> None:
        if self.records.pop(record_id, None) is not None:
            self.index.remove(record_id)

    def retrieve(self, key: np.ndarray, context: Optional[np.ndarray] = None, k: int = 1, min_similarity: float = 0.0,
                 observed_only: bool = False, allow_predictions: bool = True, context_weight: float = 0.3,
                 brute_force: bool = False) -> List[Tuple[MemoryRecord, float]]:
        key = np.asarray(key, dtype=np.float32).ravel()
        ids = list(self.records) if brute_force else [rid for rid, _ in self.index.candidates(key)]
        scored = []
        for rid in ids:
            record = self.records[rid]
            if observed_only and not (record.verified and record.provenance.is_observation):
                continue
            if not allow_predictions and record.provenance is Provenance.INTERNAL_PREDICTION:
                continue
            sim = cosine(key, record.key)
            if context is not None and record.context is not None:
                sim = (1.0 - context_weight) * sim + context_weight * cosine(np.asarray(context, dtype=np.float32), record.context)
            if sim >= min_similarity:
                scored.append((record, sim))
        scored.sort(key=lambda t: (-t[1], t[0].record_id))
        return scored[:k]

    def by_reward(self, sign: int = 1, k: int = 4) -> List[MemoryRecord]:
        """Reward-associated memories: the ``k`` most strongly rewarded (sign=1) or punished (sign=-1)."""
        chosen = [r for r in self.records.values() if r.reward * sign > 0]
        return sorted(chosen, key=lambda r: (-abs(r.reward), r.record_id))[:k]

    def decay(self, rate: float) -> None:
        for record in self.records.values():
            record.confidence = float(max(0.0, record.confidence * (1.0 - rate)))

    def downscale(self, factor: float) -> None:
        for record in self.records.values():
            record.value = record.value * float(factor)

    def forget(self, min_confidence: float) -> int:
        doomed = [rid for rid, r in self.records.items() if r.confidence < min_confidence]
        for rid in doomed:
            self.remove(rid)
        return len(doomed)

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "capacity": self.capacity, "evicted": int(self.evicted),
                "records": [r.to_dict() for r in self.records.values()]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.records, self.index.buckets, self.index.codes = {}, {}, {}
        self.evicted = int(state.get("evicted", 0))
        for data in state.get("records", []):
            record = MemoryRecord.from_dict(data)
            self.records[record.record_id] = record
            self.index.add(record.record_id, record.key)


class SemanticStore:
    """Prototype memory: running centroids of consolidated episodes."""

    def __init__(self, capacity: int = 64, radius: float = 0.9):
        self.capacity, self.radius = int(capacity), float(radius)
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.counts: List[int] = []

    def __len__(self) -> int:
        return len(self.keys)

    def absorb(self, record: MemoryRecord) -> bool:
        """Merge ``record`` into the closest prototype (cosine >= radius) or start a new one.  True if stored."""
        best, best_sim = -1, -1.0
        for i, key in enumerate(self.keys):
            sim = cosine(record.key, key)
            if sim > best_sim:
                best, best_sim = i, sim
        if best >= 0 and best_sim >= self.radius:
            c = self.counts[best]
            self.keys[best] = (self.keys[best] * c + record.key) / (c + 1)
            self.values[best] = (self.values[best] * c + record.value) / (c + 1)
            self.counts[best] = c + 1
            return True
        if len(self.keys) >= self.capacity:
            return False
        self.keys.append(record.key.copy())
        self.values.append(record.value.copy())
        self.counts.append(1)
        return True

    def retrieve(self, key: np.ndarray) -> Optional[Tuple[np.ndarray, float, int]]:
        if not self.keys:
            return None
        sims = [cosine(np.asarray(key, dtype=np.float32), k) for k in self.keys]
        i = int(np.argmax(sims))
        return self.values[i], float(sims[i]), self.counts[i]

    def state_dict(self) -> Dict[str, Any]:
        return {"keys": [k.tolist() for k in self.keys], "values": [v.tolist() for v in self.values], "counts": list(self.counts)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.keys = [np.asarray(k, dtype=np.float32) for k in state.get("keys", [])]
        self.values = [np.asarray(v, dtype=np.float32) for v in state.get("values", [])]
        self.counts = [int(c) for c in state.get("counts", [])]
