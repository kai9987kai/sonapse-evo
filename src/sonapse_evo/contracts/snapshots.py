from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import hashlib
import json
import torch
import numpy as np


def compute_tensor_digest(t: torch.Tensor | np.ndarray) -> str:
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().contiguous().numpy()
    else:
        arr = np.ascontiguousarray(t)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(list(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class SubstrateSnapshot:
    step_count: int
    total_neurons: int
    topology_digest: str
    state_digest: str
    rate_vector: torch.Tensor
    lesion_mask: Optional[torch.Tensor]
    active_grafts: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verify_integrity(self) -> bool:
        expected = compute_tensor_digest(self.rate_vector)
        return expected == self.state_digest
