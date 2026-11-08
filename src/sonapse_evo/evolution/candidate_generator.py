from __future__ import annotations

from typing import List, Tuple, Sequence, Dict, Any, Optional
import uuid
import numpy as np

from ..contracts.graft_manifest import GraftSpec, GraftType
from ..substrate.neural_ports import NeuralPort


class CandidateGenerator:
    """Generates typed, budgeted neural graft candidate specifications."""

    def __init__(
        self,
        default_rank: int = 8,
        default_max_gain: float = 0.10,
        ramp_steps: int = 50,
        recurrent_state_dim: int = 16,
        seed: int = 42,
    ):
        self.default_rank = int(default_rank)
        self.default_max_gain = float(default_max_gain)
        self.ramp_steps = int(ramp_steps)
        self.recurrent_state_dim = int(recurrent_state_dim)
        self.rng = np.random.default_rng(seed)

    def generate_candidates(
        self,
        port_pairs: Sequence[Tuple[NeuralPort, NeuralPort, float]],
        parent_digest: str,
        allowed_types: Tuple[GraftType, ...] = ("low_rank", "recurrent", "biological"),
        max_candidates: int = 64,
    ) -> List[GraftSpec]:
        """Generates candidate GraftSpec objects across allowed graft architectures."""
        candidates: List[GraftSpec] = []
        if len(port_pairs) == 0:
            return candidates

        for idx, (src_port, dst_port, priority) in enumerate(port_pairs):
            if len(candidates) >= max_candidates:
                break

            for g_type in allowed_types:
                if len(candidates) >= max_candidates:
                    break

                cid = f"graft_{g_type}_{uuid.uuid4().hex[:8]}"
                spec = GraftSpec(
                    graft_id=cid,
                    graft_type=g_type,
                    source_neurons=src_port.indices,
                    target_neurons=dst_port.indices,
                    rank=self.default_rank,
                    max_gain=self.default_max_gain,
                    ramp_steps=self.ramp_steps,
                    parent_digest=parent_digest,
                    recurrent_state_dim=self.recurrent_state_dim,
                    metadata={
                        "source_port": src_port.port_id,
                        "target_port": dst_port.port_id,
                        "priority_score": float(priority),
                    },
                )
                candidates.append(spec)

        return candidates
