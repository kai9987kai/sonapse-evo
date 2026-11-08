from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import hashlib
import numpy as np
import torch

from .topology_state import TopologyState


@dataclass(frozen=True)
class LesionSpec:
    lesion_id: str
    lesion_type: str  # 'random_edges', 'inter_region_bridge', 'sensory_pathway', 'recurrent_memory', 'targeted_hub'
    severity: float = 0.20
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_digest(self) -> str:
        payload = f"{self.lesion_id}:{self.lesion_type}:{self.severity}:{self.seed}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LesionEngine:
    """Applies controlled topological lesions to the recurrent substrate."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def create_lesion_mask(self, topology: TopologyState, spec: LesionSpec) -> np.ndarray:
        m = len(topology.base_pre)
        mask = np.ones(m, dtype=np.float32)
        if m == 0 or spec.severity <= 0.0:
            return mask

        rng = np.random.default_rng(spec.seed)

        if spec.lesion_type == "random_edges":
            k = max(1, int(m * spec.severity))
            drop_idx = rng.choice(m, size=k, replace=False)
            mask[drop_idx] = 0.0

        elif spec.lesion_type == "inter_region_bridge":
            pre_regions = topology.region_assignments[topology.base_pre]
            post_regions = topology.region_assignments[topology.base_post]
            bridge_indices = np.flatnonzero(pre_regions != post_regions)
            if len(bridge_indices) > 0:
                k = max(1, int(len(bridge_indices) * min(1.0, spec.severity * 2.0)))
                drop_idx = rng.choice(bridge_indices, size=k, replace=False)
                mask[drop_idx] = 0.0

        elif spec.lesion_type == "sensory_pathway":
            # Nodes 0 to total_neurons // 4 represent the sensory entry hub
            sensory_nodes = np.arange(0, topology.total_neurons // 4)
            sensory_edges = np.flatnonzero(np.isin(topology.base_pre, sensory_nodes))
            if len(sensory_edges) > 0:
                k = max(1, int(len(sensory_edges) * spec.severity))
                drop_idx = rng.choice(sensory_edges, size=k, replace=False)
                mask[drop_idx] = 0.0

        elif spec.lesion_type == "recurrent_memory":
            # Reciprocal edges: pairs (i, j) and (j, i)
            edge_dict = {}
            reciprocal_pairs = []
            for idx, (u, v) in enumerate(zip(topology.base_pre, topology.base_post)):
                if (v, u) in edge_dict:
                    reciprocal_pairs.append((idx, edge_dict[(v, u)]))
                else:
                    edge_dict[(u, v)] = idx
            
            if len(reciprocal_pairs) > 0:
                k = max(1, int(len(reciprocal_pairs) * spec.severity))
                chosen = rng.choice(len(reciprocal_pairs), size=k, replace=False)
                for c in chosen:
                    idx1, idx2 = reciprocal_pairs[c]
                    mask[idx1] = 0.0
                    mask[idx2] = 0.0
            else:
                # Fallback to random edges
                k = max(1, int(m * spec.severity))
                mask[rng.choice(m, size=k, replace=False)] = 0.0

        elif spec.lesion_type == "targeted_hub":
            # Identify highest-degree nodes
            degrees = np.bincount(topology.base_pre, minlength=topology.total_neurons) + \
                      np.bincount(topology.base_post, minlength=topology.total_neurons)
            num_hubs = max(1, int(topology.total_neurons * 0.05))
            top_hubs = np.argsort(degrees)[-num_hubs:]
            hub_edges = np.flatnonzero(np.isin(topology.base_pre, top_hubs) | np.isin(topology.base_post, top_hubs))
            if len(hub_edges) > 0:
                k = max(1, int(len(hub_edges) * spec.severity))
                drop_idx = rng.choice(hub_edges, size=k, replace=False)
                mask[drop_idx] = 0.0

        else:
            raise ValueError(f"Unknown lesion type: {spec.lesion_type}")

        return mask
