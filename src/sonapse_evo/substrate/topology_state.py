from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import hashlib
import json
import numpy as np
import torch


@dataclass
class TopologyState:
    total_neurons: int = 768
    num_regions: int = 3
    region_names: Tuple[str, ...] = ("flywire", "manc", "optic-lobe")
    region_assignments: Optional[np.ndarray] = None
    base_pre: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    base_post: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    base_w: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    seed: int = 42

    def __post_init__(self):
        if self.region_assignments is None or len(self.region_assignments) != self.total_neurons:
            # Distribute neurons uniformly across regions by default
            nodes_per_region = self.total_neurons // self.num_regions
            assignments = np.zeros(self.total_neurons, dtype=np.int64)
            for r in range(self.num_regions):
                start = r * nodes_per_region
                end = start + nodes_per_region if r < self.num_regions - 1 else self.total_neurons
                assignments[start:end] = r
            self.region_assignments = assignments

        if len(self.base_pre) == 0:
            self._generate_default_topology()

    def _generate_default_topology(self, mean_degree: int = 8):
        rng = np.random.default_rng(self.seed)
        num_edges = int(self.total_neurons * mean_degree)
        
        # Intra-region dense connections + sparse inter-region bridges
        pre_list = []
        post_list = []
        w_list = []
        
        # 80% intra-region, 20% inter-region
        intra_edges = int(num_edges * 0.8)
        inter_edges = num_edges - intra_edges
        
        # Intra-region
        nodes_per_region = self.total_neurons // self.num_regions
        for r in range(self.num_regions):
            r_start = r * nodes_per_region
            r_end = r_start + nodes_per_region if r < self.num_regions - 1 else self.total_neurons
            r_nodes = np.arange(r_start, r_end)
            r_edge_count = intra_edges // self.num_regions
            
            p_pre = rng.choice(r_nodes, size=r_edge_count, replace=True)
            p_post = rng.choice(r_nodes, size=r_edge_count, replace=True)
            # Avoid self-loops
            mask = p_pre != p_post
            p_pre = p_pre[mask]
            p_post = p_post[mask]
            # Dale's law: 80% excitatory (+), 20% inhibitory (-)
            signs = rng.choice([1.0, -1.0], size=len(p_pre), p=[0.8, 0.2])
            weights = signs * rng.uniform(0.05, 0.25, size=len(p_pre))
            
            pre_list.append(p_pre)
            post_list.append(p_post)
            w_list.append(weights)
            
        # Inter-region bridges
        p_pre_inter = rng.integers(0, self.total_neurons, size=inter_edges)
        p_post_inter = rng.integers(0, self.total_neurons, size=inter_edges)
        mask = self.region_assignments[p_pre_inter] != self.region_assignments[p_post_inter]
        p_pre_inter = p_pre_inter[mask]
        p_post_inter = p_post_inter[mask]
        signs_inter = rng.choice([1.0, -1.0], size=len(p_pre_inter), p=[0.8, 0.2])
        w_inter = signs_inter * rng.uniform(0.02, 0.15, size=len(p_pre_inter))
        
        pre_list.append(p_pre_inter)
        post_list.append(p_post_inter)
        w_list.append(w_inter)

        # Canonical descending sensorimotor pathway (Sensory Region 0 to Motor Region 2)
        canon_pre = np.array([0, 1, 2], dtype=np.int64)
        canon_post = np.array([512, 513, 514], dtype=np.int64)
        canon_w = np.array([0.80, 0.80, 0.80], dtype=np.float32)
        pre_list.insert(0, canon_pre)
        post_list.insert(0, canon_post)
        w_list.insert(0, canon_w)
        
        self.base_pre = np.concatenate(pre_list).astype(np.int64)
        self.base_post = np.concatenate(post_list).astype(np.int64)
        self.base_w = np.concatenate(w_list).astype(np.float32)

    def get_region_nodes(self, region_id: int) -> np.ndarray:
        return np.flatnonzero(self.region_assignments == region_id)

    def compute_digest(self) -> str:
        h = hashlib.sha256()
        h.update(f"total_neurons:{self.total_neurons}:regions:{self.num_regions}:".encode("utf-8"))
        h.update(self.region_assignments.tobytes())
        h.update(self.base_pre.tobytes())
        h.update(self.base_post.tobytes())
        h.update(self.base_w.tobytes())
        return h.hexdigest()
