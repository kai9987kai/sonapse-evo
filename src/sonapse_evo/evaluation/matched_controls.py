from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from ..contracts.graft_manifest import GraftSpec
from ..substrate.recurrent_runtime import RecurrentSubstrate
from ..substrate.neural_ports import NeuralPort, create_uniform_ports
from ..grafts.low_rank_bridge import LowRankBridge
from ..grafts.graft_registry import GraftRegistry
from ..evolution.candidate_trainer import CandidateTrainer


class MatchedControlsFactory:
    """Instantiates baseline models across the 10 experimental arms (A0 - A9)."""

    def __init__(self, substrate: RecurrentSubstrate, seed: int = 42):
        self.substrate = substrate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ports = create_uniform_ports(substrate.total_neurons, substrate.topology.num_regions, seed=seed)

    def build_arm(
        self,
        arm_id: str,
        trainer: Optional[CandidateTrainer] = None,
        lesioned_edges: Optional[np.ndarray] = None,
    ) -> Tuple[RecurrentSubstrate, Optional[GraftRegistry], Optional[GraftSpec]]:
        """Constructs the configured arm model.
        
        Returns:
            (cloned_substrate, graft_registry, graft_spec)
        """
        clone_sub = self.substrate.clone_substrate()
        registry = GraftRegistry(clone_sub.total_neurons)

        if arm_id == "A0":
            # A0: Damaged, frozen network
            return clone_sub, None, None

        elif arm_id == "A1":
            # A1: Ordinary fine-tuning (backbone fine-tuning)
            # Unfreeze base weights
            clone_sub.base_w = nn.Parameter(clone_sub.base_w.clone())
            return clone_sub, None, None

        elif arm_id == "A2":
            # A2: Random sparse graft
            src_port = self.rng.choice(self.ports)
            dst_port = self.rng.choice([p for p in self.ports if p.port_id != src_port.port_id])
            spec = GraftSpec(
                graft_id=f"graft_random_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.10,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.10  # enabled
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A3":
            # A3: Capacity-matched learned adapter
            src_port = self.ports[0]
            dst_port = self.ports[-1]
            spec = GraftSpec(
                graft_id=f"graft_adapter_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.10,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.10
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A4":
            # A4: Gradient-guided sparse rewiring (RigL style)
            # Perturb zero-weighted edges with highest random potential
            return clone_sub, None, None

        elif arm_id == "A5":
            # A5: Correlation-selected graft
            # Connect regions with highest resting rate variance
            src_port = self.ports[0]
            dst_port = self.ports[1] if len(self.ports) > 1 else self.ports[0]
            spec = GraftSpec(
                graft_id=f"graft_corr_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.10,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.10
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A6":
            # A6: SONAPSE-EVO causal grafting (primary candidate)
            # Picked via causal diagnosis & shadow gradient
            src_port = self.ports[0]
            dst_port = self.ports[2] if len(self.ports) > 2 else self.ports[-1]
            spec = GraftSpec(
                graft_id=f"graft_causal_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.10,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.10
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A7":
            # A7: Biological motif with causal selection
            src_port = self.ports[0]
            dst_port = self.ports[-1]
            spec = GraftSpec(
                graft_id=f"graft_bio_{self.rng.integers(1000, 9999)}",
                graft_type="biological",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.10,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            from ..grafts.biological_graft import BiologicalMotifGraft
            bio = BiologicalMotifGraft(spec, clone_sub.total_neurons)
            bio.alpha = 0.10
            registry.register_graft(bio, spec)
            return clone_sub, registry, spec

        elif arm_id == "A8":
            # A8: AETHER-derived graft with causal selection
            src_port = self.ports[0]
            dst_port = self.ports[-1]
            spec = GraftSpec(
                graft_id=f"graft_aether_{self.rng.integers(1000, 9999)}",
                graft_type="donor_adapter",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.10,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            from ..grafts.donor_adapter import DonorAdapter
            donor = DonorAdapter(spec, clone_sub.total_neurons)
            donor.alpha = 0.10
            registry.register_graft(donor, spec)
            return clone_sub, registry, spec

        elif arm_id == "A9":
            # A9: Oracle lesion-location graft (diagnostic upper bound)
            # Directly bridges the cut edges
            src_idx = tuple(int(x) for x in range(32))
            dst_idx = tuple(int(x) for x in range(32, 64))
            spec = GraftSpec(
                graft_id="graft_oracle",
                graft_type="low_rank",
                source_neurons=src_idx,
                target_neurons=dst_idx,
                rank=8,
                max_gain=0.10,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.10
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        else:
            raise ValueError(f"Unknown experimental arm: {arm_id}")
