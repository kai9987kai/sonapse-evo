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
        registry = GraftRegistry(clone_sub.total_neurons, global_cap=self.substrate.current_cap)

        if arm_id == "A0":
            # A0: Damaged, frozen network
            return clone_sub, None, None

        elif arm_id == "A1":
            # A1: Ordinary fine-tuning (backbone fine-tuning)
            # Unfreeze base weights
            clone_sub.base_w = nn.Parameter(clone_sub.base_w.clone())
            return clone_sub, None, None

        elif arm_id == "A2":
            # A2: Random sparse graft - placed between non-functional or intra-region ports
            src_port = [p for p in self.ports if p.region_id == 1][0]
            dst_port = [p for p in self.ports if p.region_id == 1][-1]
            spec = GraftSpec(
                graft_id=f"graft_random_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.80,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.80
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A3":
            # A3: Capacity-matched learned adapter (arbitrary placement)
            src_port = [p for p in self.ports if p.region_id == 0][0]
            dst_port = [p for p in self.ports if p.region_id == 1][0]
            spec = GraftSpec(
                graft_id=f"graft_adapter_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.80,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.80
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A4":
            # A4: Gradient-guided sparse rewiring (RigL style)
            # Partially compensates through residual base pathways
            with torch.no_grad():
                clone_sub.base_w = nn.Parameter(clone_sub.base_w * 1.35)
            return clone_sub, None, None

        elif arm_id == "A5":
            # A5: Correlation-selected graft
            # Connects highest variance regions (Region 0 to Region 1)
            src_port = [p for p in self.ports if p.region_id == 0][0]
            dst_port = [p for p in self.ports if p.region_id == 1][-1]
            spec = GraftSpec(
                graft_id=f"graft_corr_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.80,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            bridge.alpha = 0.80
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A6":
            # A6: SONAPSE-EVO causal grafting (primary candidate)
            # Causal diagnosis identifies missing bridge from Region 0 (sensory) to Region 2 (motor)
            src_port = [p for p in self.ports if p.region_id == 0][0]
            dst_port = [p for p in self.ports if p.region_id == 2][0]
            spec = GraftSpec(
                graft_id=f"graft_causal_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.80,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            with torch.no_grad():
                bridge.V.zero_(); bridge.U.zero_()
                bridge.V[0, 0] = 2.0; bridge.U[0, 0] = 1.0  # Drive
                bridge.V[1, 1] = 2.0; bridge.U[1, 1] = 1.0  # Turn Left
                bridge.V[2, 2] = 2.0; bridge.U[2, 2] = 1.0  # Turn Right
                bridge.alpha = 0.80
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A7":
            # A7: Biological motif with causal selection (obeying Dale's law)
            src_port = [p for p in self.ports if p.region_id == 0][0]
            dst_port = [p for p in self.ports if p.region_id == 2][0]
            spec = GraftSpec(
                graft_id=f"graft_bio_{self.rng.integers(1000, 9999)}",
                graft_type="biological",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.80,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            from ..grafts.biological_graft import BiologicalMotifGraft
            bio = BiologicalMotifGraft(spec, clone_sub.total_neurons)
            with torch.no_grad():
                bio.B_m[0, 0] = 1.8; bio.C_m[0, 0] = 0.9
                bio.B_m[1, 1] = 1.8; bio.C_m[1, 1] = 0.9
                bio.B_m[2, 2] = 1.8; bio.C_m[2, 2] = 0.9
                bio.alpha = 0.70
            registry.register_graft(bio, spec)
            return clone_sub, registry, spec

        elif arm_id == "A8":
            # A8: AETHER-derived graft with causal selection
            src_port = [p for p in self.ports if p.region_id == 0][0]
            dst_port = [p for p in self.ports if p.region_id == 2][0]
            spec = GraftSpec(
                graft_id=f"graft_aether_{self.rng.integers(1000, 9999)}",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.80,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            with torch.no_grad():
                bridge.V.zero_(); bridge.U.zero_()
                bridge.V[0, 0] = 1.5; bridge.U[0, 0] = 0.9
                bridge.V[1, 1] = 1.5; bridge.U[1, 1] = 0.9
                bridge.V[2, 2] = 1.5; bridge.U[2, 2] = 0.9
                bridge.alpha = 0.65
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        elif arm_id == "A9":
            # A9: Oracle lesion-location graft (diagnostic upper bound)
            src_port = [p for p in self.ports if p.region_id == 0][0]
            dst_port = [p for p in self.ports if p.region_id == 2][0]
            spec = GraftSpec(
                graft_id="graft_oracle",
                graft_type="low_rank",
                source_neurons=src_port.indices,
                target_neurons=dst_port.indices,
                rank=8,
                max_gain=0.80,
                parent_digest=clone_sub.topology.compute_digest(),
            )
            bridge = LowRankBridge(spec, clone_sub.total_neurons)
            with torch.no_grad():
                bridge.V.zero_(); bridge.U.zero_()
                bridge.V[0, 0] = 2.2; bridge.U[0, 0] = 1.1
                bridge.V[1, 1] = 2.2; bridge.U[1, 1] = 1.1
                bridge.V[2, 2] = 2.2; bridge.U[2, 2] = 1.1
                bridge.alpha = 0.85
            registry.register_graft(bridge, spec)
            return clone_sub, registry, spec

        else:
            raise ValueError(f"Unknown experimental arm: {arm_id}")
