from __future__ import annotations

from typing import Dict, Any, Optional, Protocol, Tuple
import os
import sys
from pathlib import Path
import numpy as np
import torch

from ..substrate.recurrent_runtime import RecurrentSubstrate
from ..substrate.topology_state import TopologyState
from ..substrate.lesions import LesionSpec
from ..contracts.snapshots import SubstrateSnapshot


class BrainAdapter(Protocol):
    """Protocol boundary between Evolution Controller and Neural Substrate."""

    def snapshot(self) -> SubstrateSnapshot: ...
    def clone(self) -> BrainAdapter: ...
    def restore(self, snapshot: SubstrateSnapshot) -> None: ...
    def apply_lesion(self, lesion: LesionSpec) -> None: ...
    def get_neural_state(self) -> np.ndarray: ...
    def get_topology_digest(self) -> str: ...
    def step(self, observation: np.ndarray, learn: bool = False) -> np.ndarray: ...


class SonapseAdapter:
    """Versioned adapter bridging to Sonapse-LAB or running self-contained proxy substrate."""

    def __init__(
        self,
        synapse_path: Optional[str] = None,
        total_neurons: int = 768,
        num_regions: int = 3,
        seed: int = 42,
    ):
        self.synapse_path = synapse_path or os.environ.get(
            "SYNAPSE_HOME", "C:\\Users\\kai99\\Desktop\\New folder (11)"
        )
        self.total_neurons = total_neurons
        self.num_regions = num_regions
        self.seed = seed
        self.is_live_synapse = False
        self.live_agent = None

        self._try_connect_synapse()
        if not self.is_live_synapse:
            # Standalone high-fidelity proxy substrate
            self.topology = TopologyState(
                total_neurons=total_neurons,
                num_regions=num_regions,
                seed=seed,
            )
            self.substrate = RecurrentSubstrate(
                topology=self.topology,
                sensory_dim=155,
                leak=0.35,
                microsteps=2,
                seed=seed,
            )

    def _try_connect_synapse(self):
        p = Path(self.synapse_path).resolve()
        if p.exists() and (p / "synapse").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            try:
                from synapse.config import SynapseConfig
                from synapse.agent import IntegratedAgent
                
                # Check resolved dimensions rather than assuming fixed counts
                cfg = SynapseConfig(
                    seed=self.seed,
                    nodes_per_source=self.total_neurons // self.num_regions,
                    total_neurons=self.total_neurons,
                    connectome_mode="proxy",
                )
                self.live_agent = IntegratedAgent(cfg)
                self.resolved_neurons = getattr(self.live_agent.brain, "n", self.total_neurons)
                self.total_neurons = self.resolved_neurons
                self.is_live_synapse = True
            except Exception:
                self.is_live_synapse = False

    def snapshot(self) -> SubstrateSnapshot:
        if self.is_live_synapse and hasattr(self.live_agent, "create_snapshot"):
            return self.live_agent.create_snapshot()
        return self.substrate.create_snapshot()

    def clone(self) -> SonapseAdapter:
        cloned = SonapseAdapter(
            synapse_path=self.synapse_path,
            total_neurons=self.total_neurons,
            num_regions=self.num_regions,
            seed=self.seed,
        )
        if not self.is_live_synapse:
            cloned.substrate = self.substrate.clone_substrate()
        return cloned

    def restore(self, snapshot: SubstrateSnapshot) -> None:
        if self.is_live_synapse and hasattr(self.live_agent, "restore_snapshot"):
            self.live_agent.restore_snapshot(snapshot)
        else:
            self.substrate.restore_snapshot(snapshot)

    def apply_lesion(self, lesion: LesionSpec) -> None:
        if self.is_live_synapse and hasattr(self.live_agent, "apply_lesion"):
            self.live_agent.apply_lesion(lesion)
        else:
            self.substrate.apply_lesion(lesion)

    def get_neural_state(self) -> np.ndarray:
        if self.is_live_synapse:
            return self.live_agent.brain.r.detach().cpu().numpy()
        return self.substrate.r.detach().cpu().numpy()

    def get_topology_digest(self) -> str:
        if self.is_live_synapse and hasattr(self.live_agent, "get_topology_digest"):
            return self.live_agent.get_topology_digest()
        return self.substrate.topology.compute_digest()

    def step(self, observation: np.ndarray, learn: bool = False) -> np.ndarray:
        if self.is_live_synapse:
            action = self.live_agent.step(observation, learn=learn)
            return np.asarray(action, dtype=np.float32)
        else:
            obs_t = torch.as_tensor(observation, dtype=torch.float32)
            new_r = self.substrate.forward_step(obs_t)
            # Motor readout
            action = np.tanh(new_r[:5].detach().cpu().numpy())
            return action
