from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import hashlib
import json
import numpy as np
import torch

from .topology_state import TopologyState


@dataclass
class NodeMetadata:
    """Metadata for a single neuron or functional unit in the graph."""
    node_id: int
    region_id: int = 0
    module_id: int = 0
    node_type: str = "interneuron"  # "sensory", "interneuron", "motor", "graft"
    sign: int = 1  # +1 for excitatory, -1 for inhibitory (Dale's law)
    delay_steps: int = 0
    mean_activity: float = 0.0
    activity_variance: float = 0.0
    plasticity_eligible: bool = True
    age_steps: int = 0
    functional_contribution: float = 0.0
    stability_score: float = 1.0


@dataclass
class EdgeMetadata:
    """Metadata for a directed computational connection."""
    pre: int
    post: int
    weight: float
    edge_type: str = "synapse"  # "synapse", "bridge", "graft_internal", "donor"
    delay: int = 0
    utilization: float = 0.0
    age_steps: int = 0
    plasticity_rate: float = 0.01
    frozen: bool = False


@dataclass
class ModuleMetadata:
    """Metadata for a functional sub-module within a region."""
    module_id: int
    region_id: int
    name: str
    node_indices: List[int] = field(default_factory=list)
    specialized_role: str = "general"  # "sensory_processor", "memory_buffer", "motor_pattern", "graft"


@dataclass
class PlasticityParameters:
    """Parameters governing local synaptic and structural adaptation."""
    hebbian_rate: float = 0.001
    homeostatic_target: float = 0.20
    tau_eligibility: float = 20.0
    prune_threshold: float = 0.005
    max_weight_norm: float = 2.0


class GraphTopology:
    """Generalized typed computational graph representation: G = (V, E, T, W, P, M).
    
    Fully encapsulates the existing TopologyState while supporting 
    dynamic structural variation operators (growth, pruning, rewiring, module splitting).
    """

    def __init__(
        self,
        total_neurons: int = 768,
        num_regions: int = 3,
        region_names: Tuple[str, ...] = ("flywire", "manc", "optic-lobe"),
        seed: int = 42,
    ):
        self.total_neurons = total_neurons
        self.num_regions = num_regions
        self.region_names = region_names
        self.seed = seed

        # Graph components G = (V, E, T, W, P, M)
        self.nodes: Dict[int, NodeMetadata] = {}
        self.edges: List[EdgeMetadata] = []
        self.modules: Dict[int, ModuleMetadata] = {}
        self.plasticity = PlasticityParameters()
        self.history: List[Dict[str, Any]] = []

        # Adjacency index for fast lookup: pre -> list of edge indices
        self._adj_out: Dict[int, List[int]] = {}
        self._adj_in: Dict[int, List[int]] = {}

        self._initialize_from_default()

    @classmethod
    def from_topology_state(cls, topo: TopologyState) -> GraphTopology:
        """Constructs a GraphTopology from an existing TopologyState instance."""
        graph = cls(
            total_neurons=topo.total_neurons,
            num_regions=topo.num_regions,
            region_names=topo.region_names,
            seed=topo.seed,
        )
        # Clear default edges and initialize from topo
        graph.edges = []
        graph._adj_out = {i: [] for i in range(topo.total_neurons)}
        graph._adj_in = {i: [] for i in range(topo.total_neurons)}

        # Update region assignments in node metadata
        for i in range(topo.total_neurons):
            reg = int(topo.region_assignments[i])
            ntype = "sensory" if reg == 0 else ("motor" if reg == 2 else "interneuron")
            graph.nodes[i].region_id = reg
            graph.nodes[i].node_type = ntype

        # Add edges from topo.base_pre, base_post, base_w
        for idx in range(len(topo.base_pre)):
            u = int(topo.base_pre[idx])
            v = int(topo.base_post[idx])
            w = float(topo.base_w[idx])
            edge_type = "bridge" if graph.nodes[u].region_id != graph.nodes[v].region_id else "synapse"
            edge = EdgeMetadata(pre=u, post=v, weight=w, edge_type=edge_type)
            e_idx = len(graph.edges)
            graph.edges.append(edge)
            graph._adj_out.setdefault(u, []).append(e_idx)
            graph._adj_in.setdefault(v, []).append(e_idx)

        return graph

    def _initialize_from_default(self) -> None:
        """Initializes default nodes, modules, and connectivity matching v1 TopologyState."""
        topo = TopologyState(
            total_neurons=self.total_neurons,
            num_regions=self.num_regions,
            region_names=self.region_names,
            seed=self.seed,
        )
        nodes_per_region = self.total_neurons // self.num_regions

        for i in range(self.total_neurons):
            reg = int(topo.region_assignments[i])
            ntype = "sensory" if reg == 0 else ("motor" if reg == 2 else "interneuron")
            sign = 1 if (i % 5 != 0) else -1  # 80% E, 20% I
            self.nodes[i] = NodeMetadata(
                node_id=i,
                region_id=reg,
                module_id=reg,
                node_type=ntype,
                sign=sign,
            )
            self._adj_out[i] = []
            self._adj_in[i] = []

        # Initialize default 3 modules corresponding to the 3 regions
        for r in range(self.num_regions):
            r_start = r * nodes_per_region
            r_end = r_start + nodes_per_region if r < self.num_regions - 1 else self.total_neurons
            self.modules[r] = ModuleMetadata(
                module_id=r,
                region_id=r,
                name=self.region_names[r],
                node_indices=list(range(r_start, r_end)),
                specialized_role="sensory_processor" if r == 0 else ("motor_pattern" if r == 2 else "memory_buffer"),
            )

        # Import edges from default TopologyState
        for idx in range(len(topo.base_pre)):
            u = int(topo.base_pre[idx])
            v = int(topo.base_post[idx])
            w = float(topo.base_w[idx])
            edge_type = "bridge" if self.nodes[u].region_id != self.nodes[v].region_id else "synapse"
            edge = EdgeMetadata(pre=u, post=v, weight=w, edge_type=edge_type)
            e_idx = len(self.edges)
            self.edges.append(edge)
            self._adj_out[u].append(e_idx)
            self._adj_in[v].append(e_idx)

    def to_topology_state(self) -> TopologyState:
        """Converts graph into an exact TopologyState compatible with RecurrentSubstrate."""
        region_assignments = np.zeros(self.total_neurons, dtype=np.int64)
        for i in range(self.total_neurons):
            if i in self.nodes:
                region_assignments[i] = self.nodes[i].region_id

        active_edges = [e for e in self.edges if abs(e.weight) > 1e-6]
        if not active_edges:
            pre = np.array([], dtype=np.int64)
            post = np.array([], dtype=np.int64)
            w = np.array([], dtype=np.float32)
        else:
            pre = np.array([e.pre for e in active_edges], dtype=np.int64)
            post = np.array([e.post for e in active_edges], dtype=np.int64)
            w = np.array([e.weight for e in active_edges], dtype=np.float32)

        topo = TopologyState(
            total_neurons=self.total_neurons,
            num_regions=self.num_regions,
            region_names=self.region_names,
            region_assignments=region_assignments,
            base_pre=pre,
            base_post=post,
            base_w=w,
            seed=self.seed,
        )
        return topo

    def add_edge(self, pre: int, post: int, weight: float, edge_type: str = "synapse", frozen: bool = False) -> int:
        """Adds or updates a directed edge in the graph."""
        if pre not in self.nodes or post not in self.nodes:
            raise ValueError(f"Nodes ({pre}, {post}) must exist in graph.")
        if pre == post:
            raise ValueError(f"Self-loops are forbidden (pre={pre}, post={post}).")

        # Check if edge already exists
        for e_idx in self._adj_out.get(pre, []):
            if self.edges[e_idx].post == post:
                self.edges[e_idx].weight = weight
                self.edges[e_idx].edge_type = edge_type
                self.edges[e_idx].frozen = frozen
                return e_idx

        e_idx = len(self.edges)
        edge = EdgeMetadata(pre=pre, post=post, weight=weight, edge_type=edge_type, frozen=frozen)
        self.edges.append(edge)
        self._adj_out.setdefault(pre, []).append(e_idx)
        self._adj_in.setdefault(post, []).append(e_idx)
        return e_idx

    def remove_edge(self, pre: int, post: int) -> bool:
        """Zeroes out weight or removes directed connection."""
        found = False
        for e_idx in self._adj_out.get(pre, []):
            if self.edges[e_idx].post == post:
                self.edges[e_idx].weight = 0.0
                found = True
        return found

    def compute_digest(self) -> str:
        """Cryptographic SHA-256 digest of topological graph structure and weights."""
        topo = self.to_topology_state()
        return topo.compute_digest()

    @property
    def edge_count(self) -> int:
        return sum(1 for e in self.edges if abs(e.weight) > 1e-6)

    @property
    def density(self) -> float:
        n = self.total_neurons
        return self.edge_count / max(1, n * (n - 1))

    def clone(self) -> GraphTopology:
        """Creates an independent deep clone of the graph topology."""
        new_g = GraphTopology(
            total_neurons=self.total_neurons,
            num_regions=self.num_regions,
            region_names=self.region_names,
            seed=self.seed,
        )
        new_g.nodes = {nid: NodeMetadata(**n.__dict__) for nid, n in self.nodes.items()}
        new_g.edges = [EdgeMetadata(**e.__dict__) for e in self.edges]
        new_g.modules = {mid: ModuleMetadata(**m.__dict__) for mid, m in self.modules.items()}
        new_g.plasticity = PlasticityParameters(**self.plasticity.__dict__)
        new_g._adj_out = {k: list(v) for k, v in self._adj_out.items()}
        new_g._adj_in = {k: list(v) for k, v in self._adj_in.items()}
        return new_g
