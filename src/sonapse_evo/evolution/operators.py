from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import copy
import uuid

from ..substrate.graph_topology import GraphTopology, EdgeMetadata, NodeMetadata, ModuleMetadata


@dataclass
class MutationReceipt:
    """Audit record capturing the exact changes applied by an operator for reversibility."""
    mutation_id: str
    operator_name: str
    applied: bool
    parameters: Dict[str, Any] = field(default_factory=dict)
    undo_state: Dict[str, Any] = field(default_factory=dict)
    resource_delta: Dict[str, float] = field(default_factory=dict)


class BaseStructuralOperator:
    """Abstract base class for all 20 bounded structural variation operators."""
    operator_name: str = "base_operator"

    def can_apply(self, graph: GraphTopology) -> bool:
        """Evaluates whether graph preconditions are satisfied for this operator."""
        return True

    def estimate_cost(self, graph: GraphTopology) -> Dict[str, float]:
        """Estimates computational, parameter, and memory costs."""
        return {"params": 0.0, "flops": 0.0, "memory_bytes": 0.0}

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        """Applies the mutation to the graph and returns an undoable receipt."""
        raise NotImplementedError

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        """Reverts the graph to its pre-mutation state using the receipt's undo_state."""
        raise NotImplementedError


# 1. Add Synapse Operator
class AddSynapseOperator(BaseStructuralOperator):
    operator_name = "add_synapse"

    def can_apply(self, graph: GraphTopology) -> bool:
        # Precondition: Density under 25% to maintain sparsity
        return graph.density < 0.25

    def estimate_cost(self, graph: GraphTopology) -> Dict[str, float]:
        return {"params": 1.0, "flops": 2.0, "memory_bytes": 4.0}

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        nodes = list(graph.nodes.keys())
        u = int(rng.choice(nodes))
        v = int(rng.choice(nodes))
        while u == v:
            v = int(rng.choice(nodes))

        # Respect Dale's sign of source node
        sign = graph.nodes[u].sign
        w = float(sign * rng.uniform(0.02, 0.15))
        
        # Check existing edge for undo state
        existing_idx = None
        old_w = 0.0
        for idx in graph._adj_out.get(u, []):
            if graph.edges[idx].post == v:
                existing_idx = idx
                old_w = graph.edges[idx].weight
                break

        e_idx = graph.add_edge(u, v, w)
        receipt = MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"pre": u, "post": v, "weight": w},
            undo_state={"pre": u, "post": v, "old_weight": old_w, "is_new": (existing_idx is None)},
            resource_delta={"params": 1.0},
        )
        return receipt

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        u = receipt.undo_state["pre"]
        v = receipt.undo_state["post"]
        old_w = receipt.undo_state["old_weight"]
        is_new = receipt.undo_state["is_new"]
        if is_new:
            return graph.remove_edge(u, v)
        else:
            graph.add_edge(u, v, old_w)
            return True


# 2. Remove Synapse Operator
class RemoveSynapseOperator(BaseStructuralOperator):
    operator_name = "remove_synapse"

    def can_apply(self, graph: GraphTopology) -> bool:
        return graph.edge_count > 100

    def estimate_cost(self, graph: GraphTopology) -> Dict[str, float]:
        return {"params": -1.0, "flops": -2.0, "memory_bytes": -4.0}

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        active_indices = [i for i, e in enumerate(graph.edges) if abs(e.weight) > 1e-6 and not e.frozen]
        if not active_indices:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        idx = int(rng.choice(active_indices))
        edge = graph.edges[idx]
        old_w = edge.weight
        edge.weight = 0.0

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"pre": edge.pre, "post": edge.post},
            undo_state={"edge_index": idx, "old_weight": old_w},
            resource_delta={"params": -1.0},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        idx = receipt.undo_state["edge_index"]
        graph.edges[idx].weight = receipt.undo_state["old_weight"]
        return True


# 3. Rewire Synapse Operator
class RewireSynapseOperator(BaseStructuralOperator):
    operator_name = "rewire_synapse"

    def can_apply(self, graph: GraphTopology) -> bool:
        return graph.edge_count > 50

    def estimate_cost(self, graph: GraphTopology) -> Dict[str, float]:
        return {"params": 0.0, "flops": 0.0, "memory_bytes": 0.0}

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        active = [i for i, e in enumerate(graph.edges) if abs(e.weight) > 1e-6 and not e.frozen]
        if not active:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        e_idx = int(rng.choice(active))
        edge = graph.edges[e_idx]
        old_pre, old_post, old_w = edge.pre, edge.post, edge.weight

        # Pick a new post target
        nodes = list(graph.nodes.keys())
        new_post = int(rng.choice(nodes))
        while new_post == old_pre or new_post == old_post:
            new_post = int(rng.choice(nodes))

        # Re-target edge
        edge.weight = 0.0  # Zero old
        new_idx = graph.add_edge(old_pre, new_post, old_w)

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"pre": old_pre, "old_post": old_post, "new_post": new_post, "weight": old_w},
            undo_state={"old_edge_idx": e_idx, "old_w": old_w, "new_pre": old_pre, "new_post": new_post},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        graph.edges[receipt.undo_state["old_edge_idx"]].weight = receipt.undo_state["old_w"]
        graph.remove_edge(receipt.undo_state["new_pre"], receipt.undo_state["new_post"])
        return True


# 4. Rescale Connection Operator
class RescaleConnectionOperator(BaseStructuralOperator):
    operator_name = "rescale_connection"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        active = [i for i, e in enumerate(graph.edges) if abs(e.weight) > 1e-6 and not e.frozen]
        if not active:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        idx = int(rng.choice(active))
        scale = float(rng.uniform(0.5, 1.5))
        old_w = graph.edges[idx].weight
        graph.edges[idx].weight = float(np.clip(old_w * scale, -2.0, 2.0))

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"edge_idx": idx, "scale": scale},
            undo_state={"edge_idx": idx, "old_weight": old_w},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        graph.edges[receipt.undo_state["edge_idx"]].weight = receipt.undo_state["old_weight"]
        return True


# 5. Add Neuron Operator
class AddNeuronOperator(BaseStructuralOperator):
    operator_name = "add_neuron"

    def can_apply(self, graph: GraphTopology) -> bool:
        return graph.total_neurons < 1024  # Max bound

    def estimate_cost(self, graph: GraphTopology) -> Dict[str, float]:
        return {"params": 16.0, "flops": 32.0, "memory_bytes": 128.0}

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        new_id = graph.total_neurons
        reg = int(rng.integers(0, graph.num_regions))
        sign = 1 if rng.uniform() > 0.2 else -1
        
        node = NodeMetadata(node_id=new_id, region_id=reg, module_id=reg, node_type="interneuron", sign=sign)
        graph.nodes[new_id] = node
        graph.total_neurons += 1
        graph._adj_out[new_id] = []
        graph._adj_in[new_id] = []
        graph.modules[reg].node_indices.append(new_id)

        # Wire with 4 input and 4 output connections
        reg_nodes = [nid for nid, n in graph.nodes.items() if n.region_id == reg and nid != new_id]
        new_edges = []
        if len(reg_nodes) >= 4:
            in_peers = rng.choice(reg_nodes, size=4, replace=False)
            out_peers = rng.choice(reg_nodes, size=4, replace=False)
            for p in in_peers:
                w = float(graph.nodes[int(p)].sign * rng.uniform(0.05, 0.15))
                e = graph.add_edge(int(p), new_id, w)
                new_edges.append(e)
            for p in out_peers:
                w = float(sign * rng.uniform(0.05, 0.15))
                e = graph.add_edge(new_id, int(p), w)
                new_edges.append(e)

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"new_neuron_id": new_id, "region": reg, "sign": sign},
            undo_state={"neuron_id": new_id, "region": reg, "added_edges": new_edges},
            resource_delta={"params": float(len(new_edges))},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        nid = receipt.undo_state["neuron_id"]
        reg = receipt.undo_state["region"]
        # Remove edges
        for e_idx in receipt.undo_state["added_edges"]:
            if e_idx < len(graph.edges):
                graph.edges[e_idx].weight = 0.0
        # Remove node
        if nid in graph.nodes:
            del graph.nodes[nid]
        if nid in graph.modules[reg].node_indices:
            graph.modules[reg].node_indices.remove(nid)
        graph.total_neurons -= 1
        return True


# 6. Prune Inactive Neuron Operator
class PruneNeuronOperator(BaseStructuralOperator):
    operator_name = "prune_neuron"

    def can_apply(self, graph: GraphTopology) -> bool:
        return graph.total_neurons > 512

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        # Find interneurons with lowest in-degree / out-degree weight sum
        scores = []
        for nid, n in graph.nodes.items():
            if n.node_type == "interneuron":
                out_w = sum(abs(graph.edges[i].weight) for i in graph._adj_out.get(nid, []))
                in_w = sum(abs(graph.edges[i].weight) for i in graph._adj_in.get(nid, []))
                scores.append((out_w + in_w, nid))

        if not scores:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        scores.sort()
        target_id = scores[0][1]
        
        # Zero connections
        cut_edges = []
        for i in graph._adj_out.get(target_id, []):
            cut_edges.append((i, graph.edges[i].weight))
            graph.edges[i].weight = 0.0
        for i in graph._adj_in.get(target_id, []):
            cut_edges.append((i, graph.edges[i].weight))
            graph.edges[i].weight = 0.0

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"pruned_neuron_id": target_id},
            undo_state={"target_id": target_id, "cut_edges": cut_edges},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        for idx, old_w in receipt.undo_state["cut_edges"]:
            graph.edges[idx].weight = old_w
        return True


# 7. Split Pathway Operator
class SplitPathwayOperator(BaseStructuralOperator):
    operator_name = "split_pathway"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        active = [i for i, e in enumerate(graph.edges) if abs(e.weight) > 0.05 and not e.frozen]
        if not active:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        idx = int(rng.choice(active))
        edge = graph.edges[idx]
        orig_w = edge.weight

        # Halve weight of primary pathway and add parallel bridge through intermediate node
        edge.weight = orig_w * 0.5
        reg = graph.nodes[edge.pre].region_id
        peers = [nid for nid, n in graph.nodes.items() if n.region_id == reg and nid != edge.pre and nid != edge.post]
        if not peers:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        mid = int(rng.choice(peers))
        e1 = graph.add_edge(edge.pre, mid, float(orig_w * 0.5))
        e2 = graph.add_edge(mid, edge.post, 0.5)

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"primary_edge": idx, "intermediate": mid},
            undo_state={"edge_idx": idx, "orig_w": orig_w, "e1": e1, "e2": e2, "mid": mid, "pre": edge.pre, "post": edge.post},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        graph.edges[receipt.undo_state["edge_idx"]].weight = receipt.undo_state["orig_w"]
        graph.remove_edge(receipt.undo_state["pre"], receipt.undo_state["mid"])
        graph.remove_edge(receipt.undo_state["mid"], receipt.undo_state["post"])
        return True


# 8. Local Recurrent Circuit Operator
class LocalRecurrentCircuitOperator(BaseStructuralOperator):
    operator_name = "local_recurrent_circuit"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        reg = int(rng.integers(0, graph.num_regions))
        reg_nodes = [nid for nid, n in graph.nodes.items() if n.region_id == reg]
        if len(reg_nodes) < 3:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        triplet = rng.choice(reg_nodes, size=3, replace=False)
        n0, n1, n2 = int(triplet[0]), int(triplet[1]), int(triplet[2])

        # Form a feedback loop n0 -> n1 -> n2 -> n0
        w0 = float(graph.nodes[n0].sign * 0.12)
        w1 = float(graph.nodes[n1].sign * 0.12)
        w2 = float(graph.nodes[n2].sign * 0.12)

        graph.add_edge(n0, n1, w0)
        graph.add_edge(n1, n2, w1)
        graph.add_edge(n2, n0, w2)

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"triplet": [n0, n1, n2]},
            undo_state={"triplet": [n0, n1, n2]},
            resource_delta={"params": 3.0},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        n0, n1, n2 = receipt.undo_state["triplet"]
        graph.remove_edge(n0, n1)
        graph.remove_edge(n1, n2)
        graph.remove_edge(n2, n0)
        return True


# 9. Grow Microcircuit Operator (Type B Graft Bridge)
class GrowMicrocircuitOperator(BaseStructuralOperator):
    operator_name = "grow_microcircuit"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        r_src = 0
        r_dst = 1
        src_nodes = [nid for nid, n in graph.nodes.items() if n.region_id == r_src]
        dst_nodes = [nid for nid, n in graph.nodes.items() if n.region_id == r_dst]
        s_port = rng.choice(src_nodes, size=4, replace=False)
        d_port = rng.choice(dst_nodes, size=4, replace=False)

        edges_added = []
        for s in s_port:
            for d in d_port:
                e = graph.add_edge(int(s), int(d), float(rng.uniform(0.01, 0.05)), edge_type="graft_internal")
                edges_added.append(e)

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"source_region": r_src, "target_region": r_dst, "ports": 4},
            undo_state={"edges": [(graph.edges[i].pre, graph.edges[i].post) for i in edges_added]},
            resource_delta={"params": float(len(edges_added))},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        for u, v in receipt.undo_state["edges"]:
            graph.remove_edge(u, v)
        return True


# 10. Add Low-Rank Bridge Operator (Type A Graft Bridge)
class AddLowRankBridgeOperator(BaseStructuralOperator):
    operator_name = "add_low_rank_bridge"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        r_src = int(rng.integers(0, graph.num_regions))
        r_dst = (r_src + 1) % graph.num_regions
        src_nodes = [nid for nid, n in graph.nodes.items() if n.region_id == r_src][:8]
        dst_nodes = [nid for nid, n in graph.nodes.items() if n.region_id == r_dst][:8]

        edges_added = []
        # Connect via low-rank intermediary hub
        hub_id = int(rng.choice(src_nodes))
        for d in dst_nodes:
            e = graph.add_edge(hub_id, int(d), 0.08, edge_type="bridge")
            edges_added.append((hub_id, int(d)))

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"hub": hub_id, "dst_nodes": len(dst_nodes)},
            undo_state={"edges": edges_added},
            resource_delta={"params": float(len(edges_added))},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        for u, v in receipt.undo_state["edges"]:
            graph.remove_edge(u, v)
        return True


# 11. Duplicate Module Operator
class DuplicateModuleOperator(BaseStructuralOperator):
    operator_name = "duplicate_module"

    def can_apply(self, graph: GraphTopology) -> bool:
        return len(graph.modules) < 8

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        target_mid = int(rng.choice(list(graph.modules.keys())))
        orig_mod = graph.modules[target_mid]

        new_mid = len(graph.modules)
        new_mod = ModuleMetadata(
            module_id=new_mid,
            region_id=orig_mod.region_id,
            name=f"{orig_mod.name}_copy",
            node_indices=list(orig_mod.node_indices[:16]),
            specialized_role=f"specialized_{orig_mod.specialized_role}",
        )
        graph.modules[new_mid] = new_mod

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"new_module_id": new_mid, "parent_module": target_mid},
            undo_state={"new_mid": new_mid},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        mid = receipt.undo_state["new_mid"]
        if mid in graph.modules:
            del graph.modules[mid]
        return True


# 12. Divide Module Operator
class DivideModuleOperator(BaseStructuralOperator):
    operator_name = "divide_module"

    def can_apply(self, graph: GraphTopology) -> bool:
        return any(len(m.node_indices) >= 32 for m in graph.modules.values())

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        eligible = [m for m in graph.modules.values() if len(m.node_indices) >= 32]
        target = eligible[0]
        half = len(target.node_indices) // 2

        new_mid = max(graph.modules.keys()) + 1
        moved_nodes = target.node_indices[half:]
        target.node_indices = target.node_indices[:half]

        graph.modules[new_mid] = ModuleMetadata(
            module_id=new_mid,
            region_id=target.region_id,
            name=f"{target.name}_sub",
            node_indices=moved_nodes,
        )

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"divided_module": target.module_id, "new_module_id": new_mid},
            undo_state={"parent_mid": target.module_id, "new_mid": new_mid, "moved_nodes": moved_nodes},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        p_mid = receipt.undo_state["parent_mid"]
        n_mid = receipt.undo_state["new_mid"]
        graph.modules[p_mid].node_indices.extend(receipt.undo_state["moved_nodes"])
        if n_mid in graph.modules:
            del graph.modules[n_mid]
        return True


# 13. Merge Redundant Modules Operator
class MergeModulesOperator(BaseStructuralOperator):
    operator_name = "merge_modules"

    def can_apply(self, graph: GraphTopology) -> bool:
        return len(graph.modules) > 3

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        mids = sorted(list(graph.modules.keys()))
        m_keep = mids[0]
        m_drop = mids[-1]
        dropped_nodes = list(graph.modules[m_drop].node_indices)

        graph.modules[m_keep].node_indices.extend(dropped_nodes)
        dropped_meta = graph.modules.pop(m_drop)

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"merged": m_drop, "into": m_keep},
            undo_state={"m_keep": m_keep, "m_drop": m_drop, "meta": dropped_meta},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        m_keep = receipt.undo_state["m_keep"]
        meta = receipt.undo_state["meta"]
        for nid in meta.node_indices:
            if nid in graph.modules[m_keep].node_indices:
                graph.modules[m_keep].node_indices.remove(nid)
        graph.modules[meta.module_id] = meta
        return True


# 14. Specialize Sensory Pathway Operator
class SpecializeSensoryPathwayOperator(BaseStructuralOperator):
    operator_name = "specialize_sensory_pathway"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        # Connect heading cosine/sine nodes (0, 1) directly to intermediate representation
        sensory_pre = [0, 1]
        inter_post = [256, 257]
        edges_added = []
        for u in sensory_pre:
            for v in inter_post:
                w = float(rng.uniform(0.15, 0.35))
                graph.add_edge(u, v, w, edge_type="bridge")
                edges_added.append((u, v))

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"type": "sensory_specialization"},
            undo_state={"edges": edges_added},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        for u, v in receipt.undo_state["edges"]:
            graph.remove_edge(u, v)
        return True


# 15. Specialize Memory Pathway Operator
class SpecializeMemoryPathwayOperator(BaseStructuralOperator):
    operator_name = "specialize_memory_pathway"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        # Self-sustaining recurrent feedback in Region 1
        m_nodes = [256, 257, 258, 259]
        edges_added = []
        for i in range(len(m_nodes)):
            u = m_nodes[i]
            v = m_nodes[(i + 1) % len(m_nodes)]
            graph.add_edge(u, v, 0.20)
            edges_added.append((u, v))

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"memory_ring": m_nodes},
            undo_state={"edges": edges_added},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        for u, v in receipt.undo_state["edges"]:
            graph.remove_edge(u, v)
        return True


# 16. Modulate Inter-Module Routing Operator
class ModulateInterModuleRoutingOperator(BaseStructuralOperator):
    operator_name = "modulate_inter_module_routing"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        bridges = [i for i, e in enumerate(graph.edges) if e.edge_type == "bridge" and abs(e.weight) > 1e-6]
        if not bridges:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        idx = int(rng.choice(bridges))
        old_w = graph.edges[idx].weight
        gain = float(rng.uniform(0.7, 1.4))
        graph.edges[idx].weight = float(np.clip(old_w * gain, -1.5, 1.5))

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"bridge_idx": idx, "gain": gain},
            undo_state={"bridge_idx": idx, "old_w": old_w},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        graph.edges[receipt.undo_state["bridge_idx"]].weight = receipt.undo_state["old_w"]
        return True


# 17. Tune Plasticity Operator
class TunePlasticityOperator(BaseStructuralOperator):
    operator_name = "tune_plasticity"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        rng = rng or np.random.default_rng()
        old_rate = graph.plasticity.hebbian_rate
        new_rate = float(np.clip(old_rate * rng.uniform(0.8, 1.25), 1e-5, 0.05))
        graph.plasticity.hebbian_rate = new_rate

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"new_hebbian_rate": new_rate},
            undo_state={"old_rate": old_rate},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        graph.plasticity.hebbian_rate = receipt.undo_state["old_rate"]
        return True


# 18. Adaptive Context Gating Operator
class AdaptiveContextGatingOperator(BaseStructuralOperator):
    operator_name = "adaptive_context_gating"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        # Connect inhibitory interneuron to damp down sensory pass-through during high activity
        u = 256  # central hub
        v = 512  # motor hub
        w = -0.10  # gating inhibition
        graph.add_edge(u, v, w)

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"gate_edge": (u, v)},
            undo_state={"pre": u, "post": v},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        return graph.remove_edge(receipt.undo_state["pre"], receipt.undo_state["post"])


# 19. Prune Redundant Computation Operator
class PruneRedundantComputationOperator(BaseStructuralOperator):
    operator_name = "prune_redundant_computation"

    def can_apply(self, graph: GraphTopology) -> bool:
        return graph.edge_count > 50

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        active = [i for i, e in enumerate(graph.edges) if abs(e.weight) > 1e-6 and not e.frozen]
        if not active:
            return MutationReceipt(mutation_id=uuid.uuid4().hex[:8], operator_name=self.operator_name, applied=False)

        # Sort by absolute weight magnitude and prune smallest 5 connections
        active.sort(key=lambda idx: abs(graph.edges[idx].weight))
        num_prune = min(5, len(active))
        pruned = []
        for idx in active[:num_prune]:
            pruned.append((idx, graph.edges[idx].weight))
            graph.edges[idx].weight = 0.0

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"pruned_count": len(pruned)},
            undo_state={"pruned": pruned},
            resource_delta={"params": -float(len(pruned))},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        for idx, old_w in receipt.undo_state["pruned"]:
            graph.edges[idx].weight = old_w
        return True


# 20. Transfer Donor Module Operator
class TransferDonorModuleOperator(BaseStructuralOperator):
    operator_name = "transfer_donor_module"

    def apply(self, graph: GraphTopology, rng: Optional[np.random.Generator] = None) -> MutationReceipt:
        # Imports a pre-structured microcircuit motif representing a donor pattern
        new_edges = []
        # Excitatory relay motif
        u, v = 512, 513
        e = graph.add_edge(u, v, 0.25, edge_type="donor")
        new_edges.append((u, v))

        return MutationReceipt(
            mutation_id=uuid.uuid4().hex[:8],
            operator_name=self.operator_name,
            applied=True,
            parameters={"donor_type": "poseidon_relay"},
            undo_state={"edges": new_edges},
            resource_delta={"params": float(len(new_edges))},
        )

    def undo(self, graph: GraphTopology, receipt: MutationReceipt) -> bool:
        for u, v in receipt.undo_state["edges"]:
            graph.remove_edge(u, v)
        return True


# Operator Registry Mapping 1-20
ALL_OPERATORS = [
    AddSynapseOperator(),
    RemoveSynapseOperator(),
    RewireSynapseOperator(),
    RescaleConnectionOperator(),
    AddNeuronOperator(),
    PruneNeuronOperator(),
    SplitPathwayOperator(),
    LocalRecurrentCircuitOperator(),
    GrowMicrocircuitOperator(),
    AddLowRankBridgeOperator(),
    DuplicateModuleOperator(),
    DivideModuleOperator(),
    MergeModulesOperator(),
    SpecializeSensoryPathwayOperator(),
    SpecializeMemoryPathwayOperator(),
    ModulateInterModuleRoutingOperator(),
    TunePlasticityOperator(),
    AdaptiveContextGatingOperator(),
    PruneRedundantComputationOperator(),
    TransferDonorModuleOperator(),
]


def get_operator_by_name(name: str) -> Optional[BaseStructuralOperator]:
    for op in ALL_OPERATORS:
        if op.operator_name == name:
            return op
    return None
