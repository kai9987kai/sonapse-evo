from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import uuid
import copy

from ..substrate.graph_topology import GraphTopology
from ..evolution.operators import ALL_OPERATORS, BaseStructuralOperator, MutationReceipt
from ..contracts.evidence import EvidenceTier


@dataclass
class ArchitectureGenotype:
    """Represents an architecture candidate genotype in the discovery population."""
    genotype_id: str
    parent_id: Optional[str]
    graph: GraphTopology
    applied_operators: List[str] = field(default_factory=list)
    # 8-dimensional multiobjective fitness vector:
    # [R_train, R_OOD, A, -Forget, -Latency, -Memory, -Energy, -Instability]
    fitness_vector: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    scalar_utility: float = 0.0
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    generation: int = 0
    evidence_tier: EvidenceTier = EvidenceTier.SIMULATED_ASSAY
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return self.graph.compute_digest()

    def clone(self, new_id: Optional[str] = None) -> ArchitectureGenotype:
        gid = new_id or f"gen_{uuid.uuid4().hex[:8]}"
        return ArchitectureGenotype(
            genotype_id=gid,
            parent_id=self.genotype_id,
            graph=self.graph.clone(),
            applied_operators=list(self.applied_operators),
            fitness_vector=self.fitness_vector.copy(),
            scalar_utility=self.scalar_utility,
            pareto_rank=self.pareto_rank,
            crowding_distance=self.crowding_distance,
            generation=self.generation + 1,
            evidence_tier=self.evidence_tier,
            metadata=copy.deepcopy(self.metadata),
        )


class GenotypePool:
    """Manages an evolutionary population of architecture candidates with Pareto ranking and diversity."""

    def __init__(
        self,
        base_graph: GraphTopology,
        population_size: int = 12,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)
        self.generation: int = 0
        self.population: List[ArchitectureGenotype] = []
        self.seen_digests: Set[str] = set()

        # Seed initial population from base graph with single diverse mutations
        root_gen = ArchitectureGenotype(
            genotype_id="gen_root",
            parent_id=None,
            graph=base_graph.clone(),
            generation=0,
        )
        self.population.append(root_gen)
        self.seen_digests.add(root_gen.digest)

        # Populate remaining slots with exploratory structural mutations
        while len(self.population) < self.population_size:
            child = root_gen.clone()
            op = self.rng.choice(ALL_OPERATORS)
            if op.can_apply(child.graph):
                receipt = op.apply(child.graph, rng=self.rng)
                if receipt.applied:
                    child.applied_operators.append(op.operator_name)
                    if child.digest not in self.seen_digests:
                        self.seen_digests.add(child.digest)
                        self.population.append(child)

    def select_parent_tournament(self, k: int = 3) -> ArchitectureGenotype:
        """Tournament selection favoring lower Pareto rank and higher scalar utility."""
        k = min(k, len(self.population))
        idx = self.rng.choice(len(self.population), size=k, replace=False)
        candidates = [self.population[i] for i in idx]
        # Sort by pareto rank ascending (rank 0 is best), then scalar utility descending
        candidates.sort(key=lambda g: (g.pareto_rank, -g.scalar_utility))
        return candidates[0]

    def mutate_candidate(self, parent: ArchitectureGenotype) -> Optional[ArchitectureGenotype]:
        """Creates a mutated child by applying one bounded structural variation operator."""
        child = parent.clone()
        # Shuffle operators to find an applicable one
        ops = list(ALL_OPERATORS)
        self.rng.shuffle(ops)

        for op in ops:
            if op.can_apply(child.graph):
                receipt = op.apply(child.graph, rng=self.rng)
                if receipt.applied:
                    child.applied_operators.append(op.operator_name)
                    if child.digest not in self.seen_digests:
                        self.seen_digests.add(child.digest)
                        return child
        return None

    def crossover(self, p1: ArchitectureGenotype, p2: ArchitectureGenotype) -> Optional[ArchitectureGenotype]:
        """Structure-preserving module recombination between two parent graphs."""
        child = p1.clone()
        # Transfer module 1 connections from p2 to child if regions match
        p2_mid1 = p2.graph.modules.get(1)
        if p2_mid1:
            for nid in p2_mid1.node_indices[:8]:
                for e_idx in p2.graph._adj_out.get(nid, []):
                    edge = p2.graph.edges[e_idx]
                    if edge.post in child.graph.nodes:
                        child.graph.add_edge(edge.pre, edge.post, edge.weight, edge.edge_type)
        if child.digest not in self.seen_digests:
            self.seen_digests.add(child.digest)
            child.applied_operators.append("module_crossover")
            return child
        return None

    def step_generation(self, elitism_count: int = 2) -> None:
        """Advances evolutionary population by selecting, mutating, and maintaining diversity."""
        self.generation += 1
        # Sort current population by Pareto dominance
        self.population.sort(key=lambda g: (g.pareto_rank, -g.scalar_utility))
        new_pop: List[ArchitectureGenotype] = self.population[:elitism_count]

        attempts = 0
        while len(new_pop) < self.population_size and attempts < 50:
            attempts += 1
            p = self.select_parent_tournament(k=3)
            # 80% mutation, 20% crossover
            if self.rng.uniform() < 0.8 or len(self.population) < 2:
                child = self.mutate_candidate(p)
            else:
                p2 = self.select_parent_tournament(k=3)
                child = self.crossover(p, p2) or self.mutate_candidate(p)

            if child is not None:
                new_pop.append(child)

        self.population = new_pop
