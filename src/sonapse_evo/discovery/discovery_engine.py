from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import time

from .opportunity_monitor import DiscoveryOpportunityMonitor
from .genotype_pool import GenotypePool, ArchitectureGenotype
from .pareto_optimizer import ParetoOptimizer, HardConstraints, UtilityWeights
from .multi_fidelity import MultiFidelityEvaluator
from ..substrate.graph_topology import GraphTopology


class DiscoveryEngine:
    """Coordinates autonomous connectome architecture discovery across population search and Pareto ranking."""

    def __init__(
        self,
        base_graph: GraphTopology,
        population_size: int = 12,
        constraints: Optional[HardConstraints] = None,
        weights: Optional[UtilityWeights] = None,
        seed: int = 42,
    ):
        self.base_graph = base_graph
        self.population_size = population_size
        self.constraints = constraints or HardConstraints()
        self.weights = weights or UtilityWeights()
        self.seed = seed

        self.monitor = DiscoveryOpportunityMonitor(scheduled_exploration=True)
        self.pool = GenotypePool(base_graph, population_size=population_size, seed=seed)
        self.evaluator = MultiFidelityEvaluator(constraints=self.constraints)
        self.optimizer = ParetoOptimizer(constraints=self.constraints, weights=self.weights)

    def run_discovery_cycle(
        self,
        generations: int = 3,
        verbose: bool = False,
    ) -> Tuple[ArchitectureGenotype, List[ArchitectureGenotype]]:
        """Executes evolutionary discovery across multiple generations."""
        for gen in range(generations):
            # Evaluate current unranked or mutated candidates
            for candidate in self.pool.population:
                # Find parent if available
                parent = next((g for g in self.pool.population if g.genotype_id == candidate.parent_id), None)
                self.evaluator.evaluate_genotype(candidate, parent=parent)

            # Rank population using Pareto non-dominated sorting and crowding distance
            self.optimizer.rank_population(self.pool.population)

            if verbose:
                best = self.pool.population[0]
                print(f"Gen {gen}: Best Genotype={best.genotype_id}, Rank={best.pareto_rank}, Utility={best.scalar_utility:.3f}, Return={best.fitness_vector[0]:.2f}")

            # Advance generation if more remain
            if gen < generations - 1:
                self.pool.step_generation(elitism_count=2)

        # Final Pareto sort
        self.optimizer.rank_population(self.pool.population)
        champion = self.pool.population[0]
        return champion, self.pool.population
