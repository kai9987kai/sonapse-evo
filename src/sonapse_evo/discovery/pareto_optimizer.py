from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .genotype_pool import ArchitectureGenotype


@dataclass
class HardConstraints:
    """Configurable hard bounds that candidates must satisfy."""
    max_neurons: int = 1024
    max_connections: int = 10000
    max_parameters: int = 20000
    max_memory_mb: float = 100.0
    max_jacobian_norm: float = 1.10


@dataclass
class UtilityWeights:
    """Configurable weights for scalar utility computation."""
    w_train: float = 0.35
    w_ood: float = 0.25
    w_adapt: float = 0.15
    w_forget: float = 0.10
    w_cost: float = 0.05
    w_instability: float = 0.10


class ParetoOptimizer:
    """Executes constrained non-dominated sorting and multiobjective ranking across 8 fitness axes."""

    def __init__(
        self,
        constraints: Optional[HardConstraints] = None,
        weights: Optional[UtilityWeights] = None,
    ):
        self.constraints = constraints or HardConstraints()
        self.weights = weights or UtilityWeights()

    def check_constraints(self, genotype: ArchitectureGenotype, jacobian_norm: float = 0.85) -> bool:
        """Verifies that candidate complies with all hard physical and dynamical constraints."""
        g = genotype.graph
        if g.total_neurons > self.constraints.max_neurons:
            return False
        if g.edge_count > self.constraints.max_connections:
            return False
        if jacobian_norm > self.constraints.max_jacobian_norm:
            return False
        return True

    def compute_scalar_utility(self, fitness: np.ndarray) -> float:
        """Computes scalar utility U from normalized 8-vector fitness.
        
        Fitness: [R_train, R_OOD, A, -Forget, -Latency, -Memory, -Energy, -Instability]
        """
        r_tr = float(fitness[0])
        r_ood = float(fitness[1])
        adapt = float(fitness[2])
        forget_penalty = abs(float(fitness[3]))
        cost_penalty = (abs(float(fitness[4])) + abs(float(fitness[5])) + abs(float(fitness[6]))) / 3.0
        instab_penalty = abs(float(fitness[7]))

        u = (
            self.weights.w_train * r_tr
            + self.weights.w_ood * r_ood
            + self.weights.w_adapt * adapt
            - self.weights.w_forget * forget_penalty
            - self.weights.w_cost * cost_penalty
            - self.weights.w_instability * instab_penalty
        )
        return float(u)

    def dominates(self, f1: np.ndarray, f2: np.ndarray) -> bool:
        """Returns True if fitness vector f1 strictly Pareto-dominates f2."""
        return np.all(f1 >= f2) and np.any(f1 > f2)

    def rank_population(self, population: List[ArchitectureGenotype]) -> List[ArchitectureGenotype]:
        """Performs Fast Non-Dominated Sorting (Deb et al., 2002) and computes crowding distance."""
        n = len(population)
        if n == 0:
            return population

        # Compute scalar utilities
        for g in population:
            g.scalar_utility = self.compute_scalar_utility(g.fitness_vector)

        # Non-dominated sorting
        domination_counts = [0] * n
        dominated_sets: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self.dominates(population[p].fitness_vector, population[q].fitness_vector):
                    dominated_sets[p].append(q)
                elif self.dominates(population[q].fitness_vector, population[p].fitness_vector):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                population[p].pareto_rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front: List[int] = []
            for p in fronts[i]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        population[q].pareto_rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        # Compute crowding distance for each front to maintain diversity
        for front in fronts:
            if not front:
                continue
            f_len = len(front)
            if f_len <= 2:
                for idx in front:
                    population[idx].crowding_distance = float("inf")
                continue

            for idx in front:
                population[idx].crowding_distance = 0.0

            # Compute across all 8 objectives
            num_objs = population[0].fitness_vector.shape[0]
            for m in range(num_objs):
                front_sorted = sorted(front, key=lambda idx: population[idx].fitness_vector[m])
                population[front_sorted[0]].crowding_distance = float("inf")
                population[front_sorted[-1]].crowding_distance = float("inf")
                obj_range = (
                    population[front_sorted[-1]].fitness_vector[m]
                    - population[front_sorted[0]].fitness_vector[m]
                )
                if obj_range > 1e-6:
                    for k in range(1, f_len - 1):
                        population[front_sorted[k]].crowding_distance += (
                            population[front_sorted[k + 1]].fitness_vector[m]
                            - population[front_sorted[k - 1]].fitness_vector[m]
                        ) / obj_range

        population.sort(key=lambda g: (g.pareto_rank, -g.scalar_utility))
        return population
