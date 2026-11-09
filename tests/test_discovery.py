import pytest
import numpy as np

from sonapse_evo.substrate.graph_topology import GraphTopology
from sonapse_evo.discovery.opportunity_monitor import DiscoveryOpportunityMonitor
from sonapse_evo.discovery.genotype_pool import GenotypePool, ArchitectureGenotype
from sonapse_evo.discovery.pareto_optimizer import ParetoOptimizer, HardConstraints, UtilityWeights
from sonapse_evo.discovery.multi_fidelity import MultiFidelityEvaluator
from sonapse_evo.discovery.discovery_engine import DiscoveryEngine
from sonapse_evo.contracts.evidence import EvidenceTier


def test_discovery_opportunity_monitor_exponential_smoothing():
    monitor = DiscoveryOpportunityMonitor(beta=0.90, trigger_threshold=0.45, persistence_steps=2)
    
    # Healthy run - high return, low gap, good stability -> no trigger
    trig = monitor.update(return_train=35.0, return_ood=34.0, jacobian_norm=0.70, density=0.08)
    assert trig is False
    assert monitor.current_score < 0.45

    # Plateau and large gap over consecutive updates
    for _ in range(4):
        trig = monitor.update(return_train=10.0, return_ood=2.0, jacobian_norm=1.15, density=0.01)

    assert monitor.current_score > 0.40
    assert trig is True


def test_genotype_pool_and_evolutionary_stepping():
    base_graph = GraphTopology(total_neurons=768, num_regions=3, seed=42)
    pool = GenotypePool(base_graph, population_size=6, seed=42)
    assert len(pool.population) == 6

    parent = pool.select_parent_tournament(k=2)
    assert isinstance(parent, ArchitectureGenotype)

    child = pool.mutate_candidate(parent)
    assert child is not None
    assert len(child.applied_operators) > 0

    pool.step_generation(elitism_count=2)
    assert len(pool.population) == 6
    assert pool.generation == 1


def test_pareto_optimizer_non_dominated_sorting():
    optimizer = ParetoOptimizer()

    # Create dummy genotypes with distinct fitness vectors
    g1 = ArchitectureGenotype(
        genotype_id="g1",
        parent_id=None,
        graph=GraphTopology(768, 3, seed=1),
        fitness_vector=np.array([30.0, 28.0, 5.0, -1.0, -10.0, -50.0, -100.0, -0.05], dtype=np.float32),
    )
    g2 = ArchitectureGenotype(
        genotype_id="g2",
        parent_id=None,
        graph=GraphTopology(768, 3, seed=2),
        fitness_vector=np.array([20.0, 18.0, 2.0, -5.0, -15.0, -60.0, -120.0, -0.15], dtype=np.float32),
    )
    g3 = ArchitectureGenotype(
        genotype_id="g3",
        parent_id=None,
        graph=GraphTopology(768, 3, seed=3),
        fitness_vector=np.array([35.0, 32.0, 7.0, -0.5, -8.0, -40.0, -90.0, -0.02], dtype=np.float32),
    )

    pop = [g1, g2, g3]
    ranked = optimizer.rank_population(pop)

    # g3 strictly dominates g1 and g2; g1 dominates g2
    assert ranked[0].genotype_id == "g3"
    assert ranked[0].pareto_rank == 0
    assert ranked[1].genotype_id == "g1"
    assert ranked[1].pareto_rank == 1
    assert ranked[2].genotype_id == "g2"
    assert ranked[2].pareto_rank == 2


def test_multi_fidelity_evaluator_pipeline():
    evaluator = MultiFidelityEvaluator(max_stage=4)
    base_graph = GraphTopology(total_neurons=768, num_regions=3, seed=42)
    gen = ArchitectureGenotype(genotype_id="cand_test", parent_id=None, graph=base_graph)

    passed, stages = evaluator.evaluate_genotype(gen)
    assert passed is True
    assert len(stages) >= 4
    assert stages[0].stage_name == "validation"
    assert stages[1].stage_name == "screening"
    assert stages[2].stage_name == "short_training"
    assert stages[3].stage_name == "paired_evaluation"
    assert gen.evidence_tier == EvidenceTier.PAIRED_EXPERIMENT
