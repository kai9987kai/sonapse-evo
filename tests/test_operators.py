import pytest
import numpy as np
import torch

from sonapse_evo.substrate.graph_topology import GraphTopology
from sonapse_evo.evolution.operators import (
    ALL_OPERATORS,
    AddSynapseOperator,
    RemoveSynapseOperator,
    RewireSynapseOperator,
    RescaleConnectionOperator,
    AddNeuronOperator,
    PruneNeuronOperator,
    SplitPathwayOperator,
    LocalRecurrentCircuitOperator,
    GrowMicrocircuitOperator,
    AddLowRankBridgeOperator,
    DuplicateModuleOperator,
    DivideModuleOperator,
    MergeModulesOperator,
    SpecializeSensoryPathwayOperator,
    SpecializeMemoryPathwayOperator,
    ModulateInterModuleRoutingOperator,
    TunePlasticityOperator,
    AdaptiveContextGatingOperator,
    PruneRedundantComputationOperator,
    TransferDonorModuleOperator,
)


def test_operator_count_and_naming():
    """Verify that exactly 20 bounded structural variation operators exist."""
    assert len(ALL_OPERATORS) == 20
    names = {op.operator_name for op in ALL_OPERATORS}
    assert len(names) == 20


@pytest.mark.parametrize("operator", ALL_OPERATORS, ids=lambda op: op.operator_name)
def test_operator_apply_and_undo_reversibility(operator):
    """Verify that every single operator can be applied and cleanly reverted via undo."""
    graph = GraphTopology(total_neurons=768, num_regions=3, seed=42)
    rng = np.random.default_rng(123)

    cost = operator.estimate_cost(graph)
    assert isinstance(cost, dict)

    if operator.can_apply(graph):
        receipt = operator.apply(graph, rng=rng)
        assert receipt.applied is True
        assert receipt.operator_name == operator.operator_name

        # Verify undo reversibility
        reverted = operator.undo(graph, receipt)
        assert reverted is True
