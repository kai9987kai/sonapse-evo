from __future__ import annotations

import pytest
import numpy as np

from sonapse_evo.contracts.graft_manifest import GraftSpec
from sonapse_evo.substrate.topology_state import TopologyState


def test_topology_digest_reproducibility():
    top1 = TopologyState(total_neurons=768, num_regions=3, seed=42)
    top2 = TopologyState(total_neurons=768, num_regions=3, seed=42)
    top3 = TopologyState(total_neurons=768, num_regions=3, seed=99)

    d1 = top1.compute_digest()
    d2 = top2.compute_digest()
    d3 = top3.compute_digest()

    assert d1 == d2
    assert d1 != d3


def test_graft_spec_validation():
    # Valid spec
    valid_spec = GraftSpec(
        graft_id="graft_valid",
        graft_type="low_rank",
        source_neurons=(0, 1, 2),
        target_neurons=(3, 4, 5),
        rank=4,
    )
    valid_spec.validate(total_neurons=768)

    # Out of bounds port
    invalid_spec = GraftSpec(
        graft_id="graft_invalid",
        graft_type="low_rank",
        source_neurons=(0, 800),  # > 768
        target_neurons=(3, 4),
        rank=4,
    )
    with pytest.raises(IndexError):
        invalid_spec.validate(total_neurons=768)
