from __future__ import annotations

import pytest
import torch
import numpy as np

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate


def test_snapshot_restore_exact():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, seed=42)

    # Run a few steps to create non-trivial neural rates
    obs = torch.rand(155)
    substrate.forward_step(obs)
    substrate.forward_step(obs)

    snapshot = substrate.create_snapshot()
    assert snapshot.verify_integrity()

    # Change rates
    substrate.r.fill_(0.99)
    assert not torch.all(substrate.r == snapshot.rate_vector)

    # Restore
    substrate.restore_snapshot(snapshot)
    assert torch.all(substrate.r == snapshot.rate_vector)


def test_clone_independence():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    sub1 = RecurrentSubstrate(topology=topology, seed=42)
    sub2 = sub1.clone_substrate()

    obs1 = torch.ones(155)
    obs2 = torch.zeros(155)

    sub1.forward_step(obs1)
    sub2.forward_step(obs2)

    assert not torch.all(sub1.r == sub2.r)
