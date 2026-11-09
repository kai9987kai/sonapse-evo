from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import pytest

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.neural_ports import create_uniform_ports
from sonapse_evo.diagnostics.causal_intervention import PearlianInterventionEngine


def test_pearlian_clamp_port():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, sensory_dim=155, seed=42)
    ports = create_uniform_ports(768, 3, port_width=32, seed=42)
    engine = PearlianInterventionEngine(sample_obs_dim=155)

    obs = torch.randn(155)
    clamped_rates = engine.do_clamp_port(substrate, ports[0], clamp_value=0.88, obs=obs)

    # Check that the target port indices were set to clamp_value
    assert torch.all(torch.isclose(clamped_rates[list(ports[0].indices)], torch.tensor(0.88)))


def test_average_causal_effect():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, sensory_dim=155, seed=42)
    ports = create_uniform_ports(768, 3, port_width=32, seed=42)
    engine = PearlianInterventionEngine(sample_obs_dim=155)

    readout = nn.Linear(64, 5)
    ace = engine.compute_average_causal_effect(
        substrate=substrate,
        port=ports[0],
        action_readout=readout,
        val_control=0.0,
        val_treatment=1.0,
        num_samples=3,
    )

    assert isinstance(ace, float)
    assert ace >= 0.0


def test_causal_influence_matrix():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, sensory_dim=155, seed=42)
    ports = create_uniform_ports(768, 3, port_width=32, seed=42)
    engine = PearlianInterventionEngine(sample_obs_dim=155)

    readout = nn.Linear(64, 5)
    matrix = engine.compute_causal_influence_matrix(substrate, ports[:3], readout)

    assert matrix.shape == (3,)
    assert np.all(matrix >= 0.0)
