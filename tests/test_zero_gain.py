from __future__ import annotations

import pytest
import torch
import numpy as np

from sonapse_evo.contracts.graft_manifest import GraftSpec
from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.grafts.low_rank_bridge import LowRankBridge
from sonapse_evo.grafts.recurrent_microcircuit import RecurrentMicrocircuit
from sonapse_evo.grafts.biological_graft import BiologicalMotifGraft


def test_zero_gain_low_rank_bridge():
    total_neurons = 768
    spec = GraftSpec(
        graft_id="test_bridge",
        graft_type="low_rank",
        source_neurons=tuple(range(32)),
        target_neurons=tuple(range(32, 64)),
        rank=8,
        max_gain=0.10,
        ramp_steps=50,
    )
    bridge = LowRankBridge(spec, total_neurons)
    bridge.reset_ramp(start_gain=0.0)

    # With zero gain, output must be exactly zeros
    rates = torch.rand(total_neurons)
    current = bridge(rates.unsqueeze(0)).squeeze(0)
    assert torch.all(current == 0.0)
    assert float(torch.norm(current.detach())) == 0.0


def test_zero_gain_recurrent_microcircuit():
    total_neurons = 768
    spec = GraftSpec(
        graft_id="test_microcircuit",
        graft_type="recurrent",
        source_neurons=tuple(range(32)),
        target_neurons=tuple(range(32, 64)),
        rank=8,
        max_gain=0.10,
        recurrent_state_dim=16,
    )
    micro = RecurrentMicrocircuit(spec, total_neurons)
    micro.reset_ramp(start_gain=0.0)

    rates = torch.rand(total_neurons)
    current = micro(rates.unsqueeze(0)).squeeze(0)
    assert torch.all(current == 0.0)


def test_zero_gain_biological_motif():
    total_neurons = 768
    spec = GraftSpec(
        graft_id="test_motif",
        graft_type="biological",
        source_neurons=tuple(range(32)),
        target_neurons=tuple(range(32, 64)),
        max_gain=0.10,
    )
    bio = BiologicalMotifGraft(spec, total_neurons)
    bio.reset_ramp(start_gain=0.0)

    rates = torch.rand(total_neurons)
    current = bio(rates.unsqueeze(0)).squeeze(0)
    assert torch.all(current == 0.0)


def test_substrate_dynamics_identical_with_zero_gain_graft():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    sub1 = RecurrentSubstrate(topology=topology, seed=42)
    sub2 = sub1.clone_substrate()

    spec = GraftSpec(
        graft_id="test_bridge",
        graft_type="low_rank",
        source_neurons=tuple(range(32)),
        target_neurons=tuple(range(32, 64)),
        rank=8,
        max_gain=0.10,
    )
    bridge = LowRankBridge(spec, 768)
    bridge.reset_ramp(start_gain=0.0)

    obs = torch.rand(155)

    # Step sub1 without graft
    r1 = sub1.forward_step(obs, aux_current=None)

    # Step sub2 with zero-gain graft
    zero_aux = bridge(sub2.r.unsqueeze(0)).squeeze(0)
    r2 = sub2.forward_step(obs, aux_current=zero_aux)

    # Dynamics must be bitwise or float-identical
    diff = float(torch.max(torch.abs(r1 - r2)).item())
    assert diff < 1e-7
