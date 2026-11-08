from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.contracts.graft_manifest import GraftSpec
from sonapse_evo.evolution.candidate_trainer import CandidateTrainer


def test_frozen_backbone_during_graft_training():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, seed=42)

    orig_base_w = substrate.base_w.clone()
    orig_bias = substrate.bias.clone()
    orig_B = substrate.B.clone()

    spec = GraftSpec(
        graft_id="test_train_graft",
        graft_type="low_rank",
        source_neurons=tuple(range(32)),
        target_neurons=tuple(range(32, 64)),
        rank=8,
        max_gain=0.10,
    )

    trainer = CandidateTrainer(learning_rate=0.01)
    graft = trainer.instantiate_graft(spec, 768)
    initial_graft_params = [p.clone() for p in graft.parameters()]

    train_data = [(torch.randn(155), torch.tensor([0.2, -0.1, 0.4, 0.0, 0.5])) for _ in range(8)]
    metrics = trainer.train_candidate(graft, substrate, train_data, steps=10, batch_size=4)

    # Substrate weights must remain 100% identical
    assert torch.all(substrate.base_w == orig_base_w)
    assert torch.all(substrate.bias == orig_bias)
    assert torch.all(substrate.B == orig_B)

    # Graft parameters must have updated
    updated_graft_params = list(graft.parameters())
    param_diff = sum(float(torch.norm(p1.detach() - p2.detach())) for p1, p2 in zip(initial_graft_params, updated_graft_params))
    assert param_diff > 1e-4

    # Post-training gain must be reset to 0 for safe installation protocol
    assert graft.alpha == 0.0
