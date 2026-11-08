from __future__ import annotations

import pytest
import torch
import numpy as np

from sonapse_evo.substrate.neural_ports import create_uniform_ports
from sonapse_evo.diagnostics.intervention_probes import VirtualGradientProbe
from sonapse_evo.diagnostics.causal_attribution import CausalAttributionEngine


def test_virtual_shadow_gradient():
    probe = VirtualGradientProbe()
    source_rates = torch.tensor([0.8, 0.2, 0.5])
    target_grad = torch.tensor([0.1, -0.4])

    score = probe.evaluate_shadow_gradients(source_rates, target_grad)
    assert score > 0.0
    # True outer product norm
    outer = torch.ger(target_grad, source_rates)
    expected = float(torch.norm(outer, p="fro").item())
    assert abs(score - expected) < 1e-5


def test_port_ranking_non_empty():
    ports = create_uniform_ports(768, 3, port_width=32, seed=42)
    probe = VirtualGradientProbe()
    current_rates = torch.rand(768)
    loss_grad = torch.rand(768)

    ranked_pairs = probe.rank_port_pairs(ports, current_rates, loss_grad, top_k=5)
    assert len(ranked_pairs) == 5
    # Verify descending sort order
    for i in range(len(ranked_pairs) - 1):
        assert ranked_pairs[i][2] >= ranked_pairs[i + 1][2]


def test_causal_interventional_effect():
    engine = CausalAttributionEngine(num_paired_rollouts=4)

    # Synthetic rollout function where graft adds a constant +2.5 benefit
    def mock_rollout(seed: int, graft_on: bool) -> float:
        base = float(seed % 100)
        return base + (2.5 if graft_on else 0.0)

    tau_hat, on_returns, off_returns = engine.evaluate_interventional_effect(mock_rollout, base_seed=100)
    assert len(on_returns) == 4
    assert len(off_returns) == 4
    assert abs(tau_hat - 2.5) < 1e-5
