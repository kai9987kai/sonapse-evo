from __future__ import annotations

import torch
import numpy as np
import pytest

from sonapse_evo.substrate.plasticity import (
    NeuromodulatoryState,
    NeuromodulatedPlasticityEngine,
)


def test_neuromodulatory_state_multiplier():
    state = NeuromodulatoryState(dopamine_rpe=1.0, acetylcholine_unc=0.5, noradrenaline_gain=1.0)
    m = state.compute_plasticity_multiplier()
    # m = (0.7 * 1.0 + 0.3 * 0.5) * 1.0 = 0.85
    assert abs(m - 0.85) < 1e-4

    # Test clipping
    state_extreme = NeuromodulatoryState(dopamine_rpe=10.0, noradrenaline_gain=2.0)
    assert state_extreme.compute_plasticity_multiplier() == 2.0


def test_plasticity_engine_dales_law_and_fan_in():
    engine = NeuromodulatedPlasticityEngine(num_neurons=64, eta=0.05)
    pre = torch.rand(64)
    post = torch.rand(64)

    # Run several steps of plasticity
    for _ in range(5):
        stats = engine.step_plasticity(pre, post, reward=1.0, uncertainty=0.2)
        assert "mean_delta" in stats
        assert "weight_norm" in stats

    # Invariant: Dale's Law
    # Neuron signs: [::5] are -1.0 (inhibitory), others +1.0 (excitatory)
    for i in range(64):
        if engine.neuron_signs[i] > 0:
            assert torch.all(engine.plastic_w[i, :] >= 0.0), f"Excitatory neuron {i} has negative weight"
        else:
            assert torch.all(engine.plastic_w[i, :] <= 0.0), f"Inhibitory neuron {i} has positive weight"

    # Invariant: Fan-in clipping (column sum <= 2.0)
    col_norms = torch.sum(torch.abs(engine.plastic_w), dim=0)
    assert torch.all(col_norms <= 2.0001), f"Fan-in bound violated: max {torch.max(col_norms)}"


def test_plastic_current_and_reset():
    engine = NeuromodulatedPlasticityEngine(num_neurons=64, eta=0.01)
    rates = torch.rand(64)
    current = engine.compute_plastic_current(rates)
    assert current.shape == (64,)

    engine.step_plasticity(rates, rates, reward=0.5)
    assert torch.norm(engine.eligibility) > 0.0

    engine.reset_traces()
    assert torch.norm(engine.eligibility) == 0.0
    assert engine.modulators.dopamine_rpe == 0.0
