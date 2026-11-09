from __future__ import annotations

import torch
import numpy as np
import pytest

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.diagnostics.criticality import (
    CriticalityDiagnosticsEngine,
    CriticalityReport,
)


def test_criticality_measurement():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, sensory_dim=155, seed=42)
    engine = CriticalityDiagnosticsEngine(rollout_steps=10)
    obs = torch.randn(155)

    lle = engine.measure_lyapunov_exponent(substrate, obs, num_trials=2)
    assert isinstance(lle, float)
    assert not np.isnan(lle)
    assert not np.isinf(lle)


def test_evaluate_criticality_report():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, sensory_dim=155, seed=42)
    engine = CriticalityDiagnosticsEngine(rollout_steps=10)
    obs = torch.randn(155)

    report = engine.evaluate_criticality(substrate, obs)
    assert isinstance(report, CriticalityReport)
    assert report.regime in ["SUBCRITICAL_ORDER", "EDGE_OF_CHAOS_CRITICAL", "SUPERCRITICAL_CHAOTIC"]
    assert report.effective_rank > 0.0
    assert report.jacobian_spectral_radius >= 0.0


def test_tune_to_edge_of_chaos():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, sensory_dim=155, seed=42)
    engine = CriticalityDiagnosticsEngine(rollout_steps=10)
    obs = torch.randn(155)

    initial_norm = torch.norm(substrate.base_w).item()
    scale = engine.tune_to_edge_of_chaos(substrate, obs, target_lle=0.0)
    final_norm = torch.norm(substrate.base_w).item()

    assert 0.7 <= scale <= 1.3
    assert abs(final_norm - (initial_norm * scale)) < 1e-4
