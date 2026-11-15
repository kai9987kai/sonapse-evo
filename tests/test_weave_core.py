"""Core NeuroWeave runtime: compartments, soma populations, transmission, switches, persistence, invariants."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sonapse_evo.neuroweave import (CELL_FIELDS, EDGE_FIELDS, MECHANISM_IDS, MECHANISMS, SOMA_ROLES, Mechanism, ModulatorState,
                                    Stage, WeaveConfig, build_weave_substrate, canonical_motor_head, run_weave_episode)
from sonapse_evo.neuroweave.ecology import make_environment
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.topology_state import TopologyState

CORE = [m for m in MECHANISM_IDS if MECHANISMS[m].is_core]
PLUGINS = [m for m in MECHANISM_IDS if MECHANISMS[m].is_plugin]


def _legacy():
    return RecurrentSubstrate(topology=TopologyState(768, 3, seed=42), seed=42)


def _drive(sub, steps=12, seed=0, reward=0.4):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        sub.deliver_feedback(reward=reward, issued_action=[0.4, 0.1, 0, 0, 0], intended_action=[0.4, 0.1, 0, 0, 0])
        sub.forward_step(torch.rand(155, generator=gen) * 0.6)
    return sub


@pytest.fixture(scope="module")
def trained_max():
    sub = build_weave_substrate(seed=42, profile="max", strict=True)
    env = make_environment("neuroweave_1200x660", seed=5, max_steps=60)
    run_weave_episode(env, sub, canonical_motor_head(), seed=5)
    return sub


def test_config_profiles_and_serialisation():
    cfg = WeaveConfig.from_profile("max", seed=7)
    assert cfg.enabled_ids() == MECHANISM_IDS
    again = WeaveConfig.from_dict(cfg.to_dict())
    assert again.digest() == cfg.digest()
    ablated = cfg.without("pair_stdp")
    assert not ablated.is_enabled("pair_stdp") and cfg.is_enabled("pair_stdp")
    with pytest.raises(KeyError):
        WeaveConfig(enabled={"no_such_mechanism": True})
    assert not WeaveConfig.legacy().any_enabled()


@pytest.mark.parametrize("mechanism_id", MECHANISM_IDS)
def test_every_mechanism_runs_bounded_and_finite(mechanism_id):
    sub = _drive(build_weave_substrate(seed=42, enable=[mechanism_id], strict=True), steps=8)
    assert torch.isfinite(sub.r).all() and float(sub.r.min()) >= 0.0 and float(sub.r.max()) <= 1.0
    assert not sub.weave_faults
    for name in CELL_FIELDS:
        assert torch.isfinite(getattr(sub.cell, name)).all(), name
    for name in EDGE_FIELDS:
        assert getattr(sub.syn, name).shape[0] == sub.syn.n_edges, name


@pytest.mark.parametrize("mechanism_id", ["multi_soma", "dendritic_compartments", "membrane_spiking", "axonal_branches",
                                           "sparse_distributed_code", "synaptic_delays", "local_predictive_coding"])
def test_compartments_participate_in_computation(mechanism_id):
    """Enabling a cell mechanism changes the emitted rates: it is part of the computation, not decoration."""
    legacy, sub = _legacy(), build_weave_substrate(seed=42, enable=[mechanism_id], strict=True)
    gen = torch.Generator().manual_seed(3)
    diff = 0.0
    for _ in range(10):
        obs = torch.rand(155, generator=gen)
        diff = max(diff, float((legacy.forward_step(obs) - sub.forward_step(obs)).abs().max()))
    assert diff > 1e-4


def test_multi_compartment_state_invariants(trained_max):
    cell, sub = trained_max.cell, trained_max
    assert cell.soma_matrix(sub.r).shape == (len(SOMA_ROLES), 768) and cell.dendrite_matrix().shape == (4, 768)
    assert float(cell.soma_gating.min()) >= 0.65 and float(cell.soma_gating.max()) <= 1.35
    assert float(cell.threshold.min()) >= cell.p.theta_min - 1e-6 and float(cell.threshold.max()) <= cell.p.theta_max + 1e-6
    assert set(torch.unique(cell.spike).tolist()) <= {0.0, 1.0}
    assert float(cell.refractory.min()) >= 0.0 and float(cell.resource.min()) >= 0.0 and float(cell.resource.max()) <= 1.0
    assert float(cell.homeo.min()) >= cell.p.gain_min - 1e-6 and float(cell.homeo.max()) <= cell.p.gain_max + 1e-6
    assert float(cell.myelin.min()) >= 0.2 and float(cell.myelin.max()) <= 0.98
    assert float(cell.pred_surprise.min()) >= 0.0 and float(cell.pred_surprise.max()) <= 2.0
    assert float(cell.release.min()) >= 0.0
    assert cell.branches.shape == (cell.p.n_branches, 768)
    assert float(sub.syn.routing_gain.min()) >= 0.62 and float(sub.syn.routing_gain.max()) <= 1.14
    assert int(sub.syn.delay.min()) >= 0 and int(sub.syn.delay.max()) <= 3


def test_primary_secondary_soma_relationship():
    """The primary soma is the legacy rate; the secondary soma is its persistent low-pass trace."""
    sub = build_weave_substrate(seed=42, enable=["multi_soma"], strict=True)
    for _ in range(40):
        sub.forward_step(torch.full((155,), 0.5))
    assert torch.allclose(sub.cell.soma_matrix(sub.r)[0], sub.r)
    assert float((sub.cell.soma_secondary - sub.r).abs().max()) < 0.05          # converges to the steady primary rate
    sub.forward_step(torch.zeros(155))
    assert float(sub.cell.soma_secondary.mean()) > float(sub.r.mean()) * 0.9     # and persists when input stops


def test_more_axonal_branches_can_be_configured():
    sub = build_weave_substrate(seed=42, enable=["axonal_branches"], params={"cell": {"n_branches": 4}}, strict=True)
    _drive(sub, 5)
    assert sub.cell.branches.shape[0] == 4 and int(sub.syn.branch.max()) <= 3 and torch.isfinite(sub.cell.branches).all()


def test_weight_updates_are_bounded_and_sign_preserving():
    sub = build_weave_substrate(seed=42, enable=["reward_gated_plasticity"], strict=True)
    before = sub.effective_weights().clone()
    sub.apply_weight_update(torch.full((sub.syn.n_edges,), 50.0) * sub.syn.sign)
    grown = sub.effective_weights()
    assert float(grown.abs().max()) <= sub.w_max + 1e-6
    sub.apply_weight_update(torch.full((sub.syn.n_edges,), -500.0) * sub.syn.sign)
    shrunk = sub.effective_weights()
    assert torch.all(torch.sign(shrunk) * torch.sign(before) >= 0)              # no synapse changed sign
    assert float(shrunk.abs().max()) <= 1e-6                                     # magnitude floor is zero, never negative
    assert torch.equal(sub.base_w, RecurrentSubstrate(topology=sub.topology, seed=42).base_w)   # original buffer untouched


def test_learned_state_is_inert_when_mechanisms_are_switched_off():
    sub = _drive(build_weave_substrate(seed=42, enable=["reward_gated_plasticity", "structural_plasticity"], strict=True,
                                       params={"structural_plasticity": {"period": 4}}), steps=12, reward=1.0)
    assert float(sub.syn.w_delta.abs().sum()) > 0 and sub.syn.n_grown > 0
    sub.set_mechanism("reward_gated_plasticity", False)
    sub.set_mechanism("structural_plasticity", False)
    legacy = _legacy()
    sub.reset_state()
    obs = torch.rand(155)
    assert torch.equal(sub.forward_step(obs), legacy.forward_step(obs))


def test_runtime_switching_and_freeze():
    sub = build_weave_substrate(seed=42, strict=True)
    sub.set_mechanism("fast_weights", True)
    assert "fast_weights" in sub.mechanisms and sub.weave_active
    sub.set_mechanism("fast_weights", False)
    assert "fast_weights" not in sub.mechanisms and not sub.weave_active
    sub = _drive(build_weave_substrate(seed=42, enable=["reward_gated_plasticity"], strict=True), 6)
    sub.freeze_learning(True)
    frozen = sub.syn.w_delta.clone()
    _drive(sub, 6, seed=1)
    assert torch.equal(sub.syn.w_delta, frozen)                                   # effect kept, no further learning


def test_feedback_is_lagged_by_one_step():
    sub = build_weave_substrate(seed=42, enable=["neuromodulatory_field"], strict=True)
    sub.deliver_feedback(reward=1.0)
    assert sub.signals.reward == 0.0
    sub.forward_step(torch.rand(155))
    assert sub.signals.reward == 1.0
    sub.forward_step(torch.rand(155))
    assert sub.signals.reward == 0.0


def test_clone_and_snapshot_continue_exactly(trained_max):
    clone = trained_max.clone_substrate()
    snap = trained_max.create_snapshot()
    reference = trained_max.clone_substrate()
    gen = torch.Generator().manual_seed(9)
    for _ in range(6):
        obs = torch.rand(155, generator=gen) * 0.5
        for s in (clone, reference):
            s.deliver_feedback(reward=0.3, issued_action=[0.2, 0, 0, 0, 0], intended_action=[0.2, 0, 0, 0, 0])
        assert torch.equal(clone.forward_step(obs), reference.forward_step(obs))
    restored = trained_max.clone_substrate()
    restored.forward_step(torch.rand(155))
    restored.restore_snapshot(snap)
    probe = trained_max.clone_substrate()
    obs = torch.rand(155)
    assert torch.equal(restored.forward_step(obs), probe.forward_step(obs))


def test_node_lesion_silences_neurons_and_clears():
    sub = build_weave_substrate(seed=42, strict=True)
    assert sub.set_node_lesions([5, 6, 7, 9999]) == 3
    sub.forward_step(torch.rand(155))
    assert float(sub.r[5:8].abs().sum()) == 0.0
    sub.set_node_lesions(None)
    assert not sub.weave_active


def test_graceful_degradation_and_strict_mode(monkeypatch):
    class Broken(Mechanism):
        mechanism_id, stage = "fast_weights", Stage.CONTEXT

        def begin_step(self, ctx):
            raise RuntimeError("boom")

    monkeypatch.setattr(type(MECHANISMS["fast_weights"]), "load_class", lambda self: Broken)
    sub = build_weave_substrate(seed=42, enable=["fast_weights"], strict=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sub.forward_step(torch.rand(155))
    assert caught and sub.weave_faults[0]["mechanism"] == "fast_weights" and "fast_weights" not in sub.mechanisms
    assert torch.isfinite(sub.forward_step(torch.rand(155))).all()                # the model keeps running
    assert sub.events.recent("weave.fault")
    strict = build_weave_substrate(seed=42, enable=["fast_weights"], strict=True)
    with pytest.raises(RuntimeError):
        strict.forward_step(torch.rand(155))


def test_modulator_bounds_and_legacy_bridge():
    m = ModulatorState(dopamine=5.0, serotonin=-3.0, noradrenaline=9.0).clamp()
    assert m.dopamine == 1.0 and m.serotonin == 0.0 and m.noradrenaline == 1.0
    assert ModulatorState().is_neutral()
    legacy = ModulatorState(dopamine=0.4, acetylcholine=0.3, noradrenaline=0.2).to_legacy()     # existing 4-field state
    assert abs(legacy.compute_plasticity_multiplier() - (0.7 * 0.4 + 0.3 * 0.3) * 1.2) < 1e-6
    assert abs(ModulatorState.from_legacy(legacy).dopamine - 0.4) < 1e-9


def test_max_profile_signals_stay_bounded(trained_max):
    sig = trained_max.signals
    for name, (lo, hi) in ModulatorState.BOUNDS.items():
        assert lo <= getattr(sig.modulators, name) <= hi
    for name, (lo, hi) in type(sig.affect).BOUNDS.items():
        assert lo <= getattr(sig.affect, name) <= hi
    assert 0.0 <= sig.exploration <= 1.0 and 0.0 <= sig.persistence <= 0.9 and 0.1 <= sig.plasticity_gain <= 4.0
    assert len(trained_max.mechanisms) == len(PLUGINS) and not trained_max.weave_faults
    telemetry = trained_max.weave_telemetry()
    assert set(telemetry["mechanisms"]) == set(PLUGINS) and telemetry["counters"]["weave_steps"] > 0


def test_population_scales_beyond_the_default():
    big = build_weave_substrate(total_neurons=3072, seed=1, enable=CORE, strict=True)
    _drive(big, 3)
    assert big.total_neurons == 3072 and torch.isfinite(big.r).all()
    small = build_weave_substrate(total_neurons=240, seed=1, topology_kind="multi_region", enable=CORE, strict=True)
    _drive(small, 3)
    assert small.total_neurons == 240 and small.layout.num_regions == 10 and torch.isfinite(small.r).all()
