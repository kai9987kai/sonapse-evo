"""Regression tests for the defects found by the independent review of the NeuroWeave layer.

Each test reproduces one reported failure and asserts the corrected behaviour.
"""
from __future__ import annotations

import copy
import json
import threading
import time

import numpy as np
import pytest
import torch

from sonapse_evo.evaluation.rigl import AuthenticRigL
from sonapse_evo.neuroweave import (MECHANISMS, Provenance, WeaveConfig, WeaveSubstrate, build_weave_substrate, canonical_motor_head,
                                    run_weave_episode)
from sonapse_evo.neuroweave.collective import MicroCollective
from sonapse_evo.neuroweave.connectome import apply_lesion_mask, build_lesion_mask, build_topology, custom_layout
from sonapse_evo.neuroweave.ecology import make_environment
from sonapse_evo.neuroweave.experiments import run_catalog_experiment
from sonapse_evo.neuroweave.portability import (build_runner_model, export_model_package, import_model_package,
                                                inspect_model_package, reference_inference, validate_model_package)
from sonapse_evo.neuroweave.portability.package import _digest
from sonapse_evo.neuroweave.promotion import PromotionLedger
from sonapse_evo.neuroweave.structure import grow_substrate
from sonapse_evo.observatory.weave_state import WeaveObservatoryState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.topology_state import TopologyState


def _drive(sub, steps, reward=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        sub.deliver_feedback(reward=reward, issued_action=[0.3, 0, 0, 0, 0], intended_action=[0.3, 0, 0, 0, 0])
        sub.forward_step(torch.rand(155, generator=gen) * 0.6)
    return sub


def _continue_equal(a, b, steps=8, seed=5):
    gen = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(steps):
        obs = torch.rand(155, generator=gen)
        for s in (a, b):
            s.deliver_feedback(reward=0.4, issued_action=[0.3, 0, 0, 0, 0], intended_action=[0.3, 0, 0, 0, 0])
        worst = max(worst, float((a.forward_step(obs) - b.forward_step(obs)).abs().max()))
    return worst


@pytest.fixture(scope="module")
def package():
    return json.loads(json.dumps(export_model_package(build_weave_substrate(seed=42, enable=["reward_gated_plasticity"]))))


# ------------------------------------------------------------------ packages
@pytest.mark.parametrize("name,mutate", [
    ("bias with a wrong shape", lambda p: p["substrate"]["bias"].__setitem__("shape", [768, 1])),
    ("missing num_regions", lambda p: p["topology"].pop("num_regions")),
    ("missing microsteps", lambda p: p["substrate"].pop("microsteps")),
    ("missing region_names", lambda p: p["topology"].pop("region_names")),
    ("B values shorter than rows", lambda p: p["substrate"]["B"]["values"].update(
        data=p["substrate"]["B"]["values"]["data"][:-1], shape=[len(p["substrate"]["B"]["values"]["data"]) - 1])),
    ("huge sensory_dim", lambda p: p["substrate"].__setitem__("sensory_dim", 10 ** 7)),
    ("NaN in the lesion mask", lambda p: p["substrate"]["lesion_mask"]["data"].__setitem__(0, float("nan"))),
    ("NaN in B", lambda p: p["substrate"]["B"]["values"]["data"].__setitem__(0, float("nan"))),
    ("negative microsteps", lambda p: p["substrate"].__setitem__("microsteps", -5)),
    ("NaN current cap", lambda p: p["substrate"].__setitem__("current_cap", float("nan"))),
    ("plastic edge out of range", lambda p: p["substrate"]["plastic"].update(
        pre={"__t__": "i64", "shape": [1], "data": [5]}, post={"__t__": "i64", "shape": [1], "data": [99999]},
        w={"__t__": "f32", "shape": [1], "data": [0.1]})),
    ("weave section is a list", lambda p: p.__setitem__("weave", [1])),
    ("enabled is a list", lambda p: p["weave"]["config"].__setitem__("enabled", [1])),
    ("delay out of range", lambda p: p["weave"]["synapses"]["delay"]["data"].__setitem__(0, 9)),
])
def test_validation_refuses_what_import_cannot_build(package, name, mutate):
    bad = copy.deepcopy(package)
    mutate(bad)
    if isinstance(bad.get("integrity"), dict):
        bad["integrity"]["digest"] = _digest(bad)                 # a carefully forged package, digest intact
    assert not validate_model_package(bad)["ok"], name
    with pytest.raises(ValueError):                               # never any other exception type
        import_model_package(bad)


def test_inspection_never_raises_and_staged_import_is_cleared(package):
    for text in ("[1,2]", '{"weave": [1]}', '{"weave": {"config": {"enabled": [1]}}}', '{"topology": 5}'):
        assert inspect_model_package(text)["validation"]["ok"] is False
    state = WeaveObservatoryState(seed=42)
    assert state.validate_import(json.dumps(package))["ok"] and state.staged_import is not None
    state.validate_import("[1]")
    assert state.staged_import is None and state.staged_report["ok"] is False


def test_standalone_runner_matches_a_lesioned_learned_model_with_node_lesions():
    sub = _drive(build_weave_substrate(seed=3, enable=["reward_gated_plasticity"]), 40)
    apply_lesion_mask(sub, build_lesion_mask(sub.topology, "random_edges", 0.3, seed=1))
    sub.set_node_lesions([5, 6, 7])
    pkg = json.loads(json.dumps(export_model_package(sub)))
    model = build_runner_model(pkg)
    twin, _ = import_model_package(pkg)
    twin.freeze_learning(True)
    observations = [np.random.default_rng(i).random(155) * 0.5 for i in range(5)]
    rates = np.stack([twin.forward_step(torch.as_tensor(o, dtype=torch.float32)).numpy() for o in observations])
    assert float(np.abs(reference_inference(model, observations) - rates).max()) < 1e-5


# ------------------------------------------------------------------- core
def test_restore_and_clone_after_structural_growth_are_exact():
    ids = ["structural_plasticity", "tiled_execution", "reward_gated_plasticity", "meta_plasticity", "synaptic_delays",
           "transmitter_inference", "structural_encoder", "connectivity_attention"]
    sub = build_weave_substrate(seed=1, enable=ids, params={"structural_plasticity": {"period": 5, "net_growth": 8}}, strict=True)
    _drive(sub, 8)
    snapshot, reference = sub.create_snapshot(), sub.clone_substrate()
    _drive(sub, 22, seed=1)
    sub.restore_snapshot(snapshot)
    assert _continue_equal(sub, reference, steps=12) == 0.0 and sub.weave_faults == []
    imported, _ = import_model_package(json.loads(json.dumps(export_model_package(sub))))
    assert _continue_equal(sub.clone_substrate(), imported) == 0.0


def test_neurogenesis_never_writes_into_a_shared_topology():
    topo = TopologyState(seed=4)
    digest = topo.compute_digest()
    twin = RecurrentSubstrate(topology=topo, seed=4)
    sub = WeaveSubstrate(topo, weave=WeaveConfig.with_mechanisms("neurogenesis"))
    for k in range(4):
        sub.mechanisms["neurogenesis"].run_recruitment(step=k * 1000, force=True)
    assert topo.compute_digest() == digest and twin.create_snapshot().topology_digest == digest


def test_in_place_anchor_edits_keep_learning_sign_preserving():
    sub = build_weave_substrate(seed=7, enable=["reward_gated_plasticity"])
    AuthenticRigL(sub, seed=1).prune_and_regrow()                # existing code that edits base_w in place
    _drive(sub, 30)
    anchor, effective = sub.anchor_weights(), sub.effective_weights()
    flipped = (torch.sign(effective) != torch.sign(anchor)) & (anchor != 0) & (effective != 0)
    assert int(flipped.sum()) == 0


def test_rollback_to_a_legacy_snapshot_restores_the_configuration():
    sub = build_weave_substrate(seed=2)
    snapshot = sub.create_snapshot()
    sub.set_mechanism("quantised_synapses", True)
    sub.set_mechanism("tiled_execution", True)
    sub.set_node_lesions([3])
    sub.restore_snapshot(snapshot)
    assert not sub.weave_active and sub.weight_transforms == {} and sub.traversal is None and sub.node_mask is None
    legacy = RecurrentSubstrate(topology=sub.topology, seed=2)
    obs = torch.rand(155)
    sub.reset_state()
    assert torch.equal(sub.forward_step(obs), legacy.forward_step(obs))


def test_config_and_live_mechanisms_never_drift(monkeypatch):
    sub = build_weave_substrate(seed=1, enable=["multi_region_routing"])
    sub.set_mechanism("multi_region_routing", True, adapt=True)
    assert sub.mechanisms["multi_region_routing"].params["adapt"] is True

    class Broken(MECHANISMS["fast_weights"].load_class()):
        def begin_step(self, ctx):
            raise RuntimeError("boom")

    monkeypatch.setattr(type(MECHANISMS["fast_weights"]), "load_class", lambda self: Broken)
    faulty = build_weave_substrate(seed=1, enable=["fast_weights", "multi_soma"])
    with pytest.warns(RuntimeWarning):
        faulty.forward_step(torch.rand(155))
    assert not faulty.weave.is_enabled("fast_weights") and "fast_weights" not in faulty.mechanisms
    assert "fast_weights" not in faulty.clone_substrate().mechanisms


def test_legacy_buffer_changes_keep_learned_edge_state():
    sub = _drive(build_weave_substrate(seed=3, enable=["reward_gated_plasticity"]), 10)
    learned = sub.syn.w_delta[: sub.syn.n_base].clone()
    sub.plastic_pre, sub.plastic_post, sub.plastic_w = torch.tensor([1]), torch.tensor([2]), torch.tensor([0.1])
    sub._sync_synapses()
    assert sub.syn.n_plastic == 1 and torch.equal(sub.syn.w_delta[: sub.syn.n_base], learned)


def test_frozen_evaluation_does_not_adapt_slow_cell_state():
    core = [m for m, s in MECHANISMS.items() if s.is_core]
    sub = _drive(build_weave_substrate(seed=42, enable=core), 20)
    sub.freeze_learning(True)
    slow = {n: getattr(sub.cell, n).clone() for n in ("homeo", "astro", "myelin", "rate_ema", "utility", "trace", "tm_slow")}
    _drive(sub, 10, seed=3)
    assert all(torch.equal(getattr(sub.cell, n), v) for n, v in slow.items())


def test_non_finite_feedback_is_rejected_or_ignored():
    sub = build_weave_substrate(seed=1, enable=["efference_self_model", "reward_gated_plasticity"], strict=True)
    with pytest.raises(ValueError):
        sub.deliver_feedback(reward=float("nan"))
    for i in range(6):
        sub.deliver_feedback(reward=0.3, intended_action=[0.5, 0, 0, 0, 0], issued_action=[0.5, 0, 0, 0, 0],
                             predicted_outcome=[0.5, 0.0], observed_outcome=[float("nan"), 0.2] if i == 2 else [0.5, 0.0, 0.1])
        sub.forward_step(torch.rand(155))
    assert torch.isfinite(sub.r).all() and torch.isfinite(sub.syn.w_delta).all() and sub.weave_faults == []
    assert np.isfinite(sub.signals.control_reliability)


# ---------------------------------------------------------------- systems
def test_sensor_payloads_are_validated_before_anything_is_written():
    sub = build_weave_substrate(seed=1, enable=["multimodal_sensory", "cross_modal_fusion"], strict=True)
    hub = sub.mechanisms["multimodal_sensory"]
    values, provenance = hub.values.copy(), dict(hub.manifest.provenance)
    for payload in ({"olfactory": 0.5, "telepathy": 1.0}, {"olfactory": 0.5, "gustatory": float("nan")}):
        with pytest.raises((KeyError, ValueError)):
            hub.ingest(payload)
    with pytest.raises(ValueError):
        hub.ingest({"olfactory": 0.5}, reliability={"olfactory": float("nan")})
    with pytest.raises(ValueError):
        hub.ingest_frame(np.full((10, 10, 3), np.nan))
    assert np.array_equal(values, hub.values) and provenance == hub.manifest.provenance and hub.visual.frames == 0
    with pytest.raises(ValueError):
        hub.set_mode("visual_motion", "simulated")              # no simulator exists for this channel
    hub.set_mode("thermosensation", "simulated")
    _drive(sub, 4)
    assert _continue_equal(sub, sub.clone_substrate(), steps=4) == 0.0


def test_micro_collective_adapts_after_a_distribution_shift():
    mc = MicroCollective(seed=1, lr=0.05)
    for t in range(600):
        mc.observe(np.array([0.6 * np.sin(0.2 * t + 0.4 * i) for i in range(16)]))
    for t in range(600):
        mc.observe(np.array([0.6 * np.cos(0.05 * t + 0.9 * i) for i in range(16)]))
    assert mc.skill() > 0.5 and mc.lr > mc.lr_floor and mc.consecutive_rollbacks <= mc.max_consecutive_rollbacks


def test_collective_and_specialist_accept_larger_dimensions_with_the_hub():
    sub = build_weave_substrate(seed=1, enable=["multimodal_sensory", "micro_collective", "specialist_graft"],
                                params={"micro_collective": {"dim": 24}, "specialist_graft": {"dim": 24}}, strict=True)
    _drive(sub, 5)
    assert sub.weave_faults == []


def test_ecology_sanitises_actions_and_keeps_agents_inside_the_arena():
    env = make_environment("neuroweave_1200x660", seed=1)
    obs, _, _, _ = env.step([float("nan"), float("inf")])
    assert np.all(np.isfinite(obs))
    arena = make_environment("multi_agent", seed=1, n_agents=10, preset="small_arena")
    assert max(float(np.abs(o).max()) for o in arena.reset(1)) <= 1.0


def test_canonical_pathway_never_becomes_self_loops():
    layout = custom_layout([{"name": "out", "role": "motor", "fraction": 1}, {"name": "in", "role": "sensory", "fraction": 2}], 200)
    with pytest.raises(ValueError):
        build_topology("uniform_random", 200, layout=layout)
    topo = build_topology("uniform_random", 200, layout=layout, canonical_pathway=False)
    assert bool(np.all(topo.base_pre != topo.base_post))


def test_neurogenesis_protects_the_read_out_and_growth_keeps_all_edges():
    sub = build_weave_substrate(total_neurons=96, seed=1, topology_kind="multi_region", enable=["neurogenesis"],
                                params={"neurogenesis": {"batch": 200, "max_recruits": 500, "protect_motor": 64}})
    readout = sub.motor_offset if sub.motor_offset + 64 <= 96 else 0
    sub.mechanisms["neurogenesis"].run_recruitment(step=0, force=True)
    recruits = torch.nonzero(sub.mechanisms["neurogenesis"].is_recruit).squeeze(-1)
    assert not bool(((recruits >= readout) & (recruits < readout + 64)).any()) and not bool((recruits < 3).any())
    source = _drive(build_weave_substrate(seed=3, enable=["structural_plasticity", "reward_gated_plasticity"],
                                          params={"structural_plasticity": {"period": 4, "net_growth": 6}}), 16)
    grown, receipt = grow_substrate(source, 16, seed=1)
    assert receipt["carried_grown_synapses"] == source.syn.n_grown == grown.syn.n_grown > 0


def test_memory_labels_follow_the_least_trusted_source_and_respect_freezing():
    sub = build_weave_substrate(seed=1, enable=["memory_ecology", "multimodal_sensory"], strict=True)
    run_weave_episode(make_environment("neuroweave_1200x660", seed=1, max_steps=20), sub, canonical_motor_head(), seed=1)
    eco = sub.mechanisms["memory_ecology"]
    counts = eco.telemetry()["provenance"]
    assert counts["direct_sensory"] == 0 and counts["synthetic_simulation"] > 0 and eco.telemetry()["verified"] == 0
    sub.freeze_learning(True)
    confidences = {rid: r.confidence for rid, r in eco.episodic.records.items()}
    uses = {rid: r.uses for rid, r in eco.episodic.records.items()}
    _drive(sub, 5)
    assert confidences == {rid: r.confidence for rid, r in eco.episodic.records.items()}
    assert uses == {rid: r.uses for rid, r in eco.episodic.records.items()}


# -------------------------------------------------------------------- lab
def test_quick_runs_are_smoke_tests_and_never_promotable():
    receipt = run_catalog_experiment("quantised_synapses", quick=True, seeds=(1, 2, 3, 4, 5))
    assert receipt["smoke"] and receipt["evidence_tier"] == "SMOKE_TEST" and receipt["verdict"] == "inconclusive"
    assert PromotionLedger().propose(["quantised_synapses"], receipt)["status"] == "REJECTED"


def test_promotion_ledger_is_atomic_on_write_failure(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    ledger = PromotionLedger(path=str(blocker / "ledger.json"))
    receipt = {"experiment_id": "x", "result_digest": "d", "verdict": "supported", "evidence_tier": "PAIRED_EXPERIMENT",
               "spec": {"candidate": ["pair_stdp"]}, "matched_conditions": {"candidate_cost_ratio_measured": 1.0},
               "results": {"summary": {"candidate": {"faults": 0}}}}
    with pytest.raises(OSError):
        ledger.propose(["pair_stdp"], receipt, regression={"legacy_identity": True})
    assert ledger.stable == [] and ledger.history == [] and ledger.decisions == []
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps({"format": "SONAPSE_PROMOTION_LEDGER", "stable": [], "history": [{}], "decisions": [], "failures": []}))
    assert PromotionLedger(path=str(corrupt)).rollback()["status"] == "NOTHING_TO_ROLL_BACK"


def test_observatory_rebuild_is_atomic_and_faults_never_stop_the_original_thread():
    state = WeaveObservatoryState(seed=42)
    before = (state.substrate, dict(state.weave_config))
    with pytest.raises(KeyError):
        state.rebuild(seed=7, environment="no_such_environment", total_neurons=800)
    with pytest.raises(ValueError):
        state.rebuild(seed=1, total_neurons=60, topology_kind="ring_lattice")       # smaller than the motor read-out
    assert state.substrate is before[0] and state.weave_config == before[1]
    state.set_profile("legacy")                                                       # still usable
    state.weave_running = True
    state.weave_step = lambda n: (_ for _ in ()).throw(RuntimeError("model failure"))
    ticks = state.step_idx
    worker = threading.Thread(target=lambda: [state.tick_simulation() for _ in range(3)], daemon=True)
    worker.start()
    worker.join(timeout=30)
    assert state.step_idx == ticks + 3 and state.weave_running is False and state.weave_faults
