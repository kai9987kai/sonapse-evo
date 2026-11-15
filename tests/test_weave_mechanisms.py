"""Mechanism-level tests: communication, plasticity, neuromodulation, prediction, structure, execution."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sonapse_evo.neuroweave import Provenance, build_weave_substrate
from sonapse_evo.neuroweave.comm import (PACKET_CONTRACTS, RepresentationKind, SignalPacket, compute_fingerprints, flyhash_code,
                                         vsa_bind, vsa_bundle, vsa_similarity)
from sonapse_evo.neuroweave.hardware import (available_backends, build_fpga_manifest, compact_state, from_q8_8, profile_substrate,
                                             quantise_int8, scaling_profile, to_q8_8, validate_fpga_manifest)
from sonapse_evo.neuroweave.neuromod import RECEPTORS, TRANSMITTER_CLASSES
from sonapse_evo.neuroweave.structure import grow_substrate
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.topology_state import TopologyState


def _legacy():
    return RecurrentSubstrate(topology=TopologyState(768, 3, seed=42), seed=42)


def _sub(*ids, params=None):
    return build_weave_substrate(seed=42, enable=list(ids), params=params, strict=True)


def _run(sub, steps=20, reward=0.5, seed=0, scale=0.6):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        sub.deliver_feedback(reward=reward, issued_action=[0.4, 0.1, 0, 0, 0], intended_action=[0.4, 0.1, 0, 0, 0])
        sub.forward_step(torch.rand(155, generator=gen) * scale)
    return sub


# ------------------------------------------------------------------ communication
def test_packets_have_contracts_and_round_trip():
    assert set(PACKET_CONTRACTS) == set(RepresentationKind) and len(RepresentationKind) == 16
    packet = SignalPacket(RepresentationKind.PHASE, np.array([0.5, -2.0, 3.0]), source="a", confidence=0.7, uncertainty=0.2,
                          provenance=Provenance.INTERNAL_PREDICTION).validate()
    assert float(packet.payload.min()) >= -1.0 and float(packet.payload.max()) <= 1.0          # clipped to the contract
    again = SignalPacket.from_dict(json.loads(json.dumps(packet.to_dict())))
    assert again.kind is RepresentationKind.PHASE and np.allclose(again.payload, packet.payload) and again.provenance is Provenance.INTERNAL_PREDICTION
    with pytest.raises(ValueError):
        SignalPacket(RepresentationKind.SCALAR, np.array([np.nan])).validate()
    with pytest.raises(ValueError):
        SignalPacket(RepresentationKind.SCALAR, np.array([0.1]), version="9.0").validate()
    with pytest.raises(ValueError):
        SignalPacket(RepresentationKind.SCALAR, np.array([0.1]), compartment="nucleus").validate()


def test_neurowave_bus_is_neutral_until_gated_and_carries_all_representations():
    legacy, sub = _legacy(), _sub("neurowave_packets")
    gen = torch.Generator().manual_seed(1)
    for _ in range(6):
        obs = torch.rand(155, generator=gen)
        assert torch.equal(sub.forward_step(obs), legacy.forward_step(obs))                     # zero-initialised fusion
    bus = sub.mechanisms["neurowave_packets"]
    kinds = {p.kind for p in bus.last_packets}
    assert kinds == set(RepresentationKind)
    assert all(p.version == "2.0" and p.timestamp >= 0 and 0 <= p.confidence <= 1 for p in bus.last_packets)
    assert {"vector", "phase", "uncertainty"} == set(next(p for p in bus.last_packets if p.kind is RepresentationKind.HYBRID).parts)
    assert {p.compartment for p in bus.last_packets} == {"apical", "soma"} and len({p.soma for p in bus.last_packets}) >= 5
    bus.set_gate("vector", 0.4)
    bus.receive(SignalPacket(RepresentationKind.SPECTRAL, np.ones(8) * 0.5, source="external", compartment="proximal"))
    bus.set_gate("spectral", 0.3)
    obs = torch.rand(155)
    assert float((sub.forward_step(obs) - legacy.forward_step(obs)).abs().max()) > 1e-4
    learner = _run(_sub("neurowave_packets", params={"neurowave_packets": {"learn_gates": True}}), 25, reward=1.0)
    gates = learner.mechanisms["neurowave_packets"].gates
    assert any(v != 0.0 for v in gates.values()) and all(abs(v) <= 0.5 for v in gates.values())


def test_vsa_flyhash_and_fingerprints():
    rng = np.random.default_rng(0)
    a, b, c = (rng.choice([-1.0, 1.0], size=512) for _ in range(3))
    assert abs(vsa_similarity(vsa_bind(a, b), a)) < 0.15 and vsa_similarity(vsa_bind(vsa_bind(a, b), b), a) == 1.0
    assert vsa_similarity(vsa_bundle([a, b, c]), a) > 0.3
    proj = (rng.random((64, 32)) < 0.2).astype(np.float32)
    x = rng.random(32)
    near, far = x + rng.normal(0, 0.01, 32), rng.random(32)
    code = flyhash_code(x, proj, 8)
    assert code.sum() == 8 and (code * flyhash_code(near, proj, 8)).sum() > (code * flyhash_code(far, proj, 8)).sum()
    f = compute_fingerprints(np.array([0, 1, 2, 1]), np.array([1, 0, 0, 2]), np.array([0.5, -0.2, 0.3, 0.1]), 3)
    assert f.shape == (3, 11) and f[0, 4] == 0.5 and f[1, 4] == 1.0                              # reciprocity of the 0<->1 pair
    sub = _run(_sub("structural_encoder", "connectivity_attention", "flyhash_projection", "multi_soma"), 5)
    assert sub.mechanisms["structural_encoder"].fingerprints().shape == (768, 16)
    assert float(sub.mechanisms["connectivity_attention"].attention.abs().max()) > 0
    assert sub.mechanisms["flyhash_projection"].telemetry()["active_bits"] == 8


# ------------------------------------------------------------------------ plasticity
def test_reward_gated_plasticity_needs_reward_and_keeps_signs():
    quiet = _run(_sub("reward_gated_plasticity"), 15, reward=0.0)
    assert float(quiet.syn.w_delta.abs().sum()) == 0.0 and float(quiet.syn.eligibility.abs().sum()) > 0.0
    learner = _run(_sub("reward_gated_plasticity"), 15, reward=1.0)
    w, anchor = learner.effective_weights(), learner.anchor_weights()
    assert float(learner.syn.w_delta.abs().sum()) > 0.0
    assert torch.all(torch.sign(w) * torch.sign(anchor) >= 0) and float(w.abs().max()) <= max(learner.w_max, float(anchor.abs().max())) + 1e-6
    sparse = _run(_sub("reward_gated_plasticity", params={"reward_gated_plasticity": {"mode": "sonapse"}}), 15, reward=1.0)
    fan_in = torch.zeros(768).scatter_add_(0, sparse.edge_index()[1], sparse.syn.w_delta.abs())
    assert float(sparse.syn.w_delta.abs().sum()) > 0.0 and float(fan_in.max()) <= 2.0 + 1e-5       # existing fan-in bound


def test_stdp_traces_meta_plasticity_and_sleep():
    sub = _run(_sub("membrane_spiking", "pair_stdp", "meta_plasticity", "reward_gated_plasticity"), 30, reward=1.0, scale=1.0)
    syn = sub.syn
    assert float(syn.pre_trace.max()) <= 1.5 and float(syn.post_trace.max()) <= 1.5 and float(syn.pre_trace.min()) >= 0.0
    assert sub.mechanisms["pair_stdp"].updates == 30
    assert 0.1 <= float(syn.lr_scale.min()) and float(syn.lr_scale.max()) <= 4.0 and float(syn.lr_scale.std()) > 0.0
    for rule in ("causal", "neuroweave"):
        s = _run(_sub("membrane_spiking", "pair_stdp", params={"pair_stdp": {"rule": rule}}), 10, scale=1.0)
        assert torch.isfinite(s.syn.w_delta).all()
    sleeper = _run(_sub("reward_gated_plasticity", "sleep_consolidation"), 20, reward=1.0)
    before = float(sleeper.syn.w_delta.abs().sum())
    report = sleeper.mechanisms["sleep_consolidation"].run_sleep()
    assert report["replayed"] >= 1 and report["downscaled"] > 0.0 and float(sleeper.syn.consolidation.max()) > 0.0
    only_down = _run(_sub("reward_gated_plasticity", "sleep_consolidation", params={"sleep_consolidation": {"replay": False}}), 20, reward=1.0)
    start = float(only_down.syn.w_delta.abs().sum())
    assert only_down.mechanisms["sleep_consolidation"].run_sleep()["replayed"] == 0.0
    assert before > 0 and float(only_down.syn.w_delta.abs().sum()) != start


# -------------------------------------------------------------------- neuromodulation
def test_neuromodulators_respond_stay_bounded_and_can_be_clamped():
    sub = _run(_sub("neuromodulatory_field", "affective_control", "multi_soma"), 40, reward=1.0)
    m, a = sub.signals.modulators, sub.signals.affect
    assert m.dopamine > 0.0 and a.valence > 0.0 and a.motivation > 0.5
    field = sub.mechanisms["neuromodulatory_field"]
    assert set(field.receptors) == set(RECEPTORS) and float(field.receptor_matrix().std()) > 0.0      # heterogeneous receptors
    assert len(field.regional_levels()) == 3 and float(sub.cell.soma_gating.std()) > 0.0              # local, unequal gating
    punished = _run(_sub("neuromodulatory_field", "affective_control"), 40, reward=-1.0)
    assert punished.signals.affect.stress > 0.2 and punished.signals.affect.valence < 0.0 and punished.signals.modulators.dopamine < 0.0
    clamped = _run(_sub("neuromodulatory_field", params={"neuromodulatory_field": {"clamp_values": {"dopamine": 0.8}}}), 5)
    assert clamped.signals.modulators.dopamine == pytest.approx(0.8)


def test_affect_variables_are_operational_controls():
    """Each affect variable changes a concrete control output, and each coupling can be ablated."""
    def control(**clamp):
        sub = _sub("affective_control", "membrane_spiking", params={"affective_control": {"clamp_values": clamp}})
        sub.forward_step(torch.rand(155))
        return sub

    assert control(curiosity=0.9).signals.exploration > control(curiosity=0.2).signals.exploration == 0.0
    assert control(motivation=0.95).signals.persistence > control(motivation=0.5).signals.persistence == 0.0
    assert control(valence=1.0).signals.plasticity_gain > control(valence=-1.0).signals.plasticity_gain
    assert float(control(stress=1.0).cell.threshold.mean()) < float(control(stress=0.0).cell.threshold.mean())
    calm, aroused = control(arousal=0.15), control(arousal=1.0)
    assert float((calm.r - aroused.r).abs().max()) > 0.0                                             # arousal changes response dynamics
    ablated = _sub("affective_control", params={"affective_control": {"clamp_values": {"curiosity": 0.9}, "couple_exploration": False}})
    ablated.forward_step(torch.rand(155))
    assert ablated.signals.exploration == 0.0


def test_transmitter_hypotheses_never_relabel_signs():
    sub = _run(_sub("transmitter_inference", "axonal_branches"), 3)
    mech = sub.mechanisms["transmitter_inference"]
    assert torch.allclose(mech.probabilities.sum(dim=1), torch.ones(768), atol=1e-5) and mech.probabilities.shape == (768, 6)
    assert mech.label(0) in TRANSMITTER_CLASSES and 0.94 <= float(mech.gain.min()) and float(mech.gain.max()) <= 1.06
    assert torch.equal(torch.sign(sub.effective_weights()), torch.sign(sub.anchor_weights()))


# ------------------------------------------------------------------------- prediction
def test_world_model_learns_and_counterfactuals_do_not_touch_the_model():
    sub = _sub("predictive_world_model", "uncertainty_gating")
    wm = sub.mechanisms["predictive_world_model"]
    for t in range(150):
        obs = torch.zeros(155)
        obs[:8] = torch.as_tensor([np.sin(0.2 * t + i) for i in range(8)], dtype=torch.float32) * 0.5
        sub.deliver_feedback(reward=0.2, issued_action=[0.3, 0.0, 0, 0, 0])
        sub.forward_step(obs)
        if t == 20:
            early = float(np.mean(wm.error))
    assert float(np.mean(wm.error)) < early and wm.updates > 100 and 0.0 <= sub.signals.uncertainty <= 1.0
    rates, members = sub.r.clone(), [w.copy() for w in wm.members]
    alternatives = wm.counterfactual([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [-1, 0, 0, 0, 0]])
    rollout = wm.rollout([[0.5, 0, 0, 0, 0]] * 8)
    assert len(alternatives) == 3 and len(rollout["predicted_rewards"]) == 8
    assert all(a["provenance"] == "internal_prediction" for a in alternatives)
    assert torch.equal(sub.r, rates) and all(np.array_equal(a, b) for a, b in zip(wm.members, members))
    assert wm.predict_action(wm.features, wm.features).shape == (5,)


def test_prediction_horizons_adapt_to_uncertainty():
    sub = _sub("hierarchical_prediction")
    hp = sub.mechanisms["hierarchical_prediction"]
    for t in range(260):
        obs = torch.zeros(155)
        obs[:8] = torch.as_tensor([np.sin(0.05 * t + i) for i in range(8)], dtype=torch.float32) * 0.5
        sub.forward_step(obs)
    assert hp.horizons == (1, 4, 16, 64) and hp.skill(1) > 0.0 and hp.skill(4) > 0.0     # beats the persistence baseline
    stable = hp.active_horizon
    sub.signals.uncertainty = 1.0
    sub.forward_step(obs)
    assert stable == 64 and hp.active_horizon < stable                     # high uncertainty shortens the horizon (64 -> 4)
    assert hp.params["feedback_gain"] == 0.0                               # predictive feedback starts disabled
    legacy, fb = _legacy(), _sub("hierarchical_prediction")
    for _ in range(4):
        o = torch.rand(155)
        assert torch.equal(fb.forward_step(o), legacy.forward_step(o))


def test_self_model_separates_self_caused_from_disturbed_outcomes():
    def run(observed):
        sub = _sub("efference_self_model")
        for _ in range(6):
            sub.deliver_feedback(reward=0.1, intended_action=[0.5, 0, 0, 0, 0], issued_action=[0.5, 0, 0, 0, 0],
                                 predicted_outcome=[0.5, 0.0], observed_outcome=observed)
            sub.forward_step(torch.rand(155))
        return sub

    clean, disturbed = run([0.5, 0.0]), run([-0.4, 0.6])
    r1, r2 = clean.mechanisms["efference_self_model"].report, disturbed.mechanisms["efference_self_model"].report
    assert r1["authorship"] == 1.0 and r1["disturbance"] == 0.0 and r2["authorship"] < 0.2 and r2["disturbance"] > 0.5
    assert disturbed.signals.control_reliability < clean.signals.control_reliability
    assert disturbed.signals.plasticity_gain < clean.signals.plasticity_gain              # discounted credit feeds learning
    slip = _sub("efference_self_model")
    slip.forward_step(torch.rand(155))
    slip.deliver_feedback(intended_action=[0.5, 0, 0, 0, 0], issued_action=[0.0, 0.5, 0, 0, 0], predicted_outcome=[0.5, 0], observed_outcome=[0.5, 0])
    slip.forward_step(torch.rand(155))
    assert slip.mechanisms["efference_self_model"].report["actuator_slip"] > 0.5


# -------------------------------------------------------------------------- structure
def test_structural_turnover_is_bounded_audited_and_reversible():
    sub = _run(_sub("reward_gated_plasticity", "structural_plasticity", params={"structural_plasticity": {"period": 4}}), 24, reward=1.0)
    sp = sub.mechanisms["structural_plasticity"]
    assert sp.pruned > 0 and sp.grown > 0 and sp.grown <= sp.pruned
    assert int(sub.syn.alive.sum()) <= int(sp.initial_edges * sp.params["max_edge_factor"])
    assert bool(sub.syn.alive[:3].all())                                                   # canonical pathway protected
    entries = sp.ledger.entries
    assert {e["action"] for e in entries} == {"prune_synapse", "grow_synapse"} and all("reason" in e and "step" in e for e in entries)
    pre, post = sub.edge_index()
    fixed = sub.syn.n_base + sub.syn.n_plastic
    grown_keys, base_keys = (pre * 768 + post)[fixed:], (pre * 768 + post)[:fixed]
    assert len(torch.unique(grown_keys)) == len(grown_keys) and not bool(torch.isin(grown_keys, base_keys).any())   # no duplicates
    assert bool((pre[fixed:] != post[fixed:]).all())                                        # no self-loops
    from sonapse_evo.substrate.graph_topology import GraphTopology

    synced = sp.sync_to_graph(GraphTopology(total_neurons=768, num_regions=3, seed=42))    # bridge to Mode B operators
    assert synced["added"] == sub.syn.n_grown and synced["removed"] > 0
    sp.revert()
    assert sub.syn.n_grown == 0 and bool(sub.syn.alive.all())


def test_neurogenesis_has_budgets_and_growth_leaves_the_source_untouched():
    params = {"neurogenesis": {"period": 8, "cooldown": 8, "demand_threshold": 0.0, "max_recruits": 8, "batch": 4}}
    sub = _run(_sub("neurogenesis", "local_predictive_coding", params=params), 64)
    ng = sub.mechanisms["neurogenesis"]
    assert 0 < ng.recruited <= 8 and ng.run_recruitment(force=True)["blocked"] == "lifetime_budget"
    actions = ng.ledger.summary()["counts"]
    assert actions["repurpose_neuron"] == ng.recruited and actions["form_subnetwork"] >= 1
    assert all(e["reason"] for e in ng.ledger.entries) and int(ng.is_recruit.sum()) == ng.recruited
    idle = _sub("neurogenesis", params={"neurogenesis": {"demand_threshold": 5.0}})
    assert idle.mechanisms["neurogenesis"].run_recruitment(step=10 ** 6)["blocked"] == "no_demand"   # no demand, no growth
    before = sub.r.clone()
    grown, receipt = grow_substrate(sub, 24, seed=3)
    assert grown.total_neurons == 792 and sub.total_neurons == 768 and receipt["source_untouched"] and torch.equal(sub.r, before)
    assert torch.equal(grown.r[:768], sub.r) and torch.isfinite(grown.forward_step(torch.rand(155))).all()
    with pytest.raises(ValueError):
        grow_substrate(sub, 10, max_neurons=770)


# -------------------------------------------------------------------------- execution
def test_quantisation_and_tiled_execution():
    w = torch.tensor([0.5, -0.25, 0.013, 1.5, -1.5, 0.0])
    wq, codes = quantise_int8(w, 1.5)
    assert codes.dtype == torch.int8 and int(codes.abs().max()) <= 127 and float((wq - w).abs().max()) <= 1.5 / 127 / 2 + 1e-6
    assert torch.equal(torch.sign(wq)[w.abs() > 0.01], torch.sign(w)[w.abs() > 0.01])
    assert torch.allclose(from_q8_8(to_q8_8(torch.tensor([0.5, 0.123, 1.0]))), torch.tensor([0.5, 0.123, 1.0]), atol=1 / 512)
    legacy, quant, tiled = _legacy(), _sub("quantised_synapses"), _sub("tiled_execution")
    gen = torch.Generator().manual_seed(4)
    for _ in range(6):
        obs = torch.rand(155, generator=gen)
        ref = legacy.forward_step(obs)
        assert float((quant.forward_step(obs) - ref).abs().max()) < 0.02            # small, bounded quantisation effect
        assert float((tiled.forward_step(obs) - ref).abs().max()) < 1e-5            # same result up to summation order
    assert quant.mechanisms["quantised_synapses"].telemetry()["mean_abs_quantisation_error"] > 0.0
    quant.set_mechanism("quantised_synapses", False)
    assert "int8" not in quant.weight_transforms                                    # full precision restored
    schedule = tiled.mechanisms["tiled_execution"].schedule()
    assert sum(schedule["edges_per_tile"]) == tiled.syn.n_edges and schedule["tiles"] == 12
    state = _sub("quantised_synapses", params={"quantised_synapses": {"quantise_state": True}})
    state.forward_step(torch.rand(155))
    assert torch.equal(state.r, from_q8_8(to_q8_8(state.r)))


def test_fpga_manifest_profiler_and_backends():
    sub = _run(_sub("reward_gated_plasticity"), 5, reward=1.0)
    manifest = build_fpga_manifest(sub)
    assert validate_fpga_manifest(manifest) == [] and manifest["validation"]["endpoints_in_range"] and manifest["validation"]["sign_preserved"]
    assert len(manifest["tileSchedule"]["stages"]) == 4 and "no RTL generated" in manifest["nonClaims"]
    assert manifest["neuronStateHints"]["format"].startswith("Q8.8") and max(abs(c) for c in manifest["synapseTable"]["weight_int8"]) <= 127
    tampered = json.loads(json.dumps(manifest))
    tampered["synapseTable"]["post"][0] = 10 ** 6
    assert any("endpoint" in i for i in validate_fpga_manifest(tampered)) and any("digest" in i for i in validate_fpga_manifest(tampered))
    assert validate_fpga_manifest(build_fpga_manifest(_legacy())) == []              # works on the original substrate too
    before = sub.r.clone()
    report = profile_substrate(sub, steps=6)
    assert report["measured"]["ms_per_step"] > 0 and report["measured"]["total_bytes"] > 0 and "estimated" in report
    assert torch.equal(sub.r, before)                                                # profiling restores the model state
    rows = scaling_profile(sizes=(768, 1536), steps=4)
    assert rows[1]["config"]["edges"] > rows[0]["config"]["edges"] and rows[1]["config"]["neurons"] == 1536
    compact = compact_state(sub.weave_state_dict())["measured"]
    assert compact["float_bytes_after"] * 2 == compact["float_bytes_before"] and compact["max_abs_error"] < 0.01
    backends = available_backends()
    assert backends["cpu"]["weave_path"] and not backends["fpga"]["available"] and not backends["cuda"]["weave_path"]
