"""Subsystem tests: sensory system, Micro-Collective and specialist, memory ecology, connectome, environments."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sonapse_evo.evaluation.benchmark_runner import run_benchmark_episode
from sonapse_evo.neuroweave import Provenance, build_weave_substrate, canonical_motor_head, run_weave_episode
from sonapse_evo.neuroweave.collective import DenseMLPPredictor, MicroCollective, PersistencePredictor
from sonapse_evo.neuroweave.connectome import (LESION_KINDS, TOPOLOGY_KINDS, build_lesion_mask, build_topology, compute_graph_metrics,
                                               custom_layout, degree_preserving_shuffle, edge_class_counts, legacy_layout,
                                               matched_random_mask, progressive_schedule, ten_region_layout)
from sonapse_evo.neuroweave.ecology import ECOLOGY_PRESETS, EcologyArena, MultiAgentEcology, list_environments, make_environment
from sonapse_evo.neuroweave.memory import MemoryRecord, MemoryStore, ProvenanceError, SemanticStore, SparseIndex, verify_against
from sonapse_evo.neuroweave.sensory import (SENSORY_CHANNELS, AudioFeatureExtractor, ChannelMode, SensoryManifest, SimulatedSensors,
                                            VisualFeatureExtractor)
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.topology_state import TopologyState


def _legacy():
    return RecurrentSubstrate(topology=TopologyState(768, 3, seed=42), seed=42)


def _sub(*ids, **kw):
    return build_weave_substrate(seed=42, enable=list(ids), strict=True, **kw)


def _stream(t):
    return np.array([0.6 * np.sin(0.2 * t + 0.4 * i) for i in range(16)])


# ---------------------------------------------------------------------------- sensory
def test_visual_and_audio_feature_extraction_is_local_and_keeps_no_media():
    cam = VisualFeatureExtractor()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[40:80, 30:70] = (255, 30, 30)
    still = cam.extract(frame)
    moved = cam.extract(np.roll(frame, 20, axis=1))
    assert moved["visual_motion"] > still["visual_motion"] == 0.0 and still["visual_red_green"] > 0.0 > still["visual_blue_yellow"]
    assert 0.0 < still["visual_edge_density"] < 0.5 and still["visual_centroid_x"] < 0.5 and still["visual_object_count"] > 0.0
    blue = VisualFeatureExtractor().extract(np.tile(np.array([0, 0, 255], dtype=np.uint8), (60, 80, 1)))
    assert blue["visual_blue_yellow"] > 0.5 and blue["visual_edge_density"] == 0.0
    assert max(v.size for v in vars(cam).values() if isinstance(v, np.ndarray)) <= 24 * 32 < frame.size   # only a tiny luminance grid
    t = np.arange(2048) / 16000.0
    mic = AudioFeatureExtractor()
    low, high = mic.extract(0.5 * np.sin(2 * np.pi * 150 * t)), mic.extract(0.5 * np.sin(2 * np.pi * 4000 * t))
    assert low["audio_low"] > 0.9 and low["audio_high"] < 0.05 and high["audio_high"] > 0.9 and high["audio_low"] < 0.05
    assert low["audio_periodicity"] > 0.8 and high["audio_onset"] == 1.0 and 0.3 < low["audio_amplitude"] < 0.4
    assert all(not isinstance(v, np.ndarray) or v.size <= 3 for v in vars(mic).values())               # no samples retained
    with pytest.raises(ValueError):
        mic.extract(np.array([np.nan] * 16))
    sim = SimulatedSensors(seed=1)
    assert sim.read({"food": 0.8, "energy": 0.3})["gustatory"] == pytest.approx(0.8) and sim.read({"energy": 0.3})["interoception"] == pytest.approx(0.7)
    assert set(SimulatedSensors(seed=1).read()) == set(sim.channels)


def test_sensory_hub_projects_channels_into_the_connectome():
    legacy, sub = _legacy(), _sub("multimodal_sensory")
    hub = sub.mechanisms["multimodal_sensory"]
    obs = torch.rand(155)
    assert torch.equal(sub.forward_step(obs), legacy.forward_step(obs))                     # all channels zero: exactly neutral
    assert len(SENSORY_CHANNELS) == 16 and hub.ingest({"olfactory": 0.9, "thermosensation": 5.0}) == 2
    assert hub.current_vector()[SENSORY_CHANNELS.index("thermosensation")] == 1.0           # clipped to the channel range
    diff = (sub.forward_step(obs) - legacy.forward_step(obs)).abs()
    assert float(diff.max()) > 1e-4 and float(diff[768 // 4 + 60:].max()) < float(diff[:768 // 4].max())   # enters via sensory neurons
    with pytest.raises(KeyError):
        hub.ingest({"telepathy": 1.0})
    with pytest.raises(ValueError):
        hub.ingest({"olfactory": float("nan")})
    hub.set_mode("olfactory", ChannelMode.DISABLED)
    assert hub.current_vector()[SENSORY_CHANNELS.index("olfactory")] == 0.0 and "olfactory" in hub.telemetry()["missing"]
    hub.set_mode("gustatory", "simulated")
    sub.forward_step(obs)
    assert hub.manifest.provenance["gustatory"] == Provenance.SYNTHETIC_SIMULATION.value
    hub.ingest_external({"task_context": 1.0})
    assert hub.manifest.provenance["task_context"] == Provenance.EXTERNAL_SUPPLIED.value
    frame = np.random.default_rng(0).integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    hub.ingest_frame(frame)
    hub.ingest_audio(np.sin(np.arange(1024) / 5.0) * 0.3)
    state = hub.state_dict()
    assert hub.manifest.camera_enabled and hub.manifest.microphone_enabled and state["manifest"]["raw_media_stored"] is False
    assert max(v.numel() for v in state.values() if isinstance(v, torch.Tensor)) == 16      # features only, never media
    assert SensoryManifest.from_dict(state["manifest"]).to_dict()["modes"]["olfactory"] == "disabled"
    ten = _sub("multimodal_sensory", topology_kind="multi_region")
    proj = ten.mechanisms["multimodal_sensory"].projection.to_dense()
    visual = torch.nonzero(proj[:, 0]).squeeze(-1)
    assert set(ten.region_ids[visual].tolist()) == {0}                                      # modality-specific target region


def test_cross_modal_fusion_weights_reliability_and_imputes_missing_channels():
    sub = _sub("multimodal_sensory", "cross_modal_fusion")
    hub, fusion = sub.mechanisms["multimodal_sensory"], sub.mechanisms["cross_modal_fusion"]
    rng, obs = np.random.default_rng(0), torch.rand(155)
    for i in range(70):
        values = {"olfactory": 0.5 + 0.4 * np.sin(i / 5), "gustatory": 0.5 + 0.4 * np.sin(i / 5), "thermosensation": float(rng.random())}
        if i >= 50:
            values.pop("gustatory")                                                          # this sensor drops out
        hub.ingest(values)
        sub.forward_step(obs)
    weights = fusion.telemetry()["weights"]
    assert weights["olfactory"] > weights["thermosensation"]                                 # the noisy channel is down-weighted
    idx = SENSORY_CHANNELS.index("gustatory")
    assert fusion.fused_provenance[idx] == Provenance.INTERNAL_PREDICTION.value              # imputed, never reported as observed
    assert fusion.fused_provenance[SENSORY_CHANNELS.index("olfactory")] != Provenance.INTERNAL_PREDICTION.value
    assert fusion.fused[idx] > 0.05 and fusion.imputed_count >= 1
    assert sub.signals.extra["sensory_uncertainty"] > 0.0 and "sensory_prediction_error" in sub.signals.extra
    assert float(np.abs(fusion.association[idx, SENSORY_CHANNELS.index("olfactory")])) > 0.0  # learned cross-modal association
    alone = _sub("cross_modal_fusion")
    assert torch.isfinite(alone.forward_step(obs)).all()                                     # degrades gracefully without the hub


# ------------------------------------------------------------------------- collective
def test_micro_collective_learns_validates_rolls_back_and_expands():
    mc = MicroCollective(seed=1, lr=0.05)
    mlp, persistence = DenseMLPPredictor(parameter_budget=mc.parameter_count, seed=1, lr=0.05), PersistencePredictor()
    for t in range(1200):
        for model in (mc, mlp, persistence):
            model.observe(_stream(t))
    receipt = mc.receipt()
    assert receipt["validation_loss"] < receipt["persistence_baseline_loss"] and receipt["skill_vs_persistence"] > 0.5
    assert receipt["trained"] and receipt["source"].startswith("freshly initialised") and len(mc.replay) <= 384 and len(mc.validation) <= 96
    assert abs(mlp.parameter_count - mc.parameter_count) < 40 and np.isfinite(mlp.evaluate(mc.validation))   # matched-budget baseline
    assert persistence.evaluate(mc.validation) == pytest.approx(receipt["persistence_baseline_loss"])
    x = _stream(3)
    routed = mc.route(x, track_load=False)
    assert len(routed) == 2 and sum(w for _, w in routed) == pytest.approx(1.0) and 0.0 <= mc.entropy <= 1.0
    before = mc.predict(x)
    mc.add_expert(), mc.add_expert()
    assert mc.n_experts == 6 and np.allclose(before, mc.predict(x))                          # expansion leaves predictions unchanged
    clone = MicroCollective(seed=99)
    clone.load_snapshot(mc.snapshot())
    assert np.array_equal(clone.predict(x), mc.predict(x))
    with pytest.raises(ValueError):
        clone.load_snapshot({"format": "other"})
    single = MicroCollective(seed=1, expert_routing=False)
    assert len(single.route(x, track_load=False)) == 1
    twin = MicroCollective(seed=1, lr=0.05)
    for t in range(300):
        twin.observe(_stream(t))
    again = MicroCollective(seed=1, lr=0.05)
    for t in range(300):
        again.observe(_stream(t))
    assert np.array_equal(twin.predict(x), again.predict(x))                                 # deterministic for a seed
    noisy = MicroCollective(seed=2, lr=0.3)
    rng = np.random.default_rng(0)
    for t in range(400):
        noisy.observe(_stream(t))
    for t in range(400):
        noisy.observe(rng.uniform(-1, 1, 16))                                                # the stream is corrupted
    assert noisy.rollbacks >= 1 and noisy.lr < 0.3


def test_collective_and_specialist_are_neutral_until_validated_then_contribute():
    legacy = _legacy()
    sub = _sub("micro_collective", "expert_routing", "collective_meta_learning", "specialist_graft")
    for _ in range(10):
        obs = torch.rand(155)
        assert torch.equal(sub.forward_step(obs), legacy.forward_step(obs))                  # untrained: exactly zero contribution
    for t in range(500):
        obs = torch.zeros(155)
        obs[:16] = torch.as_tensor(_stream(t), dtype=torch.float32)
        sub.forward_step(obs)
    mech, graft = sub.mechanisms["micro_collective"], sub.mechanisms["specialist_graft"]
    assert mech.skill > 0.5 and 0.0 < mech.context_gain <= 0.25 and mech.receipt()["trained"]
    assert mech.collective.expert_routing and mech.collective.meta_learning and not mech.collective.predictive_replay
    report = graft.contribution_report()
    assert report["removable"] and report["optimiser_steps"] > 0 and 0.0 <= report["gain"] <= 0.2
    knowledge = graft.export_knowledge()
    other = _sub("specialist_graft").mechanisms["specialist_graft"]
    other.import_knowledge(knowledge)
    features = np.zeros(26, dtype=np.float32)
    assert np.allclose(other.model.predict(features), graft.model.predict(features))         # explicit knowledge transfer
    with pytest.raises(ValueError):
        other.import_knowledge({"format": "nope"})
    before = sub.effective_weights().clone()
    sub.set_mechanism("specialist_graft", False)
    assert "specialist_graft" not in sub.mechanisms and torch.equal(sub.effective_weights(), before)   # removable without damage


# ----------------------------------------------------------------------------- memory
def test_provenance_rules_and_sparse_index():
    key, value = np.arange(8, dtype=np.float32), np.ones(4, dtype=np.float32)
    with pytest.raises(ProvenanceError):
        MemoryRecord(1, key, value, Provenance.INTERNAL_PREDICTION, verified=True)
    prediction = MemoryRecord(1, key, value, Provenance.INTERNAL_PREDICTION)
    with pytest.raises(ProvenanceError):
        prediction.mark_verified()
    with pytest.raises(ProvenanceError):
        MemoryRecord(2, key, value, Provenance.SYNTHETIC_SIMULATION).mark_verified()
    confirmed = verify_against(prediction, value + 0.01, tolerance=0.1, new_id=3)
    assert confirmed.verified and confirmed.provenance is Provenance.DIRECT_SENSORY and confirmed.meta["confirmed_prediction"] == 1
    assert not prediction.verified and verify_against(prediction, value + 5.0, 0.1, 4) is None    # the prediction is never relabelled
    assert MemoryRecord.from_dict(confirmed.to_dict()).verified
    rng = np.random.default_rng(0)
    store = MemoryStore(capacity=40, key_dim=16, seed=1)
    keys = rng.standard_normal((40, 16)).astype(np.float32)
    for i, k in enumerate(keys):
        store.add(MemoryRecord(i, k, rng.standard_normal(4), Provenance.DIRECT_SENSORY, verified=True, reward=float(i % 3 - 1), step=i))
    hits = sum(store.retrieve(k + 0.01, k=1)[0][0].record_id == store.retrieve(k + 0.01, k=1, brute_force=True)[0][0].record_id for k in keys)
    assert hits == 40                                                                        # sparse index agrees with brute force
    index = SparseIndex(16, seed=1)
    assert index.overlap(keys[0], keys[0] + 0.001) > index.overlap(keys[0], keys[1]) and len(index.code(keys[0])) == 12
    store.add(MemoryRecord(99, keys[0], np.zeros(4), Provenance.DIRECT_SENSORY, step=100))
    assert len(store) == 40 and store.evicted == 1                                           # capacity is never exceeded
    assert all(r.reward > 0 for r in store.by_reward(1)) and all(r.reward < 0 for r in store.by_reward(-1))
    store.decay(0.5)
    assert store.forget(0.3) > 0
    semantic = SemanticStore(capacity=4, radius=0.9)
    assert semantic.absorb(MemoryRecord(1, keys[0], np.ones(4), Provenance.DIRECT_SENSORY)) and len(semantic) == 1
    assert semantic.absorb(MemoryRecord(2, keys[0] * 1.01, np.zeros(4), Provenance.DIRECT_SENSORY)) and len(semantic) == 1
    assert semantic.counts[0] == 2 and semantic.retrieve(keys[0])[1] > 0.99


def test_memory_mechanisms_store_recall_and_keep_predictions_separate():
    legacy = _legacy()
    sub = _sub("associative_trace_memory", "fast_weights", "memory_ecology", "predictive_replay", "predictive_world_model")
    obs = torch.rand(155)
    assert torch.equal(sub.forward_step(obs), legacy.forward_step(obs))                      # empty stores: exactly neutral
    cue = torch.rand(155) * 0.8
    for t in range(60):
        sub.deliver_feedback(reward=1.0, issued_action=[0.3, 0, 0, 0, 0])
        sub.forward_step(cue if t % 2 == 0 else torch.rand(155) * 0.8)
    eco, assoc, fast, replay = (sub.mechanisms[m] for m in ("memory_ecology", "associative_trace_memory", "fast_weights", "predictive_replay"))
    assert len(assoc.episodes) >= 1 and assoc.hits > 0 and len(assoc.episodes) <= 32
    assert float(np.abs(fast.matrix).max()) > 0.0 and float(np.abs(fast.matrix).max()) <= 1.0 and fast.updates > 0
    tel = eco.telemetry()
    assert tel["episodic"] > 0 and tel["working"] <= 8 and tel["predictive"] > 0 and tel["hits"] > 0
    assert tel["provenance"]["internal_prediction"] == tel["predictive"] and tel["provenance"]["direct_sensory"] > 0
    assert tel["confirmed_predictions"] + tel["refuted_predictions"] > 0                     # predictions are checked against reality
    observed = eco.retrieve(cue.numpy(), k=5, observed_only=True)
    assert observed and all(r.verified and r.provenance.is_observation for r, _ in observed)
    assert all(not r.verified for r in eco.predictive.records.values())                      # a prediction is never an observation
    assert any(r.provenance is Provenance.INTERNAL_PREDICTION for r, _ in eco.retrieve(cue.numpy(), k=20, stores=("predictive",)))
    with pytest.raises(ValueError):
        eco.episodic.add(MemoryRecord(10 ** 6, np.ones(32), np.ones(4), Provenance.DIRECT_SENSORY)) or eco.predictive.add(
            MemoryRecord(10 ** 6 + 1, np.ones(32), np.ones(4), Provenance.DIRECT_SENSORY))
    a = eco.write_observation(np.ones(32), np.ones(768), step=1)
    b = eco.write_observation(np.ones(32), -np.ones(768), step=2)                            # same key, conflicting content
    c = eco.write_observation(-np.ones(32), np.ones(768), step=3)
    assert b.meta["interference"] > c.meta["interference"] and eco.interference > 0.0
    assert eco.write_external(np.ones(32), np.ones(768)).provenance is Provenance.EXTERNAL_SUPPLIED
    assert eco.write_simulated(np.ones(32), np.ones(768)).verified is False
    episodic_before = sum(r.confidence for r in eco.episodic.records.values())
    assert eco.consolidate() > 0 and len(eco.semantic) > 0 and sum(r.confidence for r in eco.episodic.records.values()) < episodic_before
    assert len(eco.next_replay_batch(3)) == 3 and len(eco.replay_states(2)) == 2
    eco.downscale(0.5)
    assert eco.forget(rate=1.0) > 0                                                          # forgetting experiment hook
    assert len(replay.items) <= 128 and replay.run_replay(3) == 3 and len(replay.queue) == 3
    sub.forward_step(obs)
    assert len(replay.queue) == 2 and replay.replayed >= 1 and sub.events.recent("memory.replay")


# -------------------------------------------------------------------------- connectome
@pytest.mark.parametrize("kind", TOPOLOGY_KINDS)
def test_every_topology_family_builds_a_valid_runnable_connectome(kind):
    topo = build_topology(kind, 768, seed=7)
    n = topo.total_neurons
    assert topo.base_pre.dtype == np.int64 and topo.base_w.dtype == np.float32 and len(topo.base_pre) == len(topo.base_w) > 1000
    assert topo.base_pre.min() >= 0 and topo.base_post.max() < n and np.all(np.isfinite(topo.base_w))
    m = topo.layout.motor_offset
    assert list(topo.base_pre[:3]) == [0, 1, 2] and list(topo.base_post[:3]) == [m, m + 1, m + 2] and np.allclose(topo.base_w[:3], 0.8)
    if kind != "legacy":                                           # the original generator samples with replacement
        assert len(np.unique(topo.base_pre * n + topo.base_post)) == len(topo.base_pre) and np.all(topo.base_pre != topo.base_post)
        signs = topo.layout.signs()
        assert np.all(np.sign(topo.base_w[3:]) == signs[topo.base_pre[3:]])                 # sign belongs to the source neuron
    again = build_topology(kind, 768, seed=7)
    assert np.array_equal(again.base_pre, topo.base_pre) and np.array_equal(again.base_w, topo.base_w)   # deterministic
    ret = run_benchmark_episode(make_environment("forage", seed=1), RecurrentSubstrate(topology=topo, seed=7), canonical_motor_head(),
                                seed=3, motor_offset=m)
    assert np.isfinite(ret) and ret > 0.0


def test_layouts_metrics_shuffle_and_small_populations():
    ten, legacy = ten_region_layout(768), legacy_layout(768)
    assert ten.num_regions == 10 and sum(ten.sizes) == 768 and ten.names[-1] == "motor" and len(ten.sensory_region_ids) == 7
    assert legacy.sizes == (256, 256, 256) and legacy.motor_offset == 512 and legacy.names == ("flywire", "manc", "optic-lobe")
    custom = custom_layout([{"name": "in", "role": "sensory", "fraction": 2, "inhibitory_fraction": 0.5},
                            {"name": "hub", "role": "association", "fraction": 1, "inhibitory_fraction": 0.0},
                            {"name": "out", "role": "motor", "fraction": 1}], 400)
    summary = custom.population_summary()
    assert summary[0]["neurons"] == 200 and summary[0]["inhibitory"] == 100 and summary[1]["inhibitory"] == 0
    assert type(custom).from_dict(custom.to_dict()).sizes == custom.sizes and custom.role_of("hub") == "association"
    assert torch.isfinite(RecurrentSubstrate(build_topology("hybrid", 400, layout=custom, seed=1), seed=1).forward_step(torch.rand(155))).all()
    source = build_topology("clustered_small_world", 768, seed=3)
    shuffled = degree_preserving_shuffle(source, seed=4)
    for arr in ("base_pre", "base_post"):
        assert np.array_equal(np.bincount(getattr(source, arr), minlength=768), np.bincount(getattr(shuffled, arr), minlength=768))
    assert int((source.base_post != shuffled.base_post).sum()) > 1000
    ring, rand = compute_graph_metrics(build_topology("ring_lattice", 768, seed=1)), compute_graph_metrics(build_topology("erdos_renyi", 768, seed=1))
    hub = compute_graph_metrics(build_topology("hub_skewed", 768, seed=1))
    capped = compute_graph_metrics(build_topology("hub_capped", 768, seed=1))
    regional = compute_graph_metrics(build_topology("multi_region", 768, seed=1))
    assert ring["clustering_coefficient"] > 5 * rand["clustering_coefficient"] and ring["characteristic_path_length"] > rand["characteristic_path_length"]
    assert hub["in_degree"]["cv"] > ring["in_degree"]["cv"] and hub["hub_concentration_top5pct"] > capped["hub_concentration_top5pct"]
    assert capped["in_degree"]["max"] <= 12 and regional["modularity"] > rand["modularity"] and len(regional["inter_region_density"]) == 10
    assert 0.0 < rand["global_efficiency"] <= 1.0 and rand["sources_with_mixed_sign"] <= 1 and rand["path_sources_used"] == 128     # only the canonical pathway source
    classes = edge_class_counts(build_topology("multi_region", 768, seed=1))
    assert classes["feedforward"] > classes["feedback"] > 0 and classes["local_recurrent"] > 0 and classes["long_range"] > 0
    small = build_topology("multi_region", 240, seed=1)
    assert small.total_neurons == 240 and torch.isfinite(RecurrentSubstrate(topology=small, seed=1).forward_step(torch.rand(155))).all()
    with pytest.raises(KeyError):
        build_topology("no_such_topology")
    with pytest.raises(ValueError):
        build_topology("legacy", 300)


def test_region_routing_and_lesion_library():
    sub = _sub("multi_region_routing", topology_kind="multi_region")
    legacy = RecurrentSubstrate(topology=sub.topology, seed=42)
    obs = torch.rand(155)
    assert torch.equal(sub.forward_step(obs), legacy.forward_step(obs))                      # unit gains: exactly neutral
    routing = sub.mechanisms["multi_region_routing"]
    assert routing.roles.count("sensory") == 7 and len(routing.pathway_pairs("sensory_to_association")) == 7
    routing.set_pathway("sensory_to_association", 1.4)
    routing.set_gain(0, 0, 9.0)
    assert float(routing.gain.max()) == 1.5 and routing.telemetry()["non_unit_gains"] == 8   # bounded
    assert float((sub.forward_step(obs) - legacy.forward_step(obs)).abs().max()) > 1e-5
    assert routing.telemetry()["inter_region_traffic"] > 0.0
    adaptive = _sub("multi_region_routing", topology_kind="multi_region", params={"multi_region_routing": {"adapt": True}})
    for _ in range(10):
        adaptive.deliver_feedback(reward=1.0)
        adaptive.forward_step(torch.rand(155))
    g = adaptive.mechanisms["multi_region_routing"].gain
    assert float(g.min()) >= 0.5 and float(g.max()) <= 1.5 and bool((g != 1.0).any())
    topo = build_topology("multi_region", 768, seed=3)
    for kind in LESION_KINDS:
        mask = build_lesion_mask(topo, kind, 0.3, seed=5)
        control = matched_random_mask(topo, int((mask == 0).sum()), seed=6)
        assert 0 < int((mask == 0).sum()) == int((control == 0).sum()) and mask[:3].all() and control[:3].all()
        assert np.array_equal(mask, build_lesion_mask(topo, kind, 0.3, seed=5))             # deterministic
    assert int((build_lesion_mask(topo, "sensory_pathway", 1.0, seed=1, protect_first=0)[:3] == 0).sum()) == 3
    assert build_lesion_mask(topo, "random_edges", 0.0).all() and list(progressive_schedule(4, 0.4)) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(KeyError):
        build_lesion_mask(topo, "no_such_lesion", 0.2)


# ------------------------------------------------------------------------ environments
@pytest.mark.parametrize("preset", sorted(ECOLOGY_PRESETS))
def test_every_ecology_preset_runs_with_bounded_observations(preset):
    env = make_environment(preset, seed=3)
    obs = env.reset(seed=3)
    assert obs.shape == (155,) and obs.dtype == np.float32
    for i in range(40):
        obs, reward, done, info = env.step([0.8, 0.3 * np.sin(i / 3), 0, 0, 0])
        assert np.all(np.isfinite(obs)) and float(np.abs(obs).max()) <= 1.0 + 1e-6 and np.isfinite(reward)
    channels = env.sensory_channels()
    assert list(channels) == list(SENSORY_CHANNELS) and all(-1.0 <= v <= 1.0 for v in channels.values())
    assert {"predicted_outcome", "observed_outcome", "energy", "collisions", "success"} <= set(info)
    assert env.render_state()["width"] == env.config.width and env.sensory_provenance is Provenance.SYNTHETIC_SIMULATION


def test_ecology_actions_have_real_consequences():
    a, b = make_environment("neuroweave_1200x660", seed=3), make_environment("neuroweave_1200x660", seed=3)
    assert a.config.width == 1200 and a.config.height == 660 and len(a.obstacles) == 10 and len(a.food) == 18 and len(a.thermal) == 3
    assert [a.step([0.8, 0.1])[1] for _ in range(40)] == [b.step([0.8, 0.1])[1] for _ in range(40)]     # deterministic
    idle, mover = make_environment("neuroweave_1200x660", seed=3), make_environment("neuroweave_1200x660", seed=3)
    for _ in range(20):
        idle.step([0.0, 0.0])
        mover.step([1.0, float(np.clip(2 * mover._observe()[1], -1, 1))])
    assert np.hypot(*(mover.goal - mover.pos)) < np.hypot(*(idle.goal - idle.pos)) and mover.energy < idle.energy
    env = make_environment("neuroweave_1200x660", seed=3)
    ob = env.obstacles[0]
    env.pos = np.array([ob[0] - ob[2] - 25.0, ob[1]])
    env.heading = 0.0
    _, reward, _, info = env.step([1.0, 0.0])
    assert reward == -1.0 and info["collisions"] == 1 and env.touch == 1.0                   # collision penalty and touch signal
    env.food[0][0], env.food[0][1] = float(env.pos[0]), float(env.pos[1])
    eaten = env.food_eaten
    env.step([0.0, 0.0])
    assert env.food_eaten == eaten + 1 and len(env.food) == 18                                # consumed and replenished
    env.pos = env.goal - np.array([10.0, 0.0])
    _, reward, _, info = env.step([0.0, 0.0])
    assert reward == 1.0 and info["success"] and env.score == 1
    calm, windy = make_environment("neuroweave_1200x660", seed=3), make_environment("disturbed", seed=3)
    i1, i2 = calm.step([1.0, 0.0])[3], windy.step([1.0, 0.0])[3]
    assert np.allclose(i1["predicted_outcome"], i1["observed_outcome"]) and not np.allclose(i2["predicted_outcome"], i2["observed_outcome"])
    memory = make_environment("memory_navigation", seed=3)
    flags = [int(memory.step([0.2, 0.0])[0][23]) for _ in range(10)]
    assert flags[:5] == [1] * 5 and flags[6:] == [0] * 4 and memory._observe()[1] == 0.0 and memory._observe()[2] == 0.0
    shift = make_environment("rule_shift", seed=3)
    rules = {shift.step([0.1, 0.0])[3]["rule"] for _ in range(70)}
    assert rules == {0, 1}
    snap = a.state_dict()
    restored = EcologyArena(seed=99)
    restored.load_state_dict(snap)
    assert a.step([0.5, 0.2])[1] == restored.step([0.5, 0.2])[1]


def test_multi_agent_ecology_and_environment_registry():
    arena = MultiAgentEcology(n_agents=3, seed=5)
    obs = arena.reset(5)
    assert len(obs) == 3 and all(o.shape == (155,) for o in obs) and float(obs[0][27]) > 0.0   # agents sense each other
    assert arena.agents[1].food is arena.agents[0].food                                         # one shared, contested world
    obs, rewards, done, info = arena.step([[0.5, 0.0]] * 3)
    assert len(rewards) == 3 and len(info["agents"]) == 3 and all(r > -1.0 for r in rewards)
    assert len(arena.render_state()["agents"]) == 3
    names = {e["name"]: e for e in list_environments()}
    assert names["forage"]["kind"] == "existing" and names["delayed_cue"]["kind"] == "existing" and len(names) == 15
    assert type(make_environment("forage")).__name__ == "ForageNavigationTask"                  # existing tasks are kept
    assert type(make_environment("delayed_cue")).__name__ == "DelayedCueMatchingTask"
    with pytest.raises(KeyError):
        make_environment("no_such_environment")
    with pytest.raises(KeyError):
        make_environment("small_arena", gravity=9.8)
    sub = _sub("multimodal_sensory", "efference_self_model", "reward_gated_plasticity")
    result = run_weave_episode(make_environment("disturbed", seed=2, max_steps=40), sub, canonical_motor_head(), seed=2, collect_telemetry=True)
    assert result.steps == 40 and sub.mechanisms["multimodal_sensory"].ingested == 40 * 16     # the environment feeds the connectome
    assert sub.mechanisms["efference_self_model"].report["source"] == "environment" and float(sub.syn.w_delta.abs().sum()) > 0.0
