"""Laboratory tests: experiments, promotion, model packages, standalone runner, observatory endpoints, CLI."""
from __future__ import annotations

import copy
import json
import shutil
import socketserver
import subprocess
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from sonapse_evo import cli
from sonapse_evo.neuroweave import build_weave_substrate, canonical_motor_head, run_weave_episode
from sonapse_evo.neuroweave.ecology import make_environment
from sonapse_evo.neuroweave.experiments import (EXPERIMENT_IDS, EXPERIMENTS, MIN_PAIRED_SEEDS, ExperimentSpec, compare, export_receipt,
                                                list_experiments, receipt_to_csv, run_catalog_experiment, run_experiment,
                                                run_lesion_assay, run_topology_assay)
from sonapse_evo.neuroweave.portability import (LIMITS, RUNTIME_JS, build_runner_model, export_legacy_package, export_model_package,
                                                export_snapshot, export_standalone_html, import_model_package, inspect_model_package,
                                                load_package, reference_inference, restore_snapshot, save_package,
                                                validate_model_package)
from sonapse_evo.neuroweave.promotion import PromotionLedger
from sonapse_evo.observatory.weave_server import ACTIONS, create_weave_handler
from sonapse_evo.observatory.weave_state import WeaveObservatoryState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.topology_state import TopologyState


@pytest.fixture(scope="module")
def trained():
    motor = canonical_motor_head()
    sub = build_weave_substrate(seed=42, profile="max", strict=True)
    env = make_environment("neuroweave_1200x660", seed=5, max_steps=50)
    run_weave_episode(env, sub, motor, seed=5)
    return sub, motor, env


@pytest.fixture(scope="module")
def package(trained):
    sub, motor, env = trained
    return json.loads(json.dumps(export_model_package(sub.clone_substrate(), motor_head=motor, environment=env, training_history={"episodes": 1})))


# ------------------------------------------------------------------------ experiments
def test_catalogue_covers_the_twenty_four_experiments():
    assert len(EXPERIMENT_IDS) == 24 and len({e["id"] for e in list_experiments()}) == 24
    required = {"multimodal_sensory_integration", "affective_neuromodulation", "micro_collective_learning", "expert_routing",
                "meta_plasticity", "predictive_replay", "connectivity_context_attention", "flyhash_sparse_projections", "stdp",
                "adaptive_conduction", "quantised_synapses", "hierarchical_temporal_memory", "uncertainty_gated_propagation",
                "predictive_coding", "structural_plasticity", "associative_memory", "action_grounded_self_modelling",
                "multi_soma_integration", "multi_region_routing", "lesion_robustness", "topology_optimisation", "neural_grafting",
                "cross_modal_learning", "specialist_expert_evolution"}
    assert set(EXPERIMENT_IDS) == required
    assert all(set(s.baseline) != set(s.candidate) or s.kind != "paired" or s.params for s in EXPERIMENTS.values())
    with pytest.raises(KeyError):
        ExperimentSpec("x", "x", "x", candidate=("no_such_mechanism",))
    with pytest.raises(ValueError):
        ExperimentSpec("x", "x", "x", candidate=("pair_stdp",), ablations=("multi_soma",))
    with pytest.raises(KeyError):
        run_catalog_experiment("no_such_experiment")


def test_experiments_are_reproducible_with_baseline_candidate_ablation_and_held_out_evaluation(tmp_path):
    spec = ExperimentSpec("unit", "Unit", "h", candidate=("multi_soma", "dendritic_compartments"), ablations=("dendritic_compartments",),
                          episodes_train=1, episodes_eval=1, seeds=(11, 23))
    a, b = run_experiment(spec), run_experiment(spec)
    assert a["result_digest"] == b["result_digest"]                                        # deterministic for fixed seeds
    assert run_experiment(spec, seeds=(11, 24))["result_digest"] != a["result_digest"]
    assert set(a["results"]["runs"]) == {"baseline", "candidate", "candidate_minus_dendritic_compartments"}
    assert a["results"]["summary"]["baseline"]["mechanisms"] == [] and a["held_out_seed_offset"] == 1000
    run = a["results"]["runs"]["candidate"][0]
    assert len(run["train_returns"]) == 1 and len(run["heldout_returns"]) == 1 and run["cost"]["measured_ms_per_step"] > 0
    assert run["cost"]["measured_state_bytes"] > 0 and run["faults"] == []
    assert a["matched_conditions"]["candidate_cost_ratio_measured"] > 1.0                   # extra compute is reported, not hidden
    assert a["verdict"] == "inconclusive" and a["evidence_tier"] == "SIMULATED_ASSAY"        # two seeds can never "succeed"
    assert "candidate_vs_candidate_minus_dendritic_compartments" in a["results"]["comparisons"]
    rows = receipt_to_csv(a).strip().splitlines()
    assert len(rows) == 1 + 3 * 2 and rows[0].startswith("experiment_id,arm,seed")
    files = export_receipt(a, str(tmp_path))
    assert json.loads(open(files["json"], encoding="utf-8").read())["result_digest"] == a["result_digest"] and files["csv"].endswith(".csv")


def test_verdict_rule_never_rewards_complexity_or_small_samples():
    better, base = [5.0, 6.0, 5.5, 6.2, 5.8, 6.1], [1.0, 1.2, 0.9, 1.1, 1.0, 1.3]
    strong = compare(better, base)
    assert strong["verdict"] == "supported" and strong["evidence_tier"] == "PAIRED_EXPERIMENT" and strong["wins"] == 6
    assert compare(base, better)["verdict"] == "not_supported"
    assert compare(better[:3], base[:3])["verdict"] == "inconclusive" and len(better[:3]) < MIN_PAIRED_SEEDS
    assert compare([1.0, 1.1, 0.9, 1.0, 1.2], [1.0, 1.0, 1.0, 1.1, 1.1])["verdict"] == "inconclusive"
    assert compare([3.0], [1.0])["verdict"] == "inconclusive"


def test_topology_and_lesion_assays_run_with_matched_controls():
    topo = run_topology_assay(kinds=("legacy", "ring_lattice", "multi_region"), seeds=(11, 23), episodes=1)
    assert set(topo["results"]["summary"]) == {"legacy", "ring_lattice", "multi_region"} and topo["verdict"] == "descriptive"
    assert topo["results"]["summary"]["ring_lattice"]["clustering_coefficient_mean"] > topo["results"]["summary"]["legacy"]["clustering_coefficient_mean"]
    assert set(topo["results"]["comparisons"]) == {"ring_lattice_vs_legacy", "multi_region_vs_legacy"}
    assert run_topology_assay(kinds=("legacy",), seeds=(11,), episodes=1)["result_digest"] == run_topology_assay(kinds=("legacy",), seeds=(11,), episodes=1)["result_digest"]
    lesion = run_lesion_assay(seeds=(11, 23), conditions=("sensory_pathway", "hub_targeted"), severity=0.6, recovery_episodes=1,
                              progressive_steps=3, mechanisms=("reward_gated_plasticity",))
    entry = lesion["results"]["runs"][0]["conditions"]["sensory_pathway"]
    assert entry["edges_removed"] == entry["receipt"]["edges_removed"] > 0
    assert {"return", "success_rate", "mean_steps", "retention", "functional_redundancy", "recovery_curve", "post_recovery_return",
            "connectivity_change"} <= set(entry["targeted"]) and set(entry["matched_random"]) == set(entry["targeted"])
    assert len(lesion["results"]["runs"][0]["progressive_random"]) == 3 and lesion["matched_conditions"]["intact_substrate_untouched"]
    assert "sensory_pathway_targeted_vs_matched_random" in lesion["results"]["comparisons"]
    quick = run_catalog_experiment("topology_optimisation", quick=True, kinds=("legacy", "uniform_random"))
    assert quick["kind"] == "topology" and run_catalog_experiment("lesion_robustness", quick=True)["kind"] == "lesion"


# -------------------------------------------------------------------------- promotion
def _receipt(verdict="supported", tier="PAIRED_EXPERIMENT", faults=0, cost=2.0):
    return {"experiment_id": "unit", "result_digest": "abc", "verdict": verdict, "evidence_tier": tier,
            "spec": {"candidate": ["multi_soma", "pair_stdp"]}, "matched_conditions": {"candidate_cost_ratio_measured": cost},
            "results": {"summary": {"baseline": {"faults": 0}, "candidate": {"faults": faults}}}}


def test_promotion_is_gated_atomic_and_reversible(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = PromotionLedger(str(path))
    assert ledger.stable == [] and ledger.stable_config().profile == "legacy"
    for bad in (_receipt(verdict="inconclusive"), _receipt(tier="SIMULATED_ASSAY"), _receipt(faults=1), _receipt(cost=999.0)):
        decision = ledger.propose(["multi_soma"], bad)
        assert decision["status"] == "REJECTED" and decision["failed_gates"] and ledger.stable == []
    assert ledger.propose(["fast_weights"], _receipt())["failed_gates"] == ["receipt_covers_mechanisms"]
    assert ledger.propose(["multi_soma"], _receipt(), regression={"legacy_identity": False})["failed_gates"] == ["compatibility"]
    assert len(ledger.failures()) == 6                                                     # failed experiments are kept, not deleted
    promoted = ledger.propose(["multi_soma"], _receipt())
    assert promoted["status"] == "PROMOTED" and ledger.stable == ["multi_soma"] and all(promoted["gates"].values())
    ledger.propose(["pair_stdp"], _receipt())
    assert ledger.stable == ["multi_soma", "pair_stdp"] and ledger.stable_config().enabled_ids() == ("multi_soma", "pair_stdp")
    reloaded = PromotionLedger(str(path))
    assert reloaded.stable == ledger.stable and len(reloaded.failures()) == 6 and reloaded.to_dict()["default_profile"] == "legacy"
    assert ledger.rollback()["stable"] == ["multi_soma"] and ledger.rollback()["stable"] == []
    assert ledger.rollback()["status"] == "NOTHING_TO_ROLL_BACK" and not build_weave_substrate().weave.any_enabled()


# ------------------------------------------------------------------------ portability
def test_package_round_trip_continues_exactly(trained, package, tmp_path):
    sub, motor, env = trained
    report = validate_model_package(package)
    assert report["ok"] and report["issues"] == [] and report["summary"]["neurons"] == 768
    for section in ("topology", "neuronPopulations", "substrate", "weave", "sensory", "memory", "agent", "environment", "training",
                    "experiment", "provenance", "capabilities", "integrity", "counts"):
        assert section in package, section
    weave = package["weave"]
    assert {"cell", "synapses", "signals", "mechanisms", "config", "interfaces"} <= set(weave) and len(weave["mechanisms"]) == 27
    assert package["capabilities"]["live_sensory_operation"] is False and package["sensory"]["raw_media_stored"] is False
    size = save_package(package, str(tmp_path / "model.json"))
    imported, extras = import_model_package(load_package(str(tmp_path / "model.json")), strict=True)
    assert size < LIMITS["max_bytes"] and imported is not sub and extras["motor_head"] is not None
    reference = sub.clone_substrate()
    gen = torch.Generator().manual_seed(2)
    for _ in range(6):
        obs = torch.rand(155, generator=gen) * 0.5
        for s in (reference, imported):
            s.deliver_feedback(reward=0.4, issued_action=[0.3, 0, 0, 0, 0], intended_action=[0.3, 0, 0, 0, 0])
        assert torch.equal(reference.forward_step(obs), imported.forward_step(obs))          # learning continues identically
    assert torch.equal(extras["motor_head"].weight, motor.weight) and extras["environment"]["name"] == "neuroweave_1200x660"


def test_validation_rejects_tampered_malformed_and_oversized_packages(package):
    tampered = copy.deepcopy(package)
    tampered["substrate"]["edges"]["w"]["data"][5] += 0.25
    assert any("integrity" in i for i in validate_model_package(tampered)["issues"])
    with pytest.raises(ValueError):
        import_model_package(tampered)
    cases = {
        "edges.post": lambda p: p["substrate"]["edges"]["post"]["data"].__setitem__(7, 10 ** 6),
        "non-finite": lambda p: p["substrate"]["edges"]["w"]["data"].__setitem__(3, None),
        "total_neurons": lambda p: p["topology"].__setitem__("total_neurons", 10 ** 7),
        "substrate.r": lambda p: p["substrate"]["r"]["data"].__setitem__(0, 7.0),
        "delay": lambda p: p["weave"]["synapses"]["delay"]["data"].__setitem__(0, 99),
        "booleans": lambda p: p["weave"]["config"]["enabled"].__setitem__("multi_soma", "yes"),
        "unsupported format version": lambda p: p.__setitem__("formatVersion", "9.0.0"),
    }
    for needle, mutate in cases.items():
        bad = copy.deepcopy(package)
        mutate(bad)
        result = validate_model_package(bad)
        assert not result["ok"] and any(needle in i or "malformed" in i for i in result["issues"]), needle
    assert not validate_model_package("{broken")["ok"] and not validate_model_package("[1, 2]")["ok"] and not validate_model_package({})["ok"]
    assert not validate_model_package("x" * (LIMITS["max_bytes"] + 1))["ok"]
    future = copy.deepcopy(package)
    future["weave"]["config"]["enabled"]["mechanism_from_the_future"] = True
    future.pop("integrity")
    result = validate_model_package(future)
    assert result["ok"] and any("unknown mechanisms" in w for w in result["warnings"])
    assert "mechanism_from_the_future" not in import_model_package(future)[0].weave.enabled
    info = inspect_model_package(json.dumps(package))
    assert info["imported"] is False and info["validation"]["ok"] and len(info["mechanisms"]) == 42 and info["has_agent"]


def test_legacy_packages_and_snapshots(trained):
    legacy = RecurrentSubstrate(topology=TopologyState(768, 3, seed=42), seed=42)
    legacy.forward_step(torch.rand(155))
    old = json.loads(json.dumps(export_legacy_package(legacy)))
    assert old["formatVersion"] == "2.0.0" and validate_model_package(old)["ok"]
    restored, _ = import_model_package(old)
    obs = torch.rand(155)
    assert type(restored) is RecurrentSubstrate and torch.equal(restored.forward_step(obs), legacy.forward_step(obs))
    plain = json.loads(json.dumps(export_model_package(legacy)))
    assert plain["weave"] is None and plain["capabilities"]["continued_learning"] is False and type(import_model_package(plain)[0]) is RecurrentSubstrate
    sub = trained[0].clone_substrate()
    snap = json.loads(json.dumps(export_snapshot(sub)))
    probe = sub.clone_substrate()
    sub.forward_step(torch.rand(155))
    restore_snapshot(sub, snap)
    assert torch.equal(sub.forward_step(obs), probe.forward_step(obs))
    snap["step_count"] += 1
    with pytest.raises(ValueError):
        restore_snapshot(sub, snap)


def test_standalone_runner_matches_python_and_states_its_limits(trained, package, tmp_path):
    legacy = RecurrentSubstrate(topology=TopologyState(768, 3, seed=42), seed=42)
    model = build_runner_model(json.loads(json.dumps(export_model_package(legacy, motor_head=canonical_motor_head()))))
    observations = [np.random.default_rng(i).random(155) * 0.5 for i in range(5)]
    reference = reference_inference(model, observations)
    actual = np.stack([legacy.forward_step(torch.as_tensor(o, dtype=torch.float32)).numpy() for o in observations])
    assert float(np.abs(reference - actual).max()) < 1e-5                                    # exact for the original pathway
    assert model["capabilities"]["not_executed"] == [] and model["capabilities"]["motor_readout"]
    full = build_runner_model(package)
    caps = full["capabilities"]
    assert caps["frozen_inference"] and not caps["continued_learning"] and not caps["live_sensory_operation"]
    assert len(caps["not_executed"]) >= 10 and caps["micro_collective_prediction"] in (True, False)
    page = export_standalone_html(package, str(tmp_path / "runner.html"), title="<script>alert(1)</script>")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "<script>alert(1)</script>" not in page       # title is escaped
    embedded = page.split("const MODEL=", 1)[1].split("\nclass SonapseRuntime", 1)[0]
    assert "</script" not in embedded and "getUserMedia" not in page and "fetch(" not in page                # self-contained, no devices
    assert "Not executed here" in page and "Continued learning" in page and (tmp_path / "runner.html").exists()
    with pytest.raises(ValueError):
        build_runner_model({"format": "nope"})
    if shutil.which("node") is None:
        pytest.skip("Node.js not available: JavaScript parity not checked on this machine")
    harness = ("const MODEL=" + json.dumps(full) + ";\n" + RUNTIME_JS + "\nconst rt=new SonapseRuntime(MODEL);const OBS="
               + json.dumps([o.tolist() for o in observations])
               + ";const out=[];for(const o of OBS){rt.step(o);out.push(Array.from(rt.r));}"
               + "console.log(JSON.stringify({rates:out,action:rt.action(),mc:rt.predictSensory(OBS[0].slice(0,16)),regions:rt.regionRates().length}));")
    script = tmp_path / "harness.js"
    script.write_text(harness, encoding="utf-8")
    result = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout)
    assert float(np.abs(np.array(result["rates"]) - reference_inference(full, observations)).max()) < 1e-9   # JS == Python reference
    assert len(result["action"]) == 5 and result["regions"] == 3
    if full["microCollective"] is not None:
        from sonapse_evo.neuroweave.collective import MicroCollective

        mc = MicroCollective()
        mc.load_snapshot(package["weave"]["mechanisms"]["micro_collective"]["snapshot"])
        assert float(np.abs(np.array(result["mc"]) - mc.predict(observations[0][:16])).max()) < 1e-9


# ------------------------------------------------------------------------- observatory
@pytest.fixture(scope="module")
def observatory():
    return WeaveObservatoryState(seed=42)


def test_observatory_controls_change_the_real_model(observatory):
    state = observatory
    assert state.weave_state()["telemetry"]["enabled"] == [] and len(state.weave_options()["experiments"]) == 24
    assert state.set_mechanism("multimodal_sensory", True)["attached"] and state.set_mechanism("multi_soma", True)["enabled"]
    assert "multimodal_sensory" in state.substrate.mechanisms and state.substrate.weave.is_enabled("multi_soma")
    before = int(state.substrate.step_count)
    state.weave_step(6)
    assert int(state.substrate.step_count) == before + 6 and len(state.weave_history) == 6
    assert state.ingest_sensors({"visual_luminance": 0.7, "audio_low": 0.4, "bogus": "x"})["raw_media_stored"] is False
    hub = state.substrate.mechanisms["multimodal_sensory"]
    assert hub.manifest.provenance["visual_luminance"] == "direct_sensory" and hub.manifest.camera_enabled and state.live_sensors
    state.set_channel_mode("olfactory", "disabled")
    assert hub.manifest.modes["olfactory"].value == "disabled" and state.release_sensors()["live_sensors"] is False
    assert state.apply_lesion("cross_region_communication", 0.4)["edges_removed"] > 0 and state.weave_state()["lesion"]["edges_lesioned"] > 0
    assert state.clear_lesion()["status"] == "CLEARED" and state.weave_state()["lesion"]["edges_lesioned"] == 0
    detail = state.neuron_detail(3)
    assert detail["index"] == 3 and {"prox", "soma_gating", "release", "myelin"} <= set(detail["compartments"]) and len(detail["branches"]) == 2
    view = state.weave_state()
    assert len(view["regions"]) == 3 and len(view["somas"]) == 8 and view["environment"]["width"] == 1200 and view["graph"]["edges"] > 5000
    state.set_running(True)
    state.tick_simulation()
    assert state.step_idx >= 1 and int(state.substrate.step_count) > before + 6             # original tick still runs, plus the live model
    state.set_running(False)
    assert state.set_profile("max")["mechanisms"] == list(state.substrate.weave.enabled_ids()) and len(state.substrate.mechanisms) == 27
    assert state.rebuild(seed=7, profile="cells", total_neurons=480, topology_kind="multi_region")["total_neurons"] == 480
    assert state.substrate.total_neurons == 480 and state.substrate.layout.num_regions == 10
    with pytest.raises(ValueError):
        state.rebuild(total_neurons=300)
    with pytest.raises(KeyError):
        state.set_mechanism("no_such", True)
    assert state.regression_check()["legacy_identity"] and state.profile(5)["measured"]["ms_per_step"] > 0
    assert state.fpga_manifest()["validation"]["manifest_issues"] == []


def test_observatory_import_is_staged_and_experiments_leave_the_live_model_alone(observatory, package):
    state = observatory
    state.rebuild(seed=42, profile="legacy")
    live = state.substrate
    report = state.validate_import(json.dumps(package))
    assert report["ok"] and report["active_model_changed"] is False and state.substrate is live and state.staged_import is not None
    bad = copy.deepcopy(package)
    bad["substrate"]["edges"]["w"]["data"][0] = 99.0
    assert not state.validate_import(json.dumps(bad))["ok"] and state.staged_import is None    # failure clears the staged model
    assert state.activate_import()["status"] == "NOTHING_STAGED" and state.substrate is live
    state.validate_import(json.dumps(package))
    assert "SonapseRuntime" in state.standalone_html(staged=True)
    assert state.activate_import()["status"] == "ACTIVATED" and state.substrate is not live and len(state.substrate.weave.enabled_ids()) == 42
    exported = state.export_package()
    assert validate_model_package(exported)["ok"]
    active = state.substrate
    steps = int(active.step_count)
    result = state.run_experiment("quantised_synapses", quick=True, seeds=[11])
    assert result["status"] == "DONE" and state.substrate is active and int(active.step_count) == steps
    assert state.last_receipt()["experiment_id"] == "quantised_synapses" and state.propose_promotion()["status"] == "REJECTED"
    assert state.promotion.stable == [] and state.rollback_promotion()["status"] == "NOTHING_TO_ROLL_BACK"


def test_http_endpoints_validate_input_and_keep_original_routes():
    state = WeaveObservatoryState(seed=42)
    server = socketserver.TCPServer(("127.0.0.1", 0), create_weave_handler(state))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=180) as r:
            return r.status, r.read().decode("utf-8"), dict(r.headers)

    def post(body, headers=None, raw=False):
        data = body if raw else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(base + "/api/weave/action", data=data, method="POST", headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    try:
        assert json.loads(get("/api/health")[1])["status"] == "healthy"                      # original routes still served
        assert "mode_a" in json.loads(get("/api/live_state")[1]) and "weave" in json.loads(get("/api/live_state")[1])
        assert "SONAPSE-EVO" in get("/")[1] and "NeuroWeave Observatory" in get("/weave.html")[1] and 'href="/weave.html"' in get("/")[1]
        assert len(json.loads(get("/api/weave/options")[1])["catalogue"]) == 42
        assert post({"action": "set_mechanism", "id": "fast_weights", "enabled": True})[1]["enabled"] is True
        assert "fast_weights" in json.loads(get("/api/weave/state")[1])["telemetry"]["enabled"]
        assert post({"action": "step", "n": 3})[1]["steps"] == 3 and post({"action": "run", "running": False})[1]["status"] == "PAUSED"
        assert json.loads(get("/api/weave/neuron?i=5")[1])["index"] == 5
        status, body, headers = get("/api/weave/export")
        assert status == 200 and "attachment" in headers["Content-Disposition"] and validate_model_package(body)["ok"]
        assert "SonapseRuntime" in get("/api/weave/standalone")[1] and json.loads(get("/api/weave/fpga")[1])["format"] == "SONAPSE_FPGA_MAPPING"
        assert post({"action": "validate_import", "package": body})[1]["ok"] is True
        assert post({"action": "no_such_action"})[0] == 404 and post({"nothing": 1})[0] == 400
        assert post(b"{not json", raw=True)[0] == 400 and post({"action": "set_mechanism", "id": "nope", "enabled": True})[0] == 400
        assert post({"action": "step", "n": 1}, headers={"Origin": "http://evil.example"})[0] == 403        # cross-origin refused
        assert post({"action": "step", "pad": "x" * 300_000})[0] == 413                                     # oversized body refused
        assert post({"action": "sensors", "values": {"visual_motion": 0.5}})[1]["status"] == "IGNORED"       # hub not enabled
        assert "Access-Control-Allow-Origin" not in get("/api/weave/state")[2]
        assert set(ACTIONS) >= {"set_mechanism", "rebuild", "sensors", "lesion", "experiment", "validate_import", "activate_import"}
    finally:
        server.shutdown()
        server.server_close()


# -------------------------------------------------------------------------------- CLI
def _cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr("sys.argv", ["sonapse-evo", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value.code, capsys.readouterr().out


def test_weave_cli_commands(monkeypatch, capsys, tmp_path):
    code, out = _cli(monkeypatch, capsys, "weave-mechanisms")
    assert code == 0 and "42 switchable mechanisms" in out and "31 of 31" in out
    code, out = _cli(monkeypatch, capsys, "weave-compat")
    assert code == 0 and out.count("PASS") >= 4 and "FAIL" not in out
    code, out = _cli(monkeypatch, capsys, "weave-smoke")
    assert code == 0 and "All NeuroWeave smoke checks passed" in out
    code, out = _cli(monkeypatch, capsys, "max", "--episodes", "1", "--max-steps", "30", "--export", str(tmp_path / "max"))
    assert code == 0 and "mechanisms: 42" in out and "Mechanism faults: 0" in out and "Legacy model, same seeds" in out
    assert validate_model_package((tmp_path / "max" / "model_package.json").read_text(encoding="utf-8"))["ok"]
    code, out = _cli(monkeypatch, capsys, "weave-run", "--enable", "multi_soma,pair_stdp", "--episodes", "1")
    assert code == 0 and "mechanisms: 2" in out
    code, out = _cli(monkeypatch, capsys, "weave-experiment", "--list")
    assert code == 0 and out.count("\n") == 24
    code, out = _cli(monkeypatch, capsys, "weave-experiment", "quantised_synapses", "--seeds", "11", "--export", str(tmp_path / "exp"))
    assert code == 0 and "Verdict: inconclusive" in out and (tmp_path / "exp" / "quantised_synapses.csv").exists()
    code, out = _cli(monkeypatch, capsys, "weave-export", "--out", str(tmp_path / "out"), "--profile", "cells", "--train-episodes", "0")
    assert code == 0 and all((tmp_path / "out" / f).exists() for f in ("model_package.json", "standalone.html", "fpga_mapping.json"))
    code, out = _cli(monkeypatch, capsys, "weave-validate", str(tmp_path / "out" / "model_package.json"))
    assert code == 0 and out.startswith("VALID") and "nothing was imported" in out
    broken = tmp_path / "broken.json"
    broken.write_text((tmp_path / "out" / "model_package.json").read_text(encoding="utf-8").replace('"leak":0.35', '"leak":0.36'), encoding="utf-8")
    code, out = _cli(monkeypatch, capsys, "weave-validate", str(broken))
    assert code == 1 and "REJECTED" in out and "integrity" in out
    code, out = _cli(monkeypatch, capsys, "weave-profile", "--sizes", "768", "--steps", "4")
    assert code == 0 and "ms/step" in out and "no acceleration is claimed" in out
    code, out = _cli(monkeypatch, capsys, "weave-topology", "--kinds", "legacy,ring_lattice", "--seeds", "11")
    assert code == 0 and "ring_lattice" in out
    code, out = _cli(monkeypatch, capsys, "weave-lesion", "--seeds", "11", "--severity", "0.5")
    assert code == 0 and "sensory_pathway" in out
