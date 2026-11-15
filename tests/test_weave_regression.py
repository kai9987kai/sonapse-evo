"""Regression prevention: the original SONAPSE-EVO 2.0 surface and numerics must be intact.

The baseline files in ``tests/weave_baseline/`` were generated from the untouched 2.0.0 code
(git tag ``baseline/sonapse-evo-2.0.0``) before any NeuroWeave file was wired in.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
import torch

from sonapse_evo import cli
from sonapse_evo.evaluation.benchmark_runner import ForageNavigationTask, run_benchmark_episode
from sonapse_evo.neuroweave import (MECHANISM_IDS, WeaveSubstrate, build_weave_substrate, canonical_motor_head, run_weave_episode,
                                    upgrade_substrate)
from sonapse_evo.neuroweave.compat import build_compatibility_manifest, check_compatibility, golden_baseline, verify_legacy_identity
from sonapse_evo.observatory.live_state import LiveObservatoryState
from sonapse_evo.observatory.weave_state import WeaveObservatoryState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.topology_state import TopologyState

BASELINE = Path(__file__).parent / "weave_baseline"


@pytest.fixture(scope="module")
def manifest():
    return json.loads((BASELINE / "compatibility_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden():
    return json.loads((BASELINE / "golden.json").read_text(encoding="utf-8"))


def _capture(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn(*args, **kwargs)
    return buffer.getvalue().splitlines()


def test_original_api_surface_is_intact(manifest):
    """Every recorded module, class, method, function, CLI option, endpoint and dashboard hook still exists."""
    assert check_compatibility(manifest) == []
    assert len(manifest["modules"]) >= 59 and len(manifest["cli"]["subcommands"]) == 8


def test_manifest_detects_a_removal(manifest):
    broken = json.loads(json.dumps(manifest))
    broken["cli"]["subcommands"]["no-such-command"] = []
    broken["dom_ids"].append("no-such-element")
    violations = check_compatibility(broken)
    assert any("no-such-command" in v for v in violations) and any("no-such-element" in v for v in violations)


def test_golden_numeric_baseline(golden):
    """Topology digest, 25-step rate trace, lesion masks, episode returns and snapshot digest are unchanged."""
    live = golden_baseline()
    for key in live:
        assert json.dumps(live[key], sort_keys=True) == json.dumps(golden[key], sort_keys=True), key


def test_all_mechanisms_disabled_is_bit_identical():
    report = verify_legacy_identity(steps=25)
    assert report["legacy_identity"] and report["neutral_extended_path_identity"]


def test_default_weave_substrate_has_nothing_enabled():
    sub = build_weave_substrate()
    assert isinstance(sub, RecurrentSubstrate) and isinstance(sub, WeaveSubstrate)
    assert sub.weave.profile == "legacy" and not sub.weave.any_enabled() and not sub.weave_active and not sub.mechanisms


def test_episode_runner_matches_original_on_legacy_models():
    topo = TopologyState(total_neurons=768, num_regions=3, seed=42)
    motor = canonical_motor_head()
    env = ForageNavigationTask(sensory_dim=155, max_steps=40, seed=42)
    for seed in (100, 237):
        original = run_benchmark_episode(env, RecurrentSubstrate(topology=topo, seed=42), motor, seed=seed)
        assert run_weave_episode(env, RecurrentSubstrate(topology=topo, seed=42), motor, seed=seed).total_return == original
        assert run_weave_episode(env, build_weave_substrate(seed=42), motor, seed=seed).total_return == original
        assert run_benchmark_episode(env, build_weave_substrate(seed=42), motor, seed=seed) == original


def test_upgrade_substrate_is_non_destructive():
    legacy = RecurrentSubstrate(topology=TopologyState(768, 3, seed=42), seed=42)
    for _ in range(3):
        legacy.forward_step(torch.full((155,), 0.3))
    before = legacy.r.clone()
    upgraded = upgrade_substrate(legacy)
    obs = torch.rand(155)
    assert torch.equal(upgraded.forward_step(obs), legacy.clone_substrate().forward_step(obs))
    assert torch.equal(legacy.r, before)


def test_original_cli_outputs(golden):
    assert [l for l in _capture(cli.run_lesion_repair, "forage", 0.25) if "Tx ID" not in l] == golden["cli"]["lesion_repair"]
    assert _capture(cli.run_curriculum, 42) == golden["cli"]["curriculum"]
    arms = [l for l in _capture(cli.run_benchmark_arms, 5, False) if l.startswith("Arm ")]
    assert arms[0] == golden["cli"]["benchmark_arm_means"][0] and len(arms) == 10
    crit = _capture(cli.run_criticality, 42)
    assert crit[2] == golden["cli"]["criticality"][2] and crit[3] == golden["cli"]["criticality"][3]


def test_original_cli_parser_still_accepts_every_command(monkeypatch, capsys, manifest):
    for command in manifest["cli"]["subcommands"]:
        monkeypatch.setattr("sys.argv", ["sonapse-evo", command, "--help"])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_observatory_feed_is_a_superset_of_the_original():
    """The extended live state keeps every original key and type; the new data sits under one new key."""
    original, extended = LiveObservatoryState().get_full_state(), WeaveObservatoryState().get_full_state()
    for key, value in original.items():
        assert key in extended and type(extended[key]) is type(value), key
        if isinstance(value, dict):
            assert set(value) <= set(extended[key]), key
    assert set(extended) - set(original) == {"weave"}
    assert extended["connectome"]["total_neurons"] == 768


def test_existing_subsystems_accept_a_weave_substrate():
    """Lesions, snapshots, grafts with transactions, criticality and causal interventions on the subclass."""
    from sonapse_evo.contracts.graft_manifest import GraftSpec
    from sonapse_evo.contracts.receipts import EvaluationResult
    from sonapse_evo.diagnostics.criticality import CriticalityDiagnosticsEngine
    from sonapse_evo.evolution.transaction_manager import TransactionManager
    from sonapse_evo.grafts.graft_registry import GraftRegistry
    from sonapse_evo.grafts.low_rank_bridge import LowRankBridge
    from sonapse_evo.substrate.lesions import LesionSpec

    sub = build_weave_substrate(seed=42, enable=["multi_soma", "reward_gated_plasticity"], strict=True)
    sub.apply_lesion(LesionSpec("l1", "inter_region_bridge", severity=0.25))
    assert float(sub.lesion_mask.mean()) < 1.0
    sub.clear_lesion()
    spec = GraftSpec(graft_id="g", graft_type="low_rank", source_neurons=tuple(range(32)), target_neurons=tuple(range(512, 544)),
                     rank=8, max_gain=0.1, parent_digest=sub.topology.compute_digest())
    registry = GraftRegistry(768)
    manager = TransactionManager(sub, registry)
    result = EvaluationResult("g", (1.0,), (2.0,), 1.0, 0.5, True, True, True, True, "paired_rollout")
    sub.forward_step(torch.rand(155))
    before = sub.cell.soma_fast.clone()
    receipt = manager.commit_graft(LowRankBridge(spec, 768), spec, result)
    assert receipt.status == "COMMITTED"
    sub.forward_step(torch.rand(155))
    manager.rollback_graft("g", reason="test")                       # rollback restores weave compartment state too
    assert torch.equal(sub.cell.soma_fast, before)
    clone = sub.clone_substrate()
    assert isinstance(clone, WeaveSubstrate) and clone.weave.enabled_ids() == sub.weave.enabled_ids()
    report = CriticalityDiagnosticsEngine().evaluate_criticality(sub)
    assert report.regime in ("SUBCRITICAL_ORDER", "EDGE_OF_CHAOS_CRITICAL", "SUPERCRITICAL_CHAOTIC")
    assert sub.estimate_jacobian_norm() > 0.0


def test_registry_accounts_for_every_neuroweave_switch():
    from sonapse_evo.neuroweave import MECHANISMS, NEUROWEAVE_FEATURE_IDS, NEUROWEAVE_FEATURE_MAP, PROFILES

    assert len(NEUROWEAVE_FEATURE_IDS) == 31 and all(f in NEUROWEAVE_FEATURE_MAP for f in NEUROWEAVE_FEATURE_IDS)
    assert len(MECHANISM_IDS) == 42 and PROFILES["legacy"] == () and set(PROFILES["max"]) == set(MECHANISM_IDS)
    assert all(MECHANISMS[m].status == "experimental" for m in MECHANISM_IDS)          # nothing promoted by default
    for spec in MECHANISMS.values():
        if spec.is_plugin:
            assert issubclass(spec.load_class(), __import__("sonapse_evo.neuroweave", fromlist=["Mechanism"]).Mechanism)
