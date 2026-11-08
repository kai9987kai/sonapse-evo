from __future__ import annotations

import pytest
import torch
import numpy as np

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.grafts.graft_registry import GraftRegistry
from sonapse_evo.grafts.low_rank_bridge import LowRankBridge
from sonapse_evo.evolution.transaction_manager import TransactionManager, FailureArchive
from sonapse_evo.contracts.graft_manifest import GraftSpec
from sonapse_evo.contracts.receipts import EvaluationResult


def test_stale_parent_digest_rejected():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, seed=42)
    registry = GraftRegistry(768)
    tx_manager = TransactionManager(substrate, registry)

    spec = GraftSpec(
        graft_id="stale_graft",
        graft_type="low_rank",
        source_neurons=(0, 1),
        target_neurons=(2, 3),
        parent_digest="deadbeef_incorrect_digest",
    )
    bridge = LowRankBridge(spec, 768)
    eval_res = EvaluationResult(
        candidate_id=spec.graft_id,
        baseline_returns=(1.0,),
        candidate_returns=(2.0,),
        causal_tau=1.0,
        recovery_ratio=0.5,
        retention_passed=True,
        replay_verified=True,
        resource_passed=True,
        stability_passed=True,
        evidence_kind="test",
    )

    receipt = tx_manager.commit_graft(bridge, spec, eval_res)
    assert receipt.status == "ROLLED_BACK"
    assert "digest mismatch" in receipt.reason.lower()
    assert len(registry.grafts) == 0


def test_successful_commit_and_rollback():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, seed=42)
    registry = GraftRegistry(768)
    tx_manager = TransactionManager(substrate, registry)

    parent_digest = topology.compute_digest()
    spec = GraftSpec(
        graft_id="valid_graft",
        graft_type="low_rank",
        source_neurons=tuple(range(16)),
        target_neurons=tuple(range(16, 32)),
        parent_digest=parent_digest,
    )
    bridge = LowRankBridge(spec, 768)
    eval_res = EvaluationResult(
        candidate_id=spec.graft_id,
        baseline_returns=(1.0,),
        candidate_returns=(2.0,),
        causal_tau=1.0,
        recovery_ratio=0.5,
        retention_passed=True,
        replay_verified=True,
        resource_passed=True,
        stability_passed=True,
        evidence_kind="test",
    )

    receipt = tx_manager.commit_graft(bridge, spec, eval_res)
    assert receipt.status == "COMMITTED"
    assert "valid_graft" in registry.grafts

    # Roll back
    tx_manager.rollback_graft("valid_graft", reason="Manual regression test rollback")
    assert "valid_graft" not in registry.grafts
    assert "valid_graft" not in substrate.active_grafts
    assert len(tx_manager.failure_archive.records) > 0
