import pytest
import torch
import numpy as np

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.evaluation.rigl import AuthenticRigL, RigLConfig
from sonapse_evo.diagnostics.covariance import StreamingCovarianceTracker
from sonapse_evo.evaluation.matched_controls import MatchedControlsFactory
from sonapse_evo.evaluation.generalization import DynamicFrictionForagingTask, MultiTaskContinualBenchmark


def test_authentic_rigl_pruning_and_regrowth():
    topo = TopologyState(total_neurons=768, num_regions=3, seed=42)
    sub = RecurrentSubstrate(topology=topo, seed=42)

    rigl = AuthenticRigL(sub, config=RigLConfig(initial_drop_fraction=0.20, total_steps=10), seed=42)
    initial_active = rigl.active_count
    assert initial_active > 0

    # Execute pruning and regrowth step
    receipt = rigl.prune_and_regrow()
    assert receipt["pruned"] > 0
    assert receipt["regrown"] == receipt["pruned"]
    # Total active parameters strictly conserved
    assert int(rigl.mask.sum().item()) == initial_active


def test_streaming_covariance_tracker():
    tracker = StreamingCovarianceTracker(num_neurons=16)
    rng = np.random.default_rng(42)

    for _ in range(50):
        # Generate correlated signals between node 0 and node 1
        s = rng.normal(0.0, 1.0)
        vec = rng.normal(0.0, 0.1, size=16)
        vec[0] += s * 2.0
        vec[1] += s * 2.0
        tracker.update(vec)

    cov = tracker.get_covariance()
    assert cov.shape == (16, 16)
    # Covariance between 0 and 1 should be significantly positive
    assert cov[0, 1] > 0.5

    u, v, val = tracker.get_highest_covariance_ports(np.array([0, 2]), np.array([1, 3]))
    assert u == 0
    assert v == 1
    assert val > 0.5


def test_matched_controls_authentic_and_legacy_parity():
    topo = TopologyState(total_neurons=768, num_regions=3, seed=42)
    sub = RecurrentSubstrate(topology=topo, seed=42)
    factory = MatchedControlsFactory(sub, seed=42)

    # Legacy A4 call
    clone_leg = sub.clone_substrate()
    scaled = factory.legacy_A4_weight_scaled_demo(clone_leg)
    assert scaled.base_w is not None

    # Authentic arm builds
    sub_a0, _, _ = factory.build_arm("A0", use_authentic=True)
    sub_a1, _, _ = factory.build_arm("A1", use_authentic=True)
    sub_a4, _, _ = factory.build_arm("A4", use_authentic=True)
    sub_a6, reg_a6, spec_a6 = factory.build_arm("A6", use_authentic=True)

    assert sub_a0 is not None
    assert sub_a1.base_w.requires_grad is True  # Unfrozen for fine-tuning
    assert sub_a4 is not None
    assert reg_a6 is not None
    assert spec_a6 is not None


def test_generalization_tasks_and_continual_profile():
    topo = TopologyState(total_neurons=768, num_regions=3, seed=42)
    sub = RecurrentSubstrate(topology=topo, seed=42)

    task_dyn = DynamicFrictionForagingTask(seed=42)
    obs = task_dyn.reset()
    assert obs.shape == (155,)

    obs, r, done, info = task_dyn.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert "friction" in info
    assert r > 0.0

    continual_bench = MultiTaskContinualBenchmark(seed=42)
    profile = continual_bench.evaluate_continual_profile(sub)
    assert "r1_initial" in profile
    assert "r2_ood" in profile
    assert "forgetting" in profile
    assert "retention_ratio" in profile
