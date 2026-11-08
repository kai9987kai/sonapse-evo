from __future__ import annotations

import pytest
import torch
import numpy as np

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.substrate.lesions import LesionSpec, LesionEngine


def test_lesion_mask_severity():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    engine = LesionEngine(seed=42)
    spec = LesionSpec(lesion_id="test_rand", lesion_type="random_edges", severity=0.20, seed=42)
    mask = engine.create_lesion_mask(topology, spec)

    m = len(topology.base_pre)
    lesioned_count = np.sum(mask == 0.0)
    expected = int(m * 0.20)
    assert abs(lesioned_count - expected) <= 1


def test_inter_region_bridge_lesion():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    engine = LesionEngine(seed=42)
    spec = LesionSpec(lesion_id="test_bridge", lesion_type="inter_region_bridge", severity=0.50, seed=42)
    mask = engine.create_lesion_mask(topology, spec)

    pre_regions = topology.region_assignments[topology.base_pre]
    post_regions = topology.region_assignments[topology.base_post]
    bridge_indices = np.flatnonzero(pre_regions != post_regions)

    # All lesioned edges must be bridge edges
    lesioned_indices = np.flatnonzero(mask == 0.0)
    assert len(lesioned_indices) > 0
    assert np.all(np.isin(lesioned_indices, bridge_indices))


def test_intact_weights_unmodified():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    sub = RecurrentSubstrate(topology=topology, seed=42)
    orig_w = sub.base_w.clone()

    sub.apply_lesion(LesionSpec("test_lesion", "random_edges", severity=0.30))
    # Base weights are unchanged; only lesion_mask buffer is modified
    assert torch.all(sub.base_w == orig_w)
    assert torch.mean(sub.lesion_mask).item() < 1.0

    sub.clear_lesion()
    assert torch.all(sub.lesion_mask == 1.0)
