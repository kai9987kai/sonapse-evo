"""Extended lesion library (works on any ``RecurrentSubstrate`` through its edge-level ``lesion_mask``).

The existing ``sonapse_evo.substrate.lesions`` module is untouched; this adds further lesion families, a
matched random control that removes exactly the same number of edges, progressive schedules and receipts.

Kinds
    random_edges                 uniformly random synapses
    hub_targeted                 synapses of the highest in-degree neurons
    region_targeted              synapses touching one region
    sensory_pathway              synapses leaving sensory regions
    memory_pathway               synapses touching memory regions (association when no memory region exists)
    cross_region_communication   synapses between different regions
    node_ablation                every synapse of chosen (or randomly drawn) neurons

``severity`` is the fraction of the *eligible* synapses that is removed (for node_ablation: the fraction
of neurons).  The first ``protect_first`` edges (the canonical sensorimotor pathway) are never removed unless
``protect_first=0``.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

from ...substrate.topology_state import TopologyState
from .regions import RegionLayout, legacy_layout

LESION_KINDS: Tuple[str, ...] = ("random_edges", "hub_targeted", "region_targeted", "sensory_pathway", "memory_pathway",
                                 "cross_region_communication", "node_ablation")


def _layout(topology: TopologyState, layout: Optional[RegionLayout]) -> RegionLayout:
    return layout or getattr(topology, "layout", None) or legacy_layout(topology.total_neurons)


def eligible_edges(topology: TopologyState, kind: str, layout: Optional[RegionLayout] = None, **options: Any) -> np.ndarray:
    """Indices of the edges a lesion of this kind may remove."""
    lay = _layout(topology, layout)
    pre, post, n = topology.base_pre, topology.base_post, topology.total_neurons
    a, b = lay.assignments[pre], lay.assignments[post]
    if kind == "random_edges":
        return np.arange(len(pre))
    if kind == "hub_targeted":
        in_deg = np.bincount(post, minlength=n)
        hubs = np.argsort(-in_deg, kind="stable")[: max(1, int(n * float(options.get("hub_fraction", 0.05))))]
        return np.flatnonzero(np.isin(pre, hubs) | np.isin(post, hubs))
    if kind == "region_targeted":
        rid = lay.region_id(options.get("region", lay.num_regions // 2))
        return np.flatnonzero((a == rid) | (b == rid))
    if kind == "sensory_pathway":
        return np.flatnonzero(np.isin(a, lay.sensory_region_ids))
    if kind == "memory_pathway":
        mem = lay.regions_with_role("memory") or lay.regions_with_role("association")
        return np.flatnonzero(np.isin(a, mem) | np.isin(b, mem))
    if kind == "cross_region_communication":
        return np.flatnonzero(a != b)
    if kind == "node_ablation":
        return np.arange(len(pre))
    raise KeyError(f"unknown lesion kind '{kind}'. Available: {LESION_KINDS}")


def build_lesion_mask(topology: TopologyState, kind: str, severity: float, seed: int = 0, layout: Optional[RegionLayout] = None,
                      protect_first: int = 3, **options: Any) -> np.ndarray:
    """Return a float32 edge mask (1 = intact, 0 = lesioned)."""
    m = len(topology.base_pre)
    mask = np.ones(m, dtype=np.float32)
    severity = float(np.clip(severity, 0.0, 1.0))
    if m == 0 or severity <= 0.0:
        return mask
    rng = np.random.default_rng(seed)
    if kind == "node_ablation":
        n = topology.total_neurons
        nodes = options.get("nodes")
        if nodes is None:
            candidates = np.arange(3 if protect_first else 0, n)
            nodes = rng.choice(candidates, size=max(1, int(len(candidates) * severity)), replace=False)
        drop = np.flatnonzero(np.isin(topology.base_pre, nodes) | np.isin(topology.base_post, nodes))
    else:
        pool = eligible_edges(topology, kind, layout, **options)
        pool = pool[pool >= protect_first]
        if len(pool) == 0:
            return mask
        drop = rng.choice(pool, size=max(1, int(len(pool) * severity)), replace=False)
    drop = drop[drop >= protect_first]
    mask[drop] = 0.0
    return mask


def matched_random_mask(topology: TopologyState, n_removed: int, seed: int = 0, protect_first: int = 3) -> np.ndarray:
    """Random control lesion removing exactly ``n_removed`` synapses."""
    m = len(topology.base_pre)
    mask = np.ones(m, dtype=np.float32)
    pool = np.arange(protect_first, m)
    k = int(min(n_removed, len(pool)))
    if k > 0:
        mask[np.random.default_rng(seed).choice(pool, size=k, replace=False)] = 0.0
    return mask


def apply_lesion_mask(substrate, mask: np.ndarray) -> int:
    """Install an edge mask on a substrate (legacy or weave).  Returns the number of lesioned edges."""
    tensor = torch.as_tensor(mask, dtype=torch.float32, device=substrate.lesion_mask.device)
    if tensor.shape != substrate.lesion_mask.shape:
        raise ValueError("lesion mask length does not match the substrate's base edges")
    substrate.lesion_mask.copy_(tensor)
    return int((tensor == 0).sum().item())


def clear_lesions(substrate) -> None:
    substrate.lesion_mask.fill_(1.0)
    if hasattr(substrate, "set_node_lesions"):
        substrate.set_node_lesions(None)


def lesion_receipt(kind: str, severity: float, seed: int, mask: np.ndarray, **details: Any) -> Dict[str, Any]:
    return {"kind": kind, "severity": float(severity), "seed": int(seed), "edges_total": int(len(mask)),
            "edges_removed": int((mask == 0).sum()), "fraction_removed": float((mask == 0).mean()) if len(mask) else 0.0,
            "mask_digest": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(), **details}


def progressive_schedule(steps: int = 5, max_severity: float = 0.5) -> Iterator[float]:
    """Increasing severities for progressive-damage experiments."""
    for i in range(1, int(steps) + 1):
        yield float(max_severity) * i / int(steps)


__all__ = ["LESION_KINDS", "eligible_edges", "build_lesion_mask", "matched_random_mask", "apply_lesion_mask", "clear_lesions",
           "lesion_receipt", "progressive_schedule"]
