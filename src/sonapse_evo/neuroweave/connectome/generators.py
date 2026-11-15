"""Connectome generators: thirteen topology families behind one function.

``build_topology(kind, ...)`` returns an ordinary ``TopologyState`` (explicit edge arrays, so the existing
default generator is not invoked) that every existing component accepts.  All generators are seeded and
obey the same rules:

* no self-loops, no duplicate (pre, post) pairs;
* the sign of a synapse is a property of its SOURCE neuron (``layout.signs()``);
* magnitudes are uniform in [0.05, 0.25] within a region and [0.02, 0.15] between regions (the ranges of
  the existing default graph);
* with ``canonical_pathway=True`` the first three edges are the existing sensorimotor pathway
  ``0 -> m, 1 -> m+1, 2 -> m+2`` (weight +0.80, ``m = layout.motor_offset``) so the existing foraging
  benchmark and motor read-out keep working on any topology.

No generator encodes a biological connectome; they are synthetic control conditions.  A more elaborate
graph is not assumed to be better - that is what the topology assay measures.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...substrate.topology_state import TopologyState
from .regions import ROLE_ORDER, RegionLayout, legacy_layout, ten_region_layout

TOPOLOGY_KINDS: Tuple[str, ...] = (
    "legacy", "clustered_small_world", "hub_skewed", "uniform_random", "degree_shuffled", "erdos_renyi", "hub_capped",
    "ring_lattice", "hierarchical_modular", "multi_region", "dense_multi_region", "multiscale_regional", "hybrid",
)
_HIGHWAYS = [("sensory", "association"), ("association", "memory"), ("memory", "motor"), ("association", "motor")]


def _clustered(n: int, rng: np.random.Generator, mean_degree: float, ring_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    radius = max(2, int(round(mean_degree / 2))) if ring_only else 2
    idx = np.arange(n)
    pre, post = [], []
    for r in range(1, radius + 1):
        pre += [idx, idx]
        post += [(idx + r) % n, (idx - r) % n]
    if not ring_only:
        extra = max(0, int(n * (mean_degree - 2 * radius)))
        src = rng.integers(0, n, size=extra)
        dst = rng.integers(0, n, size=extra)
        local = rng.random(extra) < 0.72
        offset = rng.integers(1, 6, size=extra) * rng.choice([-1, 1], size=extra)
        dst = np.where(local, (src + offset) % n, dst)
        pre.append(src)
        post.append(dst)
    return np.concatenate(pre), np.concatenate(post)


def _random_pairs(n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    return rng.integers(0, n, size=m), rng.integers(0, n, size=m)


def _hub_skewed(n: int, rng: np.random.Generator, mean_degree: float) -> Tuple[np.ndarray, np.ndarray]:
    weights = 1.0 / np.power(1.0 + np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(1.0, n / 2), 1.6)
    m = int(n * mean_degree)
    return rng.integers(0, n, size=m), rng.choice(n, size=m, p=weights / weights.sum())


def _within(members: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    return members[pairs[0]], members[pairs[1]]


def _dedupe(pre: np.ndarray, post: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    keep = pre != post
    pre, post = pre[keep].astype(np.int64), post[keep].astype(np.int64)
    _, first = np.unique(pre * n + post, return_index=True)
    first.sort()
    return pre[first], post[first]


def _cap_in_degree(pre: np.ndarray, post: np.ndarray, cap: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(post, kind="stable")
    sorted_post = post[order]
    start = np.searchsorted(sorted_post, sorted_post, side="left")
    keep = np.zeros(len(post), dtype=bool)
    keep[order[(np.arange(len(post)) - start) < cap]] = True
    return pre[keep], post[keep]


def _regional(layout: RegionLayout, rng: np.random.Generator, local_degree: float, global_degree: float,
              highways: bool = True, fan: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    n = layout.total_neurons
    pre, post = [], []
    for rid in range(layout.num_regions):
        members = layout.indices(rid)
        m = int(len(members) * local_degree)
        pre.append(rng.choice(members, size=m))
        post.append(rng.choice(members, size=m))
    m = int(n * global_degree)
    src, dst = rng.integers(0, n, size=m), rng.integers(0, n, size=m)
    inter = layout.assignments[src] != layout.assignments[dst]
    pre.append(src[inter])
    post.append(dst[inter])
    if highways:                                     # ordered cross-modal highways and feedback routes
        routes: List[Tuple[int, int]] = []
        for a_role, b_role in _HIGHWAYS:
            routes += [(a, b) for a in layout.regions_with_role(a_role) for b in layout.regions_with_role(b_role)]
        sensory, motor = layout.regions_with_role("sensory"), layout.regions_with_role("motor")
        if sensory and motor:
            routes.append((motor[0], sensory[-1]))                                   # motor -> interoceptive feedback
        routes += [(b, a) for a in layout.regions_with_role("association") for b in layout.regions_with_role("memory")]
        for a, b in routes:
            srcs, dsts = layout.indices(a), layout.indices(b)
            k = min(fan, len(dsts))
            for j in range(k):
                pre.append(srcs)
                post.append(dsts[(j * 17 + srcs) % len(dsts)])
        for a in layout.regions_with_role("association"):                            # sparse predictive feedback
            srcs = layout.indices(a)
            for s in sensory:
                dsts = layout.indices(s)
                pre.append(srcs[:: max(1, len(srcs) // max(1, len(dsts) // 2))])
                post.append(dsts[(7 * pre[-1]) % len(dsts)])
    return np.concatenate(pre), np.concatenate(post)


def degree_preserving_shuffle(topology: TopologyState, seed: int = 0, swaps_per_edge: int = 8, protect_first: int = 3) -> TopologyState:
    """Double-edge swaps ``(a->b, c->d) => (a->d, c->b)``: every in- and out-degree is preserved exactly."""
    rng = np.random.default_rng(seed)
    n = topology.total_neurons
    pre, post, w = topology.base_pre.copy(), topology.base_post.copy(), topology.base_w.copy()
    e = len(pre)
    pairs = set((pre * n + post).tolist())
    pool = np.arange(protect_first, e)
    if len(pool) >= 2:
        picks = rng.choice(pool, size=(len(pool) * swaps_per_edge, 2))
        for i, j in picks:
            a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
            if i == j or a == c or b == d or a == d or c == b:
                continue
            k1, k2 = a * n + d, c * n + b
            if k1 in pairs or k2 in pairs:
                continue
            pairs.discard(a * n + b)
            pairs.discard(c * n + d)
            pairs.add(k1)
            pairs.add(k2)
            post[i], post[j] = d, b
    shuffled = TopologyState(total_neurons=n, num_regions=topology.num_regions, region_names=topology.region_names,
                             region_assignments=topology.region_assignments.copy(), base_pre=pre, base_post=post, base_w=w,
                             seed=topology.seed)
    for attr in ("layout", "kind", "canonical_pathway"):
        if hasattr(topology, attr):
            setattr(shuffled, attr, getattr(topology, attr))
    shuffled.kind = "degree_shuffled"
    return shuffled


def build_topology(kind: str, total_neurons: int = 768, layout: Optional[RegionLayout] = None, seed: int = 42,
                   mean_degree: float = 8.0, canonical_pathway: bool = True, **options: Any) -> TopologyState:
    if kind not in TOPOLOGY_KINDS:
        raise KeyError(f"unknown topology kind '{kind}'. Available: {TOPOLOGY_KINDS}")
    n = int(total_neurons)
    regional = kind in ("multi_region", "dense_multi_region", "multiscale_regional")
    if layout is None:
        layout = ten_region_layout(n) if regional else legacy_layout(n)
    if layout.total_neurons != n:
        raise ValueError("layout size does not match total_neurons")
    if kind == "legacy":
        if n < 576:
            raise ValueError("the legacy generator hard-codes motor neurons 512-514 and needs at least 576 neurons")
        topo = TopologyState(total_neurons=n, num_regions=3, seed=seed)
        topo.layout, topo.kind, topo.canonical_pathway = legacy_layout(n), "legacy", True
        return topo
    rng = np.random.default_rng(seed)
    if kind in ("clustered_small_world", "degree_shuffled", "hub_capped"):
        pre, post = _clustered(n, rng, mean_degree)
    elif kind == "ring_lattice":
        pre, post = _clustered(n, rng, mean_degree, ring_only=True)
    elif kind == "hub_skewed":
        pre, post = _hub_skewed(n, rng, mean_degree)
    elif kind == "uniform_random":
        pre, post = _random_pairs(n, int(n * mean_degree), rng)
    elif kind == "erdos_renyi":
        pre, post = _random_pairs(n, int(rng.binomial(n * (n - 1), min(1.0, mean_degree / max(1, n - 1)))), rng)
    elif kind == "hierarchical_modular":
        module = int(options.get("module_size", 16))
        m = int(n * mean_degree)
        src = rng.integers(0, n, size=m)
        level = rng.choice(3, size=m, p=options.get("level_probabilities", [0.6, 0.3, 0.1]))
        in_module = (src // module) * module + rng.integers(0, module, size=m)
        starts, sizes = np.array(layout.starts), np.array(layout.sizes)
        reg = layout.assignments[src]
        in_region = starts[reg] + rng.integers(0, 1 << 30, size=m) % sizes[reg]
        dst = np.where(level == 0, np.minimum(in_module, n - 1), np.where(level == 1, in_region, rng.integers(0, n, size=m)))
        pre, post = src, dst
    elif kind == "multi_region":
        pre, post = _regional(layout, rng, options.get("local_degree", 8.0), options.get("global_degree", 8.0))
    elif kind == "dense_multi_region":
        pre, post = _regional(layout, rng, options.get("local_degree", 15.0), options.get("global_degree", 24.0))
    elif kind == "multiscale_regional":
        parts = [_regional(layout, rng, options.get("local_degree", 6.0), options.get("global_degree", 1.0), highways=True, fan=2)]
        for rid in range(layout.num_regions):                    # small-world wiring inside every region
            members = layout.indices(rid)
            parts.append(_within(members, _clustered(len(members), rng, min(6.0, mean_degree))))
        group = np.array([ROLE_ORDER[r] for r in layout.roles])[layout.assignments]
        m = int(n * options.get("group_degree", 2.0))            # shortcuts inside a functional super-group
        src, dst = rng.integers(0, n, size=m), rng.integers(0, n, size=m)
        same = (group[src] == group[dst]) & (layout.assignments[src] != layout.assignments[dst])
        parts.append((src[same], dst[same]))
        pre, post = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    else:                                                        # hybrid: a different generator per region
        per_region: Dict[str, str] = options.get("per_region", {})
        default = options.get("default", "clustered_small_world")
        parts = []
        for rid, name in enumerate(layout.names):
            members = layout.indices(rid)
            sub_kind = per_region.get(name, default)
            k = len(members)
            if sub_kind == "ring_lattice":
                local = _clustered(k, rng, mean_degree * 0.8, ring_only=True)
            elif sub_kind == "uniform_random":
                local = _random_pairs(k, int(k * mean_degree * 0.8), rng)
            elif sub_kind == "hub_skewed":
                local = _hub_skewed(k, rng, mean_degree * 0.8)
            elif sub_kind == "clustered_small_world":
                local = _clustered(k, rng, mean_degree * 0.8)
            else:
                raise KeyError(f"hybrid topologies cannot use '{sub_kind}' inside a region")
            parts.append(_within(members, local))
        m = int(n * mean_degree * 0.2)
        src, dst = rng.integers(0, n, size=m), rng.integers(0, n, size=m)
        inter = layout.assignments[src] != layout.assignments[dst]
        parts.append((src[inter], dst[inter]))
        pre, post = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    pre, post = _dedupe(np.asarray(pre), np.asarray(post), n)
    if kind == "hub_capped":
        pre, post = _cap_in_degree(pre, post, int(options.get("in_degree_cap", max(6, round(1.5 * mean_degree)))))
    m_off = layout.motor_offset
    if canonical_pathway:
        motors = layout.regions_with_role("motor")
        if not motors or layout.sizes[motors[0]] < 3 or m_off <= 2:
            raise ValueError("the canonical sensorimotor pathway needs a motor region of at least 3 neurons that does not "
                             "contain neurons 0-2; use canonical_pathway=False for this layout")
        if n < 6 or m_off + 3 > n:
            raise ValueError("population too small for the canonical sensorimotor pathway")
        canon_pre, canon_post = np.array([0, 1, 2], dtype=np.int64), np.array([m_off, m_off + 1, m_off + 2], dtype=np.int64)
        clash = np.isin(pre * n + post, canon_pre * n + canon_post)
        pre, post = pre[~clash], post[~clash]
    inter = layout.assignments[pre] != layout.assignments[post]
    magnitude = np.where(inter, rng.uniform(0.02, 0.15, size=len(pre)), rng.uniform(0.05, 0.25, size=len(pre)))
    weights = (layout.signs()[pre] * magnitude).astype(np.float32)
    if canonical_pathway:
        pre, post = np.concatenate([canon_pre, pre]), np.concatenate([canon_post, post])
        weights = np.concatenate([np.full(3, 0.80, dtype=np.float32), weights])
    topo = TopologyState(total_neurons=n, num_regions=layout.num_regions, region_names=layout.names,
                         region_assignments=layout.assignments.copy(), base_pre=pre.astype(np.int64),
                         base_post=post.astype(np.int64), base_w=weights.astype(np.float32), seed=seed)
    topo.layout, topo.kind, topo.canonical_pathway = layout, kind, bool(canonical_pathway)
    if kind == "degree_shuffled":
        topo = degree_preserving_shuffle(topo, seed=seed + 1, swaps_per_edge=int(options.get("swaps_per_edge", 8)),
                                         protect_first=3 if canonical_pathway else 0)
    return topo


def edge_class_labels(topology: TopologyState, layout: Optional[RegionLayout] = None) -> np.ndarray:
    """Label every edge: local_recurrent, feedforward, feedback (incl. predictive feedback) or lateral."""
    layout = layout or getattr(topology, "layout", None) or legacy_layout(topology.total_neurons)
    order = np.array([ROLE_ORDER[r] for r in layout.roles])
    a, b = layout.assignments[topology.base_pre], layout.assignments[topology.base_post]
    labels = np.full(len(a), "lateral", dtype=object)
    labels[a == b] = "local_recurrent"
    labels[(a != b) & (order[a] < order[b])] = "feedforward"
    labels[(a != b) & (order[a] > order[b])] = "feedback"
    return labels


def edge_class_counts(topology: TopologyState, layout: Optional[RegionLayout] = None) -> Dict[str, int]:
    labels, counts = np.unique(edge_class_labels(topology, layout), return_counts=True)
    out = {str(k): int(v) for k, v in zip(labels, counts)}
    out["long_range"] = int(sum(v for k, v in out.items() if k != "local_recurrent"))
    return out


__all__ = ["TOPOLOGY_KINDS", "build_topology", "degree_preserving_shuffle", "edge_class_labels", "edge_class_counts"]
