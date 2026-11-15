"""Graph metrics for connectome comparison (numpy / scipy.sparse only).

All quantities are computed on the live edges (non-zero weight).  Path-based metrics use breadth-first
distances from a seeded sample of source nodes and report how many sources were used; clustering is exact
up to ``exact_limit`` nodes and node-sampled above it.  These describe graph structure only - they say
nothing about whether a structure helps learning; that is measured separately.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

from ...substrate.topology_state import TopologyState
from .regions import RegionLayout


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def _degree_summary(d: np.ndarray) -> Dict[str, Any]:
    mean = float(d.mean())
    return {"mean": mean, "max": int(d.max()), "min": int(d.min()), "std": float(d.std()),
            "cv": float(d.std() / mean) if mean > 0 else 0.0, "gini": _gini(d),
            "histogram": np.bincount(np.minimum(d, 63), minlength=64).tolist()}


def compute_graph_metrics(topology: TopologyState, layout: Optional[RegionLayout] = None, seed: int = 0,
                          max_sources: int = 128, exact_limit: int = 2000) -> Dict[str, Any]:
    n = int(topology.total_neurons)
    live = np.abs(topology.base_w) > 0
    pre, post, w = topology.base_pre[live], topology.base_post[live], topology.base_w[live]
    e = int(len(pre))
    rng = np.random.default_rng(seed)
    adj = sp.csr_matrix((np.ones(e), (pre, post)), shape=(n, n))
    adj.data[:] = 1.0
    in_deg = np.asarray(adj.sum(axis=0)).ravel().astype(int)
    out_deg = np.asarray(adj.sum(axis=1)).ravel().astype(int)
    reciprocity = float(adj.multiply(adj.T).sum() / max(1, adj.nnz))

    und = ((adj + adj.T) > 0).astype(np.float64).tocsr()
    k = np.asarray(und.sum(axis=1)).ravel()
    rows = np.arange(n) if n <= exact_limit else np.sort(rng.choice(n, size=exact_limit, replace=False))
    triangles = np.asarray((und[rows] @ und).multiply(und[rows]).sum(axis=1)).ravel() / 2.0
    kr = k[rows]
    valid = kr >= 2
    clustering = float(np.mean(2.0 * triangles[valid] / (kr[valid] * (kr[valid] - 1)))) if valid.any() else 0.0

    assignments = np.asarray(layout.assignments if layout is not None else topology.region_assignments)
    regions = np.unique(assignments)
    same = assignments[pre] == assignments[post]
    modularity = 0.0
    for g in regions:
        modularity += float((same & (assignments[pre] == g)).sum()) / max(1, e) \
            - float(out_deg[assignments == g].sum()) * float(in_deg[assignments == g].sum()) / max(1, e) ** 2

    sources = np.arange(n) if n <= max_sources else np.sort(rng.choice(n, size=max_sources, replace=False))
    dist = csgraph.dijkstra(adj, directed=True, indices=sources, unweighted=True)
    mask = np.ones_like(dist, dtype=bool)
    mask[np.arange(len(sources)), sources] = False
    finite = np.isfinite(dist) & mask
    inv = np.zeros_like(dist)
    inv[finite] = 1.0 / dist[finite]
    path_length = float(dist[finite].mean()) if finite.any() else float("inf")
    efficiency = float(inv[mask].mean())

    endpoint = in_deg + out_deg
    top = max(1, int(round(0.05 * n)))
    hub_share = float(np.sort(endpoint)[-top:].sum() / max(1, endpoint.sum()))

    index = {int(g): i for i, g in enumerate(regions)}
    r = len(regions)
    counts = np.zeros((r, r))
    np.add.at(counts, (np.vectorize(index.get)(assignments[pre]), np.vectorize(index.get)(assignments[post])), 1.0)
    sizes = np.array([(assignments == g).sum() for g in regions], dtype=np.float64)
    possible = np.outer(sizes, sizes) - np.diag(sizes)
    density_matrix = counts / np.maximum(possible, 1.0)
    src_region = np.vectorize(index.get)(assignments[sources])
    pair_eff = np.zeros((r, r))
    for a in range(r):
        rows_a = np.nonzero(src_region == a)[0]
        for b in range(r):
            cols_b = np.nonzero(assignments == regions[b])[0]
            if a != b and len(rows_a) and len(cols_b):
                pair_eff[a, b] = float(inv[np.ix_(rows_a, cols_b)].mean())
    off = ~np.eye(r, dtype=bool)
    covered = pair_eff[off][pair_eff[off] > 0]
    regional_efficiency = float(covered.mean()) if covered.size else 0.0

    pos = np.bincount(pre[w > 0], minlength=n)
    neg = np.bincount(pre[w < 0], minlength=n)
    return {
        "neurons": n, "edges": e, "density": e / max(1, n * (n - 1)),
        "in_degree": _degree_summary(in_deg), "out_degree": _degree_summary(out_deg),
        "reciprocity": reciprocity, "clustering_coefficient": clustering,
        "clustering_nodes_used": int(len(rows)), "modularity": float(modularity),
        "characteristic_path_length": path_length, "global_efficiency": efficiency,
        "reachable_fraction": float(finite.mean()), "path_sources_used": int(len(sources)),
        "hub_concentration_top5pct": hub_share,
        "regions": [int(g) for g in regions], "inter_region_density": density_matrix.tolist(),
        "inter_region_edge_fraction": float(1.0 - same.mean()) if e else 0.0,
        "regional_communication_efficiency": regional_efficiency,
        "inhibitory_edge_fraction": float((w < 0).mean()) if e else 0.0,
        "sources_with_mixed_sign": int(((pos > 0) & (neg > 0)).sum()),
        "estimated_cost": {"edge_ops_per_microstep": e * 3, "note": "analytic estimate; see hardware.profiler for measured cost"},
    }


__all__ = ["compute_graph_metrics"]
