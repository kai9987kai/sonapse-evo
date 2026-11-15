"""Tiled connectome traversal with synchronous state buffers and sparse (event-driven) execution.

Edges are grouped by destination tile (``tile_neurons`` consecutive target neurons) and processed tile by
tile.  All per-edge signals are computed from the *previous* rate buffer before any accumulation, so the
update is synchronous exactly as in a double-buffered hardware schedule.  With ``sparse=True`` a tile whose
signals are all exactly zero is skipped (event-driven execution); this is numerically exact.

The result equals the single scatter-add up to floating-point summation order.  Operation and tile counts
are measured; byte figures are estimates and labelled as such.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import torch

from ..contracts import Mechanism, Stage, StepContext


class TiledExecution(Mechanism):
    mechanism_id = "tiled_execution"
    stage = Stage.TRANSMIT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"tile_neurons": 64, "sparse": True}

    def on_attach(self) -> None:
        self.tiles_run = self.tiles_skipped = self.edge_ops = self.calls = 0
        self.seconds = 0.0
        self._edges = -1
        self.substrate.traversal = self.traverse

    def on_detach(self) -> None:
        self.substrate.traversal = None

    def on_topology_change(self) -> None:
        self._edges = -1

    def _plan(self, post: torch.Tensor) -> None:
        tile = int(self.params["tile_neurons"])
        self.order = torch.argsort(post // tile, stable=True)
        counts = torch.bincount(post[self.order] // tile, minlength=(self.substrate.total_neurons + tile - 1) // tile)
        self.bounds = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), counts]), 0).tolist()
        self._edges = int(post.numel())

    def traverse(self, post: torch.Tensor, signal: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        if self._edges != int(post.numel()):
            self._plan(post)
        nxt = torch.zeros_like(like)                       # "next" buffer; ``signal`` came from the "previous" buffer
        sig, dst = signal[self.order], post[self.order]
        for lo, hi in zip(self.bounds[:-1], self.bounds[1:]):
            if hi == lo:
                continue
            chunk = sig[lo:hi]
            if self.params["sparse"] and not bool(chunk.detach().any()):
                self.tiles_skipped += 1
                continue
            nxt.scatter_add_(0, dst[lo:hi], chunk)
            self.tiles_run += 1
            self.edge_ops += hi - lo
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return nxt

    def schedule(self) -> Dict[str, Any]:
        post = self.substrate.edge_index()[1]
        if self._edges != int(post.numel()):
            self._plan(post)
        sizes = [hi - lo for lo, hi in zip(self.bounds[:-1], self.bounds[1:])]
        return {"tile_neurons": int(self.params["tile_neurons"]), "tiles": len(sizes), "edges_per_tile": sizes,
                "max_tile_edges": max(sizes) if sizes else 0}

    def telemetry(self) -> Dict[str, Any]:
        sched = self.schedule()
        return {"tiles": sched["tiles"], "tiles_run": int(self.tiles_run), "tiles_skipped": int(self.tiles_skipped),
                "edge_ops_measured": int(self.edge_ops), "traversal_ms_per_call_measured": 1000.0 * self.seconds / max(1, self.calls),
                "working_set_bytes_estimated": int(sched["max_tile_edges"] * 9 + int(self.params["tile_neurons"]) * 8)}
