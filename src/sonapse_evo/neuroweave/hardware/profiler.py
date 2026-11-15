"""Execution profiling: measured numbers are kept strictly separate from estimates.

``profile_substrate``  measured wall-clock per step, measured tensor bytes, plus clearly labelled estimates.
``scaling_profile``    the same across population sizes (sparse connectivity: edges grow linearly with N).
``compact_state``      memory-efficient storage: float16 copies of the weave state with the measured error.
``available_backends`` optional accelerator interface: reports what this runtime can actually execute.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from ...substrate.topology_state import TopologyState
from ..registry import WeaveConfig
from ..substrate import WeaveSubstrate


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def state_bytes(substrate) -> Dict[str, int]:
    """Measured bytes held in tensors (legacy buffers and, when present, weave state)."""
    legacy = sum(int(b.numel() * b.element_size()) for b in substrate.buffers())
    weave = _tensor_bytes(substrate.weave_state_dict()) if isinstance(substrate, WeaveSubstrate) else 0
    return {"legacy_buffers_bytes": legacy, "weave_state_bytes": weave, "total_bytes": legacy + weave}


def profile_substrate(substrate, steps: int = 40, seed: int = 0, warmup: int = 5) -> Dict[str, Any]:
    """Time ``forward_step`` on seeded random observations.  Restores the rate state afterwards."""
    gen = torch.Generator().manual_seed(int(seed))
    obs = [torch.rand(substrate.sensory_dim, generator=gen) * 0.5 for _ in range(steps + warmup)]
    snapshot = substrate.create_snapshot()
    with torch.no_grad():
        for o in obs[:warmup]:
            substrate.forward_step(o)
        start = time.perf_counter()
        for o in obs[warmup:]:
            substrate.forward_step(o)
        elapsed = time.perf_counter() - start
    substrate.restore_snapshot(snapshot)
    is_weave = isinstance(substrate, WeaveSubstrate)
    edges = int(substrate.syn.n_edges) if is_weave else int(len(substrate.base_w))
    n = int(substrate.total_neurons)
    return {
        "measured": {"ms_per_step": 1000.0 * elapsed / steps, "steps_per_second": steps / elapsed if elapsed > 0 else float("inf"),
                     "steps": int(steps), **state_bytes(substrate)},
        "estimated": {"edge_ops_per_step": edges * 3 * int(substrate.microsteps), "neuron_ops_per_step": n * 42 * int(substrate.microsteps),
                      "note": "operation counts are analytic estimates, not instrumented counts"},
        "config": {"neurons": n, "edges": edges, "microsteps": int(substrate.microsteps),
                   "mechanisms": list(substrate.weave.enabled_ids()) if is_weave else [], "device": str(substrate.r.device),
                   "torch_threads": torch.get_num_threads()},
    }


def scaling_profile(sizes: Iterable[int] = (768, 1536, 3072), profile: str = "legacy", steps: int = 20, seed: int = 42,
                    enable: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Measured cost versus population size (default sparse topology, mean degree 8)."""
    out = []
    for n in sizes:
        if n < 576:
            raise ValueError("the default topology generator needs at least 576 neurons; use connectome.build_topology for smaller graphs")
        weave = WeaveConfig.from_profile(profile, seed=seed)
        enable = tuple(enable)
        if enable:
            weave.enable(*enable)
        sub = WeaveSubstrate(topology=TopologyState(total_neurons=int(n), num_regions=3, seed=seed), seed=seed, weave=weave)
        out.append(profile_substrate(sub, steps=steps, seed=seed))
    return out


def compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Half-precision copy of a weave state dict.  Returns the copy plus measured size and error."""
    max_err, before, after = 0.0, 0, 0

    def walk(obj):
        nonlocal max_err, before, after
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            half = obj.to(torch.float16)
            before += obj.numel() * obj.element_size()
            after += half.numel() * half.element_size()
            if obj.numel():
                max_err = max(max_err, float((half.to(torch.float32) - obj.to(torch.float32)).abs().max()))
            return half
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    compact = walk(state)
    return {"state": compact, "measured": {"float_bytes_before": int(before), "float_bytes_after": int(after),
                                           "max_abs_error": max_err}}


def available_backends() -> Dict[str, Dict[str, Any]]:
    """Optional accelerator interface: what can actually run here (nothing is assumed or simulated)."""
    return {
        "cpu": {"available": True, "legacy_path": True, "weave_path": True},
        "cuda": {"available": bool(torch.cuda.is_available()), "legacy_path": bool(torch.cuda.is_available()), "weave_path": False,
                 "note": "weave compartment state is CPU-resident in this release; no GPU speed-up is claimed"},
        "fpga": {"available": False, "legacy_path": False, "weave_path": False,
                 "note": "only a mapping manifest can be exported (see hardware.fpga)"},
    }
