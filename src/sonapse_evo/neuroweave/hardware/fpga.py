"""FPGA mapping manifest export (a mapping description only).

The manifest lists signed-int8 synapses, quantisation parameters, Q8.8 neuron-state hints and a
four-stage systolic-style tile schedule, together with endpoint and sign validation results.

It is **not** RTL, a testbench, a vendor project, a bitstream, or a measured hardware result.  Resource
figures are arithmetic estimates.  The manifest records these non-claims explicitly.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

import torch

from ..contracts import WEAVE_VERSION
from .quantization import quantise_int8, to_q8_8

FPGA_FORMAT = "SONAPSE_FPGA_MAPPING"
FPGA_FORMAT_VERSION = "0.1.0"
_STAGES = ("fetch_presynaptic_state", "multiply_int8_weight", "accumulate_tile", "activate_and_write_back")


def build_fpga_manifest(substrate, tile_neurons: int = 64, include_synapses: bool = True) -> Dict[str, Any]:
    n = int(substrate.total_neurons)
    if hasattr(substrate, "edge_index"):
        pre, post = substrate.edge_index()
        w = substrate.effective_weights().detach()
    else:
        pre, post = substrate.base_pre, substrate.base_post
        w = (substrate.base_w * substrate.lesion_mask).detach()
    live = w != 0
    pre, post, w = pre[live], post[live], w[live]
    scale = float(w.abs().max()) if w.numel() else 1.0
    wq, codes = quantise_int8(w, scale)
    tile_of = post // tile_neurons
    n_tiles = (n + tile_neurons - 1) // tile_neurons
    per_tile = torch.bincount(tile_of, minlength=n_tiles).tolist()
    in_range = bool(((pre >= 0) & (pre < n) & (post >= 0) & (post < n)).all()) if w.numel() else True
    self_loops = int((pre == post).sum())
    sign_ok = int((torch.sign(codes.to(torch.float32)) == torch.sign(w)).sum()) if w.numel() else 0
    zeroed = int((codes == 0).sum())
    src_pos = torch.zeros(n).scatter_add_(0, pre, (w > 0).to(torch.float32))
    src_neg = torch.zeros(n).scatter_add_(0, pre, (w < 0).to(torch.float32))
    mixed_sources = int(((src_pos > 0) & (src_neg > 0)).sum())
    state_q = to_q8_8(substrate.r.detach())
    manifest: Dict[str, Any] = {
        "format": FPGA_FORMAT, "formatVersion": FPGA_FORMAT_VERSION, "projectVersion": WEAVE_VERSION,
        "hardwareProfile": "systolic-style tile array (conceptual)",
        "neurons": n, "synapses": int(w.numel()),
        "quantisation": {"weight_format": "signed int8 magnitude code", "levels": 127, "scale": scale,
                         "max_abs_error": float((wq - w).abs().max()) if w.numel() else 0.0,
                         "mean_abs_error": float((wq - w).abs().mean()) if w.numel() else 0.0,
                         "weights_quantised_to_zero": zeroed},
        "neuronStateHints": {"format": "Q8.8 signed fixed point", "bits": 16, "fractional_bits": 8,
                             "rate_q8_8_min": int(state_q.min()), "rate_q8_8_max": int(state_q.max()),
                             "leak_q8_8": int(round(float(substrate.leak) * 256)), "microsteps": int(substrate.microsteps)},
        "tileSchedule": {"tile_neurons": int(tile_neurons), "tiles": n_tiles, "stages": list(_STAGES),
                         "edges_per_tile": per_tile, "max_edges_per_tile": max(per_tile) if per_tile else 0,
                         "cycles_per_step_estimate": int(sum(per_tile) + len(_STAGES) * n_tiles)},
        "validation": {"endpoints_in_range": in_range, "self_loops": self_loops,
                       "sign_preserved_synapses": sign_ok, "sign_checked_synapses": int(w.numel()) - zeroed,
                       "sign_preserved": bool(sign_ok >= int(w.numel()) - zeroed),
                       "sources_with_mixed_sign": mixed_sources},
        "resourceEstimates": {"synapse_memory_bytes": int(w.numel()) * 5, "state_memory_bytes": n * 2 * 2,
                              "note": "arithmetic estimates, not synthesis or measurement results"},
        "nonClaims": ["no RTL generated", "no testbench or verification", "no vendor project or bitstream",
                      "no timing closure", "no measured throughput, latency or power"],
        "status": "unverified mapping manifest",
    }
    if include_synapses:
        manifest["synapseTable"] = {"pre": pre.tolist(), "post": post.tolist(), "weight_int8": codes.tolist(),
                                    "tile": tile_of.tolist()}
    body = json.dumps({k: v for k, v in manifest.items() if k != "integrity"}, sort_keys=True).encode("utf-8")
    manifest["integrity"] = {"algorithm": "sha256", "digest": hashlib.sha256(body).hexdigest()}
    return manifest


def validate_fpga_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of problems (empty when the manifest is internally consistent)."""
    issues: List[str] = []
    if manifest.get("format") != FPGA_FORMAT:
        issues.append("unknown manifest format")
    n = int(manifest.get("neurons", 0))
    table = manifest.get("synapseTable")
    if table is not None:
        lengths = {len(table[k]) for k in ("pre", "post", "weight_int8", "tile")}
        if len(lengths) != 1 or lengths.pop() != int(manifest.get("synapses", -1)):
            issues.append("synapse table length does not match the declared synapse count")
        if any(not (0 <= int(i) < n) for i in table["pre"]) or any(not (0 <= int(i) < n) for i in table["post"]):
            issues.append("synapse endpoint out of range")
        if any(not (-127 <= int(c) <= 127) for c in table["weight_int8"]):
            issues.append("weight code outside the signed int8 magnitude range")
    if sum(manifest.get("tileSchedule", {}).get("edges_per_tile", [])) != int(manifest.get("synapses", 0)):
        issues.append("tile schedule does not cover every synapse exactly once")
    stored = manifest.get("integrity", {}).get("digest")
    body = json.dumps({k: v for k, v in manifest.items() if k != "integrity"}, sort_keys=True).encode("utf-8")
    if stored != hashlib.sha256(body).hexdigest():
        issues.append("integrity digest mismatch")
    if not manifest.get("nonClaims"):
        issues.append("manifest must state its non-claims")
    return issues
