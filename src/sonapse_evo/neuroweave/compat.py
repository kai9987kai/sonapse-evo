"""Compatibility manifest and regression baseline for the pre-existing SONAPSE-EVO 2.0 surface.

``build_compatibility_manifest``  introspects every existing module (classes, methods, functions, dataclass
                                  fields with their signatures), the CLI sub-commands and options, the HTTP
                                  endpoints, the dashboard's DOM ids and JavaScript functions, structural
                                  operator names, lesion types, evidence tiers and graft types.
``check_compatibility``           compares the live code with a recorded manifest.  Anything recorded must
                                  still exist with a compatible signature: additions are allowed, removals
                                  and incompatible changes are violations.
``golden_baseline``               deterministic numeric fingerprints of the original scalar pathway.
``verify_legacy_identity``        the all-mechanisms-off regression check (bit-identical to the original).

The recorded manifest and golden values live in ``tests/weave_baseline/`` and were generated from the
untouched 2.0.0 code (git tag ``baseline/sonapse-evo-2.0.0``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import pkgutil
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_PACKAGE = "sonapse_evo"
_EXCLUDE_PREFIX = (f"{_PACKAGE}.neuroweave",)
_EXCLUDE_EXACT = (f"{_PACKAGE}.observatory.weave_state",)


def _signature(obj: Any) -> Optional[List[List[str]]]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    out = []
    for p in sig.parameters.values():
        default = "<required>" if p.default is inspect.Parameter.empty else repr(p.default)
        out.append([p.name, p.kind.name, default])
    return out


def _legacy_modules() -> List[str]:
    package = importlib.import_module(_PACKAGE)
    names = [_PACKAGE]
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{_PACKAGE}."):
        if info.name.startswith(_EXCLUDE_PREFIX) or info.name in _EXCLUDE_EXACT:
            continue
        names.append(info.name)
    return sorted(names)


def build_compatibility_manifest() -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    for name in _legacy_modules():
        module = importlib.import_module(name)
        classes, functions = {}, {}
        for attr, obj in vars(module).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != name:
                continue
            if inspect.isclass(obj):
                methods = {m: _signature(f) for m, f in vars(obj).items()
                           if (not m.startswith("_") or m == "__init__") and callable(f) and _signature(f) is not None}
                fields = [f.name for f in dataclasses.fields(obj)] if dataclasses.is_dataclass(obj) else []
                classes[attr] = {"methods": methods, "fields": fields, "bases": [b.__name__ for b in obj.__bases__]}
            elif inspect.isfunction(obj):
                functions[attr] = _signature(obj)
        modules[name] = {"classes": classes, "functions": functions, "all": sorted(getattr(module, "__all__", []))}

    root = Path(importlib.import_module(_PACKAGE).__file__).resolve().parent
    cli_src = (root / "cli.py").read_text(encoding="utf-8")
    subcommands: Dict[str, List[str]] = {}
    current = None
    for line in cli_src.splitlines():
        m = re.search(r'add_parser\("([^"]+)"', line)
        if m:
            current = m.group(1)
            subcommands.setdefault(current, [])
            var = re.match(r"\s*(\w+)\s*=", line)
            holder = var.group(1) if var else None
            for other in cli_src.splitlines():
                if holder and re.match(rf"\s*{holder}\.add_argument\(", other):
                    opt = re.search(r'add_argument\("(--[^"]+)"', other)
                    if opt and opt.group(1) not in subcommands[current]:
                        subcommands[current].append(opt.group(1))
    server_src = (root / "observatory" / "server.py").read_text(encoding="utf-8")
    html_src = (root / "observatory" / "web" / "index.html").read_text(encoding="utf-8")
    lesion_src = (root / "substrate" / "lesions.py").read_text(encoding="utf-8")
    from sonapse_evo.contracts.evidence import EvidenceTier
    from sonapse_evo.evolution.operators import ALL_OPERATORS

    manifest = {
        "package": _PACKAGE, "package_version": importlib.import_module(_PACKAGE).__version__,
        "modules": modules,
        "cli": {"subcommands": subcommands},
        "http_endpoints": sorted(set(re.findall(r'self\.path == "(/api/[^"]+)"', server_src))),
        "dom_ids": sorted(set(re.findall(r'\bid="([^"]+)"', html_src))),
        "js_functions": sorted(set(re.findall(r"function\s+([A-Za-z_]\w*)\s*\(", html_src))),
        "js_fetch_paths": sorted(set(re.findall(r"fetch\(\s*['\"]([^'\"]+)['\"]", html_src))),
        "structural_operators": [op.operator_name for op in ALL_OPERATORS],
        "lesion_types": sorted(set(re.findall(r'lesion_type == "(\w+)"', lesion_src))),
        "evidence_tiers": [t.value for t in EvidenceTier],
        "graft_types": ["low_rank", "recurrent", "biological", "donor_adapter"],
        "control_arms": [f"A{i}" for i in range(10)],
        "config_files": ["configs/smoke.yaml", "configs/lesion_repair.yaml", "configs/confirmatory_study.yaml"],
    }
    manifest["digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return manifest


def _compatible(old: Optional[List[List[str]]], new: Optional[List[List[str]]]) -> bool:
    """New signature must keep every old parameter (name, kind, default, order); extras must be optional."""
    if old is None:
        return True
    if new is None or len(new) < len(old):
        return False
    for a, b in zip(old, new):
        if a != b:
            return False
    return all(extra[2] != "<required>" or extra[1] in ("VAR_POSITIONAL", "VAR_KEYWORD") for extra in new[len(old):])


def check_compatibility(baseline: Dict[str, Any]) -> List[str]:
    """Violations of the recorded surface (an empty list means fully backward compatible)."""
    current = build_compatibility_manifest()
    violations: List[str] = []
    for mod, spec in baseline["modules"].items():
        cur = current["modules"].get(mod)
        if cur is None:
            violations.append(f"module removed: {mod}")
            continue
        for name, sig in spec["functions"].items():
            if name not in cur["functions"]:
                violations.append(f"function removed: {mod}.{name}")
            elif not _compatible(sig, cur["functions"][name]):
                violations.append(f"function signature changed: {mod}.{name}")
        for cls, cspec in spec["classes"].items():
            ccur = cur["classes"].get(cls)
            if ccur is None:
                violations.append(f"class removed: {mod}.{cls}")
                continue
            for method, sig in cspec["methods"].items():
                if method not in ccur["methods"]:
                    violations.append(f"method removed: {mod}.{cls}.{method}")
                elif not _compatible(sig, ccur["methods"][method]):
                    violations.append(f"method signature changed: {mod}.{cls}.{method}")
            missing = [f for f in cspec["fields"] if f not in ccur["fields"]]
            if missing:
                violations.append(f"dataclass fields removed from {mod}.{cls}: {missing}")
        gone = [a for a in spec["all"] if a not in cur["all"]]
        if gone:
            violations.append(f"names removed from {mod}.__all__: {gone}")
    for cmd, opts in baseline["cli"]["subcommands"].items():
        cur_opts = current["cli"]["subcommands"].get(cmd)
        if cur_opts is None:
            violations.append(f"CLI sub-command removed: {cmd}")
        else:
            violations += [f"CLI option removed: {cmd} {o}" for o in opts if o not in cur_opts]
    for key in ("http_endpoints", "dom_ids", "js_functions", "js_fetch_paths", "structural_operators", "lesion_types",
                "evidence_tiers", "graft_types", "control_arms"):
        violations += [f"{key}: '{item}' removed" for item in baseline.get(key, []) if item not in current.get(key, [])]
    root = Path(importlib.import_module(_PACKAGE).__file__).resolve().parents[2]
    violations += [f"config file removed: {f}" for f in baseline.get("config_files", []) if not (root / f).exists()]
    return violations


def _trace_digest(substrate, steps: int = 25, seed: int = 0) -> Dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    h = hashlib.sha256()
    means = []
    for _ in range(steps):
        obs = torch.rand(substrate.sensory_dim, generator=gen)
        aux = (torch.rand(substrate.total_neurons, generator=gen) - 0.5) * 0.3
        r = substrate.forward_step(obs, aux_current=aux)
        h.update(r.detach().numpy().tobytes())
        means.append(round(float(r.mean()), 8))
    return {"digest": h.hexdigest(), "mean_rates": means}


def golden_baseline() -> Dict[str, Any]:
    """Deterministic fingerprints of the ORIGINAL classes (never the weave subclass)."""
    import torch.nn as nn

    from sonapse_evo.evaluation.benchmark_runner import DelayedCueMatchingTask, ForageNavigationTask, run_benchmark_episode
    from sonapse_evo.substrate.lesions import LesionEngine, LesionSpec
    from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
    from sonapse_evo.substrate.topology_state import TopologyState

    topo = TopologyState(total_neurons=768, num_regions=3, seed=42)
    out: Dict[str, Any] = {"topology_digest": topo.compute_digest(), "edges": int(len(topo.base_w))}
    out["trace"] = _trace_digest(RecurrentSubstrate(topology=topo, seed=42))
    engine = LesionEngine(seed=42)
    out["lesion_masks"] = {
        kind: hashlib.sha256(engine.create_lesion_mask(topo, LesionSpec(f"g_{kind}", kind, 0.25, 42)).tobytes()).hexdigest()
        for kind in ("random_edges", "inter_region_bridge", "sensory_pathway", "recurrent_memory", "targeted_hub")}
    motor = nn.Linear(64, 5, bias=False)
    with torch.no_grad():
        motor.weight.zero_()
        motor.weight[0, 0] = 2.0
        motor.weight[1, 1] = 2.0
        motor.weight[1, 2] = -2.0
    env = ForageNavigationTask(sensory_dim=155, max_steps=40, seed=42)
    out["forage_returns"] = {str(s): run_benchmark_episode(env, RecurrentSubstrate(topology=topo, seed=42), motor, seed=s)
                             for s in (100, 237, 374)}
    cue = DelayedCueMatchingTask(sensory_dim=155, delay_steps=10, seed=42)
    out["delayed_cue_return"] = run_benchmark_episode(cue, RecurrentSubstrate(topology=topo, seed=42), motor, seed=7)
    sub = RecurrentSubstrate(topology=topo, seed=42)
    sub.forward_step(torch.full((155,), 0.25))
    snap = sub.create_snapshot()
    out["snapshot_state_digest"] = snap.state_digest
    return out


def verify_legacy_identity(steps: int = 20, seed: int = 0) -> Dict[str, Any]:
    """All mechanisms disabled: the weave substrate must reproduce the original bit for bit."""
    from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
    from sonapse_evo.substrate.topology_state import TopologyState

    from .runtime import build_weave_substrate

    topo = TopologyState(total_neurons=768, num_regions=3, seed=42)
    legacy = RecurrentSubstrate(topology=topo, seed=42)
    weave = build_weave_substrate(seed=42)
    neutral = build_weave_substrate(seed=42)
    neutral.node_mask = torch.ones(768)                     # forces the extended path with every field neutral
    gen = torch.Generator().manual_seed(seed)
    identical, neutral_identical = True, True
    for _ in range(steps):
        obs = torch.rand(155, generator=gen)
        aux = (torch.rand(768, generator=gen) - 0.5) * 0.3
        ref = legacy.forward_step(obs, aux_current=aux)
        identical &= bool(torch.equal(ref, weave.forward_step(obs, aux_current=aux)))
        neutral_identical &= bool(torch.equal(ref, neutral.forward_step(obs, aux_current=aux)))
    return {"legacy_identity": bool(identical), "neutral_extended_path_identity": bool(neutral_identical), "steps": int(steps)}


__all__ = ["build_compatibility_manifest", "check_compatibility", "golden_baseline", "verify_legacy_identity"]
