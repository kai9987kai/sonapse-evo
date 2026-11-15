"""Versioned model package: export, read-only inspection, validation and import.

``export_model_package``   everything needed to rebuild a model: connectome topology, neuron populations,
                           substrate buffers, all compartment / synapse / signal / mechanism state, sensory
                           manifest, agent read-out, optional environment state, training history, experiment
                           metadata, provenance and an integrity digest.
``validate_model_package`` never raises and never touches a live model: returns issues (blocking) and warnings.
``inspect_model_package``  read-only summary.
``import_model_package``   validates, then builds a NEW substrate.  It cannot overwrite an active model
                           because it never receives one.

Compatibility: format major versions 2 (legacy substrate-only packages, see ``export_legacy_package``) and 3
are accepted.  Unknown mechanism ids in a newer package are ignored with a warning.  The digest is SHA-256
over the canonical JSON; it detects modification, it is not a signature and proves nothing about authorship.
"""
from __future__ import annotations

import enum
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...substrate.recurrent_runtime import RecurrentSubstrate
from ...substrate.topology_state import TopologyState
from ..contracts import INTERFACE_VERSIONS, WEAVE_VERSION
from ..registry import MECHANISMS, WeaveConfig
from ..substrate import MAX_DELAY, WeaveSubstrate

FORMAT = "SONAPSE_MODEL_PACKAGE"
FORMAT_VERSION = "3.0.0"
SUPPORTED_MAJOR = (2, 3)
LIMITS = {"max_bytes": 64 * 1024 * 1024, "max_neurons": 65536, "max_synapses": 2_000_000}
_DTYPES = {"f32": torch.float32, "f64": torch.float64, "i64": torch.int64, "i16": torch.int16, "i8": torch.int8, "bool": torch.bool}
_NAMES = {v: k for k, v in _DTYPES.items()}


# ------------------------------------------------------------------ encoding
def to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors, arrays, enums and numpy scalars into plain JSON values."""
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        name = _NAMES.get(t.dtype)
        if name is None:
            t, name = (t.to(torch.float32), "f32") if t.is_floating_point() else (t.to(torch.int64), "i64")
        return {"__t__": name, "shape": list(t.shape), "data": t.reshape(-1).tolist()}
    if isinstance(obj, np.ndarray):
        return to_jsonable(torch.as_tensor(obj))
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def from_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__t__" in obj and "data" in obj:
            return torch.tensor(obj["data"], dtype=_DTYPES[obj["__t__"]]).reshape(obj.get("shape", [-1]))
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj


def _digest(package: Dict[str, Any]) -> str:
    body = {k: v for k, v in package.items() if k != "integrity"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _arr(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "__t__" in value:
        return np.asarray(value["data"]).reshape(value.get("shape", [-1]))
    return np.asarray(value)


# -------------------------------------------------------------------- export
def export_model_package(substrate: RecurrentSubstrate, motor_head: Optional[nn.Module] = None, environment: Any = None,
                         training_history: Optional[Dict[str, Any]] = None, experiment: Optional[Dict[str, Any]] = None,
                         notes: str = "") -> Dict[str, Any]:
    is_weave = isinstance(substrate, WeaveSubstrate)
    topo = substrate.topology
    layout = getattr(substrate, "layout", None) or getattr(topo, "layout", None)
    rows, cols = torch.nonzero(substrate.B.detach(), as_tuple=True)
    weave_state = substrate.weave_state_dict() if is_weave else None
    mech = weave_state["mechanisms"] if weave_state else {}
    hub = mech.get("multimodal_sensory", {})
    memory_ids = [m for m in ("associative_trace_memory", "fast_weights", "memory_ecology", "predictive_replay") if m in mech]
    env_state = None
    if environment is not None and hasattr(environment, "state_dict"):
        env_state = {"name": getattr(getattr(environment, "config", None), "name", type(environment).__name__),
                     "state": environment.state_dict()}
    package: Dict[str, Any] = {
        "format": FORMAT, "formatVersion": FORMAT_VERSION, "weaveVersion": WEAVE_VERSION, "interfaces": dict(INTERFACE_VERSIONS),
        "createdUnix": time.time(),
        "capabilities": {
            "frozen_inference": True,
            "continued_learning": bool(is_weave),
            "full_simulation": env_state is not None,
            "live_sensory_operation": False,
            "exported_memory": bool(memory_ids),
            "persistent_model_state": True,
            "notes": {"live_sensory_operation": "camera / microphone access is a property of the running interface, not of a package; "
                                                "the sensory manifest records channel modes only and no raw media",
                      "continued_learning": "requires importing into the Python runtime; the standalone HTML runner is frozen",
                      "full_simulation": "the environment state is included only when an environment was passed to the exporter"},
        },
        "topology": {"total_neurons": int(topo.total_neurons), "num_regions": int(topo.num_regions), "region_names": list(topo.region_names),
                     "region_assignments": torch.as_tensor(np.asarray(topo.region_assignments)), "seed": int(topo.seed),
                     "kind": getattr(topo, "kind", "legacy"), "layout": layout.to_dict() if layout is not None else None,
                     "digest": topo.compute_digest()},
        "neuronPopulations": layout.population_summary() if layout is not None else None,
        "substrate": {"class": type(substrate).__name__, "sensory_dim": int(substrate.sensory_dim), "leak": float(substrate.leak),
                      "microsteps": int(substrate.microsteps), "current_cap": float(substrate.current_cap), "seed": int(substrate.seed),
                      "step_count": int(substrate.step_count), "motor_offset": int(getattr(substrate, "motor_offset", 512)),
                      "edges": {"pre": substrate.base_pre, "post": substrate.base_post, "w": substrate.base_w},
                      "plastic": {"pre": substrate.plastic_pre, "post": substrate.plastic_post, "w": substrate.plastic_w},
                      "lesion_mask": substrate.lesion_mask, "bias": substrate.bias, "r": substrate.r,
                      "B": {"rows": rows, "cols": cols, "values": substrate.B.detach()[rows, cols]},
                      "active_grafts": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, bool))}
                                        for k, v in substrate.active_grafts.items()}},
        "weave": weave_state,
        "sensory": hub.get("manifest") if hub else None,
        "memory": {"mechanisms": memory_ids} if memory_ids else None,
        "agent": None if motor_head is None else {"motor_head": motor_head.weight.detach(), "readout_width": int(motor_head.weight.shape[1]),
                                                  "actions": int(motor_head.weight.shape[0])},
        "environment": env_state,
        "training": training_history or {},
        "experiment": experiment or {},
        "provenance": {"synthetic": True, "notes": notes,
                       "disclosure": "synthetic connectome and synthetic tasks; no biological data, no external checkpoints"},
        "counts": {"neurons": int(substrate.total_neurons), "base_synapses": int(len(substrate.base_w)),
                   "grown_synapses": int(substrate.syn.n_grown) if is_weave else 0,
                   "mechanisms_enabled": len(substrate.weave.enabled_ids()) if is_weave else 0},
    }
    package = to_jsonable(package)
    package["integrity"] = {"algorithm": "sha256", "digest": _digest(package)}
    return package


def export_legacy_package(substrate: RecurrentSubstrate) -> Dict[str, Any]:
    """The minimal format-2 package (substrate only), kept importable for backward compatibility."""
    topo = substrate.topology
    rows, cols = torch.nonzero(substrate.B.detach(), as_tuple=True)
    return to_jsonable({
        "format": FORMAT, "formatVersion": "2.0.0",
        "topology": {"total_neurons": int(topo.total_neurons), "num_regions": int(topo.num_regions), "region_names": list(topo.region_names),
                     "region_assignments": torch.as_tensor(np.asarray(topo.region_assignments)), "seed": int(topo.seed)},
        "substrate": {"sensory_dim": int(substrate.sensory_dim), "leak": float(substrate.leak), "microsteps": int(substrate.microsteps),
                      "current_cap": float(substrate.current_cap), "seed": int(substrate.seed), "step_count": int(substrate.step_count),
                      "edges": {"pre": substrate.base_pre, "post": substrate.base_post, "w": substrate.base_w},
                      "lesion_mask": substrate.lesion_mask, "bias": substrate.bias, "r": substrate.r,
                      "B": {"rows": rows, "cols": cols, "values": substrate.B.detach()[rows, cols]}},
    })


# ---------------------------------------------------------------- validation
def validate_model_package(package: Union[Dict[str, Any], str, bytes]) -> Dict[str, Any]:
    """Read-only validation.  Returns ``{"ok", "issues", "warnings", "summary"}`` and never raises."""
    issues: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}
    try:
        if isinstance(package, (str, bytes)):
            size = len(package.encode("utf-8")) if isinstance(package, str) else len(package)
            summary["bytes"] = size
            if size > LIMITS["max_bytes"]:
                return {"ok": False, "issues": [f"package is {size} bytes; limit is {LIMITS['max_bytes']}"], "warnings": [], "summary": summary}
            package = json.loads(package)
        if not isinstance(package, dict):
            return {"ok": False, "issues": ["package is not a JSON object"], "warnings": [], "summary": summary}
        if package.get("format") != FORMAT:
            issues.append(f"unknown format '{package.get('format')}'")
        version = str(package.get("formatVersion", ""))
        try:
            major = int(version.split(".")[0])
        except ValueError:
            major = -1
        if major not in SUPPORTED_MAJOR:
            issues.append(f"unsupported format version '{version}' (supported major versions: {SUPPORTED_MAJOR})")
        summary["formatVersion"] = version
        topo, sub = package.get("topology"), package.get("substrate")
        if not isinstance(topo, dict) or not isinstance(sub, dict):
            issues.append("missing 'topology' or 'substrate' section")
            return {"ok": False, "issues": issues, "warnings": warnings, "summary": summary}

        def number(section: Dict[str, Any], where: str, key: str, lo: float, hi: float, integer: bool = False,
                   required: bool = True) -> Optional[float]:
            value = section.get(key)
            if value is None and not required:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) \
                    or (integer and float(value) != int(value)):
                issues.append(f"{where}.{key} missing or not a finite {'integer' if integer else 'number'}")
                return None
            if not (lo <= value <= hi):
                issues.append(f"{where}.{key}={value} outside [{lo}, {hi}]")
                return None
            return value

        def vector(value: Any, where: str, length: Optional[int], integer: bool = False, lo: float = -math.inf,
                   hi: float = math.inf, ndim: int = 1) -> Optional[np.ndarray]:
            if isinstance(value, dict) and "__t__" in value:
                data, shape = value.get("data"), value.get("shape", [-1])
                if not isinstance(data, list) or not isinstance(shape, list) or not all(isinstance(s, int) and s >= 0 for s in shape) \
                        or int(np.prod(shape)) != len(data):
                    issues.append(f"{where}: declared shape does not match its data")
                    return None
            elif not isinstance(value, list):
                issues.append(f"{where} missing or not an array")
                return None
            try:
                arr = np.asarray(_arr(value), dtype=np.float64)
            except (TypeError, ValueError):
                issues.append(f"{where} contains non-numeric values")
                return None
            if arr.ndim != ndim or (length is not None and arr.shape[0] != length):
                issues.append(f"{where} has shape {tuple(arr.shape)}, expected {ndim}-D with leading size {length}")
                return None
            if arr.size and not np.all(np.isfinite(arr)):
                issues.append(f"{where} contains non-finite values")
                return None
            if integer and arr.size and not np.all(arr == np.round(arr)):
                issues.append(f"{where} must contain integers")
                return None
            if arr.size and (arr.min() < lo or arr.max() > hi):
                issues.append(f"{where} outside [{lo}, {hi}]")
                return None
            return arr

        n_value = number(topo, "topology", "total_neurons", 1, LIMITS["max_neurons"], integer=True)
        if n_value is None:
            return {"ok": False, "issues": issues, "warnings": warnings, "summary": summary}
        n = int(n_value)
        summary["neurons"] = n
        regions = number(topo, "topology", "num_regions", 1, n, integer=True)
        names = topo.get("region_names")
        if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
            issues.append("topology.region_names must be a non-empty list of strings")
        elif regions is not None and len(names) < int(regions):
            issues.append("topology.region_names has fewer names than num_regions")
        vector(topo.get("region_assignments"), "topology.region_assignments", n, integer=True, lo=0,
               hi=(int(regions) - 1) if regions is not None else math.inf)
        sensory_dim = number(sub, "substrate", "sensory_dim", 1, 100_000, integer=True)
        number(sub, "substrate", "microsteps", 1, 64, integer=True)
        number(sub, "substrate", "current_cap", 0.0, 100.0)
        number(sub, "substrate", "leak", 1e-9, 1.0)
        number(sub, "substrate", "step_count", 0, 1e15, integer=True, required=False)
        number(sub, "substrate", "motor_offset", 0, n - 1, integer=True, required=False)
        edges = sub.get("edges") if isinstance(sub.get("edges"), dict) else {}
        w = vector(edges.get("w"), "substrate.edges.w", None, lo=-10.0, hi=10.0)
        m = 0 if w is None else len(w)
        summary["synapses"] = m
        if m > LIMITS["max_synapses"]:
            issues.append(f"synapse count {m} exceeds limit {LIMITS['max_synapses']}")
            return {"ok": False, "issues": issues, "warnings": warnings, "summary": summary}
        pre = vector(edges.get("pre"), "substrate.edges.pre", m, integer=True, lo=0, hi=n - 1)
        post = vector(edges.get("post"), "substrate.edges.post", m, integer=True, lo=0, hi=n - 1)
        if pre is not None and post is not None and int((pre == post).sum()):
            warnings.append(f"{int((pre == post).sum())} self-loops present")
        vector(sub.get("bias"), "substrate.bias", n, lo=-10.0, hi=10.0)
        vector(sub.get("r"), "substrate.r", n, lo=0.0, hi=1.0)
        vector(sub.get("lesion_mask"), "substrate.lesion_mask", m, lo=0.0, hi=1.0)
        b = sub.get("B") if isinstance(sub.get("B"), dict) else {}
        b_rows = vector(b.get("rows"), "substrate.B.rows", None, integer=True, lo=0, hi=n - 1)
        k = 0 if b_rows is None else len(b_rows)
        vector(b.get("cols"), "substrate.B.cols", k, integer=True, lo=0, hi=(int(sensory_dim) - 1) if sensory_dim else math.inf)
        vector(b.get("values"), "substrate.B.values", k, lo=-100.0, hi=100.0)
        plastic_len = 0
        plastic = sub.get("plastic")
        if plastic is not None:
            if not isinstance(plastic, dict):
                issues.append("substrate.plastic must be an object")
            else:
                pw = vector(plastic.get("w"), "substrate.plastic.w", None, lo=-10.0, hi=10.0)
                plastic_len = 0 if pw is None else len(pw)
                vector(plastic.get("pre"), "substrate.plastic.pre", plastic_len, integer=True, lo=0, hi=n - 1)
                vector(plastic.get("post"), "substrate.plastic.post", plastic_len, integer=True, lo=0, hi=n - 1)
        agent = package.get("agent")
        if agent is not None:
            head = vector(agent.get("motor_head") if isinstance(agent, dict) else None, "agent.motor_head", None, lo=-100.0, hi=100.0, ndim=2)
            if head is not None and head.shape[1] > n:
                issues.append("agent.motor_head reads more neurons than the model has")
        weave = package.get("weave")
        if weave is not None:
            if not isinstance(weave, dict):
                issues.append("weave section must be an object")
                weave = {}
            config = weave.get("config")
            enabled = config.get("enabled") if isinstance(config, dict) else None
            if not isinstance(enabled, dict):
                issues.append("weave.config.enabled must be an object")
                enabled = {}
            unknown = [key for key in enabled if key not in MECHANISMS]
            if unknown:
                warnings.append(f"unknown mechanisms will be ignored on import: {unknown}")
            if any(not isinstance(v, bool) for v in enabled.values()):
                issues.append("mechanism flags must be booleans")
            if isinstance(config, dict) and not isinstance(config.get("params", {}), dict):
                issues.append("weave.config.params must be an object")
            summary["mechanisms_enabled"] = [key for key, v in enabled.items() if v is True and key in MECHANISMS]
            cell = weave.get("cell") if isinstance(weave.get("cell"), dict) else None
            branch_rows = 2
            if cell is None:
                issues.append("weave.cell missing")
            else:
                for name, value in cell.items():
                    if name == "branches":
                        br = vector(value, "weave.cell.branches", None, lo=-1e6, hi=1e6, ndim=2)
                        if br is not None:
                            branch_rows = br.shape[0]
                            if br.shape[1] != n or not (2 <= br.shape[0] <= 64):
                                issues.append("weave.cell.branches has an incompatible shape")
                    elif isinstance(value, (dict, list)) and name != "params":
                        vector(value, f"weave.cell.{name}", n, lo=-1e6, hi=1e6)
                if not isinstance(cell.get("params", {}), dict):
                    issues.append("weave.cell.params must be an object")
            syn = weave.get("synapses") if isinstance(weave.get("synapses"), dict) else None
            if syn is None:
                issues.append("weave.synapses missing")
            else:
                if syn.get("n_base") != m or syn.get("n_plastic") != plastic_len:
                    issues.append("weave.synapses was saved for a different base or plastic edge count")
                gw = vector(syn.get("grown_w"), "weave.synapses.grown_w", None, lo=-10.0, hi=10.0)
                grown = 0 if gw is None else len(gw)
                summary["grown_synapses"] = grown
                vector(syn.get("grown_pre"), "weave.synapses.grown_pre", grown, integer=True, lo=0, hi=n - 1)
                vector(syn.get("grown_post"), "weave.synapses.grown_post", grown, integer=True, lo=0, hi=n - 1)
                e = m + plastic_len + grown
                bounds = {"delay": (0, MAX_DELAY, True), "branch": (0, branch_rows - 1, True), "age": (0, 1e15, True),
                          "alive": (0, 1, False), "pre_trace": (0, 1.5 + 1e-6, False), "post_trace": (0, 1.5 + 1e-6, False),
                          "routing_gain": (0.62 - 1e-6, 1.14 + 1e-6, False), "consolidation": (0, 1 + 1e-6, False),
                          "lr_scale": (0, 100, False), "w_delta": (-10.0, 10.0, False), "sign": (-1.0, 1.0, False),
                          "myelin": (0.0, 1.0, False), "trace": (-1e6, 1e6, False), "eligibility": (-1e6, 1e6, False),
                          "utility": (-1e6, 1e6, False), "last_signal": (-1e6, 1e6, False)}
                for name, (lo, hi, integer) in bounds.items():
                    vector(syn.get(name), f"weave.synapses.{name}", e, integer=integer, lo=lo, hi=hi)
            mask = weave.get("node_mask")
            if mask is not None:
                vector(mask, "weave.node_mask", n, lo=0.0, hi=1.0)
            history = weave.get("history", [])
            if not isinstance(history, list) or len(history) > MAX_DELAY:
                issues.append("weave.history must be a list of at most 3 rate vectors")
            else:
                for i, h in enumerate(history):
                    vector(h, f"weave.history[{i}]", n, lo=0.0, hi=1.0)
            for section in ("signals", "mechanisms", "feedback"):
                if not isinstance(weave.get(section, {}), dict):
                    issues.append(f"weave.{section} must be an object")
            theirs = weave.get("interfaces", {})
            if isinstance(theirs, dict):
                for key, ours in INTERFACE_VERSIONS.items():
                    if key in theirs and str(theirs[key]).split(".")[0] != ours.split(".")[0]:
                        warnings.append(f"interface '{key}' is version {theirs[key]} in the package, {ours} in this runtime")
        integrity = package.get("integrity")
        if integrity is not None and not isinstance(integrity, dict):
            issues.append("integrity section must be an object")
        elif integrity:
            if integrity.get("digest") != _digest(package):
                issues.append("integrity digest mismatch: the package was modified after export")
            summary["digest"] = integrity.get("digest")
        elif major >= 3:
            warnings.append("package has no integrity digest")
        summary["capabilities"] = package.get("capabilities", {"frozen_inference": True})
    except Exception as exc:                      # malformed input must never crash the caller
        issues.append(f"malformed package: {type(exc).__name__}: {exc}")
    return {"ok": not issues, "issues": issues, "warnings": warnings, "summary": summary}


def inspect_model_package(package: Union[Dict[str, Any], str, bytes]) -> Dict[str, Any]:
    """Read-only description of a package (validation result plus a compact manifest).  Never raises."""
    report = validate_model_package(package)
    if isinstance(package, (str, bytes)):
        try:
            package = json.loads(package)
        except Exception:
            package = {}
    if not isinstance(package, dict):
        package = {}

    def section(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    weave, topo = section(package.get("weave")), section(package.get("topology"))
    enabled = section(section(weave.get("config")).get("enabled"))
    return {"validation": report, "format": package.get("format"), "formatVersion": package.get("formatVersion"),
            "weaveVersion": package.get("weaveVersion"), "counts": package.get("counts") if isinstance(package.get("counts"), dict) else None,
            "capabilities": package.get("capabilities") if isinstance(package.get("capabilities"), dict) else None,
            "topology_kind": topo.get("kind"), "regions": topo.get("region_names"),
            "mechanisms": sorted(str(k) for k, v in enabled.items() if v is True),
            "mechanism_state_sections": sorted(str(k) for k in section(weave.get("mechanisms"))),
            "has_agent": package.get("agent") is not None, "has_environment": package.get("environment") is not None,
            "sensory": package.get("sensory"), "provenance": package.get("provenance"), "imported": False}


# -------------------------------------------------------------------- import
def import_model_package(package: Union[Dict[str, Any], str, bytes], strict: bool = False) -> Tuple[RecurrentSubstrate, Dict[str, Any]]:
    """Validate, then build a NEW substrate from a package.

    Raises ``ValueError`` (and only ``ValueError``) when the package is rejected or cannot be built.  After
    construction one trial step is run on a throw-away clone, so a package that imports is known to run.
    """
    report = validate_model_package(package)
    if not report["ok"]:
        raise ValueError("model package rejected: " + "; ".join(report["issues"]))
    if isinstance(package, (str, bytes)):
        package = json.loads(package)
    try:
        substrate, extras = _build_from_package(package, report, strict)
        probe = substrate.clone_substrate()
        with torch.no_grad():
            rates = probe.forward_step(torch.zeros(probe.sensory_dim))
        if not bool(torch.isfinite(rates).all()):
            raise ValueError("trial step produced non-finite rates")
        head = extras.get("motor_head")
        if head is not None and head.weight.shape[1] > substrate.total_neurons:
            raise ValueError("motor head reads more neurons than the model has")
    except ValueError as exc:
        if str(exc).startswith("model package"):
            raise
        raise ValueError(f"model package could not be imported: {exc}") from exc
    except Exception as exc:                               # never leak an arbitrary exception type to callers
        raise ValueError(f"model package could not be imported: {type(exc).__name__}: {exc}") from exc
    return substrate, extras


def _build_from_package(package: Dict[str, Any], report: Dict[str, Any], strict: bool) -> Tuple[RecurrentSubstrate, Dict[str, Any]]:
    topo_d, sub_d = package["topology"], package["substrate"]
    n = int(topo_d["total_neurons"])
    edges = sub_d["edges"]
    topology = TopologyState(total_neurons=n, num_regions=int(topo_d["num_regions"]), region_names=tuple(topo_d["region_names"]),
                             region_assignments=_arr(topo_d["region_assignments"]).astype(np.int64),
                             base_pre=_arr(edges["pre"]).astype(np.int64), base_post=_arr(edges["post"]).astype(np.int64),
                             base_w=_arr(edges["w"]).astype(np.float32), seed=int(topo_d.get("seed", 42)))
    if topo_d.get("layout"):
        from ..connectome.regions import RegionLayout

        topology.layout = RegionLayout.from_dict(topo_d["layout"])
    topology.kind = topo_d.get("kind", "legacy")
    kwargs = dict(topology=topology, sensory_dim=int(sub_d["sensory_dim"]), leak=float(sub_d["leak"]), microsteps=int(sub_d["microsteps"]),
                  current_cap=float(sub_d["current_cap"]), seed=int(sub_d.get("seed", 42)))
    weave = package.get("weave")
    if weave:
        config = WeaveConfig.from_dict(weave["config"])
        config.strict = strict
        substrate: RecurrentSubstrate = WeaveSubstrate(weave=config, motor_offset=sub_d.get("motor_offset"), **kwargs)
    else:
        substrate = RecurrentSubstrate(**kwargs)
    with torch.no_grad():
        substrate.B.zero_()
        b = sub_d["B"]
        substrate.B[torch.as_tensor(_arr(b["rows"]).astype(np.int64)), torch.as_tensor(_arr(b["cols"]).astype(np.int64))] = \
            torch.as_tensor(_arr(b["values"]).astype(np.float32))
        substrate.bias.copy_(torch.as_tensor(_arr(sub_d["bias"]).astype(np.float32)))
        substrate.r.copy_(torch.as_tensor(_arr(sub_d["r"]).astype(np.float32)))
        substrate.lesion_mask.copy_(torch.as_tensor(_arr(sub_d["lesion_mask"]).astype(np.float32)))
        plastic = sub_d.get("plastic")
        if plastic and len(_arr(plastic["w"])):
            substrate.plastic_pre = torch.as_tensor(_arr(plastic["pre"]).astype(np.int64))
            substrate.plastic_post = torch.as_tensor(_arr(plastic["post"]).astype(np.int64))
            substrate.plastic_w = torch.as_tensor(_arr(plastic["w"]).astype(np.float32))
    substrate.step_count = int(sub_d.get("step_count", 0))
    if weave:
        state = from_jsonable(weave)
        state["config"] = substrate.weave.to_dict()
        substrate.load_weave_state_dict(state, strict=False)
    extras: Dict[str, Any] = {"validation": report, "training": package.get("training", {}), "experiment": package.get("experiment", {}),
                              "environment": from_jsonable(package.get("environment")) if package.get("environment") else None,
                              "motor_head": None, "format_version": package.get("formatVersion")}
    agent = package.get("agent")
    if agent:
        weights = torch.as_tensor(_arr(agent["motor_head"]).astype(np.float32))
        head = nn.Linear(weights.shape[1], weights.shape[0], bias=False)
        with torch.no_grad():
            head.weight.copy_(weights)
        extras["motor_head"] = head
    return substrate, extras


# ---------------------------------------------------------------------- files
def save_package(package: Dict[str, Any], path: str) -> int:
    text = json.dumps(package, separators=(",", ":"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def load_package(path: str) -> Dict[str, Any]:
    """Read a package file with the size limit applied.  Does NOT import it."""
    size = Path(path).stat().st_size
    if size > LIMITS["max_bytes"]:
        raise ValueError(f"package file is {size} bytes; limit is {LIMITS['max_bytes']}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_snapshot(substrate: WeaveSubstrate) -> Dict[str, Any]:
    """Lightweight model snapshot (weave state only) with a digest, for quick save / restore of one model."""
    snap = to_jsonable({"format": "SONAPSE_WEAVE_SNAPSHOT", "weaveVersion": WEAVE_VERSION, "r": substrate.r,
                        "step_count": int(substrate.step_count), "weave": substrate.weave_state_dict()})
    snap["integrity"] = {"algorithm": "sha256", "digest": _digest(snap)}
    return snap


def restore_snapshot(substrate: WeaveSubstrate, snap: Dict[str, Any]) -> None:
    if snap.get("format") != "SONAPSE_WEAVE_SNAPSHOT" or snap.get("integrity", {}).get("digest") != _digest(snap):
        raise ValueError("snapshot rejected: wrong format or integrity digest mismatch")
    state = from_jsonable(snap)
    with torch.no_grad():
        substrate.r.copy_(state["r"])
    substrate.step_count = int(state["step_count"])
    substrate.load_weave_state_dict(state["weave"], strict=False)


__all__ = ["FORMAT", "FORMAT_VERSION", "LIMITS", "to_jsonable", "from_jsonable", "export_model_package", "export_legacy_package",
           "validate_model_package", "inspect_model_package", "import_model_package", "save_package", "load_package",
           "export_snapshot", "restore_snapshot"]
