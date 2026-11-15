"""Live NeuroWeave observatory state (additive extension of ``LiveObservatoryState``).

``WeaveObservatoryState`` IS a ``LiveObservatoryState``: every existing field, method and JSON key of the
original dashboard feed is produced by the parent class, unchanged.  It adds a real, running
``WeaveSubstrate`` closed-loop with an ecology environment, and a set of actions that the NeuroWeave
observatory page calls.  Every control on that page maps to a method here that changes the actual model.

The numbers under the ``"weave"`` key are measured from the running substrate, not simulated telemetry.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..neuroweave.compat import verify_legacy_identity
from ..neuroweave.connectome import (LESION_KINDS, TOPOLOGY_KINDS, apply_lesion_mask, build_lesion_mask, clear_lesions,
                                     compute_graph_metrics, edge_class_counts)
from ..neuroweave.contracts import Provenance
from ..neuroweave.ecology import list_environments, make_environment
from ..neuroweave.experiments import export_receipt, list_experiments, run_catalog_experiment
from ..neuroweave.hardware import available_backends, build_fpga_manifest, profile_substrate, validate_fpga_manifest
from ..neuroweave.portability import (export_model_package, export_standalone_html, import_model_package, inspect_model_package,
                                      validate_model_package)
from ..neuroweave.promotion import PromotionLedger
from ..neuroweave.registry import MECHANISMS, PROFILES, describe_catalogue
from ..neuroweave.runtime import build_weave_substrate, canonical_motor_head
from .live_state import LiveObservatoryState


class WeaveObservatoryState(LiveObservatoryState):
    def __init__(self, seed: int = 42, profile: str = "legacy", environment: str = "neuroweave_1200x660",
                 total_neurons: int = 768, topology_kind: Optional[str] = None, steps_per_tick: int = 5):
        super().__init__()
        self.weave_lock = threading.RLock()
        self.weave_running = False
        self.steps_per_tick = int(steps_per_tick)
        self.receipts: List[Dict[str, Any]] = []
        self.staged_import: Optional[Dict[str, Any]] = None
        self.staged_report: Optional[Dict[str, Any]] = None
        self.promotion = PromotionLedger()
        self.live_sensors = False
        self.weave_faults: List[Dict[str, Any]] = []
        self.rebuild(seed=seed, profile=profile, environment=environment, total_neurons=total_neurons, topology_kind=topology_kind)

    # --------------------------------------------------------------- building
    def rebuild(self, seed: int = 42, profile: str = "legacy", environment: str = "neuroweave_1200x660",
                total_neurons: int = 768, topology_kind: Optional[str] = None, enable: Optional[List[str]] = None) -> Dict[str, Any]:
        """Construct a fresh model and environment (seed / population / topology / environment controls).

        Atomic: everything is built into local variables first and only assigned when every step succeeded, so a
        bad request (unknown environment, impossible population) leaves the running model untouched.
        """
        if profile not in PROFILES:
            raise KeyError(f"unknown profile '{profile}'")
        if topology_kind in (None, "", "legacy"):
            topology_kind = None
            if int(total_neurons) < 576:
                raise ValueError("the legacy topology needs at least 576 neurons; choose another topology for smaller populations")
        elif topology_kind not in TOPOLOGY_KINDS:
            raise KeyError(f"unknown topology '{topology_kind}'")
        motor = canonical_motor_head()
        if int(total_neurons) < motor.weight.shape[1]:
            raise ValueError(f"the motor read-out needs at least {motor.weight.shape[1]} neurons")
        substrate = build_weave_substrate(total_neurons=int(total_neurons), seed=int(seed), profile=profile,
                                          enable=tuple(enable or ()), topology_kind=topology_kind)
        env = make_environment(environment, seed=int(seed))
        obs = env.reset(seed=int(seed))
        substrate.reset_state()
        config = {"seed": int(seed), "profile": profile, "environment": environment, "total_neurons": int(total_neurons),
                  "topology_kind": topology_kind or "legacy"}
        with self.weave_lock:
            self.weave_config, self.substrate, self.motor, self.env, self._obs = config, substrate, motor, env, obs
            self.weave_history: List[Dict[str, float]] = []
            self.episode, self.episode_return, self.last_returns = 0, 0.0, []
            self._graph_cache = None
            self._prev_action = None
            self._rng = np.random.default_rng(int(seed) + 90001)
            return {"status": "REBUILT", **self.weave_config, "mechanisms": list(self.substrate.weave.enabled_ids())}

    # --------------------------------------------------------------- stepping
    def weave_step(self, n: int = 1) -> Dict[str, Any]:
        """Advance the closed loop by ``n`` environment steps (same loop as ``run_weave_episode``)."""
        with self.weave_lock:
            sub, env = self.substrate, self.env
            width = self.motor.weight.shape[1]
            readout = sub.motor_offset if sub.motor_offset + width <= sub.total_neurons else 0
            for _ in range(int(n)):
                hub = sub.mechanisms.get("multimodal_sensory")
                if hub is not None and not self.live_sensors and hasattr(env, "sensory_channels"):
                    hub.ingest(env.sensory_channels(), provenance=getattr(env, "sensory_provenance", Provenance.SYNTHETIC_SIMULATION))
                rates = sub.forward_step(torch.as_tensor(self._obs, dtype=torch.float32))
                with torch.no_grad():
                    action = torch.tanh(self.motor(rates[readout:readout + width])).numpy()
                intended = action
                if sub.signals.exploration > 0.0:
                    action = np.clip(action + sub.signals.exploration * self._rng.standard_normal(action.shape), -1, 1).astype(np.float32)
                if sub.signals.persistence > 0.0 and self._prev_action is not None:
                    action = (sub.signals.persistence * self._prev_action + (1 - sub.signals.persistence) * action).astype(np.float32)
                self._prev_action = action
                self._obs, reward, done, info = env.step(action)
                self.episode_return += float(reward)
                sub.deliver_feedback(reward=float(reward), done=bool(done), info=info, intended_action=intended, issued_action=action,
                                     predicted_outcome=info.get("predicted_outcome"), observed_outcome=info.get("observed_outcome"))
                self.weave_history.append({"step": int(sub.step_count), "reward": float(reward), "mean_rate": float(rates.mean()),
                                           "uncertainty": float(sub.signals.uncertainty), "surprise": float(sub.signals.surprise),
                                           "dopamine": float(sub.signals.modulators.dopamine)})
                self.weave_history = self.weave_history[-240:]
                if done:
                    self.last_returns = (self.last_returns + [self.episode_return])[-20:]
                    self.episode += 1
                    self.episode_return = 0.0
                    self._obs = env.reset(seed=self.weave_config["seed"] + self.episode)
                    sub.reset_state()
                    self._prev_action = None
            return {"status": "STEPPED", "steps": int(n), "step_count": int(sub.step_count), "episode": self.episode}

    def tick_simulation(self):
        super().tick_simulation()                      # the original telemetry feed, unchanged
        if self.weave_running:
            try:
                self.weave_step(self.steps_per_tick)
            except Exception as exc:                   # a failing model must never stop the original telemetry thread
                self.weave_running = False
                self.weave_faults.append({"time": time.time(), "error": f"{type(exc).__name__}: {exc}", "action": "paused"})
                self.weave_faults = self.weave_faults[-20:]

    # ------------------------------------------------------------------ views
    def _region_table(self) -> List[Dict[str, Any]]:
        sub = self.substrate
        cell, r = sub.cell, sub.r.detach()
        names = list(sub.topology.region_names)
        roles = list(sub.layout.roles) if sub.layout is not None else ["sensory", "association", "motor"]
        rows = []
        for i, g in enumerate(torch.unique(sub.region_ids).tolist()):
            idx = sub.region_ids == g
            rows.append({"id": int(g), "name": names[g] if g < len(names) else str(g), "role": roles[i] if i < len(roles) else "",
                         "neurons": int(idx.sum()), "rate": float(r[idx].mean()), "spike_rate": float(cell.spike[idx].mean()),
                         "apical": float(cell.apic[idx].abs().mean()), "distal": float(cell.dist[idx].abs().mean()),
                         "proximal": float(cell.prox[idx].abs().mean()), "context": float(cell.soma_context[idx].abs().mean()),
                         "gating": float(cell.soma_gating[idx].mean()), "release": float(cell.release[idx].mean()),
                         "surprise": float(cell.pred_surprise[idx].mean())})
        return rows

    def graph_metrics(self) -> Dict[str, Any]:
        with self.weave_lock:
            if self._graph_cache is None:
                m = compute_graph_metrics(self.substrate.topology, seed=0, max_sources=48)
                self._graph_cache = {k: m[k] for k in ("neurons", "edges", "density", "clustering_coefficient", "modularity",
                                                       "characteristic_path_length", "global_efficiency", "hub_concentration_top5pct",
                                                       "regional_communication_efficiency", "reciprocity", "inter_region_density")}
                self._graph_cache["in_degree"] = {k: m["in_degree"][k] for k in ("mean", "max", "cv")}
                self._graph_cache["edge_classes"] = edge_class_counts(self.substrate.topology)
            return self._graph_cache

    def weave_state(self) -> Dict[str, Any]:
        """Everything the NeuroWeave observatory page renders."""
        with self.weave_lock:
            sub = self.substrate
            telemetry = sub.weave_telemetry()
            r = sub.r.detach()
            stride = max(1, sub.total_neurons // 256)
            return {
                "config": dict(self.weave_config), "running": bool(self.weave_running), "live_sensors": bool(self.live_sensors),
                "runtime_faults": list(self.weave_faults[-5:]),
                "episode": int(self.episode), "episode_return": float(self.episode_return), "last_returns": list(self.last_returns),
                "telemetry": telemetry, "regions": self._region_table(), "graph": self.graph_metrics(),
                "rates_sample": [round(float(v), 4) for v in r[::stride][:256]],
                "somas": {name: round(float(v), 5) for name, v in zip(
                    ("primary", "secondary", "fast", "slow", "context", "integrator", "burst", "gating"),
                    sub.cell.soma_matrix(r).abs().mean(dim=1))},
                "environment": self.env.render_state() if hasattr(self.env, "render_state") else {"name": type(self.env).__name__},
                "history": self.weave_history[-120:],
                "mechanisms": [{**entry, "enabled": sub.weave.is_enabled(entry["id"]), "attached": entry["id"] in sub.mechanisms}
                               for entry in describe_catalogue()],
                "receipts": [{"experiment_id": x["experiment_id"], "verdict": x["verdict"], "evidence_tier": x["evidence_tier"],
                              "digest": x["result_digest"][:12], "summary": x["results"]["summary"]} for x in self.receipts[-8:]],
                "staged_import": self.staged_report,
                "promotion": {"stable": list(self.promotion.stable), "decisions": self.promotion.decisions[-5:],
                              "failures": len(self.promotion.failures())},
                "lesion": {"edges_lesioned": int((sub.lesion_mask == 0).sum()), "edges_total": int(sub.lesion_mask.numel())},
            }

    def weave_options(self) -> Dict[str, Any]:
        return {"profiles": {k: len(v) for k, v in PROFILES.items()}, "topologies": list(TOPOLOGY_KINDS),
                "environments": list_environments(), "experiments": list_experiments(), "lesion_kinds": list(LESION_KINDS),
                "backends": available_backends(), "catalogue": describe_catalogue()}

    def neuron_detail(self, index: int) -> Dict[str, Any]:
        with self.weave_lock:
            sub, cell = self.substrate, self.substrate.cell
            i = int(np.clip(index, 0, sub.total_neurons - 1))
            pre, post = sub.edge_index()
            w = sub.effective_weights().detach()
            fields = ("prox", "dist", "apic", "basal", "soma_secondary", "soma_fast", "soma_slow", "soma_context", "soma_integrator",
                      "soma_burst", "soma_gating", "membrane", "threshold", "refractory", "spike", "calcium", "adaptation", "hillock",
                      "terminal", "release", "resource", "homeo", "astro", "myelin", "utility", "trace", "pred_error", "pred_surprise",
                      "tm_fast", "tm_mid", "tm_slow")
            return {"index": i, "region": int(sub.region_ids[i]), "rate": float(sub.r[i]),
                    "sign": "inhibitory" if i % 5 == 0 else "excitatory",
                    "compartments": {name: float(getattr(cell, name)[i]) for name in fields},
                    "branches": [float(b) for b in cell.branches[:, i]],
                    "in_degree": int((post == i).sum()), "out_degree": int((pre == i).sum()),
                    "strongest_inputs": [{"from": int(pre[e]), "weight": float(w[e])}
                                         for e in torch.nonzero(post == i, as_tuple=False).squeeze(-1)[
                                             torch.argsort(w[post == i].abs(), descending=True)[:6]].tolist()]}

    def get_full_state(self) -> Dict[str, Any]:
        state = super().get_full_state()
        with self.weave_lock:
            sub = self.substrate
            state["weave"] = {"profile": sub.weave.profile, "enabled": len(sub.weave.enabled_ids()), "neurons": int(sub.total_neurons),
                              "mean_rate": float(sub.r.detach().mean()), "step_count": int(sub.step_count),
                              "running": bool(self.weave_running), "faults": len(sub.weave_faults)}
        return state

    # ---------------------------------------------------------------- actions
    def set_mechanism(self, mechanism_id: str, enabled: bool) -> Dict[str, Any]:
        with self.weave_lock:
            if mechanism_id not in MECHANISMS:
                raise KeyError(f"unknown mechanism '{mechanism_id}'")
            self.substrate.set_mechanism(mechanism_id, bool(enabled))
            self.weave_config["profile"] = self.substrate.weave.profile
            return {"status": "OK", "mechanism": mechanism_id, "enabled": self.substrate.weave.is_enabled(mechanism_id),
                    "attached": mechanism_id in self.substrate.mechanisms, "faults": self.substrate.weave_faults[-1:]}

    def set_profile(self, profile: str) -> Dict[str, Any]:
        cfg = dict(self.weave_config)
        return self.rebuild(seed=cfg["seed"], profile=profile, environment=cfg["environment"], total_neurons=cfg["total_neurons"],
                            topology_kind=cfg["topology_kind"])

    def set_running(self, running: bool) -> Dict[str, Any]:
        self.weave_running = bool(running)
        return {"status": "RUNNING" if self.weave_running else "PAUSED"}

    def ingest_sensors(self, values: Dict[str, float], provenance: str = "direct_sensory") -> Dict[str, Any]:
        """Derived sensory features from the browser (never raw media).  Requires the sensory hub to be enabled."""
        with self.weave_lock:
            hub = self.substrate.mechanisms.get("multimodal_sensory")
            if hub is None:
                return {"status": "IGNORED", "reason": "enable the 'multimodal_sensory' mechanism first"}
            if not isinstance(values, dict) or len(values) > 64:
                raise ValueError("sensor payload must be a small mapping of channel name to number")
            clean = {str(k): float(v) for k, v in values.items() if isinstance(v, (int, float))}
            accepted = hub.ingest(clean, provenance=Provenance(provenance))
            self.live_sensors = True
            hub.manifest.camera_enabled = hub.manifest.camera_enabled or any(k.startswith("visual_") for k in clean)
            hub.manifest.microphone_enabled = hub.manifest.microphone_enabled or any(k.startswith("audio_") for k in clean)
            return {"status": "OK", "accepted": int(accepted), "raw_media_stored": False}

    def release_sensors(self) -> Dict[str, Any]:
        self.live_sensors = False
        return {"status": "OK", "live_sensors": False}

    def set_channel_mode(self, channel: str, mode: str) -> Dict[str, Any]:
        with self.weave_lock:
            hub = self.substrate.mechanisms.get("multimodal_sensory")
            if hub is None:
                return {"status": "IGNORED", "reason": "enable the 'multimodal_sensory' mechanism first"}
            hub.set_mode(channel, mode)
            return {"status": "OK", "channel": channel, "mode": mode}

    def apply_lesion(self, kind: str, severity: float = 0.3) -> Dict[str, Any]:
        with self.weave_lock:
            mask = build_lesion_mask(self.substrate.topology, kind, float(severity), seed=self.weave_config["seed"] + 7, protect_first=0)
            removed = apply_lesion_mask(self.substrate, mask)
            return {"status": "LESIONED", "kind": kind, "severity": float(severity), "edges_removed": removed}

    def clear_lesion(self) -> Dict[str, Any]:
        with self.weave_lock:
            clear_lesions(self.substrate)
            return {"status": "CLEARED"}

    def run_experiment(self, experiment_id: str, quick: bool = True, seeds: Optional[List[int]] = None,
                       export_dir: Optional[str] = None) -> Dict[str, Any]:
        """Runs on separate substrates built by the experiment framework; the live model is not touched."""
        receipt = run_catalog_experiment(experiment_id, seeds=seeds, quick=bool(quick))
        self.receipts.append(receipt)
        out = {"status": "DONE", "experiment_id": receipt["experiment_id"], "verdict": receipt["verdict"],
               "evidence_tier": receipt["evidence_tier"], "summary": receipt["results"]["summary"],
               "comparisons": receipt["results"]["comparisons"], "digest": receipt["result_digest"]}
        if export_dir:
            out["files"] = export_receipt(receipt, export_dir)
        return out

    def last_receipt(self) -> Optional[Dict[str, Any]]:
        return self.receipts[-1] if self.receipts else None

    def propose_promotion(self) -> Dict[str, Any]:
        receipt = self.last_receipt()
        if receipt is None or "spec" not in receipt:
            return {"status": "NO_RECEIPT", "reason": "run a paired experiment first"}
        return self.promotion.propose(receipt["spec"]["candidate"], receipt)

    def rollback_promotion(self) -> Dict[str, Any]:
        return self.promotion.rollback()

    def export_package(self) -> Dict[str, Any]:
        with self.weave_lock:
            return export_model_package(self.substrate, motor_head=self.motor,
                                        environment=self.env if hasattr(self.env, "state_dict") else None,
                                        training_history={"episodes": int(self.episode), "returns": list(self.last_returns)})

    def validate_import(self, text: str) -> Dict[str, Any]:
        """Read-only validation of an uploaded package.  The active model is never touched."""
        self.staged_import, self.staged_report = None, None     # a new upload always replaces what was staged
        report = inspect_model_package(text)
        if report["validation"]["ok"]:
            self.staged_import = json.loads(text)
        else:
            self.staged_import = None                  # a failed import clears any previously staged model
        self.staged_report = {"ok": report["validation"]["ok"], "issues": report["validation"]["issues"],
                              "warnings": report["validation"]["warnings"], "counts": report.get("counts"),
                              "mechanisms": report.get("mechanisms"), "formatVersion": report.get("formatVersion"),
                              "capabilities": report.get("capabilities"), "active_model_changed": False}
        return self.staged_report

    def activate_import(self) -> Dict[str, Any]:
        """Explicit second step: replace the live model with the validated, staged package."""
        with self.weave_lock:
            if self.staged_import is None:
                return {"status": "NOTHING_STAGED"}
            substrate, extras = import_model_package(self.staged_import)
            if not hasattr(substrate, "weave"):
                from ..neuroweave.runtime import upgrade_substrate

                substrate = upgrade_substrate(substrate)
            head = extras.get("motor_head") or self.motor
            if substrate.total_neurons < head.weight.shape[1]:
                raise ValueError(f"the imported model has fewer neurons than the motor read-out needs ({head.weight.shape[1]})")
            if substrate.sensory_dim != getattr(self.env, "sensory_dim", substrate.sensory_dim):
                raise ValueError("the imported model's observation size does not match the current environment")
            self.substrate = substrate
            if extras.get("motor_head") is not None:
                self.motor = extras["motor_head"]
            self.weave_config.update(total_neurons=int(substrate.total_neurons), profile=substrate.weave.profile,
                                     topology_kind=getattr(substrate.topology, "kind", "legacy"))
            self._graph_cache = None
            self.staged_import = None
            return {"status": "ACTIVATED", "neurons": int(substrate.total_neurons), "mechanisms": list(substrate.weave.enabled_ids())}

    def standalone_html(self, staged: bool = False) -> str:
        package = self.staged_import if (staged and self.staged_import is not None) else self.export_package()
        return export_standalone_html(package)

    def fpga_manifest(self) -> Dict[str, Any]:
        with self.weave_lock:
            manifest = build_fpga_manifest(self.substrate, include_synapses=self.substrate.total_neurons <= 4096)
            manifest["validation"]["manifest_issues"] = validate_fpga_manifest(manifest)
            return manifest

    def profile(self, steps: int = 20) -> Dict[str, Any]:
        with self.weave_lock:
            return profile_substrate(self.substrate, steps=int(np.clip(steps, 5, 200)))

    def regression_check(self) -> Dict[str, Any]:
        return verify_legacy_identity()


__all__ = ["WeaveObservatoryState"]
