"""HTTP endpoints for the NeuroWeave observatory (added next to the existing ones, which are untouched).

``create_weave_handler`` subclasses the handler built by the existing ``create_handler``; every original
route keeps its behaviour and any path outside ``/api/weave/`` is delegated to the parent class.

    GET  /api/weave/state            live model, environment, sensory, learning and lab state
    GET  /api/weave/options          profiles, topologies, environments, experiments, mechanism catalogue
    GET  /api/weave/neuron?i=N       full compartment state of one neuron
    GET  /api/weave/export           model package (JSON download)
    GET  /api/weave/standalone       standalone HTML inference runner (download)
    GET  /api/weave/fpga             FPGA mapping manifest (JSON download)
    GET  /api/weave/receipt          last experiment receipt (JSON download)
    POST /api/weave/action           {"action": name, ...}   (see ``ACTIONS``)

Safety: request bodies are size-limited, malformed JSON is rejected with 400, and state-changing POSTs
from a foreign web origin are refused (the page itself is same-origin).  Sensor uploads carry derived
feature values only; raw camera frames or audio are never accepted by any endpoint.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict
from urllib.parse import parse_qs, urlparse

from .server import create_handler

MAX_BODY = 256 * 1024
MAX_IMPORT_BODY = 64 * 1024 * 1024


def _actions(state) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    return {
        "set_mechanism": lambda p: state.set_mechanism(str(p["id"]), bool(p["enabled"])),
        "set_profile": lambda p: state.set_profile(str(p["profile"])),
        "rebuild": lambda p: state.rebuild(seed=int(p.get("seed", 42)), profile=str(p.get("profile", "legacy")),
                                           environment=str(p.get("environment", "neuroweave_1200x660")),
                                           total_neurons=int(p.get("total_neurons", 768)), topology_kind=p.get("topology_kind")),
        "run": lambda p: state.set_running(bool(p.get("running", True))),
        "step": lambda p: state.weave_step(max(1, min(500, int(p.get("n", 1))))),
        "sensors": lambda p: state.ingest_sensors(p.get("values", {}), str(p.get("provenance", "direct_sensory"))),
        "release_sensors": lambda p: state.release_sensors(),
        "channel_mode": lambda p: state.set_channel_mode(str(p["channel"]), str(p["mode"])),
        "lesion": lambda p: state.apply_lesion(str(p["kind"]), float(p.get("severity", 0.3))),
        "clear_lesion": lambda p: state.clear_lesion(),
        "experiment": lambda p: state.run_experiment(str(p["id"]), quick=bool(p.get("quick", True)), seeds=p.get("seeds")),
        "propose_promotion": lambda p: state.propose_promotion(),
        "rollback_promotion": lambda p: state.rollback_promotion(),
        "validate_import": lambda p: state.validate_import(str(p["package"])),
        "activate_import": lambda p: state.activate_import(),
        "profile": lambda p: state.profile(int(p.get("steps", 20))),
        "regression_check": lambda p: state.regression_check(),
    }


ACTIONS = tuple(_actions(None))


def create_weave_handler(state):
    base = create_handler(state)
    actions = _actions(state)

    class WeaveObservatoryHandler(base):
        def end_headers(self):
            # static pages must be revalidated so an updated dashboard is never served from a stale browser cache
            sent = b"".join(getattr(self, "_headers_buffer", [])).lower()
            if b"cache-control" not in sent:
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def _json(self, data: Any, status: int = 200) -> None:
            payload = json.dumps(data, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _download(self, body: str, content_type: str, filename: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _same_origin(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            return urlparse(origin).netloc == self.headers.get("Host", "")

        def do_GET(self):
            url = urlparse(self.path)
            if not url.path.startswith("/api/weave/"):
                return super().do_GET()
            try:
                if url.path == "/api/weave/state":
                    self._json(state.weave_state())
                elif url.path == "/api/weave/options":
                    self._json(state.weave_options())
                elif url.path == "/api/weave/neuron":
                    self._json(state.neuron_detail(int(parse_qs(url.query).get("i", ["0"])[0])))
                elif url.path == "/api/weave/export":
                    self._download(json.dumps(state.export_package(), separators=(",", ":")), "application/json", "sonapse_model_package.json")
                elif url.path == "/api/weave/standalone":
                    staged = parse_qs(url.query).get("staged", ["0"])[0] == "1"
                    self._download(state.standalone_html(staged=staged), "text/html; charset=utf-8", "sonapse_standalone.html")
                elif url.path == "/api/weave/fpga":
                    self._download(json.dumps(state.fpga_manifest(), separators=(",", ":")), "application/json", "sonapse_fpga_mapping.json")
                elif url.path == "/api/weave/receipt":
                    receipt = state.last_receipt()
                    if receipt is None:
                        self._json({"error": "no experiment has been run yet"}, 404)
                    else:
                        self._download(json.dumps(receipt, indent=2, default=str), "application/json", f"{receipt['experiment_id']}.json")
                else:
                    self._json({"error": "unknown endpoint"}, 404)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/weave/action":
                return super().do_POST()
            if not self._same_origin():
                return self._json({"error": "cross-origin requests are not accepted"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json({"error": "invalid Content-Length"}, 400)
            if length <= 0 or length > MAX_IMPORT_BODY:
                return self._json({"error": "request body missing or too large"}, 413)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or "action" not in payload:
                    raise ValueError("body must be a JSON object with an 'action' field")
                name = str(payload["action"])
                if name not in actions:
                    return self._json({"error": f"unknown action '{name}'", "available": list(actions)}, 404)
                if name != "validate_import" and length > MAX_BODY:
                    return self._json({"error": "request body too large for this action"}, 413)
                self._json(actions[name](payload))
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    return WeaveObservatoryHandler


__all__ = ["create_weave_handler", "ACTIONS", "MAX_BODY", "MAX_IMPORT_BODY"]
