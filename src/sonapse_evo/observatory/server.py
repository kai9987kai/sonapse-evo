from __future__ import annotations

import http.server
import socketserver
import json
import threading
import time
from pathlib import Path
from typing import Optional

from .live_state import LiveObservatoryState

WEB_DIR = Path(__file__).resolve().parent / "web"


def create_handler(live_state: LiveObservatoryState):
    class ObservatoryHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        def _send_json(self, data: Any, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            payload = json.dumps(data).encode("utf-8")
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/api/live_state":
                self._send_json(live_state.get_full_state())
            elif self.path == "/api/telemetry":
                self._send_json(live_state.telemetry_history)
            elif self.path == "/api/action/discovery_step":
                res = live_state.trigger_discovery_step()
                self._send_json(res)
            elif self.path == "/api/action/induce_lesion":
                res = live_state.trigger_lesion_event()
                self._send_json(res)
            elif self.path == "/api/action/repair":
                res = live_state.trigger_repair_event()
                self._send_json(res)
            elif self.path == "/api/health":
                self._send_json({"status": "healthy", "service": "sonapse-evo-observatory", "version": "2.0.0"})
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == "/api/action/discovery_step":
                res = live_state.trigger_discovery_step()
                self._send_json(res)
            elif self.path == "/api/action/induce_lesion":
                res = live_state.trigger_lesion_event()
                self._send_json(res)
            elif self.path == "/api/action/repair":
                res = live_state.trigger_repair_event()
                self._send_json(res)
            else:
                self.send_error(404, "Endpoint not found")

    return ObservatoryHandler


def run_server(port: int = 8890, live_state: Optional[LiveObservatoryState] = None) -> None:
    state = live_state or LiveObservatoryState()

    # Start background simulation worker thread (1-second tick)
    def worker():
        while state.running:
            time.sleep(1.0)
            state.tick_simulation()

    sim_thread = threading.Thread(target=worker, daemon=True)
    sim_thread.start()

    handler = create_handler(state)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"SONAPSE-EVO 2.0 Live Observatory running on http://localhost:{port}")
        httpd.serve_forever()
