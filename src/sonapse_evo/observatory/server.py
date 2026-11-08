from __future__ import annotations

import http.server
import socketserver
import json
from pathlib import Path
from typing import Optional

from .telemetry import TelemetryLogger

WEB_DIR = Path(__file__).resolve().parent / "web"


def create_handler(telemetry: TelemetryLogger):
    class ObservatoryHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        def do_GET(self):
            if self.path == "/api/telemetry":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                data = json.dumps(telemetry.history).encode("utf-8")
                self.wfile.write(data)
            elif self.path == "/api/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "healthy", "service": "sonapse-evo-observatory"}')
            else:
                super().do_GET()

    return ObservatoryHandler


def run_server(port: int = 8890, telemetry: Optional[TelemetryLogger] = None) -> None:
    tel = telemetry or TelemetryLogger()
    handler = create_handler(tel)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"SONAPSE-EVO Observatory serving on http://localhost:{port}")
        httpd.serve_forever()
