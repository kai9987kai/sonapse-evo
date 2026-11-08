from __future__ import annotations

from typing import Dict, Any, List, Optional
import time
import json


class TelemetryLogger:
    """Records real-time experimental telemetry, deficit scores, and graft dynamics."""

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def record_step(
        self,
        step: int,
        mean_rate: float,
        deficit_score: float,
        jacobian_norm: float,
        max_aux_current: float,
        active_graft_count: int,
        task_return: float = 0.0,
    ) -> None:
        entry = {
            "step": step,
            "timestamp": time.time(),
            "mean_rate": float(mean_rate),
            "deficit_score": float(deficit_score),
            "jacobian_norm": float(jacobian_norm),
            "max_aux_current": float(max_aux_current),
            "active_graft_count": int(active_graft_count),
            "task_return": float(task_return),
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_latest(self) -> Dict[str, Any]:
        return self.history[-1] if self.history else {}

    def export_json(self) -> str:
        return json.dumps(self.history, indent=2)
