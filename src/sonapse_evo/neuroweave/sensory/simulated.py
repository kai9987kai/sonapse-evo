"""Simulated chemical, thermal, mechanical, proprioceptive and interoceptive channels.

Given an environment-state mapping the channels are deterministic functions of it; without one they are
smooth seeded signals so every channel can be exercised in isolation.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

_SIMULATED = ("olfactory", "gustatory", "thermosensation", "interoception", "proprioception", "mechanosensation")


class SimulatedSensors:
    channels = _SIMULATED

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0.0, 2 * np.pi, size=len(_SIMULATED))
        self.freq = rng.uniform(0.01, 0.06, size=len(_SIMULATED))
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def read(self, environment: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        self.t += 1
        if environment is not None:
            food = float(np.clip(environment.get("food", 0.0), 0.0, 1.0))
            energy = float(np.clip(environment.get("energy", 1.0), 0.0, 1.0))
            return {
                "olfactory": food,
                "gustatory": float(np.clip(food * (1.0 if energy < 0.55 else 0.35), 0.0, 1.0)),
                "thermosensation": float(np.clip(environment.get("temperature", 0.5), 0.0, 1.0)),
                "interoception": float(np.clip(1.0 - energy, 0.0, 1.0)),
                "proprioception": float(np.clip(environment.get("speed", 0.0), 0.0, 1.0)),
                "mechanosensation": float(np.clip(environment.get("touch", 0.0), 0.0, 1.0)),
            }
        values = 0.5 + 0.5 * np.sin(self.freq * self.t + self.phase)
        return {name: float(v) for name, v in zip(_SIMULATED, values)}
