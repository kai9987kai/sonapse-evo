"""Sensory hub: sixteen channels projected into the connectome.

Each channel drives its own small sub-population of sensory neurons through a fixed seeded sparse
projection ``P`` (N x 16, ``fan`` target neurons per channel):

    sensory_extra += gain * attention_gain * (P @ x)          -> proximal dendrite

Targets are the neurons of the region whose role matches the channel's modality when the substrate
carries a region layout (ten-region connectomes), otherwise sub-zones of the legacy sensory entry zone
(the first quarter of the population).  With every channel at zero the contribution is exactly zero.

Channels can be enabled, disabled, simulated or externally supplied individually.  Camera frames and
audio buffers are reduced to features by the extractors and discarded; the hub's state contains features,
modes, provenance and timestamps only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, Stage, StepContext
from .auditory import AudioFeatureExtractor
from .channels import CHANNEL_SPECS, SENSORY_CHANNELS, ChannelMode, SensoryManifest
from .simulated import SimulatedSensors
from .visual import VisualFeatureExtractor

_REGION_FOR_MODALITY = {"visual": "visual", "auditory": "auditory", "olfactory": "olfactory", "gustatory": "gustatory",
                        "thermal": "thermosensory", "mechanical": "mechanosensory", "proprioceptive": "mechanosensory",
                        "interoceptive": "interoceptive", "context": "association"}


class MultimodalSensory(Mechanism):
    mechanism_id = "multimodal_sensory"
    stage = Stage.SENSE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"gain": 0.35, "fan": 12, "stale_steps": 8, "modes": {}}

    def on_attach(self) -> None:
        sub, p = self.substrate, self.params
        n, c = sub.total_neurons, len(SENSORY_CHANNELS)
        rng = np.random.default_rng(sub.weave.seed + 5381)
        self.manifest = SensoryManifest()
        for name, mode in p["modes"].items():
            self.set_mode(name, mode)
        layout = getattr(sub, "layout", None)
        zone = np.arange(max(c, n // 4))
        proj = np.zeros((n, c), dtype=np.float32)
        for i, spec in enumerate(CHANNEL_SPECS):
            pool = None
            region = _REGION_FOR_MODALITY.get(spec.modality)
            if layout is not None and region in getattr(layout, "names", ()):
                pool = np.asarray(layout.indices(region))
            if pool is None or pool.size == 0:
                lo, hi = (i * len(zone)) // c, ((i + 1) * len(zone)) // c
                pool = zone[lo:max(hi, lo + 1)]
            chosen = rng.choice(pool, size=min(int(p["fan"]), pool.size), replace=False)
            proj[chosen, i] = rng.uniform(0.5, 1.0, size=chosen.size)
        self.projection = torch.as_tensor(proj).to_sparse()
        self.simulated = SimulatedSensors(seed=sub.weave.seed + 17)
        self.visual = VisualFeatureExtractor()
        self.audio = AudioFeatureExtractor()
        self.extended: Dict[str, float] = {}
        self.ingested = 0
        self.reset()

    def reset(self) -> None:
        c = len(SENSORY_CHANNELS)
        self.values = np.zeros(c, dtype=np.float32)
        self.reliability = np.ones(c, dtype=np.float32)
        self.updated_at = np.full(c, -10 ** 9, dtype=np.int64)
        self.timestamps = np.zeros(c, dtype=np.float64)
        self.clock = 0
        self.visual.reset()
        self.audio.reset()
        self.simulated.reset()

    # ----------------------------------------------------------------- inputs
    def set_mode(self, channel: str, mode: Union[str, ChannelMode]) -> None:
        if channel not in SENSORY_CHANNELS:
            raise KeyError(f"unknown sensory channel '{channel}'")
        mode = ChannelMode(mode)
        if mode is ChannelMode.SIMULATED and channel not in SimulatedSensors.channels:
            raise ValueError(f"channel '{channel}' has no simulator; simulated mode is available for {list(SimulatedSensors.channels)}")
        self.manifest.modes[channel] = mode

    def ingest(self, values: Union[Mapping[str, float], Sequence[float]], provenance: Provenance = Provenance.SYNTHETIC_SIMULATION,
               timestamp: Optional[float] = None, reliability: Optional[Mapping[str, float]] = None) -> int:
        """Update channels.  Returns how many were accepted.

        The whole payload is validated before anything is written: an unknown name or a non-finite value or
        reliability rejects the payload and leaves the hub unchanged.
        """
        if not isinstance(values, Mapping):
            seq = list(values)
            if len(seq) != len(SENSORY_CHANNELS):
                raise ValueError(f"expected {len(SENSORY_CHANNELS)} channel values, got {len(seq)}")
            values = dict(zip(SENSORY_CHANNELS, seq))
        provenance = Provenance(provenance)
        if timestamp is not None and not np.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite")
        staged = []
        for name, value in values.items():
            if name not in SENSORY_CHANNELS:
                if name in self.extended or str(name).startswith(("visual_", "audio_")):
                    continue
                raise KeyError(f"unknown sensory channel '{name}'")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"non-finite value for channel '{name}'")
            rel = None
            if reliability and name in reliability:
                rel = float(reliability[name])
                if not np.isfinite(rel):
                    raise ValueError(f"non-finite reliability for channel '{name}'")
            staged.append((name, value, rel))
        accepted = 0
        for name, value, rel in staged:
            if self.manifest.modes[name] in (ChannelMode.DISABLED, ChannelMode.SIMULATED):
                continue
            i = SENSORY_CHANNELS.index(name)
            self.values[i] = np.clip(value, CHANNEL_SPECS[i].low, CHANNEL_SPECS[i].high)
            self.updated_at[i] = self.clock
            self.timestamps[i] = float(self.clock if timestamp is None else timestamp)
            self.manifest.provenance[name] = provenance.value
            if rel is not None:
                self.reliability[i] = float(np.clip(rel, 0.0, 1.0))
            accepted += 1
        self.ingested += accepted
        return accepted

    @staticmethod
    def _finite_features(features: Mapping[str, Any]) -> None:
        for key, value in features.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            if not all(np.isfinite(float(v)) for v in values):
                raise ValueError(f"extracted feature '{key}' is not finite; the input was rejected")

    def ingest_external(self, values, timestamp: Optional[float] = None) -> int:
        return self.ingest(values, provenance=Provenance.EXTERNAL_SUPPLIED, timestamp=timestamp)

    def ingest_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Reduce a camera frame to features locally; the frame itself is not kept."""
        frame = np.asarray(frame)
        if frame.size and not np.all(np.isfinite(frame.astype(np.float64, copy=False))):
            raise ValueError("frame contains non-finite pixels")
        saved = (self.visual._previous, self.visual._last_mean, self.visual.flicker, self.visual.frames)
        try:
            features = self.visual.extract(frame)
            self._finite_features(features)
        except Exception:
            self.visual._previous, self.visual._last_mean, self.visual.flicker, self.visual.frames = saved
            raise
        self.manifest.camera_enabled = True
        self.extended.update({k: v for k, v in features.items() if k not in SENSORY_CHANNELS and not isinstance(v, list)})
        self.ingest({k: v for k, v in features.items() if k in SENSORY_CHANNELS}, provenance=Provenance.DIRECT_SENSORY)
        return features

    def ingest_audio(self, samples: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Reduce an audio buffer to features locally; the samples are not kept."""
        saved = (self.audio._shares.copy(), self.audio._amplitude, self.audio.buffers)
        try:
            features = self.audio.extract(samples, sample_rate)
            self._finite_features(features)
        except Exception:
            self.audio._shares, self.audio._amplitude, self.audio.buffers = saved
            raise
        self.manifest.microphone_enabled = True
        self.extended.update({k: v for k, v in features.items() if k not in SENSORY_CHANNELS})
        self.ingest({k: v for k, v in features.items() if k in SENSORY_CHANNELS}, provenance=Provenance.DIRECT_SENSORY)
        return features

    # ---------------------------------------------------------------- outputs
    def missing_mask(self) -> np.ndarray:
        """True where a channel is disabled or has not been refreshed within ``stale_steps``."""
        stale = (self.clock - self.updated_at) > int(self.params["stale_steps"])
        disabled = np.array([self.manifest.modes[c] is ChannelMode.DISABLED for c in SENSORY_CHANNELS])
        simulated = np.array([self.manifest.modes[c] is ChannelMode.SIMULATED for c in SENSORY_CHANNELS])
        return (stale & ~simulated) | disabled

    def current_vector(self) -> np.ndarray:
        x = self.values.copy()
        x[self.missing_mask()] = 0.0
        return x

    def current_provenance(self) -> List[str]:
        return [self.manifest.provenance[c] for c in SENSORY_CHANNELS]

    def project(self, vector: np.ndarray) -> torch.Tensor:
        return torch.sparse.mm(self.projection, torch.as_tensor(vector, dtype=torch.float32).unsqueeze(1)).squeeze(1)

    def begin_step(self, ctx: StepContext) -> None:
        self.clock += 1
        if any(mode is ChannelMode.SIMULATED for mode in self.manifest.modes.values()):
            for name, value in self.simulated.read().items():
                if self.manifest.modes[name] is ChannelMode.SIMULATED:
                    i = SENSORY_CHANNELS.index(name)
                    self.values[i], self.updated_at[i] = value, self.clock
                    self.manifest.provenance[name] = Provenance.SYNTHETIC_SIMULATION.value
        x = self.current_vector()
        ctx.scratch["sensory_vector"] = x
        ctx.scratch["sensory_missing"] = self.missing_mask()
        ctx.scratch["sensory_provenance"] = self.current_provenance()
        if np.any(x):
            ctx.sensory_extra += float(self.params["gain"]) * float(ctx.signals.attention_gain) * self.project(x)

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"values": torch.as_tensor(self.values), "reliability": torch.as_tensor(self.reliability),
                "updated_at": torch.as_tensor(self.updated_at), "timestamps": torch.as_tensor(self.timestamps),
                "clock": int(self.clock), "manifest": self.manifest.to_dict(),
                "extended": dict(self.extended), "ingested": int(self.ingested), "simulated_t": int(self.simulated.t)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.values = np.asarray(state["values"], dtype=np.float32).copy()
        self.reliability = np.asarray(state["reliability"], dtype=np.float32).copy()
        self.updated_at = np.asarray(state["updated_at"], dtype=np.int64).copy()
        self.clock = int(state.get("clock", 0))
        if "timestamps" in state:
            self.timestamps = np.asarray(state["timestamps"], dtype=np.float64).copy()
        self.manifest = SensoryManifest.from_dict(state.get("manifest", {}))
        self.extended = {k: float(v) for k, v in state.get("extended", {}).items()}
        self.ingested = int(state.get("ingested", 0))
        self.simulated.t = int(state.get("simulated_t", 0))

    def telemetry(self) -> Dict[str, Any]:
        x = self.current_vector()
        return {"channels": {name: float(v) for name, v in zip(SENSORY_CHANNELS, x)},
                "missing": [n for n, m in zip(SENSORY_CHANNELS, self.missing_mask()) if m],
                "modes": {k: v.value for k, v in self.manifest.modes.items()}, "provenance": dict(self.manifest.provenance),
                "extended": dict(self.extended), "camera_enabled": self.manifest.camera_enabled,
                "microphone_enabled": self.manifest.microphone_enabled, "raw_media_stored": False, "ingested": int(self.ingested)}
