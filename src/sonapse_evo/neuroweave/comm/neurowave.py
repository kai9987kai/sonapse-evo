"""NeuroWave-X2: multi-representational neural communication.

Instead of forcing every signal into one scalar per neuron, regions exchange
typed, versioned :class:`SignalPacket` objects carrying *several simultaneous
representations* of the same neural state:

    scalar, vector, spectral, phase, temporal, spatial, graph, sparse (SDR),
    vector-symbolic (VSA), event, population-coded, context, predictive,
    uncertainty, multi-channel compartment, and hybrid packets.

Every packet has a representation contract (:data:`PACKET_CONTRACTS`), a
timestamp, a source/target, a target compartment and soma, and confidence /
uncertainty metadata.  The :class:`NeuroWaveBus` mechanism encodes the
substrate state into packets each step and fuses them back into neural
currents:

    I = sum_kind  gate[kind] * P[kind] @ resample(payload[kind], D)

``P[kind]`` is a fixed seeded sparse projection; ``gate`` is the fusion layer.
**All gates start at exactly 0**, so enabling the bus alone leaves the scalar
pathway mathematically unchanged; gates are then configured (``params["gates"]``)
or learned by reward-correlated gate perturbation (``learn_gates=True``).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..contracts import INTERFACE_VERSIONS, Mechanism, Provenance, SignalKind, Stage, StepContext, rng_state, set_rng_state

PACKET_VERSION = INTERFACE_VERSIONS["signal_packet"]
_HISTORY = 32
_VSA_DIM = 256


class RepresentationKind(str, enum.Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    SPECTRAL = "spectral"
    PHASE = "phase"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    GRAPH = "graph"
    SPARSE = "sparse"
    VSA = "vsa"
    EVENT = "event"
    POPULATION = "population"
    CONTEXT = "context"
    PREDICTIVE = "predictive"
    UNCERTAINTY = "uncertainty"
    COMPARTMENT = "compartment"
    HYBRID = "hybrid"


def _c(low: float, high: float, description: str, compartment: str = "apical", soma: str = "context") -> Dict[str, Any]:
    return {"version": PACKET_VERSION, "low": low, "high": high, "description": description,
            "compartment": compartment, "soma": soma}


#: Representation contract per kind: value range, default target compartment / soma.
PACKET_CONTRACTS: Dict[RepresentationKind, Dict[str, Any]] = {
    RepresentationKind.SCALAR: _c(0.0, 1.0, "global mean rate", "soma", "primary"),
    RepresentationKind.VECTOR: _c(0.0, 1.0, "per-region mean rates", "soma", "primary"),
    RepresentationKind.SPECTRAL: _c(0.0, 1.0, "normalised rFFT magnitude of the recent mean-rate history"),
    RepresentationKind.PHASE: _c(-1.0, 1.0, "cos/sin of each region's dominant oscillation phase"),
    RepresentationKind.TEMPORAL: _c(-1.0, 1.0, "per-region rate change at 1, 8 and 32 step scales", "apical", "slow"),
    RepresentationKind.SPATIAL: _c(0.0, 1.0, "activity centre and spread along each region"),
    RepresentationKind.GRAPH: _c(-1.0, 1.0, "signed inter-region signal flow matrix"),
    RepresentationKind.SPARSE: _c(0.0, 1.0, "k-winner occupancy histogram (sparse distributed code)", "soma", "fast"),
    RepresentationKind.VSA: _c(-1.0, 1.0, "bundled role-filler hypervector of region activity levels"),
    RepresentationKind.EVENT: _c(0.0, 1.0, "spike and onset event rates per region", "soma", "burst"),
    RepresentationKind.POPULATION: _c(0.0, 1.0, "Gaussian population code of the mean rate"),
    RepresentationKind.CONTEXT: _c(-1.0, 1.0, "context-soma state per region"),
    RepresentationKind.PREDICTIVE: _c(-1.0, 1.0, "prediction, surprise and per-region prediction error"),
    RepresentationKind.UNCERTAINTY: _c(0.0, 1.0, "ensemble uncertainty, rate dispersion and inverse confidence", "apical", "gating"),
    RepresentationKind.COMPARTMENT: _c(-1.0, 1.0, "multi-channel dendrite and soma compartment means", "apical", "integrator"),
    RepresentationKind.HYBRID: _c(-1.0, 1.0, "vector + phase + uncertainty parts in one packet"),
}


@dataclass
class SignalPacket:
    kind: RepresentationKind
    payload: np.ndarray
    source: str = "connectome"
    target: str = "*"
    compartment: str = "apical"          # apical | proximal | soma
    soma: str = "context"                # soma population the packet addresses
    step: int = 0
    timestamp: float = 0.0
    confidence: float = 1.0
    uncertainty: float = 0.0
    provenance: Optional[Provenance] = None      # None: internal neural state
    parts: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    version: str = PACKET_VERSION

    def validate(self) -> "SignalPacket":
        kind = RepresentationKind(self.kind)
        contract = PACKET_CONTRACTS[kind]
        payload = np.asarray(self.payload, dtype=np.float32).ravel()
        if payload.size == 0 or not np.all(np.isfinite(payload)):
            raise ValueError(f"{kind.value} packet payload must be non-empty and finite")
        if self.version.split(".")[0] != PACKET_VERSION.split(".")[0]:
            raise ValueError(f"incompatible packet version {self.version} (runtime {PACKET_VERSION})")
        if self.compartment not in ("apical", "proximal", "soma"):
            raise ValueError(f"unknown target compartment '{self.compartment}'")
        if not (0.0 <= self.confidence <= 1.0) or self.uncertainty < 0.0:
            raise ValueError("confidence must be in [0, 1] and uncertainty non-negative")
        self.kind = kind
        self.payload = np.clip(payload, contract["low"], contract["high"])
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": RepresentationKind(self.kind).value, "payload": np.asarray(self.payload).tolist(), "source": self.source,
            "target": self.target, "compartment": self.compartment, "soma": self.soma, "step": int(self.step),
            "timestamp": float(self.timestamp), "confidence": float(self.confidence), "uncertainty": float(self.uncertainty),
            "provenance": None if self.provenance is None else Provenance(self.provenance).value,
            "parts": {k: np.asarray(v).tolist() for k, v in self.parts.items()}, "meta": dict(self.meta), "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalPacket":
        prov = data.get("provenance")
        return cls(
            kind=RepresentationKind(data["kind"]), payload=np.asarray(data["payload"], dtype=np.float32),
            source=data.get("source", "external"), target=data.get("target", "*"), compartment=data.get("compartment", "apical"),
            soma=data.get("soma", "context"), step=int(data.get("step", 0)), timestamp=float(data.get("timestamp", 0.0)),
            confidence=float(data.get("confidence", 1.0)), uncertainty=float(data.get("uncertainty", 0.0)),
            provenance=None if prov is None else Provenance(prov),
            parts={k: np.asarray(v, dtype=np.float32) for k, v in data.get("parts", {}).items()},
            meta=dict(data.get("meta", {})), version=str(data.get("version", PACKET_VERSION)),
        ).validate()


# -- vector-symbolic helpers (bipolar hypervectors) ---------------------------
def vsa_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def vsa_bundle(vectors: List[np.ndarray]) -> np.ndarray:
    total = np.sum(vectors, axis=0)
    return np.where(total >= 0, 1.0, -1.0).astype(np.float32)


def vsa_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a * b))


def _resample(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == dim:
        return values
    if values.size == 1:
        return np.full(dim, values[0], dtype=np.float32)
    return np.interp(np.linspace(0, values.size - 1, dim), np.arange(values.size), values).astype(np.float32)


class NeuroWaveBus(Mechanism):
    mechanism_id = "neurowave_packets"
    stage = Stage.CONTEXT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {
            "kinds": [k.value for k in RepresentationKind],
            "feature_dim": 16, "fan": 4,
            "gates": {},                 # kind -> fixed fusion gate (0 = that representation is not fused)
            "learn_gates": False, "gate_lr": 0.02, "gate_sigma": 0.02, "gate_max": 0.5,
            "dt": 1.0,
        }

    # ------------------------------------------------------------------ setup
    def on_attach(self) -> None:
        sub, p = self.substrate, self.params
        self.kinds = [RepresentationKind(k) for k in p["kinds"]]
        self.regions = [int(r) for r in torch.unique(sub.region_ids).tolist()]
        self.region_index = [torch.nonzero(sub.region_ids == r, as_tuple=False).squeeze(-1) for r in self.regions]
        n, d = sub.total_neurons, int(p["feature_dim"])
        rng = np.random.default_rng(sub.weave.seed + 271828)
        self.projection: Dict[RepresentationKind, torch.Tensor] = {}
        for kind in self.kinds:
            proj = np.zeros((n, d), dtype=np.float32)
            cols = rng.integers(0, d, size=(n, int(p["fan"])))
            signs = rng.choice([-1.0, 1.0], size=cols.shape)
            np.put_along_axis(proj, cols, signs / np.sqrt(int(p["fan"])), axis=1)
            self.projection[kind] = torch.as_tensor(proj)
        self.gates = {kind: float(p["gates"].get(kind.value, 0.0)) for kind in self.kinds}
        self.roles = rng.choice([-1.0, 1.0], size=(len(self.regions), _VSA_DIM)).astype(np.float32)
        self.levels = rng.choice([-1.0, 1.0], size=(8, _VSA_DIM)).astype(np.float32)
        self.centres = np.linspace(0.0, 1.0, 12, dtype=np.float32)
        self.rng = np.random.default_rng(sub.weave.seed + 314159)
        self.reward_baseline = 0.0
        self.inbox: List[SignalPacket] = []
        self.last_packets: List[SignalPacket] = []
        self.last_delta: Dict[RepresentationKind, float] = {}
        self.sent = 0
        self.reset()

    def reset(self) -> None:
        self.history = np.zeros((_HISTORY, len(self.regions)), dtype=np.float32)
        self.filled = 0
        self.inbox = []

    # ----------------------------------------------------------------- inputs
    def receive(self, packet: SignalPacket) -> None:
        """Queue an externally produced packet (sensory hub, Micro-Collective, another model) for the next step."""
        self.inbox.append(packet.validate())

    def set_gate(self, kind, value: float) -> None:
        kind = RepresentationKind(kind)
        self.gates[kind] = float(np.clip(value, -self.params["gate_max"], self.params["gate_max"]))

    # --------------------------------------------------------------- encoding
    def _region_mean(self, x: torch.Tensor) -> np.ndarray:
        return np.array([float(x[idx].mean()) for idx in self.region_index], dtype=np.float32)

    def encode(self, ctx: StepContext) -> List[SignalPacket]:
        sub, cell, K = self.substrate, self.substrate.cell, RepresentationKind
        r = sub.r.detach()
        rates = self._region_mean(r)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = rates
        self.filled = min(_HISTORY, self.filled + 1)
        hist = self.history[-self.filled:]
        unc = max(0.0, float(sub.signals.uncertainty))
        base = dict(step=ctx.step, timestamp=ctx.step * float(self.params["dt"]), uncertainty=unc, confidence=1.0 / (1.0 + unc))
        payloads: Dict[RepresentationKind, Any] = {}
        want = set(self.kinds)

        if K.SCALAR in want:
            payloads[K.SCALAR] = np.array([float(r.mean())])
        if K.VECTOR in want or K.HYBRID in want:
            payloads[K.VECTOR] = rates
        if K.SPECTRAL in want:
            signal = hist.mean(axis=1) - hist.mean()
            mag = np.abs(np.fft.rfft(np.pad(signal, (0, _HISTORY - len(signal)))))
            payloads[K.SPECTRAL] = mag / (mag.max() + 1e-9)
        if K.PHASE in want or K.HYBRID in want:
            spec = np.fft.rfft(np.pad(hist - hist.mean(axis=0), ((0, _HISTORY - len(hist)), (0, 0)), mode="constant"), axis=0)
            dominant = np.argmax(np.abs(spec[1:]), axis=0) + 1
            phase = np.angle(spec[dominant, np.arange(spec.shape[1])])
            payloads[K.PHASE] = np.concatenate([np.cos(phase), np.sin(phase)])
        if K.TEMPORAL in want:
            scales = [hist[-1] - hist[-min(len(hist), s + 1):].mean(axis=0) for s in (1, 8, 32)]
            payloads[K.TEMPORAL] = np.clip(np.concatenate(scales) * 4.0, -1.0, 1.0)
        if K.SPATIAL in want:
            spatial = []
            for idx in self.region_index:
                x = r[idx]
                pos = torch.linspace(0.0, 1.0, len(idx))
                mass = float(x.sum()) + 1e-9
                centre = float((x * pos).sum()) / mass
                spread = float(torch.sqrt((x * (pos - centre) ** 2).sum() / mass))
                spatial += [centre, min(1.0, 2.0 * spread)]
            payloads[K.SPATIAL] = np.array(spatial)
        if K.GRAPH in want:
            pre, post = sub.edge_index()
            signal = sub.syn.last_signal
            nr = len(self.regions)
            remap = torch.zeros(int(sub.region_ids.max()) + 1, dtype=torch.long)
            remap[torch.as_tensor(self.regions)] = torch.arange(nr)
            if signal.numel() == pre.numel():
                flat = remap[sub.region_ids[pre]] * nr + remap[sub.region_ids[post]]
                flow = torch.zeros(nr * nr).scatter_add_(0, flat, signal)
            else:
                flow = torch.zeros(nr * nr)
            payloads[K.GRAPH] = np.tanh(flow.numpy())
        if K.SPARSE in want:
            k = max(1, sub.total_neurons // 20)
            winners = torch.topk(r, k).indices.numpy()
            payloads[K.SPARSE] = np.histogram(winners, bins=int(self.params["feature_dim"]), range=(0, sub.total_neurons))[0] / k
        if K.VSA in want:
            level = np.clip((rates * 16).astype(int), 0, 7)
            payloads[K.VSA] = vsa_bundle([vsa_bind(self.roles[i], self.levels[level[i]]) for i in range(len(rates))])
        if K.EVENT in want:
            spikes = self._region_mean(cell.spike)
            onset = (hist[-1] - hist[-2] > 0.02).astype(np.float32) if len(hist) > 1 else np.zeros_like(rates)
            payloads[K.EVENT] = np.concatenate([spikes, onset])
        if K.POPULATION in want:
            payloads[K.POPULATION] = np.exp(-0.5 * ((float(r.mean()) - self.centres) / 0.08) ** 2)
        if K.CONTEXT in want:
            payloads[K.CONTEXT] = np.tanh(self._region_mean(cell.soma_context))
        if K.PREDICTIVE in want:
            payloads[K.PREDICTIVE] = np.concatenate([[np.clip(sub.signals.prediction, -1, 1), min(1.0, sub.signals.surprise)],
                                                     np.tanh(self._region_mean(cell.pred_error))])
        if K.UNCERTAINTY in want or K.HYBRID in want:
            payloads[K.UNCERTAINTY] = np.array([min(1.0, unc), min(1.0, float(rates.std()) * 4.0), 1.0 - base["confidence"]])
        if K.COMPARTMENT in want:
            dend = [np.tanh(self._region_mean(x)) for x in (cell.prox, cell.dist, cell.apic, cell.basal)]
            soma = np.tanh(cell.soma_matrix(r).mean(dim=1).numpy())
            payloads[K.COMPARTMENT] = np.concatenate(dend + [soma])

        packets: List[SignalPacket] = []
        for kind in self.kinds:
            contract = PACKET_CONTRACTS[kind]
            common = dict(base, source="connectome", compartment=contract["compartment"], soma=contract["soma"],
                          provenance=Provenance.INTERNAL_PREDICTION if kind is K.PREDICTIVE else None)
            if kind is K.HYBRID:
                parts = {k.value: np.asarray(payloads[k], dtype=np.float32) for k in (K.VECTOR, K.PHASE, K.UNCERTAINTY)}
                packets.append(SignalPacket(kind, np.concatenate(list(parts.values())), parts=parts, **common).validate())
            elif kind in payloads:
                packets.append(SignalPacket(kind, np.asarray(payloads[kind], dtype=np.float32), **common).validate())
        return packets

    # ------------------------------------------------------------------ fusion
    def begin_step(self, ctx: StepContext) -> None:
        p = self.params
        packets = self.encode(ctx) + self.inbox
        self.inbox = []
        self.last_packets = packets
        self.sent += len(packets)
        ctx.packets.extend(packets)
        self.last_delta = {}
        if ctx.learning and p["learn_gates"]:            # reward-correlated gate perturbation (REINFORCE on the gates)
            self.last_delta = {k: float(self.rng.normal(0.0, p["gate_sigma"])) for k in self.kinds}
        fields = {"apical": ctx.apical_in, "proximal": ctx.sensory_extra, "soma": ctx.extra_in}
        for packet in packets:
            gate = self.gates.get(packet.kind, 0.0) + self.last_delta.get(packet.kind, 0.0)
            if gate == 0.0 or packet.kind not in self.projection:
                continue
            feature = torch.as_tensor(_resample(packet.payload, int(p["feature_dim"])))
            fields[packet.compartment] += gate * packet.confidence * (self.projection[packet.kind] @ feature)

    def end_step(self, ctx: StepContext) -> None:
        if not (ctx.learning and self.last_delta):
            return
        p = self.params
        advantage = float(ctx.signals.reward) - self.reward_baseline
        self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * float(ctx.signals.reward)
        for kind, delta in self.last_delta.items():
            step = p["gate_lr"] * advantage * delta / (p["gate_sigma"] ** 2 + 1e-12) * p["gate_sigma"]
            self.gates[kind] = float(np.clip(self.gates[kind] + step, -p["gate_max"], p["gate_max"]))

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"gates": {k.value: v for k, v in self.gates.items()}, "reward_baseline": self.reward_baseline,
                "history": torch.as_tensor(self.history), "filled": int(self.filled), "sent": int(self.sent), "rng": rng_state(self.rng)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name, value in state.get("gates", {}).items():
            self.gates[RepresentationKind(name)] = float(value)
        self.reward_baseline = float(state.get("reward_baseline", 0.0))
        if "history" in state:
            self.history = np.asarray(state["history"], dtype=np.float32).copy()
            self.filled = int(state.get("filled", 0))
        self.sent = int(state.get("sent", 0))
        set_rng_state(self.rng, state.get("rng"))

    def telemetry(self) -> Dict[str, Any]:
        return {"packets_last_step": len(self.last_packets), "packets_total": int(self.sent),
                "kinds": [k.value for k in self.kinds],
                "gates": {k.value: round(v, 5) for k, v in self.gates.items()},
                "active_gates": int(sum(1 for v in self.gates.values() if v != 0.0)),
                "payload_dims": {pk.kind.value: int(np.asarray(pk.payload).size) for pk in self.last_packets}}


__all__ = ["PACKET_VERSION", "PACKET_CONTRACTS", "RepresentationKind", "SignalPacket", "NeuroWaveBus",
           "vsa_bind", "vsa_bundle", "vsa_similarity"]
