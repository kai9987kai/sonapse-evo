"""Typed, versioned interface contracts for the NeuroWeave evolution layer.

Everything in ``sonapse_evo.neuroweave`` is *additive*: the original
``RecurrentSubstrate`` scalar pathway is never edited.  Subsystems talk to
each other only through the objects defined here:

* :class:`StepContext`   - the per-environment-step blackboard that mechanisms
  read from and write neutral-initialised contributions into.
* :class:`Mechanism`     - the plug-in base class every switchable mechanism
  derives from.
* :class:`SharedSignals` - slowly varying control state (neuromodulators,
  affect, uncertainty, reward) that persists across steps.
* :class:`Provenance` / :class:`SignalKind` - enumerations used to label where
  information came from and what role a signal plays.

Neutral-initialisation rule (regression contract)
-------------------------------------------------
Every field of :class:`StepContext` that a mechanism may write starts at the
value that makes it a mathematical no-op (additive terms at exactly ``0``,
multiplicative terms at exactly ``1`` or ``None``).  A mechanism that is
disabled never runs, so the all-disabled configuration reproduces the legacy
update bit-for-bit.

These are computational abstractions.  Biological vocabulary (dendrite, soma,
dopamine, ...) names the *role* of a state variable; none of it is a validated
biophysical model and none of it implies subjective experience.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .substrate import WeaveSubstrate

WEAVE_VERSION = "3.0.0"

#: Version of each cross-subsystem interface.  Bump the relevant entry whenever
#: a field is removed or its meaning changes; adding optional fields is a
#: minor bump.  ``portability`` records these in every exported package.
INTERFACE_VERSIONS: Dict[str, str] = {
    "mechanism": "1.0",
    "step_context": "1.0",
    "shared_signals": "1.0",
    "synapse_state": "1.0",
    "cell_state": "1.0",
    "signal_packet": "2.0",      # NeuroWave-X2 packets
    "sensory_manifest": "1.0",
    "experiment_receipt": "1.0",
    "model_package": "3.0",
    "event": "1.0",
}


class Stage(enum.IntEnum):
    """Execution order of mechanism hooks inside one environment step."""

    SENSE = 10       # sensory encoding / fusion
    MODULATE = 20    # neuromodulators, affect, gains
    CONTEXT = 30     # attention, packets, memory retrieval, predictive feedback
    TRANSMIT = 40    # edge-level routing (core)
    INTEGRATE = 50   # multi-compartment cell (core)
    LEARN = 60       # plasticity, memory writes, model updates
    STRUCTURE = 70   # structural plasticity, neurogenesis, consolidation
    REPORT = 80      # telemetry only


class Provenance(str, enum.Enum):
    """Where a piece of information originated (memory / sensory provenance)."""

    DIRECT_SENSORY = "direct_sensory"
    INTERNAL_PREDICTION = "internal_prediction"
    EXTERNAL_SUPPLIED = "external_supplied"
    PRIOR_EXPERIENCE = "prior_experience"
    SYNTHETIC_SIMULATION = "synthetic_simulation"

    @property
    def is_observation(self) -> bool:
        """True only for information that was actually observed."""
        return self in (Provenance.DIRECT_SENSORY, Provenance.EXTERNAL_SUPPLIED)


class SignalKind(str, enum.Enum):
    """Role of a signal on the event bus / in a packet."""

    INPUT = "input"
    STATE_UPDATE = "state_update"
    PREDICTION = "prediction"
    MODULATORY = "modulatory"
    LEARNING = "learning"
    ACTION = "action"
    MEMORY_RETRIEVAL = "memory_retrieval"
    TELEMETRY = "telemetry"


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


@dataclass
class ModulatorState:
    """Six neuromodulator-like global control fields.

    Defaults are the *neutral* values: with these the modulators contribute
    nothing to gating, gain or plasticity.
    """

    dopamine: float = 0.0          # [-1, 1]  reward-prediction-error-like
    serotonin: float = 0.5         # [0, 1]   behavioural inhibition / patience
    acetylcholine: float = 0.15    # [0, 1]   expected uncertainty / attention
    noradrenaline: float = 0.0     # [0, 1]   arousal / gain
    octopamine: float = 0.0        # [0, 1]   insect arousal / surprise
    endocannabinoid: float = 0.0   # [0, 1]   retrograde suppression

    NAMES = ("dopamine", "serotonin", "acetylcholine", "noradrenaline", "octopamine", "endocannabinoid")
    BOUNDS = {
        "dopamine": (-1.0, 1.0),
        "serotonin": (0.0, 1.0),
        "acetylcholine": (0.0, 1.0),
        "noradrenaline": (0.0, 1.0),
        "octopamine": (0.0, 1.0),
        "endocannabinoid": (0.0, 1.0),
    }

    def clamp(self) -> "ModulatorState":
        for name in self.NAMES:
            lo, hi = self.BOUNDS[name]
            setattr(self, name, _clip(getattr(self, name), lo, hi))
        return self

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.NAMES}

    def load(self, values: Dict[str, float]) -> "ModulatorState":
        for name in self.NAMES:
            if name in values:
                setattr(self, name, float(values[name]))
        return self.clamp()

    def reset(self) -> None:
        neutral = ModulatorState()
        for name in self.NAMES:
            setattr(self, name, getattr(neutral, name))

    def is_neutral(self, tol: float = 0.0) -> bool:
        neutral = ModulatorState()
        return all(abs(getattr(self, n) - getattr(neutral, n)) <= tol for n in self.NAMES)

    # -- adapters to the pre-existing four-field Sonapse-EVO modulator state --
    def to_legacy(self):
        """Project onto ``substrate.plasticity.NeuromodulatoryState`` (existing API)."""
        from ..substrate.plasticity import NeuromodulatoryState

        return NeuromodulatoryState(
            dopamine_rpe=float(self.dopamine),
            acetylcholine_unc=float(self.acetylcholine),
            serotonin_risk=float(self.serotonin),
            noradrenaline_gain=float(1.0 + self.noradrenaline),
        )

    @classmethod
    def from_legacy(cls, legacy) -> "ModulatorState":
        state = cls(
            dopamine=float(legacy.dopamine_rpe),
            acetylcholine=float(legacy.acetylcholine_unc),
            serotonin=float(legacy.serotonin_risk),
            noradrenaline=float(legacy.noradrenaline_gain - 1.0),
        )
        return state.clamp()


@dataclass
class AffectState:
    """Operational affect-like control variables (not subjective emotion)."""

    valence: float = 0.0                   # [-1, 1]
    arousal: float = 0.15                  # [0, 1]
    curiosity: float = 0.2                 # [0, 1]
    stress: float = 0.0                    # [0, 1]
    motivation: float = 0.5                # [0, 1]
    novelty: float = 0.0                   # [0, 1]
    reward_prediction_error: float = 0.0   # [-1, 1]

    NAMES = ("valence", "arousal", "curiosity", "stress", "motivation", "novelty", "reward_prediction_error")
    BOUNDS = {
        "valence": (-1.0, 1.0),
        "arousal": (0.0, 1.0),
        "curiosity": (0.0, 1.0),
        "stress": (0.0, 1.0),
        "motivation": (0.0, 1.0),
        "novelty": (0.0, 1.0),
        "reward_prediction_error": (-1.0, 1.0),
    }

    def clamp(self) -> "AffectState":
        for name in self.NAMES:
            lo, hi = self.BOUNDS[name]
            setattr(self, name, _clip(getattr(self, name), lo, hi))
        return self

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.NAMES}

    def load(self, values: Dict[str, float]) -> "AffectState":
        for name in self.NAMES:
            if name in values:
                setattr(self, name, float(values[name]))
        return self.clamp()

    def reset(self) -> None:
        neutral = AffectState()
        for name in self.NAMES:
            setattr(self, name, getattr(neutral, name))


@dataclass
class FeedbackSignal:
    """Outcome information delivered by the agent loop *after* an action.

    It is consumed on the following step (lagged), so no mechanism can use the
    current target before the prediction for it has been scored.
    """

    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    intended_action: Optional[Any] = None    # what the controller meant to do
    issued_action: Optional[Any] = None      # the motor command actually sent
    predicted_outcome: Optional[Any] = None  # forward-model prediction
    observed_outcome: Optional[Any] = None   # what the environment reported


@dataclass
class SharedSignals:
    """Slow control state shared by all mechanisms of one substrate."""

    modulators: ModulatorState = field(default_factory=ModulatorState)
    affect: AffectState = field(default_factory=AffectState)
    reward: float = 0.0             # lagged reward visible to this step
    uncertainty: float = 0.0        # ensemble disagreement proxy, >= 0
    prediction: float = 0.0         # last scalar prediction of mean activity
    surprise: float = 0.0           # mean absolute local prediction error
    plasticity_gain: float = 1.0    # global multiplier on learning rates
    exploration: float = 0.0        # suggested action-noise scale (0 = none)
    persistence: float = 0.0        # suggested action momentum (0 = none)
    attention_gain: float = 1.0     # global sensory attention multiplier
    control_reliability: float = 1.0  # self-model estimate of actuator reliability
    feedback: FeedbackSignal = field(default_factory=FeedbackSignal)
    pending: Optional[FeedbackSignal] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modulators": self.modulators.as_dict(),
            "affect": self.affect.as_dict(),
            "reward": float(self.reward),
            "uncertainty": float(self.uncertainty),
            "prediction": float(self.prediction),
            "surprise": float(self.surprise),
            "plasticity_gain": float(self.plasticity_gain),
            "exploration": float(self.exploration),
            "persistence": float(self.persistence),
            "attention_gain": float(self.attention_gain),
            "control_reliability": float(self.control_reliability),
            "extra": {k: float(v) for k, v in self.extra.items()},
        }

    def load(self, values: Dict[str, Any]) -> None:
        self.modulators.load(values.get("modulators", {}))
        self.affect.load(values.get("affect", {}))
        for name in ("reward", "uncertainty", "prediction", "surprise", "plasticity_gain",
                     "exploration", "persistence", "attention_gain", "control_reliability"):
            if name in values:
                setattr(self, name, float(values[name]))
        self.extra = {k: float(v) for k, v in values.get("extra", {}).items()}

    @staticmethod
    def _feedback_dict(fb: Optional[FeedbackSignal]) -> Optional[Dict[str, Any]]:
        if fb is None:
            return None

        def arr(value):
            return None if value is None else torch.as_tensor(value, dtype=torch.float32).reshape(-1).tolist()

        return {"reward": float(fb.reward), "done": bool(fb.done), "intended_action": arr(fb.intended_action),
                "issued_action": arr(fb.issued_action), "predicted_outcome": arr(fb.predicted_outcome),
                "observed_outcome": arr(fb.observed_outcome)}

    def feedback_state(self) -> Dict[str, Any]:
        """Serialisable copy of the visible and the pending (lagged) feedback."""
        return {"feedback": self._feedback_dict(self.feedback), "pending": self._feedback_dict(self.pending)}

    def load_feedback_state(self, state: Dict[str, Any]) -> None:
        def build(data):
            if data is None:
                return None
            return FeedbackSignal(reward=float(data.get("reward", 0.0)), done=bool(data.get("done", False)),
                                  intended_action=data.get("intended_action"), issued_action=data.get("issued_action"),
                                  predicted_outcome=data.get("predicted_outcome"), observed_outcome=data.get("observed_outcome"))

        self.feedback = build(state.get("feedback")) or FeedbackSignal()
        self.pending = build(state.get("pending"))

    def reset_transient(self) -> None:
        """Clear per-episode signals; slow learned control state is kept."""
        self.reward = 0.0
        self.feedback = FeedbackSignal()
        self.pending = None


class StepContext:
    """Blackboard for one environment step.

    Mechanisms *add* into the neutral-initialised fields during ``begin_step``;
    the core then transmits, integrates the multi-compartment cell, and calls
    ``end_step`` for learning and structural updates.

    Additive fields (start at exact zero)
        ``apical_in``      current routed to the apical dendrite (top-down
                           context, attention, predictions)
        ``extra_in``       additional somatic current
        ``sensory_extra``  additional feed-forward current routed to the
                           proximal dendrite (e.g. extra sensory channels)
    Multiplicative fields (``None`` means exactly 1)
        ``gain``           per-neuron (or scalar) gain on the output activation
        ``sensory_gain``   gain on the legacy sensory projection ``B @ obs``
        ``edge_gain``      per-edge transmission gain
        ``context_weight`` per-neuron weighting of presynaptic sources for the
                           structural context message
    Scalars
        ``gating_drive``   target of the gating soma (1 = neutral)
        ``threshold_shift`` additive shift of the spike threshold (0 = neutral)
        ``leak_scale``     multiplier on the integration leak (1 = neutral)
        ``transmitter_gain`` per-neuron release gain (None = 1)
    """

    __slots__ = (
        "substrate", "observation", "step", "microstep", "signals",
        "r_prev", "sens_in", "rec_in", "ctx_in", "aux_in", "bias", "edge_signal",
        "apical_in", "extra_in", "sensory_extra",
        "gain", "sensory_gain", "edge_gain", "context_weight", "transmitter_gain",
        "gating_drive", "threshold_shift", "leak_scale",
        "packets", "memory", "predictions", "scratch", "telemetry",
        "drive", "fused", "r_new", "learning",
    )

    def __init__(self, substrate: "WeaveSubstrate", observation: torch.Tensor, step: int):
        n = substrate.total_neurons
        device = substrate.r.device
        self.substrate = substrate
        self.observation = observation
        self.step = int(step)
        self.microstep = 0
        self.signals: SharedSignals = substrate.signals

        self.r_prev: Optional[torch.Tensor] = None
        self.sens_in: Optional[torch.Tensor] = None
        self.rec_in: Optional[torch.Tensor] = None
        self.ctx_in: Optional[torch.Tensor] = None
        self.aux_in: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        self.edge_signal: Optional[torch.Tensor] = None

        self.apical_in = torch.zeros(n, dtype=torch.float32, device=device)
        self.extra_in = torch.zeros(n, dtype=torch.float32, device=device)
        self.sensory_extra = torch.zeros(n, dtype=torch.float32, device=device)

        self.gain: Optional[torch.Tensor] = None
        self.sensory_gain: Optional[torch.Tensor] = None
        self.edge_gain: Optional[torch.Tensor] = None
        self.context_weight: Optional[torch.Tensor] = None
        self.transmitter_gain: Optional[torch.Tensor] = None

        self.gating_drive: Any = 1.0
        self.threshold_shift: Any = 0.0
        self.leak_scale: float = 1.0

        self.packets: List[Any] = []
        self.memory: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
        self.scratch: Dict[str, Any] = {}
        self.telemetry: Dict[str, Any] = {}

        self.drive: Optional[torch.Tensor] = None
        self.fused: Optional[torch.Tensor] = None
        self.r_new: Optional[torch.Tensor] = None
        self.learning: bool = bool(substrate.learning_enabled)

    # -- helpers that keep multiplicative fields composable -----------------
    def scale_gain(self, factor) -> None:
        """Multiply the output gain by ``factor`` (tensor of shape (N,) or scalar)."""
        self.gain = factor if self.gain is None else self.gain * factor

    def scale_sensory_gain(self, factor) -> None:
        self.sensory_gain = factor if self.sensory_gain is None else self.sensory_gain * factor

    def scale_edge_gain(self, factor) -> None:
        """Multiply per-edge transmission gain by ``factor`` (shape (E,) or scalar)."""
        self.edge_gain = factor if self.edge_gain is None else self.edge_gain * factor

    def scale_transmitter_gain(self, factor) -> None:
        self.transmitter_gain = factor if self.transmitter_gain is None else self.transmitter_gain * factor

    def scale_gating_drive(self, factor) -> None:
        self.gating_drive = self.gating_drive * factor


class Mechanism:
    """Base class of every switchable NeuroWeave mechanism.

    Subclasses set ``mechanism_id`` (must match the registry) and override the
    hooks they need.  Contract:

    * ``begin_step`` may only *add to / scale* the neutral fields of ``ctx``
      and update ``ctx.signals``; it must never write substrate rates.
    * ``end_step`` performs learning.  It must honour ``ctx.learning`` (when
      ``False`` the mechanism keeps its effect but changes no learned state).
    * All learned state must be returned by ``state_dict`` using only python
      scalars, lists, dicts, strings and torch tensors, and restored exactly by
      ``load_state_dict``.
    * A freshly attached mechanism whose learned parameters are still at their
      initial values must be a no-op wherever the docstring says "zero-init".
    """

    mechanism_id: str = ""
    stage: Stage = Stage.CONTEXT
    #: Optional stage used to order ``end_step`` only (defaults to ``stage``).  A mechanism that acts early in a
    #: step but must observe what later stages produced (e.g. memory storing predictions) sets this.
    end_stage: Optional[Stage] = None
    interface_version: str = INTERFACE_VERSIONS["mechanism"]

    def __init__(self, substrate: "WeaveSubstrate", params: Optional[Dict[str, Any]] = None):
        self.substrate = substrate
        merged = dict(self.default_params())
        merged.update(params or {})
        self.params: Dict[str, Any] = merged

    # -- configuration ------------------------------------------------------
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {}

    # -- lifecycle ----------------------------------------------------------
    def on_attach(self) -> None:
        """Called once after construction, when the substrate is fully built."""

    def on_detach(self) -> None:
        """Called when the mechanism is disabled at runtime."""

    def reset(self) -> None:
        """Clear transient (per-episode) state.  Learned state must survive."""

    def on_topology_change(self) -> None:
        """Edges were added / removed; per-edge caches must be rebuilt."""

    # -- per-step hooks -----------------------------------------------------
    def begin_step(self, ctx: StepContext) -> None:
        """Once per environment step, before transmission and integration."""

    def end_step(self, ctx: StepContext) -> None:
        """Once per environment step, after integration (learning / structure)."""

    # -- persistence & observability ----------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None

    def telemetry(self) -> Dict[str, Any]:
        return {}

    # -- convenience --------------------------------------------------------
    def enabled(self, mechanism_id: str) -> bool:
        return self.substrate.weave.is_enabled(mechanism_id)

    def peer(self, mechanism_id: str) -> Optional["Mechanism"]:
        """Return another active mechanism instance, or ``None`` (graceful degradation)."""
        return self.substrate.mechanisms.get(mechanism_id)


def rng_state(rng) -> Dict[str, Any]:
    """Serialisable state of a ``numpy.random.Generator`` (so saved models continue the same random stream)."""
    return dict(rng.bit_generator.state)


def set_rng_state(rng, state: Optional[Dict[str, Any]]) -> None:
    if state:
        rng.bit_generator.state = state


__all__ = [
    "rng_state",
    "set_rng_state",
    "WEAVE_VERSION",
    "INTERFACE_VERSIONS",
    "Stage",
    "Provenance",
    "SignalKind",
    "ModulatorState",
    "AffectState",
    "FeedbackSignal",
    "SharedSignals",
    "StepContext",
    "Mechanism",
]
