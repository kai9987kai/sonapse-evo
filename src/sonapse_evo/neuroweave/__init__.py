"""SONAPSE-EVO NeuroWeave evolution layer (additive, default-off).

A second research layer built *next to* the SONAPSE-EVO 2.0 architecture.  It
adapts the mechanisms of NEUROWEAVE-LAB 0.4.0 (multi-compartment neurons,
multi-region connectomes, sixteen-channel sensing, affective neuromodulation,
the Micro-Collective, predictive processing, memory, structural plasticity,
hardware-inspired execution and model portability) to the SONAPSE-EVO
substrate and extends them.  Nothing here changes the behaviour of existing
code: every mechanism is an independent switch that starts disabled.

Sub-packages are imported lazily so that an unavailable optional subsystem can
never break ``import sonapse_evo``.
"""
from __future__ import annotations

from .contracts import (
    INTERFACE_VERSIONS, WEAVE_VERSION, AffectState, FeedbackSignal, Mechanism, ModulatorState, Provenance,
    SharedSignals, SignalKind, Stage, StepContext,
)
from .events import Event, EventBus
from .registry import (
    GROUPS, MECHANISM_IDS, MECHANISMS, NEUROWEAVE_FEATURE_IDS, NEUROWEAVE_FEATURE_MAP, PROFILES, MechanismSpec,
    WeaveConfig, describe_catalogue, mechanisms_in_group,
)
from .cells import CELL_FIELDS, DENDRITES, SOMA_ROLES, CellParams, WeaveCell
from .substrate import EDGE_FIELDS, SynapseState, WeaveSubstrate
from .runtime import EpisodeResult, build_weave_substrate, canonical_motor_head, run_weave_episode, upgrade_substrate

__all__ = [
    "WEAVE_VERSION", "INTERFACE_VERSIONS", "Stage", "Provenance", "SignalKind", "ModulatorState", "AffectState",
    "FeedbackSignal", "SharedSignals", "StepContext", "Mechanism", "Event", "EventBus",
    "MechanismSpec", "MECHANISMS", "MECHANISM_IDS", "GROUPS", "PROFILES", "WeaveConfig",
    "NEUROWEAVE_FEATURE_IDS", "NEUROWEAVE_FEATURE_MAP", "describe_catalogue", "mechanisms_in_group",
    "CellParams", "WeaveCell", "CELL_FIELDS", "SOMA_ROLES", "DENDRITES",
    "WeaveSubstrate", "SynapseState", "EDGE_FIELDS",
    "build_weave_substrate", "upgrade_substrate", "canonical_motor_head", "EpisodeResult", "run_weave_episode",
]
