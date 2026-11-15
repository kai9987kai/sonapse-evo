"""Mechanism registry, configuration profiles and switchable-feature catalogue.

Every NeuroWeave mechanism is declared here exactly once.  A mechanism is

* ``core``     - implemented inside the cell / transmission code of
  :class:`~sonapse_evo.neuroweave.substrate.WeaveSubstrate` and gated by its flag,
* a plug-in    - a :class:`~sonapse_evo.neuroweave.contracts.Mechanism`
  subclass loaded lazily from ``impl`` (``"package.module:ClassName"``), or
* a modifier   - a flag consumed by another mechanism (``impl is None``).

All mechanisms default to **disabled**: the ``legacy`` profile is the original
SONAPSE-EVO 2.0 behaviour, bit for bit.  Nothing is promoted into a default
profile without evidence (see ``neuroweave.promotion``).
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .contracts import Stage

_PKG = "sonapse_evo.neuroweave"


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    title: str
    group: str                       # cell | transmission | representation | learning | structure | modulation | memory | prediction | sensory | collective | execution | connectome
    stage: Stage
    summary: str
    impl: Optional[str] = None       # "core", "pkg.module:Class" or None (modifier flag)
    neuroweave_feature: Optional[str] = None   # switch id in NEUROWEAVE-LAB 0.4.0, if derived from one
    requires: Tuple[str, ...] = ()   # soft dependencies: missing peers degrade gracefully
    status: str = "experimental"     # experimental | promoted
    neutral_when: str = ""           # what makes it an exact no-op while enabled

    @property
    def is_core(self) -> bool:
        return self.impl == "core"

    @property
    def is_modifier(self) -> bool:
        return self.impl is None

    @property
    def is_plugin(self) -> bool:
        return self.impl not in (None, "core")

    def load_class(self):
        if not self.is_plugin:
            raise ValueError(f"{self.mechanism_id} has no plug-in implementation")
        module_name, _, class_name = self.impl.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


def _m(mid, title, group, stage, summary, impl=None, nw=None, requires=(), neutral=""):
    return MechanismSpec(mid, title, group, stage, summary, impl, nw, tuple(requires), "experimental", neutral)


_CATALOGUE: Tuple[MechanismSpec, ...] = (
    # ---- multi-compartment cell (core) ------------------------------------
    _m("multi_soma", "Multi-soma integration", "cell", Stage.INTEGRATE,
       "Primary, secondary, fast, slow, context, integrator, burst and gating soma populations re-mix the synaptic drive across timescales.",
       "core", "multiSoma", neutral="steady inputs: every temporal-contrast term vanishes"),
    _m("dendritic_compartments", "Dendritic compartments", "cell", Stage.INTEGRATE,
       "Proximal (feed-forward), distal (recurrent), apical (context) and basal branches with local time constants, saturation and apical gating.",
       "core", neutral="disabled: currents sum at the soma exactly as in the legacy drive"),
    _m("membrane_spiking", "Membrane, adaptive threshold and refractoriness", "cell", Stage.INTEGRATE,
       "Membrane proxy, activity-dependent threshold, refractory period, spike events and an intracellular calcium proxy.",
       "core"),
    _m("axonal_branches", "Axon hillock, branches and terminal release", "cell", Stage.INTEGRATE,
       "Hillock initiation, two or more axonal branches, terminal release with a depleting resource, and branch-specific synaptic transmission.",
       "core", "axonBranches"),
    _m("homeostatic_governor", "Local homeostatic gain", "cell", Stage.INTEGRATE,
       "Per-neuron gain drifts to hold a target rate, bounded to [0.58, 1.12].",
       "core", "glial", neutral="gain starts at exactly 1"),
    _m("astrocyte_regulation", "Astrocyte-like regulation", "cell", Stage.INTEGRATE,
       "Slow local reservoir tracking neighbourhood activity scales the output gain and incoming synaptic efficacy.",
       "core", "glial", neutral="only on its first update: the gain starts at exactly 1 and then adapts to activity"),
    _m("sparse_distributed_code", "Sparse distributed code (k-winners)", "representation", Stage.INTEGRATE,
       "Global or per-region k-winners: non-winning units are attenuated to 8% of their activation.",
       "core", "sparse"),
    _m("hierarchical_temporal_memory", "Hierarchical temporal memory traces", "memory", Stage.INTEGRATE,
       "Fast, medium and slow per-neuron activity traces inject a bounded temporal contrast into fusion.",
       "core", "hierarchicalMemory", neutral="traces start at zero"),
    _m("local_predictive_coding", "Local predictive coding", "prediction", Stage.INTEGRATE,
       "Each unit predicts its fused drive, tracks bounded error and surprise, and injects a surprise-normalised residual.",
       "core", "predictiveCoding"),
    # ---- transmission (core) ----------------------------------------------
    _m("synaptic_delays", "Synaptic delay lines", "transmission", Stage.TRANSMIT,
       "Per-edge extra conduction delay of 0-3 steps read from a rate history buffer; long-range edges are slower.",
       "core", "delays"),
    _m("adaptive_conduction", "Adaptive conduction (myelin proxy)", "transmission", Stage.TRANSMIT,
       "Activity-dependent myelin proxy shortens the delay of well-used axons.",
       "core", "myelination", requires=("synaptic_delays",)),
    _m("context_adaptive_routing", "Context-adaptive routing", "transmission", Stage.TRANSMIT,
       "Destination gating-soma and apical state scale each edge within [0.62, 1.14]; centred so the resting gain is exactly 1.",
       "core", "adaptiveRouting", neutral="gating soma at 1 and silent apical dendrite"),
    _m("uncertainty_gating", "Uncertainty-gated propagation", "transmission", Stage.TRANSMIT,
       "Ensemble disagreement attenuates recurrent transmission within [0.58, 1].",
       "core", "uncertaintyGate", neutral="uncertainty signal equal to 0"),
    # ---- representation / communication (plug-ins) ------------------------
    _m("structural_encoder", "Structural encoder", "representation", Stage.CONTEXT,
       "Sixteen-feature graph fingerprints per neuron weight the context message by source degree and reciprocity.",
       f"{_PKG}.comm.structural:StructuralEncoder", "structural"),
    _m("connectivity_attention", "Connectivity-context attention", "representation", Stage.CONTEXT,
       "Soft attention over the eight strongest active incoming neighbours feeds the apical dendrite.",
       f"{_PKG}.comm.attention:ConnectivityAttention", "wiretype"),
    _m("flyhash_projection", "FlyHash-style sparse projection", "representation", Stage.CONTEXT,
       "Fixed sparse random expansion of the observation with winner-take-all tagging feeds the apical dendrite.",
       f"{_PKG}.comm.flyhash:FlyHashProjection", "flyHash"),
    _m("neurowave_packets", "NeuroWave-X2 multi-representation packets", "representation", Stage.CONTEXT,
       "Typed, versioned scalar/vector/spectral/phase/sparse/VSA/event/population packets fused into neural currents through a zero-initialised fusion layer.",
       f"{_PKG}.comm.neurowave:NeuroWaveBus", neutral="fusion read-out weights start at zero"),
    _m("multi_region_routing", "Multi-region routing", "connectome", Stage.CONTEXT,
       "Region-pair routing gains over sensory, association, memory and motor pathways with feedback and predictive feedback routes.",
       f"{_PKG}.connectome.routing:MultiRegionRouting", neutral="all region-pair gains equal to 1"),
    # ---- learning ----------------------------------------------------------
    _m("reward_gated_plasticity", "Reward-gated three-factor plasticity", "learning", Stage.LEARN,
       "Sparse per-edge eligibility traces gated by lagged reward and neuromodulators; sign and magnitude bounded.",
       f"{_PKG}.plasticity.reward_gated:RewardGatedPlasticity", "plasticity", neutral="zero reward"),
    _m("pair_stdp", "Pair-based STDP", "learning", Stage.LEARN,
       "Pre/post spike traces produce bounded potentiation and depression; source sign is fixed.",
       f"{_PKG}.plasticity.stdp:PairSTDP", "stdp", requires=("membrane_spiking",)),
    _m("meta_plasticity", "Meta-plasticity", "learning", Stage.LEARN,
       "Sliding BCM-like thresholds and per-synapse adaptive learning-rate scales with consolidation-dependent protection.",
       f"{_PKG}.plasticity.metaplasticity:MetaPlasticity", neutral="per-synapse rate scale starts at 1"),
    _m("sleep_consolidation", "Sleep-like consolidation and downscaling", "learning", Stage.STRUCTURE,
       "Periodic offline phases replay stored activity, consolidate useful synapses and weakly downscale the rest.",
       f"{_PKG}.plasticity.consolidation:SleepConsolidation", "sleep"),
    _m("quantised_synapses", "Quantised synapses (INT8-like)", "execution", Stage.TRANSMIT,
       "Signed 127-level magnitude quantisation applied to effective weights at read time.",
       f"{_PKG}.hardware.quantization:QuantisedSynapses", "quantized"),
    _m("tiled_execution", "Tiled execution", "execution", Stage.TRANSMIT,
       "Tile-grouped edge traversal with synchronous previous/next buffers and operation accounting.",
       f"{_PKG}.hardware.tiled:TiledExecution", "tiled"),
    # ---- structure ---------------------------------------------------------
    _m("structural_plasticity", "Utility-guided structural plasticity", "structure", Stage.STRUCTURE,
       "Bounded pruning of low-utility synapses and co-activity-guided regrowth with an audit ledger.",
       f"{_PKG}.structure.structural_plasticity:StructuralPlasticity", "neurogenesis"),
    _m("neurogenesis", "Structural neurogenesis", "structure", Stage.STRUCTURE,
       "Recruits reserve neurons, repurposes under-used ones, assigns them to regions and wires them in under growth budgets.",
       f"{_PKG}.structure.neurogenesis:Neurogenesis"),
    # ---- modulation --------------------------------------------------------
    _m("neuromodulatory_field", "Neuromodulatory field", "modulation", Stage.MODULATE,
       "Dopamine, serotonin, acetylcholine, noradrenaline, octopamine and endocannabinoid-like fields with heterogeneous receptor profiles.",
       f"{_PKG}.neuromod.field:NeuromodulatoryField", "neuromod"),
    _m("affective_control", "Affective control state", "modulation", Stage.MODULATE,
       "Valence, arousal, curiosity, stress and motivation as operational control variables driving exploration, thresholds, persistence and learning.",
       f"{_PKG}.neuromod.affect:AffectiveControl", "affectiveSignals"),
    _m("transmitter_inference", "Connectivity-only transmitter hypotheses", "modulation", Stage.MODULATE,
       "Six-class transmitter hypothesis with confidence from graph features; never relabels the source sign.",
       f"{_PKG}.neuromod.transmitters:TransmitterInference", "wireTypeInference", requires=("structural_encoder",)),
    # ---- memory ------------------------------------------------------------
    _m("associative_trace_memory", "Associative trace memory", "memory", Stage.CONTEXT,
       "Bounded similarity-retrieved episodes re-inject a fraction of the stored state.",
       f"{_PKG}.memory.associative:AssociativeTraceMemory", "memory", neutral="empty store"),
    _m("fast_weights", "Universal fast weights", "memory", Stage.CONTEXT,
       "Reward-updated key/value association matrix read into the drive with a bounded coefficient.",
       f"{_PKG}.memory.fast_weights:FastWeights", "fastWeights", neutral="matrix starts at zero"),
    _m("memory_ecology", "Memory ecology with provenance", "memory", Stage.CONTEXT,
       "Working, episodic, semantic and predictive stores with sparse indexing, confidence, interference tracking and provenance.",
       f"{_PKG}.memory.ecology:MemoryEcology", neutral="empty stores"),
    _m("predictive_replay", "Predictive replay", "memory", Stage.STRUCTURE,
       "Novelty- and error-prioritised replay scheduling for consolidation and for the Micro-Collective.",
       f"{_PKG}.memory.replay:PredictiveReplay", "predictiveReplay"),
    # ---- prediction --------------------------------------------------------
    _m("predictive_world_model", "Predictive world model", "prediction", Stage.LEARN,
       "Small ensemble predicts pooled state; member disagreement is the uncertainty signal.",
       f"{_PKG}.predictive.world_model:PredictiveWorldModel", "worldModel"),
    _m("hierarchical_prediction", "Hierarchical prediction horizons", "prediction", Stage.LEARN,
       "Immediate, short, medium and long horizon predictors with an uncertainty-adaptive active horizon.",
       f"{_PKG}.predictive.horizons:HierarchicalPrediction"),
    _m("efference_self_model", "Action-grounded self-model", "prediction", Stage.LEARN,
       "Compares intended, issued, predicted and observed action outcomes to estimate authorship, disturbance, slip and control reliability.",
       f"{_PKG}.predictive.self_model:EfferenceSelfModel", "selfModel"),
    # ---- sensory -----------------------------------------------------------
    _m("multimodal_sensory", "Multimodal sensory integration", "sensory", Stage.SENSE,
       "Sixteen sensory channels (visual, auditory, olfactory, gustatory, thermal, mechano, proprio, interoceptive) projected into the connectome.",
       f"{_PKG}.sensory.hub:MultimodalSensory", "multimodalSensory", neutral="all channels at zero"),
    _m("cross_modal_fusion", "Cross-modal fusion", "sensory", Stage.SENSE,
       "Reliability-weighted fusion, sensory prediction error, missing-sensor handling and cross-modal association.",
       f"{_PKG}.sensory.fusion:CrossModalFusion", requires=("multimodal_sensory",)),
    # ---- collective --------------------------------------------------------
    _m("micro_collective", "Micro-Collective", "collective", Stage.CONTEXT,
       "Local mixture of small experts predicting the next sensory state with replay, held-out validation and rollback.",
       f"{_PKG}.collective.mechanism:MicroCollectiveMechanism", "microCollective", neutral="untrained: predictive context gain is zero"),
    _m("expert_routing", "Balanced expert routing", "collective", Stage.CONTEXT,
       "Top-k routing with load balancing (modifier of the Micro-Collective).",
       None, "expertRouting", requires=("micro_collective",)),
    _m("collective_meta_learning", "Collective meta-learning", "collective", Stage.CONTEXT,
       "Held-out validation, best-state snapshots, rollback and learning-rate decay (modifier of the Micro-Collective).",
       None, "metaLearning", requires=("micro_collective",)),
    _m("specialist_graft", "Supermix-inspired specialist graft", "collective", Stage.CONTEXT,
       "Independently trained specialist model exchanging predictions with the connectome through an explicit, removable interface.",
       f"{_PKG}.collective.specialist:SpecialistGraft", neutral="graft gain starts at zero"),
)

MECHANISMS: Dict[str, MechanismSpec] = {spec.mechanism_id: spec for spec in _CATALOGUE}
MECHANISM_IDS: Tuple[str, ...] = tuple(MECHANISMS)

#: NEUROWEAVE-LAB 0.4.0 switch id -> Sonapse mechanism ids that account for it.
NEUROWEAVE_FEATURE_MAP: Dict[str, Tuple[str, ...]] = {}
for _spec in _CATALOGUE:
    if _spec.neuroweave_feature:
        NEUROWEAVE_FEATURE_MAP.setdefault(_spec.neuroweave_feature, ())
        NEUROWEAVE_FEATURE_MAP[_spec.neuroweave_feature] += (_spec.mechanism_id,)

NEUROWEAVE_FEATURE_IDS: Tuple[str, ...] = (
    "multiSoma", "structural", "sparse", "memory", "plasticity", "delays", "glial", "tiled", "wiretype",
    "flyHash", "neuromod", "neurogenesis", "sleep", "worldModel", "myelination", "quantized", "axonBranches",
    "fastWeights", "selfModel", "predictiveCoding", "stdp", "adaptiveRouting", "hierarchicalMemory",
    "uncertaintyGate", "multimodalSensory", "affectiveSignals", "microCollective", "expertRouting",
    "metaLearning", "predictiveReplay", "wireTypeInference",
)

GROUPS: Tuple[str, ...] = (
    "cell", "transmission", "representation", "connectome", "learning", "structure", "modulation",
    "memory", "prediction", "sensory", "collective", "execution",
)


def mechanisms_in_group(group: str) -> Tuple[str, ...]:
    return tuple(s.mechanism_id for s in _CATALOGUE if s.group == group)


def _nw_equivalent() -> Tuple[str, ...]:
    out: List[str] = []
    for feature in NEUROWEAVE_FEATURE_IDS:
        for mid in NEUROWEAVE_FEATURE_MAP.get(feature, ()):
            if mid not in out:
                out.append(mid)
    return tuple(out)


#: Named configuration profiles.  ``legacy`` is and stays the default.
PROFILES: Dict[str, Tuple[str, ...]] = {
    "legacy": (),
    "cells": mechanisms_in_group("cell"),
    "transmission": mechanisms_in_group("transmission"),
    "learning": mechanisms_in_group("learning"),
    "modulation": mechanisms_in_group("modulation"),
    "memory": mechanisms_in_group("memory"),
    "prediction": mechanisms_in_group("prediction"),
    "sensory": mechanisms_in_group("sensory"),
    "collective": mechanisms_in_group("collective"),
    "neuroweave": _nw_equivalent(),          # the 31 NEUROWEAVE-LAB switches, adapted
    "max": MECHANISM_IDS,                    # MAX mode: every mechanism enabled
}


@dataclass
class WeaveConfig:
    """Which mechanisms are on, with what parameters.  Serialisable and hashable."""

    enabled: Dict[str, bool] = field(default_factory=dict)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    seed: int = 42
    weave_current_cap: float = 0.50     # |somatic extra|, |apical| and |extra sensory| bound per neuron
    learning_enabled: bool = True
    strict: bool = False                # True: mechanism faults raise; False: degrade gracefully
    profile: str = "legacy"

    def __post_init__(self):
        unknown = [k for k in self.enabled if k not in MECHANISMS]
        if unknown:
            raise KeyError(f"Unknown mechanism id(s): {unknown}")
        full = {mid: False for mid in MECHANISM_IDS}
        full.update({k: bool(v) for k, v in self.enabled.items()})
        self.enabled = full

    # -- constructors -------------------------------------------------------
    @classmethod
    def legacy(cls, seed: int = 42) -> "WeaveConfig":
        return cls(seed=seed, profile="legacy")

    @classmethod
    def from_profile(cls, name: str, seed: int = 42, **kwargs) -> "WeaveConfig":
        if name not in PROFILES:
            raise KeyError(f"Unknown profile '{name}'. Available: {sorted(PROFILES)}")
        return cls(enabled={mid: True for mid in PROFILES[name]}, seed=seed, profile=name, **kwargs)

    @classmethod
    def with_mechanisms(cls, *mechanism_ids: str, seed: int = 42, **kwargs) -> "WeaveConfig":
        return cls(enabled={mid: True for mid in mechanism_ids}, seed=seed, profile="custom", **kwargs)

    # -- queries ------------------------------------------------------------
    def is_enabled(self, mechanism_id: str) -> bool:
        return self.enabled.get(mechanism_id, False)

    def any_enabled(self) -> bool:
        return any(self.enabled.values())

    def enabled_ids(self) -> Tuple[str, ...]:
        return tuple(mid for mid in MECHANISM_IDS if self.enabled[mid])

    def params_for(self, mechanism_id: str) -> Dict[str, Any]:
        return dict(self.params.get(mechanism_id, {}))

    # -- mutation (returns self for chaining) -------------------------------
    def enable(self, *mechanism_ids: str) -> "WeaveConfig":
        for mid in mechanism_ids:
            if mid not in MECHANISMS:
                raise KeyError(f"Unknown mechanism id: {mid}")
            self.enabled[mid] = True
        self.profile = "custom"
        return self

    def disable(self, *mechanism_ids: str) -> "WeaveConfig":
        for mid in mechanism_ids:
            if mid not in MECHANISMS:
                raise KeyError(f"Unknown mechanism id: {mid}")
            self.enabled[mid] = False
        self.profile = "custom" if self.any_enabled() else "legacy"
        return self

    def set_params(self, mechanism_id: str, **params: Any) -> "WeaveConfig":
        if mechanism_id not in MECHANISMS and mechanism_id != "cell":
            raise KeyError(f"Unknown mechanism id: {mechanism_id}")
        self.params.setdefault(mechanism_id, {}).update(params)
        return self

    def copy(self) -> "WeaveConfig":
        return WeaveConfig(
            enabled=dict(self.enabled), params=copy.deepcopy(self.params), seed=self.seed,
            weave_current_cap=self.weave_current_cap, learning_enabled=self.learning_enabled,
            strict=self.strict, profile=self.profile,
        )

    def without(self, *mechanism_ids: str) -> "WeaveConfig":
        """Ablation helper: a copy with the given mechanisms switched off."""
        return self.copy().disable(*mechanism_ids)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": dict(self.enabled),
            "params": copy.deepcopy(self.params),
            "seed": int(self.seed),
            "weave_current_cap": float(self.weave_current_cap),
            "learning_enabled": bool(self.learning_enabled),
            "strict": bool(self.strict),
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaveConfig":
        enabled = {k: bool(v) for k, v in data.get("enabled", {}).items() if k in MECHANISMS}
        return cls(
            enabled=enabled,
            params=copy.deepcopy(data.get("params", {})),
            seed=int(data.get("seed", 42)),
            weave_current_cap=float(data.get("weave_current_cap", 0.50)),
            learning_enabled=bool(data.get("learning_enabled", True)),
            strict=bool(data.get("strict", False)),
            profile=str(data.get("profile", "custom")),
        )

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def describe_catalogue() -> List[Dict[str, Any]]:
    """JSON-friendly description of every mechanism (used by the CLI and observatory)."""
    return [
        {
            "id": s.mechanism_id,
            "title": s.title,
            "group": s.group,
            "stage": s.stage.name,
            "summary": s.summary,
            "kind": "core" if s.is_core else ("modifier" if s.is_modifier else "plugin"),
            "neuroweave_feature": s.neuroweave_feature,
            "requires": list(s.requires),
            "status": s.status,
            "neutral_when": s.neutral_when,
        }
        for s in _CATALOGUE
    ]


__all__ = [
    "MechanismSpec", "MECHANISMS", "MECHANISM_IDS", "GROUPS", "PROFILES", "WeaveConfig",
    "NEUROWEAVE_FEATURE_IDS", "NEUROWEAVE_FEATURE_MAP", "mechanisms_in_group", "describe_catalogue",
]
