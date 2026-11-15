"""Catalogue of the twenty-four independently switchable NeuroWeave experiments.

Each entry is a hypothesis with a matched baseline, not a result.  Run one with
``run_catalog_experiment("stdp")`` or ``sonapse-evo weave-experiment stdp``.  The existing SONAPSE-EVO
experiments (smoke, lesion-repair, benchmark, benchmark-v2, discover, criticality, curriculum) are
untouched and remain available through their own commands.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .framework import ExperimentSpec, run_experiment

_ECO = "neuroweave_1200x660"
_SHORT = {"max_steps": 80}


def _spec(eid, title, hypothesis, candidate, baseline=(), **kw) -> ExperimentSpec:
    return ExperimentSpec(experiment_id=eid, title=title, hypothesis=hypothesis, candidate=tuple(candidate), baseline=tuple(baseline), **kw)


EXPERIMENTS: Dict[str, ExperimentSpec] = {s.experiment_id: s for s in (
    _spec("multimodal_sensory_integration", "Multimodal sensory integration",
          "Sixteen ecology-derived sensory channels improve held-out foraging return over the 155-dim observation alone.",
          ("multimodal_sensory", "cross_modal_fusion"), environment=_ECO, env_options=_SHORT, ablations=("cross_modal_fusion",)),
    _spec("affective_neuromodulation", "Affective neuromodulation",
          "Neuromodulator fields and affective control improve reward-gated learning over reward gating alone.",
          ("reward_gated_plasticity", "neuromodulatory_field", "affective_control"), ("reward_gated_plasticity",),
          ablations=("affective_control",)),
    _spec("micro_collective_learning", "Micro-Collective learning",
          "Predictive sensory context from a trained Micro-Collective improves control in the ecology.",
          ("multimodal_sensory", "micro_collective", "collective_meta_learning"), ("multimodal_sensory",),
          environment=_ECO, env_options=_SHORT),
    _spec("expert_routing", "Expert routing",
          "Balanced top-2 expert routing improves on single-expert routing.",
          ("multimodal_sensory", "micro_collective", "collective_meta_learning", "expert_routing"),
          ("multimodal_sensory", "micro_collective", "collective_meta_learning"), environment=_ECO, env_options=_SHORT),
    _spec("meta_plasticity", "Meta-plasticity",
          "Adaptive per-synapse learning rates improve reward-gated learning.",
          ("reward_gated_plasticity", "meta_plasticity"), ("reward_gated_plasticity",)),
    _spec("predictive_replay", "Predictive replay",
          "Surprise-prioritised replay improves consolidation over sleep consolidation without it.",
          ("reward_gated_plasticity", "sleep_consolidation", "predictive_replay"), ("reward_gated_plasticity", "sleep_consolidation"),
          params={"sleep_consolidation": {"period": 40}, "predictive_replay": {"period": 20}}),
    _spec("connectivity_context_attention", "Connectivity-context attention",
          "Attention over strong neighbours, delivered to apical dendrites, improves control.",
          ("structural_encoder", "connectivity_attention", "dendritic_compartments"), ("dendritic_compartments",),
          ablations=("structural_encoder",)),
    _spec("flyhash_sparse_projections", "FlyHash-style sparse projections",
          "Sparse expansion tags with k-winner coding improve control.", ("flyhash_projection", "sparse_distributed_code"),
          ablations=("flyhash_projection",)),
    _spec("stdp", "Pair-based STDP", "Spike-timing plasticity on the membrane model improves held-out return.",
          ("membrane_spiking", "pair_stdp"), ("membrane_spiking",)),
    _spec("adaptive_conduction", "Adaptive conduction",
          "Activity-dependent myelin shortening recovers performance lost to synaptic delays.",
          ("synaptic_delays", "adaptive_conduction"), ("synaptic_delays",)),
    _spec("quantised_synapses", "Quantised synapses", "INT8-like weights do not reduce held-out return (non-inferiority check).",
          ("quantised_synapses",)),
    _spec("hierarchical_temporal_memory", "Hierarchical temporal memory",
          "Fast, medium and slow traces help when the goal cue disappears.", ("multi_soma", "hierarchical_temporal_memory"),
          ("multi_soma",), environment="memory_navigation", env_options=_SHORT),
    _spec("uncertainty_gated_propagation", "Uncertainty-gated propagation",
          "Attenuating recurrence under ensemble disagreement improves control.",
          ("predictive_world_model", "uncertainty_gating"), ("predictive_world_model",)),
    _spec("predictive_coding", "Local predictive coding", "Local prediction residuals improve control.",
          ("multi_soma", "local_predictive_coding"), ("multi_soma",)),
    _spec("structural_plasticity", "Structural plasticity",
          "Utility-guided pruning and regrowth improves on fixed wiring at matched synapse count.",
          ("reward_gated_plasticity", "structural_plasticity"), ("reward_gated_plasticity",),
          params={"structural_plasticity": {"period": 16}}),
    _spec("associative_memory", "Associative memory",
          "Episodic recall, fast weights and the memory ecology improve control.",
          ("associative_trace_memory", "fast_weights", "memory_ecology"), ablations=("fast_weights", "memory_ecology")),
    _spec("action_grounded_self_modelling", "Action-grounded self-modelling",
          "Discounting credit for disturbed outcomes improves learning under wind and actuator slip.",
          ("reward_gated_plasticity", "efference_self_model"), ("reward_gated_plasticity",), environment="disturbed", env_options=_SHORT),
    _spec("multi_soma_integration", "Multi-soma and multi-compartment integration",
          "The full compartment model (dendrites, soma populations, membrane, axon) improves control over point neurons.",
          ("multi_soma", "dendritic_compartments", "membrane_spiking", "axonal_branches"),
          ablations=("dendritic_compartments", "axonal_branches")),
    _spec("multi_region_routing", "Multi-region routing",
          "Strengthening sensory-to-association and memory-to-action pathways improves control on the ten-region connectome.",
          ("multi_region_routing",), topology_kind="multi_region",
          params={"multi_region_routing": {"pathways": {"sensory_to_association": 1.25, "memory_to_action": 1.25}}}),
    _spec("lesion_robustness", "Lesion robustness", "Targeted lesions impair more than matched random lesions.", (), kind="lesion"),
    _spec("topology_optimisation", "Topology optimisation",
          "Some topology family yields higher held-out return at matched size and seeds.", (), kind="topology"),
    _spec("neural_grafting", "Neural grafting (specialist)",
          "A separately trained specialist graft improves control once its held-out skill is positive.",
          ("multimodal_sensory", "specialist_graft"), ("multimodal_sensory",), environment=_ECO, env_options=_SHORT),
    _spec("cross_modal_learning", "Cross-modal learning",
          "Cross-modal association keeps performance when one modality is disabled.",
          ("multimodal_sensory", "cross_modal_fusion"), ("multimodal_sensory",), environment=_ECO, env_options=_SHORT,
          params={"multimodal_sensory": {"modes": {"olfactory": "disabled"}}}),
    _spec("specialist_expert_evolution", "Specialist expert evolution",
          "A six-expert collective outperforms the four-expert default.",
          ("multimodal_sensory", "micro_collective", "collective_meta_learning", "expert_routing"),
          ("multimodal_sensory", "micro_collective", "collective_meta_learning"), environment=_ECO, env_options=_SHORT,
          params={"micro_collective": {"experts": 6}}, notes="candidate also enables balanced routing; see expert_routing for that factor alone"),
)}

EXPERIMENT_IDS = tuple(EXPERIMENTS)


def list_experiments() -> List[Dict[str, Any]]:
    return [{"id": s.experiment_id, "title": s.title, "hypothesis": s.hypothesis, "kind": s.kind, "environment": s.environment,
             "candidate": list(s.candidate), "baseline": list(s.baseline), "ablations": list(s.ablations),
             "topology": s.topology_kind or "legacy"} for s in EXPERIMENTS.values()]


def run_catalog_experiment(experiment_id: str, seeds: Optional[Iterable[int]] = None, quick: bool = False, **overrides: Any) -> Dict[str, Any]:
    """Run a catalogued experiment.  ``quick`` shrinks it to a smoke-sized run (reported as such)."""
    if experiment_id not in EXPERIMENTS:
        raise KeyError(f"unknown experiment '{experiment_id}'. Available: {EXPERIMENT_IDS}")
    spec = EXPERIMENTS[experiment_id]
    if spec.kind == "topology":
        from .assays import run_topology_assay

        return run_topology_assay(seeds=tuple(seeds) if seeds is not None else ((11, 23) if quick else (11, 23, 37, 51)),
                                  episodes=1 if quick else 2, **overrides)
    if spec.kind == "lesion":
        from .assays import run_lesion_assay

        return run_lesion_assay(seeds=tuple(seeds) if seeds is not None else ((11,) if quick else (11, 23, 37)), **overrides)
    fields = {**spec.__dict__, **overrides}
    if quick:
        fields.update(episodes_train=1, episodes_eval=1, seeds=tuple(seeds) if seeds is not None else (11, 23), smoke=True)
    return run_experiment(ExperimentSpec(**fields), seeds=seeds)


__all__ = ["EXPERIMENTS", "EXPERIMENT_IDS", "list_experiments", "run_catalog_experiment"]
