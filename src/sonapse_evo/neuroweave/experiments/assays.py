"""Topology and lesion assays.

``run_topology_assay``  compares connectome families at matched size, seeds, mechanisms, environment and
                        episode schedule, and reports graph metrics, held-out return, robustness to a matched
                        random lesion, and measured cost.  It makes no assumption that a richer graph wins.
``run_lesion_assay``    compares intact, targeted and matched-random lesions (same number of synapses removed)
                        on clones of one substrate per seed, with optional recovery training and a
                        progressive-damage curve.  The intact substrate is never modified.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...contracts.evidence import EvidenceTier
from ..connectome import (LESION_KINDS, TOPOLOGY_KINDS, apply_lesion_mask, build_lesion_mask, build_topology, compute_graph_metrics,
                          lesion_receipt, matched_random_mask, progressive_schedule)
from ..contracts import INTERFACE_VERSIONS, WEAVE_VERSION
from ..ecology import make_environment
from ..runtime import build_weave_substrate, canonical_motor_head, run_weave_episode
from .framework import HELD_OUT_OFFSET, MIN_PAIRED_SEEDS, _digest, compare, episode_efficiency, resolve_metric

_GRAPH_KEYS = ("edges", "density", "clustering_coefficient", "modularity", "characteristic_path_length", "global_efficiency",
               "hub_concentration_top5pct", "regional_communication_efficiency", "reciprocity")


def _evaluate(substrate, env, motor, seed: int, episodes: int, learn: bool = False) -> Tuple[float, float, int, float, float]:
    """Mean return, measured ms/step, steps, success rate and goal efficiency over ``episodes`` episodes."""
    substrate.freeze_learning(not learn)
    returns, steps, elapsed, wins, eff = [], 0, 0.0, 0, []
    for k in range(episodes):
        start = time.perf_counter()
        result = run_weave_episode(env, substrate, motor, seed=seed + (0 if learn else HELD_OUT_OFFSET) + k)
        elapsed += time.perf_counter() - start
        steps += result.steps
        returns.append(result.total_return)
        wins += int(result.success)
        eff.append(episode_efficiency(result.success, result.steps, getattr(env, "max_steps", None)))
    return float(np.mean(returns)), 1000.0 * elapsed / max(1, steps), steps, wins / max(1, episodes), float(np.mean(eff))


def run_topology_assay(kinds: Sequence[str] = ("legacy", "clustered_small_world", "hub_skewed", "uniform_random", "degree_shuffled",
                                              "erdos_renyi", "hub_capped", "ring_lattice", "hierarchical_modular", "multi_region",
                                              "multiscale_regional", "hybrid"),
                       seeds: Sequence[int] = (11, 23, 37, 51), total_neurons: int = 768, mechanisms: Sequence[str] = (),
                       environment: str = "forage", env_options: Optional[Dict[str, Any]] = None, episodes: int = 2,
                       lesion_severity: float = 0.2, reference: str = "legacy") -> Dict[str, Any]:
    unknown = [k for k in kinds if k not in TOPOLOGY_KINDS]
    if unknown:
        raise KeyError(f"unknown topology kinds: {unknown}")
    motor = canonical_motor_head()
    rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in kinds}
    for kind in kinds:
        for seed in seeds:
            topo = build_topology(kind, total_neurons=total_neurons, seed=seed)
            metrics = compute_graph_metrics(topo, seed=seed, max_sources=64)
            substrate = build_weave_substrate(seed=seed, topology=topo, enable=tuple(mechanisms))
            env = make_environment(environment, seed=seed, **(env_options or {}))
            intact, ms, steps, success, intact_eff = _evaluate(substrate, env, motor, seed, episodes)
            damaged = substrate.clone_substrate()
            apply_lesion_mask(damaged, build_lesion_mask(topo, "random_edges", lesion_severity, seed=seed + 5))
            lesioned, _, _, lesioned_success, lesioned_eff = _evaluate(damaged, env, motor, seed, episodes)
            rows[kind].append({"seed": int(seed), "heldout_return": intact, "lesioned_return": lesioned,
                               "success_rate": success, "lesioned_success_rate": lesioned_success, "mean_steps": steps / max(1, episodes),
                               "efficiency": intact_eff, "lesioned_efficiency": lesioned_eff,
                               "lesion_retention": lesioned / intact if abs(intact) > 1e-9 else float("nan"),
                               "measured_ms_per_step": ms, "graph": {k: metrics[k] for k in _GRAPH_KEYS},
                               "in_degree_max": metrics["in_degree"]["max"], "in_degree_cv": metrics["in_degree"]["cv"],
                               "out_degree_cv": metrics["out_degree"]["cv"]})
    summary = {}
    for kind, records in rows.items():
        ret = [r["heldout_return"] for r in records]
        summary[kind] = {"heldout_mean": float(np.mean(ret)), "heldout_std": float(np.std(ret)),
                         "success_rate": float(np.mean([r["success_rate"] for r in records])),
                         "efficiency_mean": float(np.mean([r["efficiency"] for r in records])),
                         "lesioned_efficiency_mean": float(np.mean([r["lesioned_efficiency"] for r in records])),
                         "lesioned_success_rate": float(np.mean([r["lesioned_success_rate"] for r in records])),
                         "mean_steps": float(np.mean([r["mean_steps"] for r in records])),
                         "lesion_retention_mean": float(np.nanmean([r["lesion_retention"] for r in records])),
                         "measured_ms_per_step": float(np.mean([r["measured_ms_per_step"] for r in records])),
                         "edges_mean": float(np.mean([r["graph"]["edges"] for r in records])),
                         "in_degree_max_mean": float(np.mean([r["in_degree_max"] for r in records])),
                         "in_degree_cv_mean": float(np.mean([r["in_degree_cv"] for r in records])),
                         **{f"{k}_mean": float(np.mean([r["graph"][k] for r in records])) for k in _GRAPH_KEYS[2:]}}
    metric = resolve_metric("auto", environment)
    field = "efficiency" if metric == "efficiency" else "heldout_return"
    comparisons = {}
    if reference in rows:
        ref = [r[field] for r in rows[reference]]
        comparisons = {f"{kind}_vs_{reference}": dict(compare([r[field] for r in rows[kind]], ref), metric=metric)
                       for kind in kinds if kind != reference}
    ranking = sorted(kinds, key=lambda k: -summary[k]["heldout_mean"])
    ranking_eff = sorted(kinds, key=lambda k: (-summary[k]["efficiency_mean"], -summary[k]["heldout_mean"]))
    tier = EvidenceTier.PAIRED_EXPERIMENT if len(seeds) >= MIN_PAIRED_SEEDS else EvidenceTier.SIMULATED_ASSAY
    return {"receipt_version": INTERFACE_VERSIONS["experiment_receipt"], "weave_version": WEAVE_VERSION, "experiment_id": "topology_optimisation",
            "kind": "topology", "title": "Topology assay", "seeds": [int(s) for s in seeds], "total_neurons": int(total_neurons),
            "mechanisms": list(mechanisms), "environment": environment, "episodes": int(episodes), "lesion_severity": float(lesion_severity),
            "matched_conditions": {"same_size_seeds_mechanisms_environment_schedule": True, "edge_counts_differ": True,
                                   "note": "edge count and cost are reported per topology, not equalised"},
            "primary_metric": metric,
            "results": {"summary": summary, "comparisons": comparisons, "runs": rows, "ranking_by_heldout_return": ranking,
                        "ranking_by_efficiency": ranking_eff},
            "verdict": "descriptive", "evidence_tier": tier.value, "created_unix": time.time(),
            "result_digest": _digest({k: [[r["seed"], r["heldout_return"], r["lesioned_return"]] for r in v] for k, v in rows.items()}),
            "disclaimer": "synthetic graphs and a synthetic task; a ranking here is not a claim about biological wiring"}


def run_lesion_assay(conditions: Sequence[str] = ("hub_targeted", "region_targeted", "sensory_pathway", "memory_pathway",
                                                 "cross_region_communication"),
                     seeds: Sequence[int] = (11, 23, 37), severity: float = 0.5, topology_kind: str = "multi_region",
                     total_neurons: int = 768, mechanisms: Sequence[str] = (), environment: str = "forage",
                     env_options: Optional[Dict[str, Any]] = None, episodes: int = 2, recovery_episodes: int = 0,
                     progressive_steps: int = 0, params: Optional[Dict[str, Dict[str, Any]]] = None,
                     protect_canonical: bool = False) -> Dict[str, Any]:
    unknown = [c for c in conditions if c not in LESION_KINDS]
    if unknown:
        raise KeyError(f"unknown lesion kinds: {unknown}")
    motor = canonical_motor_head()
    protect = 3 if protect_canonical else 0          # by default a lesion may hit the canonical sensorimotor pathway
    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        topo = build_topology(topology_kind, total_neurons=total_neurons, seed=seed)
        intact_sub = build_weave_substrate(seed=seed, topology=topo, enable=tuple(mechanisms), params=params)
        env = make_environment(environment, seed=seed, **(env_options or {}))
        intact, _, intact_steps, intact_success, intact_eff = _evaluate(intact_sub, env, motor, seed, episodes)
        record: Dict[str, Any] = {"seed": int(seed), "intact_return": intact, "intact_success_rate": intact_success,
                                  "intact_efficiency": intact_eff,
                                  "intact_mean_steps": intact_steps / max(1, episodes), "conditions": {}}
        for kind in conditions:
            mask = build_lesion_mask(topo, kind, severity, seed=seed + 7, protect_first=protect)
            removed = int((mask == 0).sum())
            control = matched_random_mask(topo, removed, seed=seed + 8, protect_first=protect)
            entry: Dict[str, Any] = {"receipt": lesion_receipt(kind, severity, seed + 7, mask), "edges_removed": removed}
            for label, m in (("targeted", mask), ("matched_random", control)):
                damaged = intact_sub.clone_substrate()
                apply_lesion_mask(damaged, m)
                value, _, n_steps, succ, eff = _evaluate(damaged, env, motor, seed, episodes)
                item = {"return": value, "success_rate": succ, "mean_steps": n_steps / max(1, episodes), "efficiency": eff,
                        "retention": value / intact if abs(intact) > 1e-9 else float("nan"),
                        "functional_redundancy": float(np.clip(value / intact, 0.0, 1.0)) if abs(intact) > 1e-9 else float("nan")}
                if recovery_episodes > 0:
                    before = float(damaged.syn.w_delta.abs().sum())
                    curve = [_evaluate(damaged, env, motor, seed + 100 + j, 1, learn=True)[0] for j in range(recovery_episodes)]
                    after, _, _, after_success, after_eff = _evaluate(damaged, env, motor, seed, episodes)
                    hit = [j + 1 for j, _ in enumerate(curve) if after >= 0.9 * intact]
                    item.update(recovery_curve=curve, post_recovery_return=after, post_recovery_success_rate=after_success,
                                post_recovery_efficiency=after_eff, recovered_fraction=after / intact if abs(intact) > 1e-9 else float("nan"),
                                recovery_episodes_to_90pct=hit[0] if hit else None,
                                connectivity_change={"learned_weight_change": float(damaged.syn.w_delta.abs().sum()) - before,
                                                     "grown_synapses": int(damaged.syn.n_grown)})
                entry[label] = item
            entry["targeted_minus_random"] = entry["targeted"]["return"] - entry["matched_random"]["return"]
            record["conditions"][kind] = entry
        if progressive_steps > 0:
            curve = []
            for sev in progressive_schedule(progressive_steps, max_severity=severity):
                damaged = intact_sub.clone_substrate()
                apply_lesion_mask(damaged, build_lesion_mask(topo, "random_edges", sev, seed=seed + 9, protect_first=protect))
                ret, _, _, succ, eff = _evaluate(damaged, env, motor, seed, episodes)
                curve.append({"severity": sev, "return": ret, "success_rate": succ, "efficiency": eff})
            record["progressive_random"] = curve
        runs.append(record)
    summary, comparisons = {}, {}
    for kind in conditions:
        targeted = [r["conditions"][kind]["targeted"]["return"] for r in runs]
        random_ = [r["conditions"][kind]["matched_random"]["return"] for r in runs]
        summary[kind] = {"targeted_mean": float(np.mean(targeted)), "matched_random_mean": float(np.mean(random_)),
                         "intact_mean": float(np.mean([r["intact_return"] for r in runs])),
                         "intact_success_rate": float(np.mean([r["intact_success_rate"] for r in runs])),
                         "targeted_success_rate": float(np.mean([r["conditions"][kind]["targeted"]["success_rate"] for r in runs])),
                         "random_success_rate": float(np.mean([r["conditions"][kind]["matched_random"]["success_rate"] for r in runs])),
                         "targeted_mean_steps": float(np.mean([r["conditions"][kind]["targeted"]["mean_steps"] for r in runs])),
                         "intact_efficiency": float(np.mean([r["intact_efficiency"] for r in runs])),
                         "targeted_efficiency": float(np.mean([r["conditions"][kind]["targeted"]["efficiency"] for r in runs])),
                         "random_efficiency": float(np.mean([r["conditions"][kind]["matched_random"]["efficiency"] for r in runs])),
                         "targeted_retention": float(np.nanmean([r["conditions"][kind]["targeted"]["retention"] for r in runs])),
                         "random_retention": float(np.nanmean([r["conditions"][kind]["matched_random"]["retention"] for r in runs])),
                         "edges_removed_mean": float(np.mean([r["conditions"][kind]["edges_removed"] for r in runs]))}
        comparisons[f"{kind}_targeted_vs_matched_random"] = dict(compare(targeted, random_), metric="return")
        # damage comparison: "supported" means the targeted lesion impairs MORE than a matched random lesion
        damage_metric = resolve_metric("auto", environment)
        pick = (lambda e: e["efficiency"]) if damage_metric == "efficiency" else (lambda e: e["return"])
        comparisons[f"{kind}_targeted_damage_exceeds_random"] = dict(
            compare([pick(r["conditions"][kind]["matched_random"]) for r in runs], [pick(r["conditions"][kind]["targeted"]) for r in runs]),
            metric=damage_metric)
    tier = EvidenceTier.PAIRED_EXPERIMENT if len(seeds) >= MIN_PAIRED_SEEDS else EvidenceTier.SIMULATED_ASSAY
    return {"receipt_version": INTERFACE_VERSIONS["experiment_receipt"], "weave_version": WEAVE_VERSION, "experiment_id": "lesion_robustness",
            "kind": "lesion", "title": "Lesion robustness assay", "seeds": [int(s) for s in seeds], "severity": float(severity),
            "topology_kind": topology_kind, "total_neurons": int(total_neurons), "mechanisms": list(mechanisms),
            "environment": environment, "episodes": int(episodes), "recovery_episodes": int(recovery_episodes),
            "matched_conditions": {"random_control_removes_same_edge_count": True, "intact_substrate_untouched": True,
                                   "canonical_pathway_protected": bool(protect_canonical)},
            "results": {"summary": summary, "comparisons": comparisons, "runs": runs}, "verdict": "descriptive",
            "evidence_tier": tier.value, "created_unix": time.time(),
            "result_digest": _digest([[r["seed"], r["intact_return"], {k: [v["targeted"]["return"], v["matched_random"]["return"]]
                                                                       for k, v in r["conditions"].items()}] for r in runs]),
            "interpretation_note": "the existing forage task pays reward every step, so a slower agent can score a higher return; "
                                   "read success_rate and mean_steps alongside return",
            "disclaimer": "synthetic lesions on a synthetic connectome; not a model of biological injury or recovery"}


__all__ = ["run_topology_assay", "run_lesion_assay"]
