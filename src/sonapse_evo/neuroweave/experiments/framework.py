"""Controlled experiment framework: matched baseline / candidate / ablation arms with receipts.

For every seed each arm is built from the SAME topology, seed, environment and episode schedule; arms differ
only in which mechanisms are enabled.  The protocol per arm and seed is

    train   ``episodes_train`` episodes with learning enabled                      (seeds  s, s+1, ...)
    freeze  learning is switched off (mechanisms keep their effect)
    test    ``episodes_eval`` episodes on HELD-OUT environment seeds               (seeds  s+1000, ...)

Reported per arm: held-out return, held-out goal efficiency, success rate, steps, training return, measured
wall-clock per step, measured state bytes, synapse counts and mechanism faults.  The candidate is compared
with the baseline using the existing ``evaluation.statistical_analysis.paired_seed_analysis`` (paired
differences, effect size, bootstrap CI).

Primary metric (``ExperimentSpec.primary_metric``)
    return       total held-out episode return
    efficiency   goal efficiency per episode = success * (1 - (steps - 1) / max_steps), in [0, 1]
    auto         (default) ``efficiency`` on the existing ``forage`` task, ``return`` elsewhere.  The forage
                 task pays a positive reward on every step, so a slower agent that still reaches the goal
                 collects MORE return; return is therefore not a valid measure of competence there.
The other metric is always reported as a secondary comparison.

Verdict rule (never based on complexity or neuron count):
    supported      CI lower bound > 0 and at least ``MIN_PAIRED_SEEDS`` paired seeds
    not_supported  CI upper bound < 0 and at least ``MIN_PAIRED_SEEDS`` paired seeds
    inconclusive   otherwise (including every run with too few seeds)

Evidence tier follows the existing taxonomy: fewer than ``MIN_PAIRED_SEEDS`` seeds is a SIMULATED_ASSAY
(a smoke-sized run is a SMOKE_TEST); only larger paired runs are PAIRED_EXPERIMENT.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...contracts.evidence import EvidenceTier
from ...evaluation.statistical_analysis import paired_seed_analysis
from ..contracts import INTERFACE_VERSIONS, WEAVE_VERSION
from ..ecology import make_environment
from ..hardware.profiler import state_bytes
from ..registry import MECHANISMS
from ..runtime import build_weave_substrate, canonical_motor_head, run_weave_episode

MIN_PAIRED_SEEDS = 5
HELD_OUT_OFFSET = 1000
METRICS = ("return", "efficiency")
_PER_STEP_REWARD_TASKS = ("forage",)


def episode_efficiency(success: bool, steps: int, max_steps: Optional[int]) -> float:
    """Goal efficiency of one episode: 0 on failure, 1 for success on the first step, linear in between."""
    if not success:
        return 0.0
    horizon = max(1, int(max_steps or steps))
    return float(max(0.0, 1.0 - (int(steps) - 1) / horizon))


def resolve_metric(metric: str, environment: str) -> str:
    if metric == "auto":
        return "efficiency" if environment in _PER_STEP_REWARD_TASKS else "return"
    if metric not in METRICS:
        raise ValueError(f"unknown primary metric '{metric}' (expected one of {METRICS} or 'auto')")
    return metric


@dataclass
class ExperimentSpec:
    experiment_id: str
    title: str
    hypothesis: str
    candidate: Tuple[str, ...]
    baseline: Tuple[str, ...] = ()
    environment: str = "forage"
    env_options: Dict[str, Any] = field(default_factory=dict)
    topology_kind: Optional[str] = None
    total_neurons: int = 768
    episodes_train: int = 3
    episodes_eval: int = 2
    seeds: Tuple[int, ...] = (11, 23, 37)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ablations: Tuple[str, ...] = ()
    kind: str = "paired"                      # paired | topology | lesion
    notes: str = ""
    primary_metric: str = "auto"              # return | efficiency | auto (see module docstring)
    smoke: bool = False                       # smoke-sized run: reported as SMOKE_TEST, never a verdict

    def __post_init__(self):
        resolve_metric(self.primary_metric, self.environment)
        unknown = [m for m in tuple(self.candidate) + tuple(self.baseline) + tuple(self.ablations) if m not in MECHANISMS]
        if unknown:
            raise KeyError(f"experiment '{self.experiment_id}' references unknown mechanisms: {unknown}")
        missing = [m for m in self.ablations if m not in self.candidate]
        if missing:
            raise ValueError(f"ablations must be candidate mechanisms: {missing}")

    def arms(self) -> Dict[str, Tuple[str, ...]]:
        arms = {"baseline": tuple(self.baseline), "candidate": tuple(self.candidate)}
        for mid in self.ablations:
            arms[f"candidate_minus_{mid}"] = tuple(m for m in self.candidate if m != mid)
        return arms

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("candidate", "baseline", "seeds", "ablations"):
            data[key] = list(data[key])
        return data


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def run_arm(mechanisms: Sequence[str], seed: int, spec: ExperimentSpec, strict: bool = False) -> Dict[str, Any]:
    """Train then evaluate one arm for one seed.  Fully determined by (mechanisms, seed, spec)."""
    substrate = build_weave_substrate(total_neurons=spec.total_neurons, seed=seed, enable=tuple(mechanisms),
                                      params={k: v for k, v in spec.params.items() if k in mechanisms or k == "cell"},
                                      topology_kind=spec.topology_kind, strict=strict)
    motor = canonical_motor_head()
    env = make_environment(spec.environment, seed=seed, **spec.env_options)
    train, steps, elapsed = [], 0, 0.0
    for k in range(spec.episodes_train):
        start = time.perf_counter()
        result = run_weave_episode(env, substrate, motor, seed=seed + k)
        elapsed += time.perf_counter() - start
        steps += result.steps
        train.append(result.total_return)
    substrate.freeze_learning(True)
    held_out, successes, held_steps, efficiency = [], 0, 0, []
    horizon = getattr(env, "max_steps", None)
    for k in range(spec.episodes_eval):
        start = time.perf_counter()
        result = run_weave_episode(env, substrate, motor, seed=seed + HELD_OUT_OFFSET + k)
        elapsed += time.perf_counter() - start
        steps += result.steps
        held_steps += result.steps
        held_out.append(result.total_return)
        successes += int(result.success)
        efficiency.append(episode_efficiency(result.success, result.steps, horizon))
    telemetry = substrate.weave_telemetry()
    return {
        "mechanisms": list(mechanisms), "seed": int(seed),
        "train_returns": [float(x) for x in train], "heldout_returns": [float(x) for x in held_out],
        "heldout_mean": float(np.mean(held_out)) if held_out else float("nan"),
        "train_mean": float(np.mean(train)) if train else float("nan"),
        "heldout_success_rate": successes / max(1, spec.episodes_eval),
        "heldout_mean_steps": held_steps / max(1, spec.episodes_eval),
        "heldout_efficiency": float(np.mean(efficiency)) if efficiency else float("nan"),
        "cost": {"measured_ms_per_step": 1000.0 * elapsed / max(1, steps), "steps": int(steps),
                 "measured_state_bytes": state_bytes(substrate)["total_bytes"],
                 "synapses": telemetry["edges"], "neurons": telemetry["neurons"]},
        "learned": {"mean_abs_weight_delta": telemetry["edges"]["mean_abs_delta"], "grown_synapses": telemetry["edges"]["grown"]},
        "faults": telemetry["faults"], "config_digest": substrate.weave.digest(),
    }


def compare(candidate: Sequence[float], baseline: Sequence[float]) -> Dict[str, Any]:
    """Paired comparison with an explicit verdict and evidence tier."""
    n = len(candidate)
    if n < 2:
        return {"n_seeds": n, "mean_diff": float(candidate[0] - baseline[0]) if n == 1 else None, "verdict": "inconclusive",
                "reason": "fewer than two paired seeds", "evidence_tier": EvidenceTier.SMOKE_TEST.value}
    stats = {k: (None if v is None else (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)))
             for k, v in paired_seed_analysis(list(candidate), list(baseline)).items()}
    wins = int(sum(1 for c, b in zip(candidate, baseline) if c > b))
    if n >= MIN_PAIRED_SEEDS and stats["ci_95_lower"] > 0:
        verdict, reason = "supported", "95% bootstrap CI of the paired difference lies above zero"
    elif n >= MIN_PAIRED_SEEDS and stats["ci_95_upper"] < 0:
        verdict, reason = "not_supported", "95% bootstrap CI of the paired difference lies below zero"
    else:
        verdict = "inconclusive"
        reason = "CI includes zero" if n >= MIN_PAIRED_SEEDS else f"only {n} paired seeds (need {MIN_PAIRED_SEEDS} for any verdict)"
    tier = EvidenceTier.PAIRED_EXPERIMENT if n >= MIN_PAIRED_SEEDS else EvidenceTier.SIMULATED_ASSAY
    return {**stats, "n_seeds": n, "wins": wins, "verdict": verdict, "reason": reason, "evidence_tier": tier.value}


def run_experiment(spec: ExperimentSpec, seeds: Optional[Iterable[int]] = None, strict: bool = False) -> Dict[str, Any]:
    """Run every arm on every seed and return an experiment receipt (a plain JSON-serialisable dict)."""
    seeds = tuple(int(s) for s in (seeds if seeds is not None else spec.seeds))
    arms = spec.arms()
    runs: Dict[str, List[Dict[str, Any]]] = {name: [run_arm(mech, seed, spec, strict) for seed in seeds] for name, mech in arms.items()}
    summary = {}
    for name, records in runs.items():
        held = [r["heldout_mean"] for r in records]
        summary[name] = {"mechanisms": list(arms[name]), "heldout_mean": float(np.mean(held)), "heldout_std": float(np.std(held)),
                         "train_mean": float(np.mean([r["train_mean"] for r in records])),
                         "heldout_success_rate": float(np.mean([r["heldout_success_rate"] for r in records])),
                         "heldout_mean_steps": float(np.mean([r["heldout_mean_steps"] for r in records])),
                         "heldout_efficiency": float(np.mean([r["heldout_efficiency"] for r in records])),
                         "measured_ms_per_step": float(np.mean([r["cost"]["measured_ms_per_step"] for r in records])),
                         "measured_state_bytes": int(np.mean([r["cost"]["measured_state_bytes"] for r in records])),
                         "faults": int(sum(len(r["faults"]) for r in records))}
    metric = resolve_metric(spec.primary_metric, spec.environment)
    key = {"return": "heldout_mean", "efficiency": "heldout_efficiency"}
    values = lambda arm, m: [r[key[m]] for r in runs[arm]]
    secondary = "return" if metric == "efficiency" else "efficiency"
    comparisons = {"candidate_vs_baseline": dict(compare(values("candidate", metric), values("baseline", metric)), metric=metric),
                   f"candidate_vs_baseline_{secondary}": dict(compare(values("candidate", secondary), values("baseline", secondary)),
                                                               metric=secondary)}
    for name in arms:
        if name.startswith("candidate_minus_"):                        # contribution of the removed mechanism
            comparisons[f"candidate_vs_{name}"] = dict(compare(values("candidate", metric), values(name, metric)), metric=metric)
    if spec.smoke:
        for cmp in comparisons.values():
            cmp.update(verdict="inconclusive", evidence_tier="SMOKE_TEST", reason="smoke-sized run (quick); never used for a verdict")
    cost_ratio = summary["candidate"]["measured_ms_per_step"] / max(1e-9, summary["baseline"]["measured_ms_per_step"])
    results = {"summary": summary, "comparisons": comparisons, "runs": runs}
    deterministic = {"spec": spec.to_dict(), "seeds": list(seeds),
                     "returns": {name: [[r["train_returns"], r["heldout_returns"]] for r in records] for name, records in runs.items()}}
    return {
        "receipt_version": INTERFACE_VERSIONS["experiment_receipt"], "weave_version": WEAVE_VERSION,
        "experiment_id": spec.experiment_id, "title": spec.title, "hypothesis": spec.hypothesis, "kind": spec.kind,
        "spec": spec.to_dict(), "seeds": list(seeds), "held_out_seed_offset": HELD_OUT_OFFSET, "primary_metric": metric,
        "smoke": bool(spec.smoke),
        "matched_conditions": {"same_topology_seed_environment_schedule": True, "candidate_cost_ratio_measured": cost_ratio,
                               "note": "arms differ only in enabled mechanisms; compute is reported, not equalised"},
        "results": results, "verdict": comparisons["candidate_vs_baseline"]["verdict"],
        "evidence_tier": comparisons["candidate_vs_baseline"]["evidence_tier"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.system(), "torch_threads": torch.get_num_threads()},
        "created_unix": time.time(), "result_digest": _digest(deterministic),
        "secondary_metrics_note": f"the verdict uses the primary metric ({metric}); the {secondary} comparison is reported alongside. "
                                  "On tasks that pay reward every step a slower agent collects more return, so return alone is not a "
                                  "measure of competence there",
        "disclaimer": "synthetic benchmark; a verdict describes this task, these seeds and this budget only",
    }


def receipt_to_csv(receipt: Dict[str, Any]) -> str:
    """One row per (arm, seed) with returns and measured cost."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["experiment_id", "arm", "seed", "mechanisms", "train_mean", "heldout_mean", "heldout_success_rate",
                     "measured_ms_per_step", "measured_state_bytes", "mean_abs_weight_delta", "faults", "heldout_efficiency",
                     "heldout_mean_steps"])
    for arm, records in receipt["results"]["runs"].items():
        for r in records:
            writer.writerow([receipt["experiment_id"], arm, r["seed"], "+".join(r["mechanisms"]) or "legacy", f"{r['train_mean']:.6f}",
                             f"{r['heldout_mean']:.6f}", f"{r['heldout_success_rate']:.3f}", f"{r['cost']['measured_ms_per_step']:.4f}",
                             r["cost"]["measured_state_bytes"], f"{r['learned']['mean_abs_weight_delta']:.8f}", len(r["faults"]),
                             f"{r['heldout_efficiency']:.6f}", f"{r['heldout_mean_steps']:.2f}"])
    return buffer.getvalue()


def export_receipt(receipt: Dict[str, Any], directory: str) -> Dict[str, str]:
    """Write ``<id>.json`` and ``<id>.csv``.  Failed or inconclusive experiments are exported like any other."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    stem = receipt["experiment_id"]
    json_path, csv_path = out / f"{stem}.json", out / f"{stem}.csv"
    json_path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    csv_path.write_text(receipt_to_csv(receipt), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path)}


__all__ = ["ExperimentSpec", "METRICS", "episode_efficiency", "resolve_metric", "MIN_PAIRED_SEEDS", "HELD_OUT_OFFSET", "run_arm", "compare", "run_experiment", "receipt_to_csv",
           "export_receipt"]
