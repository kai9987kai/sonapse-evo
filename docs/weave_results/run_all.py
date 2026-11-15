"""Reproduce the recorded NeuroWeave results:  python docs/weave_results/run_all.py

Runs every catalogued paired experiment with five paired seeds, the topology assay and the lesion assay,
and writes one JSON receipt (+ CSV for paired experiments) per experiment plus ``summary.json``.
"""
import json
import sys
import time
from pathlib import Path

from sonapse_evo.neuroweave.experiments import (EXPERIMENTS, export_receipt, run_catalog_experiment, run_lesion_assay,
                                                run_topology_assay)

OUT = Path(__file__).resolve().parent
SEEDS = (11, 23, 37, 51, 67)
only = set(sys.argv[1:])
summary, start = {}, time.time()
if only and (OUT / "summary.json").exists():                 # partial re-run: keep the other entries
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
for eid, spec in EXPERIMENTS.items():
    if spec.kind != "paired" or (only and eid not in only):
        continue
    t0 = time.time()
    receipt = run_catalog_experiment(eid, seeds=SEEDS)
    export_receipt(receipt, str(OUT))
    cmp = receipt["results"]["comparisons"]["candidate_vs_baseline"]
    sec = next(v for k, v in receipt["results"]["comparisons"].items() if k.startswith("candidate_vs_baseline_"))
    s = receipt["results"]["summary"]
    summary[eid] = {"verdict": receipt["verdict"], "evidence_tier": receipt["evidence_tier"], "environment": spec.environment,
                    "baseline": s["baseline"]["heldout_mean"], "candidate": s["candidate"]["heldout_mean"],
                    "primary_metric": receipt["primary_metric"],
                    "efficiency": [s["baseline"]["heldout_efficiency"], s["candidate"]["heldout_efficiency"]],
                    "secondary": {"metric": sec["metric"], "mean_diff": sec["mean_diff"], "verdict": sec["verdict"]},
                    "success_rate": [s["baseline"]["heldout_success_rate"], s["candidate"]["heldout_success_rate"]],
                    "mean_steps": [s["baseline"]["heldout_mean_steps"], s["candidate"]["heldout_mean_steps"]],
                    "ablations": {k: v["mean_diff"] for k, v in receipt["results"]["comparisons"].items() if k.startswith("candidate_vs_candidate_minus_")},
                    "mean_diff": cmp["mean_diff"], "ci95": [cmp["ci_95_lower"], cmp["ci_95_upper"]], "cohens_d": cmp["cohens_d"],
                    "wins": cmp["wins"], "n": cmp["n_seeds"], "cost_ratio": receipt["matched_conditions"]["candidate_cost_ratio_measured"],
                    "faults": sum(v["faults"] for v in s.values()), "seconds": round(time.time() - t0, 1)}
    print(f"{eid:34s} [{receipt['primary_metric']:10s}] {receipt['verdict']:14s} diff {cmp['mean_diff']:+8.3f} CI [{cmp['ci_95_lower']:+.2f}, {cmp['ci_95_upper']:+.2f}] "
          f"cost x{summary[eid]['cost_ratio']:.1f} {summary[eid]['seconds']}s", flush=True)
if not only:
    topo = run_topology_assay(seeds=SEEDS, episodes=2)
    (OUT / "topology_optimisation.json").write_text(json.dumps(topo, indent=1, default=str), encoding="utf-8")
    summary["topology_optimisation"] = {"ranking": topo["results"]["ranking_by_heldout_return"],
                                        "ranking_by_efficiency": topo["results"]["ranking_by_efficiency"],
                                        "primary_metric": topo["primary_metric"], "summary": topo["results"]["summary"],
                                        "comparisons": {k: {"verdict": v["verdict"], "mean_diff": v["mean_diff"]} for k, v in topo["results"]["comparisons"].items()}}
    lesion = run_lesion_assay(seeds=SEEDS, severity=0.6, recovery_episodes=2, progressive_steps=4, mechanisms=("reward_gated_plasticity",))
    (OUT / "lesion_robustness.json").write_text(json.dumps(lesion, indent=1, default=str), encoding="utf-8")
    summary["lesion_robustness"] = {"summary": lesion["results"]["summary"],
                                    "comparisons": {k: {"verdict": v["verdict"], "mean_diff": v["mean_diff"], "metric": v["metric"]}
                                                    for k, v in lesion["results"]["comparisons"].items()},
                                    "progressive": [r["progressive_random"] for r in lesion["results"]["runs"]]}
summary["_meta"] = {"seeds": list(SEEDS), "total_seconds": round(time.time() - start, 1)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
print(f"done in {time.time() - start:.0f}s", flush=True)
