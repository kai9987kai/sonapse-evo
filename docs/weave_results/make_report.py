"""Build docs/NEUROWEAVE_RESULTS.md from the recorded receipts:  python docs/weave_results/make_report.py

Tables are generated from ``summary.json`` and the per-experiment receipts in this folder, plus cost
measurements taken on the machine that runs this script.  The interpretation section is read from
``interpretation.md`` (hand-written) so that numbers and commentary cannot drift apart silently.
"""
import json
import platform
from pathlib import Path

import numpy as np
import torch

from sonapse_evo.neuroweave import build_weave_substrate
from sonapse_evo.neuroweave.hardware import profile_substrate, scaling_profile

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
meta = summary.pop("_meta")
topology = summary.pop("topology_optimisation")
lesion = summary.pop("lesion_robustness")

out = ["# NeuroWeave Evolution: measured results", "",
       "Everything in this file was measured by running the code in this repository. Nothing is projected or",
       "estimated unless it says so. Reproduce with `python docs/weave_results/run_all.py` (experiments) and",
       "`python docs/weave_results/make_report.py` (this report). Receipts (JSON and CSV) are in `docs/weave_results/`.", "",
       f"Machine: {platform.system()} {platform.machine()}, Python {platform.python_version()}, PyTorch {torch.__version__} (CPU, "
       f"{torch.get_num_threads()} threads), NumPy {np.__version__}.", ""]

out += ["## 1. Cost (measured wall-clock, 768 neurons)", "", "| profile | mechanisms | ms per step | relative to legacy | tensor state |", "|---|---|---|---|---|"]
base = None
for profile in ("legacy", "cells", "transmission", "learning", "neuroweave", "max"):
    sub = build_weave_substrate(seed=42, profile=profile, strict=True)
    m = profile_substrate(sub, steps=40)["measured"]
    base = base or m["ms_per_step"]
    out.append(f"| `{profile}` | {len(sub.weave.enabled_ids())} | {m['ms_per_step']:.2f} | {m['ms_per_step'] / base:.1f}x | {m['total_bytes'] / 1e6:.2f} MB |")
out += ["", "Population scaling (default sparse topology, mean degree 8):", "", "| neurons | synapses | legacy ms/step | `cells` profile ms/step |", "|---|---|---|---|"]
sizes = (768, 1536, 3072, 6144)
legacy_rows, cell_rows = scaling_profile(sizes, "legacy", steps=15), scaling_profile(sizes, "cells", steps=15)
for a, b in zip(legacy_rows, cell_rows):
    out.append(f"| {a['config']['neurons']} | {a['config']['edges']} | {a['measured']['ms_per_step']:.2f} | {b['measured']['ms_per_step']:.2f} |")
out += ["", "No hardware acceleration was used or is claimed.", ""]

out += ["## 2. Controlled experiments", "",
        f"Protocol: seeds {meta['seeds']} (paired), 3 training episodes, learning frozen, 2 held-out episodes per seed; the candidate and",
        "baseline differ only in the listed mechanisms. The verdict uses the primary metric: **goal efficiency** on `forage` (success times",
        "the fraction of the step budget left), **return** on the ecology tasks. `diff` is the mean paired difference in that metric with its",
        "95% bootstrap interval. A verdict needs five paired seeds; `supported` / `not_supported` mean the interval excludes zero.", "",
        "| experiment | environment | metric | baseline | candidate | diff [95% CI] | wins | secondary: return diff | success (base / cand) | steps (base / cand) | cost | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for eid, r in summary.items():
    sr, st = r.get("success_rate", [float("nan")] * 2), r.get("mean_steps", [float("nan")] * 2)
    metric = r.get("primary_metric", "return")
    base, cand = (r["efficiency"] if metric == "efficiency" else (r["baseline"], r["candidate"]))
    sec = r.get("secondary", {})
    sec_text = f"{sec['mean_diff']:+.2f}" if sec.get("metric") == "return" and sec.get("mean_diff") is not None else "(primary)"
    out.append(f"| `{eid}` | {r['environment']} | {metric} | {base:.3f} | {cand:.3f} | {r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | "
               f"{r['wins']}/{r['n']} | {sec_text} | {sr[0]:.2f} / {sr[1]:.2f} | {st[0]:.1f} / {st[1]:.1f} | {r['cost_ratio']:.1f}x | {r['verdict']} |")
counts = {}
for r in summary.values():
    counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
faults = sum(r["faults"] for r in summary.values())
out += ["", f"Verdict counts: {counts}. Mechanism faults across all runs: {faults}.", ""]
ablated = {eid: r["ablations"] for eid, r in summary.items() if r.get("ablations")}
if ablated:
    out += ["Ablations (candidate minus the candidate with one mechanism removed, in the primary metric; positive means the removed mechanism helped):", "",
            "| experiment | removed mechanism | mean paired difference |", "|---|---|---|"]
    for eid, items in ablated.items():
        for name, value in items.items():
            out.append(f"| `{eid}` | {name.replace('candidate_vs_candidate_minus_', '')} | {value:+.4f} |")
    out.append("")

out += ["## 3. Topology assay", "", "Same size, seeds, mechanisms (none), task (`forage`) and schedule for every graph family; edge counts differ and are",
        "reported. Rows are ordered by goal efficiency; the last column is the efficiency after removing 20% of synapses at random.", "",
        "| topology | goal efficiency | held-out return | success | steps | synapses | clustering | modularity | path length | efficiency after lesion | ms/step | vs legacy |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for kind in topology.get("ranking_by_efficiency", topology["ranking"]):
    s = topology["summary"][kind]
    cmp = topology.get("comparisons", {}).get(f"{kind}_vs_legacy")
    verdict = "reference" if kind == "legacy" else (f"{cmp['mean_diff']:+.3f} ({cmp['verdict']})" if cmp else "-")
    out.append(f"| {kind} | {s['efficiency_mean']:.3f} | {s['heldout_mean']:.2f} | {s['success_rate']:.2f} | {s['mean_steps']:.1f} | {s['edges_mean']:.0f} | "
               f"{s['clustering_coefficient_mean']:.3f} | {s['modularity_mean']:.3f} | {s['characteristic_path_length_mean']:.2f} | "
               f"{s['lesioned_efficiency_mean']:.3f} | {s['measured_ms_per_step']:.2f} | {verdict} |")
out += [""]

out += ["## 4. Lesion robustness assay", "", "Ten-region connectome, severity 0.6, reward-gated plasticity enabled, two recovery episodes; each targeted lesion has a random",
        "control removing exactly the same number of synapses. The damage test asks whether the targeted lesion lowers goal efficiency MORE than its",
        "matched random control (`supported` = yes).", "",
        "| lesion | synapses removed | efficiency: intact | targeted | matched random | success (targeted / random) | targeted damage exceeds random? |",
        "|---|---|---|---|---|---|---|"]
for kind, s in lesion["summary"].items():
    cmp = lesion["comparisons"][f"{kind}_targeted_damage_exceeds_random"]
    out.append(f"| {kind} | {s['edges_removed_mean']:.0f} | {s['intact_efficiency']:.3f} | {s['targeted_efficiency']:.3f} | {s['random_efficiency']:.3f} | "
               f"{s['targeted_success_rate']:.2f} / {s['random_success_rate']:.2f} | {cmp['mean_diff']:+.3f} ({cmp['verdict']}) |")
prog = lesion.get("progressive", [])
if prog:
    sev = [p["severity"] for p in prog[0]]
    means = [float(np.mean([run[i]["efficiency"] for run in prog])) for i in range(len(sev))]
    out += ["", "Progressive random damage (mean goal efficiency over seeds): " + ", ".join(f"{s:.2f} -> {m:.3f}" for s, m in zip(sev, means))]
out += [""]

interpretation = HERE / "interpretation.md"
if interpretation.exists():
    out += [interpretation.read_text(encoding="utf-8").rstrip(), ""]
(HERE.parent / "NEUROWEAVE_RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote docs/NEUROWEAVE_RESULTS.md with", len(summary), "experiments")
