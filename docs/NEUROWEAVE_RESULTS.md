# NeuroWeave Evolution: measured results

Everything in this file was measured by running the code in this repository. Nothing is projected or
estimated unless it says so. Reproduce with `python docs/weave_results/run_all.py` (experiments) and
`python docs/weave_results/make_report.py` (this report). Receipts (JSON and CSV) are in `docs/weave_results/`.

Machine: Windows ARM64, Python 3.12.10, PyTorch 2.11.0+cpu (CPU, 8 threads), NumPy 2.4.3.

## 1. Cost (measured wall-clock, 768 neurons)

| profile | mechanisms | ms per step | relative to legacy | tensor state |
|---|---|---|---|---|
| `legacy` | 0 | 0.82 | 1.0x | 1.17 MB |
| `cells` | 6 | 9.45 | 11.6x | 1.17 MB |
| `transmission` | 4 | 5.66 | 6.9x | 1.17 MB |
| `learning` | 4 | 6.95 | 8.5x | 1.24 MB |
| `neuroweave` | 32 | 34.04 | 41.7x | 1.45 MB |
| `max` | 42 | 55.16 | 67.6x | 1.58 MB |

Population scaling (default sparse topology, mean degree 8):

| neurons | synapses | legacy ms/step | `cells` profile ms/step |
|---|---|---|---|
| 768 | 5727 | 0.67 | 9.26 |
| 1536 | 11433 | 0.96 | 10.43 |
| 3072 | 22920 | 1.21 | 12.06 |
| 6144 | 45896 | 1.77 | 12.73 |

No hardware acceleration was used or is claimed.

## 2. Controlled experiments

Protocol: seeds [11, 23, 37, 51, 67] (paired), 3 training episodes, learning frozen, 2 held-out episodes per seed; the candidate and
baseline differ only in the listed mechanisms. The verdict uses the primary metric: **goal efficiency** on `forage` (success times
the fraction of the step budget left), **return** on the ecology tasks. `diff` is the mean paired difference in that metric with its
95% bootstrap interval. A verdict needs five paired seeds; `supported` / `not_supported` mean the interval excludes zero.

| experiment | environment | metric | baseline | candidate | diff [95% CI] | wins | secondary: return diff | success (base / cand) | steps (base / cand) | cost | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `multimodal_sensory_integration` | neuroweave_1200x660 | return | -20.553 | -14.537 | +6.016 [-0.082, +16.589] | 3/5 | (primary) | 0.00 / 0.00 | 80.0 / 80.0 | 3.8x | inconclusive |
| `affective_neuromodulation` | forage | efficiency | 0.745 | 0.752 | +0.008 [+0.002, +0.013] | 3/5 | -0.23 | 1.00 / 1.00 | 11.2 / 10.9 | 1.1x | supported |
| `micro_collective_learning` | neuroweave_1200x660 | return | -9.646 | -14.841 | -5.195 [-17.605, +2.073] | 1/5 | (primary) | 0.00 / 0.00 | 80.0 / 80.0 | 1.3x | inconclusive |
| `expert_routing` | neuroweave_1200x660 | return | -14.841 | -14.500 | +0.341 [-0.015, +0.771] | 3/5 | (primary) | 0.00 / 0.00 | 80.0 / 80.0 | 1.1x | inconclusive |
| `meta_plasticity` | forage | efficiency | 0.745 | 0.745 | +0.000 [+0.000, +0.000] | 0/5 | -0.04 | 1.00 / 1.00 | 11.2 / 11.2 | 1.0x | inconclusive |
| `predictive_replay` | forage | efficiency | 0.745 | 0.745 | +0.000 [+0.000, +0.000] | 0/5 | +0.00 | 1.00 / 1.00 | 11.2 / 11.2 | 1.0x | inconclusive |
| `connectivity_context_attention` | forage | efficiency | 0.635 | 0.640 | +0.005 [+0.000, +0.010] | 2/5 | -0.23 | 1.00 / 1.00 | 15.6 / 15.4 | 2.7x | inconclusive |
| `flyhash_sparse_projections` | forage | efficiency | 0.742 | 0.735 | -0.007 [-0.012, -0.002] | 0/5 | +0.44 | 1.00 / 1.00 | 11.3 / 11.6 | 4.1x | not_supported |
| `stdp` | forage | efficiency | 0.745 | 0.745 | +0.000 [+0.000, +0.000] | 0/5 | -0.00 | 1.00 / 1.00 | 11.2 / 11.2 | 1.4x | inconclusive |
| `adaptive_conduction` | forage | efficiency | 0.675 | 0.677 | +0.002 [+0.000, +0.007] | 1/5 | -0.03 | 1.00 / 1.00 | 14.0 / 13.9 | 1.0x | inconclusive |
| `quantised_synapses` | forage | efficiency | 0.742 | 0.742 | +0.000 [+0.000, +0.000] | 0/5 | +0.00 | 1.00 / 1.00 | 11.3 / 11.3 | 4.0x | inconclusive |
| `hierarchical_temporal_memory` | memory_navigation | return | 4.935 | 5.022 | +0.087 [-0.005, +0.185] | 3/5 | (primary) | 0.00 / 0.00 | 80.0 / 80.0 | 1.1x | inconclusive |
| `uncertainty_gated_propagation` | forage | efficiency | 0.742 | 0.730 | -0.012 [-0.020, -0.005] | 0/5 | +0.67 | 1.00 / 1.00 | 11.3 / 11.8 | 1.0x | not_supported |
| `predictive_coding` | forage | efficiency | 0.738 | 0.742 | +0.005 [+0.000, +0.010] | 2/5 | -0.25 | 1.00 / 1.00 | 11.5 / 11.3 | 1.2x | inconclusive |
| `structural_plasticity` | forage | efficiency | 0.745 | 0.745 | +0.000 [+0.000, +0.000] | 0/5 | +0.01 | 1.00 / 1.00 | 11.2 / 11.2 | 0.9x | inconclusive |
| `associative_memory` | forage | efficiency | 0.742 | 0.745 | +0.002 [+0.000, +0.007] | 1/5 | -0.07 | 1.00 / 1.00 | 11.3 / 11.2 | 5.4x | inconclusive |
| `action_grounded_self_modelling` | disturbed | return | -6.496 | -4.121 | +2.375 [-0.728, +5.230] | 4/5 | (primary) | 0.00 / 0.00 | 80.0 / 80.0 | 1.2x | inconclusive |
| `multi_soma_integration` | forage | efficiency | 0.742 | 0.530 | -0.212 [-0.277, -0.123] | 0/5 | +10.04 | 1.00 / 1.00 | 11.3 / 19.8 | 6.2x | not_supported |
| `multi_region_routing` | forage | efficiency | 0.555 | 0.482 | -0.072 [-0.212, +0.000] | 0/5 | -0.23 | 0.70 / 0.60 | 18.5 / 21.3 | 3.5x | inconclusive |
| `neural_grafting` | neuroweave_1200x660 | return | -9.646 | -14.216 | -4.570 [-16.686, +2.369] | 4/5 | (primary) | 0.00 / 0.00 | 80.0 / 80.0 | 1.3x | inconclusive |
| `cross_modal_learning` | neuroweave_1200x660 | return | -15.162 | -14.508 | +0.654 [-0.995, +2.331] | 3/5 | (primary) | 0.10 / 0.00 | 80.0 / 80.0 | 1.2x | inconclusive |
| `specialist_expert_evolution` | neuroweave_1200x660 | return | -14.643 | -14.500 | +0.144 [-0.000, +0.428] | 4/5 | (primary) | 0.00 / 0.00 | 80.0 / 80.0 | 1.2x | inconclusive |

Verdict counts: {'inconclusive': 18, 'supported': 1, 'not_supported': 3}. Mechanism faults across all runs: 0.

Ablations (candidate minus the candidate with one mechanism removed, in the primary metric; positive means the removed mechanism helped):

| experiment | removed mechanism | mean paired difference |
|---|---|---|
| `multimodal_sensory_integration` | cross_modal_fusion | -4.8905 |
| `affective_neuromodulation` | affective_control | -0.0025 |
| `connectivity_context_attention` | structural_encoder | +0.0050 |
| `flyhash_sparse_projections` | flyhash_projection | -0.0025 |
| `associative_memory` | fast_weights | +0.0000 |
| `associative_memory` | memory_ecology | +0.0000 |
| `multi_soma_integration` | dendritic_compartments | -0.0800 |
| `multi_soma_integration` | axonal_branches | -0.1125 |

## 3. Topology assay

Same size, seeds, mechanisms (none), task (`forage`) and schedule for every graph family; edge counts differ and are
reported. Rows are ordered by goal efficiency; the last column is the efficiency after removing 20% of synapses at random.

| topology | goal efficiency | held-out return | success | steps | synapses | clustering | modularity | path length | efficiency after lesion | ms/step | vs legacy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical_modular | 0.745 | 27.21 | 1.00 | 11.2 | 5400 | 0.149 | 0.591 | 4.01 | 0.745 | 1.12 | +0.002 (inconclusive) |
| legacy | 0.742 | 27.30 | 1.00 | 11.3 | 5711 | 0.037 | 0.530 | 3.69 | 0.742 | 1.58 | reference |
| hybrid | 0.740 | 27.49 | 1.00 | 11.4 | 5129 | 0.218 | 0.506 | 4.30 | 0.743 | 1.23 | -0.002 (inconclusive) |
| hub_capped | 0.740 | 27.46 | 1.00 | 11.4 | 5074 | 0.325 | 0.547 | 4.68 | 0.738 | 1.16 | -0.002 (inconclusive) |
| degree_shuffled | 0.740 | 27.45 | 1.00 | 11.4 | 5075 | 0.015 | 0.001 | 3.65 | 0.738 | 1.47 | -0.002 (inconclusive) |
| clustered_small_world | 0.740 | 27.45 | 1.00 | 11.4 | 5075 | 0.325 | 0.547 | 4.68 | 0.740 | 1.48 | -0.002 (inconclusive) |
| hub_skewed | 0.740 | 27.45 | 1.00 | 11.4 | 6105 | 0.022 | -0.004 | 3.45 | 0.740 | 1.45 | -0.002 (inconclusive) |
| uniform_random | 0.740 | 27.45 | 1.00 | 11.4 | 6112 | 0.020 | -0.004 | 3.42 | 0.740 | 2.13 | -0.002 (inconclusive) |
| erdos_renyi | 0.738 | 27.59 | 1.00 | 11.5 | 6157 | 0.021 | -0.001 | 3.42 | 0.740 | 1.24 | -0.005 (inconclusive) |
| ring_lattice | 0.738 | 27.58 | 1.00 | 11.5 | 6147 | 0.642 | 0.656 | 43.54 | 0.677 | 1.26 | -0.005 (inconclusive) |
| multi_region | 0.555 | 26.70 | 0.70 | 18.5 | 15495 | 0.073 | 0.239 | 2.58 | 0.618 | 1.28 | -0.188 (inconclusive) |
| multiscale_regional | 0.550 | 28.66 | 0.70 | 18.7 | 11543 | 0.122 | 0.581 | 2.97 | 0.532 | 1.16 | -0.192 (not_supported) |

## 4. Lesion robustness assay

Ten-region connectome, severity 0.6, reward-gated plasticity enabled, two recovery episodes; each targeted lesion has a random
control removing exactly the same number of synapses. The damage test asks whether the targeted lesion lowers goal efficiency MORE than its
matched random control (`supported` = yes).

| lesion | synapses removed | efficiency: intact | targeted | matched random | success (targeted / random) | targeted damage exceeds random? |
|---|---|---|---|---|---|---|
| hub_targeted | 1217 | 0.555 | 0.320 | 0.460 | 0.40 / 0.60 | +0.140 (inconclusive) |
| region_targeted | 965 | 0.555 | 0.555 | 0.542 | 0.70 / 0.70 | -0.013 (inconclusive) |
| sensory_pathway | 4813 | 0.555 | 0.280 | 0.383 | 0.40 / 0.70 | +0.102 (inconclusive) |
| memory_pathway | 2289 | 0.555 | 0.552 | 0.382 | 0.70 / 0.50 | -0.170 (not_supported) |
| cross_region_communication | 5847 | 0.555 | 0.212 | 0.198 | 0.40 / 0.50 | -0.015 (inconclusive) |

Progressive random damage (mean goal efficiency over seeds): 0.15 -> 0.545, 0.30 -> 0.478, 0.45 -> 0.475, 0.60 -> 0.163

## 5. Interpretation

### What the evidence supports

* **The original model is preserved.** With every mechanism off, the model is bit-identical to SONAPSE-EVO 2.0
  (25-step trace digest, lesion masks, episode returns, CLI outputs and the full API surface all match the
  recorded baseline). This is the strongest result in this document.
* **The new subsystems work as engineered.** They run, stay bounded and finite, continue exactly after clone or
  export and import, and change the emitted rates when enabled. The Micro-Collective learns a structured
  sensory stream to about 1% of the persistence baseline's error, the forward model's error falls with
  training, prediction horizons shorten under uncertainty, and the self-model separates self-caused from
  disturbed outcomes. These are tested properties of the components, not claims about task performance.
* **Affective neuromodulation** (`supported` on goal efficiency, five paired seeds) improved efficiency by
  about 0.008, roughly a third of a step per episode on a 40-step budget. It is statistically consistent but
  practically very small, and its ablation (removing the affect controller but keeping the modulator field)
  changed nothing measurable. Not promoted.

### What it does not support

* **No mechanism improves the existing control task in a meaningful way.** Most comparisons on `forage` show a
  paired difference of exactly zero: with three training episodes and learning then frozen, plasticity,
  meta-plasticity, structural turnover, STDP, replay and quantisation do not change behaviour enough to alter
  how fast the fixed motor read-out reaches the target.
* **Multi-soma / multi-compartment integration** makes the agent markedly slower (efficiency 0.74 to 0.53, same
  success rate, about 20 instead of 11 steps). Both ablated parts (dendrites, axonal branches) contribute to
  the slow-down. The richer cell model is not better here; the canonical motor read-out was tuned for the
  plain rate neuron.
* **FlyHash projections and uncertainty-gated propagation** also slow the agent slightly (`not_supported`).
* **Ecology experiments are inconclusive.** With an 80-step budget the agent (fixed read-out, no trained policy)
  rarely reaches a goal in the 1200 x 660 arena, so neither sensory integration, the Micro-Collective, expert
  routing, the specialist graft, the self-model nor cross-modal learning could show an effect. The large
  return differences in those rows come with intervals that include zero.
* **Topology.** At matched size and seeds every single-population family performs like the original graph;
  the ten-region families are clearly worse (success 0.70 vs 1.00), because the motor population moves and
  the read-out loses its direct pathway. The ring lattice is the least robust to random damage. A richer graph
  is not automatically better.
* **Lesions.** No targeted lesion family was shown to hurt more than a matched random lesion at five seeds.
  Hub and sensory-pathway lesions trend that way; memory-pathway lesions hurt less than their random control.
  Progressive damage is strongly seed-dependent: two of five seeds collapse at low severity.

### A correction made during this work

The first full run used held-out **return** as the primary metric everywhere and reported multi-soma
integration (+10.0), uncertainty gating (+0.67), FlyHash (+0.44) and quantisation (+0.004) as `supported`.
Inspection showed that each "gain" came with longer episodes at an unchanged success rate: the existing forage
task pays a positive reward on every step, so a slower agent collects more return. The framework now uses
goal efficiency on `forage` and reports return as a secondary comparison (still visible in the table above),
which reverses three of those four verdicts. The original receipts were overwritten by the corrected run;
the reasoning is recorded here so the mistake is not repeated.

### What would be needed for stronger claims

A trained motor policy (rather than the fixed canonical read-out), longer training before freezing, tasks
whose reward does not favour slowness, matched compute budgets, thirty or more paired seeds as the existing
confirmatory design already targets, and held-out environments. The promotion ledger enforces most of this
automatically; nothing has passed it.

