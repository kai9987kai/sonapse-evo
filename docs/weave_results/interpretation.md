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
