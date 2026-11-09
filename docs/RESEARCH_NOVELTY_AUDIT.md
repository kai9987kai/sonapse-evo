# SONAPSE-EVO 2.0 Research Novelty Audit & Non-Claims

**Document Purpose:** Clear delineation of research hypotheses, technical antecedents, theoretical limits, and scientific boundaries.

---

## 1. Explicit Research Non-Claims & Scientific Boundaries

To preserve strict scientific integrity, SONAPSE-EVO 2.0 makes the following explicit non-claims:

1. **No Biological Equivalence:** While models employ concepts inspired by biological systems (e.g., Dale's law, sparse recurrent connectivity, synaptic pruning, neuromodulatory tone), the network nodes are abstract mathematical rate equations ($\phi(x) = \max(0, \tanh(x))$), not biophysical Hodgkin-Huxley or multi-compartment spiking neurons.
2. **No Claim of Artificial General Intelligence (AGI) or Sentience:** The architecture is a constrained mathematical search system for sparse recurrent weight topologies under reinforcement and task feedback. It exhibits no consciousness, subjective experience, or artificial general intelligence.
3. **No Unsubstantiated Breakthrough Claims:** All algorithmic gains are evaluated through matched baseline comparisons ($p$-values, effect sizes, bootstrap confidence intervals) and documented with explicit trade-offs (e.g., computational search cost vs. parameter efficiency).
4. **No Biological Morphogenesis:** Structural variation operators (adding neurons, splitting modules) are computational graph transformations executed in discrete sandbox steps, not continuous physical tissue morphogenesis.

---

## 2. Antecedents & Prior Art

SONAPSE-EVO builds directly upon established foundations in neural architecture search, sparse network rewiring, and modular continual learning:

| Antecedent / Field | Key Works | Contrast with SONAPSE-EVO 2.0 |
|:---|:---|:---|
| **Function-Preserving Transformations** | Net2Net (Chen et al., 2016); Network Morphism (Wei et al., 2016) | Net2Net focused on feedforward/convolutional expansion with identical function. SONAPSE-EVO generalizes this to recurrent rate substrates via zero-gain auxiliary injection ($\alpha=0 \implies \mathbf{u}_{\text{aux}}=\mathbf{0}$) with transactional atomic commits. |
| **Dynamic Sparse Rewiring** | Deep Rewiring (Bellec et al., 2018); RigL (Evci et al., 2020) | RigL applies periodic gradient-based connection regrowth to a fixed-size parameter budget. SONAPSE-EVO incorporates authentic RigL as a baseline arm while introducing causal perturbation probes and macro-modular graft topologies. |
| **Modular & Continual Learning** | Progressive Networks (Rusu et al., 2016); Modular Continual Learning (Ostapenko et al., 2021) | Progressive networks duplicate entire columns per task. SONAPSE-EVO dynamically targets low-rank bridges and localized microcircuits only between causally identified functional deficit/opportunity ports. |
| **Neural Architecture Search (NAS)** | DARTS (Liu et al., 2018); Regularized Evolution for NAS (Real et al., 2019) | Traditional NAS searches a fixed cell topology prior to full training. SONAPSE-EVO executes continuous, runtime life-cycle discovery and repair on an actively executing recurrent connectome with rollback safety. |
| **Causal Circuit Analysis** | Causal Tracing (Meng et al., 2022); Activation Patching (Wang et al., 2022) | Activation patching identifies causal circuits in frozen language models. SONAPSE-EVO uses counterfactual virtual shadow gradients at runtime to direct structural growth. |

---

## 3. Core Proposed Scientific Contributions of v2.0

1. **Unified Dual-Mode Structural Evolution:** Integration of deficiency-driven repair (Mode A) and opportunity-driven discovery (Mode B) within a single typed computational graph representation.
2. **Zero-Gain Transactional Safety Contract:** Formal proof and test verification that graph modifications can be committed with exact functional invariance at $t=0$, followed by supervised or bounded autonomous activation ramps and atomic rollback on contract breach.
3. **Multi-Fidelity Multi-Objective Pareto Discovery:** Co-optimizing task reward, held-out generalisation, memory footprint, inference latency, energy cost proxies, and spectral dynamical stability.
4. **Standardized Matched-Resource Benchmark Battery:** Fair comparisons where fine-tuning, RigL, random expansion, covariance placement, and causal grafting operate under strictly matched floating-point operations and parameter update budgets.
