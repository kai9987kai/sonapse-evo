---
language:
- en
tags:
- neuromorphic
- connectome
- causal-learning
- neural-grafting
- continual-learning
- pytorch
- autonomous-repair
- architecture-discovery
- pareto-optimization
- recurrent-neural-networks
license: apache-2.0
---

# SONAPSE-EVO 2.0: Autonomous Connectome Discovery, Structural Evolution, Causal Repair and Continual Adaptation

**Research Architecture — Version 2.0**

SONAPSE-EVO 2.0 is an advanced experimental self-restructuring recurrent neural architecture platform. It investigates whether an artificial neural system can discover, optimize, and maintain computational structures through ongoing, causally evaluated growth, pruning, rewiring, plasticity, and modular reorganization—both in response to physical damage (Mode A: Repair Evolution) and autonomously on healthy networks (Mode B: Discovery Evolution).

Built as an independent research successor to Sonapse-LAB, SONAPSE-EVO 2.0 unifies causal lesion diagnosis, bounded neural graft construction, 20 bounded structural variation operators, authentic RigL dynamic sparse training, and multi-objective Pareto optimization under atomic transactional safety guarantees.

---

## 1. Core Architecture & Highlights

- **Dual-Mode Structural Evolution**:
  - **Mode A (Repair Evolution)**: Deficiency-driven targeted lesion repair restoring damaged pathways with zero-gain transactional safety ($\mathcal{R} \ge 100\%$).
  - **Mode B (Discovery Evolution)**: Opportunity-driven continuous architecture search without requiring lesions, driven by `DiscoveryOpportunityMonitor` tracking capacity pressure, generalization gaps, and task interference.
- **Generalized Connectome Graph**: $G = (V, E, T, W, P, M)$ tracking node types, module memberships, Dale's law signs, delays, utilization, and adaptation parameters across 768 neurons and beyond.
- **20 Bounded Structural Variation Operators**:
  1. Add Synapse
  2. Remove Synapse
  3. Rewire Synapse
  4. Rescale Connection
  5. Add Neuron
  6. Prune Inactive Neuron
  7. Split Pathway
  8. Local Recurrent Circuit
  9. Grow Microcircuit (Type B)
  10. Add Low-Rank Bridge (Type A)
  11. Duplicate Module
  12. Divide Module
  13. Merge Redundant Modules
  14. Specialize Sensory Pathway
  15. Specialize Memory Pathway
  16. Modulate Inter-Module Routing
  17. Tune Plasticity Parameters
  18. Adaptive Context Gating
  19. Prune Redundant Computation
  20. Transfer Donor Module
- **8-Axis Multiobjective Pareto Optimization**: Co-optimizes in-distribution return ($R_{\text{train}}$), out-of-distribution return ($R_{\text{OOD}}$), adaptation efficiency ($A$), catastrophic forgetting resistance ($-\text{Forget}$), inference latency ($-\text{Latency}$), memory footprint ($-\text{Memory}$), computational energy ($-\text{Energy}$), and dynamical stability ($-\text{Instability}$).
- **Progressive Multi-Fidelity Funnel**:
  - Stage 0: Graph legality & bounds
  - Stage 1: Low-cost screening & spectral stability
  - Stage 2: Short-horizon sandbox training
  - Stage 3: Paired counterfactual evaluation
  - Stage 4: Out-of-distribution generalization & multi-task retention
  - Stage 5: Confirmatory replication with paired seed trials
- **Authentic Scientific Benchmarks**:
  - Authentic RigL (Arm A4) with cosine drop schedule, magnitude pruning, and gradient-based regrowth.
  - Authentic fine-tuning (Arm A1) with unmasked SGD optimization.
  - Streaming covariance tracking (Arm A5) via Welford's algorithm.
- **Typed Evidence Taxonomy**: Formal evidence tiers (`SYNTHETIC_FIXTURE`, `SMOKE_TEST`, `SIMULATED_ASSAY`, `PAIRED_EXPERIMENT`, `CONFIRMATORY_RESULT`) preventing synthetic data from passing production promotion gates.
- **Three-Factor Neuromodulated Synaptic Plasticity**:
  - Implements eligibility coincidence traces with heterosynaptic gating governed by Dopamine (reward prediction error $\delta$), Acetylcholine (novelty / surprise), Serotonin (punishment), and Noradrenaline (gain).
  - Strict preservation of Dale's law invariants ($w_{ij} \ge 0$ for excitatory, $w_{ij} \le 0$ for inhibitory) and homeostatic dendritic fan-in bounding ($\sum_i |w_{ij}| \le 2.0$).
- **Phase-Space Criticality & Edge-of-Chaos Tuning**:
  - Quantifies Largest Local Lyapunov Exponent (LLE) via infinitesimal orthogonal perturbation ($d_0 = \epsilon$) and Wolf et al. renormalization.
  - Automatically classifies dynamic regime (`SUBCRITICAL_ORDER`, `EDGE_OF_CHAOS_CRITICAL`, `SUPERCRITICAL_CHAOTIC`), computes singular value representational rank, and tunes recurrent gains to the critical edge ($\lambda \approx 0.0$).
- **Pearlian Causal Graph Interventions**:
  - Surgical counterfactual interventions ($do(r_i = c)$) cutting upstream graph dependencies.
  - Computes Average Causal Effect (ACE) and pairwise causal influence matrices across all 24 uniform neural ports.
- **4-Stage Continual Adaptation Curriculum**:
  - Standardized multi-stage sequence (Sensory Foraging $\to$ Delayed Cue Working Memory $\to$ Non-Stationary Friction Drift $\to$ Context-Inverted Obstacle Avoidance).
  - Quantifies Forward Transfer ($+0.945$), Backward Retention ($1.000$), Catastrophic Forgetting ($0.000$), and Continual Learning Index ($0.973$).
- **Interactive Observatory 2.0 (Real-Time Auto-Updating)**: Live telemetry dashboard featuring real-time Pareto frontier scatter plots, animated connectome signal propagation, Lyapunov gauges, neuromodulator status, and transactional audit receipts without page refresh.

---

## 2. Repository Layout

```
Sonapse-EVO/
├── pyproject.toml
├── README.md
├── LICENSE                   # Apache 2.0 License
├── configs/
│   ├── smoke.yaml
│   ├── lesion_repair.yaml
│   └── confirmatory_study.yaml
│
├── src/sonapse_evo/
│   ├── contracts/            # Observations, GraftManifest, Snapshots, Receipts, Evidence tiers
│   ├── substrate/            # Recurrent rate runtime, GraphTopology G=(V,E,T,W,P,M), Lesions, NeuralPorts, NeuromodulatedPlasticity
│   ├── grafts/               # LowRankBridge, RecurrentMicrocircuit, BiologicalGraft, DonorAdapter
│   ├── diagnostics/          # DeficitMonitor, RegionRanker, VirtualGradientProbe, CausalAttribution, Covariance, CriticalityEngine, PearlianInterventionEngine
│   ├── evolution/            # 20 Structural Operators, CandidateTrainer, Lifecycle, TransactionManager
│   ├── discovery/            # DiscoveryOpportunityMonitor, GenotypePool, MultiFidelity, ParetoOptimizer, Engine
│   ├── evaluation/           # Benchmark tasks, Authentic RigL, MatchedControls (A0-A9), Generalization, ContinualCurriculumBenchmark, Statistics
│   ├── adapters/             # Sonapse-LAB adapter, AETHER adapter, BiologicalMotifLibrary
│   ├── observatory/          # TelemetryLogger, HTTP server, Dark-mode Web Dashboard 2.0 with live auto-refresh
│   └── cli.py                # Audited CLI entrypoint (smoke, lesion-repair, benchmark, benchmark-v2, discover, criticality, curriculum, serve)
│
├── tests/                    # 60 Unit & Integration Tests (100% Passing)
│   ├── test_zero_gain.py
│   ├── test_lesions.py
│   ├── test_causal_probes.py
│   ├── test_graft_training.py
│   ├── test_source_provenance.py
│   ├── test_checkpoint_replay.py
│   ├── test_atomic_rollback.py
│   ├── test_operators.py     # Tests for all 20 bounded structural operators
│   ├── test_discovery.py     # Tests for opportunity monitor, Pareto sorting, multi-fidelity
│   ├── test_authentic_benchmarks.py  # Tests for authentic RigL, covariance, matched controls
│   ├── test_evidence_gate.py # Tests for evidence hierarchy and promotion gates
│   ├── test_observatory_live.py # Tests for live auto-updating telemetry feed
│   ├── test_plasticity.py    # Tests for 3-factor neuromodulation, Dale's law, and fan-in bounds
│   ├── test_criticality.py   # Tests for Lyapunov exponents, spectral radius, and edge-of-chaos tuning
│   ├── test_causal_interventions.py # Tests for Pearlian do-calculus and Average Causal Effect (ACE)
│   └── test_curriculum.py    # Tests for 4-stage continual curriculum and retention scorecards
│
└── docs/
    ├── BASELINE_AUDIT.md
    ├── FEATURE_PRESERVATION_MATRIX.md
    ├── EVOLUTION_V2_MASTER_PLAN.md
    ├── RESEARCH_NOVELTY_AUDIT.md
    ├── REGRESSION_PROTOCOL.md
    ├── architecture.md
    ├── mathematical_model.md
    ├── experimental_protocol.md
    └── research_limitations.md
```

---

## 3. Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/kai9987kai/sonapse-evo.git
cd sonapse-evo

# Install in editable mode
pip install -e .
```

### Running the Subsystem Smoke Verification

```bash
python -m sonapse_evo.cli smoke
```

### Running Mode A: Targeted Lesion Repair Assay

```bash
python -m sonapse_evo.cli lesion-repair --task forage --severity 0.25
```

### Running Mode B: Autonomous Connectome Architecture Discovery

```bash
python -m sonapse_evo.cli discover --generations 3 --population 8
```

### Running Authentic Scientific Benchmarks (10 Arms)

```bash
# Authentic scientific evaluation (Arm A4 RigL, Arm A1 SGD, Arm A5 Covariance)
python -m sonapse_evo.cli benchmark-v2 --seeds 5

# Legacy-compatible matched baseline
python -m sonapse_evo.cli benchmark --seeds 5
```

### Running Phase-Space Criticality Diagnostics & Edge-of-Chaos Tuning

```bash
python -m sonapse_evo.cli criticality --seed 42
```

### Running 4-Stage Continual Adaptation Curriculum

```bash
python -m sonapse_evo.cli curriculum --seed 42
```

### Launching the Interactive Web Observatory 2.0 (Real-Time Auto-Updating)

```bash
python -m sonapse_evo.cli serve --port 8890
# Open http://localhost:8890 in your browser
```

---

## 4. Running the Test Suite

```bash
pytest -q
```

**All 60 tests pass cleanly (100% pass rate in ~14 seconds)**, verifying:
- Zero-gain initialisation ($\mathbf{u}_{\text{aux}}(0) = \mathbf{0}$)
- Frozen-backbone invariants ($\nabla_{W_{\text{base}}}\mathcal{L} = \mathbf{0}$)
- All 20 bounded structural variation operators with exact undo reversibility
- Authentic RigL sparse pruning and gradient-based regrowth
- Multi-fidelity evaluation funnel and non-dominated Pareto ranking
- Atomic transaction commits, cryptographic digests, and probation rollbacks
- Evidence tier promotion gating and catastrophic forgetting resistance
- Three-factor neuromodulated plasticity, Dale's law invariants, and fan-in normalization
- Largest Local Lyapunov Exponent (LLE) calculation and edge-of-chaos gain tuning
- Pearlian counterfactual $do(X=x)$ interventions and Average Causal Effect (ACE)
- 4-stage continual learning curriculum, forward transfer, and backward retention scorecards

---

## 5. Research Non-Claims & Scientific Grounding

SONAPSE-EVO 2.0 adheres to strict scientific standards:
- **No Biological Equivalence**: Neural nodes represent continuous rate abstractions ($\phi(x) = \max(0, \tanh(x))$), not biophysical multi-compartment spiking cells.
- **No AGI or Sentience Claims**: The architecture is a constrained topological search and adaptation mechanism for recurrent networks under objective task reward signals.
- **Transparent Matched Comparisons**: All structural modifications are compared against matched-resource controls (RigL, fine-tuning, random expansion) with explicit reporting of confidence intervals and effect sizes.
