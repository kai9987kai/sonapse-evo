# SONAPSE-EVO 2.0 Master Implementation Plan

**Project:** Autonomous Connectome Discovery, Structural Evolution, Causal Repair and Continual Adaptation  
**Target:** SONAPSE-EVO 2.0  
**Repository:** https://github.com/kai9987kai/sonapse-evo  
**Base Commit:** `508838486a600031d95d1b793ad20a2b9984405c`  

---

## 1. Executive Summary & Central Hypothesis

SONAPSE-EVO 1.0 established that an artificial recurrent neural network can autonomously diagnose functional deficits resulting from physical injury (lesions) and install targeted, zero-gain causal neural grafts to restore functionality with atomic transactional safety.

**SONAPSE-EVO 2.0** broadens this framework to answer the central research question:
> *Can an artificial neural system discover, optimize, and maintain computational structures through ongoing, causally evaluated growth, pruning, rewiring, plasticity, and modular reorganization—without requiring prior simulated damage?*

SONAPSE-EVO 2.0 introduces:
1. **Dual Evolutionary Operation:** Mode A (Deficiency-driven Repair) and Mode B (Opportunity-driven Discovery).
2. **Generalized Connectome Graph:** $G = (V, E, T, W, P, M)$ with 20 bounded structural variation operators.
3. **Multi-Objective Optimization:** Pareto frontier search evaluating performance, out-of-distribution generalization, adaptation efficiency, forgetting resistance, latency, memory, energy cost, and dynamical stability.
4. **Authentic Benchmark Suite:** Authentic RigL gradient rewiring, genuine fine-tuning, real covariance tracking, and rigorous matched controls.
5. **Typed Evidence Framework:** Formal data hierarchy (`SYNTHETIC_FIXTURE`, `SMOKE_TEST`, `SIMULATED_ASSAY`, `PAIRED_EXPERIMENT`, `CONFIRMATORY_RESULT`) guaranteeing scientific integrity.
6. **Observatory 2.0:** Extended live interactive interface tracking population genetics, Pareto frontiers, real-time connectome rewiring, and transactional safety ledgers.

---

## 2. Architectural Blueprint

```
                      TASK & SENSORY STREAMS
                                 │
                                 ▼
                     RECURRENT RATE SUBSTRATE
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
           Motor Actuation              Connectome Telemetry
                  │                             │
                  ▼                             ▼
             Environment               Unified Diagnostics
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             MODE A: REPAIR MONITOR                            MODE B: DISCOVERY MONITOR
           (6D Lesion Deficit Vector)                       (Opportunity & Capacity Vector)
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                                    STRUCTURAL PROPOSAL ENGINE
                                                │
                      ┌─────────────────────────┼─────────────────────────┐
                      ▼                         ▼                         ▼
                 Graft Growth            Pruning & Rewiring        Modular Reorganization
                 (Types A-D)            (Synaptic/Neuron/RigL)     (Split/Merge/Specialize)
                      │                         │                         │
                      └─────────────────────────┼─────────────────────────┘
                                                ▼
                                   POPULATION GENOTYPE POOL
                               (Diversity, Elitism, Deduplication)
                                                │
                                                ▼
                                   MULTI-FIDELITY SANDBOXES
                           Stage 0: Legality & Precondition Check
                           Stage 1: Low-Cost Heuristic Screening
                           Stage 2: Short-Horizon Compute Budget
                           Stage 3: Paired Counterfactual Assay
                           Stage 4: Generalization & Multi-Task Retention
                           Stage 5: Confirmatory Replication Assay
                                                │
                                                ▼
                                  SCIENTIFIC PROMOTION GATE
                                (Typed Evidence Verification)
                                                │
                                ┌───────────────┴───────────────┐
                                ▼                               ▼
                         Atomic Commit                   Failure Archive
                                │                               │
                                ▼                               ▼
                       Zero-Gain Activation              Mutation Ledger
                                │
                                ▼
                       Retention Probation
                                │
                                ▼
                       Commit or Rollback
```

---

## 3. Implementation Phases

### Phase 1: Baseline Preservation & Audit Infrastructure
- Audit baseline Git commit `508838486a600031d95d1b793ad20a2b9984405c`.
- Establish `feature/sonapse-evo-2` branch.
- Generate mandatory audit documents (`BASELINE_AUDIT.md`, `FEATURE_PRESERVATION_MATRIX.md`, `EVOLUTION_V2_MASTER_PLAN.md`, `RESEARCH_NOVELTY_AUDIT.md`, `REGRESSION_PROTOCOL.md`).
- Implement `legacy_v1` profile and execution flags.

### Phase 2: Evidence Integrity & Transaction Hardening
- Implement typed evidence levels (`EvidenceTier` enum).
- Enhance `TransactionManager` to enforce SHA-256 parent lineage, stability invariants, zero-gain validation, and evidence promotion gating.
- Strengthen sandbox isolation and memory bounds.

### Phase 3: Generalized Connectome Graph & Structural Operators
- Implement `GraphTopology` $G = (V, E, T, W, P, M)$ extending `TopologyState`.
- Implement 20 bounded, typed, reversible structural variation operators:
  1. Add Synapse
  2. Remove Synapse
  3. Rewire Synapse
  4. Rescale Connection
  5. Add Neuron
  6. Prune Inactive Neuron
  7. Split Pathway
  8. Local Recurrent Circuit
  9. Grow Microcircuit
  10. Add Low-Rank Bridge
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
- Comprehensive unit tests verifying each operator's bounds, legality, and reversibility.

### Phase 4: Autonomous Architecture Discovery Engine
- Implement `DiscoveryOpportunityMonitor` tracking performance plateaus, capacity bottlenecks, redundancy, and drift.
- Implement population management with genetic diversity, tournament selection, crossover, and elitism.
- Implement multi-fidelity evaluation pipeline (Stages 0–5).
- Implement multi-objective Pareto optimization across 8 fitness axes.

### Phase 5: Authentic Scientific Benchmarks (A0–A9 Upgrade)
- Upgrade Arm A4 to authentic RigL (gradient-based pruning/regrowth schedule with dense gradient computation over masked sparse topology).
- Upgrade Arm A1 to authentic fine-tuning with matched parameter steps.
- Upgrade Arm A2 to authentic random graph growth with matched parameter count.
- Upgrade Arm A5 to streaming covariance estimation.
- Provide clear isolation between authentic transfer and proxy demonstrations.

### Phase 6: Extended CLI & Observatory 2.0
- Add `discover` and `evolve` CLI commands alongside existing `smoke`, `lesion-repair`, `benchmark`, and `serve`.
- Update Observatory dashboard with live Pareto front plotting, population genotype viewer, and live connectome graph inspection.
- Full verification against legacy baselines (0 regressions).
