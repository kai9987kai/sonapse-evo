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
- recurrent-neural-networks
license: apache-2.0
---

# SONAPSE-EVO: Causal Neural Circuit Evolution, Autonomous Repair and Adaptive Intelligence

**Research Architecture — Version 1.0**

SONAPSE-EVO is an experimental self-restructuring recurrent neural architecture designed to investigate whether an artificial neural system can diagnose its own functional weaknesses, identify where additional neural computation would help, develop targeted circuits, and preserve useful changes under transactional safety guarantees.

Built as an independent research successor to Sonapse-LAB, SONAPSE-EVO provides a modular framework for causal connectome diagnosis, bounded neural graft construction, isolated sandbox training, and transactional lifecycle management.

---

## 1. Core Architecture & Highlights

- **Causal Neural Deficiency Engine**: Lesion-blind diagnostic monitoring using 6-component exponentially weighted diagnostics $\mathbf{d}_t = [e_{\mathrm{pred}}, e_{\mathrm{policy}}, u_{\mathrm{unc}}, c_{\mathrm{rep}}, r_{\mathrm{task}}, s_{\mathrm{stab}}]^\top$, persistent threshold triggers, and virtual shadow-gradient candidate ranking.
- **Three Core Graft Architectures**:
  - **Type A (Sparse Low-Rank Bridge)**: Factorized $U_g V_g$ projections reducing weight parameters by 50% ($k(|S|+|T|)$ vs $|S||T|$).
  - **Type B (Recurrent Microcircuit)**: Dedicated internal memory state $\mathbf{q}_t$ for temporal delay tasks.
  - **Type C (Biological Motif)**: Signed subcircuits obeying Dale's law ($W_{ij} = s_j a_{ij}, a_{ij} \ge 0$).
  - **Type D (Donor Adapter)**: Representation alignment via regularized ridge regression for external donors (e.g. AETHER / Poseidon).
- **Exact Function-Preserving Zero-Gain Ramp**: Grafts are installed with $\alpha_g(0) = 0 \implies \mathbf{I}_g(0) = \mathbf{0}$, ensuring mathematical invariance of baseline dynamics upon commit, followed by controlled linear activation ramps.
- **Isolated Sandbox Training**: Backbone weights ($W_b, W_p, B, \mathbf{b}$) are strictly frozen during graft parameter fitting.
- **Atomic Transaction Protocol**: SHA-256 parent digest verification, pre-commit snapshots, and automatic rollback on validation failure.
- **10-Arm Matched Benchmark Controls**: Standardized evaluation comparing causal grafting against frozen damaged (A0), fine-tuning (A1), random grafts (A2), capacity-matched adapters (A3), RigL rewiring (A4), correlation grafting (A5), biological motifs (A7), donor transfer (A8), and oracle bounds (A9).
- **Interactive Observatory**: Real-time scientific telemetry web server and telemetry logger.

---

## 2. Repository Layout

```
Sonapse-EVO/
├── pyproject.toml
├── README.md
├── configs/
│   ├── smoke.yaml
│   ├── lesion_repair.yaml
│   └── confirmatory_study.yaml
│
├── src/sonapse_evo/
│   ├── contracts/            # Observations, GraftManifest, Snapshots, Receipts
│   ├── substrate/            # Recurrent rate runtime, Lesions, NeuralPorts, TopologyState
│   ├── grafts/               # LowRankBridge, RecurrentMicrocircuit, BiologicalGraft, DonorAdapter
│   ├── diagnostics/          # DeficitMonitor, RegionRanker, VirtualGradientProbe, CausalAttribution
│   ├── evolution/            # CandidateGenerator, CandidateTrainer, Selector, Lifecycle, Transactions
│   ├── evaluation/           # Benchmark tasks, MatchedControls (A0-A9), Retention, Statistics
│   ├── adapters/             # Sonapse-LAB adapter, AETHER adapter, BiologicalMotifLibrary
│   ├── observatory/          # TelemetryLogger, HTTP server, Dark-mode Web Dashboard
│   └── cli.py                # Audited CLI entrypoint
│
├── tests/                    # 17 Unit & Integration Tests (100% Passing)
│   ├── test_zero_gain.py
│   ├── test_lesions.py
│   ├── test_causal_probes.py
│   ├── test_graft_training.py
│   ├── test_source_provenance.py
│   ├── test_checkpoint_replay.py
│   └── test_atomic_rollback.py
│
└── docs/
    ├── architecture.md
    ├── mathematical_model.md
    ├── experimental_protocol.md
    └── research_limitations.md
```

---

## 3. Quick Start

### Installation

```bash
# Clone or navigate to the workspace
cd "c:\Users\kai99\Desktop\New folder (13)"

# Install in editable mode
pip install -e .
```

### Running the Subsystem Smoke Verification

```bash
python -m sonapse_evo.cli smoke
```

### Running a Lesion Repair Assay

```bash
python -m sonapse_evo.cli lesion-repair --task forage --severity 0.20
```

### Running the 10-Arm Matched Benchmark Comparison

```bash
python -m sonapse_evo.cli benchmark --seeds 5
```

### Launching the Scientific Research Observatory

```bash
python -m sonapse_evo.cli serve --port 8890
# Open http://localhost:8890 in your browser
```

---

## 4. Running the Test Suite

```bash
pytest -v
```

All 17 tests verify zero-gain initialisation, lesion immutability, virtual shadow-gradient estimation, frozen-backbone training isolation, snapshot digest integrity, and atomic transaction rollback.

---

## 5. Mathematical Summary

The recurrent substrate evolves under continuous rate dynamics:

$$\mathbf{r}_{t+1} = \operatorname{clip}_{[0, 1]} \left[ (1 - \eta)\mathbf{r}_t + \eta \phi\left( (W_b^{(\ell)} + W_p)\mathbf{r}_t + B\mathbf{o}_t + \sum_{g=1}^K \mathbf{I}_g(t) + \mathbf{b} \right) \right]$$

with $\phi(x) = \max(0, \tanh(x))$ and $\|\sum_g \mathbf{I}_g(t)\|_\infty \le 0.10$.

Targeted recovery is quantified by:

$$\mathcal{R} = \frac{R_R - R_D}{R_I - R_D}$$

where $R_I, R_D, R_R$ denote intact, damaged, and repaired returns.
