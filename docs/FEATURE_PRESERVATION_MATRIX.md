# SONAPSE-EVO 2.0 Feature Preservation Matrix

This matrix formalizes the zero-regression and backward compatibility guarantees for SONAPSE-EVO 2.0. Every preserved subsystem must maintain complete API, contract, and behavior parity when executed in `legacy_v1` mode or with default v1 parameters.

---

## 1. Subsystem Preservation Map

| Subsystem | Component / Class | v1 Contract & Invariants | v2 Preservation Guarantee | Validating Tests |
|:---|:---|:---|:---|:---|
| **Substrate** | `RecurrentSubstrate` | $\mathbf{x}_{t+1} = (1-\frac{\Delta t}{\tau})\mathbf{x}_t + \frac{\Delta t}{\tau}[\mathbf{W}_{\text{base}}\mathbf{r}_t + \mathbf{B}\mathbf{s}_t + \mathbf{b} + \mathbf{u}_{\text{aux}}]$ | Unchanged numerical dynamics; frozen backbone $\nabla_{\mathbf{W}_{\text{base}}}\mathcal{L} = \mathbf{0}$; zero-gain auxiliary current; snapshot & clone. | `tests/test_zero_gain.py`, `tests/test_checkpoint_replay.py` |
| **Substrate** | `TopologyState` | Sparse graph with 768 neurons, 3 regions, Dale's law signs, descending tracts. | Retained as default graph; extended via typed multi-graph wrapper. | `tests/test_lesions.py` |
| **Substrate** | `LesionEngine` | Bridge, focal, diffuse lesions with deterministic RNG seeds. | Retained identically. Mode A (Repair) uses this directly. | `tests/test_lesions.py` |
| **Substrate** | `NeuralPort` | Uniform port partitioning (e.g. 8 ports per region, 32 neurons each). | Retained; port indices deterministic. Extended to support dynamic ports. | `tests/test_causal_probes.py` |
| **Grafts** | `LowRankBridge` (Type A) | $\mathbf{u}_T = \alpha \mathbf{V} \phi(\mathbf{U} \mathbf{r}_S)$; initialized with $\alpha=0$ or $\mathbf{V}=0$. | Retained; zero-gain guarantee verified. | `tests/test_zero_gain.py`, `tests/test_graft_training.py` |
| **Grafts** | `RecurrentMicrocircuit` (Type B) | Internal memory state $\mathbf{q}_{t+1} = \tanh(\mathbf{W}_q \mathbf{q}_t + \mathbf{W}_{\text{in}} \mathbf{r}_S)$. | Retained; zero-gain output projection. | `tests/test_graft_training.py` |
| **Grafts** | `BiologicalMotifGraft` (Type C) | Signed E/I subcircuits obeying Dale's law ($W_{ij} = s_j a_{ij}$). | Retained; Dale's sign constraint enforced. | `tests/test_source_provenance.py` |
| **Grafts** | `DonorAdapter` (Type D) | Regularized linear projection from external donor space. | Retained; compatible with Sonapse-LAB and Supermix-Aether. | `tests/test_source_provenance.py` |
| **Grafts** | `GraftRegistry` | Factory registry for creating and tracking grafts. | Retained; v2 adds dynamic search operators. | `tests/test_graft_training.py` |
| **Diagnostics** | `DeficitMonitor` | 6D deficit vector $\mathbf{d}_t = [e_{\text{pred}}, e_{\text{policy}}, u_{\text{unc}}, c_{\text{rep}}, r_{\text{task}}, s_{\text{stab}}]^\top$. | Retained; Mode A deficit tracking intact. | `tests/test_causal_probes.py` |
| **Diagnostics** | `RegionRanker` | Virtual shadow-gradient score ranking between region ports. | Retained; ranking ordering preserved. | `tests/test_causal_probes.py` |
| **Diagnostics** | `VirtualGradientProbe` | Backpropagates counterfactual gradient $\nabla_{\mathbf{u}_T}\mathcal{L}_{\text{task}}$. | Retained; counterfactual gradient verification. | `tests/test_causal_probes.py` |
| **Diagnostics** | `CausalAttributionEngine` | Identifies best (source, target) port pair. | Retained without hardcoding lesion locations. | `tests/test_causal_probes.py` |
| **Evolution** | `CandidateGenerator` | Proposes graft types and port pairs based on diagnostics. | Retained for Mode A repair; extended for Mode B discovery. | `tests/test_graft_training.py` |
| **Evolution** | `CandidateTrainer` | Isolated sandbox training; backbone weights frozen. | Retained; supports fine-tuning & multi-step search. | `tests/test_graft_training.py` |
| **Evolution** | `CandidateSelector` | Scalar utility $U = \Delta \text{Return} - \lambda \cdot \text{Cost}$. | Preserved under `legacy_v1_selector`; extended to Pareto search. | `tests/test_atomic_rollback.py` |
| **Evolution** | `TransactionManager` | Pre-commit snapshot, parent digest verification, probation rollback. | Retained with enhanced multi-tier validation gates. | `tests/test_atomic_rollback.py` |
| **Evaluation** | `MatchedControlsFactory` (A0–A9) | Matched control arms (A0 damaged to A9 oracle). | Preserved as baseline suite; authentic RigL and trained controls added. | `tests/test_checkpoint_replay.py` |
| **Evaluation** | `RetentionEvaluator` | Catastrophic forgetting check on non-target tasks. | Retained; multi-task retention battery added. | `tests/test_checkpoint_replay.py` |
| **Evaluation** | `PromotionGate` | Validates recovery fraction $\mathcal{R} \ge \theta_{\text{recov}}$. | Extended with typed evidence levels (`SYNTHETIC` to `CONFIRMATORY`). | `tests/test_atomic_rollback.py` |
| **CLI** | `sonapse_evo.cli` | Commands: `smoke`, `lesion-repair`, `benchmark`, `serve`. | All v1 CLI commands and arguments preserved with zero regressions. | CLI smoke & repair baselines |

---

## 2. Backward Compatibility Contract (`legacy_v1` Profile)

When `SONAPSE_LEGACY_V1=1` or `profile="legacy_v1"` is passed:
1. Substrate dynamics use standard 768-neuron 3-region unweighted rate equations.
2. Graft proposals execute only when deficit thresholds are triggered (Mode A only).
3. Candidate selection follows the original scalar utility objective.
4. Benchmark comparison arms A0–A9 return standard v1 comparative returns.
5. All JSON/YAML schemas, logs, receipts, and snapshot formats maintain exact key-value backward compatibility.
