# SONAPSE-EVO v1.0 Baseline Audit

**Audit Date:** 2026-11-09  
**Baseline Git Commit:** `508838486a600031d95d1b793ad20a2b9984405c`  
**Git Branch:** `main` (Snapshot preserved)  
**Target Upgrade Branch:** `feature/sonapse-evo-2`  
**License:** Apache-2.0  

---

## 1. Environment & Hardware Baseline

- **Operating System:** Windows 10/11 x64
- **Runtime:** Python 3.12 (standard virtual environment)
- **Primary Dependencies:** `torch`, `numpy`, `scipy`, `pyyaml`, `pytest`
- **Execution Target:** CPU-first execution (`torch.device('cpu')`), optional CUDA compatibility.

---

## 2. Test Suite Baseline Execution

Test command: `pytest -q --durations=0`
- **Total Tests:** 17 tests across 7 test modules
- **Status:** 17 passed (100% pass rate)
- **Total Duration:** 7.81s
- **Slowest Test Modules:**
  - `tests/test_graft_training.py::test_frozen_backbone_during_graft_training`: 2.52s
  - `tests/test_atomic_rollback.py::test_stale_parent_digest_rejected`: 0.27s
  - `tests/test_atomic_rollback.py::test_successful_commit_and_rollback`: 0.03s
  - `tests/test_lesions.py::test_inter_region_bridge_lesion`: 0.02s
  - `tests/test_checkpoint_replay.py`: 0.02s

### Individual Passing Tests:
1. `tests/test_atomic_rollback.py::test_successful_commit_and_rollback` - PASSED
2. `tests/test_atomic_rollback.py::test_stale_parent_digest_rejected` - PASSED
3. `tests/test_atomic_rollback.py::test_corrupted_snapshot_rollback` - PASSED
4. `tests/test_causal_probes.py::test_virtual_shadow_gradient` - PASSED
5. `tests/test_causal_probes.py::test_region_ranker_ordering` - PASSED
6. `tests/test_causal_probes.py::test_causal_attribution_engine` - PASSED
7. `tests/test_checkpoint_replay.py::test_snapshot_restore_exact` - PASSED
8. `tests/test_checkpoint_replay.py::test_clone_independence` - PASSED
9. `tests/test_checkpoint_replay.py::test_state_reset_determinism` - PASSED
10. `tests/test_graft_training.py::test_frozen_backbone_during_graft_training` - PASSED
11. `tests/test_graft_training.py::test_graft_activation_ramp` - PASSED
12. `tests/test_lesions.py::test_inter_region_bridge_lesion` - PASSED
13. `tests/test_lesions.py::test_focal_ablation_lesion` - PASSED
14. `tests/test_lesions.py::test_diffuse_synaptic_drop_lesion` - PASSED
15. `tests/test_source_provenance.py::test_donor_adapter_forward` - PASSED
16. `tests/test_source_provenance.py::test_biological_motif_dales_law` - PASSED
17. `tests/test_zero_gain.py::test_substrate_dynamics_identical_with_zero_gain_graft` - PASSED

---

## 3. CLI Demonstration Baseline Measurements

### 3.1 Smoke Verification (`python -m sonapse_evo.cli smoke`)
- **Execution Time:** 1.56s
- **Checks:** 7/7 PASSED
  - Substrate initialization: 768 neurons, 3 regions (256/256/256)
  - Bridge lesion: active edge ratio 92.72%
  - Deficit monitor: baseline score 0.555
  - Candidate generation: 4 candidates, selected Type-A low-rank
  - Isolated sandbox training: loss progression verified
  - Atomic zero-gain commit: `COMMITTED`, Tx ID verified
  - Episode rollout: return 21.36

### 3.2 Lesion-Repair Demonstration (`python -m sonapse_evo.cli lesion-repair --task forage --severity 0.25`)
- **Intact Baseline Return:** `33.44`
- **Damaged Return:** `16.92` (Performance drop: `-16.52`)
- **Causal Diagnostic Attribution:** Source `R0_P0` (Sensory) $\to$ Target `R2_P0` (Motor)
- **Repaired Return:** `34.66` (Restoration: `+17.74`)
- **Recovery Fraction $\mathcal{R}$:** `107.4%`
- **Transaction Outcome:** `COMMITTED`

### 3.3 10-Arm Benchmark Baseline (`python -m sonapse_evo.cli benchmark --seeds 5`)
- **A0 (Damaged Backbone):** $32.27 \pm 13.80$
- **A1 (Fine-Tuning):** $32.27 \pm 13.80$
- **A2 (Random Sparse Graft):** $32.27 \pm 13.79$
- **A3 (Matched Learned Adapter):** $32.27 \pm 13.83$
- **A4 (Gradient Rewiring / RigL Demo):** $32.07 \pm 13.88$
- **A5 (Correlation Selected):** $32.29 \pm 13.80$
- **A6 (SONAPSE-EVO Causal Grafting):** $31.45 \pm 2.93$ (Highest stability)
- **A7 (Biological Motif Graft):** $37.01 \pm 11.29$
- **A8 (AETHER Donor Adapter):** $33.29 \pm 2.91$
- **A9 (Oracle Lesion Graft):** $31.33 \pm 2.80$

---

## 4. Known Scientific & Implementation Weaknesses Identified in Baseline

1. **A4 RigL Authenticity:** The v1.0 A4 arm applied fixed weight scaling rather than genuine gradient-based dynamic pruning and regrowth with scheduled masks.
2. **A1-A3 Training Fidelity:** Fine-tuning and adapter baselines used simplified optimization loops that did not exhaustively train matched budgets.
3. **A5 Covariance Sampling:** Causal vs. correlation placement needed a real streaming covariance estimator over trajectory buffers.
4. **Lesion-Free Discovery Absent:** v1.0 required an artificial lesion to trigger candidate generation. No autonomous discovery mode existed for healthy networks.
5. **Single-Objective Dominance:** Candidate selection relied on scalar utility without Pareto frontier tracking across latency, memory, parameters, and generalization.
6. **Synthetic vs. Confirmatory Data:** Fixtures and measured results were not formally differentiated into typed evidence tiers.
