# SONAPSE-EVO 2.0 Regression Prevention Protocol

This protocol defines the mandatory test batteries, gating criteria, and validation procedures required before any commit or release of SONAPSE-EVO 2.0.

---

## 1. Gating Hierarchy

Every code modification must pass three hierarchical verification gates:

```
[ Gate 1: Deterministic Contract Gate ]
  └─ pytest -q (All unit & integration tests pass with 0 failures)
  └─ Exact mathematical invariants verified:
     • Zero-gain insertion: ||u_aux(0)|| == 0.0
     • Frozen backbone: grad_W_base == None or 0.0
     • Exact snapshot restoration: ||x_restored - x_original|| == 0.0
     • Rollback purity: state after rollback == state before commit

[ Gate 2: Legacy Functional Gate ]
  └─ python -m sonapse_evo.cli smoke
     • All 7 checks PASS within <= 3.0 seconds
  └─ python -m sonapse_evo.cli lesion-repair --task forage --severity 0.25
     • Deficit drop > 10.0
     • Recovery fraction R >= 90.0%
     • Transaction status == COMMITTED

[ Gate 3: V2 Capability & Evidence Gate ]
  └─ python -m sonapse_evo.cli discover --budget 10
     • Generates valid Pareto frontier
     • Enforces hard parameter and memory constraints
  └─ Typed Evidence Verification:
     • Confirmatory results require PAIRED_EXPERIMENT or CONFIRMATORY_RESULT tier
     • Synthetic fixtures rejected at promotion gates
```

---

## 2. Quantitative Tolerances

| Metric | Permitted Regression | Action on Breach |
|:---|:---|:---|
| Unit Test Pass Rate | 0% (Must remain 100%) | Immediate block; revert offending commit. |
| Smoke Execution Time | $\le 4.0\text{s}$ on standard CPU | Profile slow operators; optimize tensor dispatch. |
| Intact Foraging Return | Mean Return $\ge 30.0$ | Check sensory projection and unicycle steering channels. |
| Lesion Recovery $\mathcal{R}$ | $\ge 90.0\%$ | Check causal attribution and isolated sandbox training steps. |
| Zero-Gain Tolerance | Absolute error $\le 10^{-7}$ | Verify $\alpha=0.0$ initialization and projection zeroing. |
| Backbone Freeze | Absolute gradient $\le 10^{-9}$ | Check `requires_grad=False` on baseline weights during training. |

---

## 3. Pre-Commit Checklist

Before merging `feature/sonapse-evo-2` back into `main`:
- [ ] All 5 audit documents updated and consistent.
- [ ] `pytest -q` passes 100%.
- [ ] New unit tests cover all 20 structural variation operators.
- [ ] New unit tests cover `DiscoveryOpportunityMonitor`, Pareto selection, and multi-fidelity evaluation.
- [ ] Legacy v1 commands execute identically under default or `legacy_v1` profile.
- [ ] No machine-specific absolute paths in any import or data loader.
- [ ] Clean git status with descriptive commits.
