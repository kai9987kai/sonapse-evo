# Research Limitations and Scientific Evidence Boundaries

1. **Synthetic Connectome Proxy vs Biological Brains**:
   The default 768-neuron substrate with 3 anatomical regions is a synthetic rate model. Observed recovery in continuous simulation does not constitute a biological validation of in vivo neurological repair.

2. **Causal Screening vs Optimization Proof**:
   Virtual shadow gradients $s_{ij}^{\mathrm{grad}}$ are heuristics for candidate proposal, not formal guarantees of optimal circuit topology. True causal verification relies on counterfactual outcome evaluation $\hat{\tau}_g$.

3. **Compute and Search Costs**:
   While sparse low-rank bridge projection requires only 512 operations per microstep, candidate discovery across 64 proposals incurs substantial diagnostic simulation. Candidate evaluation must strictly run outside the real-time action loop.

4. **Jacobian Contraction vs Memory Dynamics**:
   Strict contractive Jacobian conditions ($\|J_t\|_2 < 1$) guarantee long-term stability but can suppress long-horizon recurrent memory. The system monitors stability diagnostics without imposing over-damping that erases working memory.

5. **NeuroWeave Layer Abstractions**:
   Compartments, soma populations, neuromodulators, affect variables, astrocyte and myelin proxies in `sonapse_evo.neuroweave` are simplified update rules named after biological roles. They are not biophysical models and affect variables are control signals, not subjective states. Details and the full list of limits are in `NEUROWEAVE_EVOLUTION.md` (section 19).

6. **Evidence for the New Mechanisms**:
   No NeuroWeave mechanism is enabled or promoted by default. Experiment verdicts concern small synthetic tasks. The existing foraging task pays reward every step, so a slower agent collects more return; experiments on it use goal efficiency as the primary metric and report return separately (`NEUROWEAVE_RESULTS.md`).

7. **Execution Cost and Hardware**:
   The extended update path is far slower than the original scalar path on CPU, weave state is not GPU-resident, and the FPGA output is a mapping manifest only: no RTL, bitstream, verification or measured acceleration exists.
