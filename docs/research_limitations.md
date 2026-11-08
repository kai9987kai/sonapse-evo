# Research Limitations and Scientific Evidence Boundaries

1. **Synthetic Connectome Proxy vs Biological Brains**:
   The default 768-neuron substrate with 3 anatomical regions is a synthetic rate model. Observed recovery in continuous simulation does not constitute a biological validation of in vivo neurological repair.

2. **Causal Screening vs Optimization Proof**:
   Virtual shadow gradients $s_{ij}^{\mathrm{grad}}$ are heuristics for candidate proposal, not formal guarantees of optimal circuit topology. True causal verification relies on counterfactual outcome evaluation $\hat{\tau}_g$.

3. **Compute and Search Costs**:
   While sparse low-rank bridge projection requires only 512 operations per microstep, candidate discovery across 64 proposals incurs substantial diagnostic simulation. Candidate evaluation must strictly run outside the real-time action loop.

4. **Jacobian Contraction vs Memory Dynamics**:
   Strict contractive Jacobian conditions ($\|J_t\|_2 < 1$) guarantee long-term stability but can suppress long-horizon recurrent memory. The system monitors stability diagnostics without imposing over-damping that erases working memory.
