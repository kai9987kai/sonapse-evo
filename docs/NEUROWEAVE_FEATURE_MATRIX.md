# NeuroWeave Evolution: audit, feature matrix and requirement coverage

## 1. Audit summary

**Target repository.** The working tree is SONAPSE-EVO 2.0 (Python, PyTorch; 59 modules, 91 classes,
267 methods, 8 CLI commands, a dashboard with 34 hooks, 60 tests). It describes itself as the successor
of Sonapse-LAB. The request was written for "Sonapse-LAB"; it was applied to this repository, and items
that only exist in Sonapse-LAB are listed in section 5.

**Source repository.** NEUROWEAVE-LAB 0.4.0 is a single-file browser workbench (`index.html`, about
177 kB of JavaScript) with 31 switchable mechanisms, seven topologies, a 1200 x 660 ecology, a
Micro-Collective, model export and an FPGA mapping manifest. `index.html`, the architecture, model-format
and research-map documents, the changelog and the smoke test were read. Its code was treated as a
specification: nothing was copied and nothing from it is executed.

**Baseline recorded before any change.** 60/60 tests passing; git tag `baseline/sonapse-evo-2.0.0`;
`tests/weave_baseline/compatibility_manifest.json` (the full API surface) and
`tests/weave_baseline/golden.json` (numeric fingerprints and CLI outputs).

## 2. Feature comparison matrix

Status before this work: **present** in SONAPSE-EVO, **partial**, or **missing**. "How" is what was
done; nothing present was replaced.

| NEUROWEAVE capability | SONAPSE-EVO before | How it was integrated | Mechanism / module |
|---|---|---|---|
| Recurrent signed connectome | present (768 neurons, 3 regions) | kept as is; subclassed | `WeaveSubstrate` |
| Multi-soma units (fast, slow, context, integrator, burst, gating) | missing | new, plus primary and secondary somas | `multi_soma` |
| Four dendritic compartments | missing | new, with apical gating and branch saturation | `dendritic_compartments` |
| Membrane, threshold, refractory, calcium | missing | new | `membrane_spiking` |
| Axon hillock, two branches, terminal release | missing | new; N branches, depleting resource, branch-specific synapses | `axonal_branches` |
| Homeostatic gain, astrocyte reservoir | partial (homeostatic term in the dense plasticity engine) | new per-neuron gain; existing term untouched | `homeostatic_governor`, `astrocyte_regulation` |
| Synaptic delays, myelin proxy | partial (metadata fields only) | new delay lines and adaptive conduction | `synaptic_delays`, `adaptive_conduction` |
| k-winner sparse code | missing | new (global or per region) | `sparse_distributed_code` |
| Structural fingerprints | missing | new | `structural_encoder` |
| Connectivity-context attention | missing | new | `connectivity_attention` |
| FlyHash projection | missing | new (sparse expansion + winner-take-all tag) | `flyhash_projection` |
| Neuromodulators (DA, 5-HT, OA, ACh, NE, eCB) | partial (4-field state in the dense engine) | new six-field state with receptors; bridged to the existing one | `neuromodulatory_field` |
| Affect proxies | missing | new, as operational controls | `affective_control` |
| Connectivity-only transmitter hypotheses | missing | new | `transmitter_inference` |
| Reward-gated plasticity | present (dense N x N engine) | existing engine kept; sparse O(E) rule added, including a sparse port of the existing rule | `reward_gated_plasticity` |
| Pair STDP | missing | new | `pair_stdp` |
| Structural plasticity (prune / regrow) | present at graph level (20 operators, RigL arm) | kept; online utility-guided turnover added with a bridge to `GraphTopology` | `structural_plasticity` |
| Neurogenesis | present at graph level (`AddNeuronOperator`) | kept; online repurposing and whole-model growth added | `neurogenesis`, `grow_substrate` |
| Sleep-like consolidation | missing | new | `sleep_consolidation` |
| Associative episodes, fast weights | missing | new | `associative_trace_memory`, `fast_weights` |
| Hierarchical temporal memory | missing | new | `hierarchical_temporal_memory` |
| Local predictive coding | missing | new | `local_predictive_coding` |
| Predictive ensemble, uncertainty gate | missing | new, action-conditioned, with counterfactuals | `predictive_world_model`, `uncertainty_gating` |
| Context-adaptive routing | missing | new, centred at unit gain | `context_adaptive_routing` |
| Efference self-model | missing | new | `efference_self_model` |
| Sixteen sensory channels, webcam, microphone | missing | new hub, fusion, extractors, browser capture | `multimodal_sensory`, `cross_modal_fusion` |
| Micro-Collective, expert routing, meta-learning, predictive replay | missing | new | `micro_collective`, `expert_routing`, `collective_meta_learning`, `predictive_replay` |
| Tiled execution, INT8 quantisation | missing | new | `tiled_execution`, `quantised_synapses` |
| Seven topologies, ten regions | partial (one generator, three regions) | thirteen families, layouts, metrics | `connectome/` |
| Topology assay, lesion assay | partial (five lesion types, repair assay) | kept; assays with matched controls added | `experiments/assays.py` |
| 1200 x 660 ecology | missing (two tasks) | new ecology with twelve presets; existing tasks kept in the same registry | `ecology/` |
| Model JSON, import validation, standalone runner | missing (in-memory snapshots only) | new package format, validation, import, runner | `portability/` |
| FPGA mapping manifest | missing | new | `hardware/fpga.py` |
| Session JSON / CSV export | partial (telemetry JSON) | experiment receipts in JSON and CSV | `experiments/framework.py` |
| Sparse sequence-variant search assay | missing | **not ported** (a protein-landscape toy unrelated to the connectome) | none |

Beyond NEUROWEAVE: NeuroWave-X2 packets, multi-region routing, the memory ecology with provenance,
multi-horizon prediction, meta-plasticity, the specialist graft, the experiment framework with verdict
rules, the promotion ledger and the compatibility tooling.

**Conflicts found and how they were resolved**

* Sign convention: the original graph draws a sign per synapse, NEUROWEAVE per neuron. Learning keeps the
  sign of each synapse's anchor weight; new generators use a per-neuron sign.
* Auxiliary-current cap: graft currents keep their existing clamp; weave currents have a separate cap.
* Dense plasticity engine: kept; the new sparse rule is separate and can reproduce its update.
* Hard-coded 768 neurons / motor index 512: `motor_offset` and region layouts carry the real values.
* Snapshots: the existing `SubstrateSnapshot` is unchanged; weave state rides in its `metadata` dict, so
  transaction rollbacks restore compartment state as well.

## 3. Requirement coverage

| Section of the request | Status | Where | Tests |
|---|---|---|---|
| 1 Non-negotiables (no removals, no changed defaults) | met | three additive edits; everything else is new files | `test_weave_regression.py` |
| 2 NEUROWEAVE integration | met, one item not ported (sequence-search assay) | section 2 above | registry test |
| 3 Multi-compartment, multi-soma neurons | met | `cells.py`, `substrate.py` | `test_weave_core.py` |
| 4 Connectome expansion and topology experiments | met | `connectome/`, `experiments/assays.py` | `test_weave_systems.py`, `test_weave_lab.py` |
| 5 Multi-representational communication, NeuroWave-X2 | met | `comm/neurowave.py` | `test_weave_mechanisms.py` |
| 6 Plasticity and structural evolution | met | `plasticity/`, `structure/` | `test_weave_mechanisms.py` |
| 7 Neuromodulation and affective control | met | `neuromod/` | `test_weave_mechanisms.py` |
| 8 Sensory system | met | `sensory/`, `weave.html` | `test_weave_systems.py` |
| 9 Micro-Collective and specialist | met | `collective/` | `test_weave_systems.py` |
| 10 Predictive processing and self-model | met | `predictive/` | `test_weave_mechanisms.py` |
| 11 Memory ecology and provenance | met | `memory/` | `test_weave_systems.py` |
| 12 Environments | met | `ecology/` | `test_weave_systems.py` |
| 13 Twenty-four experiments and methodology | met | `experiments/` | `test_weave_lab.py` |
| 14 Lesion and robustness testing | met | `connectome/lesions.py`, `experiments/assays.py` | `test_weave_lab.py` |
| 15 Hardware and efficiency | met as software experiments; no accelerator backend | `hardware/` | `test_weave_mechanisms.py` |
| 16 Export and portability | met | `portability/` | `test_weave_lab.py` |
| 17 Observability and interface | met | `observatory/weave_*.py`, `web/weave.html` | `test_weave_lab.py`, browser check |
| 18 Population scaling | met; cost grows and is reported | `connectome/`, `hardware/profiler.py` | `test_weave_core.py` |
| 19 Integration architecture | met | `contracts.py`, `events.py`, `registry.py` | `test_weave_core.py` |
| 20 Regression prevention | met | `compat.py`, `tests/weave_baseline/` | `test_weave_regression.py` |
| 21 Documentation | met | this folder | none |
| 24 Acceptance criteria | see section 4 | | |

### The 23 required regression tests

| # | Required test | Covered by |
|---|---|---|
| 1 | Original baseline functionality | the original 60 tests (unchanged) and `test_golden_numeric_baseline` |
| 2 | Existing mechanism compatibility | `test_existing_subsystems_accept_a_weave_substrate` |
| 3 | Existing topology compatibility | `test_golden_numeric_baseline` (topology digest), `test_every_topology_family...[legacy]` |
| 4 | Existing experiment compatibility | `test_original_cli_outputs` |
| 5 | Existing import / export compatibility | `test_legacy_packages_and_snapshots`, original `test_checkpoint_replay.py` |
| 6 | Existing UI controls | `test_original_api_surface_is_intact` (34 dashboard hooks, 12 JS functions), `test_http_endpoints...` |
| 7 | Existing model persistence | original `test_checkpoint_replay.py`, `test_atomic_rollback.py` |
| 8 | Existing learning behaviour | original `test_plasticity.py`, `test_graft_training.py`, `test_authentic_benchmarks.py` |
| 9 | Scalar pathway equivalence | `test_all_mechanisms_disabled_is_bit_identical`, `test_episode_runner_matches_original...` |
| 10 | Multi-soma integration | `test_primary_secondary_soma_relationship`, `test_compartments_participate...` |
| 11 | Multi-compartment invariants | `test_multi_compartment_state_invariants` |
| 12 | Connectome structural integrity | `test_every_topology_family_builds_a_valid_runnable_connectome`, `test_structural_turnover...` |
| 13 | Plasticity bounds | `test_weight_updates_are_bounded_and_sign_preserving`, `test_reward_gated_plasticity...` |
| 14 | Neuromodulator bounds | `test_modulator_bounds_and_legacy_bridge`, `test_max_profile_signals_stay_bounded` |
| 15 | Memory integrity | `test_provenance_rules_and_sparse_index`, `test_memory_mechanisms...` |
| 16 | Sensory integration | `test_sensory_hub_projects_channels...`, `test_cross_modal_fusion...` |
| 17 | Micro-Collective training | `test_micro_collective_learns_validates_rolls_back_and_expands` |
| 18 | Model export validation | `test_validation_rejects_tampered_malformed_and_oversized_packages` |
| 19 | Legacy model import | `test_legacy_packages_and_snapshots` |
| 20 | Standalone inference | `test_standalone_runner_matches_python_and_states_its_limits` |
| 21 | Lesion experiments | `test_topology_and_lesion_assays_run_with_matched_controls`, `test_region_routing_and_lesion_library` |
| 22 | Reproducibility | `test_experiments_are_reproducible...`, determinism checks in every subsystem test |
| 23 | Performance benchmarks | `test_fpga_manifest_profiler_and_backends`; measured table in the results document |

## 4. Acceptance criteria

| Criterion | Result |
|---|---|
| All original features, experiments and workflows remain | yes: 60 original tests pass; API surface and golden values unchanged |
| Existing exports remain compatible | yes: snapshots unchanged; format-2 packages import |
| NEUROWEAVE mechanisms incorporated or accounted for | 31 of 31 switches mapped; one assay explicitly not ported |
| No unnecessary duplication | existing engines kept and bridged (plasticity, modulators, operators, statistics, evidence tiers, failure archive) |
| New neural systems functionally integrated | yes: each changes the emitted rates (tested) |
| Sensory systems operate through the connectome | yes |
| Multi-soma systems participate in computation | yes |
| Neuromodulators influence learning and behaviour | yes (plasticity gain, gating, exploration, persistence, thresholds; tested) |
| Structural plasticity operational | yes |
| Predictive processing operational | yes |
| Memory works with the expanded architecture | yes |
| Micro-Collective can be trained and evaluated | yes |
| Environments provide real feedback | yes |
| New experiments reproducible | yes (result digests) |
| Export and import validated | yes (exact continuation) |
| Original baseline recoverable | yes (git tag) |
| Regression tests pass | yes (233 tests: the original 60 plus 173 new) |
| Performance and limitations documented | yes |

### Independent review

Four reviewers, each limited to one area and required to reproduce every finding by running code before
reporting it, examined the new layer after it was complete. They reported 32 defects (11 high, 17 medium,
4 low). All 32 were fixed and each has a regression test in `tests/test_weave_review_fixes.py`. The most
important ones:

* package validation accepted packages that then crashed import or produced NaN (missing keys, wrong
  tensor shapes, NaN in the lesion mask or sensory projection); validation now checks every field import
  reads, and import raises only `ValueError` after a trial step;
* restoring a snapshot or cloning after structural growth left per-edge caches stale, so continuation was
  not exact; caches are now rebuilt on every load and saved where they depend on history;
* the substrate shared memory with the `TopologyState`, so neurogenesis relabelling changed the topology
  digest and the lesion masks of other substrates built on it;
* in-place edits of `base_w` by existing code (RigL) left stale synapse signs, so learning could flip them;
* one NaN reward or outcome permanently corrupted all learned weights;
* frozen evaluation still adapted slow cell state (homeostasis, astrocyte reservoir, myelin, traces);
* a failing model during a live run could kill the original dashboard's telemetry thread;
* the standalone runner applied the lesion mask before the learned delta, reviving lesioned synapses;
* quick (smoke-sized) experiment runs could be labelled as paired experiments and promoted;
* the memory ecology stored simulated input as verified observations.

### Re-running the original commands

Every original CLI command was re-run after the change and compared with the outputs recorded before it.
`lesion-repair` and `curriculum` are identical. `smoke`, `benchmark`, `discover` and `criticality` differ only
in lines that also differ between two runs of the **untouched** baseline code (checked in a worktree of the
baseline tag): the smoke sandbox-training loss, benchmark arms A2, A3, A5 and A7, the discovery utility (it
includes measured latency) and the criticality Lyapunov estimate all use unseeded randomness or wall-clock time
in the original implementation. Deterministic outputs (smoke return 21.36, arm A4, lesion-repair recovery
107.4%, curriculum scores) match exactly. This pre-existing nondeterminism was left as is, because fixing it
would change original behaviour; the golden-baseline test compares only the deterministic values.

## 5. Not done, partial, or different from the request

* **Sonapse-LAB-only items** (DSAG, its webcam / microphone pipeline, its sleep engine, its
  checkpoints): not present in this repository, so nothing to preserve; functional equivalents were built.
* **Sparse sequence-variant search assay** of NEUROWEAVE: not ported.
* **Accelerator backends**: only the CPU path runs; the interface reports this honestly.
* **Standalone runner**: frozen scalar pathway only.
* **Research sweep**: the literature review was limited to targeted look-ups of four recent papers plus
  well-known earlier work; it was not a systematic survey.
* **No mechanism is promoted.** Evidence so far is in the results document.
* **Security scan**: the configured HawkScan DAST scan could not run (no scanner runtime or API key in
  this environment). The new HTTP surface was hardened by hand (body limits, JSON validation,
  cross-origin POST refusal, no raw media) and those behaviours are tested.
* **Package version** is still 2.0.0; the layer carries its own version (3.0.0).
