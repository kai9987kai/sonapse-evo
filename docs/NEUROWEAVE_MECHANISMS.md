# NeuroWeave mechanism catalogue

Generated from `sonapse_evo.neuroweave.registry`. All 42 mechanisms are disabled by default; `legacy` is the original behaviour.
Kinds: **core** (inside the cell / transmission code), **plugin** (a `Mechanism` subclass), **modifier** (a flag read by another mechanism).


## cell

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `multi_soma` | core | INTEGRATE | Primary, secondary, fast, slow, context, integrator, burst and gating soma populations re-mix the synaptic drive across timescales. | multiSoma | steady inputs: every temporal-contrast term vanishes |
| `dendritic_compartments` | core | INTEGRATE | Proximal (feed-forward), distal (recurrent), apical (context) and basal branches with local time constants, saturation and apical gating. | - | disabled: currents sum at the soma exactly as in the legacy drive |
| `membrane_spiking` | core | INTEGRATE | Membrane proxy, activity-dependent threshold, refractory period, spike events and an intracellular calcium proxy. | - | - |
| `axonal_branches` | core | INTEGRATE | Hillock initiation, two or more axonal branches, terminal release with a depleting resource, and branch-specific synaptic transmission. | axonBranches | - |
| `homeostatic_governor` | core | INTEGRATE | Per-neuron gain drifts to hold a target rate, bounded to [0.58, 1.12]. | glial | gain starts at exactly 1 |
| `astrocyte_regulation` | core | INTEGRATE | Slow local reservoir tracking neighbourhood activity scales the output gain and incoming synaptic efficacy. | glial | only on its first update: the gain starts at exactly 1 and then adapts to activity |

## representation

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `sparse_distributed_code` | core | INTEGRATE | Global or per-region k-winners: non-winning units are attenuated to 8% of their activation. | sparse | - |

## memory

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `hierarchical_temporal_memory` | core | INTEGRATE | Fast, medium and slow per-neuron activity traces inject a bounded temporal contrast into fusion. | hierarchicalMemory | traces start at zero |

## prediction

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `local_predictive_coding` | core | INTEGRATE | Each unit predicts its fused drive, tracks bounded error and surprise, and injects a surprise-normalised residual. | predictiveCoding | - |

## transmission

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `synaptic_delays` | core | TRANSMIT | Per-edge extra conduction delay of 0-3 steps read from a rate history buffer; long-range edges are slower. | delays | - |
| `adaptive_conduction` | core | TRANSMIT | Activity-dependent myelin proxy shortens the delay of well-used axons. | myelination | - |
| `context_adaptive_routing` | core | TRANSMIT | Destination gating-soma and apical state scale each edge within [0.62, 1.14]; centred so the resting gain is exactly 1. | adaptiveRouting | gating soma at 1 and silent apical dendrite |
| `uncertainty_gating` | core | TRANSMIT | Ensemble disagreement attenuates recurrent transmission within [0.58, 1]. | uncertaintyGate | uncertainty signal equal to 0 |

## representation

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `structural_encoder` | plugin | CONTEXT | Sixteen-feature graph fingerprints per neuron weight the context message by source degree and reciprocity. | structural | - |
| `connectivity_attention` | plugin | CONTEXT | Soft attention over the eight strongest active incoming neighbours feeds the apical dendrite. | wiretype | - |
| `flyhash_projection` | plugin | CONTEXT | Fixed sparse random expansion of the observation with winner-take-all tagging feeds the apical dendrite. | flyHash | - |
| `neurowave_packets` | plugin | CONTEXT | Typed, versioned scalar/vector/spectral/phase/sparse/VSA/event/population packets fused into neural currents through a zero-initialised fusion layer. | - | fusion read-out weights start at zero |

## connectome

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `multi_region_routing` | plugin | CONTEXT | Region-pair routing gains over sensory, association, memory and motor pathways with feedback and predictive feedback routes. | - | all region-pair gains equal to 1 |

## learning

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `reward_gated_plasticity` | plugin | LEARN | Sparse per-edge eligibility traces gated by lagged reward and neuromodulators; sign and magnitude bounded. | plasticity | zero reward |
| `pair_stdp` | plugin | LEARN | Pre/post spike traces produce bounded potentiation and depression; source sign is fixed. | stdp | - |
| `meta_plasticity` | plugin | LEARN | Sliding BCM-like thresholds and per-synapse adaptive learning-rate scales with consolidation-dependent protection. | - | per-synapse rate scale starts at 1 |
| `sleep_consolidation` | plugin | STRUCTURE | Periodic offline phases replay stored activity, consolidate useful synapses and weakly downscale the rest. | sleep | - |

## execution

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `quantised_synapses` | plugin | TRANSMIT | Signed 127-level magnitude quantisation applied to effective weights at read time. | quantized | - |
| `tiled_execution` | plugin | TRANSMIT | Tile-grouped edge traversal with synchronous previous/next buffers and operation accounting. | tiled | - |

## structure

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `structural_plasticity` | plugin | STRUCTURE | Bounded pruning of low-utility synapses and co-activity-guided regrowth with an audit ledger. | neurogenesis | - |
| `neurogenesis` | plugin | STRUCTURE | Recruits reserve neurons, repurposes under-used ones, assigns them to regions and wires them in under growth budgets. | - | - |

## modulation

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `neuromodulatory_field` | plugin | MODULATE | Dopamine, serotonin, acetylcholine, noradrenaline, octopamine and endocannabinoid-like fields with heterogeneous receptor profiles. | neuromod | - |
| `affective_control` | plugin | MODULATE | Valence, arousal, curiosity, stress and motivation as operational control variables driving exploration, thresholds, persistence and learning. | affectiveSignals | - |
| `transmitter_inference` | plugin | MODULATE | Six-class transmitter hypothesis with confidence from graph features; never relabels the source sign. | wireTypeInference | - |

## memory

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `associative_trace_memory` | plugin | CONTEXT | Bounded similarity-retrieved episodes re-inject a fraction of the stored state. | memory | empty store |
| `fast_weights` | plugin | CONTEXT | Reward-updated key/value association matrix read into the drive with a bounded coefficient. | fastWeights | matrix starts at zero |
| `memory_ecology` | plugin | CONTEXT | Working, episodic, semantic and predictive stores with sparse indexing, confidence, interference tracking and provenance. | - | empty stores |
| `predictive_replay` | plugin | STRUCTURE | Novelty- and error-prioritised replay scheduling for consolidation and for the Micro-Collective. | predictiveReplay | - |

## prediction

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `predictive_world_model` | plugin | LEARN | Small ensemble predicts pooled state; member disagreement is the uncertainty signal. | worldModel | - |
| `hierarchical_prediction` | plugin | LEARN | Immediate, short, medium and long horizon predictors with an uncertainty-adaptive active horizon. | - | - |
| `efference_self_model` | plugin | LEARN | Compares intended, issued, predicted and observed action outcomes to estimate authorship, disturbance, slip and control reliability. | selfModel | - |

## sensory

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `multimodal_sensory` | plugin | SENSE | Sixteen sensory channels (visual, auditory, olfactory, gustatory, thermal, mechano, proprio, interoceptive) projected into the connectome. | multimodalSensory | all channels at zero |
| `cross_modal_fusion` | plugin | SENSE | Reliability-weighted fusion, sensory prediction error, missing-sensor handling and cross-modal association. | - | - |

## collective

| id | kind | stage | what it does | NEUROWEAVE switch | exact no-op when |
|---|---|---|---|---|---|
| `micro_collective` | plugin | CONTEXT | Local mixture of small experts predicting the next sensory state with replay, held-out validation and rollback. | microCollective | untrained: predictive context gain is zero |
| `expert_routing` | modifier | CONTEXT | Top-k routing with load balancing (modifier of the Micro-Collective). | expertRouting | - |
| `collective_meta_learning` | modifier | CONTEXT | Held-out validation, best-state snapshots, rollback and learning-rate decay (modifier of the Micro-Collective). | metaLearning | - |
| `specialist_graft` | plugin | CONTEXT | Independently trained specialist model exchanging predictions with the connectome through an explicit, removable interface. | - | graft gain starts at zero |

## Profiles

| profile | mechanisms |
|---|---|
| `legacy` | 0 (the original model) |
| `cells` | 6 |
| `transmission` | 4 |
| `learning` | 4 |
| `modulation` | 3 |
| `memory` | 5 |
| `prediction` | 4 |
| `sensory` | 2 |
| `collective` | 4 |
| `neuroweave` | 32 |
| `max` | 42 |

## NEUROWEAVE-LAB 0.4.0 switch map

| NEUROWEAVE switch | SONAPSE-EVO mechanism(s) |
|---|---|
| multiSoma | `multi_soma` |
| structural | `structural_encoder` |
| sparse | `sparse_distributed_code` |
| memory | `associative_trace_memory` |
| plasticity | `reward_gated_plasticity` |
| delays | `synaptic_delays` |
| glial | `homeostatic_governor`, `astrocyte_regulation` |
| tiled | `tiled_execution` |
| wiretype | `connectivity_attention` |
| flyHash | `flyhash_projection` |
| neuromod | `neuromodulatory_field` |
| neurogenesis | `structural_plasticity` |
| sleep | `sleep_consolidation` |
| worldModel | `predictive_world_model` |
| myelination | `adaptive_conduction` |
| quantized | `quantised_synapses` |
| axonBranches | `axonal_branches` |
| fastWeights | `fast_weights` |
| selfModel | `efference_self_model` |
| predictiveCoding | `local_predictive_coding` |
| stdp | `pair_stdp` |
| adaptiveRouting | `context_adaptive_routing` |
| hierarchicalMemory | `hierarchical_temporal_memory` |
| uncertaintyGate | `uncertainty_gating` |
| multimodalSensory | `multimodal_sensory` |
| affectiveSignals | `affective_control` |
| microCollective | `micro_collective` |
| expertRouting | `expert_routing` |
| metaLearning | `collective_meta_learning` |
| predictiveReplay | `predictive_replay` |
| wireTypeInference | `transmitter_inference` |

## Experiment catalogue

| id | kind | environment | candidate | baseline | ablations |
|---|---|---|---|---|---|
| `multimodal_sensory_integration` | paired | neuroweave_1200x660 | multimodal_sensory, cross_modal_fusion | legacy | cross_modal_fusion |
| `affective_neuromodulation` | paired | forage | reward_gated_plasticity, neuromodulatory_field, affective_control | reward_gated_plasticity | affective_control |
| `micro_collective_learning` | paired | neuroweave_1200x660 | multimodal_sensory, micro_collective, collective_meta_learning | multimodal_sensory | - |
| `expert_routing` | paired | neuroweave_1200x660 | multimodal_sensory, micro_collective, collective_meta_learning, expert_routing | multimodal_sensory, micro_collective, collective_meta_learning | - |
| `meta_plasticity` | paired | forage | reward_gated_plasticity, meta_plasticity | reward_gated_plasticity | - |
| `predictive_replay` | paired | forage | reward_gated_plasticity, sleep_consolidation, predictive_replay | reward_gated_plasticity, sleep_consolidation | - |
| `connectivity_context_attention` | paired | forage | structural_encoder, connectivity_attention, dendritic_compartments | dendritic_compartments | structural_encoder |
| `flyhash_sparse_projections` | paired | forage | flyhash_projection, sparse_distributed_code | legacy | flyhash_projection |
| `stdp` | paired | forage | membrane_spiking, pair_stdp | membrane_spiking | - |
| `adaptive_conduction` | paired | forage | synaptic_delays, adaptive_conduction | synaptic_delays | - |
| `quantised_synapses` | paired | forage | quantised_synapses | legacy | - |
| `hierarchical_temporal_memory` | paired | memory_navigation | multi_soma, hierarchical_temporal_memory | multi_soma | - |
| `uncertainty_gated_propagation` | paired | forage | predictive_world_model, uncertainty_gating | predictive_world_model | - |
| `predictive_coding` | paired | forage | multi_soma, local_predictive_coding | multi_soma | - |
| `structural_plasticity` | paired | forage | reward_gated_plasticity, structural_plasticity | reward_gated_plasticity | - |
| `associative_memory` | paired | forage | associative_trace_memory, fast_weights, memory_ecology | legacy | fast_weights, memory_ecology |
| `action_grounded_self_modelling` | paired | disturbed | reward_gated_plasticity, efference_self_model | reward_gated_plasticity | - |
| `multi_soma_integration` | paired | forage | multi_soma, dendritic_compartments, membrane_spiking, axonal_branches | legacy | dendritic_compartments, axonal_branches |
| `multi_region_routing` | paired | forage | multi_region_routing | legacy | - |
| `lesion_robustness` | lesion | forage | - | legacy | - |
| `topology_optimisation` | topology | forage | - | legacy | - |
| `neural_grafting` | paired | neuroweave_1200x660 | multimodal_sensory, specialist_graft | multimodal_sensory | - |
| `cross_modal_learning` | paired | neuroweave_1200x660 | multimodal_sensory, cross_modal_fusion | multimodal_sensory | - |
| `specialist_expert_evolution` | paired | neuroweave_1200x660 | multimodal_sensory, micro_collective, collective_meta_learning, expert_routing | multimodal_sensory, micro_collective, collective_meta_learning | - |
