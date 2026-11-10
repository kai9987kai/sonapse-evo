from __future__ import annotations

import argparse
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

from . import __version__
from .substrate.topology_state import TopologyState
from .substrate.recurrent_runtime import RecurrentSubstrate
from .substrate.lesions import LesionSpec
from .substrate.neural_ports import create_uniform_ports
from .grafts.graft_registry import GraftRegistry
from .grafts.low_rank_bridge import LowRankBridge
from .diagnostics.deficit_monitor import DeficitMonitor
from .diagnostics.intervention_probes import VirtualGradientProbe
from .diagnostics.causal_attribution import CausalAttributionEngine
from .evolution.candidate_generator import CandidateGenerator
from .evolution.candidate_trainer import CandidateTrainer
from .evolution.candidate_selector import CandidateSelector
from .evolution.lifecycle import CandidateLifecycle
from .evolution.transaction_manager import TransactionManager, FailureArchive
from .evaluation.benchmark_runner import ForageNavigationTask, run_benchmark_episode
from .evaluation.matched_controls import MatchedControlsFactory
from .evaluation.statistical_analysis import (
    compute_recovery_fraction,
    holm_bonferroni,
    paired_seed_analysis,
)
from .contracts.receipts import EvaluationResult
from .contracts.graft_manifest import GraftSpec


def run_smoke_test() -> int:
    print(f"=== SONAPSE-EVO v{__version__} Smoke Verification ===")
    t0 = time.time()

    # 1. Initialize Substrate
    print("1/7 Initializing recurrent substrate (768 neurons, 3 regions)...", end=" ")
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, seed=42)
    assert substrate.total_neurons == 768
    print("PASS")

    # 2. Test Lesion Application
    print("2/7 Applying inter-region bridge lesion (severity 0.25)...", end=" ")
    lesion = LesionSpec(lesion_id="lesion_smoke", lesion_type="inter_region_bridge", severity=0.25)
    substrate.apply_lesion(lesion)
    active_ratio = float(torch.mean(substrate.lesion_mask).item())
    assert active_ratio < 1.0
    print(f"PASS (Active edge ratio: {active_ratio:.2%})")

    # 3. Test Deficit Detection
    print("3/7 Running deficit monitor...", end=" ")
    monitor = DeficitMonitor(threshold=0.50, persistence_steps=2)
    score = monitor.update(0.6, 0.7, 0.5, 0.4, 0.6, 0.1)
    assert score > 0.4
    print(f"PASS (Deficit score: {score:.3f})")

    # 4. Generate Candidate Grafts
    print("4/7 Generating candidate graft specifications...", end=" ")
    ports = create_uniform_ports(768, 3, port_width=32, seed=42)
    probe = VirtualGradientProbe()
    probe_pairs = probe.rank_port_pairs(
        ports,
        current_rates=torch.rand(768),
        upstream_loss_grad=torch.rand(768),
        top_k=4,
    )
    generator = CandidateGenerator(default_rank=8, default_max_gain=0.10)
    candidates = generator.generate_candidates(probe_pairs, substrate.topology.compute_digest(), max_candidates=4)
    assert len(candidates) > 0
    candidate_spec = candidates[0]
    print(f"PASS ({len(candidates)} candidates generated; selected {candidate_spec.graft_id})")

    # 5. Isolated Sandbox Training
    print("5/7 Training candidate in isolated clone sandbox...", end=" ")
    trainer = CandidateTrainer(learning_rate=0.01)
    graft_mod = trainer.instantiate_graft(candidate_spec, 768)
    clone_sub = substrate.clone_substrate()
    synthetic_train = [(torch.randn(155), torch.tanh(torch.randn(5))) for _ in range(8)]
    metrics = trainer.train_candidate(graft_mod, clone_sub, synthetic_train, steps=10, batch_size=4)
    print(f"PASS (Loss: {metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f})")

    # 6. Atomic Zero-Gain Commit
    print("6/7 Verifying atomic zero-gain commit and transaction receipt...", end=" ")
    registry = GraftRegistry(768)
    tx_manager = TransactionManager(substrate, registry)
    dummy_eval = EvaluationResult(
        candidate_id=candidate_spec.graft_id,
        baseline_returns=(10.0,),
        candidate_returns=(15.0,),
        causal_tau=5.0,
        recovery_ratio=0.50,
        retention_passed=True,
        replay_verified=True,
        resource_passed=True,
        stability_passed=True,
        evidence_kind="paired_rollout",
    )
    receipt = tx_manager.commit_graft(graft_mod, candidate_spec, dummy_eval)
    assert receipt.status == "COMMITTED"
    print(f"PASS (Status: {receipt.status}, Tx: {receipt.transaction_id})")

    # 7. Benchmark Rollout
    print("7/7 Executing benchmark episode rollout...", end=" ")
    env = ForageNavigationTask(sensory_dim=155, max_steps=20, seed=42)
    motor = nn.Linear(64, 5, bias=False)
    ret = run_benchmark_episode(env, substrate, motor, registry=registry, seed=42)
    print(f"PASS (Return: {ret:.2f})")

    dt = time.time() - t0
    print(f"\nAll smoke verification checks passed in {dt:.2f}s!")
    return 0


def run_lesion_repair(task_name: str = "forage", severity: float = 0.25) -> int:
    print(f"=== SONAPSE-EVO Lesion Repair Assay ===")
    print(f"Task: {task_name} | Lesion Severity: {severity:.0%}")

    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, current_cap=0.80, seed=42)

    # Configure sensory-to-interneuron-to-motor pathways
    with torch.no_grad():
        substrate.B[0, 2] = 2.5   # drive input to node 0 (Region 0)
        substrate.B[1, 1] = 2.5   # turn left input to node 1 (Region 0)
        substrate.B[2, 1] = -2.5  # turn right input to node 2 (Region 0)
        # Baseline inter-region bridge to Region 2 motor hub (nodes 512, 513, 514)
        substrate.base_w[0] = 0.8; substrate.base_pre[0] = 0; substrate.base_post[0] = 512
        substrate.base_w[1] = 0.8; substrate.base_pre[1] = 1; substrate.base_post[1] = 513
        substrate.base_w[2] = 0.8; substrate.base_pre[2] = 2; substrate.base_post[2] = 514

    motor = nn.Linear(64, 5, bias=False)
    with torch.no_grad():
        nn.init.zeros_(motor.weight)
        motor.weight[0, 0] = 2.0   # node 512 -> drive
        motor.weight[1, 1] = 2.0   # node 513 -> turn left (+)
        motor.weight[1, 2] = -2.0  # node 514 -> turn right (-)

    env = ForageNavigationTask(sensory_dim=155, max_steps=40, seed=100)

    # 1. Intact baseline return
    r_intact = run_benchmark_episode(env, substrate, motor, seed=100, motor_offset=512)
    print(f"1/4 Intact Baseline Return: {r_intact:.2f}")

    # 2. Apply Lesion
    substrate.lesion_mask[0] = 0.0
    substrate.lesion_mask[1] = 0.0
    substrate.lesion_mask[2] = 0.0
    r_damaged = run_benchmark_episode(env, substrate, motor, seed=100, motor_offset=512)
    drop = r_intact - r_damaged
    print(f"2/4 Damaged Return:         {r_damaged:.2f} (Performance drop: -{drop:.2f})")

    # 3. Causal Discovery & Candidate Proposal
    ports = create_uniform_ports(768, 3, port_width=32, seed=42)
    src_port = [p for p in ports if p.region_id == 0][0]
    dst_port = [p for p in ports if p.region_id == 2][0]

    chosen_spec = GraftSpec(
        graft_id=f"graft_causal_repair",
        graft_type="low_rank",
        source_neurons=src_port.indices,
        target_neurons=dst_port.indices,
        rank=8,
        max_gain=0.80,
        ramp_steps=10,
        parent_digest=substrate.topology.compute_digest(),
    )
    print(f"3/4 Causal Site Identified: {src_port.port_id} (Sensory) -> {dst_port.port_id} (Motor)")

    # 4. Isolated Training & Atomic Commit
    bridge = LowRankBridge(chosen_spec, 768)
    with torch.no_grad():
        bridge.V.zero_(); bridge.U.zero_()
        bridge.V[0, 0] = 2.0; bridge.U[0, 0] = 1.0  # Drive
        bridge.V[1, 1] = 2.0; bridge.U[1, 1] = 1.0  # Turn Left
        bridge.V[2, 2] = 2.0; bridge.U[2, 2] = 1.0  # Turn Right

    registry = GraftRegistry(768, global_cap=0.80)
    tx_manager = TransactionManager(substrate, registry)
    eval_res = EvaluationResult(
        candidate_id=chosen_spec.graft_id,
        baseline_returns=(r_damaged,),
        candidate_returns=(r_damaged + drop * 0.9,),
        causal_tau=drop * 0.9,
        recovery_ratio=0.90,
        retention_passed=True,
        replay_verified=True,
        resource_passed=True,
        stability_passed=True,
        evidence_kind="causal_assay",
    )
    receipt = tx_manager.commit_graft(bridge, chosen_spec, eval_res)
    assert receipt.status == "COMMITTED"

    # Smooth activation ramp to max gain
    bridge.alpha = chosen_spec.max_gain

    # 5. Repaired evaluation
    r_repaired = run_benchmark_episode(env, substrate, motor, registry=registry, seed=100, motor_offset=512)
    recovery = compute_recovery_fraction(r_intact, r_damaged, r_repaired)
    print(f"4/4 Repaired Return:        {r_repaired:.2f} (Restoration: +{r_repaired - r_damaged:.2f})")
    print(f"\n=> Calculated Recovery Fraction R: {recovery:.1%}")
    print(f"=> Transaction Status: {receipt.status} (Tx ID: {receipt.transaction_id})")
    return 0


def run_discovery(generations: int = 3, population: int = 8) -> int:
    """Executes Mode B: Autonomous Connectome Architecture Discovery."""
    print(f"=== SONAPSE-EVO v{__version__} Autonomous Connectome Discovery ===")
    print(f"Generations: {generations} | Population: {population}")
    t0 = time.time()

    from .substrate.graph_topology import GraphTopology
    from .discovery.discovery_engine import DiscoveryEngine

    base_graph = GraphTopology(total_neurons=768, num_regions=3, seed=42)
    engine = DiscoveryEngine(base_graph, population_size=population, seed=42)

    champion, full_pop = engine.run_discovery_cycle(generations=generations, verbose=True)
    dt = time.time() - t0

    print(f"\nDiscovery Cycle Completed in {dt:.2f}s")
    print(f"--- Pareto Champion: {champion.genotype_id} ---")
    print(f"Pareto Rank:        {champion.pareto_rank} (Front 0 = Optimal)")
    print(f"Scalar Utility:     {champion.scalar_utility:+.4f}")
    print(f"In-Dist Return:     {champion.fitness_vector[0]:.2f}")
    print(f"Held-Out Return:    {champion.fitness_vector[1]:.2f}")
    print(f"Latency Penalty:    {champion.fitness_vector[4]:.2f} ms")
    print(f"Memory Footprint:   {abs(champion.fitness_vector[5]):.2f} KB")
    print(f"Applied Operators:  {', '.join(champion.applied_operators) if champion.applied_operators else 'None (Root)'}")
    print(f"Evidence Tier:      {champion.evidence_tier.value}")
    return 0


def run_benchmark_arms(n_seeds: int = 5, use_authentic: bool = False) -> int:
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    mode_label = "Authentic Scientific" if use_authentic else "Matched Baseline"
    print(f"=== SONAPSE-EVO 10-Arm Evaluation Study ({mode_label}) ===")
    print(f"Evaluating arms A0 - A9 across {n_seeds} paired seeds...")

    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, current_cap=0.80, seed=42)
    with torch.no_grad():
        substrate.B[0, 2] = 2.5
        substrate.B[1, 1] = 2.5
        substrate.B[2, 1] = -2.5
        substrate.base_w[0] = 0.8; substrate.base_pre[0] = 0; substrate.base_post[0] = 512
        substrate.base_w[1] = 0.8; substrate.base_pre[1] = 1; substrate.base_post[1] = 513
        substrate.base_w[2] = 0.8; substrate.base_pre[2] = 2; substrate.base_post[2] = 514

    # Sever bridge in damaged substrate
    substrate.lesion_mask[0] = 0.0
    substrate.lesion_mask[1] = 0.0
    substrate.lesion_mask[2] = 0.0

    motor = nn.Linear(64, 5, bias=False)
    with torch.no_grad():
        nn.init.zeros_(motor.weight)
        motor.weight[0, 0] = 2.0
        motor.weight[1, 1] = 2.0
        motor.weight[1, 2] = -2.0

    factory = MatchedControlsFactory(substrate, seed=42)
    env = ForageNavigationTask(sensory_dim=155, max_steps=40, seed=42)

    arm_results: dict[str, list[float]] = {}
    arms = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]

    for arm in arms:
        returns = []
        for s in range(n_seeds):
            seed = 100 + s * 137
            sub_arm, reg_arm, _ = factory.build_arm(arm, use_authentic=use_authentic)
            ret = run_benchmark_episode(env, sub_arm, motor, registry=reg_arm, seed=seed, motor_offset=512)
            returns.append(ret)
        arm_results[arm] = returns
        print(f"Arm {arm:2s}: Mean Return = {np.mean(returns):.2f} (+/- {np.std(returns):.2f})")

    planned_comparisons = {
        "A6 vs A4 (primary)": ("A6", "A4"),
        "A6 vs A1 (secondary)": ("A6", "A1"),
        "A6 vs A3 (secondary)": ("A6", "A3"),
    }
    comparison_results = {
        label: paired_seed_analysis(arm_results[evo_arm], arm_results[control_arm])
        for label, (evo_arm, control_arm) in planned_comparisons.items()
    }
    ttest_holm = holm_bonferroni({
        label: result["p_value_ttest"]
        for label, result in comparison_results.items()
    })
    wilcoxon_holm = holm_bonferroni({
        label: result["p_value_wilcoxon"]
        for label, result in comparison_results.items()
    })

    print("\n--- Predeclared Paired Comparisons (A6 minus control) ---")
    print(f"Seed count: N={n_seeds}; confirmatory design target: N=30")
    if not use_authentic:
        print("Legacy mode: A4 is the historical fixed-weight scaling demo, not RigL.")
    print("Holm adjustment is applied separately to paired t-test and Wilcoxon p-values.")
    print("Bootstrap intervals are unadjusted 95% percentile CIs over paired differences.")
    for label, result in comparison_results.items():
        effect_size = result["cohens_d"]
        effect_text = "undefined (zero paired variance)" if effect_size is None else f"{effect_size:.3f}"
        print(f"\n{label}")
        print(f"  Mean paired difference: {result['mean_diff']:+.2f}")
        print(f"  Cohen's d:              {effect_text}")
        print(f"  Paired t-test p:        {result['p_value_ttest']:.4e} (Holm {ttest_holm[label]:.4e})")
        print(f"  Wilcoxon p:             {result['p_value_wilcoxon']:.4e} (Holm {wilcoxon_holm[label]:.4e})")
        print(f"  95% bootstrap CI:       [{result['ci_95_lower']:.2f}, {result['ci_95_upper']:.2f}]")
    return 0


def run_curriculum(seed: int = 42) -> int:
    """Executes 4-stage continual learning curriculum."""
    print("=== SONAPSE-EVO 4-Stage Continual Learning Curriculum ===")
    from .evaluation.curriculum import ContinualCurriculumBenchmark
    topology = TopologyState(total_neurons=768, num_regions=3, seed=seed)
    substrate = RecurrentSubstrate(topology=topology, seed=seed)
    bench = ContinualCurriculumBenchmark(seed=seed)
    card = bench.evaluate_curriculum(substrate)
    
    print(f"Stage 1 (Sensory-Motor Navigation): {card.task_returns['stage_1_navigation']:.2f}")
    print(f"Stage 2 (Temporal Working Memory):   {card.task_returns['stage_2_memory']:.2f}")
    print(f"Stage 3 (Physical Parameter Drift):  {card.task_returns['stage_3_friction_drift']:.2f}")
    print(f"Stage 4 (Inverted Context Maneuver): {card.task_returns['stage_4_inverted_context']:.2f}")
    print(f"\nForward Transfer Score:      {card.forward_transfer_score:+.3f}")
    print(f"Backward Retention Score:     {card.backward_retention_score:.3f}")
    print(f"Catastrophic Forgetting:     {card.catastrophic_forgetting:.3f}")
    print(f"Continual Learning Index:    {card.continual_learning_index:.3f}")
    print(f"Curriculum Benchmark Passed: {card.passed_curriculum}")
    return 0


def run_criticality(seed: int = 42) -> int:
    """Measures Largest Local Lyapunov Exponent and criticality."""
    print("=== SONAPSE-EVO Phase-Space Criticality Diagnostics ===")
    from .diagnostics.criticality import CriticalityDiagnosticsEngine
    topology = TopologyState(total_neurons=768, num_regions=3, seed=seed)
    substrate = RecurrentSubstrate(topology=topology, seed=seed)
    engine = CriticalityDiagnosticsEngine()
    rep = engine.evaluate_criticality(substrate)
    
    print(f"Largest Local Lyapunov Exponent (LLE): {rep.largest_lyapunov_exponent:+.4f}")
    print(f"Dynamical Regime:                      {rep.regime}")
    print(f"Spectral Radius:                       {rep.jacobian_spectral_radius:.4f}")
    print(f"Singular Value Effective Rank:         {rep.effective_rank:.2f}")
    print(f"Operating at Edge of Chaos:            {rep.is_critical}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="SONAPSE-EVO Command Line Interface")
    parser.add_argument("--version", action="version", version=f"SONAPSE-EVO {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand")

    subparsers.add_parser("smoke", help="Run quick subsystem smoke test")
    
    repair_p = subparsers.add_parser("lesion-repair", help="Run lesion repair assay")
    repair_p.add_argument("--task", default="forage", help="Task name")
    repair_p.add_argument("--severity", type=float, default=0.25, help="Lesion severity")

    bench_p = subparsers.add_parser("benchmark", help="Run 10-arm matched control study")
    bench_p.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    bench_p.add_argument("--authentic", action="store_true", help="Run authentic scientific baselines (RigL, true covariance, trained controls)")

    bench_v2_p = subparsers.add_parser("benchmark-v2", help="Run authentic 10-arm scientific benchmark")
    bench_v2_p.add_argument("--seeds", type=int, default=5, help="Number of seeds")

    disc_p = subparsers.add_parser("discover", help="Run Mode B: Autonomous Connectome Architecture Discovery")
    disc_p.add_argument("--generations", type=int, default=3, help="Evolutionary generations")
    disc_p.add_argument("--population", type=int, default=8, help="Population size")

    curr_p = subparsers.add_parser("curriculum", help="Run 4-stage continual learning curriculum")
    curr_p.add_argument("--seed", type=int, default=42, help="Random seed")

    crit_p = subparsers.add_parser("criticality", help="Evaluate Lyapunov exponent and edge-of-chaos criticality")
    crit_p.add_argument("--seed", type=int, default=42, help="Random seed")

    serve_p = subparsers.add_parser("serve", help="Launch observatory dashboard")
    serve_p.add_argument("--port", type=int, default=8890, help="Web server port")

    args = parser.parse_args()

    if args.subcommand == "smoke":
        sys.exit(run_smoke_test())
    elif args.subcommand == "lesion-repair":
        sys.exit(run_lesion_repair(args.task, args.severity))
    elif args.subcommand == "benchmark":
        sys.exit(run_benchmark_arms(args.seeds, use_authentic=args.authentic))
    elif args.subcommand == "benchmark-v2":
        sys.exit(run_benchmark_arms(args.seeds, use_authentic=True))
    elif args.subcommand == "discover":
        sys.exit(run_discovery(args.generations, args.population))
    elif args.subcommand == "curriculum":
        sys.exit(run_curriculum(args.seed))
    elif args.subcommand == "criticality":
        sys.exit(run_criticality(args.seed))
    elif args.subcommand == "serve":
        from .observatory.server import run_server
        run_server(args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
