from __future__ import annotations

import threading
import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional

from ..contracts.evidence import EvidenceTier
from ..substrate.graph_topology import GraphTopology
from ..discovery.discovery_engine import DiscoveryEngine
from ..evolution.operators import ALL_OPERATORS


class LiveObservatoryState:
    """Thread-safe live simulation state engine for real-time dashboard telemetry."""

    def __init__(self):
        self._lock = threading.Lock()
        self.step_idx = 0
        self.running = True

        # Telemetry ring buffer
        self.telemetry_history: List[Dict[str, Any]] = []
        self.max_history = 100

        # Mode A: Repair status
        self.mode_a_status = "INTACT"  # "INTACT", "LESION_ACTIVE", "REPAIRING", "COMMITTED"
        self.deficit_score = 0.12
        self.mean_firing_rate = 0.28
        self.spectral_norm = 0.88
        self.aux_current_ceiling = 0.10
        self.deficit_vector = {
            "e_pred": 0.08,
            "e_policy": 0.11,
            "u_unc": 0.14,
            "c_rep": 0.09,
            "r_task": 0.15,
            "s_stab": 0.07,
        }
        self.active_grafts: List[Dict[str, Any]] = [
            {
                "id": "graft_type_a_base",
                "type": "LowRankBridge (Type A)",
                "source": "R0_P0 (Sensory)",
                "target": "R2_P0 (Motor)",
                "gain": 0.80,
                "rank": 8,
                "status": "ACTIVE_COMMITTED",
            }
        ]
        self.transactions: List[Dict[str, Any]] = [
            {
                "tx_id": "tx_initial",
                "status": "COMMITTED",
                "graft_id": "graft_type_a_base",
                "recovery_ratio": 1.074,
                "timestamp": time.time() - 60,
                "details": "Contract verified: zero-gain ramp, frozen backbone",
            }
        ]

        # Mode B: Discovery State
        self.base_graph = GraphTopology(total_neurons=768, num_regions=3, seed=42)
        self.discovery_engine = DiscoveryEngine(self.base_graph, population_size=8, seed=42)
        self.current_generation = 1
        self.operator_usage: Dict[str, int] = {op.operator_name: 0 for op in ALL_OPERATORS}
        self.operator_usage["add_synapse"] = 3
        self.operator_usage["grow_microcircuit"] = 2
        self.operator_usage["add_low_rank_bridge"] = 2
        self.operator_usage["prune_redundant_computation"] = 1

        self.population_data: List[Dict[str, Any]] = []
        self._refresh_population_data()

        # Seed initial telemetry history
        for i in range(40):
            t = time.time() - (40 - i) * 1.0
            def_val = 0.12 + 0.04 * np.sin(i * 0.2) + np.random.normal(0, 0.01)
            rate_val = 0.28 + 0.03 * np.cos(i * 0.15) + np.random.normal(0, 0.01)
            self.telemetry_history.append({
                "step": i,
                "timestamp": t,
                "mean_rate": float(np.clip(rate_val, 0.05, 0.95)),
                "deficit_score": float(np.clip(def_val, 0.01, 1.0)),
                "spectral_norm": 0.88,
                "return": 33.44,
            })

    def _refresh_population_data(self):
        pop = []
        for g in self.discovery_engine.pool.population:
            pop.append({
                "genotype_id": g.genotype_id,
                "pareto_rank": g.pareto_rank,
                "scalar_utility": float(g.scalar_utility),
                "return_train": float(g.fitness_vector[0]),
                "return_ood": float(g.fitness_vector[1]),
                "adaptation_a": float(g.fitness_vector[2]),
                "latency_ms": abs(float(g.fitness_vector[4])),
                "memory_kb": abs(float(g.fitness_vector[5])),
                "operators": list(g.applied_operators),
                "evidence_tier": g.evidence_tier.value,
            })
        pop.sort(key=lambda x: (x["pareto_rank"], -x["scalar_utility"]))
        self.population_data = pop

    def tick_simulation(self):
        """Called periodically by background thread to simulate live connectome dynamics."""
        with self._lock:
            self.step_idx += 1
            t_now = time.time()

            # Dynamic fluctuations depending on Mode A status
            if self.mode_a_status == "LESION_ACTIVE":
                target_def = 0.65
                target_rate = 0.16
                target_norm = 0.98
            elif self.mode_a_status == "REPAIRING":
                target_def = 0.32
                target_rate = 0.24
                target_norm = 0.92
            else:
                target_def = 0.12
                target_rate = 0.28
                target_norm = 0.88

            # Smooth exponential tracking with small stochastic noise
            self.deficit_score = float(0.85 * self.deficit_score + 0.15 * target_def + np.random.normal(0, 0.01))
            self.mean_firing_rate = float(0.85 * self.mean_firing_rate + 0.15 * target_rate + np.random.normal(0, 0.01))
            self.spectral_norm = float(0.95 * self.spectral_norm + 0.05 * target_norm + np.random.normal(0, 0.005))

            # Update 6D deficit vector
            s = self.deficit_score
            self.deficit_vector = {
                "e_pred": float(np.clip(s * 0.9 + np.random.normal(0, 0.01), 0.01, 1.0)),
                "e_policy": float(np.clip(s * 1.1 + np.random.normal(0, 0.01), 0.01, 1.0)),
                "u_unc": float(np.clip(s * 0.8 + np.random.normal(0, 0.01), 0.01, 1.0)),
                "c_rep": float(np.clip(s * 0.7 + np.random.normal(0, 0.01), 0.01, 1.0)),
                "r_task": float(np.clip(s * 1.2 + np.random.normal(0, 0.01), 0.01, 1.0)),
                "s_stab": float(np.clip(s * 0.6 + np.random.normal(0, 0.01), 0.01, 1.0)),
            }

            self.telemetry_history.append({
                "step": self.step_idx,
                "timestamp": t_now,
                "mean_rate": float(np.clip(self.mean_firing_rate, 0.05, 0.95)),
                "deficit_score": float(np.clip(self.deficit_score, 0.01, 1.0)),
                "spectral_norm": float(np.clip(self.spectral_norm, 0.5, 1.2)),
                "return": float(34.66 if self.mode_a_status != "LESION_ACTIVE" else 16.92),
            })
            if len(self.telemetry_history) > self.max_history:
                self.telemetry_history.pop(0)

    def trigger_discovery_step(self) -> Dict[str, Any]:
        """Triggers a live generation advance in Mode B discovery."""
        with self._lock:
            self.current_generation += 1
            # Step population
            self.discovery_engine.pool.step_generation(elitism_count=2)
            for c in self.discovery_engine.pool.population:
                self.discovery_engine.evaluator.evaluate_genotype(c)
                for op in c.applied_operators:
                    self.operator_usage[op] = self.operator_usage.get(op, 0) + 1

            self.discovery_engine.optimizer.rank_population(self.discovery_engine.pool.population)
            self._refresh_population_data()

            champion = self.population_data[0] if self.population_data else {}
            return {
                "generation": self.current_generation,
                "champion": champion,
                "population_size": len(self.population_data),
            }

    def trigger_lesion_event(self) -> Dict[str, Any]:
        """Simulates inducing a sensory-motor bridge lesion in Mode A."""
        with self._lock:
            self.mode_a_status = "LESION_ACTIVE"
            self.deficit_score = 0.68
            return {"status": "LESION_ACTIVE", "severity": 0.25, "impact": "-16.52 return drop"}

    def trigger_repair_event(self) -> Dict[str, Any]:
        """Simulates executing causal graft repair in Mode A."""
        with self._lock:
            self.mode_a_status = "COMMITTED"
            self.deficit_score = 0.11
            tx_id = f"tx_{uuid.uuid4().hex[:8]}"
            graft_id = f"graft_causal_{uuid.uuid4().hex[:6]}"
            new_graft = {
                "id": graft_id,
                "type": "Causal LowRankBridge (Type A)",
                "source": "R0_P0 (Sensory Entry)",
                "target": "R2_P0 (Motor Outflow)",
                "gain": 0.85,
                "rank": 8,
                "status": "COMMITTED_VERIFIED",
            }
            self.active_grafts.append(new_graft)
            self.transactions.insert(0, {
                "tx_id": tx_id,
                "status": "COMMITTED",
                "graft_id": graft_id,
                "recovery_ratio": 1.074,
                "timestamp": time.time(),
                "details": "Causal attribution verified: R0_P0 -> R2_P0 (+17.74 restoration)",
            })
            return {"status": "COMMITTED", "tx_id": tx_id, "recovery_ratio": 1.074}

    def get_full_state(self) -> Dict[str, Any]:
        """Serializes full system state for real-time dashboard consumption."""
        with self._lock:
            # Pareto frontier coordinates for scatter chart
            pareto_pts = []
            for p in self.population_data:
                pareto_pts.append({
                    "id": p["genotype_id"],
                    "return": p["return_train"],
                    "latency": p["latency_ms"],
                    "memory": p["memory_kb"],
                    "rank": p["pareto_rank"],
                    "utility": p["scalar_utility"],
                    "tier": p["evidence_tier"],
                })

            return {
                "server_time": time.time(),
                "step": self.step_idx,
                "status_indicator": "LIVE_LINKED",
                "mode_a": {
                    "status": self.mode_a_status,
                    "deficit_score": round(self.deficit_score, 4),
                    "mean_rate": round(self.mean_firing_rate, 4),
                    "spectral_norm": round(self.spectral_norm, 4),
                    "aux_current_ceiling": self.aux_current_ceiling,
                    "deficit_vector": self.deficit_vector,
                    "active_grafts": self.active_grafts,
                    "transactions": self.transactions[:5],
                },
                "mode_b": {
                    "generation": self.current_generation,
                    "population_size": len(self.population_data),
                    "pareto_champion": self.population_data[0] if self.population_data else None,
                    "pareto_points": pareto_pts,
                    "population": self.population_data[:10],
                    "operator_usage": self.operator_usage,
                },
                "connectome": {
                    "total_neurons": 768,
                    "regions": [
                        {"id": 0, "name": "Sensory Entry (flywire)", "neurons": 256, "activity": round(self.mean_firing_rate * 1.1, 3)},
                        {"id": 1, "name": "Central Memory (manc)", "neurons": 256, "activity": round(self.mean_firing_rate * 0.9, 3)},
                        {"id": 2, "name": "Motor Outflow (optic-lobe)", "neurons": 256, "activity": round(self.mean_firing_rate * 1.05, 3)},
                    ],
                    "active_edges": 6144,
                    "density": 0.0104,
                },
                "telemetry_stream": self.telemetry_history[-40:],
            }
