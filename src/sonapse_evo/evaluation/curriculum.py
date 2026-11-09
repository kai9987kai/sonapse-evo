from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from .benchmark_runner import ForageNavigationTask, DelayedCueMatchingTask, run_benchmark_episode
from .generalization import DynamicFrictionForagingTask
from ..substrate.recurrent_runtime import RecurrentSubstrate


class InvertedSteeringObstacleTask(ForageNavigationTask):
    """Context-inverted obstacle avoidance environment.
    
    Steering commands are inverted (turning left causes rightward yaw).
    Tests agent capacity to override ingrained motor habits under distribution shift.
    """

    def __init__(self, sensory_dim: int = 155, max_steps: int = 40, seed: int = 777):
        super().__init__(sensory_dim=sensory_dim, max_steps=max_steps, seed=seed)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_idx += 1
        drive = float(action[0])
        # Inverted context: turn command sign reversed
        turn = -float(action[1])

        d_theta = np.clip(turn, -1.0, 1.0) * 0.3
        self.agent_heading = float((self.agent_heading + d_theta + np.pi) % (2 * np.pi) - np.pi)
        speed = max(0.0, np.clip(drive, -1.0, 1.0)) * 0.5
        self.agent_pos[0] += speed * np.cos(self.agent_heading)
        self.agent_pos[1] += speed * np.sin(self.agent_heading)

        dist = np.linalg.norm(self.target_pos - self.agent_pos)
        reward = max(0.01, (10.0 - dist) * 0.15)
        done = bool(dist < 0.6 or self.step_idx >= self.max_steps)

        return self._get_obs(), float(reward), done, {"dist": dist, "context": "inverted_steering"}


@dataclass
class CurriculumScorecard:
    """Comprehensive performance scorecard across sequenced curriculum tasks."""
    task_returns: Dict[str, float]
    forward_transfer_score: float
    backward_retention_score: float
    catastrophic_forgetting: float
    continual_learning_index: float
    passed_curriculum: bool


class ContinualCurriculumBenchmark:
    """Sequenced 4-stage continual adaptation benchmark evaluating zero catastrophic forgetting.
    
    Curriculum:
    - Stage 1: Basic Sensory-Motor Foraging (ForageNavigationTask)
    - Stage 2: Temporal Delayed Cue Matching (DelayedCueMatchingTask)
    - Stage 3: Non-Stationary Physics Drift (DynamicFrictionForagingTask)
    - Stage 4: Inverted Context & Obstacle Maneuver (InvertedSteeringObstacleTask)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.tasks = {
            "stage_1_navigation": ForageNavigationTask(seed=seed),
            "stage_2_memory": DelayedCueMatchingTask(delay_steps=10, seed=seed + 1),
            "stage_3_friction_drift": DynamicFrictionForagingTask(seed=seed + 2),
            "stage_4_inverted_context": InvertedSteeringObstacleTask(seed=seed + 3),
        }

    def evaluate_curriculum(
        self,
        substrate: RecurrentSubstrate,
        motor_head: Optional[nn.Module] = None,
        graft_registry: Optional[Any] = None,
    ) -> CurriculumScorecard:
        """Executes full multi-stage evaluation across all curriculum tasks."""
        if motor_head is None:
            motor_head = nn.Linear(64, 5, bias=False)
            with torch.no_grad():
                motor_head.weight.zero_()
                motor_head.weight[0, 0] = 2.0
                motor_head.weight[1, 1] = 2.0
                motor_head.weight[1, 2] = -2.0

        returns = {}

        # 1. Baseline Task 1 performance
        r1_init = run_benchmark_episode(self.tasks["stage_1_navigation"], substrate, motor_head, registry=graft_registry, seed=self.seed)
        returns["stage_1_navigation"] = r1_init

        # 2. Stage 2 Working Memory
        r2 = run_benchmark_episode(self.tasks["stage_2_memory"], substrate, motor_head, registry=graft_registry, seed=self.seed + 1)
        returns["stage_2_memory"] = r2

        # 3. Stage 3 Physical Parameter Drift
        r3 = run_benchmark_episode(self.tasks["stage_3_friction_drift"], substrate, motor_head, registry=graft_registry, seed=self.seed + 2)
        returns["stage_3_friction_drift"] = r3

        # 4. Stage 4 Context Inversion
        r4 = run_benchmark_episode(self.tasks["stage_4_inverted_context"], substrate, motor_head, registry=graft_registry, seed=self.seed + 3)
        returns["stage_4_inverted_context"] = r4

        # 5. Measure Retention: Re-evaluate Stage 1 after exposures
        r1_final = run_benchmark_episode(self.tasks["stage_1_navigation"], substrate, motor_head, registry=graft_registry, seed=self.seed)

        forgetting = float(max(0.0, r1_init - r1_final))
        retention = float(r1_final / max(1e-4, r1_init))
        forward_transfer = float((r3 + r4) / max(1e-4, 2 * r1_init))
        cli = float(np.mean([retention, forward_transfer]))

        return CurriculumScorecard(
            task_returns=returns,
            forward_transfer_score=round(forward_transfer, 3),
            backward_retention_score=round(retention, 3),
            catastrophic_forgetting=round(forgetting, 3),
            continual_learning_index=round(cli, 3),
            passed_curriculum=bool(retention >= 0.85 and forgetting <= 5.0),
        )
