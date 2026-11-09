from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from .benchmark_runner import ForageNavigationTask
from ..substrate.recurrent_runtime import RecurrentSubstrate


class DynamicFrictionForagingTask(ForageNavigationTask):
    """Out-of-distribution foraging environment with non-stationary surface friction.
    
    Tests agent generalization to physical perturbations and dynamical shifts.
    """

    def __init__(self, sensory_dim: int = 155, max_steps: int = 40, seed: int = 999):
        super().__init__(sensory_dim=sensory_dim, max_steps=max_steps, seed=seed)
        self.friction = 1.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_idx += 1
        # Friction fluctuates stochastically
        self.friction = float(np.clip(self.friction + self.rng.normal(0.0, 0.1), 0.3, 2.0))
        
        drive = float(action[0]) / self.friction
        turn = float(action[1]) / self.friction

        d_theta = np.clip(turn, -1.0, 1.0) * 0.3
        self.agent_heading = float((self.agent_heading + d_theta + np.pi) % (2 * np.pi) - np.pi)
        speed = max(0.0, np.clip(drive, -1.0, 1.0)) * 0.5
        self.agent_pos[0] += speed * np.cos(self.agent_heading)
        self.agent_pos[1] += speed * np.sin(self.agent_heading)

        dist = np.linalg.norm(self.target_pos - self.agent_pos)
        reward = max(0.01, (10.0 - dist) * 0.15)
        done = bool(dist < 0.6 or self.step_idx >= self.max_steps)

        return self._get_obs(), float(reward), done, {"dist": dist, "friction": self.friction}


class MultiTaskContinualBenchmark:
    """Evaluates continual learning, forward transfer, and backward catastrophic forgetting."""

    def __init__(self, seed: int = 42):
        self.task1 = ForageNavigationTask(seed=seed)
        self.task2 = DynamicFrictionForagingTask(seed=seed + 100)

    def evaluate_continual_profile(
        self,
        substrate: RecurrentSubstrate,
        graft_registry: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Measures Task 1 baseline return, Task 2 adaptation return, and Task 1 retention."""
        # 1. Initial Task 1 performance
        r1_init = self.task1.rollout_episode(substrate, graft_registry=graft_registry, max_steps=40)

        # 2. Task 2 (OOD) performance
        r2 = self.task2.rollout_episode(substrate, graft_registry=graft_registry, max_steps=40)

        # 3. Post-adaptation Task 1 retention
        r1_post = self.task1.rollout_episode(substrate, graft_registry=graft_registry, max_steps=40)

        forgetting = float(max(0.0, r1_init - r1_post))
        retention_ratio = float(r1_post / max(1e-4, r1_init))

        return {
            "r1_initial": float(r1_init),
            "r2_ood": float(r2),
            "r1_post": float(r1_post),
            "forgetting": forgetting,
            "retention_ratio": retention_ratio,
        }
