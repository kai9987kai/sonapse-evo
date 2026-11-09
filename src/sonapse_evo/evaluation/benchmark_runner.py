from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn

from ..substrate.recurrent_runtime import RecurrentSubstrate


class ForageNavigationTask:
    """Continuous 2D sensory-motor foraging task.
    
    Sensory observation: 155-dim (distance/angles to food targets, obstacle rangefinders, velocity).
    Action: 5-dim [drive, turn, reach, height, grip].
    Goal: navigate to target positions, accumulate food reward.
    """

    def __init__(self, sensory_dim: int = 155, max_steps: int = 40, seed: int = 42):
        self.sensory_dim = sensory_dim
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.agent_pos = np.zeros(2)
        self.agent_heading = 0.0
        self.target_pos = np.zeros(2)
        self.step_idx = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = self.rng.uniform(-3.0, 3.0, size=2)
        self.agent_heading = float(self.rng.uniform(-np.pi, np.pi))
        self.target_pos = self.rng.uniform(-4.0, 4.0, size=2)
        self.step_idx = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.sensory_dim, dtype=np.float32)
        rel_pos = self.target_pos - self.agent_pos
        dist = np.linalg.norm(rel_pos)
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        heading_err = float((target_angle - self.agent_heading + np.pi) % (2 * np.pi) - np.pi)

        obs[0] = float(np.clip(dist / 10.0, 0.0, 1.0))
        obs[1] = float(np.sin(heading_err))
        obs[2] = float(np.cos(heading_err))
        # Sensory background noise
        obs[3:32] = self.rng.normal(0.0, 0.02, size=29)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_idx += 1
        drive = float(action[0])
        turn = float(action[1])

        # Proper 2D unicycle kinematic update
        d_theta = np.clip(turn, -1.0, 1.0) * 0.3
        self.agent_heading = float((self.agent_heading + d_theta + np.pi) % (2 * np.pi) - np.pi)
        speed = max(0.0, np.clip(drive, -1.0, 1.0)) * 0.5
        self.agent_pos[0] += speed * np.cos(self.agent_heading)
        self.agent_pos[1] += speed * np.sin(self.agent_heading)

        dist = float(np.linalg.norm(self.target_pos - self.agent_pos))
        reward = max(0.01, (10.0 - dist) * 0.15)

        done = (dist < 0.8) or (self.step_idx >= self.max_steps)
        if dist < 0.8:
            reward += 15.0  # target reached bonus

        return self._get_obs(), reward, done, {"dist": dist, "success": dist < 0.8}

    def rollout_episode(
        self,
        substrate: RecurrentSubstrate,
        motor_head: Optional[nn.Module] = None,
        graft_registry: Optional[Any] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> float:
        """Helper to run a full episode rollout on this task."""
        if motor_head is None:
            motor_head = nn.Linear(64, 5, bias=False)
            with torch.no_grad():
                motor_head.weight.zero_()
                motor_head.weight[0, 0] = 2.0
                motor_head.weight[1, 1] = 2.0
                motor_head.weight[1, 2] = -2.0
        return run_benchmark_episode(
            self,
            substrate,
            motor_head,
            registry=graft_registry,
            seed=seed if seed is not None else 42,
        )


class DelayedCueMatchingTask:
    """Working memory task: cue presented in step 0, delay of N steps, matching action required."""

    def __init__(self, sensory_dim: int = 155, delay_steps: int = 10, seed: int = 42):
        self.sensory_dim = sensory_dim
        self.delay_steps = delay_steps
        self.rng = np.random.default_rng(seed)
        self.target_cue = 0
        self.step_idx = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.target_cue = int(self.rng.choice([-1, 1]))
        self.step_idx = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.sensory_dim, dtype=np.float32)
        if self.step_idx == 0:
            obs[0] = float(self.target_cue)  # Cue visible only at step 0
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_idx += 1
        done = self.step_idx >= (self.delay_steps + 1)
        reward = 0.0

        if done:
            chosen = 1 if action[0] > 0 else -1
            if chosen == self.target_cue:
                reward = 10.0
            else:
                reward = -5.0

        return self._get_obs(), reward, done, {"target_cue": self.target_cue}


def run_benchmark_episode(
    env: Any,
    substrate: RecurrentSubstrate,
    motor_head: nn.Module,
    registry: Optional[Any] = None,
    seed: int = 42,
    enable_grafts: bool = True,
    motor_offset: int = 512,
) -> float:
    """Runs a single episode and returns total scalar return."""
    obs = env.reset(seed=seed)
    substrate.reset_state()
    total_reward = 0.0
    done = False

    readout_start = motor_offset if motor_offset + 64 <= substrate.total_neurons else 0

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        curr_r = substrate.r

        # Compute auxiliary graft current if enabled
        aux_current = None
        if enable_grafts and registry is not None:
            aux_current = registry.compute_total_current(curr_r.unsqueeze(0)).squeeze(0)

        # Forward substrate microsteps
        new_r = substrate.forward_step(obs_t, aux_current=aux_current)

        # Action from motor head
        with torch.no_grad():
            action = torch.tanh(motor_head(new_r[readout_start:readout_start + 64])).numpy()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return float(total_reward)
