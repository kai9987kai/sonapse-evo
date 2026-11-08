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

    def __init__(self, sensory_dim: int = 155, max_steps: int = 60, seed: int = 42):
        self.sensory_dim = sensory_dim
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.agent_pos = np.zeros(2)
        self.target_pos = np.zeros(2)
        self.step_idx = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = self.rng.uniform(-2.0, 2.0, size=2)
        self.target_pos = self.rng.uniform(-5.0, 5.0, size=2)
        self.step_idx = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.sensory_dim, dtype=np.float32)
        rel_pos = self.target_pos - self.agent_pos
        dist = np.linalg.norm(rel_pos)
        angle = np.arctan2(rel_pos[1], rel_pos[0])

        obs[0] = float(np.clip(dist / 10.0, 0.0, 1.0))
        obs[1] = float(np.sin(angle))
        obs[2] = float(np.cos(angle))
        # Add sensory noise to other dimensions
        obs[3:32] = self.rng.normal(0.0, 0.05, size=29)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_idx += 1
        drive = float(action[0])
        turn = float(action[1])

        # Kinematic update
        speed = np.clip(drive, -1.0, 1.0) * 0.5
        d_theta = np.clip(turn, -1.0, 1.0) * 0.3
        self.agent_pos[0] += speed * np.cos(d_theta)
        self.agent_pos[1] += speed * np.sin(d_theta)

        dist = float(np.linalg.norm(self.target_pos - self.agent_pos))
        # Continuous smooth distance shaping
        reward = max(0.01, (10.0 - dist) * 0.1)

        done = (dist < 0.5) or (self.step_idx >= self.max_steps)
        if dist < 0.5:
            reward += 10.0  # target reached bonus

        return self._get_obs(), reward, done, {"dist": dist, "success": dist < 0.5}


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
) -> float:
    """Runs a single episode and returns total scalar return."""
    obs = env.reset(seed=seed)
    substrate.reset_state()
    total_reward = 0.0
    done = False

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
            action = torch.tanh(motor_head(new_r[:64])).numpy()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return float(total_reward)
