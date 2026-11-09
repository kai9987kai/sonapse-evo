from __future__ import annotations

import numpy as np
import pytest

from sonapse_evo.substrate.topology_state import TopologyState
from sonapse_evo.substrate.recurrent_runtime import RecurrentSubstrate
from sonapse_evo.evaluation.curriculum import (
    InvertedSteeringObstacleTask,
    ContinualCurriculumBenchmark,
    CurriculumScorecard,
)


def test_inverted_steering_environment():
    env = InvertedSteeringObstacleTask(sensory_dim=155, max_steps=5, seed=42)
    obs = env.reset()
    assert obs.shape == (155,)

    # Step with steer command
    action = np.array([0.5, 0.8, 0.0, 0.0, 0.0])
    next_obs, reward, done, info = env.step(action)

    assert next_obs.shape == (155,)
    assert isinstance(reward, float)
    assert info["context"] == "inverted_steering"


def test_curriculum_evaluation():
    topology = TopologyState(total_neurons=768, num_regions=3, seed=42)
    substrate = RecurrentSubstrate(topology=topology, sensory_dim=155, seed=42)
    curriculum = ContinualCurriculumBenchmark(seed=42)

    # Use smaller max_steps on tasks to keep test fast
    for task in curriculum.tasks.values():
        task.max_steps = 5

    scorecard = curriculum.evaluate_curriculum(substrate)

    assert isinstance(scorecard, CurriculumScorecard)
    assert len(scorecard.task_returns) == 4
    assert "stage_1_navigation" in scorecard.task_returns
    assert "stage_2_memory" in scorecard.task_returns
    assert "stage_3_friction_drift" in scorecard.task_returns
    assert "stage_4_inverted_context" in scorecard.task_returns
    assert scorecard.forward_transfer_score >= 0.0
    assert scorecard.backward_retention_score >= 0.0
    assert scorecard.continual_learning_index >= 0.0
