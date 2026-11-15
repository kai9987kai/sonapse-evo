"""Named ecology configurations, the multi-agent ecology and the environment registry.

The registry also lists the two EXISTING tasks (``forage`` and ``delayed_cue``) so that every environment,
old or new, is selected the same way.  Nothing is replaced.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...evaluation.benchmark_runner import DelayedCueMatchingTask, ForageNavigationTask
from .arena import EcologyArena, EcologyConfig

ECOLOGY_PRESETS: Dict[str, Callable[[], EcologyConfig]] = {
    "neuroweave_1200x660": lambda: EcologyConfig(),
    "dense_obstacles": lambda: EcologyConfig(name="dense_obstacles", n_obstacles=20, obstacle_layout="random"),
    "sparse_resources": lambda: EcologyConfig(name="sparse_resources", resources="scarce", energy_decay=0.12),
    "clustered_resources": lambda: EcologyConfig(name="clustered_resources", resources="clustered"),
    "thermal_gradient": lambda: EcologyConfig(name="thermal_gradient", n_thermal=6, n_obstacles=6),
    "disturbed": lambda: EcologyConfig(name="disturbed", wind=3.0, actuator_slip=0.25),
    "dynamic_conditions": lambda: EcologyConfig(name="dynamic_conditions", dynamic=True, obstacle_speed=1.5),
    "rule_shift": lambda: EcologyConfig(name="rule_shift", rule_shift_period=60),
    "long_horizon": lambda: EcologyConfig(name="long_horizon", waypoints=5, max_steps=600),
    "memory_navigation": lambda: EcologyConfig(name="memory_navigation", memory_cue_steps=6, n_obstacles=4, moving_obstacles=False),
    "small_arena": lambda: EcologyConfig(name="small_arena", width=600.0, height=330.0, n_obstacles=4, n_food=8, max_steps=150),
    "corridor": lambda: EcologyConfig(name="corridor", obstacle_layout="corridor", n_obstacles=8),
}

_DESCRIPTIONS = {
    "neuroweave_1200x660": "reference ecology: ten moving obstacles, eighteen food patches, three thermal zones",
    "dense_obstacles": "twenty randomly placed moving obstacles",
    "sparse_resources": "a quarter of the food and faster energy loss",
    "clustered_resources": "food concentrated in three clusters",
    "thermal_gradient": "six thermal zones, fewer obstacles",
    "disturbed": "wind and actuator slip: observed motion differs from commanded motion",
    "dynamic_conditions": "obstacle speed and food density change over time",
    "rule_shift": "the reward rule flips every 60 steps (approach goal <-> avoid goal, seek food)",
    "long_horizon": "five ordered waypoints over a 600-step episode",
    "memory_navigation": "goal bearing visible for six steps per goal, then it must be remembered",
    "small_arena": "600 x 330 arena for quick experiments",
    "corridor": "obstacles arranged as a slalom corridor",
}


class MultiAgentEcology:
    """Several agents in one shared arena, competing for the same food and avoiding each other."""

    def __init__(self, config: Optional[EcologyConfig] = None, n_agents: int = 2, seed: int = 42):
        if n_agents < 1:
            raise ValueError("n_agents must be at least 1")
        self.n_agents = int(n_agents)
        self.agents = [EcologyArena(config, seed=seed + i) for i in range(self.n_agents)]
        self.max_steps = self.agents[0].max_steps
        self.sensory_dim = self.agents[0].sensory_dim
        self.reset(seed)

    def _share_world(self) -> None:
        world = self.agents[0]
        for i, agent in enumerate(self.agents):
            if i > 0:
                agent.owns_world = False
                agent.obstacles, agent.food, agent.thermal, agent.clusters = world.obstacles, world.food, world.thermal, world.clusters
            c = agent.config
            spacing = min(80.0, (c.height - 56.0) / max(1, self.n_agents))
            agent.pos = np.clip(agent.pos + np.array([0.0, spacing * (i - (self.n_agents - 1) / 2.0)]),
                                [28.0, 28.0], [c.width - 28.0, c.height - 28.0])          # spread the start line inside the walls
            agent.others = [tuple(o.pos) for j, o in enumerate(self.agents) if j != i]

    def reset(self, seed: Optional[int] = None) -> List[np.ndarray]:
        for i, agent in enumerate(self.agents):
            agent.owns_world = i == 0
            agent.reset(None if seed is None else seed + i)
        self._share_world()
        return [agent._observe() for agent in self.agents]

    def step(self, actions) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        rewards, infos, done = [], [], False
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            agent.others = [tuple(o.pos) for j, o in enumerate(self.agents) if j != i]
            _, reward, finished, info = agent.step(action)
            rewards.append(reward)
            infos.append(info)
            done = done or finished
        obs = []
        for i, agent in enumerate(self.agents):
            agent.others = [tuple(o.pos) for j, o in enumerate(self.agents) if j != i]
            obs.append(agent._observe())
        return obs, rewards, done, {"agents": infos, "food_remaining": len(self.agents[0].food),
                                    "total_food_eaten": int(sum(a.food_eaten for a in self.agents))}

    def render_state(self) -> Dict[str, Any]:
        state = self.agents[0].render_state()
        state["agents"] = [a.render_state()["agent"] for a in self.agents]
        return state


def _ecology(name: str) -> Callable[..., EcologyArena]:
    def factory(seed: int = 42, **overrides: Any) -> EcologyArena:
        config = ECOLOGY_PRESETS[name]()
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise KeyError(f"unknown ecology option '{key}'")
            setattr(config, key, value)
        return EcologyArena(config, seed=seed)

    return factory


ENVIRONMENTS: Dict[str, Callable[..., Any]] = {
    "forage": lambda seed=42, **kw: ForageNavigationTask(sensory_dim=kw.get("sensory_dim", 155), max_steps=kw.get("max_steps", 40), seed=seed),
    "delayed_cue": lambda seed=42, **kw: DelayedCueMatchingTask(sensory_dim=kw.get("sensory_dim", 155), delay_steps=kw.get("delay_steps", 10), seed=seed),
}
ENVIRONMENTS.update({name: _ecology(name) for name in ECOLOGY_PRESETS})
ENVIRONMENTS["multi_agent"] = lambda seed=42, **kw: MultiAgentEcology(ECOLOGY_PRESETS[kw.pop("preset", "neuroweave_1200x660")](),
                                                                        n_agents=kw.get("n_agents", 2), seed=seed)


def make_environment(name: str, seed: int = 42, **overrides: Any):
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown environment '{name}'. Available: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[name](seed=seed, **overrides)


def list_environments() -> List[Dict[str, Any]]:
    out = [{"name": "forage", "kind": "existing", "observation_dim": 155, "description": "existing 2-D foraging navigation task"},
           {"name": "delayed_cue", "kind": "existing", "observation_dim": 155, "description": "existing delayed cue matching (working memory)"}]
    out += [{"name": name, "kind": "ecology", "observation_dim": 155, "description": _DESCRIPTIONS[name]} for name in ECOLOGY_PRESETS]
    out.append({"name": "multi_agent", "kind": "ecology", "observation_dim": 155, "description": "several agents sharing one ecology"})
    return out


__all__ = ["ECOLOGY_PRESETS", "MultiAgentEcology", "ENVIRONMENTS", "make_environment", "list_environments"]
