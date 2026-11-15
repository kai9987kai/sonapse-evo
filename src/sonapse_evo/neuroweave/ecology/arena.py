"""Embodied ecology: a 1200 x 660 arena with moving obstacles, replenishing food, thermal zones and energy.

Adapted from the NEUROWEAVE-LAB ``World`` and exposed through the same interface as the existing tasks
(``reset(seed) -> obs``, ``step(action) -> (obs, reward, done, info)``, 155-dim observations, 5-dim actions
with ``action[0]`` = drive and ``action[1]`` = turn), so the existing agent controller and
``run_benchmark_episode`` work on it unchanged.  It is an *additional* environment; nothing is replaced.

Actions have real consequences: unicycle motion, energy use, obstacle collisions (penalty and bounce),
food consumption with replenishment, goal capture, energy depletion and respawn, thermal exposure, and
optional disturbances (wind, actuator slip) that make the observed displacement differ from the commanded
one.  ``info["predicted_outcome"]`` / ``info["observed_outcome"]`` carry both displacements for the
action-grounded self-model, and ``sensory_channels()`` returns the sixteen sensory channels.

Observation layout (all values bounded to [-1, 1]; unlisted entries are 0)
    0  distance to goal / diagonal      1  sin(heading error)        2  cos(heading error)
    3,4 sin/cos bearing of nearest obstacle   5  obstacle proximity   6  energy
    7,8 sin/cos bearing of nearest food       9  food strength       10  thermal value
    11 speed   12 collision flag   13,14 sin/cos of day phase   15-18 wall proximity (left, right, top, bottom)
    19,20 goal offset (x, y)   21 active rule   22 waypoint progress   23 goal cue visible
    24 last disturbance magnitude   25,26 sin/cos bearing of nearest other agent   27 other-agent proximity
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import Provenance
from ..sensory.channels import SENSORY_CHANNELS

_REFERENCE_OBSTACLES = [(560, 200, 52, .2), (780, 470, 62, 1.7), (370, 520, 36, 3.1), (230, 144, 26, .7), (950, 152, 38, 2.1),
                        (1010, 540, 30, 3.7), (630, 530, 24, 4.2), (140, 460, 20, 5.2), (875, 350, 28, 5.8), (430, 350, 17, 2.8)]
_REFERENCE_THERMAL = [(320, 220, 170, .25), (890, 420, 220, .85), (610, 110, 130, .55)]


@dataclass
class EcologyConfig:
    name: str = "neuroweave_1200x660"
    width: float = 1200.0
    height: float = 660.0
    n_obstacles: int = 10
    n_food: int = 18
    n_thermal: int = 3
    moving_obstacles: bool = True
    obstacle_speed: float = 1.0
    obstacle_layout: str = "reference"       # reference | random | corridor | ring
    resources: str = "uniform"               # uniform | clustered | patchy | scarce
    replenish_food: bool = True
    max_steps: int = 300
    speed: float = 14.0                      # arena units per step at full drive
    turn_rate: float = 0.3                   # radians per step at full turn
    energy_decay: float = 0.05
    goal_radius: float = 46.0
    wind: float = 0.0                        # constant push (arena units per step) plus gusts
    actuator_slip: float = 0.0               # probability that a step's motor command is degraded
    dynamic: bool = False                    # obstacle speed and food density drift over time
    rule_shift_period: int = 0               # > 0: the reward rule flips every this many steps
    waypoints: int = 0                       # > 0: long-horizon task with this many ordered waypoints
    memory_cue_steps: int = 0                # > 0: goal bearing is visible only this many steps per goal
    sensory_dim: int = 155

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class EcologyArena:
    sensory_provenance = Provenance.SYNTHETIC_SIMULATION

    def __init__(self, config: Optional[EcologyConfig] = None, seed: int = 42):
        self.config = replace(config) if config is not None else EcologyConfig()
        self.sensory_dim = int(self.config.sensory_dim)
        self.max_steps = int(self.config.max_steps)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.others: List[Tuple[float, float]] = []        # positions of other agents (multi-agent ecology)
        self.owns_world = True
        self.reset(seed)

    # ------------------------------------------------------------------ world
    def _food_position(self) -> Tuple[float, float]:
        c = self.config
        if c.resources == "clustered":
            cx, cy = self.clusters[self.rng.integers(len(self.clusters))]
            return (float(np.clip(cx + self.rng.normal(0, 60), 60, c.width - 60)), float(np.clip(cy + self.rng.normal(0, 60), 50, c.height - 50)))
        if c.resources == "patchy":
            side = self.rng.integers(2)
            return (float(self.rng.uniform(70, c.width * 0.3) if side == 0 else self.rng.uniform(c.width * 0.7, c.width - 70)),
                    float(self.rng.uniform(50, c.height - 50)))
        return float(self.rng.uniform(70, c.width - 70)), float(self.rng.uniform(50, c.height - 50))

    def _new_goal(self) -> np.ndarray:
        c = self.config
        return np.array([self.rng.uniform(90, c.width - 90), self.rng.uniform(76, c.height - 76)])

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        c = self.config
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.array([210.0 * c.width / 1200.0, c.height / 2.0])
        self.heading = float(self.rng.uniform(-0.6, 0.6))
        self.t = 0
        self.energy, self.score, self.collisions, self.food_eaten, self.respawns = 100.0, 0, 0, 0, 0
        self.reward, self.speed_now, self.touch, self.ping, self.disturbance = 0.0, 0.0, 0.0, 0.0, 0.0
        self.rule, self.waypoint_index, self.goal_age = 0, 0, 0
        self.path: List[Tuple[float, float]] = []
        self.wind_vec = c.wind * np.array([np.cos(0.7), np.sin(0.7)])
        if self.owns_world:
            sx, sy = c.width / 1200.0, c.height / 660.0
            if c.obstacle_layout == "reference":
                base = [_REFERENCE_OBSTACLES[i % 10] for i in range(c.n_obstacles)]
                self.obstacles = [[x * sx + (i // 10) * 37.0, y * sy, r, ph + 0.3 * (i // 10)] for i, (x, y, r, ph) in enumerate(base)]
            elif c.obstacle_layout == "corridor":
                self.obstacles = [[c.width * (0.25 + 0.5 * i / max(1, c.n_obstacles - 1)), c.height * (0.3 if i % 2 else 0.7), 44.0, 0.9 * i]
                                  for i in range(c.n_obstacles)]
            elif c.obstacle_layout == "ring":
                self.obstacles = [[c.width / 2 + 0.3 * c.width * np.cos(2 * np.pi * i / c.n_obstacles),
                                   c.height / 2 + 0.3 * c.height * np.sin(2 * np.pi * i / c.n_obstacles), 34.0, 0.6 * i] for i in range(c.n_obstacles)]
            else:
                self.obstacles = [[self.rng.uniform(300, c.width - 100), self.rng.uniform(80, c.height - 80), self.rng.uniform(18, 60),
                                   self.rng.uniform(0, 6.28)] for _ in range(c.n_obstacles)]
            self.clusters = [(self.rng.uniform(200, c.width - 200), self.rng.uniform(120, c.height - 120)) for _ in range(3)]
            n_food = max(1, c.n_food // 4) if c.resources == "scarce" else c.n_food
            self.food = [[*self._food_position(), 12.0, float(0.35 + 0.65 * self.rng.random())] for _ in range(n_food)]
            zones = [_REFERENCE_THERMAL[i % 3] for i in range(c.n_thermal)]
            self.thermal = [(x * sx, y * sy, r, v) for x, y, r, v in zones]
        self.route = [self._new_goal() for _ in range(c.waypoints)] if c.waypoints > 0 else []
        self.goal = self.route[0] if self.route else np.array([1040.0 * c.width / 1200.0, c.height / 2.0])
        return self._observe()

    # ---------------------------------------------------------------- sensing
    def _nearest(self, items: List[Any]) -> Tuple[Optional[Any], float]:
        if not items:
            return None, float("inf")
        d = [float(np.hypot(it[0] - self.pos[0], it[1] - self.pos[1])) for it in items]
        i = int(np.argmin(d))
        return items[i], d[i]

    def _bearing(self, x: float, y: float) -> float:
        return _wrap(np.arctan2(y - self.pos[1], x - self.pos[0]) - self.heading)

    def thermal_value(self) -> float:
        v = 0.5
        for x, y, r, value in self.thermal:
            v += (value - 0.5) * max(0.0, 1.0 - np.hypot(x - self.pos[0], y - self.pos[1]) / r)
        return float(np.clip(v, 0.0, 1.0))

    def cue_visible(self) -> bool:
        k = self.config.memory_cue_steps
        return k <= 0 or self.goal_age < k

    def _observe(self) -> np.ndarray:
        c = self.config
        obs = np.zeros(self.sensory_dim, dtype=np.float32)
        rel = self.goal - self.pos
        dist = float(np.hypot(*rel))
        if self.cue_visible():
            err = self._bearing(self.goal[0], self.goal[1])
            obs[0] = np.clip(dist / np.hypot(c.width, c.height), 0.0, 1.0)
            obs[1], obs[2] = np.sin(err), np.cos(err)
            obs[19], obs[20] = np.clip(rel[0] / c.width, -1, 1), np.clip(rel[1] / c.height, -1, 1)
        ob, od = self._nearest(self.obstacles)
        if ob is not None:
            b = self._bearing(ob[0], ob[1])
            obs[3], obs[4], obs[5] = np.sin(b), np.cos(b), np.clip(1.0 - (od - ob[2]) / 480.0, 0.0, 1.0)
        fd, fdist = self._nearest(self.food)
        if fd is not None:
            b = self._bearing(fd[0], fd[1])
            obs[7], obs[8], obs[9] = np.sin(b), np.cos(b), fd[3] * (1.0 - np.clip(fdist / 480.0, 0.0, 1.0))
        obs[6] = self.energy / 100.0
        obs[10] = self.thermal_value()
        obs[11] = np.clip(self.speed_now / max(1e-6, c.speed), 0.0, 1.0)
        obs[12] = self.touch
        phase = 2 * np.pi * (self.t % 400) / 400.0
        obs[13], obs[14] = np.sin(phase), np.cos(phase)
        obs[15], obs[16] = 1.0 - min(1.0, self.pos[0] / 200.0), 1.0 - min(1.0, (c.width - self.pos[0]) / 200.0)
        obs[17], obs[18] = 1.0 - min(1.0, self.pos[1] / 200.0), 1.0 - min(1.0, (c.height - self.pos[1]) / 200.0)
        obs[21] = float(self.rule)
        obs[22] = self.waypoint_index / max(1, c.waypoints)
        obs[23] = 1.0 if self.cue_visible() else 0.0
        obs[24] = np.clip(self.disturbance / max(1e-6, c.speed), 0.0, 1.0)
        if self.others:
            ot, odist = self._nearest([list(o) for o in self.others])
            b = self._bearing(ot[0], ot[1])
            obs[25], obs[26], obs[27] = np.sin(b), np.cos(b), np.clip(1.0 - odist / 480.0, 0.0, 1.0)
        return obs

    def sensory_channels(self) -> Dict[str, float]:
        """The sixteen sensory channels, derived from the ecology (visual / audio are scene proxies)."""
        c = self.config
        ob, od = self._nearest(self.obstacles)
        fd, fdist = self._nearest(self.food)
        prox = float(np.clip(1.0 - od / 480.0, 0.0, 1.0)) if ob is not None else 0.0
        food = float(fd[3] * (1.0 - np.clip(fdist / 480.0, 0.0, 1.0))) if fd is not None else 0.0
        thermal = self.thermal_value()
        near = sum(1 for o in self.obstacles if np.hypot(o[0] - self.pos[0], o[1] - self.pos[1]) < 240.0)
        values = [
            0.5 + 0.4 * np.sin(2 * np.pi * (self.t % 400) / 400.0),                 # visual_luminance (day cycle)
            float(np.clip(food - prox, -1.0, 1.0)),                                # visual_red_green (food vs obstacle)
            float(np.clip(2.0 * (thermal - 0.5), -1.0, 1.0)),                      # visual_blue_yellow (cool vs warm zone)
            near / max(1, len(self.obstacles)),                                    # visual_edge_density
            float(np.clip(self.speed_now / max(1e-6, c.speed) * 0.7 + 0.3 * prox * float(c.moving_obstacles), 0.0, 1.0)),
            prox,                                                                  # audio_low (rumble of the nearest obstacle)
            float(np.clip(np.linalg.norm(self.wind_vec) / 6.0 + self.disturbance / max(1e-6, c.speed), 0.0, 1.0)),
            float(np.clip(self.ping, 0.0, 1.0)),                                   # audio_high (collision ping, decays)
            float(np.clip(food, 0.0, 1.0)),                                        # olfactory
            float(np.clip(food * (1.0 if self.energy < 55 else 0.35), 0.0, 1.0)),  # gustatory
            thermal,                                                               # thermosensation
            float(np.clip(1.0 - self.energy / 100.0, 0.0, 1.0)),                   # interoception
            float(np.clip(self.speed_now / max(1e-6, c.speed), 0.0, 1.0)),         # proprioception
            float(self.touch),                                                     # mechanosensation
            prox,                                                                  # environment_state
            float(self.rule) if c.rule_shift_period > 0 else (1.0 if self.cue_visible() else 0.0),   # task_context
        ]
        return {name: float(v) for name, v in zip(SENSORY_CHANNELS, values)}

    # ------------------------------------------------------------------- step
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        c = self.config
        action = np.nan_to_num(np.asarray(action, dtype=np.float64).ravel(), nan=0.0, posinf=1.0, neginf=-1.0)
        drive, turn = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1)) if action.size > 1 else 0.0
        self.t += 1
        self.goal_age += 1
        scale = c.obstacle_speed * (1.0 + 0.6 * np.sin(self.t / 90.0) if c.dynamic else 1.0)
        if self.owns_world and c.moving_obstacles:
            for o in self.obstacles:
                o[1] += np.sin(self.t * 0.018 + o[3]) * 0.38 * 3.0 * scale
                o[0] += np.cos(self.t * 0.012 + o[3]) * 0.18 * 3.0 * scale
        if c.rule_shift_period > 0 and self.t % c.rule_shift_period == 0:
            self.rule = 1 - self.rule
        start = self.pos.copy()
        prev_dist = float(np.hypot(*(self.goal - self.pos)))
        self.heading = _wrap(self.heading + turn * c.turn_rate)
        speed = max(0.0, drive) * c.speed
        predicted = speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        actual = predicted.copy()
        if c.actuator_slip > 0 and self.rng.random() < c.actuator_slip:
            angle = self.heading + self.rng.normal(0.0, 0.5)
            actual = speed * self.rng.uniform(0.2, 0.6) * np.array([np.cos(angle), np.sin(angle)])
        if c.wind > 0:
            actual = actual + self.wind_vec * (1.0 + 0.5 * np.sin(self.t / 17.0)) + self.rng.normal(0.0, 0.15 * c.wind, size=2)
        self.pos = np.clip(self.pos + actual, [28.0, 28.0], [c.width - 28.0, c.height - 28.0])
        self.energy = float(np.clip(self.energy - c.energy_decay * (1.0 + max(0.0, drive)), 0.0, 100.0))
        dist = float(np.hypot(*(self.goal - self.pos)))
        sign = -1.0 if self.rule == 1 else 1.0
        reward = -0.02 + sign * 0.5 * (prev_dist - dist) / max(1e-6, c.speed)
        self.touch, self.ping = 0.0, self.ping * 0.7
        for o in self.obstacles + [[x, y, 14.0, 0.0] for x, y in self.others]:
            if np.hypot(o[0] - self.pos[0], o[1] - self.pos[1]) < o[2] + 18.0:
                reward, self.touch, self.ping = -1.0, 1.0, 1.0
                self.collisions += 1
                self.pos = np.clip(self.pos - 1.5 * actual, [28.0, 28.0], [c.width - 28.0, c.height - 28.0])
                break
        success = False
        if dist < c.goal_radius:
            self.score += 1
            reward = -1.0 if self.rule == 1 else 1.0
            self.energy = float(np.clip(self.energy + 11.0, 0.0, 100.0))
            if self.route:
                self.waypoint_index += 1
                success = self.waypoint_index >= len(self.route)
                self.goal = self.route[min(self.waypoint_index, len(self.route) - 1)]
            else:
                success = True
                self.goal = self._new_goal()
            self.goal_age = 0
        for f in list(self.food):
            if np.hypot(f[0] - self.pos[0], f[1] - self.pos[1]) < f[2] + 13.0:
                self.energy = float(np.clip(self.energy + f[3] * 5.0, 0.0, 100.0))
                reward = max(reward, 0.25 * f[3] * (4.0 if self.rule == 1 else 1.0))
                self.food_eaten += 1
                if c.replenish_food:
                    f[0], f[1], f[3] = *self._food_position(), float(0.35 + 0.65 * self.rng.random())
                else:
                    self.food.remove(f)
        if c.dynamic and c.replenish_food and self.t % 60 == 0 and len(self.food) < 2 * c.n_food:
            self.food.append([*self._food_position(), 12.0, float(0.35 + 0.65 * self.rng.random())])
        if self.thermal_value() > 0.8:
            reward -= 0.01
        if self.energy <= 1.0:
            self.energy, self.respawns, reward = 100.0, self.respawns + 1, -0.2
            self.pos = np.array([200.0 * c.width / 1200.0, c.height / 2.0])
        observed = self.pos - start
        self.speed_now = float(np.linalg.norm(observed))
        self.disturbance = float(np.linalg.norm(observed - predicted))
        self.reward = float(reward)
        self.path = (self.path + [(float(self.pos[0]), float(self.pos[1]))])[-90:]
        done = self.t >= self.max_steps or (bool(self.route) and self.waypoint_index >= len(self.route))
        info = {"success": bool(success or (self.route and self.waypoint_index >= len(self.route))), "dist": dist,
                "energy": self.energy, "collisions": self.collisions, "score": self.score, "food_eaten": self.food_eaten,
                "respawns": self.respawns, "rule": self.rule, "disturbance": self.disturbance,
                "predicted_outcome": predicted / max(1e-6, c.speed), "observed_outcome": observed / max(1e-6, c.speed)}
        return self._observe(), float(reward), bool(done), info

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        keys = ("pos", "heading", "t", "energy", "score", "collisions", "food_eaten", "respawns", "reward", "speed_now", "touch",
                "ping", "disturbance", "rule", "waypoint_index", "goal_age", "goal", "obstacles", "food", "thermal", "route",
                "clusters", "wind_vec", "path")
        state = {k: copy.deepcopy(getattr(self, k)) for k in keys}
        state["rng"] = copy.deepcopy(self.rng.bit_generator.state)
        state["config"] = self.config.to_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key, value in state.items():
            if key == "rng":
                self.rng.bit_generator.state = copy.deepcopy(value)
            elif key == "config":
                self.config = EcologyConfig(**value)
            else:
                setattr(self, key, copy.deepcopy(value))

    def render_state(self) -> Dict[str, Any]:
        """JSON-friendly scene description for dashboards."""
        c = self.config
        return {"name": c.name, "width": c.width, "height": c.height, "step": self.t,
                "agent": {"x": float(self.pos[0]), "y": float(self.pos[1]), "heading": self.heading, "energy": self.energy},
                "goal": {"x": float(self.goal[0]), "y": float(self.goal[1]), "radius": c.goal_radius, "visible": self.cue_visible()},
                "obstacles": [{"x": float(o[0]), "y": float(o[1]), "r": float(o[2])} for o in self.obstacles],
                "food": [{"x": float(f[0]), "y": float(f[1]), "energy": float(f[3])} for f in self.food],
                "thermal": [{"x": x, "y": y, "r": r, "value": v} for x, y, r, v in self.thermal],
                "others": [{"x": float(x), "y": float(y)} for x, y in self.others], "path": list(self.path),
                "stats": {"score": self.score, "collisions": self.collisions, "food_eaten": self.food_eaten,
                          "respawns": self.respawns, "reward": self.reward, "rule": self.rule, "disturbance": self.disturbance}}


__all__ = ["EcologyConfig", "EcologyArena"]
