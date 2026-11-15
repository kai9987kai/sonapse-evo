"""Builders and the agent loop for NeuroWeave substrates.

``run_weave_episode`` is a superset of the existing
``evaluation.benchmark_runner.run_benchmark_episode``: on a legacy substrate
(or a ``WeaveSubstrate`` with nothing enabled) it performs exactly the same
operations and returns the same value.  With mechanisms enabled it additionally

* feeds extra sensory channels from the environment into the sensory hub,
* delivers lagged outcome feedback (reward, intended / issued action, observed
  outcome) so learning, neuromodulation and the self-model have real signals,
* applies the exploration noise and action persistence requested by the
  affective control state (both are exactly 0 unless a mechanism sets them).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..substrate.recurrent_runtime import RecurrentSubstrate
from ..substrate.topology_state import TopologyState
from .contracts import Provenance
from .registry import WeaveConfig
from .substrate import WeaveSubstrate


def build_weave_substrate(
    total_neurons: int = 768,
    num_regions: int = 3,
    seed: int = 42,
    profile: str = "legacy",
    enable: Iterable[str] = (),
    disable: Iterable[str] = (),
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    topology: Optional[TopologyState] = None,
    weave: Optional[WeaveConfig] = None,
    strict: bool = False,
    topology_kind: Optional[str] = None,
    topology_options: Optional[Dict[str, Any]] = None,
    **substrate_kwargs: Any,
) -> WeaveSubstrate:
    """Create a :class:`WeaveSubstrate`.  Defaults reproduce the 768-neuron, 3-region legacy substrate.

    ``topology_kind`` selects one of ``connectome.TOPOLOGY_KINDS`` (e.g. ``"multi_region"`` for the ten-region
    connectome); leaving it ``None`` keeps the original default graph.
    """
    if topology is None and topology_kind is not None and topology_kind != "legacy":
        from .connectome import build_topology

        topology = build_topology(topology_kind, total_neurons=total_neurons, seed=seed, **(topology_options or {}))
    if topology is None:
        topology = TopologyState(total_neurons=total_neurons, num_regions=num_regions, seed=seed)
    if weave is None:
        weave = WeaveConfig.from_profile(profile, seed=seed, strict=strict)
    else:
        weave = weave.copy()
    enable, disable = tuple(enable), tuple(disable)
    if enable:
        weave.enable(*enable)
    if disable:
        weave.disable(*disable)
    for mid, values in (params or {}).items():
        weave.set_params(mid, **values)
    substrate_kwargs.setdefault("seed", seed)
    return WeaveSubstrate(topology=topology, weave=weave, **substrate_kwargs)


def upgrade_substrate(substrate: RecurrentSubstrate, weave: Optional[WeaveConfig] = None) -> WeaveSubstrate:
    """Non-destructively wrap an existing substrate's weights and state in a ``WeaveSubstrate``.

    The source substrate is left untouched; the result starts from identical buffers and rates.
    """
    upgraded = WeaveSubstrate(
        topology=substrate.topology, sensory_dim=substrate.sensory_dim, leak=substrate.leak,
        microsteps=substrate.microsteps, current_cap=substrate.current_cap, seed=substrate.seed, weave=weave,
    )
    with torch.no_grad():
        for name in ("base_pre", "base_post", "base_w", "lesion_mask", "B", "bias", "r"):
            getattr(upgraded, name).copy_(getattr(substrate, name).detach())
        upgraded.plastic_pre = substrate.plastic_pre.detach().clone()
        upgraded.plastic_post = substrate.plastic_post.detach().clone()
        upgraded.plastic_w = substrate.plastic_w.detach().clone()
    upgraded.step_count = substrate.step_count
    upgraded.active_grafts = dict(substrate.active_grafts)
    upgraded._sync_synapses()
    return upgraded


def canonical_motor_head(width: int = 64, actions: int = 5) -> nn.Linear:
    """The fixed drive / turn read-out used by the existing foraging benchmarks."""
    head = nn.Linear(width, actions, bias=False)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 2.0
        head.weight[1, 1] = 2.0
        head.weight[1, 2] = -2.0
    return head


@dataclass
class EpisodeResult:
    total_return: float
    steps: int
    rewards: List[float] = field(default_factory=list)
    success: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.total_return)


def _ingest_sensors(env: Any, substrate: WeaveSubstrate) -> None:
    hub = substrate.mechanisms.get("multimodal_sensory")
    reader = getattr(env, "sensory_channels", None)
    if hub is None or reader is None:
        return
    channels = reader()
    provenance = getattr(env, "sensory_provenance", Provenance.SYNTHETIC_SIMULATION)
    hub.ingest(channels, provenance=provenance)


def run_weave_episode(
    env: Any,
    substrate: RecurrentSubstrate,
    motor_head: nn.Module,
    registry: Optional[Any] = None,
    seed: int = 42,
    enable_grafts: bool = True,
    motor_offset: Optional[int] = None,
    readout_width: int = 64,
    max_steps: Optional[int] = None,
    on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    collect_telemetry: bool = False,
) -> EpisodeResult:
    """Run one episode.  Legacy-equivalent when no mechanism is enabled."""
    obs = env.reset(seed=seed)
    substrate.reset_state()
    is_weave = isinstance(substrate, WeaveSubstrate) and substrate.weave_active
    if isinstance(substrate, WeaveSubstrate):            # every environment here is a simulation unless it says otherwise
        substrate.observation_provenance = Provenance(getattr(env, "sensory_provenance", Provenance.SYNTHETIC_SIMULATION))
    if motor_offset is None:
        motor_offset = int(getattr(substrate, "motor_offset", 512))
    readout_start = motor_offset if motor_offset + readout_width <= substrate.total_neurons else 0
    noise_rng = np.random.default_rng(int(seed) + 90001)
    prev_action: Optional[np.ndarray] = None
    rewards: List[float] = []
    total_reward, steps, done, info = 0.0, 0, False, {}

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        curr_r = substrate.r
        aux_current = None
        if enable_grafts and registry is not None:
            aux_current = registry.compute_total_current(curr_r.unsqueeze(0)).squeeze(0)
        if is_weave:
            _ingest_sensors(env, substrate)
        new_r = substrate.forward_step(obs_t, aux_current=aux_current)
        with torch.no_grad():
            action = torch.tanh(motor_head(new_r[readout_start:readout_start + readout_width])).numpy()
        intended = action
        if is_weave:
            signals = substrate.signals
            if signals.exploration > 0.0:        # curiosity-driven exploration (0 unless a mechanism sets it)
                action = np.clip(action + signals.exploration * noise_rng.standard_normal(action.shape), -1.0, 1.0).astype(np.float32)
            if signals.persistence > 0.0 and prev_action is not None:   # motivation-driven persistence
                action = (signals.persistence * prev_action + (1.0 - signals.persistence) * action).astype(np.float32)
            prev_action = action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rewards.append(float(reward))
        steps += 1
        if is_weave:
            substrate.deliver_feedback(
                reward=float(reward), done=bool(done), info=info, intended_action=intended, issued_action=action,
                predicted_outcome=info.get("predicted_outcome") if isinstance(info, dict) else None,
                observed_outcome=info.get("observed_outcome") if isinstance(info, dict) else None,
            )
        if on_step is not None:
            on_step(steps, {"reward": float(reward), "action": action, "info": info})
        if max_steps is not None and steps >= max_steps:
            break

    telemetry = substrate.weave_telemetry() if (collect_telemetry and isinstance(substrate, WeaveSubstrate)) else {}
    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    return EpisodeResult(total_return=float(total_reward), steps=steps, rewards=rewards, success=success,
                         info=dict(info) if isinstance(info, dict) else {}, telemetry=telemetry)


__all__ = ["build_weave_substrate", "upgrade_substrate", "canonical_motor_head", "EpisodeResult", "run_weave_episode"]
