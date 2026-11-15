"""``WeaveSubstrate`` - the additive NeuroWeave runtime.

``WeaveSubstrate`` *is a* :class:`~sonapse_evo.substrate.recurrent_runtime.RecurrentSubstrate`
(it subclasses it and never edits it), so lesions, grafts, the transaction
manager, criticality and causal diagnostics, matched controls and every
existing benchmark accept it unchanged.

Regression contract
-------------------
* No mechanism enabled (the default)  ->  every overridden method delegates to
  the parent implementation.  Output is bit-identical to SONAPSE-EVO 2.0.
* Mechanisms enabled                  ->  the extended path runs.  Its neutral
  form (all context fields at their initial values) evaluates the *same
  expression in the same order* as the parent, so a mechanism whose learned
  parameters are still zero changes nothing.
* Learned weave state (weight deltas, grown synapses, compartment state) is
  owned by this class and is inert whenever the weave path is not active; it
  never leaks into the legacy buffers ``base_w`` / ``plastic_w``.
"""
from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..contracts.snapshots import SubstrateSnapshot
from ..substrate.recurrent_runtime import RecurrentSubstrate
from ..substrate.topology_state import TopologyState
from .cells import CellParams, WeaveCell
from .contracts import (
    INTERFACE_VERSIONS, WEAVE_VERSION, FeedbackSignal, Mechanism, Provenance, SharedSignals, SignalKind, StepContext,
)
from .events import EventBus
from .registry import MECHANISMS, WeaveConfig

_FLOAT_EDGE_FIELDS = (
    "w_delta", "alive", "trace", "eligibility", "pre_trace", "post_trace", "utility",
    "routing_gain", "myelin", "lr_scale", "consolidation", "last_signal", "sign",
)
_LONG_EDGE_FIELDS = ("delay", "branch", "age")
EDGE_FIELDS = _FLOAT_EDGE_FIELDS + _LONG_EDGE_FIELDS
MAX_DELAY = 3
_ROUTING_REST = math.tanh(1.0 - 0.12)


def neuron_sign(index: torch.Tensor) -> torch.Tensor:
    """Dale-style source sign convention shared with the existing code base
    (``GraphTopology`` / ``NeuromodulatedPlasticityEngine``): every fifth neuron is inhibitory."""
    return torch.where(index % 5 == 0, -1.0, 1.0).to(torch.float32)


class SynapseState:
    """Per-edge state aligned with the unified edge list ``[base | legacy plastic | grown]``."""

    def __init__(self, n_base: int, n_plastic: int, seed: int = 42, device=None):
        self.device = device or torch.device("cpu")
        self.seed = int(seed)
        self.n_base = int(n_base)
        self.n_plastic = int(n_plastic)
        self.grown_pre = torch.zeros(0, dtype=torch.long, device=self.device)
        self.grown_post = torch.zeros(0, dtype=torch.long, device=self.device)
        self.grown_w = torch.zeros(0, dtype=torch.float32, device=self.device)
        e = self.n_base + self.n_plastic
        for name in _FLOAT_EDGE_FIELDS:
            setattr(self, name, torch.zeros(e, dtype=torch.float32, device=self.device))
        for name in _LONG_EDGE_FIELDS:
            setattr(self, name, torch.zeros(e, dtype=torch.long, device=self.device))
        self.alive.fill_(1.0)
        self.routing_gain.fill_(1.0)
        self.lr_scale.fill_(1.0)
        self.sign.fill_(1.0)
        rng = np.random.default_rng(self.seed + 104729)
        self.myelin = torch.as_tensor(0.5 + 0.4 * rng.random(e), dtype=torch.float32, device=self.device)

    @property
    def n_grown(self) -> int:
        return int(self.grown_w.numel())

    @property
    def n_edges(self) -> int:
        return self.n_base + self.n_plastic + self.n_grown

    def append(self, pre: torch.Tensor, post: torch.Tensor, w: torch.Tensor, delay: Optional[torch.Tensor] = None,
               branch: Optional[torch.Tensor] = None) -> None:
        k = int(pre.numel())
        if k == 0:
            return
        self.grown_pre = torch.cat([self.grown_pre, pre.to(torch.long)])
        self.grown_post = torch.cat([self.grown_post, post.to(torch.long)])
        self.grown_w = torch.cat([self.grown_w, w.to(torch.float32)])
        fresh = {
            "alive": torch.ones(k), "routing_gain": torch.ones(k), "lr_scale": torch.ones(k),
            "myelin": torch.full((k,), 0.5), "sign": torch.sign(w.to(torch.float32)),
        }
        for name in _FLOAT_EDGE_FIELDS:
            add = fresh.get(name, torch.zeros(k)).to(torch.float32).to(self.device)
            setattr(self, name, torch.cat([getattr(self, name), add]))
        for name in _LONG_EDGE_FIELDS:
            add = torch.zeros(k, dtype=torch.long, device=self.device)
            if name == "delay" and delay is not None:
                add = delay.to(torch.long)
            if name == "branch" and branch is not None:
                add = branch.to(torch.long)
            setattr(self, name, torch.cat([getattr(self, name), add]))

    def compact_grown(self) -> int:
        """Physically remove grown edges whose ``alive`` flag is 0.  Returns the number removed."""
        fixed = self.n_base + self.n_plastic
        keep_grown = self.alive[fixed:] > 0
        removed = int((~keep_grown).sum().item())
        if removed == 0:
            return 0
        keep = torch.cat([torch.ones(fixed, dtype=torch.bool, device=self.device), keep_grown])
        for name in EDGE_FIELDS:
            setattr(self, name, getattr(self, name)[keep])
        self.grown_pre, self.grown_post, self.grown_w = self.grown_pre[keep_grown], self.grown_post[keep_grown], self.grown_w[keep_grown]
        return removed

    def reset_transient(self) -> None:
        for name in ("pre_trace", "post_trace", "last_signal"):
            getattr(self, name).zero_()

    def state_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {name: getattr(self, name).detach().clone() for name in EDGE_FIELDS}
        out.update(grown_pre=self.grown_pre.clone(), grown_post=self.grown_post.clone(), grown_w=self.grown_w.clone(),
                   n_base=self.n_base, n_plastic=self.n_plastic)
        return out

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("n_base", self.n_base)) != self.n_base or int(state.get("n_plastic", self.n_plastic)) != self.n_plastic:
            raise ValueError("synapse state was saved for a different base/plastic edge count")
        self.grown_pre = torch.as_tensor(state["grown_pre"], dtype=torch.long, device=self.device).clone()
        self.grown_post = torch.as_tensor(state["grown_post"], dtype=torch.long, device=self.device).clone()
        self.grown_w = torch.as_tensor(state["grown_w"], dtype=torch.float32, device=self.device).clone()
        e = self.n_edges
        for name in EDGE_FIELDS:
            dtype = torch.long if name in _LONG_EDGE_FIELDS else torch.float32
            value = torch.as_tensor(state[name], dtype=dtype, device=self.device)
            if value.shape != (e,):
                raise ValueError(f"synapse field '{name}' has length {value.numel()}, expected {e}")
            setattr(self, name, value.clone())


class WeaveSubstrate(RecurrentSubstrate):
    """Recurrent connectome substrate with switchable NeuroWeave mechanisms."""

    def __init__(
        self,
        topology: TopologyState,
        sensory_dim: int = 155,
        leak: float = 0.35,
        microsteps: int = 2,
        current_cap: float = 0.10,
        seed: int = 42,
        weave: Optional[WeaveConfig] = None,
        motor_offset: Optional[int] = None,
    ):
        super().__init__(topology=topology, sensory_dim=sensory_dim, leak=leak, microsteps=microsteps,
                         current_cap=current_cap, seed=seed)
        self.weave: WeaveConfig = weave.copy() if weave is not None else WeaveConfig.legacy(seed=seed)
        self.learning_enabled: bool = bool(self.weave.learning_enabled)
        self.signals = SharedSignals()
        self.events = EventBus()
        self.mechanisms: Dict[str, Mechanism] = {}
        self._ordered: List[Mechanism] = []
        self.weave_faults: List[Dict[str, Any]] = []
        self.node_mask: Optional[torch.Tensor] = None
        self.layout = getattr(topology, "layout", None)          # RegionLayout when built by connectome.build_topology
        #: where the observations fed to ``forward_step`` come from (the agent loop sets it from the environment)
        self.observation_provenance: Provenance = Provenance.DIRECT_SENSORY
        if motor_offset is not None:
            self.motor_offset: int = int(motor_offset)
        elif self.layout is not None:
            self.motor_offset = int(self.layout.motor_offset)
        else:
            self.motor_offset = 512 if self.total_neurons >= 576 else 0
        # an owned copy: neurogenesis relabels neurons and must never write into the shared TopologyState
        self.region_ids: torch.Tensor = torch.tensor(np.array(topology.region_assignments, copy=True), dtype=torch.long)
        self.weight_transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        self.traversal: Optional[Callable[..., torch.Tensor]] = None
        self.w_max: float = 1.5
        self.last_sparse_count: int = self.total_neurons
        self.weave_steps: int = 0
        self.op_count: int = 0
        self._ctx: Optional[StepContext] = None
        self._history: List[torch.Tensor] = []
        self._w_cache: Optional[torch.Tensor] = None       # effective weights, valid only inside one step
        self._wt_cache: Optional[torch.Tensor] = None      # ... after weight transforms
        self._cache_enabled: bool = False
        self.cell = WeaveCell(self.total_neurons, seed=self.seed, params=CellParams.from_dict(self.weave.params.get("cell")))
        self.syn = SynapseState(len(self.base_w), len(self.plastic_w), seed=self.seed)
        self._init_edge_attributes()
        self._build_mechanisms()

    # ------------------------------------------------------------------ setup
    @property
    def weave_active(self) -> bool:
        return self.weave.any_enabled() or self.node_mask is not None

    def _init_edge_attributes(self) -> None:
        """Seeded per-edge delay, axonal branch and anchor sign."""
        pre, post = self.edge_index()
        inter = self.region_ids[pre] != self.region_ids[post]
        rng = np.random.default_rng(self.seed + 15485863)
        delay = rng.choice(3, size=int(pre.numel()), p=[0.5, 0.3, 0.2])
        delay = torch.as_tensor(delay, dtype=torch.long) + inter.to(torch.long)      # long-range edges conduct slower
        self.syn.delay = torch.clamp(delay, 0, MAX_DELAY)
        n_branches = self.cell.branches.shape[0]
        self.syn.branch = torch.where(inter, torch.ones_like(delay), torch.zeros_like(delay)) % n_branches  # 0 local, 1 long-range
        anchor = self.anchor_weights().detach()
        sign = torch.sign(anchor)
        self.syn.sign = torch.where(sign == 0, neuron_sign(pre), sign)

    def _build_mechanisms(self) -> None:
        self.mechanisms, self._ordered = {}, []
        for mid in self.weave.enabled_ids():
            self._attach(mid)
        self._ordered.sort(key=lambda m: (int(m.stage), list(MECHANISMS).index(m.mechanism_id)))

    def _attach(self, mechanism_id: str) -> None:
        spec = MECHANISMS[mechanism_id]
        if not spec.is_plugin or mechanism_id in self.mechanisms:
            return
        try:
            cls = spec.load_class()
            mech = cls(self, self.weave.params_for(mechanism_id))
            mech.mechanism_id = mechanism_id
            self.mechanisms[mechanism_id] = mech
            self._ordered.append(mech)
            mech.on_attach()
        except Exception as exc:
            self.mechanisms.pop(mechanism_id, None)
            self._ordered = [m for m in self._ordered if m.mechanism_id != mechanism_id]
            if self.weave.strict:
                raise
            self._record_fault(mechanism_id, "attach", exc)

    def _detach_faulty(self, mech: Mechanism) -> None:
        self.mechanisms.pop(mech.mechanism_id, None)
        self._ordered = [m for m in self._ordered if m is not mech]
        try:
            mech.on_detach()
        except Exception:
            pass
        self.weave.enabled[mech.mechanism_id] = False          # the config always describes what actually runs
        self._invalidate_weights()

    def _record_fault(self, mechanism_id: str, hook: str, exc: BaseException) -> None:
        self.weave.enabled[mechanism_id] = False
        fault = {"mechanism": mechanism_id, "hook": hook, "error": f"{type(exc).__name__}: {exc}", "step": int(self.step_count)}
        self.weave_faults.append(fault)
        self.events.publish("weave.fault", SignalKind.TELEMETRY, self.step_count, fault, source=mechanism_id)
        warnings.warn(f"NeuroWeave mechanism '{mechanism_id}' failed in {hook} and was detached: {fault['error']}", RuntimeWarning, stacklevel=3)

    def set_mechanism(self, mechanism_id: str, enabled: bool, **params: Any) -> None:
        """Switch one mechanism at runtime (independently switchable experiments)."""
        if mechanism_id not in MECHANISMS:
            raise KeyError(f"Unknown mechanism id: {mechanism_id}")
        if params:
            self.weave.set_params(mechanism_id, **params)
            attached = self.mechanisms.pop(mechanism_id, None)
            if attached is not None:                         # new parameters: re-create the mechanism (its state restarts)
                self._ordered = [m for m in self._ordered if m is not attached]
                attached.on_detach()
        if enabled:
            self.weave.enable(mechanism_id)
            self._attach(mechanism_id)
        else:
            self.weave.disable(mechanism_id)
            mech = self.mechanisms.pop(mechanism_id, None)
            if mech is not None:
                self._ordered = [m for m in self._ordered if m is not mech]
                mech.on_detach()
        self._ordered.sort(key=lambda m: (int(m.stage), list(MECHANISMS).index(m.mechanism_id)))

    def freeze_learning(self, frozen: bool = True) -> None:
        """Keep every mechanism's effect but stop all learned-state updates (frozen inference)."""
        self.learning_enabled = not frozen

    # ------------------------------------------------------------ edge access
    def _sync_synapses(self) -> None:
        if self.syn.n_base != len(self.base_w) or self.syn.n_plastic != len(self.plastic_w):
            old = self.syn
            self.syn = SynapseState(len(self.base_w), len(self.plastic_w), seed=self.seed)
            self._init_edge_attributes()
            o_fixed = old.n_base + old.n_plastic
            self.syn.append(old.grown_pre, old.grown_post, old.grown_w, delay=old.delay[o_fixed:], branch=old.branch[o_fixed:])
            nb, k1, k2 = self.syn.n_base, min(old.n_base, self.syn.n_base), min(old.n_plastic, self.syn.n_plastic)
            for name in EDGE_FIELDS:                          # carry every learned per-edge value that still has an edge
                new_field, old_field = getattr(self.syn, name), getattr(old, name)
                new_field[:k1] = old_field[:k1]
                new_field[nb:nb + k2] = old_field[old.n_base:old.n_base + k2]
                if old.n_grown:
                    new_field[nb + self.syn.n_plastic:] = old_field[o_fixed:]
            self._refresh_signs()
            self.notify_topology_change()

    def _refresh_signs(self) -> None:
        """Synapse sign follows the current anchor weight (existing code such as RigL edits ``base_w`` in place)."""
        anchor = self.anchor_weights().detach()
        if anchor.numel() == self.syn.sign.numel():
            self.syn.sign = torch.where(anchor != 0, torch.sign(anchor), self.syn.sign)

    def notify_topology_change(self) -> None:
        """Edges were added, removed or reloaded: drop cached weights and let every mechanism rebuild its tables."""
        self._invalidate_weights()
        for mech in list(self._ordered):
            mech.on_topology_change()

    def edge_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified ``(pre, post)`` over base, legacy-plastic and grown edges."""
        if len(self.plastic_pre) == 0 and self.syn.n_grown == 0:
            return self.base_pre, self.base_post
        return (torch.cat([self.base_pre, self.plastic_pre, self.syn.grown_pre]),
                torch.cat([self.base_post, self.plastic_post, self.syn.grown_post]))

    def anchor_weights(self) -> torch.Tensor:
        """Weights before weave learning: ``[base_w | plastic_w | grown_w]``."""
        if len(self.plastic_w) == 0 and self.syn.n_grown == 0:
            return self.base_w
        return torch.cat([self.base_w, self.plastic_w, self.syn.grown_w])

    def edge_mask(self) -> torch.Tensor:
        """Lesion mask extended over all edges, times the weave ``alive`` flag."""
        extra = self.syn.n_edges - len(self.lesion_mask)
        mask = self.lesion_mask if extra == 0 else torch.cat([self.lesion_mask, torch.ones(extra, dtype=torch.float32)])
        return mask * self.syn.alive

    def effective_weights(self) -> torch.Tensor:
        """``(anchor + learned delta) * lesion_mask * alive`` - what transmission uses."""
        if self._w_cache is not None:
            return self._w_cache
        self._sync_synapses()
        w = (self.anchor_weights() + self.syn.w_delta) * self.edge_mask()
        if self._cache_enabled:
            self._w_cache = w
        return w

    def _invalidate_weights(self) -> None:
        self._w_cache, self._wt_cache = None, None

    def _transmission_weights(self) -> torch.Tensor:
        if self._wt_cache is not None:
            return self._wt_cache
        w = self.effective_weights()
        for transform in self.weight_transforms.values():
            w = transform(w)
        if self._cache_enabled:
            self._wt_cache = w
        return w

    def apply_weight_update(self, dw: torch.Tensor, index: Optional[torch.Tensor] = None,
                            w_min: float = 0.0, w_max: Optional[float] = None, respect_lr_scale: bool = True) -> torch.Tensor:
        """Single shared, bounded, sign-preserving weight update used by all plasticity rules.

        ``dw`` is a signed change of synaptic *magnitude*-carrying weight.  The
        result keeps each edge's anchor sign (Dale-style) and ``|w| in [w_min, w_max]``.
        Returns the change actually applied.
        """
        if not self.learning_enabled:
            return torch.zeros_like(dw)
        self._invalidate_weights()
        self._sync_synapses()
        self._refresh_signs()
        w_max = self.w_max if w_max is None else float(w_max)
        anchor = self.anchor_weights().detach()
        sign, delta, scale = self.syn.sign, self.syn.w_delta, self.syn.lr_scale
        if index is not None:
            anchor, sign, old_delta, scale = anchor[index], sign[index], delta[index], scale[index]
        else:
            old_delta = delta
        step = dw * scale if respect_lr_scale else dw
        current = anchor + old_delta
        upper = torch.clamp(anchor.abs(), min=w_max)          # never shrink a pre-existing larger anchor
        magnitude = torch.minimum(torch.clamp(sign * (current + step), min=w_min), upper)
        new_delta = sign * magnitude - anchor
        applied = new_delta - old_delta
        if index is not None:
            delta[index] = new_delta
        else:
            self.syn.w_delta = new_delta
        return applied

    def grow_edges(self, pre: torch.Tensor, post: torch.Tensor, w: torch.Tensor) -> int:
        """Add weave-owned synapses (never touches ``base_w`` / ``plastic_w``).  Skips self-loops and duplicates."""
        self._invalidate_weights()
        self._sync_synapses()
        cur_pre, cur_post = self.edge_index()
        n = self.total_neurons
        existing = set((cur_pre * n + cur_post).tolist())
        keep = [i for i, (u, v) in enumerate(zip(pre.tolist(), post.tolist()))
                if u != v and 0 <= u < n and 0 <= v < n and (u * n + v) not in existing and not existing.add(u * n + v)]
        if not keep:
            return 0
        idx = torch.as_tensor(keep, dtype=torch.long)
        p, q, ww = pre[idx].to(torch.long), post[idx].to(torch.long), w[idx].to(torch.float32)
        inter = self.region_ids[p] != self.region_ids[q]
        self.syn.append(p, q, ww, delay=inter.to(torch.long), branch=inter.to(torch.long) % self.cell.branches.shape[0])
        self.notify_topology_change()
        return len(keep)

    def prune_edges(self, index: torch.Tensor) -> int:
        """Mark edges dead (weave-level pruning; lesion masks are untouched).  Grown edges are compacted away."""
        self._invalidate_weights()
        self._sync_synapses()
        index = index.to(torch.long)
        before = int(self.syn.alive[index].sum().item())
        self.syn.alive[index] = 0.0
        self.syn.compact_grown()
        self.notify_topology_change()
        return before

    def set_node_lesions(self, node_ids) -> int:
        """Silence whole neurons (their rate is forced to 0).  ``None`` / empty clears the lesion."""
        ids = [int(i) for i in (node_ids or []) if 0 <= int(i) < self.total_neurons]
        if not ids:
            self.node_mask = None
            return 0
        mask = torch.ones(self.total_neurons, dtype=torch.float32)
        mask[torch.as_tensor(sorted(set(ids)), dtype=torch.long)] = 0.0
        self.node_mask = mask
        return int((mask == 0).sum().item())

    # ----------------------------------------------------------- transmission
    def _presynaptic(self, r: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        en = self.weave.is_enabled
        if en("synaptic_delays") and r.dim() == 1 and self._history:
            delay = self.syn.delay
            if en("adaptive_conduction"):
                delay = torch.clamp(delay - (self.cell.myelin[pre] > 0.78).to(torch.long), min=0)
            depth = min(len(self._history), MAX_DELAY)
            stack = torch.stack([r] + [self._history[-k] for k in range(1, depth + 1)])    # [now, t-1, t-2, ...]
            src = stack[torch.clamp(delay, max=depth), pre]
        else:
            src = r[..., pre]
        if en("axonal_branches") and r.dim() == 1:
            cell = self.cell
            mod = cell.branches[self.syn.branch, pre] - cell.hillock[pre]          # branch-specific transmission
            src = src * torch.clamp(1.0 + cell.p.branch_gain * mod, 0.7, 1.3)
        return src

    def _edge_gain(self, post: torch.Tensor, ctx: Optional[StepContext]) -> Optional[torch.Tensor]:
        en = self.weave.is_enabled
        gain = ctx.edge_gain if ctx is not None else None
        if en("context_adaptive_routing"):
            cell = self.cell
            raw = torch.tanh(cell.soma_gating[post].abs() + 0.3 * cell.apic[post].abs() - 0.12)
            routing = torch.clamp(1.0 + 0.24 * (raw - _ROUTING_REST), 0.62, 1.14)
            self.syn.routing_gain = routing.detach()
            gain = routing if gain is None else gain * routing
        if en("uncertainty_gating"):
            u_gain = min(1.0, max(0.58, 1.0 / (1.0 + 2.2 * max(0.0, float(self.signals.uncertainty)))))
            if u_gain != 1.0:
                gain = torch.full((int(post.numel()),), u_gain) if gain is None else gain * u_gain
        if en("astrocyte_regulation"):
            a_gain = torch.clamp(1.0 + self.cell.p.astro_syn * (self.cell.astro_gain()[post] - 1.0), 0.85, 1.15)
            gain = a_gain if gain is None else gain * a_gain
        return gain

    def _scatter(self, post: torch.Tensor, signal: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if self.traversal is not None and signal.dim() == 1:
            return self.traversal(post, signal, like)
        rec = torch.zeros_like(like)
        index = post if signal.dim() == 1 else post.expand_as(signal)
        m, p = self.syn.n_base, self.syn.n_plastic
        for lo, hi in ((0, m), (m, m + p), (m + p, signal.shape[-1])):      # same accumulation order as the parent
            if hi > lo:
                rec.scatter_add_(-1, index[..., lo:hi], signal[..., lo:hi])
        return rec

    def transmit(self, r: torch.Tensor, ctx: Optional[StepContext] = None):
        """Returns ``(recurrent_input, per-edge signal, pre, post)`` on the extended path."""
        pre, post = self.edge_index()
        w = self._transmission_weights()
        signal = self._presynaptic(r, pre) * w
        gain = self._edge_gain(post, ctx)
        if gain is not None:
            signal = signal * gain
        self.op_count += int(signal.shape[-1]) * (3 if gain is None else 5)
        return self._scatter(post, signal, r), signal, pre, post

    def compute_recurrent_input(self, r: torch.Tensor) -> torch.Tensor:
        if not self.weave_active:
            return super().compute_recurrent_input(r)
        return self.transmit(r, self._ctx)[0]

    # ----------------------------------------------------------- step pipeline
    def _run_hooks(self, hook: str, ctx: StepContext) -> None:
        order = list(self._ordered)
        if hook == "end_step":
            index = list(MECHANISMS)
            order.sort(key=lambda m: (int(m.end_stage if m.end_stage is not None else m.stage), index.index(m.mechanism_id)))
        for mech in order:
            try:
                getattr(mech, hook)(ctx)
            except Exception as exc:
                if self.weave.strict:
                    raise
                self._record_fault(mech.mechanism_id, hook, exc)
                self._detach_faulty(mech)

    def _open_context(self, observation: torch.Tensor) -> StepContext:
        self._sync_synapses()
        self._refresh_signs()
        signals = self.signals
        if signals.pending is not None:                    # lagged feedback becomes visible now
            signals.feedback, signals.pending = signals.pending, None
            signals.reward = float(signals.feedback.reward)
        else:
            signals.feedback = FeedbackSignal()
            signals.reward = 0.0
        # per-step control outputs are recomputed by the modulation stage every step
        signals.plasticity_gain, signals.attention_gain = 1.0, 1.0
        signals.exploration, signals.persistence = 0.0, 0.0
        ctx = StepContext(self, observation, self.step_count)
        self._ctx = ctx
        return ctx

    def _weave_microstep(self, observation: torch.Tensor, aux_current: Optional[torch.Tensor], ctx: StepContext) -> torch.Tensor:
        if observation.dim() != 1:
            raise ValueError("The NeuroWeave path integrates one observation vector at a time (got a batch).")
        sens_in = torch.matmul(self.B, observation)
        if aux_current is not None:
            clamped_aux = torch.clamp(aux_current, -self.current_cap, self.current_cap)
        else:
            clamped_aux = torch.zeros_like(self.r)
        ctx.r_prev = self.r
        rec_in, signal, pre, post = self.transmit(self.r, ctx)
        ctx.sens_in, ctx.rec_in, ctx.aux_in, ctx.bias, ctx.edge_signal = sens_in, rec_in, clamped_aux, self.bias, signal
        if ctx.context_weight is not None:
            ctx.ctx_in = self._scatter(post, signal * ctx.context_weight[pre], self.r)
        else:
            ctx.ctx_in = rec_in
        new_r = self.cell.integrate(self, ctx)
        self._history.append(self.r.detach())
        if len(self._history) > MAX_DELAY:
            self._history.pop(0)
        self.r = new_r
        with torch.no_grad():
            self.syn.last_signal = signal.detach()
        return self.r

    def microstep(self, observation: torch.Tensor, aux_current: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.weave_active:
            return super().microstep(observation, aux_current)
        ctx = self._ctx if self._ctx is not None else StepContext(self, observation, self.step_count)
        self._bound_context(ctx)
        return self._weave_microstep(observation, aux_current, ctx)

    def _bound_context(self, ctx: StepContext) -> None:
        cap = float(self.weave.weave_current_cap)
        ctx.apical_in = torch.clamp(ctx.apical_in, -cap, cap)
        ctx.extra_in = torch.clamp(ctx.extra_in, -cap, cap)
        ctx.sensory_extra = torch.clamp(ctx.sensory_extra, -cap, cap)

    def forward_step(self, observation, aux_current=None) -> torch.Tensor:
        if not self.weave_active:
            return super().forward_step(observation, aux_current)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.r.device)
        aux_tensor = (torch.as_tensor(aux_current, dtype=torch.float32, device=self.r.device)
                      if aux_current is not None else None)
        ctx = self._open_context(obs_tensor)
        self._invalidate_weights()
        self._cache_enabled = True              # weights are constant until the learning stage of this step
        self._run_hooks("begin_step", ctx)
        self._bound_context(ctx)
        for i in range(self.microsteps):
            ctx.microstep = i
            self._weave_microstep(obs_tensor, aux_tensor, ctx)
        self.step_count += 1
        self.weave_steps += 1
        self.signals.surprise = float(self.cell.pred_error.abs().mean().item())
        self._cache_enabled = False
        self._invalidate_weights()
        self._run_hooks("end_step", ctx)
        self._ctx = None
        return self.r

    def deliver_feedback(self, reward: float = 0.0, done: bool = False, info: Optional[Dict[str, Any]] = None,
                         intended_action=None, issued_action=None, predicted_outcome=None, observed_outcome=None) -> None:
        """Hand the outcome of the last action to the substrate.  Consumed on the *next* step (lagged).

        A non-finite reward is rejected: it would otherwise propagate into every learned weight.
        """
        if not np.isfinite(float(reward)):
            raise ValueError(f"reward must be finite, got {reward!r}")
        self.signals.pending = FeedbackSignal(
            reward=float(reward), done=bool(done), info=dict(info or {}), intended_action=intended_action,
            issued_action=issued_action, predicted_outcome=predicted_outcome, observed_outcome=observed_outcome,
        )

    def reset_state(self) -> None:
        super().reset_state()
        self.cell.reset()
        self.syn.reset_transient()
        self._history = []
        self._ctx = None
        self.signals.reset_transient()
        for mech in self._ordered:
            mech.reset()

    def estimate_jacobian_norm(self, observation: Optional[torch.Tensor] = None) -> float:
        if not self.weave_active:
            return super().estimate_jacobian_norm(observation)
        with torch.no_grad():      # linearised around the legacy activation, using the effective weave weights
            if observation is None:
                observation = torch.zeros(self.sensory_dim, dtype=torch.float32, device=self.r.device)
            pre, post = self.edge_index()
            w = self.effective_weights().detach()
            gain = self._edge_gain(post, None)
            if gain is not None:
                w = w * gain
            rec = torch.zeros_like(self.r).scatter_add_(0, post, self.r[pre] * w)
            phi_prime = self.d_phi(rec + torch.matmul(self.B, observation) + self.bias)
            v = torch.randn(self.total_neurons, dtype=torch.float32, device=self.r.device)
            v = v / (torch.norm(v) + 1e-8)
            norm_jv = torch.tensor(0.0)
            for _ in range(10):
                wv = torch.zeros_like(v).scatter_add_(0, post, v[pre] * w)
                jv = (1.0 - self.leak) * v + self.leak * (phi_prime * wv)
                norm_jv = torch.norm(jv)
                if norm_jv < 1e-12:
                    break
                v = jv / norm_jv
            self.last_jacobian_norm = float(norm_jv.item())
            return self.last_jacobian_norm

    # ------------------------------------------------------------ persistence
    def weave_state_dict(self) -> Dict[str, Any]:
        """Complete NeuroWeave state (config, compartments, synapses, signals, mechanisms)."""
        return {
            "weave_version": WEAVE_VERSION,
            "interfaces": dict(INTERFACE_VERSIONS),
            "config": self.weave.to_dict(),
            "learning_enabled": bool(self.learning_enabled),
            "cell": self.cell.state_dict(),
            "synapses": self.syn.state_dict(),
            "signals": self.signals.as_dict(),
            "feedback": self.signals.feedback_state(),
            "node_mask": None if self.node_mask is None else self.node_mask.clone(),
            "history": [h.clone() for h in self._history],
            "mechanisms": {mid: copy.deepcopy(mech.state_dict()) for mid, mech in self.mechanisms.items()},
            "counters": {"weave_steps": int(self.weave_steps), "op_count": int(self.op_count)},
            "observation_provenance": Provenance(self.observation_provenance).value,
        }

    def load_weave_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        config = WeaveConfig.from_dict(state["config"])
        live = {m.mechanism_id for m in self._ordered}
        wanted = {mid for mid in config.enabled_ids() if MECHANISMS[mid].is_plugin}
        if config.to_dict()["enabled"] != self.weave.to_dict()["enabled"] or config.params != self.weave.params or live != wanted:
            for mech in list(self._ordered):                     # detach what runs now, including transforms / traversal
                try:
                    mech.on_detach()
                except Exception:
                    pass
            self.weight_transforms.clear()
            self.traversal = None
            self.weave = config
            self.cell = WeaveCell(self.total_neurons, seed=self.seed, params=CellParams.from_dict(config.params.get("cell")))
            self._build_mechanisms()
        self.learning_enabled = bool(state.get("learning_enabled", True))
        self.cell.load_state_dict(state["cell"])
        self._sync_synapses()
        self.syn.load_state_dict(state["synapses"])
        self.signals.load(state.get("signals", {}))
        self.signals.load_feedback_state(state.get("feedback", {}))
        mask = state.get("node_mask")
        self.node_mask = None if mask is None else torch.as_tensor(mask, dtype=torch.float32).clone()
        self._history = [torch.as_tensor(h, dtype=torch.float32).clone() for h in state.get("history", [])]
        self.observation_provenance = Provenance(state.get("observation_provenance", Provenance.DIRECT_SENSORY.value))
        counters = state.get("counters", {})
        self.weave_steps = int(counters.get("weave_steps", 0))
        self.op_count = int(counters.get("op_count", 0))
        self._refresh_signs()
        self.notify_topology_change()                            # rebuild per-edge caches for the loaded edge set first
        for mid, mech_state in state.get("mechanisms", {}).items():
            mech = self.mechanisms.get(mid)
            if mech is not None:
                mech.load_state_dict(copy.deepcopy(mech_state))
            elif strict and self.weave.strict:
                raise KeyError(f"State for mechanism '{mid}' cannot be restored: mechanism is not attached")

    def create_snapshot(self) -> SubstrateSnapshot:
        snapshot = super().create_snapshot()
        # always recorded, so rolling back to a snapshot taken in legacy mode also restores the mechanism configuration
        snapshot.metadata["weave"] = self.weave_state_dict()
        return snapshot

    def restore_snapshot(self, snapshot: SubstrateSnapshot) -> None:
        super().restore_snapshot(snapshot)
        weave_state = snapshot.metadata.get("weave") if isinstance(snapshot.metadata, dict) else None
        if weave_state is not None:
            self.load_weave_state_dict(weave_state, strict=False)

    def clone_substrate(self) -> "WeaveSubstrate":
        """Isolated clone that stays a ``WeaveSubstrate`` (same mechanisms and learned state)."""
        cloned = WeaveSubstrate(
            topology=self.topology, sensory_dim=self.sensory_dim, leak=self.leak, microsteps=self.microsteps,
            current_cap=self.current_cap, seed=self.seed, weave=self.weave, motor_offset=self.motor_offset,
        )
        with torch.no_grad():
            cloned.base_w.copy_(self.base_w)
            cloned.base_pre.copy_(self.base_pre)
            cloned.base_post.copy_(self.base_post)
            cloned.B.copy_(self.B)
            cloned.bias.copy_(self.bias)
            cloned.plastic_pre = self.plastic_pre.detach().clone()
            cloned.plastic_post = self.plastic_post.detach().clone()
            cloned.plastic_w = self.plastic_w.detach().clone()
        cloned.region_ids = self.region_ids.clone()
        cloned.w_max = self.w_max
        base = RecurrentSubstrate.create_snapshot(self)
        RecurrentSubstrate.restore_snapshot(cloned, base)
        cloned.load_weave_state_dict(self.weave_state_dict(), strict=False)
        return cloned

    # ---------------------------------------------------------- observability
    def weave_telemetry(self) -> Dict[str, Any]:
        """Measured (not estimated) runtime state for dashboards and experiment receipts."""
        mech = {}
        for mid, m in self.mechanisms.items():
            try:
                mech[mid] = m.telemetry()
            except Exception as exc:   # telemetry must never break a run
                mech[mid] = {"telemetry_error": f"{type(exc).__name__}: {exc}"}
        w = self.effective_weights().detach()
        return {
            "weave_version": WEAVE_VERSION,
            "profile": self.weave.profile,
            "enabled": list(self.weave.enabled_ids()),
            "active_plugins": list(self.mechanisms),
            "faults": list(self.weave_faults),
            "learning_enabled": bool(self.learning_enabled),
            "neurons": int(self.total_neurons),
            "edges": {"base": self.syn.n_base, "legacy_plastic": self.syn.n_plastic, "grown": self.syn.n_grown,
                      "alive": int(self.syn.alive.sum().item()), "mean_abs_weight": float(w.abs().mean().item()) if w.numel() else 0.0,
                      "mean_abs_delta": float(self.syn.w_delta.abs().mean().item()) if w.numel() else 0.0,
                      "routing_gain_mean": float(self.syn.routing_gain.mean().item()) if w.numel() else 1.0},
            "mean_rate": float(self.r.detach().mean().item()),
            "sparse_count": int(self.last_sparse_count),
            "cell": self.cell.summary(),
            "signals": self.signals.as_dict(),
            "mechanisms": mech,
            "counters": {"weave_steps": int(self.weave_steps), "op_count": int(self.op_count), "steps": int(self.step_count)},
        }


__all__ = ["WeaveSubstrate", "SynapseState", "EDGE_FIELDS", "MAX_DELAY", "neuron_sign"]
