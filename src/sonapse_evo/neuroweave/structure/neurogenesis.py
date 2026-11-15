"""Structural neurogenesis: recruiting, repurposing and adding neurons under explicit budgets.

Two operations, both audited in the ledger with the measurement that triggered them.

1. In-place repurposing (the mechanism; population size fixed).  Every ``period`` steps, if the demand
   signal (mean local prediction surprise, or ``signals.surprise``) exceeds ``demand_threshold``:
   the ``batch`` neurons with the lowest utility and rate (never protected ones) are detached from their
   weakest role - their weave-learned weight changes are cleared and their outgoing/incoming synapses are
   pruned at weave level - and re-wired as a small recurrent subnetwork inside the region with the highest
   demand: each gets ``fan_in`` inputs from that region's most active neurons, ``fan_out`` outputs to its
   highest-utility neurons, and links to the other recruits (a new specialised pathway).  Their functional
   region label (``substrate.region_ids``) is updated; the original topology object is not modified.
2. True growth (``grow_substrate``).  Returns a NEW, larger ``WeaveSubstrate`` with ``n_new`` extra neurons
   appended to a chosen region, copying all weights, rates, compartment state and learned weave deltas of
   the existing neurons.  The source substrate is untouched, so growth is trivially reversible.

Safeguards against uncontrolled growth: a cooldown between events, a per-event batch size, a lifetime
budget (``max_recruits``; ``max_neurons`` for true growth), and the demand gate.  Nothing happens unless a
measured signal asks for capacity.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ...substrate.topology_state import TopologyState
from ..contracts import Mechanism, SignalKind, Stage, StepContext, rng_state, set_rng_state
from ..substrate import EDGE_FIELDS, WeaveSubstrate, neuron_sign
from .ledger import StructuralLedger


class Neurogenesis(Mechanism):
    mechanism_id = "neurogenesis"
    stage = Stage.STRUCTURE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"period": 256, "batch": 4, "fan_in": 6, "fan_out": 6, "weight": 0.12, "demand_threshold": 0.02,
                "max_recruits": 64, "cooldown": 256, "protect_first": 3, "protect_motor": 64}

    def on_attach(self) -> None:
        self.rng = np.random.default_rng(self.substrate.weave.seed + 99991)
        self.ledger = StructuralLedger()
        self.recruited = 0
        self.last_event = -10 ** 9
        self.demand_ema = 0.0
        self.is_recruit = torch.zeros(self.substrate.total_neurons, dtype=torch.bool)

    def demand_by_region(self) -> Dict[int, float]:
        sub = self.substrate
        signal = sub.cell.pred_surprise + sub.cell.pred_error.abs() + (sub.cell.rate_ema - sub.cell.p.target_rate).clamp(min=0)
        return {int(g): float(signal[sub.region_ids == g].mean()) for g in torch.unique(sub.region_ids)}

    def run_recruitment(self, step: int = 0, force: bool = False) -> Dict[str, Any]:
        sub, syn, p = self.substrate, self.substrate.syn, self.params
        demand = self.demand_by_region()
        level = max(max(demand.values()), float(sub.signals.surprise))
        self.demand_ema = 0.9 * self.demand_ema + 0.1 * level
        report = {"recruited": 0, "demand": level}
        if not sub.learning_enabled:
            return report
        budget = min(int(p["batch"]), int(p["max_recruits"]) - self.recruited)
        if budget <= 0:
            report["blocked"] = "lifetime_budget"
            return report
        if not force and (step - self.last_event < int(p["cooldown"]) or level < p["demand_threshold"]):
            report["blocked"] = "cooldown" if step - self.last_event < int(p["cooldown"]) else "no_demand"
            return report
        target_region = max(demand, key=lambda g: (demand[g], -g))
        members = torch.nonzero(sub.region_ids == target_region, as_tuple=False).squeeze(-1)
        usage = sub.cell.utility + sub.cell.rate_ema
        usage = usage.clone()
        usage[: int(p["protect_first"])] = float("inf")
        width = int(p["protect_motor"])                     # the read-out window exactly as the agent loop computes it
        readout = sub.motor_offset if sub.motor_offset + width <= sub.total_neurons else 0
        usage[readout: readout + width] = float("inf")
        usage[self.is_recruit] = float("inf")
        candidates = torch.nonzero(torch.isfinite(usage), as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            report["blocked"] = "no_candidates"
            return report
        recruits = candidates[torch.argsort(usage[candidates], stable=True)[:budget]]
        pre, post = sub.edge_index()
        touching = torch.isin(pre, recruits) | torch.isin(post, recruits)
        touching[: int(p["protect_first"])] = False
        syn.w_delta[touching] = 0.0
        sub.prune_edges(torch.nonzero(touching & (syn.alive > 0), as_tuple=False).squeeze(-1))
        active = members[torch.argsort(sub.cell.rate_ema[members], descending=True, stable=True)[: int(p["fan_in"])]]
        useful = members[torch.argsort(sub.cell.utility[members], descending=True, stable=True)[: int(p["fan_out"])]]
        new_pre, new_post = [], []
        for nrn in recruits.tolist():
            new_pre += active.tolist() + [nrn] * len(useful) + [nrn] * (len(recruits) - 1)
            new_post += [nrn] * len(active) + useful.tolist() + [m for m in recruits.tolist() if m != nrn]
        tp, tq = torch.as_tensor(new_pre, dtype=torch.long), torch.as_tensor(new_post, dtype=torch.long)
        grown = sub.grow_edges(tp, tq, neuron_sign(tp) * float(p["weight"]))
        for nrn in recruits.tolist():
            old = int(sub.region_ids[nrn])
            sub.region_ids[nrn] = int(target_region)
            self.is_recruit[nrn] = True
            self.ledger.record(step, "repurpose_neuron", f"neuron {nrn}", "under_utilised_and_region_demand",
                               from_region=old, to_region=int(target_region), usage=float(sub.cell.utility[nrn] + sub.cell.rate_ema[nrn]),
                               region_demand=float(demand[target_region]))
        self.ledger.record(step, "form_subnetwork", f"region {target_region}", "recruits_wired_recurrently",
                           neurons=recruits.tolist(), synapses=int(grown))
        self.recruited += int(recruits.numel())
        self.last_event = step
        report.update(recruited=int(recruits.numel()), region=int(target_region), synapses=int(grown))
        sub.events.publish("structure.neurogenesis", SignalKind.STATE_UPDATE, step, report, self.mechanism_id)
        return report

    def end_step(self, ctx: StepContext) -> None:
        if ctx.learning and self.substrate.weave_steps % int(self.params["period"]) == 0:
            self.run_recruitment(ctx.step)

    def state_dict(self) -> Dict[str, Any]:
        return {"recruited": int(self.recruited), "last_event": int(self.last_event), "demand_ema": float(self.demand_ema),
                "is_recruit": self.is_recruit.clone(), "region_ids": self.substrate.region_ids.clone(), "ledger": self.ledger.state_dict(), "rng": rng_state(self.rng)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.recruited, self.last_event = int(state.get("recruited", 0)), int(state.get("last_event", -10 ** 9))
        self.demand_ema = float(state.get("demand_ema", 0.0))
        if "is_recruit" in state:
            self.is_recruit = torch.as_tensor(state["is_recruit"], dtype=torch.bool).clone()
        if "region_ids" in state:
            self.substrate.region_ids = torch.as_tensor(state["region_ids"], dtype=torch.long).clone()
        self.ledger.load_state_dict(state.get("ledger", {}))
        set_rng_state(self.rng, state.get("rng"))

    def telemetry(self) -> Dict[str, Any]:
        return {"recruited": int(self.recruited), "budget_remaining": int(self.params["max_recruits"]) - int(self.recruited),
                "demand": self.demand_ema, "regional_demand": {str(k): round(v, 5) for k, v in self.demand_by_region().items()},
                "ledger": self.ledger.summary()}


def grow_substrate(substrate: WeaveSubstrate, n_new: int, region_id: Optional[int] = None, seed: int = 0,
                   fan_in: int = 6, fan_out: int = 6, weight: float = 0.12, max_neurons: int = 65536,
                   reason: str = "requested") -> Tuple[WeaveSubstrate, Dict[str, Any]]:
    """Return a NEW substrate with ``n_new`` additional neurons; the source substrate is left untouched."""
    n_old = substrate.total_neurons
    if n_new <= 0:
        raise ValueError("n_new must be positive")
    if n_old + n_new > max_neurons:
        raise ValueError(f"growth budget exceeded: {n_old} + {n_new} > max_neurons={max_neurons}")
    rng = np.random.default_rng(seed)
    regions = substrate.region_ids.clone()
    region_id = int(regions.max()) if region_id is None else int(region_id)
    members = torch.nonzero(regions == region_id, as_tuple=False).squeeze(-1)
    new_ids = np.arange(n_old, n_old + n_new)
    pre, post, w = [substrate.base_pre.numpy()], [substrate.base_post.numpy()], [substrate.base_w.detach().numpy()]
    active = members[torch.argsort(substrate.cell.rate_ema[members], descending=True, stable=True)[:fan_in]].numpy()
    useful = members[torch.argsort(substrate.cell.utility[members], descending=True, stable=True)[:fan_out]].numpy()
    for nrn in new_ids:
        sign = -1.0 if nrn % 5 == 0 else 1.0
        pre += [active, np.full(len(useful), nrn)]
        post += [np.full(len(active), nrn), useful]
        w += [np.where(active % 5 == 0, -1.0, 1.0) * weight * rng.uniform(0.6, 1.0, len(active)),
              sign * weight * rng.uniform(0.6, 1.0, len(useful))]
    topo = substrate.topology
    assignments = np.concatenate([regions.numpy(), np.full(n_new, region_id)]).astype(np.int64)
    new_topo = TopologyState(total_neurons=n_old + n_new, num_regions=topo.num_regions, region_names=topo.region_names,
                             region_assignments=assignments, base_pre=np.concatenate(pre).astype(np.int64),
                             base_post=np.concatenate(post).astype(np.int64), base_w=np.concatenate(w).astype(np.float32), seed=topo.seed)
    grown = WeaveSubstrate(topology=new_topo, sensory_dim=substrate.sensory_dim, leak=substrate.leak, microsteps=substrate.microsteps,
                           current_cap=substrate.current_cap, seed=substrate.seed, weave=substrate.weave, motor_offset=substrate.motor_offset)
    m = len(substrate.base_w)
    src, p_len = substrate.syn, len(substrate.plastic_w)
    with torch.no_grad():                                   # legacy plastic edges and weave-grown edges come along too
        grown.plastic_pre = substrate.plastic_pre.detach().clone()
        grown.plastic_post = substrate.plastic_post.detach().clone()
        grown.plastic_w = substrate.plastic_w.detach().clone()
    grown._sync_synapses()
    fixed_src = src.n_base + src.n_plastic
    grown.syn.append(src.grown_pre, src.grown_post, src.grown_w, delay=src.delay[fixed_src:], branch=src.branch[fixed_src:])
    nb = grown.syn.n_base
    for name in EDGE_FIELDS:                                 # copy every per-edge value of every edge that existed before
        field, source = getattr(grown.syn, name), getattr(src, name)
        field[:m] = source[:m]
        field[nb: nb + p_len] = source[m: m + p_len]
        if src.n_grown:
            field[nb + p_len:] = source[fixed_src:]
    grown.notify_topology_change()
    with torch.no_grad():
        grown.B.zero_(); grown.bias.zero_(); grown.r.zero_()
        grown.B[:n_old] = substrate.B
        grown.bias[:n_old] = substrate.bias
        grown.bias[n_old:] = float(substrate.bias.mean())
        grown.r[:n_old] = substrate.r.detach()
        grown.lesion_mask[:m] = substrate.lesion_mask
    cell_state = substrate.cell.state_dict()
    for name, value in cell_state.items():
        if isinstance(value, torch.Tensor) and value.dim() == 1 and value.numel() == n_old:
            getattr(grown.cell, name)[:n_old] = value
    grown.cell.branches[:, :n_old] = substrate.cell.branches
    grown.signals.load(substrate.signals.as_dict())
    grown.step_count = substrate.step_count
    receipt = {"action": "grow_population", "reason": reason, "neurons_before": n_old, "neurons_after": n_old + n_new,
               "region": region_id, "new_neurons": new_ids.tolist(), "new_synapses": int(n_new * (len(active) + len(useful))),
               "carried_plastic_synapses": int(p_len), "carried_grown_synapses": int(src.n_grown),
               "source_untouched": True, "note": "per-neuron state of plug-in mechanisms is re-initialised in the grown substrate"}
    return grown, receipt
