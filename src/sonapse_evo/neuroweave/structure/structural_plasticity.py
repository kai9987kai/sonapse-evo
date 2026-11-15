"""Utility-guided structural plasticity: bounded synaptic pruning and regrowth.

Every ``period`` steps (learning on):

    score_e   = utility_e + 0.15 |trace_e|
    prune     the ``prune_fraction`` lowest-scoring live synapses (never the protected canonical pathway,
              never synapses whose consolidation exceeds ``protect_consolidation``)
    regrow    the same number of new synapses; each is the best of ``candidates`` random (a, b) pairs under
              coactivity_a * coactivity_b + small noise, with the source neuron's sign and a small weight

Safeguards against uncontrolled growth: regrowth never exceeds pruning unless ``net_growth`` allows it, and
the live synapse count is capped at ``max_edge_factor`` times the initial count.  Every change is written
to the audit ledger with the score that caused it.  Pruning uses the weave ``alive`` flag, so lesion masks
and the original ``base_w`` buffer are never modified and the change is fully reversible (``revert()``).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..contracts import Mechanism, SignalKind, Stage, StepContext, rng_state, set_rng_state
from ..substrate import neuron_sign
from .ledger import StructuralLedger


class StructuralPlasticity(Mechanism):
    mechanism_id = "structural_plasticity"
    stage = Stage.STRUCTURE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"period": 128, "prune_fraction": 0.008, "candidates": 32, "weight_low": 0.06, "weight_high": 0.16,
                "protect_first": 3, "protect_consolidation": 0.5, "max_edge_factor": 1.25, "net_growth": 0,
                "ledger_detail": 24}

    def on_attach(self) -> None:
        self.rng = np.random.default_rng(self.substrate.weave.seed + 86243)
        self.ledger = StructuralLedger()
        self.initial_edges = int(self.substrate.syn.alive.sum().item())
        self.pruned, self.grown, self.events = 0, 0, 0

    def run_turnover(self, step: int = 0) -> Dict[str, int]:
        sub, syn, p = self.substrate, self.substrate.syn, self.params
        if not sub.learning_enabled:
            return {"pruned": 0, "grown": 0}
        n = sub.total_neurons
        live = int(syn.alive.sum().item())
        if live < max(20, 2 * n):
            return {"pruned": 0, "grown": 0}
        pre, post = sub.edge_index()
        score = syn.utility + 0.15 * syn.trace.abs()
        eligible = (syn.alive > 0) & (syn.consolidation <= p["protect_consolidation"])
        eligible[: int(p["protect_first"])] = False
        idx = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
        k = min(int(idx.numel()), max(1, int(live * p["prune_fraction"])))
        if k == 0:
            return {"pruned": 0, "grown": 0}
        order = idx[torch.argsort(score[idx], stable=True)[:k]]
        for e in order[: int(p["ledger_detail"])].tolist():
            self.ledger.record(step, "prune_synapse", f"{int(pre[e])}->{int(post[e])}", "low_utility",
                               score=float(score[e]), weight=float(sub.effective_weights()[e]))
        pruned = sub.prune_edges(order)

        budget = min(k + int(p["net_growth"]), max(0, int(self.initial_edges * p["max_edge_factor"]) - (live - pruned)))
        coact = sub.cell.rate_ema.numpy()
        new_pre, new_post, new_w, why = [], [], [], []
        for _ in range(budget):
            a = self.rng.integers(0, n, size=int(p["candidates"]))
            b = self.rng.integers(0, n, size=int(p["candidates"]))
            s = np.abs(coact[a] * coact[b]) + self.rng.random(len(a)) * 0.015
            s[a == b] = -1.0
            j = int(np.argmax(s))
            new_pre.append(int(a[j])); new_post.append(int(b[j])); why.append(float(s[j]))
            new_w.append(float(self.rng.uniform(p["weight_low"], p["weight_high"])))
        grown = 0
        if new_pre:
            tp = torch.as_tensor(new_pre, dtype=torch.long)
            tw = torch.as_tensor(new_w, dtype=torch.float32) * neuron_sign(tp)
            before = sub.syn.n_grown
            grown = sub.grow_edges(tp, torch.as_tensor(new_post, dtype=torch.long), tw)
            for i in range(min(grown, int(p["ledger_detail"]))):
                e = sub.syn.n_base + sub.syn.n_plastic + before + i
                gp, gq = sub.edge_index()
                self.ledger.record(step, "grow_synapse", f"{int(gp[e])}->{int(gq[e])}", "coactivity",
                                   coactivity_score=why[min(i, len(why) - 1)], weight=float(sub.syn.grown_w[before + i]))
        self.pruned += pruned
        self.grown += grown
        self.events += 1
        sub.events.publish("structure.turnover", SignalKind.STATE_UPDATE, step, {"pruned": pruned, "grown": grown}, self.mechanism_id)
        return {"pruned": pruned, "grown": grown}

    def end_step(self, ctx: StepContext) -> None:
        if ctx.learning and self.substrate.weave_steps % int(self.params["period"]) == 0:
            self.run_turnover(ctx.step)

    def revert(self) -> None:
        """Undo all weave-level structural change: revive pruned synapses and drop grown ones."""
        syn = self.substrate.syn
        fixed = syn.n_base + syn.n_plastic
        syn.alive[:fixed] = 1.0
        syn.alive[fixed:] = 0.0
        syn.compact_grown()
        self.substrate.notify_topology_change()            # other mechanisms rebuild their per-edge tables
        self.ledger.record(self.substrate.step_count, "revert", "all", "manual_rollback")

    def sync_to_graph(self, graph) -> Dict[str, int]:
        """Mirror the learned structure into an existing ``GraphTopology`` (bridge to Mode B discovery operators)."""
        sub, syn = self.substrate, self.substrate.syn
        pre, post = sub.edge_index()
        w = sub.effective_weights().detach()
        fixed = syn.n_base + syn.n_plastic
        added = removed = 0
        for e in torch.nonzero(syn.alive[:fixed] == 0, as_tuple=False).squeeze(-1).tolist():
            removed += int(graph.remove_edge(int(pre[e]), int(post[e])))
        for e in range(fixed, syn.n_edges):
            graph.add_edge(int(pre[e]), int(post[e]), float(w[e]), edge_type="synapse")
            added += 1
        return {"added": added, "removed": removed}

    def state_dict(self) -> Dict[str, Any]:
        return {"pruned": int(self.pruned), "grown": int(self.grown), "events": int(self.events),
                "initial_edges": int(self.initial_edges), "ledger": self.ledger.state_dict(), "rng": rng_state(self.rng)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.pruned, self.grown, self.events = int(state.get("pruned", 0)), int(state.get("grown", 0)), int(state.get("events", 0))
        self.initial_edges = int(state.get("initial_edges", self.initial_edges))
        self.ledger.load_state_dict(state.get("ledger", {}))
        set_rng_state(self.rng, state.get("rng"))

    def telemetry(self) -> Dict[str, Any]:
        syn = self.substrate.syn
        return {"pruned": int(self.pruned), "grown": int(self.grown), "turnover_events": int(self.events),
                "live_synapses": int(syn.alive.sum().item()), "grown_synapses_present": int(syn.n_grown),
                "edge_budget": int(self.initial_edges * self.params["max_edge_factor"]), "ledger": self.ledger.summary()}
