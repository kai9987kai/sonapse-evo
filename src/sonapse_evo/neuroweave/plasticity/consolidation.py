"""Sleep-like consolidation, replay and synaptic downscaling.

Every ``period`` environment steps (or on ``run_sleep()``) an offline phase runs:

1. Replay.  Stored activity patterns (from ``predictive_replay``,
   ``memory_ecology`` or ``associative_trace_memory`` when present, otherwise the
   mechanism's own small buffer of high-reward states) are replayed and
   co-active pairs are potentiated:   d|w_ij| = eta_replay * x_i * x_j.
2. Consolidation.  Neurons: ``c <- 0.96 c + 0.04 |trace|``; synapses in the top
   utility fraction have their consolidation raised (protecting them when
   meta-plasticity is on).
3. Downscaling (synaptic-homeostasis flavour).  Every *unconsolidated* learned
   weight shrinks:  |w| <- |w| * (1 - downscale * (1 - consolidation)); utility
   decays by 0.995.

Controls for experiments: ``replay`` and ``downscale`` can be switched off
independently (replay-only, downscale-only, no-sleep).
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from ..contracts import Mechanism, SignalKind, Stage, StepContext


class SleepConsolidation(Mechanism):
    mechanism_id = "sleep_consolidation"
    stage = Stage.STRUCTURE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"period": 240, "replay": True, "downscale": True, "downscale_rate": 0.0005, "eta_replay": 0.002,
                "replay_batch": 4, "protect_fraction": 0.2, "buffer": 16, "reward_threshold": 0.35}

    def on_attach(self) -> None:
        self.sleep_cycles = 0
        self.replayed = 0
        self.buffer: List[torch.Tensor] = []
        self.last_downscaled = 0.0

    def _replay_patterns(self) -> List[torch.Tensor]:
        n = int(self.params["replay_batch"])
        for peer_id in ("predictive_replay", "memory_ecology", "associative_trace_memory"):
            peer = self.peer(peer_id)
            getter = getattr(peer, "replay_states", None) if peer is not None else None
            if getter is not None:
                states = [torch.as_tensor(s, dtype=torch.float32) for s in getter(n)]
                states = [s for s in states if s.numel() == self.substrate.total_neurons]
                if states:
                    return states
        return self.buffer[-n:]

    def run_sleep(self, step: int = 0) -> Dict[str, float]:
        sub, syn, cell, p = self.substrate, self.substrate.syn, self.substrate.cell, self.params
        if not sub.learning_enabled:
            return {"skipped": 1.0}
        pre, post = sub.edge_index()
        replayed = 0
        if p["replay"]:
            for state in self._replay_patterns():
                sub.apply_weight_update(syn.sign * p["eta_replay"] * state[pre] * state[post], respect_lr_scale=False)
                cell.trace = torch.clamp(cell.trace + 0.002 * state.abs(), 0.0, 2.0)
                replayed += 1
        cell.consolidation = 0.96 * cell.consolidation + 0.04 * cell.trace.abs()
        cell.trace = cell.trace * 0.998
        if syn.utility.numel():
            k = max(1, int(syn.utility.numel() * p["protect_fraction"]))
            threshold = torch.topk(syn.utility, k).values[-1]
            useful = (syn.utility >= threshold) & (syn.utility > 0)
            syn.consolidation = torch.clamp(syn.consolidation + 0.05 * useful.to(torch.float32), 0.0, 1.0)
        shrunk = 0.0
        if p["downscale"]:
            w = sub.effective_weights().detach()
            applied = sub.apply_weight_update(-p["downscale_rate"] * (1.0 - syn.consolidation) * w, respect_lr_scale=False)
            shrunk = float(applied.abs().sum())
            syn.utility = syn.utility * 0.995
        self.sleep_cycles += 1
        self.replayed += replayed
        self.last_downscaled = shrunk
        report = {"cycle": float(self.sleep_cycles), "replayed": float(replayed), "downscaled": shrunk}
        sub.events.publish("plasticity.sleep", SignalKind.LEARNING, step, report, self.mechanism_id)
        return report

    def end_step(self, ctx: StepContext) -> None:
        p = self.params
        if ctx.learning and ctx.signals.reward > p["reward_threshold"]:
            self.buffer.append(self.substrate.r.detach().clone())
            self.buffer = self.buffer[-int(p["buffer"]):]
        if ctx.learning and self.substrate.weave_steps % int(p["period"]) == 0:
            self.run_sleep(ctx.step)

    def state_dict(self) -> Dict[str, Any]:
        return {"sleep_cycles": int(self.sleep_cycles), "replayed": int(self.replayed), "buffer": [b.clone() for b in self.buffer]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.sleep_cycles = int(state.get("sleep_cycles", 0))
        self.replayed = int(state.get("replayed", 0))
        self.buffer = [torch.as_tensor(b, dtype=torch.float32).clone() for b in state.get("buffer", [])]

    def telemetry(self) -> Dict[str, Any]:
        return {"sleep_cycles": int(self.sleep_cycles), "replayed_patterns": int(self.replayed),
                "last_downscaled": self.last_downscaled, "buffered_states": len(self.buffer),
                "synapse_consolidation_mean": float(self.substrate.syn.consolidation.mean()),
                "neuron_consolidation_mean": float(self.substrate.cell.consolidation.mean())}
