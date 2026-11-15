"""Vectorised multi-compartment, multi-soma cell model.

One :class:`WeaveCell` holds the state of *all* neurons of a substrate as
``(N,)`` tensors and advances them one microstep at a time.  It adapts the
NEUROWEAVE-LAB compartment model to SONAPSE-EVO's rate convention
(``r in [0, 1]``, ``phi(x) = max(0, tanh x)``) and extends it:

    dendrites   proximal (feed-forward), distal (recurrent + context),
                apical (top-down), basal (proximal + distal)
    somas       primary (the legacy rate ``r``), secondary (persistent trace
                of the primary), fast, slow, context, integrator, burst, gating
    membrane    membrane proxy, adaptive threshold, refractory period, spike,
                calcium proxy, adaptation
    axon        hillock, N branches, terminal, release with depleting resource
    glia        homeostatic gain, astrocyte-like reservoir, myelin proxy

Design rule: **every compartment is a residual around the legacy drive**

    drive0 = W r + B s + aux + bias            (exactly the legacy expression)

so that (a) with no cell mechanism enabled the update *is* the legacy update,
operation for operation, and (b) each enabled compartment changes temporal
dynamics, saturation, gating or release rather than adding an unbounded gain.
The compartments are not decorative: when enabled, synaptic current reaches the
soma only through the dendrites, the output rate only through the axon.

All coefficients are plain parameters (see :class:`CellParams`) and none of
them is fitted to biological data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    from .contracts import StepContext
    from .substrate import WeaveSubstrate


@dataclass
class CellParams:
    # dendritic time constants (per-microstep retention) and coupling
    alpha_prox: float = 0.30      # proximal: fast feed-forward integration
    alpha_dist: float = 0.60      # distal: slower recurrent integration
    alpha_apic: float = 0.80      # apical: slow contextual integration
    kappa_ctx: float = 0.30       # share of the structural context message in the distal input
    dend_saturation: float = 3.0  # branch saturation scale: z_sat * tanh(z / z_sat)
    apical_gate: float = 0.25     # multiplicative apical gating of the basal drive
    apical_weight: float = 0.30   # direct apical contribution to the soma
    # soma populations
    beta_fast: float = 0.42
    beta_slow: float = 0.94
    beta_ctx: float = 0.69
    beta_secondary: float = 0.90
    ctx_apical: float = 0.34
    w_fast: float = 0.35          # onset emphasis: (fast - slow)
    w_slow: float = 0.18          # long integration: (slow - instantaneous)
    w_ctx: float = 0.19           # neighbourhood / graph context
    w_secondary: float = 0.12     # persistence of the primary soma's past rate
    # membrane / spike process
    theta0: float = 0.22
    theta_min: float = 0.12
    theta_max: float = 0.78
    theta_adapt: float = 0.18
    refractory_steps: float = 2.0
    spike_boost: float = 0.12
    spike_reset: float = 0.26
    # axon
    n_branches: int = 2
    axon_mix: float = 0.25        # share of terminal release in the emitted rate
    tau_recovery: float = 20.0    # release-resource recovery (microsteps)
    release_use: float = 0.20     # fraction of resource used per unit release
    branch_gain: float = 0.30     # branch-specific modulation of transmission
    # sparse code
    sparsity: float = 0.20        # fraction of winners
    loser_scale: float = 0.08
    sparse_per_region: bool = False
    # glia / homeostasis
    target_rate: float = 0.20
    homeo_rate: float = 0.015
    gain_min: float = 0.58
    gain_max: float = 1.12
    astro_syn: float = 0.50       # coupling of the astrocyte reservoir to incoming synaptic efficacy
    rate_ema_rate: float = 0.02
    # predictive coding
    pc_gain: float = 0.055

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "CellParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (values or {}).items() if k in known})


#: per-neuron state vectors.  ``persistent`` ones survive an episode reset.
_TRANSIENT = (
    "prox", "dist", "apic", "basal",
    "soma_secondary", "soma_fast", "soma_slow", "soma_context", "soma_integrator", "soma_burst",
    "membrane", "refractory", "spike", "calcium", "adaptation",
    "hillock", "terminal", "release",
    "pred_estimate", "pred_error", "pred_surprise", "tm_fast", "tm_mid",
)
_PERSISTENT = ("homeo", "astro", "myelin", "utility", "trace", "consolidation", "rate_ema", "tm_slow")   # adapt only while learning
_UNIT_INIT = ("soma_gating", "resource")        # transient, reset to 1
_THRESHOLD = ("threshold",)                     # transient, reset to theta0
CELL_FIELDS = _TRANSIENT + _UNIT_INIT + _THRESHOLD + _PERSISTENT

SOMA_ROLES = ("primary", "secondary", "fast", "slow", "context", "integrator", "burst", "gating")
DENDRITES = ("proximal", "distal", "apical", "basal")


class WeaveCell:
    """State and dynamics of the multi-compartment population."""

    def __init__(self, n: int, seed: int = 42, params: Optional[CellParams] = None, device=None):
        self.n = int(n)
        self.seed = int(seed)
        self.p = params or CellParams()
        self.device = device or torch.device("cpu")
        self.branches = torch.zeros((max(2, int(self.p.n_branches)), self.n), dtype=torch.float32, device=self.device)
        self.spike_count = 0
        self.steps = 0
        self.hard_reset()

    # -- state management ---------------------------------------------------
    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self.n, dtype=torch.float32, device=self.device)

    def _astro_rest(self) -> float:
        return 1.0 / (1.0 + 1.6 * self.p.target_rate)

    def hard_reset(self) -> None:
        """Restore every compartment to its initial value (including slow state)."""
        for name in _TRANSIENT + _PERSISTENT:
            setattr(self, name, self._zeros())
        rng = np.random.default_rng(self.seed + 7919)
        self.myelin = torch.as_tensor(0.55 + 0.35 * rng.random(self.n), dtype=torch.float32, device=self.device)
        self.homeo = torch.ones(self.n, dtype=torch.float32, device=self.device)
        self.astro = torch.full((self.n,), self._astro_rest(), dtype=torch.float32, device=self.device)
        self.rate_ema = torch.full((self.n,), float(self.p.target_rate), dtype=torch.float32, device=self.device)
        self._reset_unit_fields()
        self.spike_count = 0
        self.steps = 0

    def _reset_unit_fields(self) -> None:
        self.soma_gating = torch.ones(self.n, dtype=torch.float32, device=self.device)
        self.resource = torch.ones(self.n, dtype=torch.float32, device=self.device)
        self.threshold = torch.full((self.n,), float(self.p.theta0), dtype=torch.float32, device=self.device)
        self.branches = torch.zeros_like(self.branches)

    def reset(self) -> None:
        """Episode reset: fast compartment state is cleared, slow regulation is kept."""
        for name in _TRANSIENT:
            setattr(self, name, self._zeros())
        self._reset_unit_fields()

    def state_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {name: getattr(self, name).detach().clone() for name in CELL_FIELDS}
        out["branches"] = self.branches.detach().clone()
        out["params"] = asdict(self.p)
        out["spike_count"] = int(self.spike_count)
        out["steps"] = int(self.steps)
        return out

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "params" in state:
            self.p = CellParams.from_dict(state["params"])
        for name in CELL_FIELDS:
            if name in state:
                value = torch.as_tensor(state[name], dtype=torch.float32, device=self.device)
                if value.shape != (self.n,):
                    raise ValueError(f"cell field '{name}' has shape {tuple(value.shape)}, expected ({self.n},)")
                setattr(self, name, value.clone())
        if "branches" in state:
            branches = torch.as_tensor(state["branches"], dtype=torch.float32, device=self.device)
            if branches.dim() != 2 or branches.shape[1] != self.n:
                raise ValueError("cell branches have an incompatible shape")
            self.branches = branches.clone()
        self.spike_count = int(state.get("spike_count", self.spike_count))
        self.steps = int(state.get("steps", self.steps))

    def soma_matrix(self, rates: torch.Tensor) -> torch.Tensor:
        """(8, N) matrix of the soma populations in ``SOMA_ROLES`` order."""
        return torch.stack([
            rates.detach(), self.soma_secondary, self.soma_fast, self.soma_slow,
            self.soma_context, self.soma_integrator, self.soma_burst, self.soma_gating,
        ])

    def dendrite_matrix(self) -> torch.Tensor:
        return torch.stack([self.prox, self.dist, self.apic, self.basal])

    def astro_gain(self) -> torch.Tensor:
        """Astrocyte reservoir expressed as a gain that is exactly 1 at its resting value."""
        rest = torch.full_like(self.astro, self._astro_rest())     # same float32 rounding as the initial fill: x / x == 1
        return self.astro / rest

    # -- dynamics -----------------------------------------------------------
    @staticmethod
    def _sat(z: torch.Tensor, scale: float) -> torch.Tensor:
        return scale * torch.tanh(z / scale)

    def integrate(self, sub: "WeaveSubstrate", ctx: "StepContext") -> torch.Tensor:
        """Advance all compartments one microstep and return the new rate vector."""
        p = self.p
        en = sub.weave.is_enabled
        multi_soma = en("multi_soma")
        r = ctx.r_prev
        q = ctx.rec_in                                   # recurrent current   -> distal dendrite
        c = ctx.ctx_in                                   # structural context message
        a = ctx.apical_in                                # top-down current    -> apical dendrite
        sens = ctx.sens_in if ctx.sensory_gain is None else ctx.sens_in * ctx.sensory_gain
        s = sens + ctx.sensory_extra                     # feed-forward current -> proximal dendrite

        # ---- dendritic compartments: local integration with branch-specific dynamics
        prox = p.alpha_prox * self.prox + (1.0 - p.alpha_prox) * s
        dist_in = q if c is q else q + p.kappa_ctx * (c - q)
        dist = p.alpha_dist * self.dist + (1.0 - p.alpha_dist) * dist_in
        apic = p.alpha_apic * self.apic + (1.0 - p.alpha_apic) * a
        basal = prox + dist
        if en("dendritic_compartments"):
            gate = 1.0 + p.apical_gate * torch.tanh(apic)                    # context-dependent gating
            dend = gate * self._sat(basal, p.dend_saturation) + p.apical_weight * self._sat(apic, p.dend_saturation)
            drive = dend + ctx.aux_in + ctx.bias + ctx.extra_in
        else:
            drive = q + sens + ctx.aux_in + ctx.bias                         # legacy expression, same op order
            drive = drive + ctx.sensory_extra + ctx.extra_in + p.apical_weight * a

        # ---- soma populations (always tracked, mixed into the drive when enabled)
        sq = s + q
        fast = p.beta_fast * self.soma_fast + (1.0 - p.beta_fast) * sq
        slow = p.beta_slow * self.soma_slow + (1.0 - p.beta_slow) * sq
        context = p.beta_ctx * self.soma_context + (1.0 - p.beta_ctx) * (c + p.ctx_apical * a)
        secondary = p.beta_secondary * self.soma_secondary + (1.0 - p.beta_secondary) * r
        if multi_soma:
            fused = (drive
                     + p.w_fast * (fast - slow)            # fast soma: rapid, onset-weighted response
                     + p.w_slow * (slow - sq)              # slow soma: long-window integration
                     + p.w_ctx * (context - q)             # context soma: neighbourhood information
                     + p.w_secondary * (secondary - r))    # secondary soma: persistent context
        else:
            fused = drive

        tm_fast, tm_mid, tm_slow = self.tm_fast, self.tm_mid, self.tm_slow
        if en("hierarchical_temporal_memory"):
            fused = fused + 0.045 * (tm_fast - tm_mid) + 0.035 * (tm_mid - tm_slow)

        pred_estimate, pred_error, pred_surprise = self.pred_estimate, self.pred_error, self.pred_surprise
        if en("local_predictive_coding"):
            expected = fused - sq + (0.65 * fast + 0.35 * slow)     # drive expected from the soma-filtered input
            prediction = 0.58 * pred_estimate + 0.42 * expected
            pred_error = fused - prediction
            pred_surprise = torch.clamp(0.94 * pred_surprise + 0.06 * pred_error.abs(), 0.0, 2.0)
            fused = fused + torch.clamp(pred_error, -1.5, 1.5) * p.pc_gain / (1.0 + pred_surprise)
            pred_estimate = 0.91 * prediction + 0.09 * fused

        integrator = 0.91 * self.soma_integrator + 0.09 * fused                       # cross-compartment integrator
        burst = 0.72 * self.soma_burst + 0.28 * torch.relu(fused - self.threshold)    # suprathreshold activity
        gating = torch.clamp(0.75 * self.soma_gating + 0.25 * ctx.gating_drive, 0.65, 1.35)
        eff = fused * gating if multi_soma else fused

        # ---- membrane, adaptive threshold, refractory period, calcium
        membrane, threshold, refractory = self.membrane, self.threshold, self.refractory
        adaptation, calcium = self.adaptation, self.calcium
        spiking = en("membrane_spiking")
        if spiking:
            adaptation = 0.995 * adaptation + 0.005 * r
            threshold = torch.clamp(p.theta0 + p.theta_adapt * adaptation + ctx.threshold_shift, p.theta_min, p.theta_max)
            m_drive = 0.74 * integrator + 0.18 * burst + 0.08 * gating * fused if multi_soma else fused
            membrane = 0.78 * membrane + 0.22 * m_drive - 0.06 * adaptation
            refractory = torch.clamp(refractory - 1.0, min=0.0)
            spike = ((membrane > threshold) & (refractory <= 0.0)).to(torch.float32)
            membrane = torch.where(spike > 0, p.spike_reset * membrane, membrane)
            refractory = torch.where(spike > 0, torch.full_like(refractory, p.refractory_steps), refractory)
            calcium = 0.965 * calcium + 0.075 * spike + 0.008 * fused.abs()
        else:
            spike = torch.zeros_like(r)

        # ---- axon hillock and activation
        hill_target = torch.where(spike > 0, torch.ones_like(eff), torch.tanh(eff))
        hillock = 0.66 * self.hillock + 0.34 * hill_target
        act = sub.activation_phi(eff)
        if spiking:
            act = torch.clamp(act + p.spike_boost * spike * hillock, 0.0, 1.0)

        # ---- axonal branches, terminal and release
        branches, terminal, release, resource = self.branches, self.terminal, self.release, self.resource
        if en("axonal_branches"):
            nb = branches.shape[0]
            rows = []
            for k in range(nb):
                alpha = 0.72 + 0.12 * (k / max(1, nb - 1))
                couple = 0.2 * apic if k == 0 else (-0.15 * dist if k == 1 else 0.0)
                rows.append(alpha * branches[k] + (1.0 - alpha) * (hillock + couple))
            branches = torch.stack(rows)
            if nb == 2:
                terminal = 0.6 * branches[0] + 0.4 * branches[1]
            else:
                terminal = branches.mean(dim=0)
            resource = torch.clamp(resource + (1.0 - resource) / p.tau_recovery - p.release_use * resource * release, 0.0, 1.0)
            release = torch.relu(torch.tanh(terminal)) * resource
            if ctx.transmitter_gain is not None:
                release = release * ctx.transmitter_gain
            act = (1.0 - p.axon_mix) * act + p.axon_mix * release

        # ---- sparse distributed code (k-winners)
        if en("sparse_distributed_code"):
            act = self._k_winners(sub, act)

        # ---- glial / homeostatic regulation
        # slow regulation adapts only while learning; when frozen its current value is applied unchanged
        adapt = bool(ctx.learning)
        homeo, astro = self.homeo, self.astro
        local_gain = None
        if en("homeostatic_governor"):
            if adapt:
                homeo = torch.clamp(homeo + p.homeo_rate * (p.target_rate - self.rate_ema), p.gain_min, p.gain_max)
            local_gain = homeo
        if en("astrocyte_regulation"):
            a_gain = self.astro_gain()                                      # apply the stored reservoir, then update it
            local_gain = a_gain if local_gain is None else local_gain * a_gain
            if adapt:
                pool = 0.5 * r.detach() + 0.5 * torch.tanh(q.detach().abs())    # own + neighbourhood activity
                astro = 0.97 * astro + 0.03 / (1.0 + 1.6 * pool)
        if local_gain is not None:
            act = act * torch.clamp(local_gain, p.gain_min, p.gain_max)
        if ctx.gain is not None:
            act = act * ctx.gain

        # ---- leaky rate integration (identical expression to the legacy update)
        leak = sub.leak if ctx.leak_scale == 1.0 else float(min(1.0, max(0.0, sub.leak * ctx.leak_scale)))
        new_r = (1.0 - leak) * r + leak * act
        new_r = torch.clamp(new_r, 0.0, 1.0)
        if sub.node_mask is not None:
            new_r = new_r * sub.node_mask

        # ---- commit compartment state (truncated: state never carries a graph)
        rd = new_r.detach()
        self.prox, self.dist, self.apic, self.basal = prox.detach(), dist.detach(), apic.detach(), basal.detach()
        self.soma_fast, self.soma_slow = fast.detach(), slow.detach()
        self.soma_context, self.soma_secondary = context.detach(), secondary.detach()
        self.soma_integrator, self.soma_burst, self.soma_gating = integrator.detach(), burst.detach(), gating.detach()
        self.membrane, self.threshold, self.refractory = membrane.detach(), threshold.detach(), refractory.detach()
        self.spike, self.calcium, self.adaptation = spike.detach(), calcium.detach(), adaptation.detach()
        self.hillock, self.branches = hillock.detach(), branches.detach()
        self.terminal, self.release, self.resource = terminal.detach(), release.detach(), resource.detach()
        self.pred_estimate, self.pred_error, self.pred_surprise = pred_estimate.detach(), pred_error.detach(), pred_surprise.detach()
        self.homeo, self.astro = homeo.detach(), astro.detach()

        if adapt:
            self.trace = 0.992 * self.trace + 0.008 * rd
        if en("hierarchical_temporal_memory"):
            self.tm_fast = 0.62 * tm_fast + 0.38 * rd
            self.tm_mid = 0.94 * tm_mid + 0.06 * self.tm_fast
            if adapt:
                self.tm_slow = 0.994 * tm_slow + 0.006 * self.tm_mid
        if adapt:
            self.utility = 0.985 * self.utility + 0.015 * (rd * self.soma_integrator).abs()
            self.rate_ema = (1.0 - p.rate_ema_rate) * self.rate_ema + p.rate_ema_rate * rd
        if en("adaptive_conduction") and adapt:
            self.myelin = torch.clamp(self.myelin + 0.0005 * self.hillock.abs() - 0.00008, 0.2, 0.98)

        self.spike_count += int(spike.sum().item())
        self.steps += 1
        ctx.drive, ctx.fused, ctx.r_new = drive, fused, new_r
        return new_r

    def _k_winners(self, sub: "WeaveSubstrate", act: torch.Tensor) -> torch.Tensor:
        p = self.p

        def select(values: torch.Tensor) -> torch.Tensor:
            n = values.numel()
            k = max(1, int(round(n * p.sparsity)))
            if k >= n:
                return torch.ones_like(values, dtype=torch.bool)
            kth = torch.topk(values.detach(), k).values[-1]
            return values.detach() >= kth

        if p.sparse_per_region and sub.region_ids is not None:
            mask = torch.zeros_like(act, dtype=torch.bool)
            for region in torch.unique(sub.region_ids):
                idx = torch.nonzero(sub.region_ids == region, as_tuple=False).squeeze(-1)
                mask[idx] = select(act[idx])
        else:
            mask = select(act)
        sub.last_sparse_count = int(mask.sum().item())
        return torch.where(mask, act, act * p.loser_scale)

    # -- observability ------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        def m(x: torch.Tensor) -> float:
            return float(x.abs().mean().item())

        return {
            "dend_proximal": m(self.prox), "dend_distal": m(self.dist), "dend_apical": m(self.apic), "dend_basal": m(self.basal),
            "soma_secondary": m(self.soma_secondary), "soma_fast": m(self.soma_fast), "soma_slow": m(self.soma_slow),
            "soma_context": m(self.soma_context), "soma_integrator": m(self.soma_integrator), "soma_burst": m(self.soma_burst),
            "soma_gating": float(self.soma_gating.mean().item()),
            "membrane": m(self.membrane), "threshold": float(self.threshold.mean().item()),
            "spike_rate": float(self.spike.mean().item()), "calcium": m(self.calcium),
            "hillock": m(self.hillock), "terminal": m(self.terminal), "release": m(self.release),
            "resource": float(self.resource.mean().item()),
            "homeo": float(self.homeo.mean().item()), "astro_gain": float(self.astro_gain().mean().item()),
            "myelin": float(self.myelin.mean().item()), "utility": m(self.utility),
            "surprise": float(self.pred_surprise.mean().item()), "prediction_error": m(self.pred_error),
            "tm_fast": m(self.tm_fast), "tm_mid": m(self.tm_mid), "tm_slow": m(self.tm_slow),
        }


__all__ = ["CellParams", "WeaveCell", "CELL_FIELDS", "SOMA_ROLES", "DENDRITES"]
