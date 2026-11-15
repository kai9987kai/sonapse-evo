"""INT8-like signed synaptic quantisation and Q8.8 fixed-point state experiments.

    code = sign(w) * round(|w| / w_scale * 127)        code in [-127, 127]
    w_q  = code / 127 * w_scale

Applied to the *effective* weights at read time through ``substrate.weight_transforms``; the stored
floating-point weights are untouched, so disabling the mechanism restores full precision exactly.
Q8.8: ``round(x * 256)`` clipped to int16; optionally applied to the rate vector after each step
(``quantise_state=True``).  These are software experiments, not hardware results.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ..contracts import Mechanism, Stage, StepContext


def quantise_int8(w: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dequantised weights, int8 codes)``."""
    codes = torch.clamp(torch.round(w / scale * 127.0), -127, 127)
    return codes / 127.0 * scale, codes.to(torch.int8)


def to_q8_8(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x * 256.0), -32768, 32767).to(torch.int16)


def from_q8_8(q: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) / 256.0


class QuantisedSynapses(Mechanism):
    mechanism_id = "quantised_synapses"
    stage = Stage.TRANSMIT

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"quantise_state": False}

    def on_attach(self) -> None:
        self.last_error = 0.0
        self.levels_used = 0
        self.substrate.weight_transforms["int8"] = self._transform

    def on_detach(self) -> None:
        self.substrate.weight_transforms.pop("int8", None)

    def _transform(self, w: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.substrate.w_max), float(w.detach().abs().max()) if w.numel() else 1.0)
        wq, codes = quantise_int8(w, scale)
        self.last_error = float((wq - w).detach().abs().mean()) if w.numel() else 0.0
        self.levels_used = int(torch.unique(codes).numel())
        return w + (wq - w).detach()          # straight-through: quantised value, full-precision gradient

    def end_step(self, ctx: StepContext) -> None:
        if self.params["quantise_state"]:
            self.substrate.r = from_q8_8(to_q8_8(self.substrate.r.detach()))

    def telemetry(self) -> Dict[str, Any]:
        return {"mean_abs_quantisation_error": self.last_error, "int8_levels_used": self.levels_used,
                "state_quantised_q8_8": bool(self.params["quantise_state"])}
