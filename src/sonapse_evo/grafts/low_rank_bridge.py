from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts.graft_manifest import GraftSpec


class LowRankBridge(nn.Module):
    """Type A Graft: Sparse Low-Rank Bridge.
    
    I_g = alpha_g(t) * g_g(c_t) * P_T tanh(U_g * V_g * r_{S,t})
    """

    def __init__(self, spec: GraftSpec, total_neurons: int, context_dim: int = 64):
        super().__init__()
        self.spec = spec
        self.total_neurons = total_neurons
        self.context_dim = context_dim
        self.source_width = len(spec.source_neurons)
        self.target_width = len(spec.target_neurons)
        self.rank = spec.rank
        self.max_gain = spec.max_gain
        self.ramp_steps = spec.ramp_steps

        # Register port index buffers
        self.register_buffer("source_idx", torch.as_tensor(spec.source_neurons, dtype=torch.long))
        self.register_buffer("target_idx", torch.as_tensor(spec.target_neurons, dtype=torch.long))

        # Low-rank factorized matrices: V_g (k x |S|), U_g (|T| x k)
        self.V = nn.Parameter(torch.empty(self.rank, self.source_width))
        self.U = nn.Parameter(torch.empty(self.target_width, self.rank))
        self.bias = nn.Parameter(torch.zeros(self.target_width))

        # Context gate: g_g(c_t) = sigmoid(w_c^T c_t + b_c)
        self.gate_w = nn.Parameter(torch.empty(self.context_dim))
        self.gate_b = nn.Parameter(torch.zeros(1))

        # Runtime state
        self.alpha = 0.0
        self.current_step = 0
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.V, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=np.sqrt(5))
        nn.init.zeros_(self.gate_w)
        nn.init.constant_(self.gate_b, 2.0)  # default gate open ~0.88

    def reset_ramp(self, start_gain: float = 0.0):
        self.alpha = float(start_gain)
        self.current_step = 0

    def step_ramp(self):
        if self.ramp_steps <= 0:
            self.alpha = self.max_gain
        else:
            self.current_step += 1
            progress = min(1.0, self.current_step / float(self.ramp_steps))
            self.alpha = self.max_gain * progress

    def forward(
        self,
        full_rates: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes full-dimensional injection current I_g(t)."""
        if self.alpha <= 0.0 and not self.training:
            return torch.zeros_like(full_rates)

        # 1. Extract source activity r_{S,t}
        r_S = full_rates[..., self.source_idx]  # (..., |S|)

        # 2. Factorized projection: U * V * r_S
        intermediate = F.linear(r_S, self.V)  # (..., rank)
        local_target = F.linear(intermediate, self.U, self.bias)  # (..., |T|)
        act = torch.tanh(local_target)

        # 3. Context gating
        if context is not None:
            gate_logit = F.linear(context, self.gate_w.unsqueeze(0), self.gate_b)
            gate = torch.sigmoid(gate_logit)
            act = act * gate

        # 4. Scale by residual gain alpha_g
        scaled = self.alpha * act

        # 5. Scatter to target neurons P_T
        out = torch.zeros_like(full_rates)
        out[..., self.target_idx] = scaled
        return out

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
