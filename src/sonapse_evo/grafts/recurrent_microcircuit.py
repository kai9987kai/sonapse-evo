from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts.graft_manifest import GraftSpec


class RecurrentMicrocircuit(nn.Module):
    """Type B Graft: Recurrent Microcircuit Graft with Internal Memory State q_t.
    
    q_{t+1} = (1 - nu) * q_t + nu * tanh(A_g * q_t + B_g * r_{S,t} + b_q)
    I_g(t) = alpha_g(t) * g_g(c_t) * P_T tanh(C_g * q_{t+1} + b_c)
    """

    def __init__(
        self,
        spec: GraftSpec,
        total_neurons: int,
        context_dim: int = 64,
        leak_nu: float = 0.40,
    ):
        super().__init__()
        self.spec = spec
        self.total_neurons = total_neurons
        self.context_dim = context_dim
        self.source_width = len(spec.source_neurons)
        self.target_width = len(spec.target_neurons)
        self.state_dim = spec.recurrent_state_dim
        self.max_gain = spec.max_gain
        self.ramp_steps = spec.ramp_steps
        self.leak_nu = float(leak_nu)

        self.register_buffer("source_idx", torch.as_tensor(spec.source_neurons, dtype=torch.long))
        self.register_buffer("target_idx", torch.as_tensor(spec.target_neurons, dtype=torch.long))

        # Weight matrices
        self.B_g = nn.Parameter(torch.empty(self.state_dim, self.source_width))
        self.A_g = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
        self.C_g = nn.Parameter(torch.empty(self.target_width, self.state_dim))
        self.bias_q = nn.Parameter(torch.zeros(self.state_dim))
        self.bias_c = nn.Parameter(torch.zeros(self.target_width))

        # Context gate
        self.gate_w = nn.Parameter(torch.empty(self.context_dim))
        self.gate_b = nn.Parameter(torch.zeros(1))

        # Internal state buffer
        self.register_buffer("q", torch.zeros(self.state_dim, dtype=torch.float32))

        self.alpha = 0.0
        self.current_step = 0
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.B_g, a=np.sqrt(5))
        nn.init.orthogonal_(self.A_g, gain=0.8)
        nn.init.kaiming_uniform_(self.C_g, a=np.sqrt(5))
        nn.init.zeros_(self.gate_w)
        nn.init.constant_(self.gate_b, 2.0)

    def reset_state(self):
        self.q.zero_()

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
        if self.alpha <= 0.0 and not self.training:
            return torch.zeros_like(full_rates)

        r_S = full_rates[..., self.source_idx]  # (..., |S|)

        # Advance internal recurrent state
        # A_g * q_t + B_g * r_{S,t} + b_q
        q_in = F.linear(self.q, self.A_g) + F.linear(r_S, self.B_g, self.bias_q)
        q_act = torch.tanh(q_in)
        new_q = (1.0 - self.leak_nu) * self.q + self.leak_nu * q_act
        self.q = new_q.detach() if not self.training else new_q

        # Output projection
        out_local = F.linear(new_q, self.C_g, self.bias_c)
        act = torch.tanh(out_local)

        if context is not None:
            gate_logit = F.linear(context, self.gate_w.unsqueeze(0), self.gate_b)
            gate = torch.sigmoid(gate_logit)
            act = act * gate

        scaled = self.alpha * act

        out = torch.zeros_like(full_rates)
        out[..., self.target_idx] = scaled
        return out

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
