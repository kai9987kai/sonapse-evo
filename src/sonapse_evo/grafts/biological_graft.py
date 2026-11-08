from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts.graft_manifest import GraftSpec


class BiologicalMotifGraft(nn.Module):
    """Type C Graft: Biological Motif Graft with Signed Connectome Structure.
    
    q_{t+1} = phi(W_m * q_t + B_m * r_{S,t})
    I_g(t) = alpha_g(t) * P_T C_m * q_{t+1}
    W_m obeys Dale's Law: W_ij = s_j * a_ij with a_ij >= 0, s_j in {-1, +1}.
    """

    def __init__(
        self,
        spec: GraftSpec,
        total_neurons: int,
        motif_nodes: int = 16,
        excitatory_ratio: float = 0.8,
        freeze_motif: bool = True,
    ):
        super().__init__()
        self.spec = spec
        self.total_neurons = total_neurons
        self.source_width = len(spec.source_neurons)
        self.target_width = len(spec.target_neurons)
        self.motif_nodes = int(motif_nodes)
        self.max_gain = spec.max_gain
        self.ramp_steps = spec.ramp_steps
        self.freeze_motif = freeze_motif

        self.register_buffer("source_idx", torch.as_tensor(spec.source_neurons, dtype=torch.long))
        self.register_buffer("target_idx", torch.as_tensor(spec.target_neurons, dtype=torch.long))

        # Dale's law signs: 80% excitatory (+1), 20% inhibitory (-1)
        rng = np.random.default_rng(spec.metadata.get("seed", 42))
        signs = rng.choice([1.0, -1.0], size=self.motif_nodes, p=[excitatory_ratio, 1.0 - excitatory_ratio])
        self.register_buffer("dale_signs", torch.as_tensor(signs, dtype=torch.float32))

        # Magnitude parameter a_ij >= 0 (enforced via softplus or relu)
        raw_magnitudes = rng.uniform(0.05, 0.3, size=(self.motif_nodes, self.motif_nodes)).astype(np.float32)
        np.fill_diagonal(raw_magnitudes, 0.0)  # no direct self-excitation
        if self.freeze_motif:
            self.register_buffer("raw_magnitudes", torch.as_tensor(raw_magnitudes, dtype=torch.float32))
        else:
            self.raw_magnitudes = nn.Parameter(torch.as_tensor(raw_magnitudes, dtype=torch.float32))

        # Trainable input and output adapters
        self.B_m = nn.Parameter(torch.empty(self.motif_nodes, self.source_width))
        self.C_m = nn.Parameter(torch.empty(self.target_width, self.motif_nodes))
        self.bias_q = nn.Parameter(torch.zeros(self.motif_nodes))
        self.bias_c = nn.Parameter(torch.zeros(self.target_width))

        # Motif state buffer
        self.register_buffer("q", torch.zeros(self.motif_nodes, dtype=torch.float32))

        self.alpha = 0.0
        self.current_step = 0
        self._init_adapters()

    def _init_adapters(self):
        nn.init.kaiming_uniform_(self.B_m, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.C_m, a=np.sqrt(5))

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

    @property
    def signed_motif_matrix(self) -> torch.Tensor:
        # Enforce non-negativity of magnitudes: a_ij >= 0
        mags = F.relu(self.raw_magnitudes)
        # Column j has sign s_j
        return mags * self.dale_signs.unsqueeze(0)

    def forward(
        self,
        full_rates: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.alpha <= 0.0 and not self.training:
            return torch.zeros_like(full_rates)

        r_S = full_rates[..., self.source_idx]

        # W_m * q_t + B_m * r_{S,t} + bias_q
        w_m = self.signed_motif_matrix
        motif_rec = F.linear(self.q, w_m)
        motif_in = motif_rec + F.linear(r_S, self.B_m, self.bias_q)
        new_q = F.relu(torch.tanh(motif_in))
        self.q = new_q.detach() if not self.training else new_q

        # Output adapter: C_m * q_{t+1} + bias_c
        out_local = F.linear(new_q, self.C_m, self.bias_c)
        act = torch.tanh(out_local)

        scaled = self.alpha * act
        out = torch.zeros_like(full_rates)
        out[..., self.target_idx] = scaled
        return out

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
