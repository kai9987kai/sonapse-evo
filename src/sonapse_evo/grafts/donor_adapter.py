from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts.graft_manifest import GraftSpec


class DonorAdapter(nn.Module):
    """Type D Graft: External Donor Adapter (e.g. AETHER / Poseidon representation transfer).
    
    z_S = P_theta * z_A
    I_g(t) = alpha_g(t) * P_T * tanh(z_S)
    """

    def __init__(
        self,
        spec: GraftSpec,
        total_neurons: int,
        donor_dim: int = 64,
    ):
        super().__init__()
        self.spec = spec
        self.total_neurons = total_neurons
        self.target_width = len(spec.target_neurons)
        self.donor_dim = donor_dim
        self.max_gain = spec.max_gain
        self.ramp_steps = spec.ramp_steps

        self.register_buffer("target_idx", torch.as_tensor(spec.target_neurons, dtype=torch.long))

        # Trainable projection matrix P_theta
        self.P_proj = nn.Parameter(torch.empty(self.target_width, self.donor_dim))
        self.bias = nn.Parameter(torch.zeros(self.target_width))

        self.alpha = 0.0
        self.current_step = 0
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.P_proj, a=np.sqrt(5))

    def fit_ridge_regression(
        self,
        donor_features: np.ndarray,
        target_activations: np.ndarray,
        reg_lambda: float = 1e-2,
    ) -> float:
        """Solves P^{*T} = (Z_A^T Z_A + lambda * I)^{-1} Z_A^T Z_S in closed form."""
        Z_A = np.asarray(donor_features, dtype=np.float64)  # (M, d_A)
        Z_S = np.asarray(target_activations, dtype=np.float64)  # (M, |T|)

        M, d_A = Z_A.shape
        ATA = Z_A.T @ Z_A + reg_lambda * np.eye(d_A)
        ATB = Z_A.T @ Z_S

        # Solve for P_trans = (d_A, |T|)
        P_trans = np.linalg.solve(ATA, ATB)
        P_opt = P_trans.T  # (|T|, d_A)

        # Compute mean squared error on training features
        pred = Z_A @ P_trans
        mse = float(np.mean((pred - Z_S) ** 2))

        # Copy into PyTorch parameter
        with torch.no_grad():
            self.P_proj.copy_(torch.as_tensor(P_opt, dtype=torch.float32))

        return mse

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
        donor_representation: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.alpha <= 0.0 and not self.training:
            return torch.zeros_like(full_rates)

        if donor_representation is None:
            # Fallback to zero current if no donor features provided
            return torch.zeros_like(full_rates)

        z_S = F.linear(donor_representation, self.P_proj, self.bias)
        act = torch.tanh(z_S)
        scaled = self.alpha * act

        out = torch.zeros_like(full_rates)
        out[..., self.target_idx] = scaled
        return out

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
