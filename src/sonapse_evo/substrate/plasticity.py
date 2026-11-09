from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import numpy as np


@dataclass
class NeuromodulatoryState:
    """Dynamic neuromodulatory chemical field governing learning rates and plasticity."""
    dopamine_rpe: float = 0.0     # Reward prediction error (DA)
    acetylcholine_unc: float = 0.0 # Unexpected uncertainty / novelty (ACh)
    serotonin_risk: float = 0.0    # Behavioral inhibition / punishment (5-HT)
    noradrenaline_gain: float = 1.0 # Global arousal / signal-to-noise (NA)

    def compute_plasticity_multiplier(self) -> float:
        """Computes aggregate scalar gating multiplier M(t)."""
        # DA promotes consolidation of beneficial actions; ACh expands learning window
        m = (0.7 * self.dopamine_rpe + 0.3 * self.acetylcholine_unc) * self.noradrenaline_gain
        return float(np.clip(m, -2.0, 2.0))


class NeuromodulatedPlasticityEngine:
    """Implements biological three-factor synaptic plasticity with Dale's law constraints.
    
    Update Equation:
        dw_ij/dt = eta * M(t) * e_ij(t) - gamma * w_ij(t) - kappa * (r_j - target) * w_ij(t)
        
    Where:
        - e_ij: Synaptic eligibility trace tracking pre-post coincidence.
        - M(t): Neuromodulatory field combining DA, ACh, and 5-HT.
        - Dale's Law: Strictly preserves signs of excitatory/inhibitory projections.
        - Homeostatic Norm: Bounds total dendritic synaptic input.
    """

    def __init__(
        self,
        num_neurons: int = 768,
        eta: float = 0.005,
        tau_eligibility: float = 20.0,
        decay_rate: float = 0.0005,
        target_rate: float = 0.20,
        homeostatic_gain: float = 0.001,
        max_weight_bound: float = 1.50,
    ):
        self.num_neurons = num_neurons
        self.eta = eta
        self.lambda_e = float(np.exp(-1.0 / max(1.0, tau_eligibility)))
        self.decay = decay_rate
        self.target_rate = target_rate
        self.kappa = homeostatic_gain
        self.max_bound = max_weight_bound

        # Sparse plastic weight matrix and eligibility traces
        self.eligibility = torch.zeros((num_neurons, num_neurons), dtype=torch.float32)
        self.plastic_w = torch.zeros((num_neurons, num_neurons), dtype=torch.float32)
        self.neuron_signs = torch.ones(num_neurons, dtype=torch.float32)
        # 20% inhibitory interneurons by default
        self.neuron_signs[::5] = -1.0

        self.modulators = NeuromodulatoryState()
        self.update_count = 0

    def update_eligibility(self, pre_rates: torch.Tensor, post_rates: torch.Tensor) -> torch.Tensor:
        """Updates eligibility trace: e_ij(t) = lambda * e_ij(t-1) + pre_i * (post_j - mean_post)."""
        centered_post = post_rates - torch.mean(post_rates)
        outer_coincidence = torch.outer(pre_rates, centered_post)
        self.eligibility = self.lambda_e * self.eligibility + (1.0 - self.lambda_e) * outer_coincidence
        return self.eligibility

    def step_plasticity(
        self,
        pre_rates: torch.Tensor,
        post_rates: torch.Tensor,
        reward: float = 0.0,
        uncertainty: float = 0.0,
    ) -> Dict[str, float]:
        """Applies three-factor neuromodulated plasticity update under Dale's law."""
        self.update_count += 1

        # 1. Update neuromodulatory state
        # Dopamine RPE: reward - baseline
        self.modulators.dopamine_rpe = float(reward)
        self.modulators.acetylcholine_unc = float(uncertainty)
        m_factor = self.modulators.compute_plasticity_multiplier()

        # 2. Update eligibility trace
        self.update_eligibility(pre_rates, post_rates)

        # 3. Three-factor delta calculation
        dw = self.eta * m_factor * self.eligibility

        # 4. Decay and Homeostatic term
        homeo = self.kappa * (post_rates - self.target_rate).unsqueeze(0)
        self.plastic_w = self.plastic_w + dw - (self.decay * self.plastic_w) - (homeo * self.plastic_w)

        # 5. Strict Dale's Law Enforcement: Sign(w_ij) == Sign(pre_i)
        # Signs matrix where row i matches neuron_signs[i]
        signs_mat = self.neuron_signs.unsqueeze(1).expand_as(self.plastic_w)
        # Excitatory nodes must remain >= 0; inhibitory nodes <= 0
        self.plastic_w = torch.where(
            signs_mat > 0,
            torch.clamp(self.plastic_w, min=0.0, max=self.max_bound),
            torch.clamp(self.plastic_w, min=-self.max_bound, max=0.0),
        )

        # 6. Homeostatic Fan-in bound: sum_i |w_ij| <= 2.0
        col_norms = torch.sum(torch.abs(self.plastic_w), dim=0, keepdim=True)
        scale = torch.clamp(col_norms / 2.0, min=1.0)
        self.plastic_w = self.plastic_w / scale

        mean_delta = float(torch.mean(torch.abs(dw)).item())
        weight_norm = float(torch.norm(self.plastic_w).item())

        return {
            "mean_delta": mean_delta,
            "weight_norm": weight_norm,
            "neuromodulator_m": m_factor,
            "dopamine_rpe": self.modulators.dopamine_rpe,
            "acetylcholine_unc": self.modulators.acetylcholine_unc,
        }

    def compute_plastic_current(self, rates: torch.Tensor) -> torch.Tensor:
        """Computes auxiliary input current generated by plastic connections: I_plastic = W_p * r."""
        return torch.matmul(self.plastic_w.t(), rates)

    def reset_traces(self) -> None:
        """Clears eligibility traces and zero-centers modulators."""
        self.eligibility.zero_()
        self.modulators = NeuromodulatoryState()
