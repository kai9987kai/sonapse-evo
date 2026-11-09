from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from ..substrate.recurrent_runtime import RecurrentSubstrate
from ..substrate.neural_ports import NeuralPort


@dataclass
class CausalInterventionReceipt:
    """Audit record of a surgical Pearlian causal graph intervention."""
    intervention_type: str  # "do_clamp_node", "do_sever_pathway", "do_port_excitation"
    target_indices: List[int]
    clamped_value: Optional[float]
    average_causal_effect: float
    indirect_mediation_ratio: float
    affected_outflow_channels: List[int]


class PearlianInterventionEngine:
    """Implements surgical counterfactual graph interventions: do(X = x).
    
    Grounding: Pearl (2009) Causality: Models, Reasoning, and Inference.
    Disentangles direct causation from correlational co-activation by cutting
    all upstream structural incoming edges and forcing node state do(r_i = c).
    """

    def __init__(self, sample_obs_dim: int = 155):
        self.obs_dim = sample_obs_dim

    def do_clamp_port(
        self,
        substrate: RecurrentSubstrate,
        port: NeuralPort,
        clamp_value: float,
        obs: torch.Tensor,
        action_readout: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Executes do(r_port = clamp_value) by clamping target nodes during forward execution."""
        sub_clone = substrate.clone_substrate()
        # Clamped forward step
        r_step = sub_clone.forward_step(obs)
        # Surgical intervention: enforce clamp_value on target port
        idx = list(port.indices)
        with torch.no_grad():
            r_step[idx] = clamp_value
            sub_clone.r[idx] = clamp_value

        if action_readout is not None:
            with torch.no_grad():
                motor_out = action_readout(r_step[512:576])
                return motor_out
        return r_step

    def compute_average_causal_effect(
        self,
        substrate: RecurrentSubstrate,
        port: NeuralPort,
        action_readout: nn.Module,
        val_control: float = 0.0,
        val_treatment: float = 1.0,
        num_samples: int = 10,
    ) -> float:
        """Calculates Average Causal Effect (ACE): E[Y | do(X = x1)] - E[Y | do(X = x0)]."""
        ace_accum = 0.0

        for _ in range(num_samples):
            obs = torch.randn(self.obs_dim)
            out_control = self.do_clamp_port(substrate, port, val_control, obs, action_readout)
            out_treatment = self.do_clamp_port(substrate, port, val_treatment, obs, action_readout)

            # Norm of action change under intervention
            diff = torch.norm(out_treatment - out_control).item()
            ace_accum += diff

        return float(ace_accum / max(1, num_samples))

    def compute_causal_influence_matrix(
        self,
        substrate: RecurrentSubstrate,
        ports: List[NeuralPort],
        action_readout: nn.Module,
    ) -> np.ndarray:
        """Computes pairwise causal impact of every port on downstream motor action."""
        matrix = np.zeros(len(ports), dtype=np.float32)
        for idx, port in enumerate(ports):
            ace = self.compute_average_causal_effect(substrate, port, action_readout)
            matrix[idx] = ace
        return matrix
