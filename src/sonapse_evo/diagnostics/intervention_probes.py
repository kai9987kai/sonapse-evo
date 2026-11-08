from __future__ import annotations

from typing import List, Tuple, Dict, Any, Sequence
import numpy as np
import torch
import torch.nn as nn

from ..substrate.neural_ports import NeuralPort


class VirtualGradientProbe:
    """Evaluates virtual connection shadow-gradients |d L / d W_{ji}^{virtual}|.
    
    Screens candidate source-target port pairs without physically modifying the live connectome.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def evaluate_shadow_gradients(
        self,
        source_rates: torch.Tensor,  # (B, |S|)
        target_upstream_grad: torch.Tensor,  # (B, |T|)
    ) -> float:
        """Computes the Frobenius norm of virtual cross-weight gradient:
        
        dL / dW_{ji}^{virtual} = target_upstream_grad^T * source_rates
        """
        if source_rates.dim() == 1:
            source_rates = source_rates.unsqueeze(0)
        if target_upstream_grad.dim() == 1:
            target_upstream_grad = target_upstream_grad.unsqueeze(0)

        # Gradient outer product
        grad_W = torch.matmul(target_upstream_grad.t(), source_rates)  # (|T|, |S|)
        score = float(torch.norm(grad_W, p="fro").item())
        return score

    def rank_port_pairs(
        self,
        ports: Sequence[NeuralPort],
        current_rates: torch.Tensor,
        upstream_loss_grad: torch.Tensor,
        top_k: int = 16,
    ) -> List[Tuple[NeuralPort, NeuralPort, float]]:
        """Ranks all directed port pairs (source -> target) by virtual shadow gradient."""
        pairs = []
        for src in ports:
            s_act = src.extract(current_rates)
            for dst in ports:
                if src.port_id == dst.port_id:
                    continue  # skip direct identity self-ports
                d_grad = dst.extract(upstream_loss_grad)
                score = self.evaluate_shadow_gradients(s_act, d_grad)
                pairs.append((src, dst, score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]
