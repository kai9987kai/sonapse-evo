from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..substrate.recurrent_runtime import RecurrentSubstrate
from ..grafts.low_rank_bridge import LowRankBridge
from ..grafts.recurrent_microcircuit import RecurrentMicrocircuit
from ..grafts.biological_graft import BiologicalMotifGraft
from ..grafts.donor_adapter import DonorAdapter
from ..contracts.graft_manifest import GraftSpec


class CandidateTrainer:
    """Isolated sandbox for Stage 1 frozen-backbone graft parameter optimization."""

    def __init__(
        self,
        learning_rate: float = 3e-3,
        lambda_distill: float = 0.50,
        lambda_reg: float = 1e-4,
        lambda_stability: float = 0.10,
        rho_limit: float = 1.05,
    ):
        self.lr = float(learning_rate)
        self.lambda_distill = float(lambda_distill)
        self.lambda_reg = float(lambda_reg)
        self.lambda_stability = float(lambda_stability)
        self.rho_limit = float(rho_limit)

    def instantiate_graft(self, spec: GraftSpec, total_neurons: int) -> nn.Module:
        if spec.graft_type == "low_rank":
            return LowRankBridge(spec, total_neurons)
        elif spec.graft_type == "recurrent":
            return RecurrentMicrocircuit(spec, total_neurons)
        elif spec.graft_type == "biological":
            return BiologicalMotifGraft(spec, total_neurons)
        elif spec.graft_type == "donor_adapter":
            return DonorAdapter(spec, total_neurons)
        else:
            raise ValueError(f"Unknown graft type: {spec.graft_type}")

    def train_candidate(
        self,
        candidate_module: nn.Module,
        isolated_substrate: RecurrentSubstrate,
        training_data: List[Tuple[torch.Tensor, torch.Tensor]],  # list of (observation, target_action)
        retention_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # (obs, ref_action)
        motor_head: Optional[nn.Module] = None,
        steps: int = 64,
        batch_size: int = 16,
    ) -> Dict[str, float]:
        """Trains candidate graft parameters with backbone strictly frozen."""
        if motor_head is None:
            # Default linear readout from first 64 neurons to 5 actions
            motor_head = nn.Linear(64, 5, bias=False)
            nn.init.orthogonal_(motor_head.weight)

        # Freeze substrate and motor head
        isolated_substrate.requires_grad_(False)
        motor_head.requires_grad_(False)

        # Ensure candidate parameters require grad
        candidate_module.requires_grad_(True)
        # Enable nonzero effective gain during training sandbox
        if hasattr(candidate_module, "alpha"):
            candidate_module.alpha = getattr(candidate_module, "max_gain", 0.10)

        optimizer = torch.optim.Adam(candidate_module.parameters(), lr=self.lr)

        loss_history = []
        task_losses = []
        distill_losses = []

        candidate_module.train()

        num_samples = len(training_data)
        if num_samples == 0:
            return {"final_loss": 0.0, "mean_task_loss": 0.0}

        for step in range(steps):
            optimizer.zero_grad()

            # Mini-batch sampling
            indices = np.random.choice(num_samples, size=min(batch_size, num_samples), replace=False)
            batch_loss = torch.tensor(0.0, requires_grad=True)
            batch_task = torch.tensor(0.0)
            batch_distill = torch.tensor(0.0)

            for idx in indices:
                obs, target_action = training_data[idx]

                # Step clone forward with candidate graft
                curr_r = isolated_substrate.r.clone()
                aux_i = candidate_module(curr_r.unsqueeze(0)).squeeze(0)

                # Differentiable forward rate update (1 microstep)
                sens_in = torch.matmul(isolated_substrate.B, obs)
                rec_in = isolated_substrate.compute_recurrent_input(curr_r)
                drive = rec_in + sens_in + aux_i + isolated_substrate.bias
                act = isolated_substrate.activation_phi(drive)
                next_r = (1.0 - isolated_substrate.leak) * curr_r + isolated_substrate.leak * act

                # Motor prediction from first 64 features
                pred_action = torch.tanh(motor_head(next_r[:64]))

                # 1. Task loss
                l_task = F.mse_loss(pred_action, target_action)
                batch_task = batch_task + l_task.detach()

                # 2. Distillation on retention data
                l_distill = torch.tensor(0.0)
                if retention_data and len(retention_data) > 0:
                    ret_idx = np.random.randint(len(retention_data))
                    ret_obs, ref_act = retention_data[ret_idx]
                    sens_ret = torch.matmul(isolated_substrate.B, ret_obs)
                    drive_ret = rec_in + sens_ret + aux_i + isolated_substrate.bias
                    act_ret = isolated_substrate.activation_phi(drive_ret)
                    next_ret_r = (1.0 - isolated_substrate.leak) * curr_r + isolated_substrate.leak * act_ret
                    pred_ret = torch.tanh(motor_head(next_ret_r[:64]))
                    l_distill = F.mse_loss(pred_ret, ref_act)
                    batch_distill = batch_distill + l_distill.detach()

                # 3. Regularization: parameter norm + current norm
                l_reg = sum(torch.sum(p ** 2) for p in candidate_module.parameters()) + torch.mean(aux_i ** 2)

                # 4. Stability penalty
                j_norm = isolated_substrate.estimate_jacobian_norm(obs)
                l_stab = torch.tensor((max(0.0, j_norm - self.rho_limit)) ** 2)

                total_l = l_task + self.lambda_distill * l_distill + self.lambda_reg * l_reg + self.lambda_stability * l_stab
                batch_loss = batch_loss + total_l

            batch_loss = batch_loss / len(indices)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate_module.parameters(), max_norm=1.0)
            optimizer.step()

            loss_history.append(float(batch_loss.item()))
            task_losses.append(float(batch_task.item() / len(indices)))
            distill_losses.append(float(batch_distill.item() / len(indices)))

        # Reset graft gain to zero for safe post-training installation protocol
        if hasattr(candidate_module, "alpha"):
            candidate_module.alpha = 0.0

        candidate_module.eval()

        return {
            "initial_loss": float(loss_history[0]) if loss_history else 0.0,
            "final_loss": float(loss_history[-1]) if loss_history else 0.0,
            "mean_task_loss": float(np.mean(task_losses)) if task_losses else 0.0,
            "mean_distill_loss": float(np.mean(distill_losses)) if distill_losses else 0.0,
        }
