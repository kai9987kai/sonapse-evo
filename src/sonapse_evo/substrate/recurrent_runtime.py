from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology_state import TopologyState
from .lesions import LesionSpec, LesionEngine
from ..contracts.snapshots import SubstrateSnapshot, compute_tensor_digest


class RecurrentSubstrate(nn.Module):
    """Dynamical recurrent connectome substrate with causal lesion support and auxiliary current injection."""

    def __init__(
        self,
        topology: TopologyState,
        sensory_dim: int = 155,
        leak: float = 0.35,
        microsteps: int = 2,
        current_cap: float = 0.10,
        seed: int = 42,
    ):
        super().__init__()
        self.topology = topology
        self.total_neurons = topology.total_neurons
        self.sensory_dim = sensory_dim
        self.leak = float(leak)
        self.microsteps = int(microsteps)
        self.current_cap = float(current_cap)
        self.seed = int(seed)

        # Base connectivity buffers
        self.register_buffer("base_pre", torch.as_tensor(topology.base_pre, dtype=torch.long))
        self.register_buffer("base_post", torch.as_tensor(topology.base_post, dtype=torch.long))
        self.register_buffer("base_w", torch.as_tensor(topology.base_w, dtype=torch.float32))

        # Lesion mask buffer (1.0 = active, 0.0 = lesioned)
        m = len(topology.base_pre)
        self.register_buffer("lesion_mask", torch.ones(m, dtype=torch.float32))

        # Plastic connectivity (sparse indices and weights)
        self.register_buffer("plastic_pre", torch.zeros(0, dtype=torch.long))
        self.register_buffer("plastic_post", torch.zeros(0, dtype=torch.long))
        self.register_buffer("plastic_w", torch.zeros(0, dtype=torch.float32))

        # Sensory projection matrix B
        rng = np.random.default_rng(seed)
        b_init = rng.normal(0.0, 0.05, size=(self.total_neurons, sensory_dim)).astype(np.float32)
        # Sparse sensory inputs into first 25% of neurons
        sensory_targets = self.total_neurons // 4
        b_init[sensory_targets:] = 0.0
        # Canonical sensorimotor alignment (nodes 0, 1, 2)
        b_init[0, 2] = 2.5   # Drive (cos heading)
        b_init[1, 1] = 2.5   # Turn Left (sin heading)
        b_init[2, 1] = -2.5  # Turn Right (-sin heading)
        self.register_buffer("B", torch.as_tensor(b_init, dtype=torch.float32))

        # Biases and rate states
        bias_init = rng.normal(0.02, 0.01, size=self.total_neurons).astype(np.float32)
        self.register_buffer("bias", torch.as_tensor(bias_init, dtype=torch.float32))
        self.register_buffer("r", torch.zeros(self.total_neurons, dtype=torch.float32))

        # Tracking and metadata
        self.step_count = 0
        self.last_jacobian_norm = 0.0
        self.active_grafts: Dict[str, Any] = {}

    def activation_phi(self, x: torch.Tensor) -> torch.Tensor:
        """phi(x) = max(0, tanh(x))"""
        return F.relu(torch.tanh(x))

    def d_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of phi(x): (1 - tanh^2(x)) * (x > 0)"""
        t = torch.tanh(x)
        dt = 1.0 - t * t
        return dt * (x > 0.0).float()

    def reset_state(self) -> None:
        self.r.zero_()
        self.step_count = 0

    def apply_lesion(self, spec: LesionSpec) -> None:
        """Applies a controlled lesion mask."""
        engine = LesionEngine(seed=spec.seed)
        mask = engine.create_lesion_mask(self.topology, spec)
        self.lesion_mask.copy_(torch.as_tensor(mask, dtype=torch.float32, device=self.lesion_mask.device))

    def clear_lesion(self) -> None:
        self.lesion_mask.fill_(1.0)

    def compute_recurrent_input(self, r: torch.Tensor) -> torch.Tensor:
        """Computes (W_b^{(l)} + W_p) * r using sparse scatter_add."""
        # Active base edges
        active_w = self.base_w * self.lesion_mask
        pre_acts = r[..., self.base_pre]
        transmitted = pre_acts * active_w

        rec_input = torch.zeros_like(r)
        if len(self.base_post) > 0:
            rec_input.scatter_add_(-1, self.base_post, transmitted)

        # Plastic edges if any
        if len(self.plastic_w) > 0 and len(self.plastic_post) > 0:
            p_acts = r[..., self.plastic_pre] * self.plastic_w
            rec_input.scatter_add_(-1, self.plastic_post, p_acts)

        return rec_input

    def microstep(
        self,
        observation: torch.Tensor,
        aux_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes a single continuous rate microstep."""
        # Compute sensory projection
        if observation.dim() == 1:
            sens_in = torch.matmul(self.B, observation)
        else:
            sens_in = torch.matmul(observation, self.B.t())

        # Bound total auxiliary current: ||I||_inf <= B
        if aux_current is not None:
            clamped_aux = torch.clamp(aux_current, -self.current_cap, self.current_cap)
        else:
            clamped_aux = torch.zeros_like(self.r)

        rec_in = self.compute_recurrent_input(self.r)
        drive = rec_in + sens_in + clamped_aux + self.bias
        act = self.activation_phi(drive)

        new_r = (1.0 - self.leak) * self.r + self.leak * act
        self.r = torch.clamp(new_r, 0.0, 1.0)
        return self.r

    def forward_step(
        self,
        observation: torch.Tensor,
        aux_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes the specified number of microsteps for one environment transition."""
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.r.device)
        aux_tensor = (
            torch.as_tensor(aux_current, dtype=torch.float32, device=self.r.device)
            if aux_current is not None
            else None
        )

        for _ in range(self.microsteps):
            self.microstep(obs_tensor, aux_tensor)

        self.step_count += 1
        return self.r

    def estimate_jacobian_norm(self, observation: Optional[torch.Tensor] = None) -> float:
        """Estimates ||J_t||_2 = ||d r_{t+1} / d r_t||_2 using power iteration."""
        with torch.no_grad():
            if observation is None:
                observation = torch.zeros(self.sensory_dim, dtype=torch.float32, device=self.r.device)

            rec_in = self.compute_recurrent_input(self.r)
            sens_in = torch.matmul(self.B, observation)
            drive = rec_in + sens_in + self.bias
            phi_prime = self.d_phi(drive)  # (N,)

            # Operator: v -> (1 - eta) v + eta * phi_prime * (W_b^{(l)} v)
            v = torch.randn(self.total_neurons, dtype=torch.float32, device=self.r.device)
            v = v / (torch.norm(v) + 1e-8)

            active_w = self.base_w * self.lesion_mask

            for _ in range(10):
                # Matrix-vector multiply W * v
                pre_v = v[self.base_pre]
                wv = torch.zeros_like(v)
                wv.scatter_add_(0, self.base_post, pre_v * active_w)
                
                # J * v
                jv = (1.0 - self.leak) * v + self.leak * (phi_prime * wv)
                norm_jv = torch.norm(jv)
                if norm_jv < 1e-12:
                    break
                v = jv / norm_jv

            self.last_jacobian_norm = float(norm_jv.item())
            return self.last_jacobian_norm

    def create_snapshot(self) -> SubstrateSnapshot:
        """Creates an immutable SubstrateSnapshot."""
        rate_copy = self.r.detach().cpu().clone()
        mask_copy = self.lesion_mask.detach().cpu().clone()
        digest = compute_tensor_digest(rate_copy)

        return SubstrateSnapshot(
            step_count=self.step_count,
            total_neurons=self.total_neurons,
            topology_digest=self.topology.compute_digest(),
            state_digest=digest,
            rate_vector=rate_copy,
            lesion_mask=mask_copy,
            active_grafts=copy.deepcopy(self.active_grafts),
            metadata={
                "leak": self.leak,
                "microsteps": self.microsteps,
                "current_cap": self.current_cap,
            },
        )

    def restore_snapshot(self, snapshot: SubstrateSnapshot) -> None:
        """Restores state from an immutable snapshot."""
        if snapshot.total_neurons != self.total_neurons:
            raise ValueError(
                f"Neuron count mismatch: {snapshot.total_neurons} vs {self.total_neurons}"
            )
        self.r.copy_(snapshot.rate_vector.to(self.r.device))
        if snapshot.lesion_mask is not None:
            self.lesion_mask.copy_(snapshot.lesion_mask.to(self.lesion_mask.device))
        self.step_count = snapshot.step_count
        self.active_grafts = copy.deepcopy(snapshot.active_grafts)

    def clone_substrate(self) -> RecurrentSubstrate:
        """Creates an isolated clone with identical state and detached parameters."""
        cloned = RecurrentSubstrate(
            topology=self.topology,
            sensory_dim=self.sensory_dim,
            leak=self.leak,
            microsteps=self.microsteps,
            current_cap=self.current_cap,
            seed=self.seed,
        )
        cloned.base_w.copy_(self.base_w)
        cloned.base_pre.copy_(self.base_pre)
        cloned.base_post.copy_(self.base_post)
        cloned.B.copy_(self.B)
        cloned.bias.copy_(self.bias)
        cloned.restore_snapshot(self.create_snapshot())
        return cloned
