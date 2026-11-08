from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import torch


@dataclass(frozen=True)
class ObservationSpec:
    sensory_dim: int = 155
    context_dim: int = 64
    action_dim: int = 5
    value_range: tuple[float, float] = (-10.0, 10.0)
    dtype: str = "float32"

    def validate_sensory(self, data: np.ndarray | torch.Tensor) -> bool:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return data.shape[-1] == self.sensory_dim and np.all(np.isfinite(data))


@dataclass(frozen=True)
class SensoryObservation:
    data: torch.Tensor
    step_idx: int = 0
    context: Optional[torch.Tensor] = None
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.data, torch.Tensor):
            object.__setattr__(self, "data", torch.as_tensor(self.data, dtype=torch.float32))
        if self.context is not None and not isinstance(self.context, torch.Tensor):
            object.__setattr__(self, "context", torch.as_tensor(self.context, dtype=torch.float32))


@dataclass(frozen=True)
class ActionCommand:
    actions: torch.Tensor
    raw_logits: Optional[torch.Tensor] = None

    def __post_init__(self):
        if not isinstance(self.actions, torch.Tensor):
            tensor_actions = torch.as_tensor(self.actions, dtype=torch.float32)
        else:
            tensor_actions = self.actions.to(dtype=torch.float32)
        clamped = torch.clamp(tensor_actions, -1.0, 1.0)
        object.__setattr__(self, "actions", clamped)

    def to_numpy(self) -> np.ndarray:
        return self.actions.detach().cpu().numpy()
