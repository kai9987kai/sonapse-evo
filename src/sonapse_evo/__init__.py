"""SONAPSE-EVO: Autonomous Connectome Discovery, Structural Evolution, Causal Repair and Continual Adaptation.

Research Architecture — Version 2.0
"""

__version__ = "2.0.0"

from .contracts.observations import ObservationSpec, SensoryObservation, ActionCommand
from .contracts.graft_manifest import GraftSpec, GraftType
from .contracts.snapshots import SubstrateSnapshot
from .contracts.receipts import EvaluationResult, TransactionReceipt

__all__ = [
    "__version__",
    "ObservationSpec",
    "SensoryObservation",
    "ActionCommand",
    "GraftSpec",
    "GraftType",
    "SubstrateSnapshot",
    "EvaluationResult",
    "TransactionReceipt",
]
