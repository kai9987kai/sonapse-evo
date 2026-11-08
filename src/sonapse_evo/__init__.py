"""SONAPSE-EVO: Causal Neural Circuit Evolution, Autonomous Repair and Adaptive Intelligence.

Proposed Research Architecture — Version 1.0
"""

__version__ = "1.0.0"

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
