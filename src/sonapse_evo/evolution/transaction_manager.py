from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import copy
import uuid
import time
import torch
import torch.nn as nn

from ..contracts.graft_manifest import GraftSpec
from ..contracts.receipts import TransactionReceipt, EvaluationResult
from ..contracts.snapshots import SubstrateSnapshot
from ..substrate.recurrent_runtime import RecurrentSubstrate
from ..grafts.graft_registry import GraftRegistry
from .lifecycle import CandidateLifecycle, CandidateState


class FailureArchive:
    """Immutable audit archive recording rejected and rolled-back candidates."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record_failure(self, candidate_id: str, reason: str, details: Dict[str, Any]) -> None:
        self.records.append({
            "candidate_id": candidate_id,
            "timestamp": time.time(),
            "reason": reason,
            "details": details,
        })


class TransactionManager:
    """Coordinates atomic, verified installation and rollback of neural grafts."""

    def __init__(
        self,
        substrate: RecurrentSubstrate,
        registry: GraftRegistry,
        failure_archive: Optional[FailureArchive] = None,
    ):
        self.substrate = substrate
        self.registry = registry
        self.failure_archive = failure_archive or FailureArchive()
        self.rollback_checkpoints: Dict[str, SubstrateSnapshot] = {}
        self.receipts: List[TransactionReceipt] = []

    def commit_graft(
        self,
        graft_module: nn.Module,
        spec: GraftSpec,
        eval_result: EvaluationResult,
        lifecycle: Optional[CandidateLifecycle] = None,
    ) -> TransactionReceipt:
        """Executes atomic graft installation transaction."""
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"

        # 1. Pre-condition checks: Verify parent topology digest
        current_digest = self.substrate.topology.compute_digest()
        if spec.parent_digest and spec.parent_digest != current_digest:
            reason = f"Parent digest mismatch: expected {spec.parent_digest}, got {current_digest}"
            self.failure_archive.record_failure(spec.graft_id, reason, {"tx_id": tx_id})
            return TransactionReceipt(
                transaction_id=tx_id,
                status="ROLLED_BACK",
                graft_id=spec.graft_id,
                parent_digest=current_digest,
                committed_digest=current_digest,
                reason=reason,
            )

        # 2. Verify port bounds
        spec.validate(self.substrate.total_neurons)

        # 3. Create pre-commit rollback snapshot
        pre_snapshot = self.substrate.create_snapshot()
        self.rollback_checkpoints[spec.graft_id] = pre_snapshot

        # 4. Exact zero-effect initialisation test
        if hasattr(graft_module, "reset_ramp"):
            graft_module.reset_ramp(start_gain=0.0)

        with torch.no_grad():
            dummy_rates = torch.rand(self.substrate.total_neurons, dtype=torch.float32)
            init_current = graft_module(dummy_rates.unsqueeze(0)).squeeze(0)
            max_dev = float(torch.max(torch.abs(init_current)).item())

            if max_dev > 1e-6:
                reason = f"Zero-gain initialization violation: max current dev {max_dev:.6f} > 0"
                self.failure_archive.record_failure(spec.graft_id, reason, {"tx_id": tx_id})
                return TransactionReceipt(
                    transaction_id=tx_id,
                    status="ROLLED_BACK",
                    graft_id=spec.graft_id,
                    parent_digest=current_digest,
                    committed_digest=current_digest,
                    reason=reason,
                )

        # 5. Atomic installation into registry
        try:
            self.registry.register_graft(graft_module, spec)
            self.substrate.active_grafts[spec.graft_id] = {
                "graft_type": spec.graft_type,
                "rank": spec.rank,
                "installed_at": time.time(),
            }
            if lifecycle:
                lifecycle.transition_to(CandidateState.ACTIVE, reason="Committed in zero-gain state")
                lifecycle.transition_to(CandidateState.MONITORING, reason="Initiating activation ramp")

            receipt = TransactionReceipt(
                transaction_id=tx_id,
                status="COMMITTED",
                graft_id=spec.graft_id,
                parent_digest=current_digest,
                committed_digest=self.substrate.create_snapshot().state_digest,
                reason="Successfully installed under atomic contract",
                metrics={
                    "causal_tau": eval_result.causal_tau,
                    "recovery_ratio": eval_result.recovery_ratio,
                    "parameters": graft_module.get_parameter_count() if hasattr(graft_module, "get_parameter_count") else 0,
                },
            )
            self.receipts.append(receipt)
            return receipt

        except Exception as e:
            # Clean rollback
            self.rollback_graft(spec.graft_id, reason=f"Installation exception: {str(e)}")
            return TransactionReceipt(
                transaction_id=tx_id,
                status="ROLLED_BACK",
                graft_id=spec.graft_id,
                parent_digest=current_digest,
                committed_digest=current_digest,
                reason=f"Exception during registration: {str(e)}",
            )

    def rollback_graft(self, graft_id: str, reason: str = "") -> None:
        """Restores pre-commit state and deregisters graft."""
        if graft_id in self.rollback_checkpoints:
            snapshot = self.rollback_checkpoints[graft_id]
            self.substrate.restore_snapshot(snapshot)
            del self.rollback_checkpoints[graft_id]

        self.registry.unregister_graft(graft_id)
        if graft_id in self.substrate.active_grafts:
            del self.substrate.active_grafts[graft_id]

        self.failure_archive.record_failure(graft_id, reason, {"action": "rollback"})
