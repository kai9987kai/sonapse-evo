"""Atomic promotion and rollback of experimental mechanisms.

Nothing in the NeuroWeave layer is enabled by default, and promotion never changes that: the package
default stays the ``legacy`` profile.  Promotion maintains a separate, explicit *stable set* of mechanisms
that have earned their place with evidence.  ``PromotionLedger.propose`` promotes a set of mechanisms only
if every gate passes:

    evidence_tier   the experiment receipt is PAIRED_EXPERIMENT or CONFIRMATORY_RESULT (existing taxonomy)
    verdict         the receipt's verdict is ``supported``
    no_faults       no mechanism fault occurred in any arm
    compatibility   the all-off identity check passes (legacy behaviour is intact)
    cost            measured candidate / baseline cost ratio is within ``max_cost_ratio``

The stable set changes atomically (all gates or nothing), the previous stable set is kept for
``rollback()``, and every rejected proposal is archived with its receipt digest and reasons - failed
experiments are evidence too and are never deleted.  The ledger can be persisted to JSON with an atomic
write (temporary file, then replace).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..contracts.evidence import is_promotable_evidence
from ..evolution.transaction_manager import FailureArchive
from .compat import verify_legacy_identity
from .registry import MECHANISMS, WeaveConfig

LEDGER_FORMAT = "SONAPSE_PROMOTION_LEDGER"


class PromotionLedger:
    def __init__(self, path: Optional[str] = None, max_cost_ratio: float = 25.0, failure_archive: Optional[FailureArchive] = None):
        self.path = Path(path) if path else None
        self.max_cost_ratio = float(max_cost_ratio)
        self.stable: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.failure_archive = failure_archive or FailureArchive()
        if self.path is not None and self.path.exists():
            self.load()

    # ----------------------------------------------------------------- gates
    def evaluate_gates(self, mechanism_ids: Iterable[str], receipt: Dict[str, Any],
                       regression: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        regression = regression if regression is not None else verify_legacy_identity()
        summary = receipt.get("results", {}).get("summary", {})
        faults = sum(int(arm.get("faults", 0)) for arm in summary.values())
        cost = float(receipt.get("matched_conditions", {}).get("candidate_cost_ratio_measured", float("inf")))
        candidate = set(receipt.get("spec", {}).get("candidate", []))
        return {
            "known_mechanisms": all(m in MECHANISMS for m in mechanism_ids),
            "receipt_covers_mechanisms": set(mechanism_ids) <= candidate,
            "evidence_tier": is_promotable_evidence(receipt.get("evidence_tier", "")),
            "verdict": receipt.get("verdict") == "supported",
            "no_faults": faults == 0,
            "compatibility": bool(regression.get("legacy_identity")) and bool(regression.get("neutral_extended_path_identity", True)),
            "cost": cost <= self.max_cost_ratio,
        }

    # ------------------------------------------------------------- decisions
    def _commit(self, stable: List[str], history: List[Dict[str, Any]], decisions: List[Dict[str, Any]],
                failures: List[Dict[str, Any]]) -> None:
        """Persist the new ledger first, then adopt it in memory.  A failed write changes nothing."""
        self.save(stable=stable, history=history, decisions=decisions, failures=failures)
        self.stable, self.history, self.decisions = stable, history, decisions
        self.failure_archive.records = failures

    def propose(self, mechanism_ids: Iterable[str], receipt: Dict[str, Any], regression: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Promote ``mechanism_ids`` if every gate passes; otherwise archive the failure.  Atomic either way."""
        ids = [m for m in dict.fromkeys(mechanism_ids)]
        gates = self.evaluate_gates(ids, receipt, regression)
        decision = {"mechanisms": ids, "gates": gates, "experiment_id": receipt.get("experiment_id"),
                    "result_digest": receipt.get("result_digest"), "evidence_tier": receipt.get("evidence_tier"),
                    "verdict": receipt.get("verdict"), "timestamp": time.time()}
        stable, history, failures = list(self.stable), list(self.history), list(self.failure_archive.records)
        if all(gates.values()):
            history.append({"stable": list(stable), "replaced_at": decision["timestamp"], "by": receipt.get("experiment_id")})
            stable = [m for m in dict.fromkeys(stable + ids)]
            decision["status"] = "PROMOTED"
        else:
            failed = [name for name, ok in gates.items() if not ok]
            decision["status"] = "REJECTED"
            decision["failed_gates"] = failed
            failures.append({"candidate_id": "+".join(ids) or "<none>", "timestamp": decision["timestamp"],
                             "reason": "promotion gates failed: " + ", ".join(failed),
                             "details": {"experiment_id": receipt.get("experiment_id"), "result_digest": receipt.get("result_digest"),
                                         "gates": gates}})
        self._commit(stable, history, list(self.decisions) + [decision], failures)
        return decision

    def rollback(self) -> Dict[str, Any]:
        """Restore the previous stable set (atomic).  A no-op report is returned when there is nothing to undo."""
        if not self.history:
            return {"status": "NOTHING_TO_ROLL_BACK", "stable": list(self.stable)}
        entry = self.history[-1]
        previous = entry.get("stable") if isinstance(entry, dict) else None
        if not isinstance(previous, list) or not all(m in MECHANISMS for m in previous):
            raise ValueError("the last history entry of the promotion ledger is corrupted; nothing was changed")
        report = {"status": "ROLLED_BACK", "from": list(self.stable), "stable": list(previous), "timestamp": time.time()}
        self._commit(list(previous), list(self.history[:-1]), list(self.decisions) + [report], list(self.failure_archive.records))
        return report

    def stable_config(self, seed: int = 42, **kwargs: Any) -> WeaveConfig:
        """A ``WeaveConfig`` with exactly the promoted mechanisms enabled (opt-in; never the package default)."""
        config = WeaveConfig.with_mechanisms(*self.stable, seed=seed, **kwargs)
        config.profile = "promoted" if self.stable else "legacy"
        return config

    def failures(self) -> List[Dict[str, Any]]:
        return list(self.failure_archive.records)

    # ----------------------------------------------------------- persistence
    def to_dict(self, stable=None, history=None, decisions=None, failures=None) -> Dict[str, Any]:
        return {"format": LEDGER_FORMAT, "version": "1.0", "stable": list(self.stable if stable is None else stable),
                "history": list(self.history if history is None else history),
                "decisions": list(self.decisions if decisions is None else decisions),
                "failures": list(self.failures() if failures is None else failures), "max_cost_ratio": self.max_cost_ratio,
                "default_profile": "legacy"}

    def save(self, **pending: Any) -> None:
        """Atomic write (temporary file, then replace).  ``pending`` saves a not-yet-adopted state."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(**pending), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("format") != LEDGER_FORMAT:
            raise ValueError("not a promotion ledger file")
        self.stable = [m for m in data.get("stable", []) if m in MECHANISMS]
        self.history = [h for h in data.get("history", []) if isinstance(h, dict) and isinstance(h.get("stable"), list)]
        self.decisions = list(data.get("decisions", []))
        self.failure_archive.records = list(data.get("failures", []))
        self.max_cost_ratio = float(data.get("max_cost_ratio", self.max_cost_ratio))


__all__ = ["PromotionLedger", "LEDGER_FORMAT"]
