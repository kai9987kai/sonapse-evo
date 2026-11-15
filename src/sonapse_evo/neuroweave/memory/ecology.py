"""Memory ecology: working, episodic, semantic and predictive stores with provenance.

Every write carries a :class:`~sonapse_evo.neuroweave.contracts.Provenance`.  Predicted or simulated
information is stored, retrievable and *labelled*; it is never returned by ``observed_only`` retrieval
and can never be marked verified.  When a stored prediction is later confirmed by an actual observation
a new observed record is created that references it.

Timescales   working (few slots, fast decay)  ->  episodic (bounded, prioritised)  ->  semantic (prototypes)
Retrieval    sparse-index candidates, cosine similarity, optional context (affect) similarity
Confidence   raised when a retrieved memory is followed by reward, decayed otherwise
Interference overlap of sparse keys between records whose stored values disagree
Sharing      a memory written while region ``g`` dominated is re-injected at full gain into ``g`` and at
             ``share_gain`` into the other regions (cross-region memory sharing)
Recall current   extra_in += gain * confidence * similarity * (value * region_mask)      (0 with empty stores)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..contracts import Mechanism, Provenance, SignalKind, Stage, StepContext
from .provenance import MemoryRecord, provenance_counts, verify_against
from .stores import MemoryStore, SemanticStore, cosine

_PRED = (Provenance.INTERNAL_PREDICTION,)


class MemoryEcology(Mechanism):
    mechanism_id = "memory_ecology"
    stage = Stage.CONTEXT
    end_stage = Stage.REPORT          # store after the prediction and learning stages have produced this step's results

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"key_dim": 32, "working_capacity": 8, "episodic_capacity": 256, "semantic_capacity": 64,
                "predictive_capacity": 64, "gain": 0.08, "min_similarity": 0.35, "write_every": 8,
                "reward_threshold": 0.3, "working_decay": 0.2, "forget_rate": 0.002, "min_confidence": 0.02,
                "share_gain": 0.25, "verify_tolerance": 0.15, "consolidate_every": 200, "semantic_radius": 0.9}

    def on_attach(self) -> None:
        p, seed = self.params, self.substrate.weave.seed
        kd = int(p["key_dim"])
        self.working = MemoryStore(p["working_capacity"], kd, seed + 1, "working")
        self.episodic = MemoryStore(p["episodic_capacity"], kd, seed + 2, "episodic")
        self.predictive = MemoryStore(p["predictive_capacity"], kd, seed + 3, "predictive", accept=_PRED)
        self.semantic = SemanticStore(p["semantic_capacity"], p["semantic_radius"])
        self.next_id = 1
        self.interference = 0.0
        self.stats = {"writes": 0, "retrievals": 0, "hits": 0, "confirmed_predictions": 0, "refuted_predictions": 0,
                      "consolidations": 0, "forgotten": 0}
        self.last_retrieved: Optional[MemoryRecord] = None
        self.pending_prediction: Optional[MemoryRecord] = None

    # ------------------------------------------------------------------ keys
    def key_of(self, observation) -> np.ndarray:
        kd = int(self.params["key_dim"])
        key = np.asarray(observation, dtype=np.float32).ravel()[:kd]
        return np.pad(key, (0, kd - key.size))

    def context_of(self) -> np.ndarray:
        a = self.substrate.signals.affect
        return np.array([a.valence, a.arousal, a.curiosity, a.stress, a.motivation], dtype=np.float32)

    def dominant_region(self) -> int:
        sub = self.substrate
        regions = torch.unique(sub.region_ids)
        means = torch.stack([sub.r.detach()[sub.region_ids == g].mean() for g in regions])
        return int(regions[int(torch.argmax(means))])

    _TRUST = {Provenance.INTERNAL_PREDICTION: 0, Provenance.SYNTHETIC_SIMULATION: 1, Provenance.PRIOR_EXPERIENCE: 2,
              Provenance.EXTERNAL_SUPPLIED: 3, Provenance.DIRECT_SENSORY: 4}

    def observation_provenance(self, ctx: StepContext) -> Provenance:
        """The LEAST trusted source among the inputs of this step (an observation is only as trusted as its weakest part)."""
        sources = [Provenance(getattr(self.substrate, "observation_provenance", Provenance.DIRECT_SENSORY))]
        reported, missing = ctx.scratch.get("sensory_provenance"), ctx.scratch.get("sensory_missing")
        if reported:
            for i, value in enumerate(reported):
                if missing is None or not bool(missing[i]):
                    sources.append(Provenance(value))
        weakest = min(sources, key=lambda p: self._TRUST[p])
        return Provenance.SYNTHETIC_SIMULATION if weakest is Provenance.INTERNAL_PREDICTION else weakest

    # ---------------------------------------------------------------- writing
    def write(self, key, value, provenance: Provenance, reward: float = 0.0, confidence: float = 0.5,
              store: str = "episodic", region: Optional[int] = None, step: int = 0, meta: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        """Store an item.  ``provenance`` is mandatory; predictions go to the predictive store only."""
        provenance = Provenance(provenance)
        if provenance is Provenance.INTERNAL_PREDICTION:
            store = "predictive"
        record = MemoryRecord(record_id=self.next_id, key=self.key_of(key), value=np.asarray(value, dtype=np.float32),
                              provenance=provenance, confidence=confidence, reward=float(reward), step=int(step),
                              region=region, context=self.context_of(), verified=provenance.is_observation,
                              priority=abs(float(reward)) + float(self.substrate.signals.surprise), meta=dict(meta or {}))
        self.next_id += 1
        target: MemoryStore = getattr(self, store)
        if store == "episodic":
            self._track_interference(record)
        target.add(record)
        self.stats["writes"] += 1
        return record

    def write_observation(self, key, value, **kw) -> MemoryRecord:
        return self.write(key, value, Provenance.DIRECT_SENSORY, **kw)

    def write_prediction(self, key, value, **kw) -> MemoryRecord:
        return self.write(key, value, Provenance.INTERNAL_PREDICTION, **kw)

    def write_external(self, key, value, **kw) -> MemoryRecord:
        return self.write(key, value, Provenance.EXTERNAL_SUPPLIED, **kw)

    def write_simulated(self, key, value, **kw) -> MemoryRecord:
        return self.write(key, value, Provenance.SYNTHETIC_SIMULATION, **kw)

    def _track_interference(self, record: MemoryRecord) -> float:
        score = 0.0
        for rid, overlap in self.episodic.index.candidates(record.key, min_overlap=self.episodic.index.active // 2)[:8]:
            other = self.episodic.records[rid]
            score = max(score, (overlap / self.episodic.index.active) * (1.0 - max(0.0, cosine(record.value, other.value))))
        self.interference = 0.95 * self.interference + 0.05 * score
        record.meta["interference"] = float(score)
        return score

    # -------------------------------------------------------------- retrieval
    def retrieve(self, key, context=None, k: int = 1, observed_only: bool = False, allow_predictions: bool = True,
                 stores: Tuple[str, ...] = ("working", "episodic", "predictive")) -> List[Tuple[MemoryRecord, float]]:
        """Best matches across stores.  ``observed_only`` returns verified observations exclusively."""
        key = self.key_of(key)
        found: List[Tuple[MemoryRecord, float]] = []
        for name in stores:
            if name == "predictive" and (observed_only or not allow_predictions):
                continue
            found += getattr(self, name).retrieve(key, context=context, k=k, min_similarity=self.params["min_similarity"],
                                                  observed_only=observed_only, allow_predictions=allow_predictions)
        found.sort(key=lambda t: (-t[1], t[0].record_id))
        self.stats["retrievals"] += 1
        return found[:k]

    # ---------------------------------------------------------------- per step
    def begin_step(self, ctx: StepContext) -> None:
        p, sub = self.params, self.substrate
        if self.last_retrieved is not None and not ctx.learning:
            self.last_retrieved = None                       # frozen: no credit now, and none applied late after unfreezing
        if self.last_retrieved is not None:                  # credit the memory recalled before the last action
            delta = 0.1 * float(np.clip(ctx.signals.reward, -1.0, 1.0)) - 0.01
            self.last_retrieved.confidence = float(np.clip(self.last_retrieved.confidence + delta, 0.0, 1.0))
            self.last_retrieved = None
        key = self.key_of(ctx.observation.detach().numpy())
        if not np.any(key) or (len(self.episodic) == 0 and len(self.working) == 0):
            return
        matches = self.retrieve(key, context=self.context_of(), allow_predictions=False, stores=("working", "episodic"))
        if not matches:
            return
        record, sim = matches[0]
        if record.value.size != sub.total_neurons:
            return
        if ctx.learning:
            record.uses += 1
        value = torch.as_tensor(record.value)
        if record.region is not None:
            value = value * torch.where(sub.region_ids == int(record.region), 1.0, float(p["share_gain"]))
        ctx.extra_in += float(p["gain"] * record.confidence * sim) * value
        ctx.memory["ecology"] = {"record_id": record.record_id, "similarity": float(sim), "confidence": record.confidence,
                                 "provenance": record.provenance, "verified": record.verified}
        self.last_retrieved = record
        self.stats["hits"] += 1

    def end_step(self, ctx: StepContext) -> None:
        if not ctx.learning:
            return
        p, sub = self.params, self.substrate
        obs = ctx.observation.detach().numpy()
        rates = sub.r.detach().numpy()
        if self.pending_prediction is not None:               # check the previous prediction against reality
            wm = self.peer("predictive_world_model")
            actual = wm.features if wm is not None else None
            if actual is not None:
                confirmed = verify_against(self.pending_prediction, actual, p["verify_tolerance"], self.next_id, ctx.step)
                if confirmed is not None:
                    self.next_id += 1
                    self.stats["confirmed_predictions"] += 1
                    self.pending_prediction.meta["outcome"] = "confirmed"
                else:
                    self.stats["refuted_predictions"] += 1
                    self.pending_prediction.meta["outcome"] = "refuted"
            self.pending_prediction = None
        prediction = ctx.predictions.get("world_model")
        if prediction is not None:
            self.pending_prediction = self.write_prediction(obs, prediction["next_features"], step=ctx.step,
                                                            confidence=1.0 / (1.0 + 10.0 * prediction["disagreement"]))
        self.working.decay(p["working_decay"])
        self.working.forget(p["min_confidence"])
        provenance = self.observation_provenance(ctx)
        reward = float(ctx.signals.reward)
        if np.any(self.key_of(obs)):
            self.write(obs, rates, provenance, reward=reward, store="working", region=self.dominant_region(), step=ctx.step)
            if abs(reward) > p["reward_threshold"] or ctx.step % int(p["write_every"]) == 0:
                self.write(obs, rates, provenance, reward=reward, store="episodic", region=self.dominant_region(),
                           step=ctx.step, confidence=min(1.0, 0.5 + 0.5 * abs(np.clip(reward, -1, 1))))
        if sub.weave_steps % int(p["consolidate_every"]) == 0:
            self.consolidate()
            self.forget()

    # ----------------------------------------------------- offline operations
    def consolidate(self, max_records: int = 64) -> int:
        """Move the most-used episodic records into semantic prototypes; down-weight the originals."""
        moved = 0
        chosen = sorted(self.episodic.records.values(), key=lambda r: (-(r.uses + abs(r.reward)), r.record_id))[:max_records]
        for record in chosen:
            if record.provenance.is_observation and self.semantic.absorb(record):
                record.confidence *= 0.8
                moved += 1
        self.stats["consolidations"] += 1
        self.substrate.events.publish("memory.consolidate", SignalKind.LEARNING, self.substrate.step_count,
                                      {"moved": moved, "prototypes": len(self.semantic)}, self.mechanism_id)
        return moved

    def forget(self, rate: Optional[float] = None) -> int:
        """Forgetting experiment hook: decay confidences and drop records below ``min_confidence``."""
        self.episodic.decay(self.params["forget_rate"] if rate is None else rate)
        removed = self.episodic.forget(self.params["min_confidence"]) + self.predictive.forget(self.params["min_confidence"])
        self.stats["forgotten"] += removed
        return removed

    def downscale(self, factor: float) -> None:
        """Downscaling experiment hook: shrink every stored episodic value."""
        self.episodic.downscale(factor)

    def next_replay_batch(self, n: int) -> List[MemoryRecord]:
        ranked = sorted(self.episodic.records.values(), key=lambda r: (-(r.priority * r.confidence), r.record_id))
        return ranked[:n]

    def replay_states(self, n: int) -> List[torch.Tensor]:
        return [torch.as_tensor(r.value) for r in self.next_replay_batch(n)]

    def all_records(self) -> List[MemoryRecord]:
        return list(self.working.records.values()) + list(self.episodic.records.values()) + list(self.predictive.records.values())

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"working": self.working.state_dict(), "episodic": self.episodic.state_dict(),
                "predictive": self.predictive.state_dict(), "semantic": self.semantic.state_dict(),
                "next_id": int(self.next_id), "interference": float(self.interference), "stats": dict(self.stats),
                "last_retrieved": None if self.last_retrieved is None else int(self.last_retrieved.record_id),
                "pending_prediction": None if self.pending_prediction is None else int(self.pending_prediction.record_id)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name in ("working", "episodic", "predictive", "semantic"):
            if name in state:
                getattr(self, name).load_state_dict(state[name])
        self.next_id = int(state.get("next_id", self.next_id))
        self.interference = float(state.get("interference", 0.0))
        self.stats.update(state.get("stats", {}))
        def find(record_id):
            if record_id is None:
                return None
            for store in (self.working, self.episodic, self.predictive):
                if int(record_id) in store.records:
                    return store.records[int(record_id)]
            return None

        self.last_retrieved = find(state.get("last_retrieved"))
        self.pending_prediction = find(state.get("pending_prediction"))

    def telemetry(self) -> Dict[str, Any]:
        records = self.all_records()
        return {"working": len(self.working), "episodic": len(self.episodic), "semantic_prototypes": len(self.semantic),
                "predictive": len(self.predictive), "provenance": provenance_counts(records),
                "verified": int(sum(1 for r in records if r.verified)), "interference": float(self.interference),
                "mean_confidence": float(np.mean([r.confidence for r in records])) if records else 0.0, **self.stats}
