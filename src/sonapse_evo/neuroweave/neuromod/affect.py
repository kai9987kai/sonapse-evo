"""Affective control state: valence, arousal, curiosity, stress and motivation.

These are *operational control variables* computed from reward, surprise and
activity.  They are not, and do not model, subjective feelings.

Appraisal (``s`` = surprise, ``R`` = lagged reward clipped to [-1, 1])

    surprise   s = clip(2 u + |a - p|, 0, 1)           u = ensemble uncertainty, a / p = actual / predicted mean rate
    novelty    <- 0.88 novelty + 0.12 s
    expected   <- 0.94 expected + 0.06 R ;   RPE = clip(R - expected, -1, 1)
    valence    <- 0.965 valence + 0.035 R
    arousal    <- 0.92 arousal + 0.08 (0.15 + 0.55 s + 0.3 |R|)
    curiosity  <- 0.90 curiosity + 0.10 novelty
    stress     <- 0.97 stress + 0.03 max(0, -R) + 0.015 s
    motivation <- 0.97 motivation + 0.03 (0.5 + 0.5 valence - 0.25 stress)

and the modulators it drives:  NA <- 0.9 NA + 0.1 (0.6 arousal + 0.4 s),
eCB <- 0.96 eCB + 0.04 |R| a,  DA <- 0.94 DA + 0.06 RPE,  ACh <- 0.92 ACh + 0.08 (0.6 s + 0.4 curiosity).

Control couplings (each individually switchable for ablation; all are exactly
neutral at the resting state):

    curiosity  -> exploration noise       exploration = k_e * max(0, curiosity - 0.2)
    stress     -> response threshold      threshold_shift -= k_t * stress ;  attention_gain *= 1 + 0.3 stress
    motivation -> behavioural persistence persistence = k_p * max(0, motivation - 0.5)
    valence    -> reward-related learning plasticity_gain *= clip(1 + 0.3 valence, 0.7, 1.3)
    arousal    -> processing dynamics     leak_scale = clip(1 + 0.4 (arousal - 0.15), 0.8, 1.4)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..contracts import AffectState, Mechanism, Stage, StepContext

_REST = AffectState()


class AffectiveControl(Mechanism):
    mechanism_id = "affective_control"
    stage = Stage.MODULATE

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {"couple_exploration": True, "couple_threshold": True, "couple_persistence": True,
                "couple_learning": True, "couple_dynamics": True, "drive_modulators": True,
                "exploration_gain": 0.5, "threshold_gain": 0.08, "persistence_gain": 0.6,
                "clamp_values": {}}       # name -> fixed value (intervention on an affect variable)

    def on_attach(self) -> None:
        self.expected_reward = 0.0
        self.last_surprise = 0.0
        self.steps = 0

    def begin_step(self, ctx: StepContext) -> None:
        sig, p = ctx.signals, self.params
        a, m = sig.affect, sig.modulators
        reward = float(np.clip(sig.reward, -1.0, 1.0))
        activity = float(self.substrate.r.detach().abs().mean())
        gap = abs(activity - sig.prediction) if sig.prediction else 0.0
        surprise = float(np.clip(2.0 * max(0.0, sig.uncertainty) + gap, 0.0, 1.0))
        rpe = float(np.clip(reward - self.expected_reward, -1.0, 1.0))
        self.expected_reward = 0.94 * self.expected_reward + 0.06 * reward
        a.novelty = 0.88 * a.novelty + 0.12 * surprise
        a.reward_prediction_error = rpe
        a.valence = 0.965 * a.valence + 0.035 * reward
        a.arousal = 0.92 * a.arousal + 0.08 * (0.15 + 0.55 * surprise + 0.3 * abs(reward))
        a.curiosity = 0.90 * a.curiosity + 0.10 * (0.2 + 0.8 * a.novelty)
        a.stress = 0.97 * a.stress + 0.03 * max(0.0, -reward) + 0.015 * surprise
        a.motivation = 0.97 * a.motivation + 0.03 * (0.5 + 0.5 * a.valence - 0.25 * a.stress)
        for name, value in p["clamp_values"].items():
            setattr(a, name, float(value))
        a.clamp()
        if p["drive_modulators"]:
            m.noradrenaline = 0.9 * m.noradrenaline + 0.1 * (0.6 * (a.arousal - _REST.arousal) + 0.4 * surprise)
            m.endocannabinoid = 0.96 * m.endocannabinoid + 0.04 * abs(reward) * activity
            m.dopamine = 0.94 * m.dopamine + 0.06 * rpe
            m.acetylcholine = 0.92 * m.acetylcholine + 0.08 * (0.15 + 0.6 * surprise + 0.4 * (a.curiosity - _REST.curiosity))
            m.clamp()
        self.last_surprise = surprise

        sig.exploration = p["exploration_gain"] * max(0.0, a.curiosity - _REST.curiosity) if p["couple_exploration"] else 0.0
        sig.persistence = min(0.9, p["persistence_gain"] * max(0.0, a.motivation - _REST.motivation)) if p["couple_persistence"] else 0.0
        if p["couple_threshold"] and a.stress > 0.0:
            ctx.threshold_shift = ctx.threshold_shift - p["threshold_gain"] * a.stress
            sig.attention_gain = float(sig.attention_gain * (1.0 + 0.3 * a.stress))
        if p["couple_learning"]:
            sig.plasticity_gain = float(sig.plasticity_gain * np.clip(1.0 + 0.3 * a.valence, 0.7, 1.3))
        if p["couple_dynamics"]:
            ctx.leak_scale = float(ctx.leak_scale * np.clip(1.0 + 0.4 * (a.arousal - _REST.arousal), 0.8, 1.4))
        self.steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"expected_reward": float(self.expected_reward), "steps": int(self.steps), "last_surprise": float(self.last_surprise)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.expected_reward = float(state.get("expected_reward", 0.0))
        self.steps = int(state.get("steps", 0))
        self.last_surprise = float(state.get("last_surprise", 0.0))

    def telemetry(self) -> Dict[str, Any]:
        sig = self.substrate.signals
        return {"affect": sig.affect.as_dict(), "surprise": self.last_surprise, "expected_reward": self.expected_reward,
                "exploration": float(sig.exploration), "persistence": float(sig.persistence)}
