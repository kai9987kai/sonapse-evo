"""Small typed event bus used for cross-subsystem communication and telemetry.

Publishing never raises: a failing subscriber is counted and skipped so that an
observer (dashboard, logger, experiment recorder) can never break the neural
update loop.  History is a bounded ring buffer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from .contracts import INTERFACE_VERSIONS, SignalKind


@dataclass(frozen=True)
class Event:
    topic: str
    kind: SignalKind
    step: int
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    version: str = INTERFACE_VERSIONS["event"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic, "kind": self.kind.value, "step": self.step,
            "payload": self.payload, "source": self.source, "version": self.version,
        }


Subscriber = Callable[[Event], None]


class EventBus:
    """Topic-based publish/subscribe with prefix wildcards (``"plasticity.*"``)."""

    def __init__(self, history: int = 512):
        self._subs: Dict[str, List[Subscriber]] = {}
        self._history: Deque[Event] = deque(maxlen=int(history))
        self.published = 0
        self.subscriber_errors = 0

    def subscribe(self, topic: str, fn: Subscriber) -> Callable[[], None]:
        self._subs.setdefault(topic, []).append(fn)

        def unsubscribe() -> None:
            subs = self._subs.get(topic, [])
            if fn in subs:
                subs.remove(fn)

        return unsubscribe

    def publish(self, topic: str, kind: SignalKind = SignalKind.TELEMETRY, step: int = 0,
                payload: Optional[Dict[str, Any]] = None, source: str = "") -> Event:
        event = Event(topic=topic, kind=kind, step=int(step), payload=payload or {}, source=source)
        self._history.append(event)
        self.published += 1
        for pattern, subs in list(self._subs.items()):
            if pattern == topic or pattern == "*" or (pattern.endswith(".*") and topic.startswith(pattern[:-1])):
                for fn in list(subs):
                    try:
                        fn(event)
                    except Exception:  # observers must never break the update loop
                        self.subscriber_errors += 1
        return event

    def recent(self, topic_prefix: str = "", limit: int = 50) -> List[Event]:
        events = [e for e in self._history if e.topic.startswith(topic_prefix)]
        return events[-int(limit):]

    def clear(self) -> None:
        self._history.clear()


__all__ = ["Event", "EventBus"]
