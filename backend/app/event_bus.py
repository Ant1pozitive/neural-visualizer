# app/event_bus.py
"""
Abstract event bus used to decouple training engine from API layer.
"""

from typing import Callable, Dict, List, Any
from .models import ActivationSnapshot


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[ActivationSnapshot], Any]]] = {}

    def subscribe(self, run_id: str, callback: Callable[[ActivationSnapshot], Any]):
        self._subscribers.setdefault(run_id, []).append(callback)

    def unsubscribe(self, run_id: str, callback: Callable):
        if run_id in self._subscribers and callback in self._subscribers[run_id]:
            self._subscribers[run_id].remove(callback)

    def publish(self, run_id: str, snapshot: ActivationSnapshot):
        for cb in self._subscribers.get(run_id, []):
            cb(snapshot)


# singleton
event_bus = EventBus()
