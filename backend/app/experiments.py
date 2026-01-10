# app/experiments.py
"""
ExperimentManager and MockTrainer

- ExperimentManager keeps in-memory registry of runs, allows clients to fetch snapshots,
  and manages WebSocket subscribers via a simple pub-sub.
- MockTrainer is a demo producer that simulates a training loop and emits ActivationSnapshots.
  It is intentionally simple and useful for development/integration testing.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Callable, Any
from .models import ActivationSnapshot, TimeIndexedTensor
from . import storage
import numpy as np
from datetime import datetime

# In-memory structure mapping run_id -> list of websocket send coroutines
_subscribers: Dict[str, List[Callable[[ActivationSnapshot], Any]]] = {}

# in-memory snapshots cache for quick access (bounded)
_snapshots_cache: Dict[str, List[ActivationSnapshot]] = {}


def _notify_subscribers(run_id: str, snapshot: ActivationSnapshot):
    """
    Synchronously call subscriber callbacks (which are expected to handle async).
    """
    subs = _subscribers.get(run_id, [])
    for cb in subs:
        try:
            cb(snapshot)
        except Exception as e:
            # subscriber may be a coroutine wrapper that schedules sending
            # errors are logged at the caller
            print(f"Subscriber callback error: {e}")


def subscribe(run_id: str, callback: Callable[[ActivationSnapshot], Any]):
    """
    Register callback for live updates.
    """
    _subscribers.setdefault(run_id, []).append(callback)


def unsubscribe(run_id: str, callback: Callable[[ActivationSnapshot], Any]):
    lst = _subscribers.get(run_id, [])
    if callback in lst:
        lst.remove(callback)


def register_snapshot(run_id: str, snapshot: ActivationSnapshot):
    """
    Persist snapshot and notify subscribers.
    """
    # append to storage
    storage.append_snapshot(run_id, snapshot)
    # update in-memory cache (bounded to last 1000)
    lst = _snapshots_cache.setdefault(run_id, [])
    lst.append(snapshot)
    if len(lst) > 2000:
        lst.pop(0)
    # notify subscribers
    _notify_subscribers(run_id, snapshot)


def get_snapshots(run_id: str, from_step: int = 0, to_step: int = 0) -> List[ActivationSnapshot]:
    """
    Try in-memory cache first, otherwise fall back to storage.
    """
    if run_id in _snapshots_cache:
        out = [s for s in _snapshots_cache[run_id] if s.step >= from_step and (to_step == 0 or s.step <= to_step)]
        if out:
            return out
    return storage.load_snapshots(run_id, from_step=from_step, to_step=to_step)


class MockTrainer:
    """
    Simple loop that simulates training and emits ActivationSnapshot messages.
    Useful to test the frontend without real model training.
    """

    def __init__(self, run_id: str, steps: int = 200, delay: float = 0.2):
        self.run_id = run_id
        self.steps = steps
        self.delay = delay
        self._task: asyncio.Task | None = None
        self._stopped = False

    def start(self):
        if self._task and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(self._run())

    def stop(self):
        self._stopped = True
        if self._task:
            self._task.cancel()

    async def _run(self):
        """
        Each step produces:
         - attention: small random matrix [heads, seq, seq]
         - read/write weights: random [heads, slots]
         - memory_slots: [slots, dim]
         - reconstructions: [seq, dim]
        """
        for step in range(1, self.steps + 1):
            if self._stopped:
                break
            # synthetic sizes
            heads = 8
            seq = min(32, 4 + step // 5)
            slots = 16
            dim = 32

            attention = TimeIndexedTensor(
                time=step,
                shape=[heads, seq, seq],
                buffer=np.random.rand(heads * seq * seq).astype(float).tolist()
            )
            read_w = TimeIndexedTensor(
                time=step,
                shape=[heads, slots],
                buffer=np.abs(np.random.randn(heads * slots).astype(float)).tolist()
            )
            write_w = TimeIndexedTensor(
                time=step,
                shape=[heads, slots],
                buffer=np.abs(np.random.randn(heads * slots).astype(float)).tolist()
            )
            memory_slots = TimeIndexedTensor(
                time=step,
                shape=[slots, dim],
                buffer=np.random.randn(slots * dim).astype(float).tolist()
            )
            recon = TimeIndexedTensor(
                time=step,
                shape=[seq, dim],
                buffer=np.random.randn(seq * dim).astype(float).tolist()
            )
            loss = float(np.exp(-step / 100.0) + 0.02 * np.random.rand())

            snapshot = ActivationSnapshot(
                step=step,
                epoch=step // 100,
                attention=attention,
                read_weights=read_w,
                write_weights=write_w,
                memory_slots=memory_slots,
                reconstructions=recon,
                loss=loss,
                metrics={"aux_metric": float(np.random.rand())}
            )

            # register (persist + notify)
            register_snapshot(self.run_id, snapshot)

            # throttle to simulate time
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                break


# convenience mapping to keep live trainers per run
_trainers: Dict[str, MockTrainer] = {}


def start_mock_run(name: str = "demo", model_type: str = "memnet") -> str:
    """
    Create a new run and start a MockTrainer producing snapshots.
    Returns run_id.
    """
    run_id = str(uuid.uuid4())
    storage.register_experiment(run_id=run_id, name=name, model_type=model_type)
    trainer = MockTrainer(run_id=run_id, steps=1000, delay=0.15)
    _trainers[run_id] = trainer
    trainer.start()
    return run_id


def stop_mock_run(run_id: str):
    t = _trainers.get(run_id)
    if t:
        t.stop()
        _trainers.pop(run_id, None)
