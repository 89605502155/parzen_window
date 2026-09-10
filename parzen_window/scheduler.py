"""Thread-safety around dataset compaction.

While :meth:`~parzen_window.core.ParzenWindowClassifier.compact` runs, the
training set is being rewritten in place, so calls to ``predict`` or
``partial_fit`` made from other threads cannot run concurrently with it.
Rather than failing, such calls are queued and executed automatically, in
arrival order, right after compaction finishes -- acting as a small
scheduler for deferred work.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable


class _PendingCall:
    __slots__ = ("_func", "_args", "_kwargs", "_done", "_result", "_error")

    def __init__(self, func: Callable, args: tuple, kwargs: dict):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._done = threading.Event()
        self._result = None
        self._error = None

    def run(self) -> None:
        try:
            self._result = self._func(*self._args, **self._kwargs)
        except Exception as error:  # re-raised in the waiting thread
            self._error = error
        finally:
            self._done.set()

    def wait(self):
        self._done.wait()
        if self._error is not None:
            raise self._error
        return self._result


class CompactionScheduler:
    """Serializes access to model data and defers calls during compaction."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._compacting = threading.Event()
        self._queue: deque[_PendingCall] = deque()
        self._queue_lock = threading.Lock()

    @property
    def is_compacting(self) -> bool:
        return self._compacting.is_set()

    def call(self, func: Callable, *args, **kwargs):
        """Run ``func`` now, or queue it while compaction is in progress."""
        if self._compacting.is_set():
            pending = _PendingCall(func, args, kwargs)
            with self._queue_lock:
                self._queue.append(pending)
            return pending.wait()
        with self._lock:
            return func(*args, **kwargs)

    def run_exclusively(self, func: Callable, *args, **kwargs):
        """Run ``func`` (compaction) with exclusive access to the data."""
        with self._lock:
            self._compacting.set()
            try:
                return func(*args, **kwargs)
            finally:
                self._compacting.clear()
                self._drain_queue()

    def _drain_queue(self) -> None:
        with self._queue_lock:
            pending_calls = list(self._queue)
            self._queue.clear()
        for pending in pending_calls:
            pending.run()
