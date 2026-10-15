"""Per-model FIFO ticket queue for the autoloader.

The autoloader has two independent paths to a model's engine HTTP port:
  - the raw TCP forwarder (LLM chat completions), and
  - the FastAPI state operations (save / restore).

The engine's own lock serializes operations but does NOT order them, so a fast
LLM request can win the race against a pending state restore and force a full
prompt reprocess instead of resuming from restored state. This module provides a
per-model FIFO *ticket* queue: every operation acquires a ticket (which blocks
until it is next in FIFO order), performs its I/O while holding the ticket, then
releases it. The queue grants tickets in strict submission order, giving the
ordering guarantee the engine's lock cannot.

Design reference: reports/strata-restore-fifo-design.md §3.5 (ticket-based v2).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, Optional, Tuple

log = logging.getLogger("autoloader")


class ModelCommandQueue:
    """Per-model FIFO ticket queue.

    Callers acquire a ticket (which blocks until the operation is next in FIFO
    order), perform their work, then release the ticket. This enforces strict
    FIFO ordering without moving the actual I/O logic into the queue.

    Thread-safe: works with both threading (TCP forwarder) and asyncio
    (FastAPI via asyncio.to_thread) callers. Uses only threading primitives.

    Usage (preferred — context manager enforces release):
        with queue.ticket(timeout=300.0):
            self._pipe_to_backend(...)

    Usage (manual):
        ticket = queue.acquire_ticket(timeout=300.0)
        try:
            ...
        finally:
            queue.release_ticket(ticket)
    """

    def __init__(self, port: int, model_id: str):
        self.port = port
        self.model_id = model_id
        self._pending: Deque[Tuple[int, threading.Event]] = deque()  # FIFO queue
        self._granted_id: Optional[int] = None   # ticket ID currently held
        self._granted_event: Optional[threading.Event] = None
        self._lock = threading.Lock()
        self._order = 0
        self._stopped = False

    def acquire_ticket(self, timeout: float = 300.0) -> int:
        """Block until this caller's turn. Returns the ticket ID.

        Raises TimeoutError if the ticket is not granted within timeout.
        Raises RuntimeError if the queue has been stopped.
        """
        event = threading.Event()
        with self._lock:
            if self._stopped:
                raise RuntimeError(f"FIFO queue stopped for model {self.model_id}")
            self._order += 1
            ticket_id = self._order
            if self._granted_id is None:
                # No current holder: grant immediately. _pending holds only
                # not-yet-granted waiters, so the granted ticket is NOT added to
                # it — otherwise release_ticket would pop and re-grant this same
                # ticket and the queue would deadlock after the first holder.
                self._granted_id = ticket_id
                self._granted_event = event
                event.set()
            else:
                # A holder exists: queue as a waiter in strict FIFO order.
                self._pending.append((ticket_id, event))

        # Wait for our turn (no-op if we were granted immediately).
        if not event.wait(timeout=timeout):
            with self._lock:
                # We may have been granted at the same instant the wait timed out;
                # honor the grant rather than erroring (a granted_id no one will
                # release would otherwise wedge the queue).
                if self._granted_id == ticket_id:
                    return ticket_id
                # Otherwise drop our entry so it cannot leak or be granted later.
                try:
                    self._pending.remove((ticket_id, event))
                except ValueError:
                    pass  # already removed (e.g. by stop())
            raise TimeoutError(
                f"FIFO queue timeout for model {self.model_id} "
                f"(ticket {ticket_id}, {len(self._pending)} queued)")

        # Confirm we were actually granted (not woken by stop()).
        if self._granted_id != ticket_id:
            raise RuntimeError(f"FIFO queue stopped while waiting for model {self.model_id}")
        return ticket_id

    def release_ticket(self, ticket_id: int) -> None:
        """Release the current ticket, granting the next one in FIFO order.

        Verifies ticket ownership: only the currently granted ticket can be
        released. A spurious release (wrong ticket) is logged and ignored.
        """
        with self._lock:
            if self._granted_id != ticket_id:
                log.warning(
                    f"release_ticket({ticket_id}) but granted is {self._granted_id} "
                    f"for model {self.model_id} — ignoring spurious release")
                return
            self._granted_id = None
            self._granted_event = None
            # Grant the next pending ticket.
            if self._pending:
                next_id, next_event = self._pending.popleft()
                self._granted_id = next_id
                self._granted_event = next_event
                next_event.set()

    @contextmanager
    def ticket(self, timeout: float = 300.0):
        """Context manager that acquires and releases a ticket.

        Guarantees release even on exception (prevents a deadlock from a
        crashed caller).
        """
        ticket_id = self.acquire_ticket(timeout=timeout)
        try:
            yield ticket_id
        finally:
            self.release_ticket(ticket_id)

    def stop(self) -> None:
        """Wake all waiting tickets with an error. No new tickets are granted."""
        with self._lock:
            self._stopped = True
            # Wake all pending waiters.
            for _, event in self._pending:
                event.set()
            self._pending.clear()
            # Wake the current holder (if any) — they'll see the _granted_id mismatch.
            if self._granted_event is not None:
                self._granted_event.set()

    def pending_count(self) -> int:
        """Number of callers currently waiting for a ticket (observability)."""
        with self._lock:
            return len(self._pending)


class ModelQueueRegistry:
    """Registry of per-model FIFO ticket queues, keyed by engine port."""

    def __init__(self):
        self._queues: Dict[int, ModelCommandQueue] = {}
        self._lock = threading.Lock()

    def get_or_create(self, port: int, model_id: str) -> ModelCommandQueue:
        """Return the queue for ``port``, creating it on first use."""
        with self._lock:
            q = self._queues.get(port)
            if q is None:
                q = ModelCommandQueue(port, model_id)
                self._queues[port] = q
            elif q.model_id != model_id:
                # Expected: aliases (e.g. engine name vs autoloader internal ID) resolve
                # to the same port. Debug-level only to avoid spam on every request.
                log.debug(
                    f"queue for port {port} created with model_id={q.model_id!r}, "
                    f"requested with {model_id!r} — returning existing queue")
            return q

    def remove(self, port: int) -> None:
        """Stop and drop the queue for ``port`` (no-op if absent)."""
        with self._lock:
            q = self._queues.pop(port, None)
        if q is not None:
            q.stop()

    def stop_all(self) -> None:
        """Stop every queue (shutdown path)."""
        with self._lock:
            queues = list(self._queues.values())
            self._queues.clear()
        for q in queues:
            q.stop()
