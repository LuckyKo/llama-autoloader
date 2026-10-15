"""Unit tests for the per-model FIFO ticket queue (model_queue.py).

These exercise the queue's concurrency contract with plain threads only — no
sockets, no event loop, no network. They lock in the properties the ordering fix
depends on:

  * tickets are granted in strict FIFO order (granted order == ticket order),
  * a state op enqueued before an LLM request is granted first,
  * a timed-out waiter is removed from the pending list (no leak),
  * stop() wakes waiters and refuses new tickets,
  * a spurious release (wrong ticket id) is ignored,
  * the context manager releases the ticket even when the body raises,
  * the registry is idempotent and stops cleanly.

The original design (reports/strata-restore-fifo-design.md §3.5) appended a
ticket to the pending list *before* granting it, so release_ticket re-granted the
same ticket and the queue deadlocked after the first holder. test_no_self_regrant
guards against that regression.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Ensure repo root is importable regardless of cwd (matches tests/conftest.py approach).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model_queue import ModelCommandQueue, ModelQueueRegistry  # noqa: E402


class TestFifoOrdering:
    def test_granted_order_matches_ticket_order_under_contention(self):
        """Many concurrent acquirers must be granted in strict ticket order."""
        q = ModelCommandQueue(9001, "m")
        n = 8
        granted: list[int] = []
        glock = threading.Lock()

        def worker():
            tid = q.acquire_ticket(timeout=10.0)
            with glock:
                granted.append(tid)
            time.sleep(0.005)  # hold briefly so others contend
            q.release_ticket(tid)

        ts = [threading.Thread(target=worker) for _ in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=15.0)

        assert granted == sorted(granted), f"not FIFO: {granted}"
        assert sorted(granted) == list(range(1, n + 1)), f"missing/dup: {granted}"
        assert q.pending_count() == 0

    def test_no_self_regrant(self):
        """Regression for the §3.5 design bug: releasing must grant the NEXT
        waiter, never re-grant the just-released ticket."""
        q = ModelCommandQueue(9002, "m")
        first = q.acquire_ticket(timeout=5.0)   # granted immediately (no contention)
        assert first == 1

        got: list[int] = []
        def waiter():
            got.append(q.acquire_ticket(timeout=5.0))

        w = threading.Thread(target=waiter)
        w.start()
        time.sleep(0.05)                         # ensure the waiter is queued
        assert q.pending_count() == 1
        q.release_ticket(first)                  # must grant ticket 2, not re-grant 1
        w.join(timeout=5.0)
        assert got == [2], f"self-regrant / wrong grant: {got}"

    def test_state_op_before_llm_is_granted_first(self):
        """The core fix: a restore enqueued before an LLM request runs first."""
        q = ModelCommandQueue(9003, "m")
        order: list[str] = []

        def restore():
            tid = q.acquire_ticket(timeout=10.0)
            order.append("restore")
            time.sleep(0.05)
            q.release_ticket(tid)

        def llm():
            tid = q.acquire_ticket(timeout=10.0)
            order.append("llm")
            q.release_ticket(tid)

        r = threading.Thread(target=restore)
        r.start()
        time.sleep(0.02)                         # restore appends first
        l = threading.Thread(target=llm)
        l.start()
        r.join(timeout=10.0)
        l.join(timeout=10.0)
        assert order == ["restore", "llm"], f"ordering broken: {order}"


class TestTimeoutAndStop:
    def test_timeout_removes_pending_entry(self):
        """A waiter that times out must not leak into the pending list."""
        q = ModelCommandQueue(9004, "m")
        held = q.acquire_ticket(timeout=5.0)
        try:
            q.acquire_ticket(timeout=0.2)
            raise AssertionError("expected TimeoutError")
        except TimeoutError:
            pass
        assert q.pending_count() == 0
        q.release_ticket(held)
        # Queue still usable after a timeout. Ticket ids are monotonic: the timed-out
        # acquire consumed id 2, so the next granted id is 3.
        nxt = q.acquire_ticket(timeout=5.0)
        assert nxt == 3
        q.release_ticket(nxt)

    def test_stop_wakes_waiter_and_refuses_new_tickets(self):
        q = ModelCommandQueue(9005, "m")
        held = q.acquire_ticket(timeout=5.0)
        err: dict[str, str] = {}

        def waiter():
            try:
                q.acquire_ticket(timeout=5.0)
            except RuntimeError as e:
                err["e"] = str(e)

        w = threading.Thread(target=waiter)
        w.start()
        time.sleep(0.05)
        q.stop()
        w.join(timeout=5.0)
        assert "e" in err, "waiter not woken by stop()"

        # New acquires after stop() raise immediately.
        try:
            q.acquire_ticket(timeout=1.0)
            raise AssertionError("expected RuntimeError after stop()")
        except RuntimeError:
            pass
        # The original holder can still release cleanly after stop().
        q.release_ticket(held)


class TestReleaseSafety:
    def test_spurious_release_ignored(self):
        q = ModelCommandQueue(9006, "m")
        tid = q.acquire_ticket(timeout=5.0)
        q.release_ticket(999)                    # wrong id -> ignored, no wrong grant
        assert q.pending_count() == 0
        # The real holder can still release.
        q.release_ticket(tid)
        nxt = q.acquire_ticket(timeout=5.0)
        assert nxt == 2
        q.release_ticket(nxt)

    def test_context_manager_releases_on_exception(self):
        """A body that raises must still release the ticket (no deadlock)."""
        q = ModelCommandQueue(9007, "m")
        try:
            with q.ticket(timeout=5.0):
                raise ValueError("boom")
        except ValueError:
            pass
        # Ticket was released despite the exception.
        tid = q.acquire_ticket(timeout=5.0)
        assert tid == 2
        q.release_ticket(tid)


class TestRegistry:
    def test_get_or_create_is_idempotent(self):
        reg = ModelQueueRegistry()
        a = reg.get_or_create(9008, "x")
        b = reg.get_or_create(9008, "x")
        assert a is b
        # Different port -> different queue.
        c = reg.get_or_create(9009, "y")
        assert c is not a

    def test_remove_and_stop_all_are_idempotent(self):
        reg = ModelQueueRegistry()
        q = reg.get_or_create(9010, "x")
        reg.remove(9010)
        reg.remove(9010)                         # no-op on absent port
        reg.stop_all()                           # no-op on empty registry
        # A removed queue is stopped: new acquires raise.
        try:
            q.acquire_ticket(timeout=1.0)
            raise AssertionError("expected RuntimeError after remove()")
        except RuntimeError:
            pass
