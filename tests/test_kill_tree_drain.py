"""Regression tests for BUG_0012: _kill_proc must be tree-complete.

The engine grandchildren (Strata engine binary + vision encoder) are orphaned
when the tracked wrapper exits first: they survive the primary terminate/kill
path (in the incident, a psutil.AccessDenied on a descendant, pid 7908) and
hold the model's committed memory. The old code only LOGGED such descendants
("reaped by the OS") and returned success, so the immediate state-restore
auto-load died on the Windows commit limit (VirtualAlloc error 1314).

The fix re-enumerates the whole process tree after the parent is verified gone,
terminates any survivors, and bounded-waits until the ENTIRE tree is gone
before reporting success. These tests drive REAL process trees through the
real _kill_proc / _tree_survivors path (test 1) and pin the bounded-timeout
failure verdict (test 2).
"""
from __future__ import annotations

import logging
import subprocess  # noqa: TCH003 - used for real subprocesses
import sys
from pathlib import Path
from unittest.mock import MagicMock

import psutil
import pytest

import server
from server import LoadedModel, ModelConfig


# ---------------------------------------------------------------------------
# Log capture (mirrors tests/test_safe_mode_slot_ordering.py:autoloader_logs)
# ---------------------------------------------------------------------------
@pytest.fixture()
def autoloader_logs():
    """Capture records emitted on the "autoloader" logger for one test.

    Production sets the level via init_logging(); tests never call it, so the
    logger still defaults to WARNING and would drop lower lines. Raise the
    level for the duration of the test and restore it after.
    """
    recs: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            recs.append(record.getMessage())

    logger = logging.getLogger("autoloader")
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # keep pytest's caplog/root handlers out of the way
    handler = _Cap()
    logger.addHandler(handler)
    try:
        yield recs
    finally:
        logger.removeHandler(handler)
        logger.level = old_level
        logger.propagate = old_propagate


def _add_resident(mgr, model_id="engine.gguf", pid=424242, process=None):
    """Register a resident model, mirroring tests/test_unload_reporting.py."""
    cfg = ModelConfig.default_for(Path(model_id))
    lm = LoadedModel(
        model_id=model_id,
        gguf_path=Path(model_id),
        config=cfg,
        port=19555,
        process=process if process is not None else MagicMock(),
        pid=pid,
    )
    lm.ready = True
    mgr.models[model_id] = cfg
    mgr.loaded[model_id] = lm
    return lm


def _safe_kill(pid):
    """Best-effort teardown of a leftover real process (idempotent)."""
    try:
        psutil.Process(pid).kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Test 1 — the tree-drain waits for a real descendant, THEN succeeds
# ---------------------------------------------------------------------------
# Real 3-tier tree: tracked parent -> child -> grandchild.
#   * the child's terminate/kill is forced to raise AccessDenied, reproducing the
#     incident's signature (a descendant whose termination is denied and which
#     therefore survives the primary path);
#   * the child then exits on its own after a short delay, so the correct
#     outcome is: the tree-drain WAITS for it and only THEN reports success.
# Coverage note: the primary path's enumeration (parent.children(recursive=True),
# NOT stubbed here) puts BOTH the child and the grandchild in the pre-kill
# snapshot, so the drain reaches them via the BFS from the live child — this
# test proves the drain WAITS for a real descendant and reports success only
# once the whole tree is gone. The process-table (orphan) pass and the pid-
# recycling guard are exercised separately by test_tree_survivors_orphan_pass
# below, which stubs the process table directly.
PARENT_CODE = (
    "import subprocess, sys, time\n"
    "c = subprocess.Popen([sys.executable, '-c', "
    "\"import subprocess, sys, time; "
    "g=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
    "print('G', g.pid, flush=True); time.sleep(0.6)\"])\n"
    "print('C', c.pid, flush=True)\n"
    "time.sleep(30)\n"
)


@pytest.mark.asyncio
async def test_tree_drain_waits_for_grandchild(isolated_manager, autoloader_logs,
                                               monkeypatch, request):
    mgr = isolated_manager

    parent = subprocess.Popen([sys.executable, "-c", PARENT_CODE],
                              stdout=subprocess.PIPE)
    request.addfinalizer(lambda: (parent.kill(), parent.wait()))
    child_pid = int(parent.stdout.readline().decode().split()[1])
    grand_pid = int(parent.stdout.readline().decode().split()[1])
    request.addfinalizer(lambda: _safe_kill(child_pid))
    request.addfinalizer(lambda: _safe_kill(grand_pid))

    _add_resident(mgr, model_id="engine.gguf", pid=parent.pid)

    # Reproduce the defect: the child cannot be terminated (AccessDenied), so
    # the primary path leaves it alive. wait_procs is stubbed fast (as the
    # existing kill tests do) so the primary path does not itself wait out the
    # child's self-exit — the child must be reaped by the tree-drain.
    real_term, real_kill = psutil.Process.terminate, psutil.Process.kill

    def _term(self, *a, **k):
        if self.pid == child_pid:
            raise psutil.AccessDenied(self.pid)
        return real_term(self, *a, **k)

    def _kill(self, *a, **k):
        if self.pid == child_pid:
            raise psutil.AccessDenied(self.pid)
        return real_kill(self, *a, **k)

    monkeypatch.setattr(psutil.Process, "terminate", _term, raising=False)
    monkeypatch.setattr(psutil.Process, "kill", _kill, raising=False)
    monkeypatch.setattr(server.psutil, "wait_procs",
                        lambda procs, timeout: ([], list(procs)))

    result = await mgr.unload_model("engine.gguf")

    # (a) A live descendant WAS present and success was NOT reported before it
    # was gone: the tree-drain warning fired while the child was still alive.
    warnings = [m for m in autoloader_logs
                if "still running after unload" in m]
    assert warnings, ("tree-drain warning never fired: success was reported "
                      "with a live descendant")
    # (c) The warning now carries ppid so future occurrences are attributable
    # (the BUG_0012 forensics gap: the old note logged pids only).
    assert "ppid=" in warnings[0], "warning must include ppid for attribution"

    # (b) It returned success only AFTER the whole tree was gone.
    assert result is not None and result.unloaded is True
    assert not psutil.pid_exists(child_pid), (
        "unload reported success while the descendant was still alive")
    assert not psutil.pid_exists(grand_pid), (
        "grandchild survived the tree drain")


# ---------------------------------------------------------------------------
# Test 2 — a descendant that never dies turns the unload into an explicit
# failure once the bounded drain window expires.
# ---------------------------------------------------------------------------
class _ParentGone:
    """Parent mock: stops cleanly so the code reaches the tree-drain phase."""

    def __init__(self, pid):
        self.pid = pid

    def children(self, recursive=True):  # noqa: ARG002 - psutil API
        return []

    def terminate(self):
        pass

    def kill(self):
        pass

    def is_running(self) -> bool:
        return False

    def status(self) -> str:
        return psutil.STATUS_ZOMBIE


class _StubbornProc:
    """A tree survivor that ignores every termination attempt (the scan is
    stubbed, so only these attributes are exercised)."""

    def __init__(self, pid):
        self.pid = pid

    def name(self) -> str:
        return "stubborn-engine"

    def ppid(self) -> int:
        return 1

    def terminate(self):
        pass

    def kill(self):
        pass


@pytest.mark.asyncio
async def test_tree_drain_times_out_reports_failure(isolated_manager, autoloader_logs,
                                                    monkeypatch):
    mgr = isolated_manager
    _add_resident(mgr, model_id="engine.gguf", pid=424242)

    # Parent is verifiably gone, so the code reaches the tree-drain phase.
    parent = _ParentGone(424242)
    monkeypatch.setattr(server.psutil, "Process", lambda pid: parent)
    monkeypatch.setattr(server.psutil, "wait_procs",
                        lambda procs, timeout: ([], list(procs)))

    # The tree scan is stubbed to a persistent survivor that never dies, and the
    # bounded window is shortened for the test. The real psutil scan is covered
    # by test_tree_drain_waits_for_grandchild; this pins the timeout verdict.
    survivor = _StubbornProc(999999)
    monkeypatch.setattr(server, "_tree_survivors",
                        lambda tree_pids, expected_ctime: [survivor])
    monkeypatch.setattr(server, "KILL_TREE_DRAIN_TIMEOUT_S", 0.5)

    result = await mgr.unload_model("engine.gguf")

    # The unload must NOT claim success while a descendant is still running.
    assert result is not None and result.unloaded is False
    assert "tree drain timed out" in result.reason
    assert "999999" in result.reason

    # The model stays tracked (the engine is still resident).
    assert "engine.gguf" in mgr.loaded

    # An ERROR is logged with the surviving pid (diagnosable, not silent).
    errors = [m for m in autoloader_logs if "Tree drain timed out" in m]
    assert errors and "999999" in errors[0]


# ---------------------------------------------------------------------------
# Unit test — the process-table (orphan) pass and the pid-recycling guard.
# test 1 reaches descendants via the pre-kill snapshot + BFS, so the orphan pass
# and the create_time recycling guard are never exercised there. This stubs the
# process table directly to prove both, deterministically and without real
# subprocesses.
# ---------------------------------------------------------------------------
class _FakeProc:
    """Minimal psutil.Process stand-in for the tree scan."""

    def __init__(self, pid, *, live=True, ctime=100.0, ppid=None, kids=()):
        self.pid = pid
        self._live, self._ctime, self._ppid = live, ctime, ppid
        self._kids = list(kids)

    def is_running(self) -> bool:
        return self._live

    def status(self) -> str:
        # Anything that is not STATUS_ZOMBIE reads as live to the scan.
        return psutil.STATUS_ZOMBIE if not self._live else "sleeping"

    def create_time(self) -> float:
        return self._ctime

    def children(self, recursive=False):  # noqa: ARG002 - psutil API
        return list(self._kids)

    def ppid(self) -> int:
        return self._ppid

    def name(self) -> str:
        return "fake-engine"


class _Row:
    def __init__(self, pid, ppid):
        self.info = {"pid": pid, "ppid": ppid}


def _patch_table(monkeypatch, table, rows):
    """Route psutil.Process / process_iter in server through a fixed table."""
    def _factory(pid):
        if pid in table:
            return table[pid]
        raise psutil.NoSuchProcess(pid)

    def _iter(*args, **kwargs):
        return iter(_Row(pid, ppid) for pid, ppid in rows)

    monkeypatch.setattr(server.psutil, "Process", _factory)
    monkeypatch.setattr(server.psutil, "process_iter", _iter)


def test_tree_survivors_orphan_pass_and_recycling_guard(monkeypatch):
    # Case A — a real orphan: its parent (a tree pid) is dead, so the BFS
    # cannot descend from it; only the process-table pass recovers it.
    dead_parent = _FakeProc(1000, live=False)
    orphan = _FakeProc(2000, live=True, ppid=1000)
    _patch_table(monkeypatch, {1000: dead_parent, 2000: orphan},
                 rows=[(2000, 1000)])
    survivors = server._tree_survivors({1000}, {1000: 100.0})
    assert [p.pid for p in survivors] == [2000], (
        "the process-table pass must recover an orphan of a dead tree pid")

    # Case B — pid recycling: the tree pid is now occupied by an UNRELATED live
    # process (create_time differs from the snapshot). Neither the BFS nor the
    # orphan pass may descend into it or its child.
    recycled = _FakeProc(3000, live=True, ctime=999.0, kids=[_FakeProc(4000)])
    unrelated_child = _FakeProc(4000, live=True, ppid=3000)
    _patch_table(monkeypatch, {3000: recycled, 4000: unrelated_child},
                 rows=[(4000, 3000)])
    survivors = server._tree_survivors({3000}, {3000: 100.0})
    assert survivors == [], (
        "a pid recycled to an unrelated process must not be descended into")
