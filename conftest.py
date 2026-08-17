"""Root-level conftest: exclude pre-existing ad-hoc scripts from test collection.

The repo contains manual/debug scripts named ``test_*.py`` that make LIVE HTTP
calls to a running server (127.0.0.1) at import time. They are NOT unit tests and
would crash pytest collection. ``collect_ignore`` tells pytest to skip them while
still collecting the real tests (root-level test_server.py + the tests/ package).

These scripts are intentionally left in place (out of scope to rename/move); they
are simply excluded from automated discovery.
"""
import os

_here = os.path.dirname(__file__)
collect_ignore = [
    os.path.join(_here, "test_direct.py"),
    os.path.join(_here, "test_kv_cache_reuse.py"),
    os.path.join(_here, "test_kv_fix.py"),
    os.path.join(_here, "test_slot_restore.py"),
]
