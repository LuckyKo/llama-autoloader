"""BUG_0187 — save the caller's retained session, not a sibling's slot 0.

NInfer exposes one retained KV session per catalog cell through ``GET /slots`` and
reports the cell's ``session_digest`` in the Chat Completions response. The autoloader
used to POST ``/slots/0?action=save`` unconditionally, so under concurrent fleet
traffic the state file captured whichever sibling agent dominated that cell.

These tests lock the autoloader half:

* a supplied digest resolves to the matching retained cell (which is usually not 0),
* the save carries an ``if_digest`` precondition so the cell cannot change mid-save,
* an unresolvable digest fails honestly with ``slot_session_mismatch`` and does NOT
  fall back to saving slot 0,
* omitting the digest keeps the legacy single-slot behaviour byte-for-byte.
"""

from unittest.mock import AsyncMock, patch

import pytest

import server


class _Resp:
    def __init__(self, status_code=200, text='', payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError('not json')
        return self._payload


class _FakeClient:
    """Minimal httpx.AsyncClient double: records GETs and POSTs with their bodies."""

    def __init__(self, *, get=None, post=None):
        self._get = list(get or [])
        self._post = list(post or [])
        self.get_calls = []
        self.post_calls = []

    async def get(self, url, timeout=None):
        self.get_calls.append(url)
        return self._get.pop(0)

    async def post(self, url, json=None, timeout=None):
        self.post_calls.append((url, json))
        return self._post.pop(0)

    async def aclose(self):
        pass


def _slot(slot_id, digest, retained=True):
    return {'id': slot_id, 'retained': retained, 'session_digest': digest}


@pytest.fixture()
def mgr(isolated_manager):
    """Isolated manager with one ready model, mirroring tests/test_regression.py."""
    from pathlib import Path
    from unittest.mock import MagicMock

    cfg = server.ModelConfig.default_for(Path('m.gguf'))
    lm = MagicMock()
    lm.port = 19500
    lm.ready = True
    lm.config = cfg
    lm.state_path = None
    isolated_manager.models['m.gguf'] = cfg
    isolated_manager.loaded['m.gguf'] = lm
    return isolated_manager


# ── Digest resolution ─────────────────────────────────────────────────────────


async def test_save_with_digest_targets_the_matching_cell(mgr):
    client = _FakeClient(
        get=[_Resp(payload=[_slot(0, 'sibling'), _slot(2, 'mine')])],
        post=[_Resp(200)],
    )
    mgr.client = client
    path = mgr.save_state_dir / 'm.gguf.default.bin'

    assert await mgr._perform_slot_action(
        'save', 19500, path, model_id='m.gguf', session_digest='mine') is True

    assert client.get_calls == ['http://127.0.0.1:19500/slots?model=m.gguf']
    assert client.post_calls == [(
        'http://127.0.0.1:19500/slots/2?action=save&model=m.gguf',
        {'filename': 'm.gguf.default.bin', 'if_digest': 'mine'},
    )]


async def test_save_with_digest_ignores_non_retained_cell(mgr):
    """A matching digest in an active (not retained) cell is not saveable."""
    client = _FakeClient(
        get=[_Resp(payload=[_slot(1, 'mine', retained=False)])],
        post=[_Resp(200)],
    )
    mgr.client = client
    path = mgr.save_state_dir / 'm.gguf.default.bin'

    with pytest.raises(server.EngineSlotError) as exc:
        await mgr._perform_slot_action(
            'save', 19500, path, model_id='m.gguf', session_digest='mine')

    assert exc.value.code == 'slot_session_mismatch'
    assert exc.value.status_code == 409
    assert client.post_calls == []  # never saved a sibling's slot


async def test_unresolvable_digest_refuses_instead_of_falling_back_to_slot_zero(mgr):
    client = _FakeClient(get=[_Resp(payload=[_slot(0, 'sibling')])], post=[_Resp(200)])
    mgr.client = client
    path = mgr.save_state_dir / 'm.gguf.default.bin'

    with pytest.raises(server.EngineSlotError) as exc:
        await mgr._perform_slot_action(
            'save', 19500, path, model_id='m.gguf', session_digest='gone')

    assert exc.value.code == 'slot_session_mismatch'
    assert exc.value.status_code == 409
    assert 'gone' in exc.value.message
    assert client.post_calls == []


async def test_slot_listing_failure_refuses_instead_of_guessing(mgr):
    client = _FakeClient(get=[_Resp(404, text='not found')], post=[_Resp(200)])
    mgr.client = client
    path = mgr.save_state_dir / 'm.gguf.default.bin'

    with pytest.raises(server.EngineSlotError) as exc:
        await mgr._perform_slot_action(
            'save', 19500, path, model_id='m.gguf', session_digest='mine')

    assert exc.value.code == 'slot_session_mismatch'
    assert client.post_calls == []


# ── Legacy path is unchanged when no digest is supplied ──────────────────────


async def test_save_without_digest_still_uses_slot_zero_and_skips_listing(mgr):
    client = _FakeClient(post=[_Resp(200)])
    mgr.client = client
    path = mgr.save_state_dir / 'm.gguf.default.bin'

    assert await mgr._perform_slot_action(
        'save', 19500, path, model_id='m.gguf') is True

    assert client.get_calls == []  # no digest, no /slots probe
    assert client.post_calls == [(
        'http://127.0.0.1:19500/slots/0?action=save&model=m.gguf',
        {'filename': 'm.gguf.default.bin'},
    )]


async def test_restore_with_digest_is_still_a_plain_restore(mgr):
    """Restore does not resolve a cell: the digest names the SAVE source only."""
    client = _FakeClient(post=[_Resp(200)])
    mgr.client = client
    path = mgr.save_state_dir / 'm.gguf.default.bin'
    path.write_bytes(b'state')

    assert await mgr._perform_slot_action(
        'restore', 19500, path, model_id='m.gguf', session_digest='mine') is True

    assert client.get_calls == []
    assert client.post_calls == [(
        'http://127.0.0.1:19500/slots/0?action=restore&model=m.gguf',
        {'filename': 'm.gguf.default.bin'},
    )]


# ── save_state threads the digest through ───────────────────────────────────


async def test_save_state_forwards_session_digest(mgr):
    path = mgr.save_state_dir / 'm.gguf.default.bin'
    path.write_bytes(b'state')
    with patch.object(mgr, '_perform_slot_save', new=AsyncMock(return_value=True)) as save:
        await mgr.save_state('m.gguf', 'default', session_digest='mine')

    kwargs = save.call_args.kwargs
    assert kwargs['model_id'] == 'm.gguf'
    assert kwargs['session_digest'] == 'mine'
