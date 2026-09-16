"""Tests for the GUI's log-tail WebSocket (``WS /api/logs``).

This endpoint had no coverage at all, which is how three separate faults
survived in it: it streamed ANY readable file on the machine (gui.md invariant
5 says file-serving endpoints stay confined to their intended tree), it
accepted cross-origin handshakes (WebSocket handshakes bypass CORS, so any page
the user visits could open it against the loopback server), and it never awaited
a client frame, so a tail on a quiet file looped forever holding an open file
handle after the browser had gone.

The streaming/rotation behavior is driven through a fake WebSocket rather than
the TestClient: it lets the tests assert that the coroutine RETURNS, which is
precisely what the disconnect fix is about.
"""

import asyncio
import contextlib
import time

import pytest


class _FakeWebSocket:
    """A WebSocket that records sends and disconnects after a set delay."""

    def __init__(self, disconnect_after=0.4):
        self.sent = []
        self._disconnect_after = disconnect_after
        self._deadline = None

    async def send_text(self, text):
        self.sent.append(text)

    async def receive(self):
        if self._deadline is None:
            self._deadline = time.monotonic() + self._disconnect_after
        remaining = self._deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        return {"type": "websocket.disconnect"}


def _run_tail(path, disconnect_after=0.4, timeout=10.0):
    """Run ``_tail_log`` against a fake socket; return the lines it sent."""
    pytest.importorskip("fastapi")
    from exozippy.gui.app import _tail_log

    ws = _FakeWebSocket(disconnect_after=disconnect_after)

    async def main():
        await asyncio.wait_for(
            _tail_log(ws, str(path), poll_s=0.01), timeout=timeout
        )

    asyncio.run(main())
    return ws.sent


# --- streaming ---------------------------------------------------------------


def test_tail_sends_the_existing_tail_then_returns_on_disconnect(tmp_path):
    """
    Given a log file with content,
    When a client tails it and then disconnects,
    Then the existing lines are sent and the coroutine returns.
    """
    log = tmp_path / "run.log"
    log.write_text("first\nsecond\n")

    sent = _run_tail(log, disconnect_after=0.1)

    assert sent[:2] == ["first", "second"]


def test_tail_of_a_quiet_file_ends_when_the_client_disconnects(tmp_path):
    """
    Given a log file that never grows,
    When the client disconnects,
    Then the tail returns instead of polling forever.

    This is the regression: the handler only noticed a disconnect when it next
    tried to SEND, so a quiet file (a finished run's log, or one that never
    appears) left an immortal poller holding a file handle -- one per file
    switch, since LogTerminal opens a fresh socket each time. The
    ``asyncio.wait_for`` in ``_run_tail`` is what fails if that regresses.
    """
    log = tmp_path / "quiet.log"
    log.write_text("only line\n")

    sent = _run_tail(log, disconnect_after=0.2)

    assert sent == ["only line"]


def test_tail_of_a_missing_file_ends_when_the_client_disconnects(tmp_path):
    """
    Given a log file that does not exist yet,
    When the client disconnects,
    Then the tail returns (the wait-for-creation loop is not immortal either).
    """
    sent = _run_tail(tmp_path / "not-created-yet.log", disconnect_after=0.2)

    assert sent == []


def test_tail_streams_a_file_created_after_the_connection(tmp_path):
    """
    Given a tail attached to a log the fit has not written yet,
    When the file appears,
    Then its content is streamed from the start -- there is no earlier tail
    that could already have carried it.
    """
    pytest.importorskip("fastapi")
    from exozippy.gui.app import _tail_log

    log = tmp_path / "late.log"
    ws = _FakeWebSocket(disconnect_after=1.0)

    async def main():
        async def create():
            await asyncio.sleep(0.15)
            log.write_text("first line of the run\n")

        await asyncio.gather(
            asyncio.wait_for(_tail_log(ws, str(log), poll_s=0.01), timeout=10),
            create(),
        )

    asyncio.run(main())

    assert ws.sent == ["first line of the run"]


def test_tail_follows_appended_content(tmp_path):
    """
    Given a tail in progress,
    When lines are appended,
    Then they are streamed without resending the seeded tail.
    """
    pytest.importorskip("fastapi")
    from exozippy.gui.app import _tail_log

    log = tmp_path / "run.log"
    log.write_text("seed\n")
    ws = _FakeWebSocket(disconnect_after=1.0)

    async def main():
        async def append():
            await asyncio.sleep(0.15)
            with open(log, "a") as fh:
                fh.write("appended\n")

        await asyncio.gather(
            asyncio.wait_for(_tail_log(ws, str(log), poll_s=0.01), timeout=10),
            append(),
        )

    asyncio.run(main())

    assert ws.sent == ["seed", "appended"]


def test_tail_follows_rotation_to_a_new_inode(tmp_path):
    """
    Given a tail in progress,
    When the file is rotated away and a fresh one takes its place,
    Then the new file's content is streamed from its start.
    """
    pytest.importorskip("fastapi")
    from exozippy.gui.app import _tail_log

    log = tmp_path / "run.log"
    log.write_text("old\n")
    ws = _FakeWebSocket(disconnect_after=1.0)

    async def main():
        async def rotate():
            await asyncio.sleep(0.15)
            log.rename(tmp_path / "run.log.1")
            (tmp_path / "run.log").write_text("rotated\n")

        await asyncio.gather(
            asyncio.wait_for(_tail_log(ws, str(log), poll_s=0.01), timeout=10),
            rotate(),
        )

    asyncio.run(main())

    assert ws.sent[0] == "old"
    assert "rotated" in ws.sent


# --- confinement + origin (via the real endpoint) -----------------------------


@pytest.fixture
def project_client(tmp_path):
    """A TestClient whose server has ``tmp_path`` open as its project."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    client = TestClient(create_app(project_dir=str(tmp_path)))
    client.post("/api/project/open", json={"path": str(tmp_path)})
    return client


@contextlib.contextmanager
def _open_tail(client, url, **kwargs):
    """Open the log socket and tolerate the TestClient's teardown race.

    ``WebSocketTestSession.__exit__`` sends the close frame and then STOPS THE
    PORTAL, cancelling the app task if it has not yet processed the
    disconnect, and re-raises that cancellation at exit. That is a property of
    the harness's teardown, not of the endpoint -- a real server simply waits
    for the handler to return -- and it is timing-dependent: it showed up only
    on the slower macOS CI runner. Every assertion runs before teardown, so
    suppressing it there hides nothing. The handler's own "does it stop when
    the client goes away" contract is pinned by the fake-socket tests above,
    which assert the coroutine RETURNS.
    """
    session = client.websocket_connect(url, **kwargs)
    socket = session.__enter__()
    try:
        yield socket
    finally:
        with contextlib.suppress(BaseException):
            session.__exit__(None, None, None)


def test_logs_socket_streams_a_file_inside_the_project(
    project_client, tmp_path
):
    """
    Given a file inside the open project,
    When the log socket is opened on it,
    Then its content is streamed.
    """
    log = tmp_path / "fit.log"
    log.write_text("hello\n")

    with _open_tail(project_client, f"/api/logs?file={log}") as socket:
        assert socket.receive_text() == "hello"


def test_logs_socket_refuses_a_file_outside_the_project(
    project_client, tmp_path
):
    """
    Given a readable file OUTSIDE the open project,
    When the log socket is asked to tail it,
    Then it refuses instead of streaming.

    The path arrived verbatim in the query string with no confinement at all,
    so this socket was an arbitrary-file read on the user's machine -- reachable
    from any web page, since WebSocket handshakes are not subject to CORS.
    """
    outside = tmp_path.parent / "outside-the-project.txt"
    outside.write_text("secret\n")
    try:
        with _open_tail(project_client, f"/api/logs?file={outside}") as socket:
            message = socket.receive_text()
        assert "refusing" in message
        assert "secret" not in message
    finally:
        outside.unlink()


def test_logs_socket_refuses_a_cross_origin_handshake(
    project_client, tmp_path
):
    """
    Given a handshake carrying a non-local Origin,
    When it reaches the log socket,
    Then it is closed before being accepted.
    """
    from starlette.websockets import WebSocketDisconnect

    log = tmp_path / "fit.log"
    log.write_text("hello\n")

    with pytest.raises(WebSocketDisconnect):
        with project_client.websocket_connect(
            f"/api/logs?file={log}",
            headers={"origin": "http://evil.example"},
        ) as socket:
            socket.receive_text()


def test_logs_socket_allows_a_localhost_origin(project_client, tmp_path):
    """
    Given a handshake from the GUI's own page,
    When it reaches the log socket,
    Then it is accepted -- the Origin check must not break the real client.
    """
    log = tmp_path / "fit.log"
    log.write_text("hello\n")

    with _open_tail(
        project_client,
        f"/api/logs?file={log}",
        headers={"origin": "http://127.0.0.1:8931"},
    ) as socket:
        assert socket.receive_text() == "hello"


def test_origin_predicate_accepts_only_loopback():
    """Given assorted Origins, When checked, Then only loopback ones pass.

    A MISSING Origin is allowed on purpose: browsers always send one on a WS
    handshake, so its absence means a non-browser client (curl, a script, the
    test client), which was never the threat this check addresses.
    """
    pytest.importorskip("fastapi")
    from exozippy.gui.app import _origin_is_local

    assert _origin_is_local(None)
    assert _origin_is_local("")
    assert _origin_is_local("http://127.0.0.1:5173")
    assert _origin_is_local("http://localhost:8000")
    assert not _origin_is_local("http://evil.example")
    assert not _origin_is_local("https://127.0.0.1.evil.example")
