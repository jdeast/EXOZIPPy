"""A crashed fit must render as crashed, never as the previous run's "done".

Every run at a given output prefix writes the same ``<prefix>_gui_status.json``,
so a fit that dies before writing anything of its own used to leave the reader
looking at the file an EARLIER run left behind -- the user read "done" about a
run that had just died. These tests pin the two halves of the fix:

* the status document carries the launch's run id, and a document from any
  other launch is refused ("unknown", never "wrong");
* a run that died is reported as "error" with its exit status and the tail of
  its captured console, which is the only place a crash BEFORE the fit
  installs its reporter (a bad config, an import error) leaves a traceback.

Mostly unit-speed: RunHandle needs only ``poll()``/``pid`` from its process, so
a fake stands in. One end-to-end test launches a real subprocess that really
crashes, to prove the console capture and the whole chain.
"""

import json
import os
import time
from pathlib import Path

import pytest

from exozippy.gui import runner
from exozippy.gui.runner import RunHandle
from exozippy.gui.status import GuiReporter

PREFIX = "out/planet"


class _FakeProc:
    """The slice of Popen RunHandle uses: a pid and a poll()."""

    def __init__(self, returncode=None, pid=999999):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _write_status(tmp_path, **doc):
    path = tmp_path / (PREFIX + "_gui_status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _handle(tmp_path, returncode=None, run_id="RUN-NEW"):
    return RunHandle(
        _FakeProc(returncode), PREFIX, str(tmp_path), "cfg.yaml", run_id=run_id
    )


# ---------------------------------------------------------------------------
# A stale status file is never mistaken for the current run
# ---------------------------------------------------------------------------


def test_crashed_run_does_not_inherit_the_previous_runs_done(tmp_path):
    """
    Given a finished run left a "done" status file at this prefix,
    When a new run dies before writing any status of its own,
    Then it reports "error", not the previous run's "done".
    """
    _write_status(tmp_path, phase="done", run_id="RUN-OLD", pid=111, state={})
    handle = _handle(tmp_path, returncode=1)

    status = handle.status()

    assert status["phase"] == "error"
    assert status["alive"] is False
    assert status["stale_status"] is True
    assert "exit" not in status  # the reason carries it, not a bare key
    assert "code 1" in status["error"]
    assert status["returncode"] == 1


def test_stale_status_is_unknown_not_wrong_while_the_new_run_starts(tmp_path):
    """
    Given the same stale "done" file,
    When the new run is still alive but has not written its status yet,
    Then it reports "starting" -- unknown, rather than someone else's answer.
    """
    _write_status(tmp_path, phase="done", run_id="RUN-OLD", pid=111)
    handle = _handle(tmp_path, returncode=None)

    status = handle.status()

    assert status["phase"] == "starting"
    assert status["alive"] is True
    assert status["stale_status"] is True


def test_this_runs_own_terminal_status_is_reported_verbatim(tmp_path):
    """
    Given a run that wrote its own "done" status,
    When it has exited,
    Then "done" is reported unchanged -- the run-id check must not turn a
      genuine success into a failure.
    """
    _write_status(tmp_path, phase="done", run_id="RUN-NEW", pid=222, state={})
    handle = _handle(tmp_path, returncode=0)

    status = handle.status()

    assert status["phase"] == "done"
    assert status["alive"] is False
    assert not status.get("stale_status")


def test_this_runs_recorded_error_traceback_survives(tmp_path):
    """
    Given a fit that crashed inside run_fit and recorded its traceback
      (the PR #46 mechanism),
    When the handle reports status,
    Then phase and traceback are passed through untouched.
    """
    _write_status(
        tmp_path,
        phase="error",
        run_id="RUN-NEW",
        pid=222,
        error="Traceback (most recent call last):\nValueError: boom",
    )
    handle = _handle(tmp_path, returncode=1)

    status = handle.status()

    assert status["phase"] == "error"
    assert "ValueError: boom" in status["error"]


def test_a_run_that_died_mid_phase_reports_error(tmp_path):
    """
    Given this run's own status file stuck on a non-terminal phase,
    When the process is gone,
    Then the phase is forced to "error" with the exit status.
    """
    _write_status(tmp_path, phase="sampling", run_id="RUN-NEW", pid=222)
    handle = _handle(tmp_path, returncode=-9)

    status = handle.status()

    assert status["phase"] == "error"
    assert "code -9" in status["error"]


def test_no_status_file_at_all_is_an_error_when_dead(tmp_path):
    """
    Given a run that never wrote a status file,
    When the process is gone,
    Then it reports "error" with the exit status (never-run, running,
      finished and crashed all stay distinguishable).
    """
    handle = _handle(tmp_path, returncode=2)

    status = handle.status()

    assert status["phase"] == "error"
    assert "code 2" in status["error"]
    assert not status.get("stale_status")


def test_handle_without_a_run_id_keeps_the_old_behavior(tmp_path):
    """
    Given a handle with no run id (a hand-built one, e.g. in a test),
    When a status document is present,
    Then it is read as before -- the check only tightens runner-launched fits.
    """
    _write_status(tmp_path, phase="done", pid=111)
    handle = _handle(tmp_path, returncode=0, run_id=None)

    assert handle.status()["phase"] == "done"


# ---------------------------------------------------------------------------
# The crash itself is visible
# ---------------------------------------------------------------------------


def test_console_tail_is_reported_as_the_error(tmp_path):
    """
    Given a crashed run whose captured console holds a traceback,
    When the handle reports status,
    Then the traceback is in the reported error, labeled as console output.
    """
    console = tmp_path / (PREFIX + "_gui_console.log")
    console.parent.mkdir(parents=True, exist_ok=True)
    console.write_text(
        "Traceback (most recent call last):\n"
        '  File "cli.py", line 1\n'
        "ValueError: bad config\n"
    )
    handle = _handle(tmp_path, returncode=1)

    error = handle.status()["error"]

    assert "ValueError: bad config" in error
    assert "console" in error
    assert "code 1" in error  # the exit status is labeled, not implied


def test_console_tail_is_bounded(tmp_path):
    """
    Given a long-running fit that printed megabytes before dying,
    When its console tail is read,
    Then only the last few KB are reported.
    """
    console = tmp_path / (PREFIX + "_gui_console.log")
    console.parent.mkdir(parents=True, exist_ok=True)
    console.write_text("x" * 200_000 + "\nthe last line\n")
    handle = _handle(tmp_path, returncode=1)

    error = handle.status()["error"]

    assert "the last line" in error
    assert len(error) < runner.MAX_CONSOLE_TAIL + 500


# ---------------------------------------------------------------------------
# The reporter's half of the contract
# ---------------------------------------------------------------------------


def test_reporter_stamps_the_run_id_from_the_environment(
    tmp_path, monkeypatch
):
    """
    Given a fit launched by the runner (which exports the run id),
    When the reporter writes a status file,
    Then the document carries that id, which is what the reader matches on.
    """
    monkeypatch.setenv(runner.RUN_ID_ENV, "RUN-NEW")

    GuiReporter(tmp_path / PREFIX, enabled=True).phase("preparing")

    doc = json.loads((tmp_path / (PREFIX + "_gui_status.json")).read_text())
    assert doc["run_id"] == "RUN-NEW"


def test_reporter_omits_the_run_id_outside_a_runner_launch(
    tmp_path, monkeypatch
):
    """
    Given a fit started by hand with gui.snapshot on,
    When the reporter writes a status file,
    Then no run id is stamped and nothing else changes.
    """
    monkeypatch.delenv(runner.RUN_ID_ENV, raising=False)

    GuiReporter(tmp_path / PREFIX, enabled=True).phase("preparing")

    doc = json.loads((tmp_path / (PREFIX + "_gui_status.json")).read_text())
    assert "run_id" not in doc
    assert doc["phase"] == "preparing"


def test_list_runs_reports_the_run_id(tmp_path):
    """
    Given a status file with a run id,
    When list_runs summarizes the tree,
    Then the id is carried through for callers that match on it.
    """
    _write_status(
        tmp_path, phase="done", run_id="RUN-OLD", pid=1, updated_at=1.0
    )

    runs = runner.list_runs(tmp_path)

    assert [r["run_id"] for r in runs] == ["RUN-OLD"]


# ---------------------------------------------------------------------------
# The status endpoint the GUI polls
# ---------------------------------------------------------------------------


def test_status_endpoint_renders_a_crash_as_failed(tmp_path, monkeypatch):
    """
    Given a run that crashed after a previous run left a "done" file,
    When the GUI polls GET /api/run/status,
    Then the payload says the run is over and failed, and carries the error --
      instead of repeating the previous run's success.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    _write_status(tmp_path, phase="done", run_id="RUN-OLD", pid=1)
    console = tmp_path / (PREFIX + "_gui_console.log")
    console.write_text("ValueError: bad config\n")
    handle = _handle(tmp_path, returncode=1)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: handle)

    client = TestClient(create_app())
    client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )
    body = client.get("/api/run/status").json()

    assert body["phase"] == "error"
    assert body["terminal"] is True
    assert body["alive"] is False
    assert body["stale_status"] is True
    assert "ValueError: bad config" in body["error"]
    assert body["console_path"].endswith("_gui_console.log")


def test_status_endpoint_marks_a_finished_run_terminal(tmp_path, monkeypatch):
    """
    Given a run that finished normally,
    When the GUI polls status,
    Then it is terminal but not an error -- crashed and finished stay
      distinguishable.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    _write_status(tmp_path, phase="done", run_id="RUN-NEW", pid=1, state={})
    handle = _handle(tmp_path, returncode=0)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: handle)

    client = TestClient(create_app())
    client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )
    body = client.get("/api/run/status").json()

    assert body["phase"] == "done"
    assert body["terminal"] is True
    assert body["error"] is None


# ---------------------------------------------------------------------------
# End to end: a real subprocess that really crashes
# ---------------------------------------------------------------------------


def test_real_crashing_subprocess_reports_its_traceback(tmp_path):
    """
    Given a previous run's "done" status file, and a config the fit refuses,
    When the run is launched for real and dies before it can report anything,
    Then the GUI-facing status is "error" carrying the child's traceback --
      the audit's scenario, end to end.
    """
    config = tmp_path / "bad.yaml"
    # Refused by the shared YAML guard in exozippy.cli, i.e. a crash BEFORE
    # run_fit exists to record a traceback of its own.
    config.write_text(
        f"prefix: {PREFIX}\nlens:\n  - name: L\n    finite_source: no\n"
    )
    _write_status(tmp_path, phase="done", run_id="RUN-OLD", pid=1)

    handle = runner.start_run("bad.yaml", cwd=str(tmp_path))
    assert handle.wait(timeout=300) != 0

    status = handle.status()

    assert status["phase"] == "error"
    assert status["stale_status"] is True
    assert "finite_source" in status["error"]
    assert os.path.exists(handle.console_path)
