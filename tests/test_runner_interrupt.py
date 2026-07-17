"""End-to-end GUI-runner interrupt test, split into its own file so xdist's
`--dist loadfile` scheduler runs it on a separate worker from the other slow
subprocess-runner tests. Shared helpers/fixture imported from test_runner.
"""
import os

import pytest

from exozippy.gui import TERMINAL_PHASES
from exozippy.gui import runner

from test_runner import (  # noqa: E402  (tests/ is on sys.path via conftest)
    REACH_SAMPLING_TIMEOUT,
    _poll_until,
    _write_ptde_config,
    kelt4_workdir,
)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_interrupt_during_prepare_leaves_terminal_phase(kelt4_workdir, tmp_path):
    """
    Given a fit interrupted almost immediately (before/around prepare),
    When it is stopped and exits,
    Then the status is never left stranded on a non-terminal phase.
    """
    out_prefix = tmp_path / "out2" / "RUN"
    config_name = _write_ptde_config(kelt4_workdir, out_prefix)

    handle = runner.start_run(config_name, cwd=kelt4_workdir)
    try:
        # Wait only until the run has entered run_fit (status file exists),
        # then interrupt -- this is the prepare/compile window, well before
        # any draws are stored.
        appeared = _poll_until(
            lambda: os.path.exists(handle.status_path) or not handle.is_alive(),
            timeout=REACH_SAMPLING_TIMEOUT)
        assert appeared, "run never wrote an initial status or exited"

        handle.stop(force=True)
        rc = handle.wait(timeout=120.0)
        assert rc is not None, "process did not exit after stop"
    finally:
        if handle.is_alive():
            handle.stop(force=True)
            handle.wait(timeout=60.0)

    final = handle.status()
    assert final["phase"] in TERMINAL_PHASES, \
        f"status left on non-terminal phase: {final}"

    # list_runs finds the run and reports the same terminal phase.
    summaries = runner.list_runs(tmp_path)
    matching = [s for s in summaries
                if s["status_path"] == handle.status_path]
    assert matching, "list_runs did not find the run"
    assert matching[0]["phase"] in TERMINAL_PHASES
