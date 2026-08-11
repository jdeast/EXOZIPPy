"""End-to-end GUI-runner interrupt test, split into its own file so xdist's
`--dist loadfile` scheduler runs it on a separate worker from the other slow
subprocess-runner tests. Shared helpers/fixture imported from test_runner.
"""

import os

import pytest
from test_runner import (  # noqa: E402  (tests/ is on sys.path via conftest)
    REACH_STATUS_FILE_TIMEOUT,
    _poll_until,
    _write_ptde_config,
    kelt4_workdir,
)

from exozippy.gui import TERMINAL_PHASES, runner

# Budget: 180 s to see the first status doc (no model compile is involved --
# see test_runner.REACH_STATUS_FILE_TIMEOUT) + 60 s to reap a process that
# stop(force=True) has already SIGINT-SIGINT-SIGKILLed (that call itself blocks
# ~50 s) + 60 s for the same in teardown = ~350 s worst case, well under the
# 900 s mark below, so a hang fails on the assertion rather than the guard.
FORCE_EXIT_TIMEOUT = 60.0


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_interrupt_during_prepare_leaves_terminal_phase(
    kelt4_workdir, tmp_path
):
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
            lambda: (
                os.path.exists(handle.status_path) or not handle.is_alive()
            ),
            timeout=REACH_STATUS_FILE_TIMEOUT,
        )
        assert appeared, (
            f"run never wrote an initial status at {handle.status_path} nor "
            f"exited within {REACH_STATUS_FILE_TIMEOUT}s; pid={handle.pid}"
        )

        handle.stop(force=True)
        rc = handle.wait(timeout=FORCE_EXIT_TIMEOUT)
        assert rc is not None, (
            f"process {handle.pid} did not exit within {FORCE_EXIT_TIMEOUT}s "
            "after stop(force=True) (which already SIGKILLed it)"
        )
    finally:
        if handle.is_alive():
            handle.stop(force=True)
            handle.wait(timeout=FORCE_EXIT_TIMEOUT)

    final = handle.status()
    assert final["phase"] in TERMINAL_PHASES, (
        f"status left on non-terminal phase: {final}"
    )

    # list_runs finds the run and reports the same terminal phase.
    summaries = runner.list_runs(tmp_path)
    matching = [s for s in summaries if s["status_path"] == handle.status_path]
    assert matching, "list_runs did not find the run"
    assert matching[0]["phase"] in TERMINAL_PHASES
