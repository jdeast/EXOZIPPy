"""End-to-end GUI-runner lifecycle test, split into its own file so the xdist
`--dist loadfile` scheduler runs it on a separate worker from the other slow
subprocess-runner tests (they used to share tests/test_runner.py and therefore
one worker, serializing ~3 minutes of real fits). Shared helpers and the
kelt4 workdir fixture are imported from test_runner (tests/ is on sys.path).

See test_runner.py for why these must be real subprocess fits.
"""

import os

import arviz as az
import numpy as np
import pytest
from test_runner import (  # noqa: E402  (tests/ is on sys.path via conftest)
    REACH_SAMPLING_TIMEOUT,
    _poll_until,
    _write_ptde_config,
    kelt4_workdir,
)

from exozippy.gui import TERMINAL_PHASES, runner

# Poll budgets. Every wait here must add up to comfortably LESS than the
# @pytest.mark.timeout below, or the guard kills the test before its own polls
# give up and the failure is reported as a dead xdist worker with no test name
# instead of the assertion that says what was actually stuck.
#
#   REACH_SAMPLING_TIMEOUT (360 s, from test_runner)  subprocess + imports +
#       model build + a COLD pytensor compile + tune + 100 draws
#   SNAPSHOT_TIMEOUT       30 s   file-visibility lag only (see below)
#   GRACEFUL_EXIT_TIMEOUT  240 s  wrap-up: save trace + reports/plots
#   FORCE_EXIT_TIMEOUT     30 s   reaping a process already sent SIGKILL
#
# Worst case, counting the ~50 s that handle.stop(force=True) itself blocks for
# (graceful_timeout 30 + kill_timeout 10 + 10): 360 + 30 + 240 + 50 + 30 + 50 +
# 30 = 790 s < 900 s. Warm, the whole test measures 57-81 s.
SNAPSHOT_TIMEOUT = 30.0
GRACEFUL_EXIT_TIMEOUT = 240.0
FORCE_EXIT_TIMEOUT = 30.0


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_run_lifecycle_status_snapshot_and_graceful_stop(
    kelt4_workdir, tmp_path
):
    """
    Given a fit launched via start_run with the GUI flag,
    When it reaches the sampling phase and is then stopped,
    Then status.json advances to "sampling", a snapshot npz appears, the run
    exits on a terminal phase, and a valid trace .nc is left behind.
    """
    out_prefix = tmp_path / "out" / "RUN"
    config_name = _write_ptde_config(kelt4_workdir, out_prefix)

    handle = runner.start_run(config_name, cwd=kelt4_workdir)
    forced = []  # records any force escalation, for the final message
    try:
        # 1. reaches a convergence check during sampling. gui.phase("sampling")
        # is written the instant sampling starts, but n_draws only appears once
        # the first geometric convergence check (>=100 stored draws) fires the
        # progress hook -- that is the meaningful "sampling with progress" state.
        # (This is also a status.json update DURING sampling, satisfying the
        # "updates at least once" requirement.)
        def _sampling_with_progress():
            st = handle.status()
            if not handle.is_alive() and st.get("phase") not in ("sampling",):
                return True  # died/finished; assertion below inspects it
            return (
                st.get("phase") == "sampling"
                and st.get("state", {}).get("n_draws", 0) >= 100
            )

        assert _poll_until(_sampling_with_progress, REACH_SAMPLING_TIMEOUT), (
            f"run never reported n_draws>=100 during sampling within "
            f"{REACH_SAMPLING_TIMEOUT}s; alive={handle.is_alive()}, "
            f"last status: {handle.status()}"
        )
        status = handle.status()
        assert status["phase"] == "sampling", f"unexpected phase {status}"
        assert status["state"].get("n_draws", 0) >= 100

        # 2. the snapshot artifacts written by that same convergence check
        # exist. The npz is written by the very convergence check that
        # published n_draws>=100, so it is already on disk (or one atomic
        # rename away) by the time the poll above returns: this budget covers
        # file-visibility lag, not model work.
        snap_npz = os.path.join(handle.snapshot_dir, "partial.npz")
        found = _poll_until(
            lambda: os.path.exists(snap_npz), timeout=SNAPSHOT_TIMEOUT
        )
        listing = (
            sorted(os.listdir(handle.snapshot_dir))
            if os.path.isdir(handle.snapshot_dir)
            else "<snapshot dir does not exist>"
        )
        assert found, (
            f"snapshot npz {snap_npz} never appeared within "
            f"{SNAPSHOT_TIMEOUT}s; snapshot dir contains {listing}"
        )
        snap = np.load(snap_npz)
        assert "_lp" in snap and any(k.endswith("_raw") for k in snap.files)

        # 3. graceful stop (single SIGINT) as early as possible -> the run
        # wraps up on its own (saves the partial trace, writes reports) and
        # exits on a terminal phase, without a premature force escalation.
        # Wrap-up is ~20-30 s warm (trace save + report/plot generation, which
        # compiles its own pytensor plotters); 240 s is ~8x that, room for a
        # cold compile cache, and still leaves the escalation below inside the
        # 900 s mark.
        handle.stop(force=False)
        ended = _poll_until(
            lambda: not handle.is_alive(), timeout=GRACEFUL_EXIT_TIMEOUT
        )
        if not ended:
            forced.append("graceful stop timed out; escalated to force")
            handle.stop(force=True)
            handle.wait(timeout=FORCE_EXIT_TIMEOUT)
    finally:
        if handle.is_alive():
            forced.append("still alive at teardown; forced")
            handle.stop(force=True)
            handle.wait(timeout=FORCE_EXIT_TIMEOUT)

    final = handle.status()
    assert final["phase"] in {"stopped", "done"}, (
        f"non-terminal end: {final}"
        + (f"; escalations: {forced}" if forced else "")
    )
    assert final["phase"] in TERMINAL_PHASES

    # 4. a usable trace was written and opens in arviz.
    trace_path = str(out_prefix) + "_trace.nc"
    assert os.path.exists(trace_path), "no trace .nc written"
    idata = az.from_netcdf(trace_path)
    assert "posterior" in idata
