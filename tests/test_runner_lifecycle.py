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
def test_run_lifecycle_status_snapshot_and_graceful_stop(kelt4_workdir, tmp_path):
    """
    Given a fit launched via start_run with the GUI flag,
    When it reaches the sampling phase and is then stopped,
    Then status.json advances to "sampling", a snapshot npz appears, the run
    exits on a terminal phase, and a valid trace .nc is left behind.
    """
    out_prefix = tmp_path / "out" / "RUN"
    config_name = _write_ptde_config(kelt4_workdir, out_prefix)

    handle = runner.start_run(config_name, cwd=kelt4_workdir)
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
                return True   # died/finished; assertion below inspects it
            return (st.get("phase") == "sampling"
                    and st.get("state", {}).get("n_draws", 0) >= 100)

        assert _poll_until(_sampling_with_progress, REACH_SAMPLING_TIMEOUT), \
            "run never reported n_draws>=100 during sampling"
        status = handle.status()
        assert status["phase"] == "sampling", f"unexpected phase {status}"
        assert status["state"].get("n_draws", 0) >= 100

        # 2. the snapshot artifacts written by that same convergence check exist.
        snap_npz = os.path.join(handle.snapshot_dir, "partial.npz")
        assert _poll_until(lambda: os.path.exists(snap_npz), timeout=60.0), \
            "snapshot npz never appeared"
        snap = np.load(snap_npz)
        assert "_lp" in snap and any(k.endswith("_raw") for k in snap.files)

        # 3. graceful stop (single SIGINT) as early as possible -> the run
        # wraps up on its own (saves the partial trace, writes reports) and
        # exits on a terminal phase, without a premature force escalation.
        handle.stop(force=False)
        ended = _poll_until(lambda: not handle.is_alive(), timeout=600.0)
        if not ended:
            handle.stop(force=True)
            handle.wait(timeout=60.0)
    finally:
        if handle.is_alive():
            handle.stop(force=True)
            handle.wait(timeout=60.0)

    final = handle.status()
    assert final["phase"] in {"stopped", "done"}, f"non-terminal end: {final}"
    assert final["phase"] in TERMINAL_PHASES

    # 4. a usable trace was written and opens in arviz.
    trace_path = str(out_prefix) + "_trace.nc"
    assert os.path.exists(trace_path), "no trace .nc written"
    idata = az.from_netcdf(trace_path)
    assert "posterior" in idata
