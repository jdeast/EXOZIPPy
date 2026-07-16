"""Tests for the GUI subprocess runner (exozippy.gui.runner) and the status /
snapshot artifacts a fit emits when the GUI flag is on.

The end-to-end tests launch a real ``exozippy`` fit in a subprocess against a
tiny, fast PTDE config built from the kelt4 RV-only example, then drive it
through the runner API (start_run / status / stop / list_runs). They poll on
conditions with generous timeouts rather than sleeping a fixed amount, so a
slow machine only makes them slower, never flaky.
"""
import json
import os
import shutil
import time
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import yaml

from exozippy.gui import TERMINAL_PHASES
from exozippy.gui import runner
from exozippy.gui.status import GuiReporter, gui_enabled

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4"

# Generous ceilings: subprocess start + import + graph compile + reaching the
# first convergence check (100 stored draws) all happen inside these.
REACH_SAMPLING_TIMEOUT = 360.0
POLL_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poll_until(predicate, timeout, interval=POLL_INTERVAL):
    """Return the first truthy predicate() value within timeout, else None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def _write_ptde_config(work_dir, out_prefix, *, draws=100_000):
    """Write a fast PTDE run config into work_dir; return the config filename.

    min_ess / max_rhat are set impossibly strict so the run NEVER auto-stops
    on convergence: it keeps sampling (emitting snapshots) until the test
    stops it, which is what the lifecycle test needs. tune is tiny and the
    ladder is minimal (2 rungs) so the run reaches the first convergence
    check (100 draws) quickly.
    """
    with open(EXAMPLE_DIR / "kelt4_rvonly.yaml") as fh:
        config = yaml.safe_load(fh)
    config["prefix"] = str(out_prefix)
    config["sampler"] = {
        "method": "ptde",
        "tune": 30,
        "draws": draws,
        "n_temps": 2,
        "T_max": 5.0,
        "n_chains": 4,
        # cores=1 -> serial in-process logp eval: no fork-pool IPC overhead, so
        # the run reaches the first convergence check (100 draws) in seconds
        # instead of minutes, while still exercising the same PTDE code path.
        "cores": 1,
        "check_curvatures": False,
        "recompute_trace": True,
        "min_ess": 100_000_000,   # unreachable -> never converge
        "max_rhat": 1.0000001,    # unreachable -> never converge
    }
    config_name = "run_ptde.yaml"
    with open(work_dir / config_name, "w") as fh:
        yaml.safe_dump(config, fh)
    return config_name


@pytest.fixture
def kelt4_workdir(tmp_path):
    """A throwaway copy of the kelt4 example directory (data + params)."""
    work_dir = tmp_path / "kelt4"
    shutil.copytree(
        EXAMPLE_DIR, work_dir,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )
    return work_dir


# ---------------------------------------------------------------------------
# GuiReporter unit behavior (fast, no subprocess)
# ---------------------------------------------------------------------------

def test_gui_enabled_reads_config_and_env(monkeypatch):
    """
    Given a config flag and/or the EXOZIPPY_GUI_SNAPSHOT env var,
    When gui_enabled is evaluated,
    Then either source turns it on and neither turns it off by default.
    """
    monkeypatch.delenv("EXOZIPPY_GUI_SNAPSHOT", raising=False)
    assert gui_enabled({}) is False
    assert gui_enabled({"gui": {"snapshot": True}}) is True
    monkeypatch.setenv("EXOZIPPY_GUI_SNAPSHOT", "1")
    assert gui_enabled({}) is True


def test_disabled_reporter_writes_nothing(tmp_path):
    """
    Given a GuiReporter with enabled=False,
    When phase() and terminal() are called,
    Then no status or snapshot files are created.
    """
    prefix = tmp_path / "off" / "planet"
    reporter = GuiReporter(prefix, enabled=False)

    reporter.phase("preparing")
    reporter.progress_callback({"n_draws": 100, "n_chains": 4,
                                "stored_raw": {"x": np.zeros((4, 100))},
                                "stored_lp": np.zeros((4, 100)),
                                "raw_var_names": ["x"]})
    reporter.terminal("done")

    assert not os.path.exists(str(prefix) + "_gui_status.json")
    assert not os.path.exists(str(prefix) + "_gui_snapshot")


def test_snapshot_is_thinned_and_atomic(tmp_path):
    """
    Given 500 stored draws/chain and a 200-draw snapshot cap,
    When progress_callback writes a snapshot,
    Then the npz is thinned to <=200 draws/chain and carries every var name.
    """
    prefix = tmp_path / "run" / "planet"
    reporter = GuiReporter(prefix, enabled=True)
    stored_raw = {"x": np.random.randn(4, 500), "y": np.random.randn(4, 500)}

    reporter.progress_callback({
        "n_draws": 500, "n_chains": 4, "max_rhat": 1.03, "min_ess": 250.0,
        "elapsed_s": 2.0, "stop_reason": None,
        "stored_raw": stored_raw, "stored_lp": np.random.randn(4, 500),
        "raw_var_names": ["x", "y"]})

    snap = np.load(str(prefix) + "_gui_snapshot" + os.sep + "partial.npz")
    assert set(snap.keys()) == {"x", "y", "_lp"}
    assert snap["x"].shape[0] == 4
    assert snap["x"].shape[1] <= 200
    meta = json.load(open(str(prefix) + "_gui_snapshot" + os.sep + "partial.json"))
    assert meta["n_draws"] == 500 and meta["n_kept"] <= 200


# ---------------------------------------------------------------------------
# End-to-end subprocess lifecycle
# ---------------------------------------------------------------------------

# Real subprocess fits (startup + graph compile + a real fit + full report/plot
# output on graceful stop) legitimately exceed the 300s global pytest timeout,
# so give the two end-to-end tests a wider per-test budget.
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


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_run_without_flag_writes_no_status(kelt4_workdir, tmp_path):
    """
    Given a fit launched WITHOUT the GUI flag or env var,
    When it runs to completion,
    Then no _gui_status.json or snapshot dir is written.
    """
    import subprocess
    import sys

    out_prefix = tmp_path / "noflag" / "RUN"
    # Tiny NUTS config: 2 tune / 1 draw so this finishes fast in-process-free.
    with open(EXAMPLE_DIR / "kelt4_rvonly.yaml") as fh:
        config = yaml.safe_load(fh)
    config["prefix"] = str(out_prefix)
    config["sampler"] = {"method": "nuts", "tune": 2, "draws": 1,
                         "chains": 1, "cores": 1, "check_curvatures": False,
                         "recompute_trace": True}
    with open(kelt4_workdir / "noflag.yaml", "w") as fh:
        yaml.safe_dump(config, fh)

    env = dict(os.environ)
    env.pop("EXOZIPPY_GUI_SNAPSHOT", None)   # ensure the flag is OFF
    subprocess.run(
        [sys.executable, "-m", "exozippy.cli", "noflag.yaml"],
        cwd=str(kelt4_workdir), env=env, timeout=600, check=True)

    assert not os.path.exists(str(out_prefix) + "_gui_status.json")
    assert not os.path.exists(str(out_prefix) + "_gui_snapshot")
    # sanity: the run really did complete (trace exists)
    assert os.path.exists(str(out_prefix) + "_trace.nc")
