"""
Tests for the interim per-mode output loop (run.py, notes/multimode_
implementation.txt P7): loop the existing single-posterior corner/component
plot calls once per detected mode, restricted to that mode's draws.

Per-mode LaTeX columns and CSV rows already exist (outputs/latex.py's
mode_report kwarg) and are not touched here -- these tests cover only the
new plot-output loop (_emit_per_mode_outputs), the mode-filtered draw
extraction (get_draws(mode=...)), and the mode-restricted idata builder
(_idata_for_mode).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import arviz as az
import pytest

from exozippy.outputs.modes import identify_modes, mode_suffix
from exozippy.run import get_draws, _idata_for_mode, _emit_per_mode_outputs


N_CHAIN, N_DRAW = 8, 500
N = N_CHAIN * N_DRAW


def _make_idata(posterior, lp):
    return az.from_dict({
        "posterior": {k: np.asarray(v).reshape(N_CHAIN, N_DRAW)
                      for k, v in posterior.items()},
        "sample_stats": {"lp": np.asarray(lp).reshape(N_CHAIN, N_DRAW)},
    })


def _two_mode_idata(rng, w2=0.3, sep=8.0):
    """Two Gaussian modes mixed within every chain (mirrors
    tests/test_mode_report.py's _two_mode_idata). Every sampled variable
    gets both a physical name ('x') and a '_raw' companion ('x_raw',
    identical values here) to mirror production traces, where identify_modes
    clusters on the *_raw variables but the per-mode corner plot draws the
    physical ones (make_corner excludes anything with '_raw' in its name).
    """
    labels = (rng.random(N) < w2).astype(int)
    x = rng.normal(0, 1, N) + sep * labels
    y = rng.normal(0, 1, N) - 0.6 * sep * labels
    lp = rng.normal(1000, 3, N) - 5 * labels
    return _make_idata({"x": x, "x_raw": x, "y": y, "y_raw": y}, lp)


def _single_mode_idata(rng):
    x = rng.normal(0, 1, N)
    y = rng.normal(0, 1, N)
    lp = rng.normal(1000, 3, N)
    return _make_idata({"x": x, "x_raw": x, "y": y, "y_raw": y}, lp)


class _FakeSystem:
    """Minimal System stand-in: no components, empty param lookup -- enough
    to exercise _emit_per_mode_outputs' plotting loop without building a
    real component graph."""
    active_components = {}

    def get_parameter_lookup(self):
        return {}


# ----------------------------------------------------------------------
# get_draws(mode=...)
# ----------------------------------------------------------------------

def test_get_draws_mode_filter_disjoint():
    """
    Given a two-mode trace,
    When get_draws is called once per mode,
    Then the two draw sets are disjoint and mode 0 (the majority, centered
      near x=0) has a smaller mean |x| than mode 1.
    """
    rng = np.random.default_rng(1)
    idata = _two_mode_idata(rng)
    rep = identify_modes(idata)
    assert rep.n_modes == 2

    draws0 = get_draws(idata, n_draws=200, mode=0)
    draws1 = get_draws(idata, n_draws=200, mode=1)

    x0 = {float(d["x"]) for d in draws0}
    x1 = {float(d["x"]) for d in draws1}
    assert x0.isdisjoint(x1)
    assert np.mean(np.abs(list(x0))) < np.mean(np.abs(list(x1)))


def test_get_draws_mode_missing_variable_raises():
    """
    Given an idata with no posterior['mode'] variable,
    When get_draws is called with an explicit mode,
    Then it raises rather than silently returning the whole posterior.
    """
    rng = np.random.default_rng(2)
    idata = az.from_dict({"posterior": {"a": rng.normal(size=(2, 50))}})

    with pytest.raises(ValueError, match="mode"):
        get_draws(idata, mode=0)


# ----------------------------------------------------------------------
# _idata_for_mode
# ----------------------------------------------------------------------

def test_idata_for_mode_restricts_and_flattens():
    """
    Given a two-mode trace,
    When _idata_for_mode builds a per-mode view,
    Then it contains only that mode's draws, flattened into a single
      synthetic chain, with no 'mode' variable of its own, and the two
      modes' 'x' distributions are cleanly separated.
    """
    rng = np.random.default_rng(3)
    idata = _two_mode_idata(rng)
    rep = identify_modes(idata)

    idata0 = _idata_for_mode(idata, 0)
    idata1 = _idata_for_mode(idata, 1)

    assert "mode" not in idata0.posterior
    assert idata0.posterior.sizes["chain"] == 1
    assert idata0.posterior.sizes["draw"] == rep.modes[0].n_draws
    assert idata1.posterior.sizes["draw"] == rep.modes[1].n_draws

    x0 = idata0.posterior["x"].values
    x1 = idata1.posterior["x"].values
    assert x0.mean() < x1.mean() - 3  # sep=8, both std~1: no overlap


# ----------------------------------------------------------------------
# _emit_per_mode_outputs
# ----------------------------------------------------------------------

def test_emit_per_mode_outputs_writes_named_corner_files(tmp_path):
    """
    Given a two-mode trace and a component-free fake system,
    When _emit_per_mode_outputs runs,
    Then a corner PNG is written per mode, suffixed with mode_suffix's
      naming (modeone/modetwo), and nothing else is written (no components
      means no comp.plot output, no CSV/latex duplication).
    """
    rng = np.random.default_rng(4)
    idata = _two_mode_idata(rng)
    rep = identify_modes(idata)
    assert rep.n_modes == 2

    prefix = tmp_path / "fit"
    _emit_per_mode_outputs(_FakeSystem(), None, idata, rep, prefix)

    expected = sorted([
        f"fit_corner_{mode_suffix(0)}.png",
        f"fit_corner_{mode_suffix(1)}.png",
    ])
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == expected
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0


def test_emit_per_mode_outputs_calls_component_plot_per_mode(tmp_path):
    """
    Given a two-mode trace and a fake system with one recording component,
    When _emit_per_mode_outputs runs,
    Then comp.plot is called once per mode with a mode-specific
      filename_prefix and a draws list restricted to that mode.
    """
    rng = np.random.default_rng(6)
    idata = _two_mode_idata(rng)
    rep = identify_modes(idata)

    calls = []

    class _RecordingComponent:
        def plot(self, system, points, filename_prefix="debug"):
            calls.append((filename_prefix, points))

    class _SystemWithComponent(_FakeSystem):
        active_components = {"dummy": _RecordingComponent()}

    prefix = tmp_path / "fit"
    _emit_per_mode_outputs(_SystemWithComponent(), None, idata, rep, prefix)

    assert len(calls) == 2
    prefixes = sorted(p for p, _ in calls)
    assert prefixes == sorted([
        f"{prefix}_mcmc_{mode_suffix(0)}",
        f"{prefix}_mcmc_{mode_suffix(1)}",
    ])
    for _, points in calls:
        assert len(points) > 0


# ----------------------------------------------------------------------
# single-mode guard (mirrors the `mode_report.n_modes > 1` check in run_fit)
# ----------------------------------------------------------------------

def test_single_mode_report_skips_the_loop_guard():
    """
    Given a unimodal trace,
    When identify_modes runs,
    Then n_modes == 1, so run_fit's guard (`mode_report.n_modes > 1`) is
      False and _emit_per_mode_outputs is never invoked -- single-mode runs
      get zero new per-mode files.
    """
    rng = np.random.default_rng(5)
    idata = _single_mode_idata(rng)

    rep = identify_modes(idata)

    assert rep.n_modes == 1
    assert not (rep.n_modes > 1)
