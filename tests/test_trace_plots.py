"""
Tests for the detailed trace-plot output (run.py).

The ArviZ 1.0 migration silently changed az.plot_trace to render only the
trace lines; the old dist + trace two-column layout is now plot_trace_dist.
These tests pin the restored layout.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import arviz as az
import matplotlib.pyplot as plt
import pytest

import exozippy.run as run_mod
from exozippy.run import _render_trace_page, save_multipage_trace


@pytest.fixture
def small_idata():
    rng = np.random.default_rng(42)
    return az.from_dict(
        {
            "posterior": {
                "a": rng.normal(size=(2, 60)),
                "b": rng.normal(size=(2, 60)),
            }
        }
    )


def test_trace_page_has_dist_and_trace_columns(small_idata):
    """
    Given a posterior with two scalar variables,
    When one trace page is rendered,
    Then the figure has one distribution axis AND one trace axis per
    variable (the pymc6/arviz-1.0 migration had dropped the dist column).
    """
    # ACT
    fig = _render_trace_page(small_idata, ["a", "b"], n_rows=2, title="t")

    try:
        # ASSERT: 2 variables x (dist + trace) = 4 axes
        assert len(fig.axes) == 4
        # ASSERT: every axis has drawn content (per-chain lines)
        for ax in fig.axes:
            assert len(ax.lines) >= 2
    finally:
        plt.close(fig)


def test_save_multipage_trace_writes_pdf(small_idata, tmp_path):
    """
    Given a small posterior,
    When save_multipage_trace runs,
    Then a non-trivial PDF is written without error.
    """
    # ACT
    out = tmp_path / "trace.pdf"
    save_multipage_trace(small_idata, ["a", "b"], str(out))

    # ASSERT
    assert out.exists()
    assert out.stat().st_size > 5000


def _with_mode_var(idata, mode_vals):
    """Attach a posterior['mode'] (chain, draw) int array, mirroring what
    identify_modes.attach() writes onto a real trace."""
    import xarray as xr

    post = idata.posterior
    post["mode"] = xr.DataArray(
        np.asarray(mode_vals, dtype=int),
        dims=("chain", "draw"),
        coords={"chain": post.chain, "draw": post.draw},
    )
    return idata


def test_render_trace_page_shades_multimodal_by_mode(small_idata):
    """
    Given a posterior with a mode label that switches partway through every
      chain (mode-hopping),
    When one trace page is rendered,
    Then each trace axis gains a per-draw scatter overlay colored by mode
      (visualizing the hop), on top of the existing dist+trace line layout.
    """
    n_chain, n_draw = (
        small_idata.posterior.sizes["chain"],
        small_idata.posterior.sizes["draw"],
    )
    mode_vals = np.zeros((n_chain, n_draw), dtype=int)
    mode_vals[:, n_draw // 2 :] = 1
    idata = _with_mode_var(small_idata, mode_vals)

    # ACT
    fig = _render_trace_page(idata, ["a", "b"], n_rows=2, title="t")

    try:
        # ASSERT: trace axes (odd index) gain scatter collections, dist axes don't
        for i, ax in enumerate(fig.axes):
            if i % 2 == 1:
                assert len(ax.collections) == n_chain
            else:
                assert len(ax.collections) == 0
    finally:
        plt.close(fig)


def test_render_trace_page_unimodal_mode_var_unchanged(small_idata):
    """
    Given a posterior whose 'mode' variable is a single label (0) everywhere
      (unimodal run; identify_modes still attaches 'mode' by default),
    When one trace page is rendered,
    Then no scatter overlay is added -- single-mode output renders exactly
      as it did before mode-shading existed.
    """
    n_chain, n_draw = (
        small_idata.posterior.sizes["chain"],
        small_idata.posterior.sizes["draw"],
    )
    idata = _with_mode_var(small_idata, np.zeros((n_chain, n_draw), dtype=int))

    # ACT
    fig = _render_trace_page(idata, ["a", "b"], n_rows=2, title="t")

    try:
        # ASSERT
        assert all(len(ax.collections) == 0 for ax in fig.axes)
    finally:
        plt.close(fig)


def test_render_trace_page_no_mode_var_unchanged(small_idata):
    """
    Given a posterior with no 'mode' variable at all (old trace file, or
      mode identification failed/was skipped),
    When one trace page is rendered,
    Then no scatter overlay is added and rendering does not error.
    """
    # ACT
    fig = _render_trace_page(small_idata, ["a", "b"], n_rows=2, title="t")

    try:
        # ASSERT
        assert all(len(ax.collections) == 0 for ax in fig.axes)
    finally:
        plt.close(fig)


@pytest.fixture
def many_chain_idata():
    """70 chains x 4000 draws -- the shape a production PTDE run produces."""
    rng = np.random.default_rng(7)
    return az.from_dict(
        {
            "posterior": {
                "a": rng.normal(size=(70, 4000)),
                "b": rng.normal(size=(70, 4000)),
            },
            "sample_stats": {"lp": rng.normal(size=(70, 4000))},
        }
    )


def test_thinning_keeps_draws_per_chain_not_total(many_chain_idata, tmp_path):
    """
    Given a many-chain posterior far larger than the plotting budget,
    When save_multipage_trace thins it,
    Then each chain keeps draws_per_chain points rather than
      max_samples/n_chains (~28 for 70 chains), which starved every chain
      regardless of how long it actually sampled.
    """
    # ARRANGE
    captured = {}
    import exozippy.run as run_mod

    real = run_mod._render_trace_page

    def spy(idata, *a, **kw):
        captured.setdefault("draws", idata.posterior.draw.size)
        return real(idata, *a, **kw)

    run_mod._render_trace_page = spy
    try:
        # ACT
        save_multipage_trace(
            many_chain_idata,
            ["a", "b"],
            str(tmp_path / "t.pdf"),
            draws_per_chain=100,
        )
    finally:
        run_mod._render_trace_page = real

    # ASSERT
    assert captured["draws"] == 100


def test_thinning_preserves_true_draw_numbers(many_chain_idata, tmp_path):
    """
    Given a posterior thinned for plotting by save_multipage_trace,
    When the thinned data reaches the page renderer,
    Then its `draw` coordinate still holds true (unthinned) draw numbers, so
      the trace x axis spans the real run -- round-tripping through
      az.from_dict dropped the coordinate and relabelled the axis 0..n_thinned.
    """
    # ARRANGE
    captured = {}
    import exozippy.run as run_mod

    real = run_mod._render_trace_page

    def spy(idata, *a, **kw):
        captured.setdefault(
            "draw_coord", np.asarray(idata.posterior.draw.values)
        )
        return real(idata, *a, **kw)

    run_mod._render_trace_page = spy
    try:
        # ACT
        save_multipage_trace(
            many_chain_idata,
            ["a"],
            str(tmp_path / "t.pdf"),
            draws_per_chain=100,
        )
    finally:
        run_mod._render_trace_page = real

    # ASSERT: last thinned point carries its original draw number, not its
    # position along the thinned axis.
    coord = captured["draw_coord"]
    assert coord.max() > 3000
    assert coord[1] > 1


def test_dist_column_is_density_not_ecdf(many_chain_idata):
    """
    Given few draws per chain after thinning,
    When one trace page is rendered,
    Then the dist column shows a density, not an ECDF -- arviz's default
      kind="auto" switches to an ECDF below 100 draws per chain, drawing a
      cumulative curve that plateaus at 1.0 and reads as a clipped posterior.
    """
    # ARRANGE: 50 draws/chain, below arviz's auto-ECDF threshold
    thinned = many_chain_idata.isel(draw=slice(None, 50))

    # ACT
    fig = _render_trace_page(thinned, ["a"], n_rows=1, title="t")

    try:
        # ASSERT: a CDF is monotonically non-decreasing and tops out at 1.0
        ys = [
            np.asarray(l.get_ydata())
            for l in fig.axes[0].lines
            if len(l.get_ydata()) > 2
        ]
        assert ys, "dist column drew nothing"
        assert not all(np.all(np.diff(y) >= -1e-12) for y in ys)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Degenerate (density-less) variables
#
# arviz_stats' KDE builds a fixed 512-interval grid spanning exactly the
# [min, max] of the finite draws, so np.histogram raises
#   ValueError: Too many bins for data range. Cannot create 512 finite-sized
#   bins.
# whenever that range spans fewer than ~512 float64 steps.  That took down the
# whole PDF -- and, since save_multipage_trace runs inside _run_fit's wrap-up,
# every other output of a gracefully stopped fit with it.
# ---------------------------------------------------------------------------


def _idata_with(a_values, other=None, n_chain=2, n_draw=60):
    """Posterior with variable 'a' set to a_values and a healthy 'b'."""
    rng = np.random.default_rng(3)
    post = {
        "a": np.asarray(a_values, dtype=float),
        "b": rng.normal(size=(n_chain, n_draw)) if other is None else other,
    }
    return az.from_dict({"posterior": post})


def _constant_but_one_ulp(value=1.0, shape=(2, 60)):
    """Finite, not constant, but spanning a single float64 step.

    This is the shape a stopped, unmixed run leaves behind: a chain that never
    really moved, but whose last bits differ -- the exact condition that
    crashed CI (n_draws=137, max_rhat=6.42).
    """
    arr = np.full(shape, value)
    arr[0, 0] = np.nextafter(value, value + 1.0)
    return arr


@pytest.mark.parametrize(
    "label, values",
    [
        ("one float64 step", _constant_but_one_ulp()),
        ("exactly constant", np.full((2, 60), 3.5)),
        ("no finite draws", np.full((2, 60), np.nan)),
        ("all infinite", np.full((2, 60), np.inf)),
        (
            "hundred float64 steps",
            1.0
            + np.random.default_rng(1).integers(0, 100, size=(2, 60))
            * np.spacing(1.0),
        ),
    ],
)
def test_degenerate_variable_does_not_kill_the_pdf(label, values, tmp_path):
    """
    Given a posterior in which one variable admits no density (it is
      constant, non-finite, or spans fewer float64 steps than the KDE grid
      has points),
    When save_multipage_trace runs,
    Then a complete PDF is still written -- the whole of wrap-up used to die
      with numpy's "Too many bins for data range".
    """
    # ARRANGE
    idata = _idata_with(values)
    out = tmp_path / "trace.pdf"

    # ACT
    save_multipage_trace(idata, ["a", "b"], str(out))

    # ASSERT
    assert out.exists(), label
    assert out.stat().st_size > 5000, label


def test_degenerate_variable_is_reported_not_swallowed(caplog, tmp_path):
    """
    Given a variable with no density,
    When save_multipage_trace runs,
    Then it says so per variable, naming the variable and the reason -- a
      missing panel must never be silent.
    """
    # ARRANGE
    idata = _idata_with(np.full((2, 60), 3.5))

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.run"):
        save_multipage_trace(idata, ["a", "b"], str(tmp_path / "t.pdf"))

    # ASSERT
    messages = [r.getMessage() for r in caplog.records]
    assert any("a" in m and "constant at 3.5" in m for m in messages), messages
    assert not any("b" in m and "constant" in m for m in messages), messages


def test_degenerate_variable_keeps_its_trace_panel(tmp_path):
    """
    Given a flat variable,
    When its page is rendered,
    Then the trace panel still carries one line per chain (a flat chain is
      exactly what the reader needs to see) and the density panel carries the
      reason and value instead of a density.
    """
    # ARRANGE
    idata = _idata_with(np.full((3, 40), 3.5), other=None, n_chain=3)
    rng = np.random.default_rng(4)
    idata.posterior["b"] = (("chain", "draw"), rng.normal(size=(3, 40)))
    rows = run_mod._split_degenerate_vars(idata, ["a"])[1]

    # ACT
    fig = run_mod._render_degenerate_page(idata, rows, title="t")

    try:
        # ASSERT
        assert len(fig.axes) == 2
        ax_dist, ax_trace = fig.axes
        assert len(ax_trace.lines) == 3
        texts = " ".join(t.get_text() for t in ax_dist.texts)
        assert "no density" in texts
        assert "constant at 3.5" in texts
    finally:
        plt.close(fig)


def test_partially_pinned_vector_keeps_its_sampled_densities(tmp_path):
    """
    Given a vector variable with some elements pinned constant and some
      sampled (GP and robust-likelihood hyperparameters are full-length
      vectors with the non-opted-in files pinned via sigma: 0, and such a
      vector IS tracked as a Deterministic),
    When the pages are split,
    Then only the pinned elements lose their density; the sampled ones stay
      on an ArviZ page via a coords selection.
    """
    # ARRANGE
    rng = np.random.default_rng(5)
    vec = rng.normal(size=(2, 60, 4))
    vec[:, :, 1] = 0.5
    vec[:, :, 3] = -2.0
    idata = az.from_dict(
        {"posterior": {"gp_sigma": vec, "b": rng.normal(size=(2, 60))}}
    )

    # ACT
    specs, degenerate = run_mod._split_degenerate_vars(
        idata, ["gp_sigma", "b"]
    )

    # ASSERT
    gp_spec = [s for s in specs if s[0] == "gp_sigma"]
    assert len(gp_spec) == 1
    _name, coords, n_rows = gp_spec[0]
    assert n_rows == 2
    assert list(coords.values())[0] == [0, 2]
    assert sorted(r.label for r in degenerate) == [
        "gp_sigma[1]",
        "gp_sigma[3]",
    ]
    # b is healthy and must not be touched
    assert ("b", None, 1) in specs


def test_degenerate_lp_does_not_kill_the_pdf(tmp_path):
    """
    Given a stopped fit whose log-posterior never moved (its own page comes
      from sample_stats, not posterior),
    When save_multipage_trace runs,
    Then the PDF is still written.
    """
    # ARRANGE
    rng = np.random.default_rng(6)
    idata = az.from_dict(
        {
            "posterior": {"a": rng.normal(size=(2, 60))},
            "sample_stats": {"lp": np.full((2, 60), -123.5)},
        }
    )
    out = tmp_path / "t.pdf"

    # ACT
    save_multipage_trace(idata, ["a"], str(out))

    # ASSERT
    assert out.exists()
    assert out.stat().st_size > 5000


def test_healthy_posterior_takes_the_unchanged_path(small_idata):
    """
    Given a posterior with no degenerate element,
    When the split runs,
    Then it is the identity: every variable keeps coords=None, so the ArviZ
      call is exactly the one made before degeneracy handling existed.
    """
    # ACT
    specs, degenerate = run_mod._split_degenerate_vars(small_idata, ["a", "b"])

    # ASSERT
    assert specs == [("a", None, 1), ("b", None, 1)]
    assert degenerate == []


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), None),
        (np.full((2, 2), 7.0), "constant at 7"),
        (np.full((2, 2), np.nan), "no finite draws"),
        (np.array([[np.nan, np.inf], [-np.inf, np.nan]]), "no finite draws"),
        (_constant_but_one_ulp(), "float64 steps"),
        # non-finite values are filtered before the test, exactly as
        # arviz_stats filters them, so a lone NaN must not change the verdict
        (np.array([[1.0, 2.0], [3.0, np.nan]]), None),
    ],
)
def test_dist_degeneracy_verdicts(values, expected):
    """
    Given each degenerate shape,
    When _dist_degeneracy inspects it,
    Then it returns None only when a 512-interval KDE grid can actually be
      built, and otherwise a reason naming what is wrong.
    """
    # ACT
    reason = run_mod._dist_degeneracy(values)

    # ASSERT
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason
