"""
Tests for the detailed trace-plot output (run.py).

The ArviZ 1.0 migration silently changed az.plot_trace to render only the
trace lines; the old dist + trace two-column layout is now plot_trace_dist.
These tests pin the restored layout.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
import pytest

from exozippy.run import save_multipage_trace, _render_trace_page


@pytest.fixture
def small_idata():
    rng = np.random.default_rng(42)
    return az.from_dict({
        "posterior": {
            "a": rng.normal(size=(2, 60)),
            "b": rng.normal(size=(2, 60)),
        }
    })


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
    post["mode"] = xr.DataArray(np.asarray(mode_vals, dtype=int),
                                dims=("chain", "draw"),
                                coords={"chain": post.chain, "draw": post.draw})
    return idata


def test_render_trace_page_shades_multimodal_by_mode(small_idata):
    """
    Given a posterior with a mode label that switches partway through every
      chain (mode-hopping),
    When one trace page is rendered,
    Then each trace axis gains a per-draw scatter overlay colored by mode
      (visualizing the hop), on top of the existing dist+trace line layout.
    """
    n_chain, n_draw = small_idata.posterior.sizes["chain"], small_idata.posterior.sizes["draw"]
    mode_vals = np.zeros((n_chain, n_draw), dtype=int)
    mode_vals[:, n_draw // 2:] = 1
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
    n_chain, n_draw = small_idata.posterior.sizes["chain"], small_idata.posterior.sizes["draw"]
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
    return az.from_dict({
        "posterior": {
            "a": rng.normal(size=(70, 4000)),
            "b": rng.normal(size=(70, 4000)),
        },
        "sample_stats": {"lp": rng.normal(size=(70, 4000))},
    })


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
        save_multipage_trace(many_chain_idata, ["a", "b"],
                             str(tmp_path / "t.pdf"), draws_per_chain=100)
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
        captured.setdefault("draw_coord",
                            np.asarray(idata.posterior.draw.values))
        return real(idata, *a, **kw)

    run_mod._render_trace_page = spy
    try:
        # ACT
        save_multipage_trace(many_chain_idata, ["a"], str(tmp_path / "t.pdf"),
                             draws_per_chain=100)
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
        ys = [np.asarray(l.get_ydata()) for l in fig.axes[0].lines
              if len(l.get_ydata()) > 2]
        assert ys, "dist column drew nothing"
        assert not all(np.all(np.diff(y) >= -1e-12) for y in ys)
    finally:
        plt.close(fig)
