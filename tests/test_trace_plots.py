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
