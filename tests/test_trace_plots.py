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
