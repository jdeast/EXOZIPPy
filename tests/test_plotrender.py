"""Tests for the generic Chart -> matplotlib renderer (plotrender.py).

The renderer is the saved-PDF half of the two-renderer pair (the GUI's
plotly-adapter.ts is the other); these tests pin its file naming, the
points-list spaghetti semantics, and the meta/style vocabulary with cheap
synthetic specs -- no System or pytensor compile.

Tests follow AAA with Given/When/Then docstrings.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from exozippy.chart import Chart, Trace
from exozippy.plotrender import plot_via_specs, render_spec_groups


def _spec(spec_id="demo.panel", meta=None, extra_traces=()):
    x = np.linspace(0.0, 10.0, 20)
    traces = [
        Trace(
            name="inst0",
            role="data",
            kind="scatter",
            x=x,
            y=np.sin(x),
            yerr=np.full_like(x, 0.1),
            style={"series_index": 0},
        ),
        Trace(name="model", role="model", kind="line", x=x, y=np.sin(x)),
        *extra_traces,
    ]
    return Chart(
        id=spec_id,
        component={"yaml_key": "demo", "instance": None},
        title="Demo",
        xlabel="x",
        ylabel="y",
        traces=traces,
        meta=dict(meta or {}),
    )


def test_render_writes_pdf_named_by_file_tag(tmp_path):
    """Given a spec with meta file_tag, When rendered, Then the PDF is
    {prefix}_{file_tag}.pdf and is nonempty."""
    spec = _spec(meta={"file_tag": "RV_unphased"})

    written = render_spec_groups([[spec]], filename_prefix=str(tmp_path / "p"))

    assert written == [str(tmp_path / "p_RV_unphased.pdf")]
    assert (tmp_path / "p_RV_unphased.pdf").stat().st_size > 0


def test_render_falls_back_to_spec_id_for_filename(tmp_path):
    """Given a spec without file_tag, When rendered, Then the id (dots ->
    underscores) names the file."""
    spec = _spec(spec_id="rvinstrument.phased.b")

    written = render_spec_groups([[spec]], filename_prefix=str(tmp_path / "p"))

    assert written == [str(tmp_path / "p_rvinstrument_phased_b.pdf")]


def test_render_full_meta_vocabulary(tmp_path):
    """Given every meta decoration and an asymmetric-error sky trace, When
    rendered, Then it draws without error (vocabulary smoke test)."""
    sky = Trace(
        name="sky",
        role="data",
        kind="scatter",
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        yerr=np.array([[0.1, 0.2], [0.3, 0.4]]),
        xerr=np.array([0.05, 0.06]),
        style={"color": "k", "marker": "."},
    )
    scatter_model = Trace(
        name="epochs",
        role="model",
        kind="scatter",
        x=np.array([1.0, 2.0]),
        y=np.array([3.1, 3.9]),
    )
    spec = _spec(
        meta={
            "file_tag": "everything",
            "figsize": (8, 4),
            "hline_y": 0.0,
            "x_range": [0.5, 9.5],
            "y_inverted": True,
            "aspect_equal": True,
        },
        extra_traces=(sky, scatter_model),
    )

    written = render_spec_groups([[spec]], filename_prefix=str(tmp_path / "p"))

    assert (tmp_path / "p_everything.pdf").stat().st_size > 0
    assert len(written) == 1


class _FakeComponent:
    """plot_data stub: fails on request for a given draw index."""

    prefix = "demo"

    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = set(fail_on)

    def plot_data(self, system, point):
        idx = self.calls
        self.calls += 1
        if idx in self.fail_on:
            raise RuntimeError("bad draw")
        return [_spec(meta={"file_tag": "demo"})]


def test_plot_via_specs_spaghetti_skips_bad_draws(tmp_path):
    """Given three draws whose middle plot_data raises, When plot_via_specs
    runs, Then the bad draw is skipped and the PDF still renders."""
    comp = _FakeComponent(fail_on={1})

    written = plot_via_specs(
        comp, None, [{}, {}, {}], filename_prefix=str(tmp_path / "p")
    )

    assert comp.calls == 3
    assert written == [str(tmp_path / "p_demo.pdf")]


def test_plot_via_specs_reference_point_failure_raises(tmp_path):
    """Given the FIRST (reference) draw fails, When plot_via_specs runs,
    Then it raises -- the reference supplies the data traces themselves."""
    comp = _FakeComponent(fail_on={0})

    with pytest.raises(RuntimeError, match="bad draw"):
        plot_via_specs(
            comp, None, [{}, {}], filename_prefix=str(tmp_path / "p")
        )


def test_plot_via_specs_single_dict_point(tmp_path):
    """Given a bare point dict (pre-flight style), When plot_via_specs runs,
    Then it is treated as a one-point list."""
    comp = _FakeComponent()

    written = plot_via_specs(
        comp, None, {}, filename_prefix=str(tmp_path / "p")
    )

    assert comp.calls == 1
    assert len(written) == 1
