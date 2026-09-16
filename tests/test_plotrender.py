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


def _spec(spec_id="demo.panel", meta=None, extra_traces=(), **geometry):
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
        **geometry,
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
            "aspect_equal": True,
        },
        x_range=[0.5, 9.5],
        y_inverted=True,
        extra_traces=(sky, scatter_model),
    )

    written = render_spec_groups([[spec]], filename_prefix=str(tmp_path / "p"))

    assert (tmp_path / "p_everything.pdf").stat().st_size > 0
    assert len(written) == 1


def test_the_promoted_geometry_fields_actually_reach_the_axes(tmp_path):
    """
    Given a chart with log, inverted and explicit-range geometry set as
      FIELDS,
    When it is rendered,
    Then the axes carry each one.

    The vocabulary smoke test above asserts only that a PDF was written, so it
    cannot tell a honored axis from an ignored one -- which is exactly how
    these could have been promoted out of `meta` while the renderer kept
    reading the old dict. This asserts the axis state instead (review 4.11.3).
    """
    # Arrange
    import matplotlib.pyplot as plt

    from exozippy.plotrender import _apply_axes

    spec = _spec(
        x_range=[1.0, 9.0],
        y_range=[-2.0, 2.0],
        x_log=True,
        y_inverted=True,
    )
    fig, ax = plt.subplots()

    # Act
    _apply_axes(ax, spec)

    # Assert
    assert ax.get_xscale() == "log"
    assert ax.get_xlim() == (1.0, 9.0)
    # Inverted axes report their limits descending.
    assert ax.get_ylim() == (2.0, -2.0)
    plt.close(fig)


def test_geometry_left_in_meta_is_no_longer_honored(tmp_path):
    """
    Given geometry passed the OLD way, as `meta` keys,
    When the chart is rendered,
    Then it is ignored.

    Pins the migration rather than the feature. A component that still writes
    `meta["x_log"]` must not half-work -- silently linear in the PDF and log in
    the GUI, or the reverse -- so the old spelling is inert on BOTH sides and
    this records that it is deliberate.
    """
    # Arrange
    import matplotlib.pyplot as plt

    from exozippy.plotrender import _apply_axes

    spec = _spec(
        meta={"x_log": True, "y_inverted": True, "x_range": [2.0, 3.0]}
    )
    fig, ax = plt.subplots()

    # Act
    _apply_axes(ax, spec)

    # Assert
    assert ax.get_xscale() == "linear"
    assert ax.get_ylim()[0] < ax.get_ylim()[1]
    plt.close(fig)


def test_a_trace_alpha_overrides_the_role_default():
    """
    Given traces with and without an explicit alpha,
    When the renderer resolves opacity,
    Then the override wins and the role default applies otherwise.

    The role default is the value the GUI adapter mirrors (review 4.11.6):
    the two renderers had disagreed, this one drawing data at 0.6 and the GUI
    drawing it opaque, so the same fit looked different depending on where it
    was viewed.
    """
    # Arrange
    from exozippy.plotrender import _DATA_ALPHA, _trace_alpha

    plain = Trace(name="d", role="data", kind="scatter", x=[0.0], y=[0.0])
    loud = Trace(
        name="d", role="data", kind="scatter", x=[0.0], y=[0.0], alpha=0.25
    )

    # Act / Assert
    assert _trace_alpha(plain, _DATA_ALPHA) == _DATA_ALPHA
    assert _trace_alpha(loud, _DATA_ALPHA) == 0.25


def test_geometry_serializes_only_when_set():
    """
    Given two charts, one with geometry and one without,
    When each is serialized,
    Then only the set fields appear in the payload.

    Same idiom as `yerr` and `style`: the default state of an axis is
    "whatever the renderer would do anyway", so a payload full of nulls and
    falses would be noise on every chart the GUI fetches.
    """
    # Arrange
    bare = _spec()
    shaped = _spec(x_log=True, y_range=[0.0, 1.0])

    # Act
    bare_json, shaped_json = bare.to_json(), shaped.to_json()

    # Assert
    for key in (
        "x_log",
        "y_log",
        "x_inverted",
        "y_inverted",
        "x_range",
        "y_range",
    ):
        assert key not in bare_json
    assert shaped_json["x_log"] is True
    assert shaped_json["y_range"] == [0.0, 1.0]
    assert "x_range" not in shaped_json


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
