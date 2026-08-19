"""
Corner plots must survive flat / non-finite parameters.

corner.corner() builds every panel from np.histogram, so it raises outright
on a column with no dynamic range and on one holding a non-finite value --
and the caller then loses the WHOLE corner plot for one flat parameter.  Both
shapes come out of real fits: an element pinned with ``sigma: 0`` inside an
otherwise sampled vector is exactly constant across every draw (four of
examples/gj1214's band.thermal elements are), and a short or stopped run
leaves variables that never moved.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import pytest

from exozippy.corner_utils import _drop_undrawable, save_corner_plot


@pytest.fixture
def healthy():
    rng = np.random.default_rng(11)
    return rng.normal(size=(200, 3)), ["x", "y", "z"]


def test_healthy_samples_pass_through_untouched(healthy):
    """
    Given samples with no flat or non-finite column,
    When the drop pass runs,
    Then nothing is removed -- a normal fit's corner plot is byte-identical
      to what it was before degeneracy handling existed.
    """
    # ARRANGE
    samples, labels = healthy

    # ACT
    out, out_labels = _drop_undrawable(samples, labels, "f.png")

    # ASSERT
    assert out.shape == samples.shape
    assert np.array_equal(out, samples)
    assert out_labels == labels


def test_constant_column_is_dropped_and_named(healthy, caplog):
    """
    Given one exactly constant column (a sigma: 0 pinned vector element),
    When the drop pass runs,
    Then only that column goes, and the warning names the parameter -- the
      other parameters are still plotted.
    """
    # ARRANGE
    samples, labels = healthy
    samples[:, 1] = 4.25

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.corner_utils"):
        out, out_labels = _drop_undrawable(samples, labels, "f.png")

    # ASSERT
    assert out_labels == ["x", "z"]
    assert out.shape == (200, 2)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "y" in msgs and "constant at 4.25" in msgs


def test_all_nan_column_is_dropped_without_deleting_every_draw(
    healthy, caplog
):
    """
    Given a column with no finite value at all,
    When the drop pass runs,
    Then the column is removed but every draw survives -- dropping
      non-finite ROWS first would have deleted the entire sample.
    """
    # ARRANGE
    samples, labels = healthy
    samples[:, 0] = np.nan

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.corner_utils"):
        out, out_labels = _drop_undrawable(samples, labels, "f.png")

    # ASSERT
    assert out_labels == ["y", "z"]
    assert out.shape == (200, 2)
    assert "no finite draws" in " ".join(
        r.getMessage() for r in caplog.records
    )


def test_partially_non_finite_column_drops_only_those_draws(healthy, caplog):
    """
    Given a column with a handful of non-finite draws,
    When the drop pass runs,
    Then those draws are dropped and the column is kept -- corner's default
      range is the raw min/max, so a single NaN raised "Axis limits cannot be
      NaN or Inf" and killed the whole plot.
    """
    # ARRANGE
    samples, labels = healthy
    samples[3, 1] = np.nan
    samples[7, 2] = np.inf

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.corner_utils"):
        out, out_labels = _drop_undrawable(samples, labels, "f.png")

    # ASSERT
    assert out_labels == labels
    assert out.shape == (198, 3)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(
            lambda s: s.__setitem__((slice(None), 1), 0.0), id="flat"
        ),
        pytest.param(
            lambda s: s.__setitem__((slice(None), 1), np.nan), id="all-nan"
        ),
        pytest.param(lambda s: s.__setitem__((3, 1), np.nan), id="one-nan"),
    ],
)
def test_save_corner_plot_still_writes_the_file(spoil, tmp_path):
    """
    Given samples spoiled by a flat or non-finite column,
    When save_corner_plot runs,
    Then the PNG is written anyway -- corner used to raise and the caller's
      except-branch turned the whole plot into a log line.
    """
    # ARRANGE
    rng = np.random.default_rng(12)
    samples = rng.normal(size=(120, 3))
    spoil(samples)
    out = tmp_path / "corner.png"

    # ACT
    save_corner_plot(samples, ["x", "y", "z"], str(out))

    # ASSERT
    assert out.exists()
    assert out.stat().st_size > 5000


def test_everything_flat_skips_with_a_warning(tmp_path, caplog):
    """
    Given samples in which nothing can be drawn at all,
    When save_corner_plot runs,
    Then it skips and says so rather than raising or writing a broken file.
    """
    # ARRANGE
    samples = np.zeros((50, 2))
    out = tmp_path / "corner.png"

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.corner_utils"):
        save_corner_plot(samples, ["x", "y"], str(out))

    # ASSERT
    assert not out.exists()
    assert "nothing left to plot" in " ".join(
        r.getMessage() for r in caplog.records
    )


def test_column_with_sub_bin_range_is_dropped_and_named(healthy, caplog):
    """
    Given a column that is not exactly constant but spans only a few float64
      steps (a short or stopped run whose chain moved in the last bits only),
    When the drop pass runs,
    Then it is dropped and named just like an exactly constant one -- corner
      builds its panels from a linspace bin grid over the column's raw
      min/max, and over such a range that grid collapses onto repeated
      edges, so the panel is either a raise or a full-width spike that reads
      as a measured posterior.
    """
    # ARRANGE
    samples, labels = healthy
    base = 1.0
    rng = np.random.default_rng(5)
    samples[:, 1] = base + rng.integers(0, 6, size=200) * np.spacing(base)
    assert samples[:, 1].min() != samples[:, 1].max()

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.corner_utils"):
        out, out_labels = _drop_undrawable(samples, labels, "f.png")

    # ASSERT
    assert out_labels == ["x", "z"]
    assert out.shape == (200, 2)
    assert "y" in " ".join(r.getMessage() for r in caplog.records)


def test_a_genuinely_narrow_but_drawable_column_is_kept(healthy):
    """
    Given a column whose spread is tiny in absolute terms but still spans far
      more than the bin grid's float64 steps,
    When the drop pass runs,
    Then it is kept -- the test is on representability, not on smallness, so
      a well-measured parameter with a 1e-12 posterior width still plots.
    """
    # ARRANGE
    samples, labels = healthy
    samples[:, 1] = 1.0 + samples[:, 1] * 1e-12

    # ACT
    out, out_labels = _drop_undrawable(samples, labels, "f.png")

    # ASSERT
    assert out_labels == labels
    assert out.shape == (200, 3)
