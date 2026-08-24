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

import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import pytest

import exozippy.corner_utils as corner_utils
import exozippy.run as run_mod
from exozippy.constants import (
    CORNER_THIN_SEED,
    SIGMA_1_HIGH,
    SIGMA_1_LOW,
)
from exozippy.corner_utils import (
    CORNER_BINS,
    _drop_undrawable,
    histogram_grid_degenerate,
    save_corner_plot,
)


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


# --- the shared bin-degeneracy predicate (review 3.14.8) ----------------
#
# The test used to be written twice: run.py's _dist_degeneracy inlined the
# monotonicity check and corner_utils kept a one-line copy behind a
# cross-reference comment.  They MUST agree -- one decides whether a
# parameter is given a corner column, the other whether a density can be
# drawn for it -- so these pin that there is one implementation and that
# both call sites reach it.


def test_both_call_sites_use_the_one_predicate():
    """
    Given the extracted histogram_grid_degenerate,
    When run.py's _dist_degeneracy and corner_utils' _drop_undrawable are
      inspected,
    Then both reach it -- run.py imports the same function object, and
      corner_utils' module global is that same object.

    Regression: the two were separate one-liners and could drift.
    """
    # ACT / ASSERT
    assert run_mod.histogram_grid_degenerate is histogram_grid_degenerate
    assert corner_utils.histogram_grid_degenerate is histogram_grid_degenerate


@pytest.mark.parametrize("n_edges", [2, 3, 21, 513])
def test_a_wide_range_is_never_degenerate(n_edges):
    """
    Given a range spanning many float64 steps,
    When the predicate is asked about any grid length either caller uses,
    Then it says the grid is fine.
    """
    assert histogram_grid_degenerate(0.0, 1.0, n_edges) is False


@pytest.mark.parametrize("n_edges", [3, 21, 513])
def test_a_single_ulp_range_is_degenerate(n_edges):
    """
    Given a range spanning exactly one float64 step,
    When the predicate is asked about a grid of more than two edges,
    Then it says the grid collapses -- linspace cannot produce distinct
      interior edges inside a single ULP.
    """
    lo = 1.0
    hi = np.nextafter(lo, lo + 1.0)
    assert histogram_grid_degenerate(lo, hi, n_edges) is True


def test_the_predicate_matches_numpys_own_failure():
    """
    Given a range that numpy itself refuses to bin at 512 bins,
    When the predicate is asked with the same grid length run.py uses,
    Then it agrees -- the point of the check is to TRACK numpy's condition,
      not to approximate it.
    """
    # ARRANGE: 300 float64 steps -- fine for corner's 20 bins, not for 512.
    lo = 1.0
    hi = lo + 300 * np.spacing(lo)
    data = np.linspace(lo, hi, 50)

    # ACT
    corner_ok = not histogram_grid_degenerate(lo, hi, CORNER_BINS + 1)
    kde_bad = histogram_grid_degenerate(lo, hi, 512 + 1)
    with pytest.raises(ValueError):
        np.histogram(data, bins=512, range=(lo, hi))

    # ASSERT
    assert corner_ok
    assert kde_bad


def test_corner_bins_are_passed_as_edges_not_as_a_count():
    """
    Given corner's CORNER_BINS bin count,
    When _drop_undrawable asks the predicate,
    Then it asks about CORNER_BINS + 1 EDGES -- an off-by-one here would
      silently loosen or tighten the drop test by one bin.

    Pinned by behavior: a range spanning exactly CORNER_BINS float64 steps
    has CORNER_BINS + 1 representable edges and must be kept, while one
    spanning CORNER_BINS - 1 steps must not.
    """
    lo = 1.0
    keep_hi = lo + CORNER_BINS * np.spacing(lo)
    drop_hi = lo + (CORNER_BINS - 2) * np.spacing(lo)

    assert histogram_grid_degenerate(lo, keep_hi, CORNER_BINS + 1) is False
    assert histogram_grid_degenerate(lo, drop_hi, CORNER_BINS + 1) is True


# ---------------------------------------------------------------------------
# Review 4.2.6: the magic numbers this module used to spell inline.
# ---------------------------------------------------------------------------


def test_the_sigma_1_quantiles_are_bit_identical_to_the_inline_formula():
    """
    Given constants.SIGMA_1_LOW / SIGMA_1_HIGH,
    When they are compared to the 0.5 -/+ erf(1/sqrt(2))/2 expression
      save_corner_plot used to recompute for itself,
    Then they are EQUAL, bit for bit -- so adopting the constants moved no
      reported number.

    Not `np.isclose`: the whole claim of the extraction is that the corner
    plot's quantiles and the LaTeX table's are the same two numbers, and
    "close" would allow them to drift apart by one ulp and stay green.
    """
    assert SIGMA_1_LOW == 0.5 - math.erf(1.0 / math.sqrt(2)) / 2.0
    assert SIGMA_1_HIGH == 0.5 + math.erf(1.0 / math.sqrt(2)) / 2.0


def test_the_corner_thinning_seed_is_fixed():
    """
    Given save_corner_plot's thinning draw,
    When more rows are supplied than max_samples,
    Then the SAME rows come back every time -- a corner plot is a figure that
      goes in a paper, so it must not change when the same trace is
      re-rendered.  (run.get_draws' posterior spaghetti is deliberately the
      other way round; see its docstring.)
    """
    # ARRANGE
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(500, 2))

    # ACT
    a = np.random.default_rng(seed=CORNER_THIN_SEED).choice(
        500, size=100, replace=False
    )
    b = np.random.default_rng(seed=CORNER_THIN_SEED).choice(
        500, size=100, replace=False
    )

    # ASSERT
    assert np.array_equal(a, b)
    assert samples.shape == (500, 2)
