"""SED color rows: a measured color between two filters (review 8.9.10).

A .sed row was absolute, or a SAME-FILTER differential between stars
(-2.5 log10(F_pos/F_neg) in the row's one filter).  A COLOR row names a
second filter for its neg side (``neg_filter:``), so it is
-2.5 log10(F_pos(filter) / F_neg(neg_filter)) -- the V-I of a microlensing
source from the light-curve regression, or a source-minus-clump color
offset, each far more precise than the two magnitudes it is a difference
of.  Each side has its own magnitude system (``magsys:``, ``neg_magsys:``;
unstated = the filter's native one), and the fitted Vega color is the
reported one minus the pos side's m_AB - m_Vega plus the neg side's.

The spelling and the double-counting warning are in
src/exozippy/components/sed/sed.md, "Color rows".
"""

from types import SimpleNamespace

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import yaml
from scipy.stats import norm

from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT
from exozippy.components.sed.magsys import AB, VEGA, read_magsys_table

_MODEL_ROOT = DEFAULT_MODEL_ROOT


def _sed_path(tmp_path, rows, name="color.sed"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"model": "NextGen", "filters": rows}))
    return str(p)


def _loaded_sed(tmp_path, rows, star_names=("A",)):
    """A SED component around ``rows``, through load_data."""
    from exozippy.components.sed.sed import SED
    from exozippy.config import ConfigManager

    cm = ConfigManager(
        {
            "star.teffsed": {"initval": 5778.0},
            "star.feh": {"initval": 0.0},
            "star.av": {"initval": 0.0},
        }
    )
    sed = SED(
        {"file": _sed_path(tmp_path, rows), "model_root": str(_MODEL_ROOT)},
        cm,
    )
    star = SimpleNamespace(names=list(star_names), n_elements=len(star_names))
    sed.load_data(SimpleNamespace(star=star, active_components={}))
    return sed


def _offset(column):
    return float(
        read_magsys_table(_MODEL_ROOT, "NextGen").loc[column, "ab_minus_vega"]
    )


def _row(name, mag, err=0.02, **kw):
    return {"name": name, "mag": mag, "err": err, **kw}


# ---------------------------------------------------------------------------
# Section 1 -- the bookkeeping: two filters per row, one grid column each
# ---------------------------------------------------------------------------


def test_a_color_row_reads_a_second_grid_column_and_may_name_one_star_twice(
    tmp_path,
):
    """
    Given a two-star file with an absolute row, a same-filter differential
      row and a color row naming the SAME star on both sides,
    When the SED loads it,
    Then the color row's neg side reads a grid column appended after the
      rows' own filters, the other rows read their own column, and pos and
      neg may share a star on the color row (the color of one star).
    """
    # ARRANGE
    rows = [
        _row("2MASS/2MASS.J", 10.0, photType={"pos": ["S"]}),
        _row("2MASS/2MASS.H", 1.5, photType={"pos": ["S"], "neg": ["L"]}),
        _row(
            "2MASS/2MASS.J",
            0.6,
            neg_filter="2MASS/2MASS.Ks",
            photType={"pos": ["S"], "neg": ["S"]},
        ),
    ]

    # ACT
    sed = _loaded_sed(tmp_path, rows, star_names=("L", "S"))

    # ASSERT
    assert sed.row_filters == [
        "2MASS/2MASS.J",
        "2MASS/2MASS.H",
        "2MASS/2MASS.J",
        "2MASS/2MASS.Ks",
    ]
    assert sed.neg_col.tolist() == [0, 1, 3]
    assert sed.color_rows.tolist() == [False, False, True]
    assert sed.mist_filters[:4] == [
        "2MASS_J",
        "2MASS_H",
        "2MASS_J",
        "2MASS_Ks",
    ]
    assert sed.pos_matrix.tolist() == [[0, 1], [0, 1], [0, 1]]
    assert sed.neg_matrix.tolist() == [[0, 0], [1, 0], [0, 1]]
    assert sed.combo_labels[2] == "S [2MASS/2MASS.J] - S [2MASS/2MASS.Ks]"
    # The second filter is an ordinary grid column for the predict_* API.
    assert sed.has_filter("2MASS/2MASS.Ks")


def test_a_color_row_predicts_the_flux_ratio_across_two_filters(tmp_path):
    """
    Given per-star predicted magnitudes in every grid column, and rows that
      exercise each kind -- a blended absolute row, a same-filter
      differential, a one-star color and a blended color (A+B in J against
      B in Ks),
    When _combined_appmag_node is evaluated,
    Then every row equals -2.5 log10(F_pos / F_neg) computed here in numpy
      from the same per-star magnitudes, each side in its own filter.
    """
    # ARRANGE
    rows = [
        _row("2MASS/2MASS.J", 9.0, photType={"pos": ["A", "B"]}),
        _row("2MASS/2MASS.H", 1.0, photType={"pos": ["A"], "neg": ["B"]}),
        _row(
            "2MASS/2MASS.J",
            0.5,
            neg_filter="2MASS/2MASS.Ks",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
        _row(
            "2MASS/2MASS.J",
            0.1,
            neg_filter="2MASS/2MASS.Ks",
            photType={"pos": ["A", "B"], "neg": ["B"]},
        ),
    ]
    sed = _loaded_sed(tmp_path, rows, star_names=("A", "B"))
    # columns: J, H, J, J (rows), Ks, Ks (the two color rows' neg sides)
    assert sed.mist_filters[:6] == [
        "2MASS_J",
        "2MASS_H",
        "2MASS_J",
        "2MASS_J",
        "2MASS_Ks",
        "2MASS_Ks",
    ]
    n_cols = len(sed.mist_filters)
    rng = np.random.default_rng(8910)
    m_star = 8.0 + 4.0 * rng.random((2, n_cols))
    sed._m_pred_matrix = pt.as_tensor_variable(m_star)

    # ACT
    combined = sed._combined_appmag_node(system=None).eval()

    # ASSERT
    F = 10 ** (-0.4 * m_star)
    # each row's pos side reads its own column (0..3), a color row's neg
    # side its appended one (4, 5)
    assert sed.neg_col.tolist() == [0, 1, 4, 5]
    expected = [
        -2.5 * np.log10(F[0, 0] + F[1, 0]),
        -2.5 * np.log10(F[0, 1] / F[1, 1]),
        -2.5 * np.log10(F[0, 2] / F[0, 4]),
        -2.5 * np.log10((F[0, 3] + F[1, 3]) / F[1, 5]),
    ]
    np.testing.assert_allclose(combined, expected, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Section 2 -- the magnitude system of each side
# ---------------------------------------------------------------------------


def test_each_side_of_a_color_moves_onto_vega_by_its_own_offset(tmp_path):
    """
    Given color rows mixing AB and Vega sides -- AB minus Vega, Vega minus
      AB, AB minus AB, and an unstated SDSS side (AB-native) --
    When the SED loads them,
    Then the fitted color is the reported one minus the pos side's
      m_AB - m_Vega (when AB) plus the neg side's (when AB), and each
      side's system is recorded.
    """
    # ARRANGE
    color = 0.75
    rows = [
        _row(
            "SLOAN/SDSS.g",
            color,
            magsys="AB",
            neg_filter="2MASS/2MASS.J",
            neg_magsys="Vega",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
        _row(
            "2MASS/2MASS.J",
            color,
            magsys="Vega",
            neg_filter="SLOAN/SDSS.g",
            neg_magsys="AB",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
        _row(
            "SLOAN/SDSS.g",
            color,
            magsys="AB",
            neg_filter="SLOAN/SDSS.r",
            neg_magsys="AB",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
        _row(
            "SLOAN/SDSS.g",
            color,
            neg_filter="SLOAN/SDSS.r",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
    ]
    g, r = _offset("SDSS_g"), _offset("SDSS_r")

    # ACT
    sed = _loaded_sed(tmp_path, rows)

    # ASSERT
    assert sed.magsys == [AB, VEGA, AB, AB]
    assert sed.neg_magsys == [VEGA, AB, AB, AB]
    np.testing.assert_array_equal(
        sed.mag,
        [color - g, color + g, color - (g - r), color - (g - r)],
    )
    np.testing.assert_array_equal(sed.mag_reported, [color] * 4)
    np.testing.assert_array_equal(sed.err, [0.02] * 4)


def test_a_mixed_system_color_is_the_difference_of_the_converted_magnitudes(
    tmp_path,
):
    """
    Given an AB g magnitude, a Vega J magnitude, and a g-J color reported
      as the difference of those two numbers,
    When the SED loads them,
    Then the fitted color equals the fitted g minus the fitted J -- the
      color and the two magnitudes are converted consistently.
    """
    # ARRANGE
    g_ab, j_vega = 12.3, 10.1
    rows = [
        _row("SLOAN/SDSS.g", g_ab, magsys="AB"),
        _row("2MASS/2MASS.J", j_vega, magsys="Vega"),
        _row(
            "SLOAN/SDSS.g",
            g_ab - j_vega,
            magsys="AB",
            neg_filter="2MASS/2MASS.J",
            neg_magsys="Vega",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
    ]

    # ACT
    sed = _loaded_sed(tmp_path, rows)

    # ASSERT
    assert sed.mag[2] == pytest.approx(sed.mag[0] - sed.mag[1], abs=1e-12)


def test_an_all_vega_color_row_is_fitted_as_reported(tmp_path):
    """
    Given a color between two Vega-native filters, systems unstated,
    When the SED loads it,
    Then nothing is shifted and no AB prose is due.
    """
    # ARRANGE
    rows = [
        _row(
            "2MASS/2MASS.J",
            0.512345,
            neg_filter="2MASS/2MASS.Ks",
            photType={"pos": ["A"], "neg": ["A"]},
        )
    ]

    # ACT
    sed = _loaded_sed(tmp_path, rows)

    # ASSERT
    assert sed.magsys == [VEGA] and sed.neg_magsys == [VEGA]
    assert sed.mag[0] == 0.512345
    assert not np.any(sed.ab_minus_vega)


# ---------------------------------------------------------------------------
# Section 3 -- spellings that raise, naming the file and the row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row, match",
    [
        # A misspelled key would otherwise silently drop the second filter.
        (
            _row(
                "2MASS/2MASS.J",
                0.5,
                neg_fliter="2MASS/2MASS.Ks",
                photType={"pos": ["A"], "neg": ["A"]},
            ),
            r"filter row 0 \('2MASS/2MASS.J'\): unknown key\(s\) "
            r"\['neg_fliter'\]",
        ),
        (
            _row("2MASS/2MASS.J", 0.5, neg_magsys="Vega"),
            r"filter row 0 .*neg_magsys without neg_filter",
        ),
        (
            _row(
                "2MASS/2MASS.J",
                0.5,
                neg_filter="2MASS/2MASS.Ks",
                photType={"pos": ["A"]},
            ),
            r"filter row 0 .*needs a non-empty photType 'neg'",
        ),
        (
            _row("2MASS/2MASS.J", 0.5, neg_filter="2MASS/2MASS.Ks"),
            r"filter row 0 .*needs a non-empty photType 'neg'",
        ),
        (
            _row(
                "2MASS/2MASS.J",
                0.5,
                neg_filter=["2MASS/2MASS.Ks"],
                photType={"pos": ["A"], "neg": ["A"]},
            ),
            r"filter row 0 .*neg_filter must be a filter name",
        ),
        # The same filter under another alias: a same-filter differential,
        # whose one spelling has no neg_filter.
        (
            _row(
                "2MASS/2MASS.J",
                0.5,
                neg_filter="2MASS.J",
                photType={"pos": ["A"], "neg": ["B"]},
            ),
            r"filter row 0 .*is the row's own filter \(BC column 2MASS_J\)",
        ),
        # Case-sensitive, like every magnitude system (magsys.parse_magsys).
        (
            _row(
                "2MASS/2MASS.J",
                0.5,
                neg_filter="2MASS/2MASS.Ks",
                neg_magsys="vega",
                photType={"pos": ["A"], "neg": ["A"]},
            ),
            r"filter row 0 .*neg_filter '2MASS/2MASS.Ks'",
        ),
    ],
)
def test_a_malformed_color_row_raises_naming_the_file_and_row(
    tmp_path, row, match
):
    """
    Given a color row spelled wrongly (a misspelled key, neg_magsys with no
      second filter, no neg stars, a non-string filter, the row's own
      filter under another alias, a case-variant system),
    When the SED loads it,
    Then a ValueError names the .sed file, the row and the problem.
    """
    # ARRANGE / ACT / ASSERT
    with pytest.raises(ValueError, match=match) as exc:
        _loaded_sed(tmp_path, [row], star_names=("A", "B"))
    assert "color.sed" in str(exc.value)


def test_a_same_filter_row_still_refuses_one_star_on_both_sides(tmp_path):
    """
    Given a same-filter differential row naming one star on both sides
      (identically zero),
    When the SED loads it,
    Then it raises as before -- only a color row may share a star.
    """
    rows = [_row("2MASS/2MASS.J", 0.0, photType={"pos": ["A"], "neg": ["A"]})]
    with pytest.raises(ValueError, match="both pos and neg"):
        _loaded_sed(tmp_path, rows, star_names=("A", "B"))


# ---------------------------------------------------------------------------
# Section 4 -- end to end: the likelihood a color row adds
# ---------------------------------------------------------------------------


def test_a_color_row_beside_both_magnitudes_adds_its_own_chi2(tmp_path):
    """
    Given a one-star System whose .sed has V, I and a V-I color row,
    When the model is built and the SED likelihood evaluated at the start,
    Then it equals the sum of three Gaussian log densities computed here
      from the per-star predicted magnitudes: V and I against their
      columns, the color against m_V - m_I (one star: -2.5 log10(F_V/F_I)),
      each with sigma = err * errscale.

    (This is the configuration sed.md warns about for CATALOG magnitudes
    with independent errors -- the color then counts the same information
    twice.  The arithmetic is what is pinned here, not the advice.)
    """
    from exozippy.system import System

    # ARRANGE
    rows = [
        _row("Generic/Bessell.V", 15.20, err=0.05, magsys="Vega"),
        _row("Generic/Cousins.I", 14.30, err=0.06, magsys="Vega"),
        _row(
            "Generic/Bessell.V",
            0.87,
            err=0.01,
            magsys="Vega",
            neg_filter="Generic/Cousins.I",
            neg_magsys="Vega",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
    ]
    system = System(
        {
            "star": [{"name": "A", "mist": False}],
            "sed": {
                "file": _sed_path(tmp_path, rows),
                "model_root": str(_MODEL_ROOT),
            },
        },
        {},
    )
    system.prepare()
    model = system.build_model()
    sed = system.sed
    start = system.get_raw_start(model)
    values = [start[v.name] for v in model.value_vars]

    # ACT
    lp_sed = float(
        model.compile_logp(vars=[model["sed.model"]], sum=True)(start)
    )

    # ASSERT
    m_all, errscale = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values(
            [sed._predicted_appmag_node(system), sed.errscale.value]
        ),
        on_unused_input="ignore",
    )(*values)
    v_col, i_col = 0, int(sed.neg_col[2])
    assert sed.mist_filters[i_col] == sed.mist_filters[1]
    pred = np.array(
        [m_all[0, v_col], m_all[0, 1], m_all[0, v_col] - m_all[0, i_col]]
    )
    sigma = np.array([0.05, 0.06, 0.01]) * float(np.squeeze(errscale))
    expected = norm.logpdf([15.20, 14.30, 0.87], loc=pred, scale=sigma).sum()
    assert lp_sed == pytest.approx(expected, rel=1e-10, abs=1e-9)
    # The star is read by the color row like any other row.
    assert bool(sed.seen_star_mask(system)[0])


# ---------------------------------------------------------------------------
# Section 5 -- the data-only chart leaves a color out
# ---------------------------------------------------------------------------


def test_the_data_only_preview_leaves_color_rows_out(tmp_path):
    """
    Given a .sed with two magnitudes and a color,
    When the data-only GUI preview (no model point) is built,
    Then it plots the two magnitudes only: a color has no place on a
      magnitude-vs-wavelength axis.
    """
    # ARRANGE
    rows = [
        _row("2MASS/2MASS.J", 10.0),
        _row("2MASS/2MASS.Ks", 9.5),
        _row(
            "2MASS/2MASS.J",
            0.5,
            neg_filter="2MASS/2MASS.Ks",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
    ]
    sed = _loaded_sed(tmp_path, rows)

    # ACT
    (chart,) = sed.plot_data(system=None, point=None)

    # ASSERT
    (trace,) = chart.traces
    np.testing.assert_array_equal(trace.y, [10.0, 9.5])
    assert len(trace.x) == 2
