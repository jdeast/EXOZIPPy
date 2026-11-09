"""SED differential color rows: four fluxes in two filters (review 8.9.11).

A row with ``color_filter: b`` is the difference of two same-filter
differentials, [m_pos(a) - m_neg(a)] - [m_pos(b) - m_neg(b)] -- the
Delta(V-I) of a microlensing source from the red clump, which no color row
(-2.5 log10 F_pos(a)/F_neg(b), one star set per side) can spell.  Every
magnitude system cancels.  sed.md, "Differential color rows".
"""

from types import SimpleNamespace

import numpy as np
import pytensor.tensor as pt
import pytest
import yaml

from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT


def _loaded_sed(tmp_path, rows, star_names=("A",)):
    """A SED component around ``rows``, through load_data (the
    test_sed_color_rows helper)."""
    from exozippy.components.sed.sed import SED
    from exozippy.config import ConfigManager

    path = tmp_path / "dcolor.sed"
    path.write_text(yaml.safe_dump({"model": "NextGen", "filters": rows}))
    cm = ConfigManager(
        {
            "star.teffsed": {"initval": 5778.0},
            "star.feh": {"initval": 0.0},
            "star.av": {"initval": 0.0},
        }
    )
    sed = SED({"file": str(path), "model_root": str(DEFAULT_MODEL_ROOT)}, cm)
    star = SimpleNamespace(names=list(star_names), n_elements=len(star_names))
    sed.load_data(SimpleNamespace(star=star, active_components={}))
    return sed


def _row(name, mag, err=0.02, **kw):
    return {"name": name, "mag": mag, "err": err, **kw}


def test_a_differential_color_row_is_the_four_flux_expression(tmp_path):
    """
    Given two stars and rows of every kind, including two differential
      color rows (one with a blended pos side),
    When _combined_appmag_node is evaluated on known per-star magnitudes,
    Then each differential color row equals the four-flux expression
      computed here in numpy, and every other row is unchanged.
    """
    # ARRANGE
    rows = [
        _row("2MASS/2MASS.J", 9.0, photType={"pos": ["A"]}),
        _row(
            "2MASS/2MASS.J",
            0.4,
            neg_filter="2MASS/2MASS.Ks",
            photType={"pos": ["A"], "neg": ["A"]},
        ),
        _row(
            "2MASS/2MASS.J",
            0.1,
            color_filter="2MASS/2MASS.Ks",
            photType={"pos": ["A"], "neg": ["B"]},
        ),
        _row(
            "2MASS/2MASS.H",
            -0.2,
            color_filter="2MASS/2MASS.J",
            photType={"pos": ["A", "C"], "neg": ["B"]},
        ),
    ]
    sed = _loaded_sed(tmp_path, rows, star_names=("A", "B", "C"))
    # row columns 0..3, the color row's neg side (4), then the two
    # differential colors' second filters (5, 6)
    assert sed.neg_col.tolist() == [0, 4, 2, 3]
    assert sed.color_col.tolist() == [0, 1, 5, 6]
    assert sed.mist_filters[4:7] == ["2MASS_Ks", "2MASS_Ks", "2MASS_J"]
    m = 8.0 + 4.0 * np.random.default_rng(8911).random(
        (3, len(sed.mist_filters))
    )
    sed._m_pred_matrix = pt.as_tensor_variable(m)

    # ACT
    got = sed._combined_appmag_node(system=None).eval()

    # ASSERT
    F = 10 ** (-0.4 * m)
    expected = [
        -2.5 * np.log10(F[0, 0]),
        -2.5 * np.log10(F[0, 1] / F[0, 4]),
        -2.5
        * (
            np.log10(F[0, 2])
            - np.log10(F[1, 2])
            - np.log10(F[0, 5])
            + np.log10(F[1, 5])
        ),
        -2.5
        * (
            np.log10(F[0, 3] + F[2, 3])
            - np.log10(F[1, 3])
            - np.log10(F[0, 6] + F[2, 6])
            + np.log10(F[1, 6])
        ),
    ]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert sed.dcolor_rows.tolist() == [False, False, True, True]


def test_a_file_without_one_builds_the_same_graph(tmp_path):
    """No differential color row: the combined node is the color-row
    graph (no set_subtensor), so every shipped start logp is unchanged."""
    rows = [_row("2MASS/2MASS.J", 9.0, photType={"pos": ["A"]})]
    sed = _loaded_sed(tmp_path, rows)
    sed._m_pred_matrix = pt.as_tensor_variable(np.full((1, 1), 9.0))
    node = sed._combined_appmag_node(system=None)
    from pytensor.graph.traversal import ancestors

    assert not any(
        "IncSubtensor" in type(getattr(v.owner, "op", None)).__name__
        for v in ancestors([node])
        if v.owner is not None
    )


def test_every_magnitude_system_cancels(tmp_path):
    """
    Given the same differential color stated all-Vega and with an AB side,
    When the SED loads them,
    Then both are fitted at the reported number (no offset applies), and
      each filter's system is still resolved.
    """
    common = dict(
        color_filter="2MASS/2MASS.J", photType={"pos": ["A"], "neg": ["B"]}
    )
    vega = _loaded_sed(
        tmp_path, [_row("SLOAN/SDSS.g", 0.3, **common)], ("A", "B")
    )
    ab = _loaded_sed(
        tmp_path,
        [_row("SLOAN/SDSS.g", 0.3, magsys="AB", **common)],
        ("A", "B"),
    )
    np.testing.assert_array_equal(vega.mag, [0.3])
    np.testing.assert_array_equal(ab.mag, [0.3])
    assert ab.magsys == ["AB"]


@pytest.mark.parametrize(
    "extra, match",
    [
        (
            {
                "neg_filter": "2MASS/2MASS.H",
                "photType": {"pos": ["A"], "neg": ["B"]},
            },
            "color_filter and neg_filter",
        ),
        ({"photType": {"pos": ["A"]}}, "non-empty 'neg'"),
        (
            {"photType": {"pos": ["A"], "neg": ["B"]}, "neg_magsys": "Vega"},
            "neg_magsys",
        ),
    ],
)
def test_malformed_rows_raise(tmp_path, extra, match):
    row = _row("2MASS/2MASS.J", 0.1, color_filter="2MASS/2MASS.Ks", **extra)
    with pytest.raises(ValueError, match=match):
        _loaded_sed(tmp_path, [row], ("A", "B"))


def test_naming_the_rows_own_filter_raises(tmp_path):
    """color_filter naming the row's own column (another alias) would be
    identically zero."""
    row = _row(
        "2MASS/2MASS.J",
        0.1,
        color_filter="2MASS_J",
        photType={"pos": ["A"], "neg": ["B"]},
    )
    with pytest.raises(ValueError, match="identically zero"):
        _loaded_sed(tmp_path, [row], ("A", "B"))
