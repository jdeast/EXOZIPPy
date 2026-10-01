"""AB photometry in the SED (review 1.9.1).

Every NextGen BC column is Vega-referenced, so an AB magnitude has to be
moved onto Vega by its band's m_AB - m_Vega before it is fitted; until this
was fixed an SDSS, PS1 or GALEX row was fitted as though it were Vega, wrong
by up to ~2 mag against errors of a few hundredths.  The offset is shipped by
the BC generator beside the tables (``NextGen.magsys.csv``), computed from
the same filter profile, flux weighting and Vega zeropoint as the column's
BCs, and is checked against the column's own metadata at load.

A row that states no system is in its filter's NATIVE system (JDE ruling
2026-10-01); mkticsed writes the system on every row regardless (same day).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from exozippy.components.sed.bc_grid import (
    DEFAULT_FILTER_ROOT,
    DEFAULT_MODEL_ROOT,
    read_bc_meta,
)
from exozippy.components.sed.magsys import (
    AB,
    VEGA,
    check_record_matches_column,
    magsys_table_path,
    parse_magsys,
    read_magsys_table,
)

_MODEL_ROOT = DEFAULT_MODEL_ROOT


def _sed_file(tmp_path, rows):
    """Write a one-star .sed file from (name, mag, magsys, photType) rows;
    magsys None leaves the key out."""
    filters = []
    for name, mag, magsys, phot in rows:
        row = {"name": name, "mag": mag, "err": 0.02}
        if magsys is not None:
            row["magsys"] = magsys
        if phot is not None:
            row["photType"] = phot
        filters.append(row)
    p = tmp_path / "test_star.sed"
    p.write_text(
        yaml.safe_dump({"model": "NextGen", "nstars": 1, "filters": filters})
    )
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
    config = {
        "file": _sed_file(tmp_path, rows),
        "model_root": str(_MODEL_ROOT),
    }
    sed = SED(config, cm)
    star = SimpleNamespace(names=list(star_names), n_elements=len(star_names))
    sed.load_data(SimpleNamespace(star=star))
    return sed


def _offset(column):
    return float(
        read_magsys_table(_MODEL_ROOT, "NextGen").loc[column, "ab_minus_vega"]
    )


# ---------------------------------------------------------------------------
# Section 1 -- the conversion
# ---------------------------------------------------------------------------


def test_an_ab_row_is_fitted_on_vega_by_exactly_the_recorded_offset(tmp_path):
    """
    Given an SDSS g row stated as AB,
    When the SED loads it,
    Then the fitted magnitude is the reported one minus the column's
      recorded m_AB - m_Vega, and its error is unchanged.

    Fails on master: the row was fitted at its AB value against a Vega BC.
    """
    # ARRANGE
    rows = [
        ("SLOAN/SDSS.g", 12.0, "AB", None),
        ("2MASS/2MASS.J", 10.0, "Vega", None),
    ]

    # ACT
    sed = _loaded_sed(tmp_path, rows)

    # ASSERT
    assert sed.magsys == [AB, VEGA]
    assert sed.mag[0] == 12.0 - _offset("SDSS_g")
    assert sed.mag_reported[0] == 12.0
    assert sed.err[0] == 0.02
    # The sign and size of the shipped offset: Vega is m_AB ~ -0.11 in g.
    assert _offset("SDSS_g") == pytest.approx(-0.111, abs=0.005)


def test_a_vega_row_is_untouched_bit_for_bit(tmp_path):
    """
    Given Vega rows (stated and unstated in a Vega-native band),
    When the SED loads them,
    Then the fitted magnitude IS the reported one -- the shipped all-Vega
      examples keep a bit-identical start logp.
    """
    # ARRANGE
    rows = [
        ("2MASS/2MASS.J", 10.123456789, "Vega", None),
        ("WISE/WISE.W1", 9.87654321, None, None),
    ]

    # ACT
    sed = _loaded_sed(tmp_path, rows)

    # ASSERT
    assert sed.magsys == [VEGA, VEGA]
    np.testing.assert_array_equal(sed.mag, sed.mag_reported)
    assert not np.any(sed.ab_minus_vega)


def test_an_unstated_row_takes_its_filters_native_system(tmp_path):
    """
    Given rows with no magsys in an AB-native band (GALEX FUV) and a
      Vega-native one (Gaia G),
    When the SED loads them,
    Then the GALEX row is read as AB and converted, the Gaia row as Vega.
    """
    # ARRANGE
    rows = [
        ("GALEX/GALEX.FUV", 18.0, None, None),
        ("GAIA/GAIA2r.G", 10.0, None, None),
    ]

    # ACT
    sed = _loaded_sed(tmp_path, rows)

    # ASSERT
    assert sed.magsys == [AB, VEGA]
    assert sed.mag[0] == 18.0 - _offset("GALEX_FUV")
    assert sed.mag[1] == 10.0


def test_a_differential_ab_row_is_not_shifted(tmp_path):
    """
    Given a differential (photType neg) row stated as AB,
    When the SED loads it,
    Then it is not shifted: -2.5 log10(F_pos/F_neg) in one band is the same
      number on either system.
    """
    # ARRANGE
    rows = [
        ("SLOAN/SDSS.i", 1.5, "AB", {"pos": ["B"], "neg": ["A"]}),
        ("2MASS/2MASS.J", 10.0, None, None),
    ]

    # ACT
    sed = _loaded_sed(tmp_path, rows, star_names=("A", "B"))

    # ASSERT
    assert sed.magsys[0] == AB
    assert sed.mag[0] == 1.5


@pytest.mark.parametrize("spelling", ["AB", "Vega"])
def test_the_exact_spellings_are_accepted(spelling):
    """
    Given the exact spelling of a supported system,
    When it is parsed at the .sed boundary,
    Then it is returned unchanged as the internal spelling.
    """
    # ACT / ASSERT
    assert parse_magsys(spelling, "row") == {"AB": AB, "Vega": VEGA}[spelling]


@pytest.mark.parametrize(
    "spelling, hint",
    [
        ("ab", "AB"),
        ("Ab", "AB"),
        (" AB ", "AB"),
        ("vega", "Vega"),
        ("VEGA", "Vega"),
    ],
)
def test_a_case_variant_raises_with_a_hint(spelling, hint):
    """
    Given a case (or whitespace) variant of a supported system,
    When it is parsed at the .sed boundary,
    Then it RAISES, suggesting the exact spelling -- user-facing names are
      case-sensitive (JDE 2026-10-01: "g is different than G"); suggesting
      is fine, accepting is not.
    """
    # ACT / ASSERT
    with pytest.raises(ValueError, match=f"Did you mean '{hint}'"):
        parse_magsys(spelling, "row 0")


def test_a_lowercase_ab_row_raises_naming_the_row(tmp_path):
    """
    Given a .sed row written `magsys: ab`,
    When the SED reads the file,
    Then it raises naming the row and filter, with the "did you mean" hint.
    """
    # ARRANGE
    rows = [("SLOAN/SDSS.g", 12.0, "ab", None)]

    # ACT / ASSERT
    with pytest.raises(
        ValueError, match=r"filter row 0 .*SDSS\.g.*Did you mean 'AB'"
    ):
        _loaded_sed(tmp_path, rows)


def test_an_unknown_system_raises_naming_the_row(tmp_path):
    """
    Given a row stating a system the SED does not support,
    When the SED reads the file,
    Then it raises naming the file, the row and the spelling.
    """
    # ARRANGE
    rows = [("SLOAN/SDSS.r", 12.0, "STMAG", None)]

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"filter row 0 .*SDSS\.r.*STMAG"):
        _loaded_sed(tmp_path, rows)


def test_a_filter_with_no_native_system_must_state_one(tmp_path):
    """
    Given a Zorro row (speckle contrast photometry, no native system),
    When it states no magsys,
    Then the SED raises naming the filter; stating one is accepted.
    """
    # ARRANGE
    rows = [("Gemini/Zorro.EO_562", 1.0, None, {"pos": ["B"], "neg": ["A"]})]

    # ACT / ASSERT
    with pytest.raises(ValueError, match="Zorro_EO_562 has no native"):
        _loaded_sed(tmp_path, rows, star_names=("A", "B"))
    rows = [("Gemini/Zorro.EO_562", 1.0, "Vega", {"pos": ["B"], "neg": ["A"]})]
    assert _loaded_sed(tmp_path, rows, star_names=("A", "B")).mag[0] == 1.0


# ---------------------------------------------------------------------------
# Section 2 -- the offsets are the generator's, for the columns that ship
# ---------------------------------------------------------------------------


def test_the_record_describes_every_shipped_column_as_it_stands():
    """
    Given the shipped BC tables and the generator's FILTER_SETS,
    When the magnitude-system record is read,
    Then it has exactly one row per shipped column, each matching that
      column's svo_id, flux weighting and Vega zeropoint.

    This is the drift check the SED runs at load, applied to every column.
    """
    from exozippy.components.sed.bc_grid import find_bc_table
    from exozippy.models.NextGen.generate_NextGen_BC_Tables import FILTER_SETS

    # ARRANGE
    record = read_magsys_table(_MODEL_ROOT, "NextGen")
    path = magsys_table_path(_MODEL_ROOT, "NextGen")
    seen = []

    # ACT / ASSERT
    for fac in FILTER_SETS:
        meta = read_bc_meta(find_bc_table(_MODEL_ROOT, "NextGen", fac))
        assert meta["mag_system"] == VEGA
        for col, fmeta in meta["filters"].items():
            check_record_matches_column(record.loc[col], col, fmeta, path)
            seen.append(col)
    assert sorted(seen) == sorted(record.index)


def test_the_recorded_offsets_are_what_the_generator_computes():
    """
    Given filters whose profiles ship with the package (so no SVO fetch),
      including two AB-native Roman bands,
    When the generator's magsys_table recomputes their rows,
    Then the shipped record matches them -- the record is the generator's
      output, not a hand-copied table.
    """
    from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen

    # ARRANGE
    sets = {
        "2MASS": ["2MASS/2MASS.J", "2MASS/2MASS.H", "2MASS/2MASS.Ks"],
        "WISE": ["WISE/WISE.W1", "WISE/WISE.W2"],
        "Roman": ["Roman/WFI.F087", "Roman/WFI.F146"],
    }
    record = read_magsys_table(_MODEL_ROOT, "NextGen")

    # ACT
    fresh = gen.magsys_table(sets).set_index("column")

    # ASSERT
    for col, row in fresh.iterrows():
        assert record.loc[col, "native_system"] == row["native_system"]
        assert record.loc[col, "ab_minus_vega"] == pytest.approx(
            row["ab_minus_vega"], abs=1e-8
        )
    assert fresh.loc["WFI_F087", "native_system"] == AB


def test_the_ab_offset_is_the_ab_magnitude_of_the_vega_zero():
    """
    Given a column's Vega zeropoint and its recorded offset,
    When a star's flux is computed from its AB magnitude (AB zero point
      <F_AB> = ZP_Vega * 10**(0.4 * offset)) and from the converted Vega one,
    Then the two fluxes agree -- the conversion has the right sign.

    Checked independently for 2MASS J against the textbook value: Vega is
    m_AB ~ +0.89 in J (Blanton & Roweis 2007 quote 0.91).
    """
    # ARRANGE
    record = read_magsys_table(_MODEL_ROOT, "NextGen")
    zp_vega = record.loc["2MASS_J", "zeropoint_Fl_Vega"]
    off = record.loc["2MASS_J", "ab_minus_vega"]
    m_ab = 11.0

    # ACT
    flux_from_ab = zp_vega * 10 ** (0.4 * off) * 10 ** (-0.4 * m_ab)
    flux_from_vega = zp_vega * 10 ** (-0.4 * (m_ab - off))

    # ASSERT
    assert flux_from_ab == pytest.approx(flux_from_vega, rel=1e-12)
    assert off == pytest.approx(0.89, abs=0.03)


def test_native_systems_agree_with_mist_where_mist_lists_the_filter():
    """
    Given MIST's filter_magsys.txt,
    When every record row MIST also lists is compared,
    Then the native systems agree (SDSS/PS1/GALEX AB; 2MASS/Gaia/WISE Vega).
    """
    # ARRANGE
    mist = pd.read_csv(DEFAULT_FILTER_ROOT / "filter_magsys.txt", sep=r"\s+")
    mist = dict(zip(mist["filter"], mist["system"]))
    record = read_magsys_table(_MODEL_ROOT, "NextGen")

    # ACT
    both = [c for c in record.index if c in mist]

    # ASSERT
    assert {"SDSS_u", "PS_z", "GALEX_FUV", "2MASS_J"} <= set(both)
    for col in both:
        assert record.loc[col, "native_system"] == mist[col], col


def test_a_record_for_a_different_zeropoint_is_refused():
    """
    Given a record row whose Vega zeropoint is not the column's,
    When it is checked against the column's filter_meta,
    Then it raises naming the column -- a regenerated column cannot keep a
      stale offset.
    """
    # ARRANGE
    record = read_magsys_table(_MODEL_ROOT, "NextGen")
    row = record.loc["SDSS_u"].copy()
    meta = {
        "svo_id": row["svo_id"],
        "flux_weighting": row["flux_weighting"],
        "zeropoint_Fl_Vega": row["zeropoint_Fl_Vega"] * 1.01,
    }

    # ACT / ASSERT
    with pytest.raises(ValueError, match="SDSS_u.*zeropoint_Fl_Vega"):
        check_record_matches_column(row, "SDSS_u", meta, "rec.csv")


# ---------------------------------------------------------------------------
# Section 3 -- the figure shows what the likelihood fits
# ---------------------------------------------------------------------------


def _likelihood_observed(sed, monkeypatch):
    """The observed vector the SED likelihood is built on."""
    import pymc as pm
    import pytensor.tensor as pt

    monkeypatch.setattr(
        sed, "_combined_appmag_node", lambda system: pt.zeros(sed.nfilters)
    )
    monkeypatch.setattr(sed, "seen_star_mask", lambda system: [True])
    monkeypatch.setattr(sed, "_declare_grid_support", lambda system: None)
    sed.errscale = SimpleNamespace(value=1.0)
    val = SimpleNamespace(value=pt.ones(1))
    star = SimpleNamespace(teff=val, teffsed=val, fbol=val, fbolsed=val)
    system = SimpleNamespace(star=star)
    with pm.Model() as model:
        sed.build_likelihood(model, system)
    return np.asarray(model["sed_mag_data"].get_value()), system


def test_the_plotted_magnitude_is_the_fitted_one(tmp_path, monkeypatch):
    """
    Given an SED with an AB row,
    When the likelihood is built and the figures read their photometry,
    Then the GUI chart's points, the PDF plot's mag_obs and the
      likelihood's observed vector are the same converted numbers, and the
      caption names the plotted system.
    """
    from exozippy.components.sed.plot import Plot

    # ARRANGE
    # Roman F146: AB-native AND its profile ships, so the chart (which reads
    # effective wavelengths) needs no SVO fetch.
    rows = [
        ("Roman/WFI.F146", 13.0, "AB", None),
        ("2MASS/2MASS.J", 12.0, None, None),
    ]
    sed = _loaded_sed(tmp_path, rows)
    for name in (
        "_load_spectra_data",
        "_load_model_grid_yaml",
        "_load_filter_data",
    ):
        monkeypatch.setattr(Plot, name, lambda self: None)

    # ACT
    observed, system = _likelihood_observed(sed, monkeypatch)
    chart = sed.plot_data(system, point=None)[0]
    plot_obj = Plot(
        SimpleNamespace(sed=sed, star=SimpleNamespace(names=["A"])), [{}]
    )

    # ASSERT
    assert observed[0] == 13.0 - _offset("WFI_F146")
    np.testing.assert_array_equal(np.asarray(chart.traces[0].y), observed)
    np.testing.assert_array_equal(plot_obj.mag_obs, observed)
    assert "(Vega)" in chart.ylabel


def test_the_conversion_declares_its_sentence(tmp_path, monkeypatch):
    """
    Given an SED with an AB row,
    When the likelihood is built,
    Then the modeling draft says the AB photometry was converted, citing
      Oke & Gunn (1983) -- and an all-Vega SED declares nothing.
    """
    from exozippy.outputs.prose import get_collector

    # ARRANGE
    ab = _loaded_sed(tmp_path, [("SLOAN/SDSS.z", 13.0, "AB", None)])
    vega = _loaded_sed(tmp_path, [("2MASS/2MASS.J", 12.0, None, None)])

    # ACT
    _, sys_ab = _likelihood_observed(ab, monkeypatch)
    _, sys_vega = _likelihood_observed(vega, monkeypatch)

    # ASSERT
    text_ab = " ".join(s.text for s in get_collector(sys_ab).sentences())
    assert r"\citep{Oke:1983}" in text_ab
    assert "SDSS.z" in text_ab
    assert "sed.magsys" in get_collector(sys_ab)
    assert "sed.magsys" not in get_collector(sys_vega)
