"""
Tests for the getstar.pro / read_mamajek.pro port
(components/star/mamajek.py) against the shipped EEM dwarf table.
"""

import numpy as np
import pytest

from exozippy.components.star.mamajek import get_star, read_mamajek


def test_read_mamajek_parses_the_whole_table():
    """118 spectral types, 30 numeric columns plus SpT, placeholders NaN."""
    t = read_mamajek()
    assert len(t["SpT"]) == 118
    assert len(t) == 31
    g2 = t["SpT"].index("G2V")
    assert t["Teff"][g2] == 5770.0
    assert np.isclose(t["Msun"][g2], 1.00)
    # hyphenated headers are pythonized
    assert "B_V" in t
    # O-star placeholder columns parse as NaN, not as a crash
    assert np.isnan(t["M_J"][t["SpT"].index("O3V")])


def test_minmass_filters_rows():
    t = read_mamajek(minmass=0.1)
    assert np.all(t["Msun"] >= 0.1)
    assert len(t["SpT"]) == len(t["Msun"])


def test_get_star_interpolates_every_column():
    """One measured quantity yields the full dwarf description."""
    s = get_star("teff", 5772.0)
    assert np.isclose(s["Msun"], 1.0, atol=0.02)
    assert np.isclose(s["R_Rsun"], 1.01, atol=0.02)
    assert s["SpT"] == "G2V"
    # case-insensitive tagname, either direction of monotonicity
    s = get_star("MSUN", 0.6)
    assert 3800.0 < s["Teff"] < 4100.0


def test_get_star_spt_exact_match_only():
    s = get_star("SpT", "M0V")
    assert np.isclose(s["Teff"], 3850.0, atol=100.0)
    with pytest.raises(ValueError, match="cannot be interpolated"):
        get_star("SpT", "G2.7V")


def test_get_star_unknown_tagname_names_the_columns():
    with pytest.raises(ValueError, match="not supported"):
        get_star("bogus", 1.0)


def test_no_extrapolation():
    """A value beyond the table clamps to the end row (getstar.pro's
    documented behavior), never extrapolates."""
    s = get_star("Msun", 1000.0)
    t = read_mamajek()
    assert s["Teff"] == max(v for v in t["Teff"] if np.isfinite(v))
