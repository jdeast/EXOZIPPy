"""Tests for the mkticsed catalog-to-SED utility.

Every catalog query is monkeypatched with a synthetic astropy Table, so no
test here touches Vizier, MAST or IRSA. Assertions are on the rows actually
written into the .sed file, not on module internals.
"""

import numpy as np
import pytest
from astropy.table import Table

from exozippy.utilities import mkticsed as mk

# --- synthetic catalogs -------------------------------------------------------

TARGET_RA = 100.0
TARGET_DEC = 20.0
# A contaminant 30" away -- inside the default 120" cone, but not the target.
OTHER_RA = TARGET_RA + 30.0 / 3600.0 / np.cos(np.radians(TARGET_DEC))
OTHER_DEC = TARGET_DEC


def _tic_table():
    """One-row TICv8.2 result for TIC 12345678."""
    return Table(
        {
            "TIC": ["12345678"],
            "Disp": [""],
            "m_TIC": ["-1"],
            "RAJ2000": [TARGET_RA],
            "DEJ2000": [TARGET_DEC],
            "Mass": [1.0],
            "Rad": [1.0],
            "Teff": [5800.0],
            "[M/H]": [0.0],
            "e_[M/H]": [0.08],
            "E_B-V": [0.01],
            "s_E_B-V": [0.01],
            "GAIA": ["999"],
            "_2MASS": ["j2m"],
            "WISEA": ["wise"],
            "TYC": ["1234-5678-1"],
        }
    )


def _tycho_table(rows):
    """Tycho-2 cone result. `rows` = list of (ra, dec, BTmag, VTmag) tuples."""
    return Table(
        {
            "RAmdeg": [r[0] for r in rows],
            "DEmdeg": [r[1] for r in rows],
            "BTmag": [r[2] for r in rows],
            "e_BTmag": [0.03] * len(rows),
            "VTmag": [r[3] for r in rows],
            "e_VTmag": [0.03] * len(rows),
        }
    )


def _ucac_table(bmag=11.0, vmag=10.5, eg=3.0, er=3.0, ei=3.0):
    """One-row UCAC4/APASS cone result centered on the target.

    UCAC4 stores the APASS errors as hundredths of a magnitude; 99 is the
    catalog's "no data" sentinel.
    """
    return Table(
        {
            "RAJ2000": [TARGET_RA],
            "DEJ2000": [TARGET_DEC],
            "Bmag": [bmag],
            "e_Bmag": [3.0],
            "Vmag": [vmag],
            "e_Vmag": [3.0],
            "gmag": [10.8],
            "e_gmag": [eg],
            "rmag": [10.3],
            "e_rmag": [er],
            "imag": [10.1],
            "e_imag": [ei],
        }
    )


@pytest.fixture
def patched_catalogs(monkeypatch):
    """Route every network call through a dict of synthetic tables.

    Yields the dict; tests fill in the catalogs they care about and every
    other catalog resolves to None (not found).
    """
    catalogs = {}

    def fake_query_id(catalog, target_id):
        return catalogs.get(catalog)

    def fake_query_region(catalog, ra, dec, radius_arcmin):
        return catalogs.get(catalog)

    monkeypatch.setattr(mk, "query_id", fake_query_id)
    monkeypatch.setattr(mk, "query_region", fake_query_region)
    monkeypatch.setattr(mk, "schlegel_av", lambda ra, dec: 0.5)

    catalogs["IV/39/tic82"] = _tic_table()
    return catalogs


# --- .sed reader --------------------------------------------------------------


def _read_sed_rows(path):
    """Parse a written .sed YAML into {band_name: (mag, err)} for live rows.

    Commented-out (disabled) entries are deliberately excluded -- they are not
    part of the fit.
    """
    rows = {}
    name = None
    entry = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            if name is not None:
                rows[name] = (entry.get("mag"), entry.get("err"))
            name = stripped.split(":", 1)[1].strip().strip('"')
            entry = {}
        elif name is not None and stripped.startswith(("mag:", "err:")):
            field, val = stripped.split(":", 1)
            entry[field.strip()] = float(val)
    if name is not None:
        rows[name] = (entry.get("mag"), entry.get("err"))
    return rows


def _run(tmp_path, **kwargs):
    """Run mkticsed into tmp_path and return the parsed .sed rows."""
    mk.mkticsed(
        ticid="12345678",
        star_name="Host",
        outpath=str(tmp_path),
        ucac=True,
        tycho=True,
        **kwargs,
    )
    return _read_sed_rows(tmp_path / "12345678.sed")


# --- 2.5.5: APASS-vs-Tycho dedup must use the matched Tycho row ---------------


def test_apass_dedup_uses_matched_tycho_row_not_row_zero(
    tmp_path, patched_catalogs
):
    """
    Given a Tycho cone whose row 0 is a 30" contaminant and whose *matched*
      (nearest) row is the target, and an APASS B/V that duplicates the
      matched star's BT/VT,
    When mkticsed writes the SED,
    Then the duplicate APASS B and V rows are dropped (the dedup compared
      against the matched star, not against row 0).
    """
    # Arrange: row 0 = contaminant (BT 15), row 1 = target (BT 11, VT 10.5).
    patched_catalogs["I/259/TYC2"] = _tycho_table(
        [
            (OTHER_RA, OTHER_DEC, 15.0, 14.5),
            (TARGET_RA, TARGET_DEC, 11.0, 10.5),
        ]
    )
    patched_catalogs["UCAC4"] = _ucac_table(bmag=11.0, vmag=10.5)

    # Act
    rows = _run(tmp_path, priorfile=str(tmp_path / "p.yaml"))

    # Assert: the Tycho rows are the matched star's, and APASS B/V are deduped.
    assert rows["TYCHO/TYCHO.B"][0] == pytest.approx(11.0)
    assert rows["TYCHO/TYCHO.V"][0] == pytest.approx(10.5)
    assert "Generic/Bessell.B" not in rows
    assert "Generic/Bessell.V" not in rows


def test_apass_kept_when_only_unmatched_tycho_row_duplicates_it(
    tmp_path, patched_catalogs
):
    """
    Given a Tycho cone whose row 0 is a contaminant that happens to share the
      APASS B/V magnitudes, while the matched row is a different star,
    When mkticsed writes the SED,
    Then the APASS B and V rows survive -- an unrelated star must not suppress
      the target's photometry.
    """
    # Arrange: row 0 = contaminant with the same mags as APASS; row 1 = target.
    patched_catalogs["I/259/TYC2"] = _tycho_table(
        [
            (OTHER_RA, OTHER_DEC, 11.0, 10.5),
            (TARGET_RA, TARGET_DEC, 15.0, 14.5),
        ]
    )
    patched_catalogs["UCAC4"] = _ucac_table(bmag=11.0, vmag=10.5)

    # Act
    rows = _run(tmp_path, priorfile=str(tmp_path / "p.yaml"))

    # Assert
    assert rows["TYCHO/TYCHO.B"][0] == pytest.approx(15.0)
    assert rows["Generic/Bessell.B"][0] == pytest.approx(11.0)
    assert rows["Generic/Bessell.V"][0] == pytest.approx(10.5)


# --- 2.5.6: the 99 "no data" sentinel must gate g/r/i too ---------------------


def test_apass_gri_with_sentinel_error_is_skipped(tmp_path, patched_catalogs):
    """
    Given APASS g/r/i whose UCAC4 errors are the 99 "no data" sentinel,
    When mkticsed writes the SED,
    Then no SDSS g/r/i rows are written (pre-fix they were written with a
      fabricated 0.99 mag error).
    """
    # Arrange
    patched_catalogs["UCAC4"] = _ucac_table(eg=99.0, er=99.0, ei=99.0)

    # Act
    rows = _run(tmp_path, priorfile=str(tmp_path / "p.yaml"))

    # Assert
    assert "SLOAN/SDSS.g" not in rows
    assert "SLOAN/SDSS.r" not in rows
    assert "SLOAN/SDSS.i" not in rows


def test_apass_gri_with_real_errors_is_written(tmp_path, patched_catalogs):
    """
    Given APASS g/r/i with genuine UCAC4 errors (hundredths of a magnitude),
    When mkticsed writes the SED,
    Then the SDSS g/r/i rows are written with those errors -- the sentinel
      guard must not swallow real data.
    """
    # Arrange: 3 hundredths -> 0.03 mag, above the 0.02 floor.
    patched_catalogs["UCAC4"] = _ucac_table(eg=3.0, er=3.0, ei=5.0)

    # Act
    rows = _run(tmp_path, priorfile=str(tmp_path / "p.yaml"))

    # Assert
    assert rows["SLOAN/SDSS.g"] == (pytest.approx(10.8), pytest.approx(0.03))
    assert rows["SLOAN/SDSS.r"] == (pytest.approx(10.3), pytest.approx(0.03))
    assert rows["SLOAN/SDSS.i"] == (pytest.approx(10.1), pytest.approx(0.05))


def test_apass_bv_sentinel_error_is_skipped(tmp_path, patched_catalogs):
    """
    Given APASS B/V errors of 99 and no Tycho counterpart,
    When mkticsed writes the SED,
    Then no Bessell B/V rows are written -- the pre-existing B/V sentinel
      guard is pinned alongside the new g/r/i one.
    """
    # Arrange
    ucac = _ucac_table(bmag=11.0, vmag=10.5)
    ucac["e_Bmag"] = [99.0]
    ucac["e_Vmag"] = [99.0]
    patched_catalogs["UCAC4"] = ucac

    # Act
    rows = _run(tmp_path, priorfile=str(tmp_path / "p.yaml"))

    # Assert
    assert "Generic/Bessell.B" not in rows
    assert "Generic/Bessell.V" not in rows
