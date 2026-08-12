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


def _gaia_dr3_table(plx=4.587, e_plx=0.0143):
    """One-row Gaia DR3 cone result for the target, out of zero-point range.

    Solved = 3 keeps it outside the Lindegren+2021 prescription, so the
    zero point is 0 and the written prior is the raw catalog parallax --
    which keeps the assertions on the value/error, not on gaiadr3-zeropoint
    (an optional dependency).
    """
    return Table(
        {
            "Source": ["999"],
            "RA_ICRS": [TARGET_RA],
            "DE_ICRS": [TARGET_DEC],
            "Plx": [plx],
            "e_Plx": [e_plx],
            "Gmag": [10.0],
            "e_Gmag": [0.002],
            "BPmag": [10.4],
            "e_BPmag": [0.003],
            "RPmag": [9.5],
            "e_RPmag": [0.003],
            "RUWE": [1.0],
            "nueff": [1.5],
            "pscol": [1.4],
            "ELAT": [0.0],
            "Solved": [3],
            "pmRA": [10.0],
            "pmDE": [-5.0],
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


def _read_priors(path):
    """Parse a written params YAML into {param path: {field: float}}."""
    out = {}
    key = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.rstrip().endswith(":"):
            key = line.rstrip()[:-1]
            out[key] = {}
        elif key is not None and ":" in line:
            field, val = line.split(":", 1)
            out[key][field.strip()] = float(val)
    return out


def _prior_notes(path):
    """The commented header lines of a written params YAML."""
    return [
        ln.lstrip("# ").rstrip()
        for ln in path.read_text().splitlines()
        if ln.startswith("#")
    ]


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


def test_apass_bv_kept_when_no_tycho_source_exists(tmp_path, patched_catalogs):
    """
    Given APASS B/V photometry and no Tycho-2 source anywhere in the cone,
    When mkticsed writes the SED,
    Then the Bessell B and V rows are written -- with no Tycho counterpart
      there is nothing to deduplicate against, so the NaN reference must not
      suppress them (mkticsed.pro uses a -99 sentinel and keeps them).
    """
    # Arrange: no "I/259/TYC2" entry at all -> the query returns None.
    patched_catalogs["UCAC4"] = _ucac_table(bmag=11.0, vmag=10.5)

    # Act
    rows = _run(tmp_path, priorfile=str(tmp_path / "p.yaml"))

    # Assert
    assert rows["Generic/Bessell.B"] == (
        pytest.approx(11.0),
        pytest.approx(0.03),
    )
    assert rows["Generic/Bessell.V"] == (
        pytest.approx(10.5),
        pytest.approx(0.03),
    )


# --- 3.9: the astrometric prior belongs in parallax space ---------------------


def test_gaia_parallax_written_as_a_parallax_prior(tmp_path, patched_catalogs):
    """
    Given a Gaia DR3 row with a positive parallax,
    When mkticsed writes the params file,
    Then it carries a Gaussian prior on star.<name>.parallax whose mu is
      the (zero-point-corrected) measurement in mas and whose sigma is
      sqrt(e_Plx^2 + 0.01^2) -- and no distance prior, which was a
      first-order propagation of a nonlinear map (EXOFASTv2's
      mkticsed.pro writes `parallax`, never a distance prior).
    """
    # Arrange
    patched_catalogs["I/355/gaiadr3"] = _gaia_dr3_table(
        plx=4.587, e_plx=0.0143
    )
    priorfile = tmp_path / "p.yaml"

    # Act
    _run(tmp_path, priorfile=str(priorfile))
    priors = _read_priors(priorfile)

    # Assert
    assert "star.Host.parallax" in priors
    assert priors["star.Host.parallax"]["mu"] == pytest.approx(4.587)
    assert priors["star.Host.parallax"]["sigma"] == pytest.approx(
        np.sqrt(0.0143**2 + 0.01**2), abs=1e-5
    )
    assert "star.Host.distance" not in priors


def test_negative_parallax_is_still_written_as_a_prior(
    tmp_path, patched_catalogs
):
    """
    Given a Gaia DR3 row whose parallax is negative,
    When mkticsed writes the params file,
    Then the prior is written live with the negative mu (in parallax
      space `distance` is the sampled, positive-bounded coordinate, so a
      Gaussian centred below zero is a finite one-sided penalty), no
      distance seed is written -- nothing may invert a negative parallax
      -- and the header says so.
    """
    # Arrange
    patched_catalogs["I/355/gaiadr3"] = _gaia_dr3_table(plx=-0.5, e_plx=0.2)
    priorfile = tmp_path / "p.yaml"

    # Act
    _run(tmp_path, priorfile=str(priorfile))
    priors = _read_priors(priorfile)
    notes = " ".join(_prior_notes(priorfile))

    # Assert
    assert priors["star.Host.parallax"]["mu"] == pytest.approx(-0.5)
    assert priors["star.Host.parallax"]["sigma"] == pytest.approx(
        np.sqrt(0.2**2 + 0.01**2), abs=1e-5
    )
    assert "star.Host.distance" not in priors
    assert "NEGATIVE" in notes


def test_dr2_fallback_also_writes_a_parallax_prior(tmp_path, patched_catalogs):
    """
    Given no Gaia DR3 row but a matched Gaia DR2 one,
    When mkticsed writes the params file,
    Then the Lindegren+2018-corrected DR2 parallax is written as a
      parallax prior too, with sigma = sqrt((1.08*e_Plx)^2 + sigma_s^2)
      exactly as mkticsed.pro writes it.
    """
    # Arrange: DR2 only. Gmag > 13 selects the 0.043 mas systematic.
    patched_catalogs["I/345/gaia2"] = Table(
        {
            "Source": ["999"],
            "Plx": [2.0],
            "e_Plx": [0.05],
            "Gmag": [14.0],
        }
    )
    priorfile = tmp_path / "p.yaml"

    # Act
    _run(tmp_path, priorfile=str(priorfile))
    priors = _read_priors(priorfile)

    # Assert
    assert priors["star.Host.parallax"]["mu"] == pytest.approx(2.030)
    assert priors["star.Host.parallax"]["sigma"] == pytest.approx(
        np.sqrt((1.08 * 0.05) ** 2 + 0.043**2), abs=1e-5
    )
    assert "star.Host.distance" not in priors
