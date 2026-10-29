"""AB <-> Vega magnitude-system offsets: published known answers and the
SED -> zeropoint round trip.

Review 7.16.2.  ``NextGen.magsys.csv`` records each BC column's
m_AB - m_Vega, and two consumers apply it in opposite directions: the SED
moves an AB catalog row onto the Vega-referenced BC (``m - offset``), and a
microlensing light curve whose zeropoint is AB moves the SED's Vega
prediction back (``MulensInstrument._predicted_mag_on_zp_system``,
``m + offset``).  tests/test_sed_magsys.py pins that the record is what the
generator computes and checks 2MASS J against one textbook value; it does
not compare the rest of the record with anything published, so a generator
that integrated every profile with the wrong flux weighting (photon vs
energy) or the wrong Vega spectrum would rewrite the whole table and stay
green.

The references, all m_AB - m_Vega in mag:

* Blanton & Roweis (2007, AJ 133, 734), Table 1 -- Bessell UBVRI, 2MASS
  JHKs, SDSS ugriz.
* Jarrett et al. (2011, ApJ 735, 112), Section 2.2 -- WISE W1-W4.
* Riello et al. (2021, A&A 649, A3), Table 3 -- the Gaia EDR3 AB and Vega
  zeropoints, whose difference is the offset.

Each published number came from a different filter curve, Vega spectrum
and integration from ours, so agreement is to a few hundredths, not to the
digits: the tolerance (0.04 mag) is the largest scatter the published
tables show among themselves for these bands, and the failure it exists for
-- an offset of the wrong sign or a weighting slip -- is 0.1 to 13 mag.
"""

import numpy as np
import pytest

from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT
from exozippy.components.sed.magsys import read_magsys_table

# Riello et al. (2021) Table 3: (Vega ZP, AB ZP) per EDR3 band.
_RIELLO = {
    "Gaia_G_EDR3": (25.6874, 25.8010),
    "Gaia_BP_EDR3": (25.3385, 25.3540),
    "Gaia_RP_EDR3": (24.7479, 25.1040),
}

PUBLISHED_AB_MINUS_VEGA = {
    # Blanton & Roweis (2007) Table 1
    "Bessell_U": 0.79,
    "Bessell_B": -0.09,
    "Bessell_V": 0.02,
    "Bessell_R": 0.21,
    "Bessell_I": 0.45,
    "2MASS_J": 0.91,
    "2MASS_H": 1.39,
    "2MASS_Ks": 1.85,
    "SDSS_u": 0.91,
    "SDSS_g": -0.08,
    "SDSS_r": 0.16,
    "SDSS_i": 0.37,
    "SDSS_z": 0.54,
    # Jarrett et al. (2011)
    "WISE_W1": 2.699,
    "WISE_W2": 3.339,
    "WISE_W3": 5.174,
    "WISE_W4": 6.620,
    # Riello et al. (2021): ZP_AB - ZP_Vega
    **{k: ab - vega for k, (vega, ab) in _RIELLO.items()},
}

TOLERANCE_MAG = 0.04


@pytest.fixture(scope="module")
def record():
    return read_magsys_table(DEFAULT_MODEL_ROOT, "NextGen")


@pytest.mark.parametrize("column", sorted(PUBLISHED_AB_MINUS_VEGA))
def test_recorded_ab_minus_vega_matches_the_published_value(record, column):
    """
    Given the shipped m_AB - m_Vega for a BC column,
    When it is compared with the published offset for that band,
    Then the two agree to 0.04 mag (sign included).
    """
    got = float(record.loc[column, "ab_minus_vega"])
    assert got == pytest.approx(
        PUBLISHED_AB_MINUS_VEGA[column], abs=TOLERANCE_MAG
    ), column


def test_ab_row_to_vega_and_back_to_an_ab_zeropoint_round_trips(
    tmp_path, record
):
    """
    Given an AB catalog row in SDSS g and a Vega row in 2MASS J,
    When the SED moves them onto its Vega BC system (m - offset) and a
      light curve whose zeropoint is AB moves the Vega value back the way
      MulensInstrument._predicted_mag_on_zp_system does (m + offset),
    Then the reported AB magnitude is recovered to machine precision, and
      the Vega row passes through bit for bit.
    """
    from types import SimpleNamespace

    import yaml

    from exozippy.components.sed.sed import SED
    from exozippy.config import ConfigManager

    rows = [
        {
            "name": "SLOAN/SDSS.g",
            "mag": 12.3456789,
            "err": 0.02,
            "magsys": "AB",
        },
        {
            "name": "2MASS/2MASS.J",
            "mag": 10.9876543,
            "err": 0.02,
            "magsys": "Vega",
        },
    ]
    p = tmp_path / "star.sed"
    p.write_text(
        yaml.safe_dump({"model": "NextGen", "nstars": 1, "filters": rows})
    )
    cm = ConfigManager(
        {
            "star.teffsed": {"initval": 5778.0},
            "star.feh": {"initval": 0.0},
            "star.av": {"initval": 0.0},
        }
    )
    sed = SED({"file": str(p), "model_root": str(DEFAULT_MODEL_ROOT)}, cm)
    star = SimpleNamespace(names=["A"], n_elements=1)
    sed.load_data(SimpleNamespace(star=star))

    off_g = float(record.loc["SDSS_g", "ab_minus_vega"])
    # The SED's forward conversion moved the row by the recorded offset ...
    assert sed.mag[0] != sed.mag_reported[0]
    # ... and the zeropoint side's inverse restores it.
    assert sed.mag[0] + off_g == pytest.approx(
        sed.mag_reported[0], abs=4 * np.spacing(12.0)
    )
    assert sed.mag[1] == sed.mag_reported[1]
