"""Gaia epoch astrometry read as the archive ships it (`data_format:
gaia_epoch`), checked against DPAC's own numbers on the real DR4
pre-release (examples/GaiaBH3/GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml,
ESA/Gaia/DPAC, June 2026: 12 sources, CCD-level along-scan astrometry).

Pinned here:

1. The reader keeps exactly the CCD observations AGIS used along scan --
   558 for Gaia BH3, the count DPAC's own BH3 notebook (esa/gaia-bhthree,
   branch gaia-dr4-prerelease) prints after its filter -- and builds the
   barycentric TCB instant from the documented ISO origin.
2. The model's along-scan parallax factors (the Gaia spacecraft ephemeris
   at the reception instant, from BJD_TDB times) agree with DPAC's
   parallax_factor_al on all twelve sources.
3. The five-parameter solution of Gaia BH3 formed from what the PREPARED
   SYSTEM holds (its times, abscissae, errors, scan angles and its own
   parallax factors) reproduces the one DPAC's notebook fits to the same
   file -- an external number, so the test carries no constant of its own.
4. Configuration errors raise, naming the fix.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time, TimeDelta

from exozippy.components.astrometryinstrument.astrometryinstrument import (
    read_gaia_epoch_astrometry,
)
from exozippy.constants import DAYS_PER_YEAR
from exozippy.ephemeris import observer_position_at_reception
from exozippy.skyframe import parallax_factors
from exozippy.system import System

FILE = (
    Path(__file__).parent.parent
    / "examples"
    / "GaiaBH3"
    / "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"
)
GAIA4 = 1457486023639239296
BH3 = 4318465066420528000
# J2017.5 TCB, the DR4 reference epoch, as BJD_TDB
DR4_EPOCH = Time(2017.5, format="jyear", scale="tcb").tdb.jd

# DPAC's single-star fit of Gaia BH3 to this file (esa/gaia-bhthree,
# Gaia-DR4-prerelease_BH3_fit_astrometric_orbit.ipynb, "Fit the standard
# astrometric model"): value, error, in mas and mas/yr at J2017.5.
DPAC_BH3_FIVE = {
    "ra": (0.71001, 0.00506),
    "dec": (0.26549, 0.00452),
    "parallax": (0.98329, 0.00623),
    "mura": (-30.94602, 0.00362),
    "mudec": (-148.54974, 0.00281),
}


def _star_config(source_id, **extra):
    return {
        "name": "gaiaepoch",
        "star": [{"name": "A", "mist": False}],
        "astrometryinstrument": [
            {
                "name": "G",
                "file": str(FILE),
                "mode": "gaia",
                "data_format": "gaia_epoch",
                "source_id": source_id,
                "epoch": DR4_EPOCH,
                **extra,
            }
        ],
    }


def _star_params(source_id):
    _, meta = read_gaia_epoch_astrometry(FILE, source_id)
    return {
        "star.A.ra": {"initval": meta["ra0"]},
        "star.A.dec": {"initval": meta["dec0"]},
        "star.A.distance": {"initval": 500.0},
    }


# ----------------------------------------------------------------------
# 1. The reader
# ----------------------------------------------------------------------
def test_reader_keeps_the_ccds_agis_used_and_the_tcb_instant():
    """
    Given the DR4 pre-release file,
    When Gaia BH3 and Gaia-4 are read,
    Then BH3 keeps DPAC's 558 AGIS-used along-scan CCD observations and
      Gaia-4 its 824; a multi-source file without source_id, and a
      source_id not in it, raise; and the first Gaia-4 time is the
      barycentric TCB instant built from the ISO origin
      2010-01-01T00:00:00 TCB plus obs_time_tcb + obs_time_bary_corr.
    """
    bh3, _ = read_gaia_epoch_astrometry(FILE, BH3)
    assert len(bh3) == 558
    df, meta = read_gaia_epoch_astrometry(FILE, GAIA4)
    assert len(df) == 824 and meta["n_transits"] == 115

    with pytest.raises(ValueError, match="name one with `source_id:`"):
        read_gaia_epoch_astrometry(FILE)
    with pytest.raises(ValueError, match="is not in the file"):
        read_gaia_epoch_astrometry(FILE, 42)

    table = Table.read(FILE, format="votable")
    rows = table[table["source_id"] == GAIA4]
    for r in rows:
        used = np.ma.filled(np.ma.asarray(r["used_by_agis_al"]), False)
        if used.any():
            k = int(np.flatnonzero(used)[0])
            ns = int(r["obs_time_tcb"][k]) + float(r["obs_time_bary_corr"])
            break
    origin = Time("2010-01-01T00:00:00", format="isot", scale="tcb")
    expect = (origin + TimeDelta(ns * 1e-9, format="sec")).jd
    # the file's order is kept, so its first used CCD is the first record
    assert abs(df[0].iloc[0] - expect) * 86400.0 < 1e-4  # < 0.1 ms


# ----------------------------------------------------------------------
# 2. Parallax factors
# ----------------------------------------------------------------------
def test_parallax_factors_agree_with_dpac_on_every_source():
    """
    Given each of the twelve pre-release sources, its times converted to
      BJD_TDB the way the reader's time spec converts them,
    When the model's along-scan parallax factor is formed from the Gaia
      spacecraft ephemeris at the reception instant,
    Then it matches DPAC's parallax_factor_al to 1e-4 (measured: <= 1.6e-5).
    """
    ids = np.unique(
        np.asarray(Table.read(FILE, format="votable")["source_id"])
    )
    assert len(ids) == 12
    for sid in ids:
        df, meta = read_gaia_epoch_astrometry(FILE, int(sid))
        t = Time(df[0].values, format="jd", scale="tcb").tdb.jd
        ra, dec = np.radians(meta["ra0"]), np.radians(meta["dec0"])
        xyz = observer_position_at_reception(t, ra, dec, "gaia")
        P_E, P_N = parallax_factors(xyz, ra, dec)
        psi = np.radians(df[3].values)
        ours = P_E * np.sin(psi) + P_N * np.cos(psi)
        assert np.max(np.abs(ours - df[4].values)) < 1e-4, sid


# ----------------------------------------------------------------------
# 3. The prepared system reproduces DPAC's solution
# ----------------------------------------------------------------------
def test_prepared_bh3_reproduces_the_dpac_five_parameter_fit():
    """
    Given a single-star system reading Gaia BH3 with data_format:
      gaia_epoch,
    When the system is prepared and the five-parameter weighted
      least-squares solution is formed from the dataset it holds (BJD_TDB
      times, w, errors, scan angles and the MODEL's parallax factors),
    Then every parameter and error matches DPAC's notebook fit to 0.02 of
      its error, the time spec is TCB -> BJD_TDB, and the tangent point is
      the file's reference point.
    """
    system = System(_star_config(BH3), user_params=_star_params(BH3))
    system.prepare()
    inst = system.astrometryinstrument
    d = inst.datasets[0]
    _, meta = read_gaia_epoch_astrometry(FILE, BH3)
    assert inst.time_specs[0]["scale"] == "tcb"
    assert inst.time_specs[0]["frame"] == "bjd"
    assert d["ra_ref"] == pytest.approx(np.radians(meta["ra0"]), abs=0)
    assert d["dec_ref"] == pytest.approx(np.radians(meta["dec0"]), abs=0)

    s, c = d["sin_psi"], d["cos_psi"]
    f_plx = d["P_E"] * s + d["P_N"] * c
    dt = (d["time"] - DR4_EPOCH) / DAYS_PER_YEAR
    A = np.column_stack([s, c, f_plx, dt * s, dt * c])
    W = 1.0 / d["err"] ** 2
    normal = A.T @ (W[:, None] * A)
    p = np.linalg.solve(normal, A.T @ (W * d["w"]))
    e = np.sqrt(np.diag(np.linalg.inv(normal)))
    for k, (name, (v, sig)) in enumerate(DPAC_BH3_FIVE.items()):
        assert abs(p[k] - v) < 0.02 * sig, (name, p[k], v)
        assert abs(e[k] - sig) < 0.02 * sig, (name, e[k], sig)


# ----------------------------------------------------------------------
# 4. Configuration errors
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "extra, match",
    [
        ({"mode": "abs"}, "mode: gaia"),
        ({"time_scale": "tcb"}, "would describe them a second time"),
        ({"time_offset": 2455197.5}, "would describe them a second time"),
        ({"columns": {"time": 0}}, "would describe them a second time"),
        ({"observer_location": "earth"}, "wrong observer"),
        ({"data_format": "votable"}, "data_format must be one of"),
        ({"source_id": "Gaia-4"}, "integer Gaia source_id"),
    ],
)
def test_gaia_epoch_misconfigurations_raise(extra, match):
    cfg = _star_config(GAIA4)
    cfg["astrometryinstrument"][0].update(extra)
    with pytest.raises(ValueError, match=match):
        System(cfg, user_params={})


def test_source_id_on_a_columns_file_raises(tmp_path):
    cfg = {
        "name": "x",
        "star": [{"name": "A"}],
        "astrometryinstrument": [
            {
                "name": "a",
                "file": str(tmp_path / "a.dat"),
                "mode": "gaia",
                "source_id": GAIA4,
            }
        ],
    }
    with pytest.raises(ValueError, match="source_id selects a source"):
        System(cfg, user_params={})
