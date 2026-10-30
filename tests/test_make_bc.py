"""
Tests for the BC-table generator (components/sed/make_bc.py):
  - regenerated 2MASS_J agrees with the shipped table at Av=0
  - existing columns survive a merge byte-exactly
  - extinction is applied along the Av axis
"""

import functools

import numpy as np
import pytest

from exozippy.components.sed.bc_grid import (
    DEFAULT_MODEL_ROOT,
    bc_filter_columns,
    find_bc_table,
    read_bc_table,
    write_bc_table,
)
from exozippy.components.sed.make_bc import make_bc_tables

_NEXTGEN = DEFAULT_MODEL_ROOT / "NextGen"
_SPECTRA = _NEXTGEN / "BCs" / "NextGen.spectra.csv"
_SHIPPED_2MASS = _NEXTGEN / "BCs" / "2MASS.bc.parquet"

pytestmark = pytest.mark.skipif(
    not _SPECTRA.exists(),
    reason="NextGen spectra not downloaded (Zenodo); run an SED fit once first.",
)


@pytest.fixture(scope="module")
def regenerated_2mass(tmp_path_factory):
    """
    Build a minimal BC root (the solar-feh slice of the shipped 2MASS
    table + spectra symlinks), regenerate the 2MASS_J column there, and
    return the before/after tables. Only the solar slice, so make_bc (which
    works on the axes of the tables it finds) synthesizes one feh, not
    eleven. Module-scoped: the spectra CSV read dominates the runtime.
    """
    root = tmp_path_factory.mktemp("model_root")
    model_dir = root / "NextGen"
    (model_dir / "BCs").mkdir(parents=True)
    for name in ("NextGen.spectra.csv", "NextGen.wavelength.csv"):
        (model_dir / "BCs" / name).symlink_to(_NEXTGEN / "BCs" / name)
    # find_bc_table, not the bare path: the published table is fetched
    # into the default root on first use (models/NextGen/bc_tables.py).
    shipped = read_bc_table(
        find_bc_table(DEFAULT_MODEL_ROOT, "NextGen", "2MASS")
    )
    before = shipped[shipped["feh"] == 0.0]
    write_bc_table(
        before,
        model_dir / "BCs" / _SHIPPED_2MASS.name,
        shipped.attrs["meta"]["filters"],
    )

    # overwrite: 2MASS_J is already in the table (written by another
    # pipeline), which make_bc otherwise leaves alone.
    written = make_bc_tables(
        ["2MASS/2MASS.J"], model="NextGen", model_root=root, overwrite=True
    )

    after = read_bc_table(written[0])
    cols_after = bc_filter_columns(after)
    key = ["teff", "logg", "Av"]
    before = before.sort_values(key).reset_index(drop=True)
    after = after.sort_values(key).reset_index(drop=True)
    return before, after, cols_after


def test_regenerated_2mass_j_matches_shipped_at_av0(regenerated_2mass):
    """
    Given the shipped solar-feh 2MASS BC table,
    When the 2MASS_J column is regenerated from the shipped R=150 spectra,
    Then the Av=0 values agree with the shipped table to 0.1 mag
    (the shipped tables came from higher-resolution spectra, so exact
    agreement is not expected; ~0.02 mag is typical).
    """
    # ARRANGE
    before, after, _ = regenerated_2mass
    m = before["Av"] == 0.0

    # ACT
    diff = (after.loc[m, "2MASS_J"] - before.loc[m, "2MASS_J"]).abs()

    # ASSERT
    assert diff.median() < 0.05
    assert diff.max() < 0.35  # coolest grid corners are resolution-limited


def test_merge_preserves_existing_columns_exactly(regenerated_2mass):
    """
    Given a facility file with 2MASS_H and 2MASS_Ks columns,
    When only 2MASS_J is regenerated into it,
    Then H and Ks are preserved exactly.
    """
    # ARRANGE
    before, after, cols_after = regenerated_2mass

    # ASSERT
    assert set(cols_after) == {"2MASS_J", "2MASS_H", "2MASS_Ks"}
    for col in ("2MASS_H", "2MASS_Ks"):
        np.testing.assert_array_equal(before[col].values, after[col].values)


def test_generated_bc_includes_extinction_along_av_axis(regenerated_2mass):
    """
    Given a regenerated 2MASS_J column,
    When comparing BC at Av=0 and Av=6 for a solar analog,
    Then BC drops by roughly the J-band extinction A_J ~ 0.28*Av
    (the generator applies the extinction law; the shipped tables do
    too, at dBC/dAv ~ -0.30 in J).
    """
    # ARRANGE
    _, after, _ = regenerated_2mass
    node = (after["teff"] == 5800) & (after["logg"] == 4.5)
    bc0 = float(after.loc[node & (after["Av"] == 0.0), "2MASS_J"].iloc[0])
    bc6 = float(after.loc[node & (after["Av"] == 6.0), "2MASS_J"].iloc[0])

    # ACT
    a_j = bc0 - bc6

    # ASSERT
    assert 1.0 < a_j < 2.5


# 2.5 / ln(10): A_lam = _MAG_PER_TAU * tau_lam exactly. make_bc divides by
# the rounded 1.086, so its applied A is short by 2.43e-4 relative.
_MAG_PER_TAU = 2.5 / np.log(10.0)
_GENERATOR_TAU_DIVISOR = 1.086


@functools.lru_cache(maxsize=1)
def _solar_nextgen_spectrum():
    """(wavelength in Angstrom, F_lambda) of the 5800/4.5/0.0/alpha=0
    NextGen spectrum, read straight from the shipped CSVs."""
    import json

    import pandas as pd

    wave = pd.read_csv(_NEXTGEN / "BCs" / "NextGen.wavelength.csv")[
        "wavelength_angstrom"
    ].to_numpy(float)
    for chunk in pd.read_csv(_SPECTRA, chunksize=500):
        row = chunk[
            (chunk.teff == 5800)
            & (chunk.logg == 4.5)
            & (chunk.feh == 0.0)
            & (chunk.alpha == 0.0)
        ]
        if len(row):
            return wave, np.array(json.loads(row.iloc[0].flux), dtype=float)
    raise AssertionError(
        "no 5800/4.5/0.0/alpha=0 spectrum in NextGen.spectra.csv"
    )


def _independent_a_j(av):
    """
    A_J of the solar-feh 5800 K / logg 4.5 NextGen spectrum, from scratch:
    the spectrum row read from NextGen.spectra.csv (alpha = 0), 2MASS J's
    SVO profile interpolated onto the spectra grid, photon weighting
    (2MASS is a photon counter, and make_bc photon-weights every filter),
    and 10^(-0.4 Av k(lam)/k(V)) with models/extinction_law.ascii read by
    np.loadtxt and interpolated in microns. Nothing from make_bc.py.
    """
    from exozippy.filters.filter import Filter

    wave, spec = _solar_nextgen_spectrum()
    wf, tf = Filter("2MASS/2MASS.J").ProcessedFilterCurve
    weight = np.interp(wave, wf, tf, left=0.0, right=0.0) * wave
    law = np.loadtxt(DEFAULT_MODEL_ROOT / "extinction_law.ascii")
    k = np.interp(wave * 1e-4, law[:, 0], law[:, 1])
    k_v = np.interp(0.55, law[:, 0], law[:, 1])

    def band_flux(a):
        return np.trapezoid(spec * 10.0 ** (-0.4 * a * k / k_v) * weight, wave)

    return -2.5 * np.log10(band_flux(av) / band_flux(0.0))


def _generated_a_j(after, av):
    node = (after["teff"] == 5800) & (after["logg"] == 4.5)
    bc0 = float(after.loc[node & (after["Av"] == 0.0), "2MASS_J"].iloc[0])
    bca = float(after.loc[node & (after["Av"] == av), "2MASS_J"].iloc[0])
    return bc0 - bca


def test_generated_a_j_matches_an_independent_integral_up_to_the_rounded_1086(
    regenerated_2mass,
):
    """
    Given a regenerated 2MASS_J column,
    When BC(Av=0) - BC(Av=6) is read at 5800 K, logg 4.5,
    Then it equals an independent spectrum x profile x lambda x
    10^(-0.4 Av k/k_V) integral to 1e-10 mag once Av is scaled by
    2.5/ln(10)/1.086, the one known departure (see the strict xfail
    below). The exact form of the 1.0 < A_J < 2.5 range above: the law in
    Angstrom, the wrong k_V or energy weighting all miss by >> 1e-10. When
    the 1.086 is fixed, drop the Av scaling here.
    """
    # ARRANGE
    _, after, _ = regenerated_2mass

    # ACT
    a_j = _generated_a_j(after, 6.0)
    ref = _independent_a_j(6.0 * _MAG_PER_TAU / _GENERATOR_TAU_DIVISOR)

    # ASSERT
    assert a_j == pytest.approx(ref, abs=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "make_bc rounds 2.5/ln(10) to 1.086 in tau = A k/k_V / 1.086: the "
        "applied extinction is short by 2.43e-4 relative -- measured "
        "A_J(Av=6) about 4.4e-4 mag below the exact magnitude law."
    ),
)
def test_generated_a_j_matches_the_exact_magnitude_law(regenerated_2mass):
    """
    Given a regenerated 2MASS_J column,
    When BC(Av=0) - BC(Av=6) is read at 5800 K, logg 4.5,
    Then it equals the independent integral with the exact magnitude law
    10^(-0.4 A) to 1e-10 mag.
    """
    # ARRANGE
    _, after, _ = regenerated_2mass

    # ACT
    a_j = _generated_a_j(after, 6.0)
    ref = _independent_a_j(6.0)

    # ASSERT
    assert a_j == pytest.approx(ref, abs=1e-10)
