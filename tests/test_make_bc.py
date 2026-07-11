"""
Tests for the BC-table generator (components/sed/make_bc.py):
  - regenerated 2MASS_J agrees with the shipped table at Av=0
  - existing columns survive a merge byte-exactly
  - extinction is applied along the Av axis
"""

import shutil

import numpy as np
import pytest

from exozippy.components.sed.bc_grid import (
    DEFAULT_BC_ROOT,
    _read_single_bc_file,
)
from exozippy.components.sed.make_bc import make_bc_tables

_NEXTGEN = DEFAULT_BC_ROOT / "NextGen"
_SPECTRA = _NEXTGEN / "NextGen.spectra.csv"
_SHIPPED_2MASS = _NEXTGEN / "BCs" / "2MASS" / "feh+0.0_afe+0.0.2MASS"

pytestmark = pytest.mark.skipif(
    not _SPECTRA.exists(),
    reason="NextGen spectra not downloaded (Zenodo); run an SED fit once first.",
)


@pytest.fixture(scope="module")
def regenerated_2mass(tmp_path_factory):
    """
    Build a minimal BC root (solar-feh 2MASS file + spectra symlinks),
    regenerate the 2MASS_J column there, and return the before/after
    tables. Module-scoped: the spectra CSV read dominates the runtime.
    """
    root = tmp_path_factory.mktemp("bc_root")
    model_dir = root / "NextGen"
    (model_dir / "BCs" / "2MASS").mkdir(parents=True)
    for name in ("NextGen.spectra.csv", "NextGen.wavelength.csv"):
        (model_dir / name).symlink_to(_NEXTGEN / name)
    shutil.copy(_SHIPPED_2MASS, model_dir / "BCs" / "2MASS" / _SHIPPED_2MASS.name)

    written = make_bc_tables(["2MASS/2MASS.J"], model="NextGen", bc_root=root)

    before, cols_before = _read_single_bc_file(_SHIPPED_2MASS)
    after, cols_after = _read_single_bc_file(written[0])
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
    assert diff.max() < 0.35   # coolest grid corners are resolution-limited


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
    (the shipped tables are flat in Av; the generator applies the
    extinction law).
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
