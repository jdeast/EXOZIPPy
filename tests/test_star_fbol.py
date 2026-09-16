"""star.fbol is a bolometric flux in erg s-1 cm-2, and that is pinned NUMERICALLY.

Review 1.8.1: `FBOL_CONST` was `1/(4 pi (pc/R_sun)^2)`, which is
dimensionless -- so `calc_fbol(L[solLum], d[pc])` returned solLum/solRad^2
while star/defaults.yaml declared the result `erg s-1 cm-2` with no
conversion anywhere.  Every published F_Bol was low by exactly
L_sun/R_sun^2 = 7.909e11.

The bug survived because it is invisible to every check that does not put a
number against physics: the constant cancels out of the only likelihood term
that reads fbol (sed.py's fbolsed floor), the unit strings agree with each
other, and a self-consistency test comparing calc_fbol to itself passes
under any constant at all.  So these tests are deliberately ANCHORED to an
externally known flux -- the solar constant -- rather than to the code.
Review 7.8.1(a) asked for exactly this pin.
"""

import pathlib

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytensor.tensor as pt
import yaml

from exozippy.components.star import physics
from exozippy.constants import FBOL_CONST

STAR_DEFAULTS = (
    pathlib.Path(__file__).parent
    / ".."
    / "src"
    / "exozippy"
    / "components"
    / "star"
    / "defaults.yaml"
)

# The Sun seen from 10 pc, in erg s-1 cm-2, computed from astropy constants
# INDEPENDENTLY of exozippy: L_sun / (4 pi (10 pc)^2).
SUN_AT_10PC_CGS = float(
    (const.L_sun / (4.0 * np.pi * (10.0 * u.pc) ** 2))
    .to(u.erg / u.s / u.cm**2)
    .value
)

# The factor the pre-2026-08 constant was wrong by, restated from its own
# definition so a regression names the mistake rather than just a number.
LSUN_OVER_RSUN2 = float((const.L_sun / const.R_sun**2).cgs.value)


def _fbol(luminosity_sollum, distance_pc):
    """calc_fbol evaluated through pytensor, as the model calls it.

    pt.dscalar, never a bare Python float: pytensor autocasts a literal to
    the smallest dtype that represents it, and the model always feeds
    float64.
    """
    lum = pt.dscalar("luminosity")
    dist = pt.dscalar("distance")
    out = physics.calc_fbol(lum, dist)
    return float(out.eval({lum: luminosity_sollum, dist: distance_pc}))


def test_sun_at_10pc_is_the_known_bolometric_flux():
    """Given the Sun (1 solLum) at 10 pc, when calc_fbol runs, then it
    returns 3.20e-7 erg s-1 cm-2 -- the flux astropy's own constants give."""
    # Arrange / Act
    fbol = _fbol(1.0, 10.0)

    # Assert -- against the independently computed value, and against the
    # literal quoted in review 7.8.1(a) so a plausible-looking constant
    # cannot drift past both.
    assert np.isclose(fbol, SUN_AT_10PC_CGS, rtol=1e-12)
    assert np.isclose(fbol, 3.1993e-07, rtol=1e-4)


def test_sun_at_1au_is_the_solar_constant():
    """Given the Sun at 1 au, when calc_fbol runs, then it returns the solar
    constant, ~1.361e6 erg s-1 cm-2 (1361 W m-2).

    A second, wholly independent anchor: the solar constant is a measured
    quantity, not a definition, so agreeing with it is a statement about the
    physics rather than about astropy's bookkeeping.
    """
    # Arrange
    au_in_pc = float((1.0 * u.au).to(u.pc).value)

    # Act
    fbol = _fbol(1.0, au_in_pc)

    # Assert -- 1361 W m-2 = 1.361e6 erg s-1 cm-2, to the 0.1% the nominal
    # L_sun is quoted to.
    assert np.isclose(fbol, 1.361e6, rtol=2e-3)


def test_fbol_scales_as_luminosity_over_distance_squared():
    """Given a star twice as luminous at half the distance, when calc_fbol
    runs, then the flux is 8x -- the inverse-square law itself."""
    # Arrange / Act
    base = _fbol(1.0, 10.0)
    scaled = _fbol(2.0, 5.0)

    # Assert
    assert np.isclose(scaled, 8.0 * base, rtol=1e-12)


def test_fbol_const_is_not_the_dimensionless_solrad_form():
    """Given the constant, when compared with the pre-fix expression, then it
    differs by exactly L_sun/R_sun^2 -- the whole content of review 1.8.1.

    Stated as a ratio rather than as a forbidden literal so it keeps meaning
    if astropy's nominal constants are ever revised.
    """
    # Arrange -- the expression that shipped until 2026-08.
    old = 1.0 / (4.0 * np.pi * float((const.pc / const.R_sun).value) ** 2.0)

    # Act
    ratio = FBOL_CONST / old

    # Assert
    assert np.isclose(ratio, LSUN_OVER_RSUN2, rtol=1e-10)
    assert np.isclose(LSUN_OVER_RSUN2, 7.909e11, rtol=1e-3)


def test_fbol_declares_cgs_and_needs_no_conversion():
    """Given star/defaults.yaml, when fbol/fbolsed are read, then both
    declare unit == internal_unit == erg s-1 cm-2.

    That equality is why the constant has to carry the whole conversion:
    Parameter applies a factor of exactly 1, so a wrong constant reaches the
    table unmodified.  If someone ever fixes this the other way -- by giving
    fbol a different user unit -- this test says the constant's contract
    changed with it.
    """
    # Arrange
    with open(STAR_DEFAULTS) as handle:
        defaults = yaml.safe_load(handle)["star"]

    # Act / Assert
    for name in ("fbol", "fbolsed"):
        entry = defaults[name]
        assert entry["unit"] == "erg s-1 cm-2"
        assert entry["internal_unit"] == entry["unit"]
        assert entry["expressions"]["default"]["func_name"] == "calc_fbol"
