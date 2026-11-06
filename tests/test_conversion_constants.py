"""Known-answer pins on the conversion CONSTANTS every unit boundary uses.

Review 7.16.2: every frame or unit conversion gets a known-answer test
against an external reference AND a round-trip test.  The constants in
``exozippy.constants`` are the multipliers behind most of those
conversions (R_sun <-> AU in the light-travel time, mas/yr <-> km/s in the
Galactic prior, kappa in theta_E, the internal G behind every Kepler
solve), and each is computed from astropy at import time.  That makes the
module self-consistent by construction and blind to a changed astropy
default, a dropped equivalency or a ``.to()`` on the wrong unit -- so here
each one is compared with a number written out from the PUBLISHED source
(IAU 2012 Resolution B2, IAU 2015 Resolution B3, Gould 2000), not with
another astropy expression.

The round-trip half: each constant that has a reciprocal partner elsewhere
in the module (R_sun/AU vs the speed of light in two units, M_sun in
M_earth vs the planet mass conversion) is checked to close to machine
precision.
"""

import numpy as np
import pytest

from exozippy import constants as c
from exozippy.config import unit_conversion

# --- the published numbers, written out (never recomputed via astropy) ---
# IAU 2012 Resolution B2: the astronomical unit, exactly, in m.
AU_M = 149597870700.0
# SI: the speed of light, exactly, in m/s.
C_M_PER_S = 299792458.0
# IAU 2015 Resolution B3 nominal values.
R_SUN_M = 6.957e8
GM_SUN = 1.3271244e20  # m^3 s^-2
GM_EARTH = 3.986004e14
GM_JUP = 1.2668653e17
R_EARTH_EQ_M = 6.3781e6
# The Julian year (IAU), the "yr" of every mas/yr rate (DAYS_PER_YEAR).
JULIAN_YEAR_S = 365.25 * 86400.0
DAY_S = 86400.0


def test_days_per_year_is_the_julian_year():
    """Given the one definition of the year behind every mas/yr rate,
    When compared with the IAU Julian year,
    Then it is exactly 365.25 d."""
    assert c.DAYS_PER_YEAR == 365.25


def test_rsun_to_au_matches_the_iau_nominal_ratio():
    """Given RSUN_TO_AU (orbit.a in R_sun -> AU, the LTT and parallax scale),
    When compared with IAU 2015 B3 R_sun^N over the IAU 2012 au,
    Then they agree to machine precision."""
    assert c.RSUN_TO_AU == pytest.approx(R_SUN_M / AU_M, rel=1e-15)


def test_speed_of_light_in_both_internal_units():
    """Given c in R_sun/day (ltt.py) and in AU/day (the Roemer delay),
    When compared with the SI value and the exact au,
    Then both agree, and their ratio is the AU/R_sun ratio exactly
    (the round trip between the two unit systems)."""
    assert c.C_MPS == C_M_PER_S
    assert c.C_AU_PER_DAY == pytest.approx(C_M_PER_S * DAY_S / AU_M, rel=1e-15)
    assert c.C_LIGHT_RSUN_PER_DAY == pytest.approx(
        C_M_PER_S * DAY_S / R_SUN_M, rel=1e-15
    )
    # Round trip: (R_sun/day -> AU/day) * (AU -> R_sun) returns 1.
    assert (
        c.C_LIGHT_RSUN_PER_DAY * c.RSUN_TO_AU / c.C_AU_PER_DAY
    ) == pytest.approx(1.0, rel=1e-15)
    # The published light time for 1 au: 499.004784 s.
    assert DAY_S / c.C_AU_PER_DAY == pytest.approx(499.004784, abs=1e-6)


def test_internal_g_matches_the_iau_solar_mass_parameter():
    """Given G in the internal (R_sun, M_sun, day) system,
    When compared with GM_sun^N (IAU 2015 B3) in those units,
    Then they agree to 1e-9 (astropy's M_sun is GM_sun^N / G_CODATA, so
    G*M_sun is GM_sun^N up to rounding), and Kepler's third law returns
    the Gaussian year (365.2568983 d) for 1 au around 1 M_sun."""
    expected = GM_SUN * DAY_S**2 / R_SUN_M**3
    assert c.G == pytest.approx(expected, rel=1e-9)
    # Kepler's third law in the internal units: one year at 1 au around
    # one solar mass.  a^3 = G M P^2 / (4 pi^2), a in R_sun, P in days.
    a_rsun = AU_M / R_SUN_M
    period_days = 2.0 * np.pi * np.sqrt(a_rsun**3 / c.G)
    assert period_days == pytest.approx(365.2568983, rel=1e-6)  # Gaussian year


def test_kappa_matches_gould_2000():
    """Given KAPPA = 4G/(c^2 au) in mas/M_sun (theta_E^2 = kappa M pi_rel),
    When compared with Gould (2000, ApJ 542, 785)'s 8.144 mas/M_sun and
    with the closed form from the IAU constants,
    Then it matches both."""
    rad_to_mas = 180.0 / np.pi * 3600.0e3
    closed_form = 4.0 * GM_SUN / (C_M_PER_S**2 * AU_M) * rad_to_mas
    assert c.KAPPA == pytest.approx(closed_form, rel=1e-9)
    assert c.KAPPA == pytest.approx(8.144, abs=5e-4)


def test_proper_motion_velocity_factor_is_4_74047():
    """Given K_VEL_CONVERSION, the km/s per (mas/yr * kpc) factor,
    When compared with the textbook 4.74047 and the closed form au/yr,
    Then both agree."""
    assert c.K_VEL_CONVERSION == pytest.approx(
        AU_M / JULIAN_YEAR_S / 1e3, rel=1e-15
    )
    assert c.K_VEL_CONVERSION == pytest.approx(4.74047, abs=1e-5)


def test_solar_unit_ratios_are_the_iau_nominal_ones():
    """Given the planet-side unit ratios (internal solar -> Earth units)
    and astropy's mass units as config.unit_conversion uses them,
    When compared with the IAU 2015 B3 nominal GM and radius ratios,
    Then they agree, and Earth -> Sun -> Earth closes."""
    assert c.MSUN_TO_MEARTH == pytest.approx(GM_SUN / GM_EARTH, rel=1e-12)
    assert c.RSUN_TO_REARTH == pytest.approx(R_SUN_M / R_EARTH_EQ_M, rel=1e-15)
    assert unit_conversion("solMass", "jupiterMass", "t") == pytest.approx(
        GM_SUN / GM_JUP, rel=1e-12
    )
    # Round trip through config's user -> internal multiplier.
    there = unit_conversion("earthMass", "solMass", "t")
    back = unit_conversion("solMass", "earthMass", "t")
    assert there * back == pytest.approx(1.0, rel=1e-15)


def test_solar_luminosity_and_gravity_are_the_iau_nominal_sun():
    """
    Given the luminosity (R_sun, K -> L_sun) and log g (dex M_sun, R_sun ->
      dex cgs) conversions,
    When the IAU 2015 B3 nominal Sun is fed in (R = 1, Teff = 5772 K,
      logmass = 0),
    Then L = 1 L_sun^N (3.828e26 W, defined to match sigma T^4 4 pi R^2 to
      the quoted digits) and log g = log10(GM_sun^N / R_sun^N^2) = 4.43807
      in cgs; and inverting each closes (round trip).
    """
    import pytensor.tensor as pt

    from exozippy.components.star.physics import (
        calc_logg_from_logmass,
        calc_luminosity,
    )

    r, teff, logm = pt.dscalar("r"), pt.dscalar("teff"), pt.dscalar("logm")
    lum = float(calc_luminosity(r, teff).eval({r: 1.0, teff: 5772.0}))
    # IAU 2015 B3: L_sun^N = 3.828e26 W, and Teff_sun^N = 5772 K is the
    # value that reproduces it with R_sun^N; quoted to 4 digits.
    assert lum == pytest.approx(1.0, abs=5e-4)
    sigma_sb = 5.670374419e-8  # CODATA 2018, W m^-2 K^-4 (exact)
    l_watts = 4.0 * np.pi * R_SUN_M**2 * sigma_sb * 5772.0**4
    # 1e-9: astropy derives sigma_sb from h, k, c (3e-11 apart).
    assert lum == pytest.approx(l_watts / 3.828e26, rel=1e-9)
    # Round trip: the Teff that gives L at R = 1.
    teff_back = (lum / c.LUM_CONST) ** 0.25
    assert teff_back == pytest.approx(5772.0, rel=1e-14)

    logg = float(calc_logg_from_logmass(logm, r).eval({logm: 0.0, r: 1.0}))
    assert logg == pytest.approx(
        np.log10(GM_SUN * 1e6 / (R_SUN_M * 1e2) ** 2), abs=1e-12
    )
    assert logg == pytest.approx(4.43807, abs=1e-5)


def test_rsun_per_day_to_mps_is_the_closed_form():
    """Given SOLRAD_PER_DAY_TO_MPS (internal velocities -> m/s for RVs),
    When compared with R_sun^N / 86400 s,
    Then it is exact; and the Earth's mean orbital speed comes out at the
    familiar 29.78 km/s through it."""
    assert c.SOLRAD_PER_DAY_TO_MPS == pytest.approx(R_SUN_M / DAY_S, rel=1e-15)
    v_earth_rsun_per_day = 2.0 * np.pi * (AU_M / R_SUN_M) / 365.2568983
    v_kms = v_earth_rsun_per_day * c.SOLRAD_PER_DAY_TO_MPS / 1e3
    assert v_kms == pytest.approx(29.78, abs=0.01)
