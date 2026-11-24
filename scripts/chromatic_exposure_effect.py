"""How much does a CHROMATIC exposure meter matter for raw redshifts?

A spectrum's measured redshift is the photon-weighted mean of the
instantaneous one over its exposure (Tronsgaard et al. 2019), and in a
cross-dispersed spectrograph each order has its OWN photon arrival
distribution: the atmosphere's extinction, the seeing and the atmospheric
dispersion are all functions of wavelength, so as the airmass and the
seeing change during an exposure the blue and the red photons arrive with
different mean times.  An achromatic (white-light) meter averages the
model with the wrong weights for every order but the "white" one.

This script measures the difference with EXOZIPPy's own redshift model
(``src/exozippy/components/rvinstrument/redshift.py``: ``observer_terms``,
``barycentric_redshift``), for a companion-free star:

    c z_meas(order) = c [ <1 / (1 + z_B)>_order - 1 ]

averaged once with each order's own photon weights and once with the
white-light meter's, on a 1-s grid.  The table it prints is the one in
``docs/raw_redshift_data.md`` ("A chromatic exposure meter") and in
``src/exozippy/components/rvinstrument/rvinstrument.md``.

The flux model (per order, at time t; deliberately simple and stated):

* extinction ``k(lambda) = 0.0095 lambda_um^-4.05 (P/1013.25) + 0.015
  lambda_um^-1.3`` mag per airmass (Rayleigh at the site's pressure plus a
  clean-site aerosol term): 0.33 mag at 380 nm, 0.03 at 900 nm;
* seeing ``FWHM = s0(t) (lambda / 500 nm)^-0.2 X^0.6`` with ``s0`` either
  steady at ``SEEING_STEADY`` or drifting linearly from ``SEEING0`` to
  ``SEEING1`` over the exposure, a Gaussian PSF and a circular fiber of
  radius ``FIBER_R`` arcsec;
* atmospheric dispersion: the image at lambda sits ``(R(lambda) -
  R(GUIDE_NM))`` from the fiber center along the parallactic direction,
  ``R = 206265 (n(lambda) - 1) tan z`` (Edlen 1966 at the site's pressure
  and temperature), times the ADC residual fraction (``ADC_RESIDUAL``; 1 =
  no ADC).  The guider holds GUIDE_NM on the fiber.

The fiber coupling of an offset Gaussian in a circular aperture is the
noncentral chi-square CDF with two degrees of freedom.  The white-light
meter's counts are the sum over 350-950 nm of a 5800 K photon spectrum
times each wavelength's flux model.

    PYTHONPATH=src python scripts/chromatic_exposure_effect.py
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy.stats import ncx2

from exozippy.components.rvinstrument.redshift import (
    barycentric_redshift,
    observer_terms,
)
from exozippy.constants import C_MPS

#: Las Campanas (G-CLEF's site): East longitude, latitude (deg), height (m).
SITE = (-70.6915, -29.0146, 2380.0)
PRESSURE_HPA = 770.0
TEMP_K = 285.0
#: Fiber radius (arcsec) and seeing at 500 nm at zenith (arcsec): steady,
#: or drifting from SEEING0 to SEEING1 over the exposure.
FIBER_R = 0.35
SEEING_STEADY = 0.75
SEEING0, SEEING1 = 0.65, 0.85
GUIDE_NM = 600.0
ORDERS_NM = (380.0, 450.0, 550.0, 700.0, 900.0)
WHITE_NM = np.linspace(350.0, 950.0, 61)
#: The night the exposures are centered in (the Earth's orbital terms are
#: irrelevant at this level; the rotation dominates).
NIGHT_UTC = "2027-04-15T04:00:00"


def extinction_mag(lam_nm):
    lam = lam_nm / 1000.0
    return 0.0095 * lam**-4.05 * (PRESSURE_HPA / 1013.25) + 0.015 * lam**-1.3


def refractivity(lam_nm):
    """n - 1 of air (Edlen 1966), scaled to the site's P and T."""
    s2 = (1000.0 / lam_nm) ** 2
    n15 = 1e-8 * (8342.13 + 2406030.0 / (130.0 - s2) + 15997.0 / (38.9 - s2))
    return n15 * (PRESSURE_HPA / 1013.25) * (288.15 / TEMP_K)


def photon_spectrum(lam_nm, teff=5800.0):
    """Relative photon rate per nm of a blackbody (shape only)."""
    x = 1.4388e7 / (lam_nm * teff)
    return lam_nm**-4 / np.expm1(x)


def flux(lam_nm, zen_rad, frac, adc_residual, drift):
    """The flux model above at wavelength(s) ``lam_nm`` (shape (L, 1)) and
    zenith angle(s) ``zen_rad`` (shape (T,)), ``frac`` in [0, 1] through
    the exposure."""
    X = 1.0 / np.cos(zen_rad)
    trans = 10.0 ** (-0.4 * extinction_mag(lam_nm) * X)
    s0 = SEEING0 + (SEEING1 - SEEING0) * frac if drift else SEEING_STEADY
    sig = s0 * (lam_nm / 500.0) ** -0.2 * X**0.6 / 2.3548
    off = (
        adc_residual
        * 206265.0
        * (refractivity(lam_nm) - refractivity(GUIDE_NM))
        * np.tan(zen_rad)
    )
    couple = ncx2.cdf((FIBER_R / sig) ** 2, 2, (off / sig) ** 2)
    return trans * couple


def exposure_case(dec_deg, ha_hours, minutes, adc_residual, drift):
    """One exposure centered at hour angle ``ha_hours``: per order, the
    chromatic-minus-achromatic c z_meas (m/s) and mean-time shift (s)."""
    lon, lat, h = SITE
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h * u.m)
    t0 = Time(NIGHT_UTC, scale="utc", location=loc)
    lst = t0.sidereal_time("apparent").deg
    ra = (lst - 15.0 * ha_hours) % 360.0
    dt_s = np.arange(-30.0 * minutes, 30.0 * minutes + 0.5, 1.0)
    t = t0 + dt_s * u.s
    t_tdb = t.tdb.jd
    obs = observer_terms(t_tdb, f"{lon},{lat},{h}", "builtin")
    star = (np.radians(ra), np.radians(dec_deg), 0.0, 0.0, 10.0, 0.0)
    inv = 1.0 / (1.0 + barycentric_redshift(t_tdb, obs, star, t_tdb[0]))
    # Zenith angle from the hour angle along the exposure.
    ha = np.radians(ha_hours * 15.0 + dt_s * 15.0 / 3600.0 * 1.0027379)
    d, p = np.radians(dec_deg), np.radians(lat)
    cosz = np.sin(p) * np.sin(d) + np.cos(p) * np.cos(d) * np.cos(ha)
    zen = np.arccos(cosz)
    frac = (dt_s - dt_s[0]) / (dt_s[-1] - dt_s[0])
    white_f = flux(WHITE_NM[:, None], zen, frac, adc_residual, drift)
    white = np.sum(photon_spectrum(WHITE_NM)[:, None] * white_f, axis=0)
    w_white = white / white.sum()
    cz_white = C_MPS * (np.sum(w_white * inv) - 1.0)
    tmean_white = np.sum(w_white * dt_s)
    out = {}
    for lam in ORDERS_NM:
        f = flux(np.array([[lam]]), zen, frac, adc_residual, drift)[0]
        w = f / f.sum()
        out[lam] = (
            C_MPS * (np.sum(w * inv) - 1.0) - cz_white,
            np.sum(w * dt_s) - tmean_white,
        )
    return out, float(np.degrees(zen[0])), float(np.degrees(zen[-1]))


#: The table's cases: (ADC residual fraction, seeing drift).
OPTICS = (
    (0.05, False, "ADC, steady seeing"),
    (0.05, True, "ADC, seeing 0.65->0.85 arcsec"),
    (1.0, True, "no ADC, seeing 0.65->0.85 arcsec"),
)
SHOWN_NM = (380.0, 550.0, 900.0)


def main():
    import argparse

    argparse.ArgumentParser(
        description=(
            "Print the chromatic-minus-achromatic exposure-average table "
            "(docs/raw_redshift_data.md, 'Per-order rows and a chromatic "
            "meter'). No options."
        )
    ).parse_args()
    print(
        "| dec | HA | T | optics | zenith angle | "
        + " | ".join(f"{int(lam)} nm" for lam in SHOWN_NM)
        + " |"
    )
    print("|" + "---|" * (5 + len(SHOWN_NM)))
    for dec in (0.0, -70.0):
        for ha in (0.0, 3.0):
            for minutes in (30.0, 60.0):
                for adc, drift, label in OPTICS:
                    res, z0, z1 = exposure_case(dec, ha, minutes, adc, drift)
                    cells = " | ".join(
                        f"{res[lam][0] * 100:+.1f} cm/s ({res[lam][1]:+.0f} s)"
                        for lam in SHOWN_NM
                    )
                    print(
                        f"| {dec:+.0f} | {ha:+.0f} h | {minutes:.0f} min | "
                        f"{label} | {z0:.0f}-{z1:.0f} deg | {cells} |"
                    )


if __name__ == "__main__":
    main()
