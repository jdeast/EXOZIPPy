import numpy as np
import pytensor.tensor as pt

from ...constants import (
    C_LIGHT_RSUN_PER_DAY,
    C_MPS,
    DENSITY_CONST,
    EARTH_INSOLATION_CGS,
    KEPLER_CONST,
    LOGG_CONST,
    SIGMA_SB_CGS,
    SOLRAD_PER_DAY_TO_MPS,
    TWOPI,
    G,
)
from ...physics_registry import register_physics

# Sphere geometry is not planet-specific, and PHYSICS_REGISTRY is a flat
# namespace keyed by function name -- so "calc_density" must have exactly one
# owner. planet/defaults.yaml still resolves it by name through the registry;
# this import just keeps the name available here too.
from ..star.physics import calc_density  # noqa: F401


@register_physics
def calc_logg_from_mass(mass, radius):
    """
    Calculates surface gravity (logg) from mass and radius.
    mass: planet mass, in solar masses
    radius: solar radii
    returns: cgs (log10)
    Note: this odd form of logg is designed to simplify the symbolic math and chain rule derivatives
    """
    return LOGG_CONST + pt.log10(mass) - 2.0 * pt.log10(radius)


@register_physics
def calc_mass_from_log_q(log_q, star_mass):
    """Planet mass from the sampled log10 mass ratio and the host mass.

    log_q: log10(m_planet / m_host), dimensionless
    star_mass: solar masses
    returns: solar masses (always positive -- see
    Planet._resolve_mass_parameterization for when this coordinate applies)
    """
    return pt.power(10.0, log_q) * star_mass


@register_physics
def calc_m_total(planet_mass, star_mass):
    return pt.maximum(star_mass + planet_mass, 1e-9)


@register_physics
# The parameter this feeds is named `a` (user unit AU); the FUNCTION keeps
# the arsun name because it computes the internal value, which is in solRad
# -- the one layer where the unit is fixed by convention rather than config.
def calc_arsun(m_total, period):
    m13 = pt.power(m_total, 1.0 / 3.0)
    p2 = pt.sqr(period)
    p23 = pt.power(p2, 1.0 / 3.0)
    return KEPLER_CONST * m13 * p23


@register_physics
def calc_arstar(a, rstar):
    return a / rstar


@register_physics
def calc_p(radius, star_radius):
    return radius / star_radius


@register_physics
def calc_msini(mass, sini):
    """Minimum mass, M_P sin i (EXOFASTv2 derivepars.pro's msini).

    Internal units in and out (solMass), like `mass`; defaults.yaml converts
    to the user's unit.  Signed like `mass`: a linear-mode planet may cross
    zero, and so does its minimum mass.
    """
    return mass * sini


@register_physics
# Not `calc_q`: PHYSICS_REGISTRY is a flat namespace and mulensing owns that
# name for the lens mass ratio (see components.md, "Physics registry").
def calc_planet_mass_ratio(mass, star_mass):
    """q = M_P / M_*, in linear form for EVERY planet (derivepars.pro's q).

    The log_q mass mode samples log10 of this and reports that coordinate,
    but a linear-mode planet has no log_q at all, so without this row the
    mass ratio of every RV- or astrometry-measured planet went unreported.
    """
    return mass / star_mass


@register_physics
def calc_K(mass, m_total, ecc, a, sini, period):
    ecc_factor = 1.0 / pt.sqrt(1.0 - pt.sqr(ecc))
    return 2.0 * np.pi * (a * sini * (mass / m_total) * ecc_factor / period)


# ----------------------------------------------------------------------
# Transit and occultation durations (review 8.8.7).
#
# Winn (2010, "Transits and Occultations", arXiv:1001.2010) eqs 14-16, with
# the eccentricity correction of his eq 16:
#
#   T_14 = (P/pi) arcsin[ sqrt((1+p)^2 - b^2) / (a/R* sin i) ] * K_e
#   T_23 = (P/pi) arcsin[ sqrt((1-p)^2 - b^2) / (a/R* sin i) ] * K_e
#
# with K_e = sqrt(1-e^2)/(1 + e sin omega) at the PRIMARY transit and
# sqrt(1-e^2)/(1 - e sin omega) at the occultation, because the planet sits
# at r = a(1-e^2)/(1 +/- e sin omega) at the two conjunctions.  From those
# two, EXOFASTv2's derivepars.pro convention:
#
#   T_FWHM = (T_14 + T_23)/2      tau = (T_14 - T_23)/2
#
# i.e. the half-depth duration and the ingress/egress duration.
#
# These moved here from `transit.py`, which built them inline as bare
# Deterministics.  They are geometry, not photometry: every input is a
# planet or orbit parameter, nothing about a light curve enters, and an
# RV-only fit has just as well-defined a transit duration -- which is the
# whole point of review 8.8.7's inference path, where a published duration
# CONSTRAINS e and omega through a Gaussian on the derived parameter.  Having
# them on `planet` also gives them table rows, LaTeX macros, units and a
# user-settable prior, none of which a hand-built Deterministic has.
#
# Not ported: EXOFASTv2's `tt`.  The item's list names it, but its definition
# could not be checked from this tree, and a duration reported under a name
# whose meaning we guessed is worse than one absent -- the same discipline
# orbit.md applies to EXOFASTv2's V_c/V_e Jacobian sign.
#
# The floor below is `transit.py`'s `_GEOM_EPS`, carried over unchanged: it
# keeps the arcsin arguments strictly inside (-1, 1), where the derivative is
# finite, and the denominators away from zero, so a leapfrog excursion cannot
# put a NaN in the gradient.  Values at any real posterior mode are orders of
# magnitude away from it.
# ----------------------------------------------------------------------

_GEOM_EPS = 1e-6


def _conjunction_denominator(esinw, secondary, xp=pt):
    """`1 + e sin omega` at the transit, `1 - e sin omega` at the eclipse."""
    signed = -esinw if secondary else esinw
    return xp.clip(1.0 + signed, _GEOM_EPS, np.inf)


def contact_duration(
    ar, cosi, sini, ecc, esinw, p, period, secondary, edge, xp=pt
):
    """One of Winn 2010's two arcsin durations.

    `edge` is `+1` for the 1st-to-4th contact duration (radius sum, T_14) and
    `-1` for the 2nd-to-3rd (radius difference, T_23).  `secondary` selects
    the occultation conjunction.

    Backend-agnostic through `xp=`, the `skyframe.py` idiom, and that is not
    decoration: review 8.8.7's seed solver has to evaluate exactly the
    duration the likelihood does, from inside the relaxation engine, where
    there is no tensor graph.  A numpy transcription would be a second copy
    of the physics free to drift from this one -- and the seed and the
    likelihood disagreeing about the duration is precisely the failure the
    seeding exists to prevent.  Only `xp.clip`, `xp.sqrt`, `xp.abs` and
    `xp.arcsin` are used, which numpy and pytensor spell identically;
    `pt.sqr` deliberately is not.
    """
    denom = _conjunction_denominator(esinw, secondary, xp=xp)
    ecc_factor = xp.sqrt(xp.clip(1.0 - ecc * ecc, _GEOM_EPS, 1.0))
    impact = ar * cosi * (1.0 - ecc * ecc) / denom
    edge_sum = 1.0 + edge * p
    radicand = xp.clip(edge_sum * edge_sum - impact * impact, 0.0, np.inf)
    arg = xp.clip(
        xp.sqrt(radicand) / xp.clip(xp.abs(sini * ar), _GEOM_EPS, np.inf),
        -1.0 + _GEOM_EPS,
        1.0 - _GEOM_EPS,
    )
    return (period / np.pi) * xp.arcsin(arg) * ecc_factor / denom


def _contact_duration(ar, cosi, sini, ecc, esinw, p, period, secondary, edge):
    return contact_duration(
        ar, cosi, sini, ecc, esinw, p, period, secondary, edge, xp=pt
    )


def duration_pair(ar, cosi, sini, ecc, esinw, p, period, secondary, xp=pt):
    """`(T_14, T_23)` at one conjunction -- the pair every duration is made of."""
    kw = dict(secondary=secondary, xp=xp)
    return (
        contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=1.0, **kw
        ),
        contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=-1.0, **kw
        ),
    )


@register_physics
def calc_impact_secondary(ar, cosi, ecc, esinw):
    """Occultation impact parameter -- `calc_b` at the other conjunction.

    The planet is at `r = a(1-e^2)/(1 - e sin omega)` there, so this is
    `calc_b` with the sign of `e sin omega` flipped.  A separate function and
    not a flag on `calc_b`, because a manifest entry names one function and
    the two are different parameters (`b` and `bs`).
    """
    return (
        ar * cosi * (1.0 - pt.sqr(ecc)) / _conjunction_denominator(esinw, True)
    )


@register_physics
def calc_t14(ar, p, cosi, sini, ecc, esinw, period):
    """Total transit duration, 1st to 4th contact."""
    return _contact_duration(
        ar, cosi, sini, ecc, esinw, p, period, secondary=False, edge=1.0
    )


@register_physics
def calc_t14s(ar, p, cosi, sini, ecc, esinw, period):
    """Total occultation duration, 1st to 4th contact."""
    return _contact_duration(
        ar, cosi, sini, ecc, esinw, p, period, secondary=True, edge=1.0
    )


@register_physics
def calc_tfwhm(ar, p, cosi, sini, ecc, esinw, period):
    """FWHM transit duration, `(T_14 + T_23)/2`."""
    kw = dict(secondary=False)
    return 0.5 * (
        _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=1.0, **kw
        )
        + _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=-1.0, **kw
        )
    )


@register_physics
def calc_tfwhms(ar, p, cosi, sini, ecc, esinw, period):
    """FWHM occultation duration, `(T_{S,14} + T_{S,23})/2`."""
    kw = dict(secondary=True)
    return 0.5 * (
        _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=1.0, **kw
        )
        + _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=-1.0, **kw
        )
    )


@register_physics
def calc_tau(ar, p, cosi, sini, ecc, esinw, period):
    """Ingress/egress transit duration, `(T_14 - T_23)/2`."""
    kw = dict(secondary=False)
    return 0.5 * (
        _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=1.0, **kw
        )
        - _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=-1.0, **kw
        )
    )


@register_physics
def calc_taus(ar, p, cosi, sini, ecc, esinw, period):
    """Ingress/egress occultation duration, `(T_{S,14} - T_{S,23})/2`."""
    kw = dict(secondary=True)
    return 0.5 * (
        _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=1.0, **kw
        )
        - _contact_duration(
            ar, cosi, sini, ecc, esinw, p, period, edge=-1.0, **kw
        )
    )


@register_physics
def calc_max_ecc(ar, p):
    return 1.0 - 1.0 / ar - p / ar


# ----------------------------------------------------------------------
# A priori transit and eclipse probabilities (Winn 2010 eq 9; EXOFASTv2
# derivepars.pro's pt/ptg/ps/psg).  For an isotropic orientation, the
# probability that the planet's disk overlaps the star's at conjunction is
#
#   P = (R_* +/- R_P) / r_conj = (1 +/- p)/(a/R_*) * (1 +/- e sin omega)/(1 - e^2)
#
# with `+p` counting grazing geometries (any overlap) and `-p` only full
# ones (non-grazing), and r_conj = a(1-e^2)/(1 +/- e sin omega) the
# star-planet separation at the transit (+) or eclipse (-) conjunction --
# the same two denominators the durations above use, so
# `_conjunction_denominator` is reused for the sign.  The (1 - e^2) floor is
# `contact_duration`'s.  These are probabilities only for ar large enough
# that the expression stays below 1; Winn's formula is not clipped and
# neither is this, since a value above 1 at a grazing-geometry start is a
# legitimate readout, not a NaN.
# ----------------------------------------------------------------------


def _conjunction_probability(ar, p, ecc, esinw, secondary, edge):
    """One of the four Winn 2010 eq 9 probabilities.

    `edge` is `+1` for any overlap (grazing included) and `-1` for a full
    transit; `secondary` selects the eclipse conjunction.
    """
    numer = _conjunction_denominator(esinw, secondary)
    one_minus_e2 = pt.clip(1.0 - pt.sqr(ecc), _GEOM_EPS, 1.0)
    return (1.0 + edge * p) / ar * numer / one_minus_e2


@register_physics
def calc_ptg(ar, p, ecc, esinw):
    """A priori transit probability, grazing geometries included."""
    return _conjunction_probability(
        ar, p, ecc, esinw, secondary=False, edge=1.0
    )


@register_physics
def calc_pt(ar, p, ecc, esinw):
    """A priori non-grazing (full) transit probability."""
    return _conjunction_probability(
        ar, p, ecc, esinw, secondary=False, edge=-1.0
    )


@register_physics
def calc_psg(ar, p, ecc, esinw):
    """A priori eclipse probability, grazing geometries included."""
    return _conjunction_probability(
        ar, p, ecc, esinw, secondary=True, edge=1.0
    )


@register_physics
def calc_ps(ar, p, ecc, esinw):
    """A priori non-grazing (full) eclipse probability."""
    return _conjunction_probability(
        ar, p, ecc, esinw, secondary=True, edge=-1.0
    )


# ----------------------------------------------------------------------
# Irradiation (EXOFASTv2 derivepars.pro's teq and fave).
#
#   T_eq = T_eff * sqrt(R_* / (2 a))
#
# is the equilibrium temperature of a zero-albedo planet that re-radiates
# over its whole surface (f = 1, A_B = 0 in the general
# T_eff sqrt(R_*/a) [f (1 - A_B)]^(1/4)); it is evaluated at the semi-major
# axis, i.e. without the eccentricity average -- both as EXOFASTv2 has it.
#
#   <F> = sigma_sb T_eff^4 / (a/R_* (1 + e^2/2))^2
#
# is the time-averaged incident flux over an eccentric orbit: <a^2/r^2>
# over time is (1 - e^2)^(-1/2), which EXOFASTv2 approximates by its
# second-order expansion 1 + e^2/2 -- and here too the expansion is kept
# so that the two codes report the same number.  EXOFASTv2 reports it in
# 10^9 erg s-1 cm-2; this one reports it in units of Earth's insolation
# (constants.EARTH_INSOLATION_CGS, 1361 W m-2), so the sigma_sb and the
# unit both live in cgs and the ratio is dimensionless.  No conversion is
# declared in defaults.yaml because "Earth insolation" is not an astropy
# unit -- the physics function IS the unit.
# ----------------------------------------------------------------------


@register_physics
def calc_delta(p):
    """Geometric transit depth, (R_P/R_*)^2 -- no limb darkening.

    EXOFASTv2's `delta`.  Its `depth` (the flux decrement at mid-transit
    with the band's limb darkening) is a different quantity and is not
    ported here: it needs a band, and a planet may transit in several.
    """
    return pt.sqr(p)


# Tidal circularization timescale, Adams & Laughlin (2006) eq 2 as
# EXOFASTv2's derivepars.pro evaluates it, with the planet's tidal quality
# factor fixed at Q_P = 1e6:
#
#   tau_circ = (4 Q_P / 63) sqrt(a^3 / (G M_*)) (M_P / M_*) (a / R_P)^5
#              (1 - e^2)^(13/2) / (1 + 6 e^2)
#
# sqrt(a^3 / (G M_*)) is in DAYS in the internal unit system (a in solRad,
# M_* in solMass, G in solRad^3 solMass^-1 d^-2), so the only conversion is
# days -> Gyr, which the physics applies because "Gyr" IS the declared unit
# (defaults.yaml keeps user and internal unit both Gyr, like fave).  Q_P is
# not a fitted or configurable quantity in either code: a planet's tidal Q
# is unknown to an order of magnitude, and the reported timescale carries
# that caveat by construction.
TIDAL_QP = 1.0e6
DAYS_PER_GYR = 365.25e9


@register_physics
def calc_tcirc(a, ar, p, mass, star_mass, ecc):
    """Tidal circularization timescale (Gyr); signed like mass."""
    e2 = pt.sqr(ecc)
    tau_days = (
        (4.0 * TIDAL_QP / 63.0)
        * pt.sqrt(pt.power(a, 3) / (G * star_mass))
        * (mass / star_mass)
        * pt.power(ar / p, 5)
        * pt.power(1.0 - e2, 6.5)
        / (1.0 + 6.0 * e2)
    )
    return tau_days / DAYS_PER_GYR


@register_physics
def calc_omegagr(a, star_mass, period, ecc):
    """General-relativistic apsidal precession rate, in rad/day.

    omega_dot = 3 G M_* n / (a c^2 (1 - e^2)), n = 2 pi / P -- the
    Einstein precession as derivepars.pro writes it (host mass alone in
    the numerator; the mean motion carries the total mass through P).
    Internal rad/day; defaults.yaml reports deg/century, EXOFASTv2's unit,
    through the `century` unit exozippy/units.py registers with astropy.
    Mercury: 43 arcsec/century.
    """
    n = TWOPI / period
    return (
        3.0
        * G
        * star_mass
        * n
        / (a * pt.sqr(C_LIGHT_RSUN_PER_DAY) * (1.0 - pt.sqr(ecc)))
    )


@register_physics
def calc_teq(teff, ar):
    """Equilibrium temperature (K): zero albedo, full redistribution."""
    return teff * pt.sqrt(1.0 / (2.0 * ar))


@register_physics
def calc_fave(teff, ar, ecc):
    """Time-averaged incident flux, in units of Earth's insolation."""
    flux_cgs = (
        SIGMA_SB_CGS
        * pt.power(teff, 4)
        / pt.sqr(ar * (1.0 + pt.sqr(ecc) / 2.0))
    )
    return flux_cgs / EARTH_INSOLATION_CGS


# Bolometric approximation of the Doppler beaming factor (Faigler & Mazeh
# 2011, eq. 1: A_beam = (4-alpha)*K/c, with the bandpass-dependent spectral
# index alpha set to 0). Confirmed against EXOFASTv2's step2pars.pro line
# 260, which uses beam = 4*K/c -- i.e. alpha_beam=1, not alpha=0 as the
# 2011 paper's bolometric case would give. alpha_beam's true value runs
# 0.8-1.2 depending on bandpass, but EXOFASTv2 fixes it at 1 (factor of 4),
# so we match that rather than the paper's exact bolometric limit.
BEAM_FACTOR = 4.0


@register_physics
def calc_beam_from_K(K):
    """Doppler beaming amplitude (ppm) from the RV semi-amplitude K.

    K arrives in its internal unit (solRad/d, see planet/defaults.yaml);
    converted to m/s before forming the dimensionless K/c ratio.

    step2pars.pro:258 itself stores the dimensionless 4*K/c straight into
    a field documented (and later consumed elsewhere) as ppm -- it never
    multiplies by 1e6, so downstream code that treats it as ppm silently
    divides by 1e6 again to compensate. That's a bug in EXOFASTv2, not a
    convention to match: the `* 1e6` here is the physically correct ppm
    value, so a future exofast_tran.pro parity check should NOT "fix"
    this back down to match step2pars.pro's unscaled number.
    """
    k_mps = K * SOLRAD_PER_DAY_TO_MPS
    return BEAM_FACTOR * (k_mps / C_MPS) * 1e6


# --- Chen & Kipping 2017 mass-radius relation -------------------------------
# Ported from EXOFASTv2's massradius_chen.pro (Chen & Kipping 2017, ApJ 834,
# 17, Table 2).
# https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
# A continuous broken power law R(M) in Earth units: the segment
# normalizations chain so adjacent segments meet at the break masses.
# Segments: Terran worlds (<= 2.04 Mearth) / Neptunian worlds (<= 0.414
# Mjup) / Jovian worlds (<= 0.08 Msun) / Stars.

CHEN_MASS_BREAKS = (2.04, 131.58079, 26644.8321)  # Mearth
CHEN_EXPONENTS = (0.279, 0.589, -0.044, 0.881)
# Per-segment scatter, as a fraction of the predicted radius.
CHEN_RP_FRAC = (0.0403, 0.1460, 0.0737, 0.0443)

_CHEN_T1, _CHEN_T2, _CHEN_T3 = CHEN_MASS_BREAKS
_CHEN_S1, _CHEN_S2, _CHEN_S3, _CHEN_S4 = CHEN_EXPONENTS
_CHEN_N1 = 1.0
_CHEN_N2 = _CHEN_T1 ** (_CHEN_S1 - _CHEN_S2)
_CHEN_N3 = _CHEN_N2 * _CHEN_T2 ** (_CHEN_S2 - _CHEN_S3)
_CHEN_N4 = _CHEN_N3 * _CHEN_T3 ** (_CHEN_S3 - _CHEN_S4)


@register_physics
def calc_chen_radius(mpearth):
    """Chen & Kipping 2017 predicted radius (Rearth) from mass (Mearth).

    mpearth must be positive: every switch branch (and its gradient) is
    evaluated for every input, and a non-integer power of a negative mass
    is NaN.  Callers clip first (EXOFASTv2 uses mpearth > 1d-10).
    """
    return pt.switch(
        mpearth <= _CHEN_T1,
        _CHEN_N1 * mpearth**_CHEN_S1,
        pt.switch(
            mpearth <= _CHEN_T2,
            _CHEN_N2 * mpearth**_CHEN_S2,
            pt.switch(
                mpearth <= _CHEN_T3,
                _CHEN_N3 * mpearth**_CHEN_S3,
                _CHEN_N4 * mpearth**_CHEN_S4,
            ),
        ),
    )


@register_physics
def calc_chen_radius_sigma(mpearth):
    """Scatter (Rearth) of the Chen & Kipping prediction at mass mpearth.

    Fractional per segment, so it scales with the prediction.  Same
    positive-mass requirement as calc_chen_radius.
    """
    frac = pt.switch(
        mpearth <= _CHEN_T1,
        CHEN_RP_FRAC[0],
        pt.switch(
            mpearth <= _CHEN_T2,
            CHEN_RP_FRAC[1],
            pt.switch(
                mpearth <= _CHEN_T3,
                CHEN_RP_FRAC[2],
                CHEN_RP_FRAC[3],
            ),
        ),
    )
    return calc_chen_radius(mpearth) * frac
