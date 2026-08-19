import numpy as np
import pytensor.tensor as pt

from ...constants import (
    C_MPS,
    DENSITY_CONST,
    KEPLER_CONST,
    LOGG_CONST,
    SOLRAD_PER_DAY_TO_MPS,
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
