"""
MIST evolutionary-track systematic-error floor.

Translated from EXOFASTv2's ``massradius_mist.pro`` -- specifically
its ``percenterror`` expression, which is the mass-dependent floor
EXOFASTv2 applies to the Teff/radius/age/[Fe/H] penalties between a star's
sampled parameters and the MIST-interpolated prediction, in the absence of a
user-supplied floor.
"""

import pytensor.tensor as pt

# Fit coefficients for percenterror(mstar) below; EXOFASTv2's massradius_mist.pro.
_A, _B, _C = 0.03, -0.025, 0.045


def percent_error_from_logmass(log_m):
    """The floor as a function of log10(mstar/solMass).

    Pure arithmetic, so it evaluates under numpy and under pytensor alike --
    which is why the polynomial lives here rather than inside
    :func:`calc_mist_percent_error`.  ``build_likelihood`` reaches it through
    that wrapper with a symbolic ``star.mass``; the component's EEP seed
    search reaches it directly with the starting ``star.logmass``, which is
    ALREADY log10(mass), so routing that through a ``pt.log10(10**x)`` round
    trip would both lose precision and hand back a symbolic node the numpy
    search cannot use.
    """
    return _A + _B * log_m + _C * log_m * log_m


def calc_mist_percent_error(mstar):
    """Mass-dependent fractional systematic-error floor (EXOFASTv2).

    ``mstar`` in solMass. Returns a dimensionless fraction: 3.00% at 1
    solMass, rising steeply toward low mass (5.54% at 0.3 solMass, where the
    models are least trustworthy).

    It is an upward parabola in log10(mstar), but NOT one centered on the
    Sun: its vertex is at ``-_B / (2 * _C)`` = 0.2778 dex, i.e. 1.90 solMass,
    where it bottoms out at 2.65%.  So the floor is slightly LOOSER at 1
    solMass than at 2, and 3 solMass (2.83%) is still below the solar value
    -- do not "fix" a test that finds the floor decreasing between 1 and 2
    solMass, and do not describe this as rising away from the Sun in both
    directions.

    Used as the default floor for the Teff/radius/age penalties
    (fractional) and, unusually, for [Fe/H] too (there it is used as an
    absolute dex floor rather than a fraction of the prediction -- [Fe/H]
    can be zero or negative, so a fractional floor is meaningless; this is
    exactly EXOFASTv2's own convention, not an approximation of it).
    """
    return percent_error_from_logmass(pt.log10(mstar))
