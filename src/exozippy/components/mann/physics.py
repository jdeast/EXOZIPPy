"""Mann+ empirical mass/radius relations for late-type dwarfs.

Translated from EXOFASTv2's massradius_mann.pro (the orphaned numpy
version that used to live in exozippy/evolutionary_model/massradius_mann.py
tracked the same source).  The polynomials are written against PyTensor so
they sit inside the NUTS gradient graph.

References
----------
Mann et al. 2015, ApJ 804, 64   -- radius from absolute Ks (Table 1)
Mann et al. 2019, ApJ 871, 63   -- mass from absolute Ks (Table 6, n=5)
"""

import pytensor.tensor as pt

from ...physics_registry import register_physics

# Mann+2015 Table 1: radius from absolute Ks.
_R_COEFF_NOFEH = (1.9515, -0.3520, 0.01680)              # eq 4
_R_COEFF_FEH = (1.9305, -0.3466, 0.01647, 0.04458)       # eq 5

# Mann+2019 Table 6 (n=5): mass from absolute Ks.
_M_ZP = 7.5
_M_COEFF_NOFEH = (-0.642, -0.208, -8.43e-4, 7.87e-3, 1.42e-4, -2.13e-4)   # eq 4
_M_COEFF_FEH = (-0.647, -0.207, -6.53e-4, 7.13e-3, 1.84e-4, -1.6e-4)      # eq 5
_M_FEH_TERM = -0.0035

# Fractional scatter of each relation, keyed by whether the [Fe/H]-dependent
# form is used.  These are EXOFASTv2's rstaremfloor/mstaremfloor defaults.
RSTAR_FLOOR = {False: 0.0289, True: 0.027}
MSTAR_FLOOR = {False: 0.020, True: 0.021}

# Calibration ranges.  EXOFASTv2 (exofast_chi2v2.pro) warns outside these;
# see Mann.register_parameters, which warns at startup for the same reason.
FEH_RANGE = (-0.6, 0.4)
MSTAR_RANGE = (0.075, 0.7)


def _poly(x, coeff):
    """Horner evaluation of sum_i coeff[i] * x**i on a PyTensor node."""
    out = coeff[-1]
    for c in reversed(coeff[:-1]):
        out = out * x + c
    return out


@register_physics
def calc_mann_radius(absks, feh=None):
    """Stellar radius (solRad) from absolute Ks (mag), optionally [Fe/H] (dex).

    feh=None selects Mann+2015 eq 4; otherwise eq 5, which adds a linear
    metallicity term to the same quadratic in Ks.
    """
    if feh is None:
        return _poly(absks, _R_COEFF_NOFEH)
    return _poly(absks, _R_COEFF_FEH[:3]) + _R_COEFF_FEH[3] * feh


@register_physics
def calc_mann_mass(absks, feh=None):
    """Stellar mass (solMass) from absolute Ks (mag), optionally [Fe/H] (dex).

    feh=None selects Mann+2019 eq 4; otherwise eq 5, which scales the same
    quintic by a linear metallicity factor.
    """
    x = absks - _M_ZP
    if feh is None:
        return 10.0 ** _poly(x, _M_COEFF_NOFEH)
    return (1.0 + feh * _M_FEH_TERM) * 10.0 ** _poly(x, _M_COEFF_FEH)
