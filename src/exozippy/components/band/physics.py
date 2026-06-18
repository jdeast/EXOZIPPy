import numpy as np
from scipy.interpolate import RegularGridInterpolator
from exozippy.physics_registry import register_physics


@register_physics
def claret_ld_quadratic(teff, logg, feh, grid):
    """Interpolate quadratic LD coefficients (u1, u2) from a Claret grid.

    grid must be a dict with keys 'teff', 'logg', 'feh', 'u1', 'u2'
    (1D axes and matching N-D value arrays).
    """
    axes = (grid['teff'], grid['logg'], grid['feh'])
    interp_u1 = RegularGridInterpolator(axes, grid['u1'], method='linear',
                                         bounds_error=False, fill_value=None)
    interp_u2 = RegularGridInterpolator(axes, grid['u2'], method='linear',
                                         bounds_error=False, fill_value=None)
    pt = np.array([[float(teff), float(logg), float(feh)]])
    return float(interp_u1(pt)[0]), float(interp_u2(pt)[0])
