"""
Quadratic limb-darkened occultation flux -- the single shared implementation.

exoplanet_core's ``quad_solution_vector(b, r)`` returns the occultation
solution vector ``s`` in starry's Green's basis, NOT in powers of mu.  Feeding
it the naive ``[1 - u1 - u2, u1, u2]`` intensity coefficients is a silent,
u2-only error (it is exact at u2 = 0), so every call site in the package goes
through ``quad_limb_darkened_flux`` below rather than open-coding the
change-of-basis.

Change of basis, Agol, Luger & Foreman-Mackey (2020), matching
``exoplanet.light_curves.limb_dark.get_cl()``:

    c0 = 1 - u1 - 1.5*u2,  c1 = u1 + 2*u2,  c2 = -0.25*u2
    norm = dot(s_off, c) = pi*(c0 + c1/1.5),  s_off = [pi, 2pi/3, 0]

Verified against brute-force disk integration of
``I(mu) = 1 - u1(1 - mu) - u2(1 - mu)^2`` (tests/test_limbdark.py).
"""

import numpy as np
import pytensor.tensor as pt
from exoplanet_core.pymc import ops


def quad_limb_darkened_flux(b, r, u1, u2):
    """Normalized flux of a quadratically limb-darkened star occulted by an
    opaque disk: 1.0 when there is no overlap, < 1.0 during the occultation.

    Parameters
    ----------
    b : tensor
        Sky-plane separation of the centres, in units of the stellar radius.
    r : tensor or float
        Occulter radius in units of the stellar radius.  Broadcast against
        ``b`` (``quad_solution_vector`` needs the two the same shape).
    u1, u2 : tensor or float
        Quadratic limb-darkening coefficients, broadcastable against ``b``.

    Notes
    -----
    This makes no distinction between a transit and a secondary eclipse --
    the caller is responsible for zeroing out the far-side geometry.
    """
    sol = ops.quad_solution_vector(b, r + pt.zeros_like(b))  # (..., 3)
    c0 = 1.0 - u1 - 1.5 * u2
    c1 = u1 + 2.0 * u2
    c2 = -0.25 * u2
    norm = np.pi * (c0 + c1 / 1.5)
    return (sol[..., 0] * c0 + sol[..., 1] * c1 + sol[..., 2] * c2) / norm
