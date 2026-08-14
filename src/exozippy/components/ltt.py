"""
Analytic light-travel-time (Roemer) delay -- the shared implementation for
any consumer needing the retarded time at which to evaluate an orbiting
body's Keplerian position, so the resulting signal (flux dip, RV,
photocenter position) arrives at the observer at the given observed time.

EXOFASTv2's target2bjd.pro/bjd2target.pro solve this with a while-loop
fixed-point iteration to a tolerance -- data-dependent control flow with no
static PyTensor graph, unusable under NUTS. This instead follows exoplanet's
closed-form approach (KeplerianOrbit._get_retarded_position; derivation
credited there to Luger/Agol in starry issue #66,
https://github.com/rodluger/starry/issues/66): a single Kepler solve gives
the body's line-of-sight position, velocity, and acceleration at the
UNCORRECTED time; the retardation condition c*delay = z0 - z(t - delay) is
then solved analytically from a 2nd-order Taylor expansion of z(t) about
delay=0 (a quadratic in delay), with a linear fallback when the quadratic
coefficient (the acceleration term) is negligible. No loop, no tolerance --
truncation error is third order in delay/period, sub-microsecond for
anything transiting.

Sign convention: z, vz, az here are derived from scratch (standard two-body
calculus, not copied from exoplanet) for EXOZIPPy's OWN line-of-sight
convention -- transit.py's Z = r_norm*sin_wf*sin_i, "+ = toward the
observer" (in front of the star during primary transit). This is
deliberately NOT exoplanet's own convention: exoplanet's _rotate_vector
computes Z = -r*sin(f+omega)*sin(i) (verified by reading its source), the
opposite sign. The overall sign of the correction (t - delay vs t + delay)
is pinned empirically by tests/test_ltt.py's analytic spot-checks against
known physics (the 2a/c secondary-eclipse offset, the a/c delay amplitude),
not asserted here on paper.

Units: everything here is in EXOZIPPy's native internal unit system --
R_sun, M_sun, day (see constants.py) -- not AU/second like EXOFASTv2 or
AU/day like exoplanet's own c_light. `a_rel` is the physical semi-major axis
of the RELATIVE (e.g. star-planet) orbit in R_sun -- orbit.a.value, already
exists (its internal_unit is solRad; not orbit.arsun, which does not exist
as a manifest key despite the expression function being named calc_arsun);
no separate stellar-radius multiply is needed since that physical scale is
already baked into a.

Barycentric scaling: the event (transit/eclipse/RM: the planet; RV reflex:
the star) sits at a fixed fraction of the relative-orbit separation from the
system barycenter -- m_primary/m_total for the planet, m_companion/m_total
for the star (EXOFASTv2's target2bjd.pro factor = q/(1+q), q = M1/M2, is the
same relation). Since (a_body/r_body(t)) = (a_rel/r_rel(t)) exactly (the
factor cancels in that ratio), only z/vz/az themselves need the factor
multiply, not the (a/r)^3 term inside az.
"""

import pytensor.tensor as pt
from exoplanet_core.pymc import ops

from ..constants import C_LIGHT_RSUN_PER_DAY


def line_of_sight_kinematics(
    t, tp, n, ecc, sinw, cosw, sin_i, a_rel, factor=1.0
):
    """Line-of-sight position, velocity, and acceleration of an orbiting
    body at the given (uncorrected) time -- one Kepler solve, everything
    else closed-form in the true anomaly it returns.

    Parameters
    ----------
    t : tensor
        Time(s) to evaluate at [days]. Broadcastable against tp/n/ecc/etc.
    tp : tensor
        Time of periastron of the RELATIVE orbit [days].
    n : tensor
        Mean motion, 2*pi/period [rad/day] (e.g. orbit.n.value).
    ecc : tensor
        Eccentricity of the relative orbit.
    sinw, cosw : tensor
        sin/cos of the argument of periastron.
    sin_i : tensor
        sin of the orbital inclination.
    a_rel : tensor
        Physical semi-major axis of the RELATIVE orbit [R_sun] (orbit.a.value
        -- internal_unit solRad; NOT orbit.arsun, which does not exist).
    factor : tensor or float
        Barycentric scaling for the body whose kinematics are wanted --
        m_primary/m_total for the planet's event (transit/eclipse/RM),
        m_companion/m_total for the star's (RV reflex). Default 1.0 (the
        un-scaled relative orbit, EXOFASTv2's infinite-mass-ratio limit).

    Returns
    -------
    z, vz, az : tensor
        Line-of-sight position [R_sun], velocity [R_sun/day], and
        acceleration [R_sun/day^2], each scaled by `factor`.
    """
    M = (t - tp) * n
    sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

    ecc_factor = pt.sqrt(1.0 - pt.sqr(ecc))
    r_over_a = (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)  # r / a_rel
    sin_wf = sinw * cosf + cosw * sinf  # sin(f + w)
    cos_wf = cosw * cosf - sinw * sinf  # cos(f + w)

    r = a_rel * r_over_a  # physical separation, relative orbit [R_sun]
    z_rel = r * sin_wf * sin_i

    # d/dt[r*sin(f+w)]*sin_i, derived directly (not copied) from
    # dr/dt = n*a*e*sinf/sqrt(1-e^2) and r^2*df/dt = n*a^2*sqrt(1-e^2):
    # standard two-body calculus, matching the classic RV-curve form.
    vamp = n * a_rel / ecc_factor
    vz_rel = vamp * sin_i * (ecc * cosw + cos_wf)

    # Line-of-sight component of the two-body acceleration, -(GM/r^3)*z
    # with GM = n^2*a^3 (Kepler's third law) -- a coordinate-sign-agnostic
    # relation (Newton's second law applies component-wise regardless of
    # the sign convention chosen for z), so this holds unchanged for
    # EXOZIPPy's z sign despite differing from exoplanet's.
    az_rel = -pt.sqr(n) * (1.0 / r_over_a) ** 3 * z_rel

    return z_rel * factor, vz_rel * factor, az_rel * factor


def solve_delay(z, vz, az, z0=0.0, c=C_LIGHT_RSUN_PER_DAY):
    """Closed-form light-travel-time delay from a 2nd-order Taylor
    expansion of z(t) about the uncorrected time (exoplanet's
    KeplerianOrbit._get_retarded_position; derivation credited there to
    Luger/Agol in starry issue #66).

    Parameters
    ----------
    z, vz, az : tensor
        Line-of-sight position [R_sun], velocity [R_sun/day], and
        acceleration [R_sun/day^2] of the body at the uncorrected time
        (e.g. from line_of_sight_kinematics).
    z0 : tensor or float
        Reference point along the line of sight whose light-travel time is
        taken to be zero. Default 0.0 (the system barycenter).
    c : float
        Speed of light [R_sun/day]. Defaults to C_LIGHT_RSUN_PER_DAY.

    Returns
    -------
    delay : tensor
        Light-travel-time delay [days], same shape as z. See the module
        docstring: the sign here is whatever falls out of this formula --
        tests/test_ltt.py is what confirms t - delay is the right
        direction, not this docstring.

    Notes
    -----
    pt.switch evaluates BOTH branches eagerly (needed for autodiff through
    the selected one), so whenever the linear branch is selected (az == 0,
    e.g. a face-on orbit where z/vz/az are all identically 0), the
    quadratic branch still gets evaluated with az == 0 in its denominator.
    A literal divide-by-zero there produces a forward-pass inf (confirmed:
    c/0 -> inf, and the isolated expression's own local gradient is -inf
    at az == 0) -- the shape of the JAX/PyTensor "where-trap" that can
    poison pt.grad through a switch/where even on the discarded branch. az
    is floored away from exactly 0 (az_safe) everywhere it appears in the
    quadratic branch (both the leading c/az and the radicand's az term,
    not just the division -- a large-z, hand-set-az=0 call would otherwise
    still risk a negative radicand under a denominator-only guard), which
    at minimum removes the forward RuntimeWarning and is the defensively
    correct fix regardless of gradient behavior. NOTE: tests/test_ltt.py's
    test_retarded_time_gradient_is_finite_at_az_zero_via_numpyro checked
    directly (pt.grad, both the default and mode="JAX" backends, scalar
    and vectorized) whether the where-trap actually reproduces for THIS
    formula, and found pt.grad of the full switch expression stayed
    finite at az == 0 even WITHOUT this guard -- pt.switch's gradient did
    not propagate the unselected branch's singularity here. So this fix
    is not a demonstrated bugfix for a reproduced gradient failure; see
    that test's docstring for the full finding. This only touches the
    UNSELECTED branch's numerics when az == 0 exactly; the selection
    condition (pt.lt(pt.abs(az), 1e-10)) and the quadratic branch's math
    for any az != 0 are unchanged.
    """
    az_safe = pt.where(pt.eq(az, 0.0), 1.0, az)
    return pt.switch(
        pt.lt(pt.abs(az), 1.0e-10),
        (z0 - z) / (c + vz),
        (c / az_safe)
        * (
            (1.0 + vz / c)
            - pt.sqrt((1.0 + vz / c) ** 2 - 2.0 * az_safe * (z0 - z) / c**2)
        ),
    )


def retarded_time(
    t,
    tp,
    n,
    ecc,
    sinw,
    cosw,
    sin_i,
    a_rel,
    factor=1.0,
    z0=0.0,
    c=C_LIGHT_RSUN_PER_DAY,
):
    """Observed time -> target-frame time, correcting for the light-travel
    time across the target system (one Kepler solve; the caller re-solves
    Kepler's equation at the returned time using its own existing M/ops.
    kepler code, rather than this module duplicating that per consumer --
    see the quad_limb_darkened_flux precedent in limbdark.py).

    Parameters are exactly line_of_sight_kinematics's plus z0/c, forwarded
    to solve_delay. See those two functions' docstrings for units and the
    sign-convention caveat.

    Returns
    -------
    t_corrected : tensor
        t - delay: the target-frame time at which to evaluate the body's
        Keplerian position (feed this into the consumer's own M = (t - tp)
        * n / ops.kepler code in place of the observed t).
    delay : tensor
        The delay itself [days], same shape as t -- returned alongside for
        diagnostics/testing.
    """
    z, vz, az = line_of_sight_kinematics(
        t, tp, n, ecc, sinw, cosw, sin_i, a_rel, factor
    )
    delay = solve_delay(z, vz, az, z0, c)
    return t - delay, delay
