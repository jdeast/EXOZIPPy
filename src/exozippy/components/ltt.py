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
opposite sign. Because the conventions differ, solve_delay's quadratic is
solved here from scratch too -- see its docstring; transcribing exoplanet's
expression entered vz with the wrong sign (14 ms, fixed 2026-08-15). Signs
are pinned by tests/test_ltt.py against a brentq solve of the retardation
condition itself, sampled AWAY from conjunction: the original spot-checks
(the 2a/c secondary-eclipse offset, the a/c amplitude, the face-on case)
all sit where vz == 0 exactly and so constrain nothing about its sign.

Barycentric factor: see `retarded_time`'s `factor` argument. For an
OCCULTATION it is (m_primary - m_companion)/m_total, NOT either body's own
barycentric fraction -- a transit is a two-body seam, not an emission
event.

Units: everything here is in EXOZIPPy's native internal unit system --
R_sun, M_sun, day (see constants.py) -- not AU/second like EXOFASTv2 or
AU/day like exoplanet's own c_light. `a_rel` is the physical semi-major axis
of the RELATIVE (e.g. star-planet) orbit in R_sun -- orbit.a.value, already
exists (its internal_unit is solRad; not orbit.arsun, which does not exist
as a manifest key despite the expression function being named calc_arsun);
no separate stellar-radius multiply is needed since that physical scale is
already baked into a.

Barycentric scaling: an EMISSION event (RV reflex: the star; thermal or
reflected light: the planet) sits at a fixed fraction of the relative-orbit
separation from the system barycenter -- m_companion/m_total for the star,
m_primary/m_total for the planet (EXOFASTv2's target2bjd.pro factor =
q/(1+q), q = M1/M2, is the same relation). An OCCULTATION (transit,
secondary eclipse, RM) is NOT an emission event and takes the mass
DIFFERENCE instead; `retarded_time`'s `factor` argument documents why.
Since (a_body/r_body(t)) = (a_rel/r_rel(t)) exactly (the factor cancels in
that ratio), only z/vz/az themselves need the factor multiply, not the
(a/r)^3 term inside az.
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
        Scaling applied to z/vz/az. Which value is correct depends on what
        KIND of observable is being retarded, not on which body it is
        "about":

        - OCCULTATION seam (transit, secondary eclipse, RM): use
          ``(m_primary - m_companion)/m_total``. A transit is not an
          emission event -- the planet emits nothing, it blocks light the
          STAR emitted -- so the observed separation needs BOTH bodies at
          their own retarded times, ``rho_p(t - z_p/c) - rho_s(t - z_s/c)``.
          Expanding both leaves the relative orbit evaluated at
          ``t - [(m_primary - m_companion)/m_total]*z_rel/c``: the star's
          own delay partially CANCELS the planet's. Using m_primary/m_total
          (EXOFASTv2's target2bjd.pro; this module until 2026-08-15) puts
          the star at the barycenter and so drops z_s/c entirely; using 1.0
          (exoplanet's get_relative_position) drops it and the planet's
          barycentric fraction too. All three agree to O(q) for a planet
          (~tens of ms), but for comparable masses only this one is right:
          at m_primary == m_companion the true primary-to-secondary offset
          is EXACTLY zero by symmetry (the bodies are always diametrically
          opposite), the standard eclipsing-binary result used to measure
          mass ratios (Kaplan 2010, Fabrycky 2010), while m_primary/m_total
          predicts a/c and 1.0 predicts 2a/c.
        - Light EMITTED by one body (RV reflex, Doppler beaming,
          ellipsoidal -- all from the star): that body's own barycentric
          fraction, ``m_companion/m_total`` for the primary.
        - Light emitted by the SECONDARY (thermal emission, reflected-light
          phase): ``m_primary/m_total``.

        Default 1.0 (the un-scaled relative orbit, the infinite-mass-ratio
        limit).

        MULTIPLE COMPANIONS. Everything above is the two-body result, and
        this module is called per orbit, so it cannot see the rest of the
        system. With N companions the star's reflex is
        ``z_star = -sum_i (M_i/M_tot) z_rel,i`` and planet b's occultation
        picks up three things a per-orbit call misses:

        1. its own factor becomes ``(M_star - M_b + sum_{i!=b} M_i)/M_tot``
           over the WHOLE system mass -- an O(q) correction to an O(q)
           correction, 0.05 ms for a two-Jupiter system, ignorable;
        2. a time shift ``+ sum_{i!=b} (M_i/M_tot) z_rel,i/c`` from the
           other companions moving the star along the line of sight. This
           is the classic light-time (LITE / LTT) eclipse-timing signal
           used to DETECT outer companions -- 2.5 SECONDS in amplitude for
           a Jupiter at 5 AU, varying on the OUTER period, so it reads as
           TTV rather than as an offset. Not modeled here or anywhere else
           in EXOZIPPy; it needs the full system state, not one orbit;
        3. a transverse displacement ``(z_rel,b/c)*sum_{i!=b}
           (M_i/M_tot)*d(rho_rel,i)/dt`` -- the star moves ACROSS the sky
           between the two retarded times. ~2 ms equivalent, and notably
           NOT expressible as a shift of any time axis, which is the point
           at which "just retard the timestamps" stops being the right
           mental model at all.

        (2) is the one that matters and is the one worth implementing if
        anyone needs it; (1) and (3) are sub-millisecond.

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
    expansion of z(t) about the uncorrected time (the approach of
    exoplanet's KeplerianOrbit._get_retarded_position; derivation credited
    there to Luger/Agol in starry issue #66).

    The quadratic is solved here from scratch rather than transcribed,
    because exoplanet's variables are NOT the ones this module builds and
    its expression is NOT the root of this quadratic. Two separate points:

    1. exoplanet's ``vz`` is ``-dZ/dt`` for its own ``Z`` -- its
       ``_rotate_vector`` returns ``Z = -r*sin(f+omega)*sin(i)`` (minus)
       while its ``vz`` carries a plus -- so its ``(c + vz)`` is really
       ``(c - dZ/dt)``. ``line_of_sight_kinematics`` above returns a
       genuinely self-consistent ``(z, dz/dt, d2z/dt2)``, so transcribing
       ``(c + vz)`` here would enter vz with the WRONG SIGN. It did until
       2026-08-15: a 14 ms error (a=0.05 AU, P=3 d, e=0.3, i=89), ~1000x
       the truncation error this expansion is supposed to have.
    2. exoplanet's expression is ``(c/az)*(A - sqrt(A^2 - B))`` in the
       variables below, which is ``-root(-B)``: the small root of the same
       quadratic with ``az -> -az``. It agrees with the true root in the
       part odd in B (the whole leading term) and flips the even part, so
       it is correct to first order in the acceleration and wrong at
       second, by ``(c/az)*B^2/(4*A^3)`` -- 9 us for that orbit, vs the 4 ns
       genuine third-order truncation of the correct root used here.
       Negligible for exoplanet's purposes; free to not inherit.

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
        Light-travel-time delay [days], same shape as z, satisfying
        ``c*delay = z0 - z(t - delay)`` -- so consumers evaluate at
        ``t - delay``. Every sign here is derived from that condition (see
        above); tests/test_ltt.py checks it against a brentq solve of the
        condition itself AT PHASES WHERE vz != 0, which is what the old
        conjunction-and-face-on spot-checks could not do.

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
    # A = 1 - vz/c and B = 2*az*(z0 - z)/c**2, so the quadratic
    # 0.5*az*d^2 + (c - vz)*d + (z - z0) = 0 has small root
    # (c/az)*(sqrt(A^2 + B) - A), whose az -> 0 limit is (z0 - z)/(c - vz).
    a_term = 1.0 - vz / c
    b_term = 2.0 * az_safe * (z0 - z) / c**2
    return pt.switch(
        pt.lt(pt.abs(az), 1.0e-10),
        (z0 - z) / (c - vz),
        (c / az_safe) * (pt.sqrt(a_term**2 + b_term) - a_term),
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


#: Orbit parameters `retarded_time` callers need. `Orbit.register_parameters`
#: only declares these when `Orbit._validate_bodies` passes (it warns
#: "mass/scale parameters (m_total, a, K) are disabled" otherwise), and only
#: manifest entries get an attribute -- so a geometry-only orbit has none of
#: them and reading `orbit.a.value` raises AttributeError.
REQUIRED_ORBIT_PARAMS = ("a", "m_primary", "m_companion", "m_total")


def orbit_supports_ltt(orbit_component):
    """Whether `orbit_component` carries the parameters the Roemer-delay
    correction needs.

    False for an orbit whose bodies did not resolve. Callers must treat
    that as "light travel time off for this orbit" rather than letting the
    AttributeError escape: `light_travel_time` defaults to ON, so an
    unguarded read turns a previously-working geometry-only fit into a
    crash at build_model() time.
    """
    return all(hasattr(orbit_component, p) for p in REQUIRED_ORBIT_PARAMS)
