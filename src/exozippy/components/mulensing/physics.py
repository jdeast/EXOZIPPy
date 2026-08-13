import numpy as np
import pytensor.tensor as pt

from ...constants import KAPPA, RSUN_TO_AU
from ...physics_registry import register_physics

# Positive floors for the two quantities whose logarithm the event-rate prior
# takes (lens.build_likelihood).  Both are ~6 orders of magnitude below the
# 1e-6 turn-on of the matching soft bounds there, so the barrier is already
# fully engaged wherever the floor bites and no reachable posterior region is
# affected -- the floors only replace a -inf/NaN wall with a finite plateau
# the soft bounds can push off of.
THETA_E_FLOOR = 1e-12  # mas
MU_REL_FLOOR = 1e-12  # mas/yr


@register_physics
def calc_pi_rel(dist_lens, dist_source):
    # Parallax = 1000 / distance (pc) -> mas
    # no matter what we do, we must not compute a NaN.
    # we make up values so we can compute some likelihood
    # then introduce penalties (see lens.build_likelihood) that will reject such non-physical solutions
    return (1000.0 / dist_lens) - (1000.0 / dist_source)


@register_physics
def calc_theta_E(mass_lens, pi_rel):
    # Angular Einstein Radius in mas.
    # Guard against negative pi_rel (source in front of lens): no lensing occurs,
    # but we must return a finite value so downstream parameters (rho, pi_E) don't
    # propagate NaN into the Op.  The lens.build_likelihood potentials penalise
    # this unphysical configuration so the sampler rejects it.
    #
    # mass_lens is guarded for the same reason: a lens body sampling a linear
    # mass may go negative (planet.mass allows it), and mlens_total is a plain
    # sum, so sqrt() would return NaN -- which build_likelihood then feeds to
    # log(theta_E), poisoning the logp and its gradient over a whole region
    # instead of penalising it.  The theta_E_singularity soft bound there does
    # the penalising once the value is finite.
    #
    # The whole radicand is floored at THETA_E_FLOOR**2 rather than clipping
    # theta_E afterwards: sqrt'(0) is infinite, and pt.maximum's zero gradient
    # on the clamped side then makes 0 * inf = NaN, so log(theta_E) had a NaN
    # gradient over the entire pi_rel <= 0 region even with a floor applied
    # downstream.  Flooring the argument keeps sqrt' finite, so the gradient
    # is a clean zero there and the source_behind_lens bound supplies the
    # restoring force.
    radicand = KAPPA * pt.maximum(mass_lens, 1e-12) * pt.maximum(pi_rel, 0.0)
    return pt.sqrt(pt.maximum(radicand, THETA_E_FLOOR**2))


@register_physics
def calc_mu_ra_rel(pm_ra_lens, pm_ra_source):
    return pm_ra_lens - pm_ra_source


@register_physics
def calc_mu_dec_rel(pm_dec_lens, pm_dec_source):
    return pm_dec_lens - pm_dec_source


@register_physics
def calc_mu_rel_mag(mu_ra_rel, mu_dec_rel):
    # Floored for the same reason as calc_theta_E's radicand: the two
    # components start equal (star pm_ra/pm_dec share one default), so an
    # exactly-zero relative proper motion is reachable, and sqrt'(0) = inf
    # poisons log(mu_rel_geo) in the event-rate prior -- and every
    # mu_*_rel / mu_rel_mag ratio -- with a NaN gradient.
    return pt.sqrt(
        pt.maximum(pt.sqr(mu_ra_rel) + pt.sqr(mu_dec_rel), MU_REL_FLOOR**2)
    )


# Heliocentric -> geocentric frame conversion (Gould 2004):
#   mu_geo = mu_helio - pi_rel * v_earth_perp(t0_par) / AU
# The star pm_ra/pm_dec are barycentric observables (what the galactic
# kinematic prior and any Gaia prior constrain), but the light-curve
# trajectory is in the Skowron+2011 GEOCENTRIC convention (Earth's position
# AND velocity at t0_par define the frame; see MulensInstrument), so t_E and
# the pi_E direction must use the geocentric relative proper motion.
# earth_vperp_* are Earth's velocity at t0_par projected on the sky
# (East, North) in AU/yr -- context constants injected by Lens.add_parameter
# (pi_rel in mas makes the product mas/yr).  The SIGN is pinned numerically
# against the trajectory formula in tests/test_mu_rel_geo.py: the flipped
# sign changes t_E by tens of percent at pi_rel ~ 0.35 mas.
@register_physics
def calc_mu_ra_rel_geo(mu_ra_rel, pi_rel, earth_vperp_e):
    return mu_ra_rel - pi_rel * earth_vperp_e


@register_physics
def calc_mu_dec_rel_geo(mu_dec_rel, pi_rel, earth_vperp_n):
    return mu_dec_rel - pi_rel * earth_vperp_n


@register_physics
def calc_t_E(theta_E, mu_rel_mag):
    # Convert mu_rel_mag from mas/yr to mas/day, then divide theta_E
    return theta_E / (mu_rel_mag / 365.25)


@register_physics
def calc_pi_E_N(pi_rel, theta_E, mu_dec_rel, mu_rel_mag):
    # pi_E points in the direction of relative proper motion
    pi_E_mag = pi_rel / theta_E
    return pi_E_mag * (mu_dec_rel / mu_rel_mag)


@register_physics
def calc_pi_E_E(pi_rel, theta_E, mu_ra_rel, mu_rel_mag):
    pi_E_mag = pi_rel / theta_E
    return pi_E_mag * (mu_ra_rel / mu_rel_mag)


@register_physics
def calc_q(*masses):
    # (companion_1, ..., companion_k, primary) -> per-companion mass ratios
    # q_j = M_companion_j / M_primary.  Each dep arrives as a length-1 slice
    # (scalar bracket maps in Lens.build_maps); k companions concatenate to a
    # shape-(k,) vector.
    companions, primary = masses[:-1], masses[-1]
    if len(companions) == 1:
        return companions[0] / primary
    return pt.concatenate([pt.atleast_1d(c) for c in companions]) / primary


# --- Mass-ratio sanitization ------------------------------------------------
# Validity range of the binary/multiple-lens magnification backends.  Both
# MulensModel and VBMicrolensing require a strictly positive, finite q, and the
# caustic solvers stop converging well before these limits.  Q_MIN sits a decade
# below lens.q's own defaults.yaml `lower: 1e-8` soft barrier, so on a fit that
# is behaving the clip is never active: the barrier turns the sampler around
# first.  Q_MAX is exactly lens.q's `upper` (q > 1 is legal -- it is the same
# geometry with the two bodies relabelled).
#
# This is a RANGE decision and ONLY a range decision.  It must never be paired
# with a NaN substitution -- see clip_q.
Q_MIN = 1e-9
Q_MAX = 100.0

_Q_NAN_ADVICE = (
    "q = M_companion / M_primary, so a non-finite q means one of the lens "
    "body masses is already non-finite.  Check the initval (and any "
    "'sigma: 0' link expression) on star.<primary lens>.logmass and on the "
    "companion's mass -- planet.<name>.log_q in log_q mode, "
    "planet.<name>.mass in linear mode, star.<name>.logmass for a stellar "
    "companion."
)


def clip_q(q):
    """Clip a symbolic mass ratio into the magnification model's valid range.

    Deliberately NOT paired with ``pt.nan_to_num``, which is what this used to
    be -- ``pt.clip(pt.nan_to_num(q, nan=Q_MIN), Q_MIN, Q_MAX)``, copied to
    five sites (review item 4.5).  Dropping the scrub is safe *and* strictly
    better:

    * It was unreachable.  ``q`` is ``m_companion / m_primary``; every mass in
      the chain descends from a logit-bounded sampled coordinate (star.logmass
      in [-9, 2.5] dex, planet.log_q, or a linear planet.mass with finite hard
      bounds -- and user bounds may only be tightened), so both operands are
      finite and the denominator, a stellar mass, has a hard floor of 1e-9
      solMass.  Measured on examples/DC2018_128 and examples/ob161003: q stays
      finite over the entire raw support out to raw = +/-1e12, and 2400 random
      raw points per event produced no NaN and no inf.  A 300-tune/300-draw
      PTDE fit of DC2018_128 (2 rungs x 36 chains, plus the whitening probe,
      the DE polish and the seed polish) never once entered the branch.
    * If it *were* reached it could only do harm.  q is NaN only when an input
      is already NaN, i.e. the raw vector itself carries a NaN -- and that raw
      variable's own N(0, 1) prior term already makes the total logp NaN, so
      the proposal is rejected whatever q does (verified: setting
      ``star.logmass_raw`` or ``planet.log_q_raw`` to NaN gives logp = nan).
      Substituting Q_MIN could therefore never rescue a sample; it only
      invented a mass ratio -- with a *zero* gradient, since nan_to_num is a
      switch -- in place of the one quantity that would have named the
      failure.

    So a NaN now propagates to logp, which is the sampler's own reject signal,
    and no gradient is poisoned.  A finite-but-out-of-range q is a different
    thing entirely and is still clipped, because that is a genuine modelling
    decision about where the backends are defined.
    """
    return pt.clip(q, Q_MIN, Q_MAX)


def clip_q_value(q, label="lens.q"):
    """Numeric counterpart of :func:`clip_q` for the backend Op and bootstrap
    paths, which see concrete floats rather than tensors.

    Raises ``ValueError`` on a NaN ``q`` instead of handing it to
    MulensModel/VBMicrolensing.  Every caller already has an error path that
    turns this into the right thing: the magnification Ops catch it and return
    NaN magnifications (logp = -inf, proposal rejected) after warning once with
    this message, and the flux bootstrap catches it, warns, and falls back to
    the PSPL columns.  The point is that the message names the parameter --
    ``np.clip(nan, ...)`` returned nan and let MulensModel report the generic
    "Wrong number of solutions to the lens equation" three frames away.

    The infinities are NOT an error: they carry a sign, so they are ordinary
    out-of-range values and are clipped to the bound exactly as np.clip -- and
    as the symbolic clip_q -- does.  NaN is the case with no value at all.
    """
    q = float(q)
    if np.isnan(q):
        raise ValueError(
            f"{label} is nan: a mass ratio must be a number.  {_Q_NAN_ADVICE}"
        )
    return float(np.clip(q, Q_MIN, Q_MAX))


# --- Trajectory (PSPL) parameter floors -------------------------------------
# The three RANGE decisions Lens._get_safe_mm_params makes on the single-source
# trajectory parameters before handing them to a magnification backend.  Like
# Q_MIN/Q_MAX above, each is a statement about where the model is DEFINED, and
# each must stand alone: none of them may be paired with a NaN substitution.
#
# T_E_FLOOR  every backend divides by t_E, and t_E <= 0 is not a timescale.
#            The same 1e-4 d is applied on the numeric Op path
#            (op._base_mm_params), so the two paths agree exactly.
# U_0_FLOOR  A = (u^2+2)/(u*sqrt(u^2+4)) diverges as u -> 0, so the peak
#            magnification of an exactly-central trajectory is infinite.  The
#            floor is applied to |u_0| with the sign kept, which keeps the
#            u_0 -> -u_0 reflection exact (a real degeneracy: ob140939 has four
#            Yee+2015 basins that differ by a sign flip).  It is applied by
#            apply_u_0_floor / floor_u_0_value below and NOWHERE else -- there
#            is one number and one expression, on every path.
#
#            It used to be TWO numbers: 1e-6 here and a hard-coded 1e-9 in
#            op._base_mm_params (and a third copy, also 1e-9, in the flux
#            bootstrap), so a fit visiting 1e-9 <= |u_0| < 1e-6 got a different
#            answer depending on which backend it was on.  Unified at the
#            looser 1e-9: the floor is a validity limit, not a preference, so
#            the model should be clamped as little as the arithmetic allows,
#            and 1e-9 costs nothing.  A(u) -> 1/u as u -> 0, so A(1e-9) = 1e9:
#            finite in float64 with the FULL 16 digits intact, because the
#            only term lost is the u^2 in (u^2 + 2), whose relative weight is
#            5e-19 -- three orders below eps = 2.2e-16 -- and every operation
#            in (u^2+2)/(u*sqrt(u^2+4)) is a product or a sum of positives, so
#            there is no cancellation to be catastrophic.  Measured:
#            A(U_0_FLOOR) equals 1/U_0_FLOOR exactly, to the bit.  Downstream
#            the flux model f_s*A + f_b is linear in A and the Gaussian logp
#            quadratic, so overflow needs f_s > 1e145 (F^2 < 1.8e308); a 1%
#            error bar at A = 1e9 gives chi2 = 1e22, huge but finite, which is
#            the sampler being pushed off the singularity as intended.
#            No shipped example is anywhere near the floor: |u_0| ~ 0.5
#            (ob08092), 0.14 (DC2018_128), 0.029 (KMT-2019-BLG-1806).
# THETA_E_LENSING_MIN
#            pi_E = pi_rel/theta_E, so as theta_E -> 0 the parallax vector
#            diverges while the event itself stops being a lensing event at
#            all.  Below this the trajectory is evaluated WITHOUT parallax
#            (pi_E_N = pi_E_E = 0) rather than with a diverging one; the
#            source_behind_lens / theta_E_singularity soft bounds in
#            Lens.build_likelihood are what actually push the sampler out.
#            It is a comparison, and a comparison against NaN is False, so
#            this branch never needed a NaN substitution to begin with.
T_E_FLOOR = 1e-4  # days
U_0_FLOOR = 1e-9  # Einstein radii
THETA_E_LENSING_MIN = 1e-6  # mas

# --- Binary/finite-source floors --------------------------------------------
# The same rule as the three above -- a statement about where the model is
# DEFINED -- for the two parameters that only exist once the lens is binary or
# the source is resolved.  They lived as five hard-coded literals (s three
# times, rho twice) rather than here, which is how U_0_FLOOR came to disagree
# with itself across backends; naming them is what stops that recurring.
#
# S_FLOOR    the binary-lens equation places the components at +/- s/2, so
#            s <= 0 puts the secondary on the wrong side of the primary (or on
#            top of it) and the caustic topology is undefined.  s -> 0 is also
#            the close-binary limit where the central caustic shrinks as s^2
#            and VBM's contour integration needs ever finer sampling, so a
#            floor is a numerical necessity as well as a physical one.
# RHO_FLOOR  the source radius in Einstein radii.  rho <= 0 is unphysical and
#            the finite-source methods integrate over the source disc, so a
#            non-positive radius is not merely small but meaningless.  It is
#            floored rather than clipped away because rho -> 0 IS the correct
#            point-source limit -- the floor only has to keep the integration
#            well-posed, and 1e-9 is far below any resolvable source.
#
# Both are the values that were already in force everywhere they appeared, so
# naming them changed no result; see the PR that introduced them.
S_FLOOR = 1e-6  # Einstein radii (binary separation)
RHO_FLOOR = 1e-9  # Einstein radii (source radius)


def apply_u_0_floor(u_0):
    """Move u_0 out of the open interval (-U_0_FLOOR, U_0_FLOOR) -- symbolic.

    Written as a nearest-endpoint clip::

        u_0 < 0  ->  min(u_0, -U_0_FLOOR)
        u_0 >= 0 ->  max(u_0, +U_0_FLOOR)

    which is exactly ``sign(u_0) * max(|u_0|, U_0_FLOOR)`` everywhere except at
    u_0 = 0, and that exception is the point.  ``sign(0) = 0``, so the old
    spelling returned ``0 * U_0_FLOOR = 0``: the floor did not engage at the
    one value it exists to protect, and the peak magnification of an exactly
    central trajectory stayed infinite.  ``u_0: 0`` is a perfectly plausible
    seed for a high-magnification event, so this was reachable by typing a
    round number.

    **Zero goes to +U_0_FLOOR.**  The sign of a central crossing is genuinely
    undefined -- the trajectory passes through the lens, there is no side --
    and the two branches are physically the same event under the exact
    reflection ``(u_0, pi_E_N, pi_E_E) -> (-u_0, -pi_E_N, -pi_E_E)``, so either
    choice is defensible and what matters is that it is finite, deterministic
    and written down.  Positive is chosen because it makes this map
    monotonically non-decreasing (the interval's two endpoints are the only
    candidates; 0 is equidistant, and ties break upward, the ordinary
    round-half-up convention), because u_0 > 0 is the convention a PSPL
    solution is quoted in when nothing breaks the degeneracy, and because it
    keeps the two IEEE zeros together: ``-0.0 < 0`` is False, so +0.0 and -0.0
    both map to +U_0_FLOOR rather than to opposite endpoints.

    Gradient: piecewise constant inside the gap, the identity outside -- the
    same derivative the sign/abs spelling had (``sign(u)**2 = 1``), so no fit
    that stays outside the floor feels this at all.  Both branches of the
    symbolic switch are finite for every input, including +/-inf, so there is
    no NaN hiding in the unselected branch (the JAX where-trap).  A NaN input
    still propagates: ``nan < 0`` is False and ``maximum(nan, F)`` is nan,
    which is what PR #142 wants -- the floor is a range decision and must never
    double as a NaN substitution.
    """
    return pt.where(
        pt.lt(u_0, 0.0),
        pt.minimum(u_0, -U_0_FLOOR),
        pt.maximum(u_0, U_0_FLOOR),
    )


def floor_u_0_value(u_0):
    """Numeric counterpart of :func:`apply_u_0_floor` -- same number, same
    expression, same treatment of exactly zero.  See that function for why.

    Used by the MulensModel/VBM Op path (``op._base_mm_params``) and by the
    flux bootstrap (``MulensInstrument``), which had their own hard-coded
    copies of the clip until the floors were unified.
    """
    v = float(u_0)
    return float(min(v, -U_0_FLOOR) if v < 0.0 else max(v, U_0_FLOOR))


_MM_NAN_ADVICE = (
    "The trajectory parameters are derived from the sampled coordinates: "
    "t_E and pi_E_N/pi_E_E from theta_E and the relative proper motion, "
    "theta_E from the lens mass and pi_rel, pi_rel from the two "
    "star.<...>.distance values.  Check the initval (and any 'sigma: 0' "
    "link expression) on star.<lens>.logmass, star.<lens>.distance, "
    "star.<source>.distance and the star.<...>.pm_ra/pm_dec pair, plus "
    "lens.<...>.t_0 and lens.<...>.u_0, which are sampled directly."
)


def require_mm_number(value, label):
    """Numeric guard for a trajectory parameter on its way to a backend.

    The counterpart of :func:`clip_q_value` for the five quantities
    ``Lens._get_safe_mm_params`` handles, and it exists for the same reason:
    ``pt.nan_to_num`` used to replace a NaN t_E with 100 d, a NaN u_0 with 1,
    and a NaN theta_E/pi_E_N/pi_E_E with 0 -- a complete, fabricated PSPL model
    in place of the one quantity that would have named the failure.  That scrub
    is gone (see ``Lens._get_safe_mm_params``); on the symbolic path a NaN now
    reaches logp, which is the sampler's own reject signal, and on this numeric
    path it raises with a message that says which parameter it was.

    Every caller already has the error path that turns the raise into the right
    thing: ``_MagOpBase.perform`` catches ValueError, warns once, and returns
    NaN magnifications -- logp = -inf, proposal rejected, exactly what a NaN
    handed to MulensModel produced anyway, only three frames further away and
    under a generic message.

    The infinities are NOT an error, the same split :func:`clip_q_value` makes:
    an infinity carries a sign and is an ordinary out-of-range value, which the
    caller's own floor then handles.  NaN is the case with no value at all.
    """
    v = float(value)
    if np.isnan(v):
        raise ValueError(
            f"{label} is nan: a trajectory parameter must be a number.  "
            f"{_MM_NAN_ADVICE}"
        )
    return v


@register_physics
def calc_mlens_total(*masses):
    # Total lens mass: sum over all lens bodies.  theta_E, t_E, rho, and pi_E
    # are referenced to the TOTAL mass for multi-body lenses (community
    # convention for binary-lens parameters).
    total = masses[0]
    for m in masses[1:]:
        total = total + m
    return total


@register_physics
def calc_f_source(log_f_total, q_source):
    return pt.power(10, log_f_total) * q_source


@register_physics
def calc_f_blend(log_f_total, q_source):
    return pt.power(10, log_f_total) * (1.0 - q_source)


@register_physics
def calc_zeropoint(m_source_pred, zp_center, sed_constrained, f_source):
    """Photometric zeropoint of one light curve's own flux system.

    ``zp = m_SED + 2.5*log10(f_source)`` -- the offset between the SED's
    calibrated source magnitude and the instrumental one, which is what the
    ``zeropoint`` Gaussian prior constrains (see
    ``MulensInstrument._build_sed_flux_constraint``).

    The flux floor mirrors the one the hand-built node carried: f_source's
    own lower bound is 0, and log10(0) is -inf.

    ``sed_constrained`` is a 0/1 constant, zero for a light curve with no
    band reference or whose band filter is missing from the SED's BC grid.
    Such an element has no SED prediction to tie to, so it reports
    ``zp_center`` (the resolved prior center) and its Gaussian penalty is
    then exactly zero -- the same "no constraint" the hand-built loop
    expressed by skipping the element.  It is a ``switch`` on an explicit
    0/1 mask, not a NaN test: switch's gradient multiplies the unselected
    branch by zero, and 0 * NaN is NaN.
    """
    zp = m_source_pred + 2.5 * pt.log10(pt.maximum(f_source, 1e-30))
    return pt.switch(sed_constrained, zp, zp_center)


@register_physics
def calc_rho(radius, distance, theta_E):
    theta_star_mas = (radius * RSUN_TO_AU / distance) * 1000.0
    theta_E_safe = pt.maximum(pt.nan_to_num(theta_E, nan=0.0), 1e-10)
    return theta_star_mas / theta_E_safe


@register_physics
def calc_alpha(xalpha, yalpha):
    return pt.arctan2(yalpha, xalpha)


@register_physics
def calc_s(log_s):
    # Projected binary separation from the sampled log10(s).  Sampling log_s
    # makes close/wide an exact reflection log_s -> -log_s (|J| = 1); see
    # notes/multimode_implementation.txt P2.
    return pt.power(10.0, log_s)
