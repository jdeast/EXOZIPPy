from collections import namedtuple

import numpy as np
import pytensor.tensor as pt
from exoplanet_core.pymc import ops

from ...constants import TWOPI
from ...physics_registry import register_physics


def solve_kepler(M, ecc, circular=False):
    """``(sin f, cos f)`` from the mean anomaly -- the ONE Kepler solve.

    `circular=True` skips the Newton iteration entirely and returns
    `(sin M, cos M)`, which for `e = 0` is not an approximation: the
    eccentric and true anomalies both equal the mean anomaly there (review
    6.8.2).  A circular fit is a common one -- `secosw`/`sesinw` pinned at 0,
    as `examples/kelt17` writes it -- and it was paying an exoplanet-core
    Newton solve per epoch, per orbit, for an answer available from one sine
    and one cosine.

    `circular` is a STRUCTURAL claim the caller makes, never a runtime test,
    and both halves matter.  Structural, because the two branches must not
    both be built: a `pt.switch` would evaluate the Newton solve anyway
    (which is the whole cost) and hand `where`'s VJP a branch to multiply by
    zero.  A claim about the PIN and not about the value, because an
    eccentricity that merely starts at zero can move, and a fit whose e is
    free would then be solved as if it were circular for the whole run --
    silently, and with a wrong RV phase.  `Orbit.circular_orbits` is the one
    place that judgement is made.

    The `ecc + pt.zeros_like(M)` broadcast in the eccentric branch is
    load-bearing and pre-existing: exoplanet-core's op wants the two arrays
    the same shape.
    """
    if circular:
        return pt.sin(M), pt.cos(M)
    return ops.kepler(M, ecc + pt.zeros_like(M))


@register_physics
def calc_period(logP):
    return 10**logP


@register_physics
def calc_group_mass(*group_masses):
    """Total mass of a body group: sum of the per-component-type weighted
    sums injected by Orbit.add_parameter (one term per component type)."""
    total = group_masses[0]
    for m in group_masses[1:]:
        total = total + m
    return total


@register_physics
def calc_n(period):
    return TWOPI / period


# Hard ceiling on the eccentricity handed to the forward model.  A Kepler
# solve is meaningless at e >= 1, and calc_K's sqrt(1 - e^2) / calc_tp's
# sqrt(1 - e) are NaN there, so calc_ecc clips.  The soft bound that is
# supposed to keep the sampler out of that region must NOT be applied to the
# clipped node -- see Orbit._add_eccentricity_bound.
MAX_ECC = 0.9999


# Floor on the radicand of sqrt(e) (review 1.8.2).  An exactly circular
# seed -- `secosw: 0, sesinw: 0` in a params file, which is how a user spells
# "start circular" -- makes d(sqrt e)/de infinite, and `de/d(secosw) = 2 secosw`
# is exactly zero there, so pytensor's chain rule multiplies inf by 0 and the
# START GRADIENT is NaN with nothing naming the cause.  The floor goes on the
# RADICAND and never on the result, the house rule (calc_theta_E, calc_jitter,
# _vcve_quadratic): clamping after the root multiplies sqrt'(0) = inf by
# pt.maximum's zero gradient on the clamped side, which is the same NaN again.
# 1e-30 leaves sqrt(e) = 1e-15, a quantization far below any eccentricity a fit
# can distinguish, and it is inert for every e above it -- pt.maximum returns
# its argument bit-for-bit, so no non-circular fit moves.
ECC_FLOOR = 1e-30


# Floor on the chord's radicand (review 1.8.5) -- ECC_FLOOR's sibling, and the
# SAME BUG one function over.  1.8.2's comment above states the house rule as
# "floor the radicand, never the result", and the chord code followed that in
# letter while flooring the radicand AT ZERO -- which reintroduces the exact
# product the sentence names, because `pt.maximum(x, 0.0)` on the clamped side
# has gradient zero and `sqrt'(0)` is infinite.  So the rule needs its second
# half, and this constant is it: THE FLOOR MUST BE STRICTLY POSITIVE.
#
# Measured cost of the old 0.0 (2026-08-27, 7.13.1's transit leg): `d cos i /
# d chord` is NaN for every chord past `1 + p`, which is ~46% of the sampled
# support, and one NaN poisons the whole gradient VECTOR -- so the soft bound
# in `Orbit._add_chord_terms`, which exists precisely to push the chain back,
# had no finite total gradient to act through.  Two of four chains sat at the
# denormal minimum step size for 2000 draws without moving; `planet.b` came
# back ESS 8.0 / Rhat 1.43 while `orbit.period` in the same fit had ESS 3414.
# That is the failure's signature: the run completes, most parameters look
# healthy, and only the inclination-dependent ones are destroyed.
#
# WHY 1e-30 AND NOT THE 1e-12 THE REVIEW SUGGESTED.  Any positive floor fixes
# the gradient, but 1e-12 leaves b = 1e-6, which rises ABOVE the independent
# `log(pt.maximum(b, 1e-12))` floor inside `chord_log_jacobian` and moves that
# term by 13.8 nats in the forbidden region -- a silent change to the penalty
# landscape.  1e-30 leaves b = 1e-15, still under that floor, so the Jacobian's
# VALUE is unchanged and only its gradient becomes finite.  Verified inert for
# real geometry: bit-identical to the 0.0 floor at chord = 0, 0.5, 1.0, 1.0759.
CHORD_RADICAND_FLOOR = 1e-30


#: Field-by-field meaning documented in `state_vector_terms`.
StateVectorTerms = namedtuple(
    "StateVectorTerms",
    "sinf cosf r_over_a sinwf coswf vx_phase vz_phase ecc_factor",
)


def state_vector_terms(t, tp, n, ecc, sinw=None, cosw=None, circular=False):
    """The shared Kepler-to-state kernel (review 4.8.2).

    Every element-to-observable projection in the tree -- `get_true_anomaly`,
    `get_sky_position` / `state_vectors`, `get_radial_velocity`, the transit
    evaluators' (b, Z) geometry, and `ltt.line_of_sight_kinematics` -- is a
    composition of these terms with its own amplitude and orientation
    factors.  One Kepler solve, one set of angle identities, one place an
    N-body backend has to replace (review 8.8.15).

    Inputs are raw broadcastable tensors, already indexed/broadcast by the
    caller (`Orbit.state_vectors` does the `[orbit_map][None, :]` dance; the
    transit evaluators hand in their own retarded-time grids).  `circular`
    is the caller's STRUCTURAL claim, forwarded to `solve_kepler` (review
    6.8.2).  With `sinw`/`cosw` omitted (`get_true_anomaly`'s case) the
    omega-dependent terms are returned as None.

    Returns a `StateVectorTerms` named tuple:

      sinf, cosf   -- sin/cos of the true anomaly
      r_over_a     -- (1 - e^2) / (1 + e cos f), separation in units of a
      sinwf, coswf -- sin/cos(omega + f)
      vx_phase     -- -(sinwf + e sinw):  d/dt[r_over_a * coswf] = k * vx_phase
      vz_phase     --  (coswf + e cosw):  d/dt[r_over_a * sinwf] = k * vz_phase
                      with k = n / sqrt(1 - e^2); so a velocity component is
                      (n * a / ecc_factor) times its phase term.  Named for
                      the axes they feed in the zero-node edge-on frame.
      ecc_factor   -- sqrt(1 - e^2)
    """
    M = (t - tp) * n
    sinf, cosf = solve_kepler(M, ecc, circular=circular)
    r_over_a = (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)
    if sinw is None:
        sinwf = coswf = vx_phase = vz_phase = None
    else:
        coswf = cosw * cosf - sinw * sinf
        sinwf = sinw * cosf + cosw * sinf
        vx_phase = -(sinwf + ecc * sinw)
        vz_phase = coswf + ecc * cosw
    ecc_factor = pt.sqrt(1.0 - pt.sqr(ecc))
    return StateVectorTerms(
        sinf, cosf, r_over_a, sinwf, coswf, vx_phase, vz_phase, ecc_factor
    )


def thiele_innes_xy(r, coswf, sinwf, cosi, bigomega):
    """Sky-plane position (X = North, Y = East) of an orbiting body from the
    in-plane factors -- the Thiele-Innes projection, hoisted out of
    `Orbit.state_vectors` so the microlensing keplerian mode (which needs
    the same projection per epoch, in Einstein units) consumes ONE owner
    rather than a second copy of these two lines.

    `r` is the separation in the caller's amplitude units;
    `coswf`/`sinwf` come from `state_vector_terms`.  PA measured East of
    North: at omega + f = 0 (ascending node) the body sits at
    PA = bigomega.
    """
    cosO = pt.cos(bigomega)
    sinO = pt.sin(bigomega)
    X = r * (cosO * coswf - sinO * sinwf * cosi)
    Y = r * (sinO * coswf + cosO * sinwf * cosi)
    return X, Y


def _sqrt_ecc(ecc):
    """``sqrt(e)`` with the radicand floored -- see ECC_FLOOR."""
    return pt.sqrt(pt.maximum(ecc, ECC_FLOOR))


def _circular_bias(ecc_raw):
    """``1.0`` where the sqrt(e) pair is EXACTLY zero, ``0.0`` elsewhere.

    An ADDITIVE nudge, deliberately, rather than a `pt.switch` selecting
    between an angle and a convention.  A switch is the where-trap's exact
    shape: `arctan2(0, 0)` has a NaN gradient, the switch keeps it as the
    UNSELECTED branch, and `where`'s VJP multiplies that branch by zero.
    Measured, and worth stating precisely: on this PyTensor and the C
    backend the two switch-guarded angles (`calc_omega`, `calc_lam_from_sv`)
    do NOT in fact produce a NaN -- a rewrite gets there first -- so unlike
    `calc_tp` and the linear e vector they were not a live bug.  They are
    written this way anyway, because the guard must not depend on a rewrite
    surviving a backend change, and because biasing the arctan2 ARGUMENTS
    costs nothing: one branch, no unselected one to poison anything, exactly
    inert away from the origin (`x + 0.0` is bit-identical) and at the origin
    it reproduces the convention it replaces.
    """
    return pt.switch(pt.eq(ecc_raw, 0.0), 1.0, 0.0)


def ecc_from_sqrte(secosw, sesinw):
    """Unclipped eccentricity, secosw^2 + sesinw^2.

    This is what a soft bound must see: the clipped calc_ecc below is flat
    above MAX_ECC, and a flat penalty has no gradient for NUTS to follow.
    """
    return pt.sqr(sesinw) + pt.sqr(secosw)


@register_physics
def calc_ecc(secosw, sesinw):
    # Clipped for the FORWARD MODEL only (see MAX_ECC above).  The lower clip
    # never binds -- a sum of squares is already >= 0 -- but is kept so a
    # future parameterization cannot sneak a negative e into the Kepler solve.
    return pt.clip(ecc_from_sqrte(secosw, sesinw), 0.0, MAX_ECC)


@register_physics
def calc_omega(secosw, sesinw):
    """Argument of periastron from the sqrt(e) pair.

    At exactly zero the angle is undefined and the convention is omega = 90
    deg, so the RV phase stays perfectly aligned.  The convention is applied
    by biasing the arctan2 argument rather than by switching on the answer --
    `arctan2(1, 0)` IS pi/2 -- so that `arctan2(0, 0)` never appears as an
    unselected branch for `where`'s VJP to multiply by zero (see
    _circular_bias, review 1.8.2, including the measurement that says this
    one was hardening rather than a live bug).  Every non-circular value is
    bit-identical: the bias is exactly 0.0 there.
    """
    e_raw = pt.sqr(sesinw) + pt.sqr(secosw)
    return pt.arctan2(sesinw + _circular_bias(e_raw), secosw)


# ----------------------------------------------------------------------
# The V_c/V_e eccentricity parameterization (Eastman 2024, PASP,
# arXiv:2309.14410).
#
# V_c/V_e is the ratio of the speed a planet WOULD have on a circular orbit of
# the same period to its actual speed at transit, and the transit duration
# constrains it far better than it constrains (e, omega) separately -- which is
# why a transit-only fit in sqrt(e)cos/sin(omega) recovers eccentricities that
# are, in the paper's measurement over 330 simulated systems, simply wrong.
#
#   V_c/V_e = sqrt(1 - e^2) / (1 + e sin omega)                        (eq 4)
#
# Inverting for e gives a QUADRATIC (eq 5; EXOFASTv2's vcve2e.pro):
#
#   A e^2 + B e + C = 0,  A = 1 + x^2 sin^2 w,  B = 2 x^2 sin w,  C = x^2 - 1
#   B^2 - 4AC = 4 (1 - x^2 cos^2 w)          (so the half-discriminant is neat)
#   e_hi, e_lo = (-B +/- sqrt(B^2 - 4AC)) / (2A),  and e_hi > e_lo since A > 0
#
# Both roots can be physical at once, and the paper picks one with a discrete
# sign parameter S.  That is fine for DE-MC and useless to a gradient sampler:
# S is piecewise constant (no gradient) and the logp jumps across S = 1.  Here
# BOTH roots are built and the likelihood is marginalized over them
# (System._add_vcve_branch_mixture), so nothing discrete enters the sampler.
# The functions below therefore come in pairs, and each is shielded so that
# every (vcve, omega) -- including combinations where NO real root exists --
# yields a finite value with a finite gradient.
# ----------------------------------------------------------------------


def _vcve_quadratic(vcve, omega):
    """``(A, B, sqrt(B^2 - 4AC))`` for the eccentricity quadratic.

    The discriminant is returned WHOLE, not halved: ``B^2 - 4AC`` simplifies to
    ``4 (1 - x^2 cos^2 w)``, so the square root is ``2 sqrt(1 - x^2 cos^2 w)``
    and the roots are ``(-B +/- that) / 2A``.  Dropping the factor of 2 gives
    eccentricities that look plausible and are wrong -- caught here by
    round-tripping the inversion against the forward relation, which is why
    ``tests/test_vcve.py`` does that over a grid rather than trusting the
    algebra.

    The square root's argument is floored at zero (the HARD shield): outside
    the real region the two roots coincide at -B/2A instead of being NaN.
    Flooring the RADICAND rather than the result is the house rule
    (calc_theta_E, calc_jitter): ``sqrt'(0)`` is infinite, so clamping after
    the root multiplies that infinity by ``pt.maximum``'s zero gradient and
    gives NaN -- the exact failure the shields exist to prevent.  The soft
    shield that keeps the sampler out of the region lives in
    ``Orbit._add_vcve_shield``, on the unfloored quantity, where it has a
    gradient to offer.
    """
    x2 = pt.sqr(vcve)
    sinw, cosw = pt.sin(omega), pt.cos(omega)
    a = 1.0 + x2 * pt.sqr(sinw)
    b = 2.0 * x2 * sinw
    root = 2.0 * pt.sqrt(pt.maximum(1.0 - x2 * pt.sqr(cosw), 0.0))
    return a, b, root


def vcve_discriminant(vcve, omega):
    """``1 - (V_c/V_e)^2 cos^2 omega``: negative where no real e exists.

    Unfloored on purpose -- this is what a soft bound must see, for the same
    reason ``ecc_from_sqrte`` is unclipped: the shielded root above is flat
    across the whole imaginary region and a flat penalty has no gradient for
    NUTS to follow back out.
    """
    return 1.0 - pt.sqr(vcve) * pt.sqr(pt.cos(omega))


@register_physics
def calc_ecc_from_vcve(vcve, omega):
    """The UPPER root of the V_c/V_e quadratic -- the primary branch.

    Upper, not lower, and the reason is worth keeping.  The product of the
    roots is ``C/A = (x^2 - 1)/A``, so for ``x = V_c/V_e < 1`` the two roots
    have OPPOSITE signs and the physical one is always the upper: a "prefer the
    lower eccentricity" primary would sit clipped at zero over that whole
    region, with no gradient and a forward model pinned circular.  (EXOFASTv2
    does prefer the lower root, but only among the roots that are physical --
    vcve2e.pro tests ``eneg ge 0 and eneg lt 1`` first.  A fixed choice cannot
    reproduce that test without a switch, and the switch is what the mixture
    exists to remove.)  Where BOTH roots are physical -- ``x > 1`` with
    ``sin omega < 0`` -- this is the higher-eccentricity solution and the
    mixture carries the other one's likelihood, so the choice decides only
    which branch the trace reports as ``orbit.ecc``.
    """
    a, b, root = _vcve_quadratic(vcve, omega)
    return pt.clip((-b + root) / (2.0 * a), 0.0, MAX_ECC)


@register_physics
def calc_ecc_from_vcve_lo(vcve, omega):
    """The LOWER root of the V_c/V_e quadratic -- the alternate branch.

    Not reachable through a manifest ``expr_key``: nothing samples it and
    nothing derives from it.  ``Orbit`` builds it directly for the branch
    mixture, and registering it keeps the two roots' code in one place.
    """
    a, b, root = _vcve_quadratic(vcve, omega)
    return pt.clip((-b - root) / (2.0 * a), 0.0, MAX_ECC)


def ecc_from_vcve_unclipped(vcve, omega, upper=True):
    """A V_c/V_e root without the [0, MAX_ECC] clip.

    What ``Orbit._add_eccentricity_bound`` needs (see ``ecc_from_sqrte``, same
    argument): the clipped root is flat past the collision limit, so a barrier
    applied to it has nothing to push against.  Defaults to the upper root
    because that is the primary branch.  The discriminant is still shielded --
    an imaginary root has no value to bound at all, and its own soft shield
    covers that region.
    """
    a, b, root = _vcve_quadratic(vcve, omega)
    return (-b + root) / (2.0 * a) if upper else (-b - root) / (2.0 * a)


@register_physics
def calc_omega_from_xy(xomega, yomega):
    """omega from a direction vector, exactly as bigomega and alpha are.

    The paper samples ``L cos omega``/``L sin omega`` inside the unit disk and
    marginalizes over the magnitude ``L``; with the root chosen by
    marginalization rather than by ``|L|`` (which is what EXOFASTv2's fallback
    did), the magnitude carries no information at all, so this is the plain
    direction-vector trick already used for the ascending node and the
    microlensing trajectory angle: a free positive scale absorbed by the
    N(0, 1) priors, leaving a uniform marginal on the angle.
    """
    return pt.arctan2(yomega, xomega)


@register_physics
def calc_vcve(ecc, omega):
    """V_c/V_e from (e, omega) -- eq 4, the forward direction.

    Used to REPORT V_c/V_e on an orbit that samples sqrt(e)cos/sin(omega)
    (manifest role 3), and as the relaxation engine's bridge between the two
    parameterizations.  The denominator is floored: it vanishes only in the
    unreachable corner e -> 1 with sin(omega) -> -1, and a reported quantity
    must not be the thing that puts an inf in the trace.
    """
    denom = 1.0 + ecc * pt.sin(omega)
    return pt.sqrt(pt.maximum(1.0 - pt.sqr(ecc), 0.0)) / pt.maximum(
        denom, 1e-12
    )


def vcve_log_jacobian(ecc, omega):
    """``log|d(V_c/V_e)/de|`` at fixed omega.  Consumers SUBTRACT this.

    Differentiating eq 4::

        d(V_c/V_e)/de = -(e + sin w) / (sqrt(1 - e^2) (1 + e sin w)^2)

    and the log of its magnitude is the expression below.

    WHICH DIRECTION.  This function returns the derivative itself, because that
    is the quantity a finite difference can check; the prior correction is its
    NEGATIVE, and the distinction is the whole point of the term.  V_c/V_e is
    the SAMPLED coordinate and is uniform over its bounds, so the induced
    density on the derived eccentricity is ``p(e) = p(v) |dv/de| ~ |dv/de|`` --
    which grows without limit as e -> 1, and is exactly the "non-physical prior
    that strongly biases e toward high eccentricities" of the paper's section 3.
    Flattening it means dividing by that factor, i.e. ADDING
    ``log|de/d(V_c/V_e)| = -log|d(V_c/V_e)/de|`` to the log posterior.  Adding
    the derivative instead would DOUBLE the bias rather than remove it, and it
    is not a difference any check of the derivative's magnitude can see -- so
    the direction is pinned by measuring the implied prior on e for flatness
    (tests/test_vcve.py).  EXOFASTv2 applies a Jacobian for this same
    transformation; its sign convention is not restated here because it could
    not be checked from this tree.

    Every argument of a log is floored.  The interesting one is |e + sin w|,
    which vanishes on a real curve through the parameter space (e = -sin w):
    there the map v(e) is stationary -- it is the fold where the two roots of
    the inversion meet -- so a narrow band of V_c/V_e covers a wide range of e
    and the correction genuinely diverges.  It is an INTEGRABLE divergence (the
    density goes as |v - v_fold|^(-1/2)), so the floor is a cap on a real
    feature, not a patch over a wrong one; without it the term is +inf on that
    curve.  It is also not a sampler trap worth softening: |e + sin w| grows as
    sqrt(|v - v_fold|), so the reward exceeds 10 nats only within ~1e-9 of the
    fold in V_c/V_e -- a volume no step ever lands in, and one the trajectory
    passes straight through if it does.  Not registered as physics: it is a prior term, not a parameter's
    value, and it is applied as a potential in Orbit.build_likelihood.
    """
    sinw = pt.sin(omega)
    return (
        pt.log(pt.maximum(pt.abs(ecc + sinw), 1e-12))
        - 0.5 * pt.log(pt.maximum(1.0 - pt.sqr(ecc), 1e-12))
        - 2.0 * pt.log(pt.maximum(1.0 + ecc * sinw, 1e-12))
    )


@register_physics
def calc_esinw_from_ecc(ecc, omega):
    """e sin(omega) straight from (e, omega).

    Same quantity as `calc_esinw`, which reaches it as sqrt(e) * sesinw and so
    consumes the sqrt(e) pair -- unavailable to a V_c/V_e orbit, which REPORTS
    that pair (see calc_tp_from_ecc for the same argument at length).
    """
    return ecc * pt.sin(omega)


@register_physics
def calc_ecosw_from_ecc(ecc, omega):
    """e cos(omega) straight from (e, omega); see calc_esinw_from_ecc."""
    return ecc * pt.cos(omega)


@register_physics
def calc_secosw_from_ecc(ecc, omega):
    """sqrt(e) cos(omega) -- reported on a V_c/V_e orbit (manifest role 3)."""
    return pt.sqrt(pt.maximum(ecc, 0.0)) * pt.cos(omega)


@register_physics
def calc_sesinw_from_ecc(ecc, omega):
    """sqrt(e) sin(omega) -- reported on a V_c/V_e orbit (manifest role 3)."""
    return pt.sqrt(pt.maximum(ecc, 0.0)) * pt.sin(omega)


@register_physics
def calc_bigomega(xbigomega, ybigomega):
    # Longitude of the ascending node from its direction vector; the radius
    # is a free positive scale absorbed by the N(0,1) priors (same geometry
    # sampler trick as the microlensing trajectory angle alpha).
    return pt.arctan2(ybigomega, xbigomega)


@register_physics
def calc_sinw(omega):
    return pt.sin(omega)


@register_physics
def calc_cosw(omega):
    return pt.cos(omega)


@register_physics
def calc_esinw(ecc, sesinw):
    """e sin(omega) as sqrt(e) * sesinw.  Radicand floored -- see ECC_FLOOR."""
    return _sqrt_ecc(ecc) * sesinw


@register_physics
def calc_ecosw(ecc, secosw):
    """e cos(omega) as sqrt(e) * secosw.  Radicand floored -- see ECC_FLOOR."""
    return _sqrt_ecc(ecc) * secosw


@register_physics
def calc_inc(cosi):
    return pt.arccos(cosi)


@register_physics
def calc_sini(inc):
    return pt.sin(inc)


@register_physics
def calc_b(ar, cosi, ecc, esinw):
    # Primary-transit impact parameter, Winn 2010 eq 7: the transit is at
    # true anomaly pi/2 - omega, where r = a(1-e^2)/(1 + esinw).
    return ar * cosi * (1.0 - pt.sqr(ecc)) / (1.0 + esinw)


@register_physics
def calc_tp(ecc, sesinw, secosw, tc, n):
    """Time of periastron from the sqrt(e) pair (the Tc -> Tp inversion).

    `arctan(x/y) -> arctan2(x, y)`, with both arguments multiplied through by
    sqrt(e) so the sampled pair appears directly and the 1/sqrt(e) singularity
    of the naive form cancels.

    Two shields, both for review 1.8.2's exactly-circular seed, and neither
    changes any other start by a single bit:

    * `sqrt(e)` takes the floored radicand (ECC_FLOOR), because
      `d(sqrt e)/de = inf` times `de/d(secosw) = 0` is NaN.
    * both arctan2 arguments vanish together at e = 0 -- the pair IS the two
      arguments -- and `arctan2(0, 0)` has a NaN gradient.  `_circular_bias`
      pushes the x argument to 1 there, giving `E0 = 2 arctan2(0, 1) = 0` and
      so `tp = tc`.  That is the same orbit `calc_tp_from_ecc` describes at
      (e = 0, omega = pi/2), i.e. it agrees with `calc_omega`'s convention for
      the very configuration that made omega undefined.

    `calc_tp_from_ecc` remains the better-behaved form on general grounds (it
    cancels the sqrt(e) factor algebraically rather than flooring it) and is
    what a V_c/V_e orbit uses; this one is kept because the sqrt(e)cos/sin
    path's reported `tp` -- which differs from the other form by a whole
    period on part of the domain -- must not move for every shipped fit.
    """
    E0 = 2.0 * pt.arctan2(
        pt.sqrt(1.0 - ecc) * (_sqrt_ecc(ecc) - sesinw),
        pt.sqrt(1.0 + ecc) * secosw + _circular_bias(ecc),
    )
    M0 = E0 - ecc * pt.sin(E0)
    return tc - M0 / n


@register_physics
def calc_tp_from_ecc(ecc, omega, tc, n):
    """Time of periastron from (e, omega) instead of the sqrt(e) pair.

    Needed because a V_c/V_e orbit does not SAMPLE `secosw`/`sesinw` -- it
    reports them (manifest role 3), and a reported element is by definition
    consumed by nothing.  `calc_tp` above consumes them, so on a V_c/V_e orbit
    it would read their pre-patch placeholder: a silently wrong periastron.
    This is the same quantity written in the coordinates that orbit does have.

    It is also numerically better behaved, which is why the two are not merged.
    The true anomaly at conjunction is `f = pi/2 - omega`, and

        tan(E/2) = sqrt((1 - e)/(1 + e)) tan(f/2)

    so the sqrt(e) common factor that appears in `calc_tp`'s arctan2 arguments
    cancels identically -- removing both of that form's singularities at e = 0
    (an infinite d(sqrt e)/de, and arctan2(0, 0) when the pair is exactly zero;
    review 1.8.2).  `sin(f/2)` and `cos(f/2)` never vanish together, so this
    form is finite and differentiable everywhere in e and omega.

    The two agree up to a whole period in `tp` (arctan2 puts the doubled angle
    in a different revolution when `sin(f/2) < 0`), which the model is exactly
    invariant to -- every consumer sees `tp` only through `n (t - tp)` modulo
    2 pi.  `calc_tp` is left untouched so the sqrt(e)cos/sin path stays
    bit-identical.
    """
    half_f = 0.5 * (0.5 * np.pi - omega)
    E0 = 2.0 * pt.arctan2(
        pt.sqrt(pt.maximum(1.0 - ecc, 0.0)) * pt.sin(half_f),
        pt.sqrt(1.0 + ecc) * pt.cos(half_f),
    )
    M0 = E0 - ecc * pt.sin(E0)
    return tc - M0 / n


@register_physics
def calc_ts(ecc, omega, tc, period):
    """Time of the SECONDARY eclipse (occultation), review 8.8.7.

    The primary transit is at true anomaly `f_c = pi/2 - omega` and the
    occultation at the opposite conjunction `f_s = -pi/2 - omega`, so the two
    are separated by the difference of their mean anomalies::

        t_s - t_c = (M_s - M_c) P / 2 pi

    which is `P/2` exactly for a circular orbit and departs from it mainly
    with `e cos omega` -- the classical statement that the eclipse PHASE
    measures `e cos omega` (Winn 2010 eq 33; Charbonneau et al. 2005).  That
    is what makes `ts` an INFERENCE path and not just a table row: a user's
    Gaussian on this derived parameter constrains the eccentricity vector
    through exactly this expression, with a gradient.

    Written from the exact anomalies rather than the `(1 + 4 e cos omega/pi)
    P/2` series, because the series is only first order in e and the exact
    form costs the same two arctan2s.
    """
    return ts_from_ecc_omega(ecc, omega, tc, period, xp=pt)


def mean_anomaly_at_true_anomaly(ecc, true_anomaly, xp=pt):
    """Mean anomaly at a true anomaly, in radians.

    `tan(E/2) = sqrt((1-e)/(1+e)) tan(f/2)` then `M = E - e sin E`, written
    with arctan2 on the half-angle so nothing divides by a vanishing cosine.
    The same algebra `calc_tp_from_ecc` uses at `f = pi/2 - omega`; a helper
    because `calc_ts` needs it at two different anomalies.

    Backend-agnostic through `xp=` for the reason
    `planet.physics.contact_duration` is: review 8.8.7's seed solver must
    evaluate the same eclipse-timing model the likelihood does, from inside
    the relaxation engine, and a numpy transcription would be a second copy
    free to drift from this one.
    """
    half_f = 0.5 * true_anomaly
    e_anom = 2.0 * xp.arctan2(
        xp.sqrt(xp.maximum(1.0 - ecc, 0.0)) * xp.sin(half_f),
        xp.sqrt(1.0 + ecc) * xp.cos(half_f),
    )
    return e_anom - ecc * xp.sin(e_anom)


def ts_from_ecc_omega(ecc, omega, tc, period, xp=pt):
    """`calc_ts`'s body; see that docstring and mean_anomaly_at_true_anomaly.

    Both anomalies come out in `(-pi, pi]`, so their difference puts
    `t_s - t_c` in `(-P, P)`; the branch below reports the eclipse AFTER
    `tc`, the convention a reader expects and the one EXOFASTv2 uses.  It is
    a `where` whose two branches are `dt` and `dt + P`, both finite and both
    differentiable, so it is not the where-trap (that is about a branch which
    is itself NaN or infinite).  The step it carries is real and sits at
    `t_s = t_c`, reachable only as `e -> 1` at grazing geometry.
    """
    m_c = mean_anomaly_at_true_anomaly(ecc, 0.5 * np.pi - omega, xp=xp)
    m_s = mean_anomaly_at_true_anomaly(ecc, -0.5 * np.pi - omega, xp=xp)
    dt = (m_s - m_c) * period / TWOPI
    return tc + xp.where(dt < 0.0, dt + period, dt)


def mean_anomaly_at_conjunction(ecc, omega):
    """``M`` at primary conjunction, in radians -- pure numpy.

    The numpy twin of the Kepler algebra inside `calc_tp_from_ecc`: the true
    anomaly at conjunction is `f = pi/2 - omega`, so

        tan(E/2) = sqrt((1 - e)/(1 + e)) tan(f/2),   M = E - e sin E

    and `tp = tc - M/n` is exactly that function.  Written in numpy because
    its consumers are the RELAXATION ENGINE and the stage-3 tc window, neither
    of which has a tensor graph -- and written as its own function, rather than
    inline at either call site, so the two cannot drift apart from each other
    or from the pytensor form.  `tests/test_tp_seed.py` pins all three against
    one another.

    Vectorized: every argument may be an array.
    """
    half_f = 0.5 * (0.5 * np.pi - np.asarray(omega, dtype=float))
    ecc = np.asarray(ecc, dtype=float)
    E0 = 2.0 * np.arctan2(
        np.sqrt(np.maximum(1.0 - ecc, 0.0)) * np.sin(half_f),
        np.sqrt(1.0 + ecc) * np.cos(half_f),
    )
    return E0 - ecc * np.sin(E0)


def tc_from_tp(tp, ecc, omega, period):
    """Time of conjunction implied by a time of PERIASTRON (review 8.1.1).

    `calc_tp` and `calc_tp_from_ecc` both compute `tp = tc - M/n`; this is
    that equation read the other way, `tc = tp + M P / 2 pi`.  The inversion
    is CLOSED FORM and needs no Newton iteration, because the conjunction is
    defined by a true anomaly (`f = pi/2 - omega`) rather than by a time: it
    is the mapping `M -> E` that is transcendental, and this direction never
    needs it.  That is also why this is a standalone SOLVER for the engine
    rather than a sympy relation -- see orbit/symbolic_physics.py.

    `omega` is in RADIANS and `tp`/`period` in days, i.e. the internal units
    the engine's `resolved` dict holds.
    """
    m_c = mean_anomaly_at_conjunction(ecc, omega)
    return np.asarray(tp, dtype=float) + m_c * np.asarray(
        period, dtype=float
    ) / (2.0 * np.pi)


# ----------------------------------------------------------------------
# The TRANSIT CHORD parameterization (Eastman 2024, arXiv:2309.14410).
#
# The other half of the pair whose eccentricity half is the V_c/V_e block
# above, and it exists for the same reason: a transit measures a DURATION,
#
#     T ~ (P / pi) (1 / a_R) * chord * (V_c/V_e) / sin i,
#
# so the chord -- the sky-projected path of the planet across the stellar
# disc, `sqrt((1 + p)^2 - b^2)` in units of R_* -- and V_c/V_e are the two
# coordinates the data actually constrain.  Sampling cos i instead spends the
# sampler's effort on a direction the duration only reaches through
# `b = kappa cos i`, whose scale factor kappa is itself a fitted quantity.
#
# The inversion is a square root rather than a quadratic, so unlike the
# eccentricity half there is no second branch and no mixture: chord >= 0 and
# b >= 0 give exactly one geometry, and the sign of cos i is not a root choice
# at all but the i <-> 180 - i convention the orbit already carries (i180).
# What IS shared is the shield discipline -- the radicand is floored inside
# the sqrt, at a STRICTLY POSITIVE floor (CHORD_RADICAND_FLOOR; flooring at
# 0.0 is what made a NaN not merely buildable but guaranteed, review 1.8.5),
# while the UNFLOORED radicand drives the soft bound in
# Orbit._add_chord_terms, because the floored one is flat over the whole
# non-transiting region and a flat penalty has no gradient.
# ----------------------------------------------------------------------


def chord_kappa(ar, ecc, esinw):
    """``b / cos i`` -- the scale factor of Winn 2010 eq 7.

    Written from the same three quantities `calc_b` uses, and deliberately
    not from `calc_b` itself: this is needed where cos i is what we are
    solving FOR, so it cannot be reached through a function that consumes it.
    The denominator is floored: `1 + e sin(omega)` vanishes only at e = 1 with
    omega = -90 deg, which MAX_ECC already excludes, but a NaN there would
    poison the gradient of every orbit sharing the vector.
    """
    return ar * (1.0 - pt.sqr(ecc)) / pt.maximum(1.0 + esinw, 1e-12)


def chord_radicand(chord, p):
    """``(1 + p)^2 - chord^2``, UNFLOORED: negative where no geometry exists.

    `b^2` in disguise.  Kept unfloored for the same reason
    `vcve_discriminant` is: this is what the soft bound must see, since the
    floored version is flat across the entire forbidden region.  A chord
    longer than the stellar diameter plus the planet is not a grazing transit
    or a miss -- it is no transit at all, and the barrier's job is to say so
    with a gradient rather than with a wall.
    """
    return pt.sqr(1.0 + p) - pt.sqr(chord)


@register_physics
def calc_cosi_from_chord(chord, p, ar, ecc, esinw, chord_sign):
    """cos i from the sampled chord (the shielded inverse of `calc_chord`).

    `b = sqrt((1 + p)^2 - chord^2)` and `cos i = b / kappa`.  The radicand is
    floored at `CHORD_RADICAND_FLOOR` -- the HARD shield -- so a chord past
    `1 + p` gives b = 1e-15 (a central transit, to fifteen digits) rather than
    NaN, and the soft bound supplies the restoring force.  Flooring the
    RADICAND and not the result is the house rule (calc_theta_E, calc_jitter,
    _vcve_quadratic), and the floor must be STRICTLY POSITIVE: `sqrt'(0)` is
    infinite, and `pt.maximum`'s zero gradient on the clamped side turns that
    into `0 * inf = NaN`.  This line floored at 0.0 until 2026-08-27 and did
    exactly that across ~46% of the chord's support -- see
    CHORD_RADICAND_FLOOR for the measured consequence (review 1.8.5).

    `chord_sign` is +1 or -1, injected per orbit by `Orbit.add_parameter`.
    The chord is even in cos i -- a transit at i and at 180 - i are the same
    transit -- so the parameterization cannot recover the sign, and it is not
    trying to: the sign is the orbit's existing `i180:` convention, which is
    what `cosi`'s own `lower` bound encodes when cos i is sampled.  Making it
    a context node rather than a sampled quantity is the point; it is a
    discrete label, and sampling it would be exactly the piecewise-constant
    coordinate the V_c/V_e half went out of its way to avoid.
    """
    b = pt.sqrt(pt.maximum(chord_radicand(chord, p), CHORD_RADICAND_FLOOR))
    return chord_sign * b / pt.maximum(chord_kappa(ar, ecc, esinw), 1e-12)


@register_physics
def calc_chord_from_cosi(cosi, p, ar, ecc, esinw):
    """The chord of the transit an orbit sampling cos i implies.

    The forward direction, used to REPORT the chord (manifest role 3) on an
    orbit that does not sample it, so both parameterizations produce the same
    table rows and a params file survives flipping `fitchord`.  `|cos i|`
    because the chord is even in it, and the radicand is floored for the same
    reason as above: a non-transiting geometry (b > 1 + p) reports chord ~ 0,
    which is what a reader should see, rather than NaN.  Floored positively
    like its inverse even though this is a reporting path: the two must not
    drift, and a reported Deterministic is one refactor away from a gradient.
    """
    b = chord_kappa(ar, ecc, esinw) * pt.abs(cosi)
    return pt.sqrt(
        pt.maximum(pt.sqr(1.0 + p) - pt.sqr(b), CHORD_RADICAND_FLOOR)
    )


def chord_log_jacobian(chord, p, ar, ecc, esinw):
    """``log|d(chord)/d(cos i)|`` at fixed everything else.  SUBTRACT this.

    Same contract, and the same trap, as `vcve_log_jacobian`: the function
    returns the honest derivative because that is what a finite difference can
    check, and the PRIOR correction is its negative.  The chord is the sampled
    coordinate and is uniform over its bounds, so the density it induces on
    the derived cos i is `p(cos i) = |d(chord)/d(cos i)|`; flattening that back
    to the isotropic uniform-in-cos-i prior -- which is what sampling cos i
    directly gives, and what this parameterization must not silently change --
    means ADDING `log|d(cos i)/d(chord)|`, i.e. subtracting this.

    With `chord^2 = (1 + p)^2 - kappa^2 cos^2 i`,

        d(chord)/d(cos i) = -kappa^2 |cos i| / chord = -kappa b / chord,

    so the log of its magnitude is `log kappa + log b - log chord`.  It
    diverges as chord -> 0 (a grazing transit, where a wide range of cos i maps
    into a vanishing range of chord) and vanishes at b -> 0 (a central
    transit).  Both are floored; the divergence is integrable, exactly as the
    V_c/V_e fold is, so the floor caps a real feature rather than hiding a
    wrong one.
    """
    kappa = chord_kappa(ar, ecc, esinw)
    b = pt.sqrt(pt.maximum(chord_radicand(chord, p), CHORD_RADICAND_FLOOR))
    return (
        pt.log(pt.maximum(kappa, 1e-12))
        + pt.log(pt.maximum(b, 1e-12))
        - pt.log(pt.maximum(chord, 1e-12))
    )


# ---- Rossiter-McLaughlin sqrt(vsini)cos/sin(lambda) reparameterization ----
# Mirror of calc_ecc / calc_omega for the secosw/sesinw pair.
@register_physics
def calc_vsini_from_sv(svcoslam, svsinlam):
    return pt.sqr(svcoslam) + pt.sqr(svsinlam)


@register_physics
def calc_lam_from_sv(svcoslam, svsinlam):
    """Spin-orbit angle from the sqrt(vsini) pair.

    At exactly 0 the angle is undefined and the convention is 0 (aligned) --
    and `svcoslam: 0, svsinlam: 0` is precisely how an aligned start is
    spelled, so this is the same case `calc_omega` has, and it is hardened
    the same way and for the same reason: biasing the x argument to
    `arctan2(0, 1) = 0` keeps the convention with ONE branch, leaving no
    unselected `arctan2(0, 0)` for `where`'s VJP to multiply by zero
    (see _circular_bias, review 1.8.2).
    """
    v_raw = pt.sqr(svcoslam) + pt.sqr(svsinlam)
    return pt.arctan2(svsinlam, svcoslam + _circular_bias(v_raw))
