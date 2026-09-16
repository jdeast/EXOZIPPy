"""One-compartment pharmacokinetics with first-order absorption.

See this directory's README.md before relying on anything here: the component
set is written by an astrophysicist and an LLM with no domain reviewer, and the
modelling choices are unreviewed.

THE MODEL
---------

A single oral dose ``D`` entering a depot compartment, absorbed first-order at
rate ``ka`` into a central compartment of volume ``V``, and eliminated
first-order at rate ``ke = CL / V``.  This is ADVAN2/TRANS2 in NONMEM and
``SSfol`` in R's nlme, and it is the most-fit model in the field::

    C(t) = (F * D * ka) / (V * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

``F`` (bioavailability) is exactly confounded with ``V`` and ``CL`` when only
oral data are available, so it is pinned at 1 and the estimates are the
*apparent* CL/F and V/F.  That is standard practice, and the component's
labels say "apparent" for the reason given in the README.

THE ka == ke SINGULARITY
------------------------

``(ka - ke)`` is a denominator and the sampler WILL visit ``ka ~ ke``, because
the flip-flop degeneracy (README, and the component docstring) puts a ridge
through exactly that plane.  The singularity is REMOVABLE -- the limit is
``C = (D*ka/V) * t * exp(-ka*t)`` -- but a naive expression evaluates 0/0
there and poisons the gradient.

``pt.where`` over the two branches is banned (CLAUDE.md's where-trap: its VJP
multiplies the unselected branch by zero, and ``0 * NaN`` poisons the gradient
of the whole expression on every backend).  It is also unnecessary.  Writing
``d = (ka - ke) * t`` and factoring::

    C(t) = (F*D*ka/V) * t * exp(-ke*t) * phi(d),    phi(d) = (1 - exp(-d)) / d

``phi`` is entire with ``phi(0) = 1``.  Using ``1 - exp(-d) = 2 exp(-d/2)
sinh(d/2)`` and ``s = d/2``::

    exp(-ke*t) * phi(d) = exp(-m) * sinh(s) / s,    m = (ka + ke) t / 2

-- i.e. the damping factor and the bracket combine into one ``sinh``-shaped
term whose two exponents are the MEAN and HALF-DIFFERENCE of ``ke*t`` and
``ka*t``.  ``sinh(s)/s`` is EVEN and entire, so a floor goes on ``s**2``,
where there is no sign to lose and the error is O(eps**2)::

    y = sqrt(s*s + KA_KE_FLOOR**2)     # >= KA_KE_FLOOR > 0, always

That floor is the ``CHORD_RADICAND_FLOOR`` pattern in
``components/orbit/physics.py``, and it is STRICTLY POSITIVE for the reason
stated there: ``d/ds sqrt(s**2 + eps**2) = s/y`` is bounded by 1 and is exactly
0 at ``s = 0``, so there is one graph, no branch, and a finite gradient
everywhere.  Flooring at zero would rebuild the very ``0 * inf`` the rule
exists to prevent, because ``sqrt'(0)`` is infinite.

**How the floored term is actually spelled is a separate question, and two
plausible answers are wrong.**  ``exp(-m) * sinh(y)/y`` overflows; the obvious
overflow-safe rewrite loses six digits at small ``y``.  ``_damped_sinhc``
below carries the working form and the measurements, and both failures are
regression-pinned in ``tests/test_pharmacokinetics_physics.py``.
"""

import numpy as np
import pytensor.tensor as pt

from ...physics_registry import register_physics

# Strictly positive floor on |s| = |ka - ke| * t / 2, which is dimensionless.
# The induced relative error is O(KA_KE_FLOOR**2 / 6) ~ 1.7e-13 -- far below
# the precision of any concentration measurement.
#
# DO NOT set this to 0.  See the module docstring: a zero floor has infinite
# derivative on the clamped side and rebuilds the 0 * inf it exists to remove.
KA_KE_FLOOR = 1.0e-6


def _damped_sinhc(m, s):
    """``exp(-m) * sinh(y) / y`` with ``y = sqrt(s**2 + KA_KE_FLOOR**2)``.

    Evaluated as ``exp(y - m) * (-expm1(-2y)) / (2y)``, which is the same
    quantity written so that neither end of the range loses anything.

    THE SPELLING IS THE WHOLE POINT, and two earlier ones were wrong:

    * ``exp(-m) * sinh(y) / y`` -- the obvious form -- OVERFLOWS.  ``sinh``
      reaches ``inf`` near ``y = 710`` while ``exp(-m)`` has underflowed to
      ``0``, so the product is ``0 * inf`` -> NaN: exactly the gradient
      poisoning this module's floor exists to prevent, reintroduced at the
      other end of the range.  It is reachable -- ``s = (ka - ke) t / 2``, so
      an excursion to a large ``ka`` over a 25-hour window crosses it, with
      only the parameter bounds in the way.  Found by numerical test (an AUC
      integral over a long grid returned NaN), not by inspection.
    * ``(exp(y - m) - exp(-y - m)) / (2y)`` fixes the overflow and then loses
      SIX DIGITS to cancellation at small ``y``, where it subtracts two
      numbers that agree to one part in ``1/y``.  Measured: 1.1e-10 relative
      error at the ``ka == ke`` limit, against 1e-16 for the form above.

    ``expm1`` is what makes the small-``y`` end exact -- ``-expm1(-2y)/(2y)``
    is computed without ever forming ``1 - (1 - 2y + ...)`` -- while the
    ``exp(y - m)`` prefactor keeps the large-``y`` end finite.

    ``exp(y - m)`` cannot overflow whenever ``m >= |s|``, since ``y`` exceeds
    ``|s|`` only by the floor.  That holds for the concentration curve, where
    ``m = (ka + ke) t / 2`` and ``s = (ka - ke) t / 2`` with ka, ke > 0.  A
    caller that cannot promise it (``calc_pk_tmax``) owes its own argument for
    what bounds the exponent.

    The floor on ``s**2`` is unchanged and is the point of the sinh form:
    ``sinh(y)/y`` is EVEN, so there is no sign to lose, and the floor is
    STRICTLY POSITIVE so the radicand's derivative ``s/y`` stays bounded by 1
    and is exactly 0 at ``s = 0``.
    """
    y = pt.sqrt(s * s + KA_KE_FLOOR * KA_KE_FLOOR)
    return pt.exp(y - m) * (-pt.expm1(-2.0 * y)) / (2.0 * y)


def concentration_reference(t, dose, ka, ke, v):
    """Numpy reference for the curve, as the textbook writes it.  Tests only.

    The unfactored closed form, so a test can assert the implementation agrees
    with it away from ``ka == ke``.  Deliberately not registered and never
    used by the model: it is 0/0 at ``ka == ke``, which is the whole problem.
    """
    t, ka, ke = (np.asarray(a, dtype=float) for a in (t, ka, ke))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (
            (dose * ka) / (v * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
        )


@register_physics
def calc_pk_concentration(t, dose, ka, ke, v):
    """Central-compartment concentration at times ``t``.

    Parameters are broadcast against ``t``, so a per-observation ``t`` with
    per-observation ``dose``/``ka``/``ke``/``v`` (gathered through a subject
    map) is the normal call.

    Units are the component's internal ones and must be consistent: ``t`` in
    hours, ``dose`` in mg, ``ka``/``ke`` in 1/hr, ``v`` in L, giving mg/L.
    Nothing here checks that; the Parameter layer converts before this is
    called.
    """
    # The damping factor exp(-ke*t) is folded INTO the bracket rather than
    # multiplied on afterwards -- that is what makes the whole expression
    # overflow-proof.  m and s are the mean and half-difference of the two
    # exponents ke*t and ka*t, so m >= |s| for any positive ka, ke, t, which
    # is exactly _damped_sinhc's precondition.
    m = 0.5 * (ka + ke) * t
    s = 0.5 * (ka - ke) * t
    return (dose * ka / v) * t * _damped_sinhc(m, s)


@register_physics
def calc_pk_pow10(log_x):
    """``10**log_x``.  The sampled-in-log -> physical bridge.

    ONE function serving CL, V and ka rather than three identical ones: the
    physics registry is a flat namespace keyed by function name, so three
    spellings of ``10**x`` would be three names to keep in step for no gain.
    Each parameter selects its own input through its ``deps``.

    The rates and volumes are sampled in log10 for the usual two reasons --
    they are positive scale parameters spanning decades, and a log coordinate
    makes the prior scale-invariant -- and for one specific to this model: the
    flip-flop degeneracy (see the module docstring) is a reflection that swaps
    ka and ke, which in log space is a translation of their difference and
    keeps both modes at comparable sampler scales.  Mirrors ``star.logmass ->
    mass`` and ``mulensing.log_s -> s``.
    """
    return pt.power(10.0, log_x)


@register_physics
def calc_pk_ke_from_cl(cl, v):
    """Elimination rate constant ``ke = CL / V``.

    The TRANS2 -> TRANS1 bridge: the component samples (CL, V) by default and
    derives ke, which is what the forward model actually consumes.
    """
    return cl / v


@register_physics
def calc_pk_cl_from_ke(ke, v):
    """Apparent clearance ``CL = ke * V``.

    The reverse bridge, used when the instance samples (ke, V) instead.  CL is
    then a REPORTED quantity -- the forward model never reads it -- but it is
    what every reader wants, which is exactly what the `reported` element role
    is for.
    """
    return ke * v


@register_physics
def calc_pk_log_ke_from_cl(cl, v):
    """``log10(ke) = log10(CL / V)``.  The REPORTED log-rate under TRANS2.

    An instance that samples (CL, V) does not sample ``log_ke``, but the
    quantity exists and has a computable inverse, so it is reported rather
    than left inactive (the rule is in ``parameter.md``): a user's prior or
    bound written against ``log_ke`` survives the flip to TRANS1 only if the
    element stays in the table.

    It reads ``cl`` and ``v`` rather than the already-derived ``ke`` for no
    numerical reason -- the two agree to a rounding -- but because under
    TRANS1 the mirror of this function must not read ``cl`` at all (that is a
    reported element there), and the two expressions are easier to keep
    honest when they take the same shape.
    """
    return pt.log10(cl / v)


@register_physics
def calc_pk_log_cl_from_ke(ke, v):
    """``log10(CL) = log10(ke * V)``.  The REPORTED log-clearance under TRANS1.

    The mirror of :func:`calc_pk_log_ke_from_cl`.  It takes ``(ke, v)`` and
    NOT ``cl``, even though ``cl`` is right there with the same value: under
    TRANS1 ``cl`` is itself reported, and a reported element may not be
    consumed.
    """
    return pt.log10(ke * v)


@register_physics
def calc_pk_v_from_rates(cl, ke):
    """Apparent volume ``V = CL / ke``.

    The third bridge, for the basis that samples the two RATES and derives the
    volume.  That is what R's ``SSfol`` -- and so the canonical ``nlme`` fit of
    the Theophylline data -- is parameterized in, which is why it exists here:
    between-subject variability is defined IN a basis, so reproducing a
    published set of random effects means fitting in the basis they were
    estimated in.
    """
    return cl / ke


@register_physics
def calc_pk_log_v_from_rates(cl, ke):
    """``log10(V) = log10(CL / ke)``.  The REPORTED log-volume under `cl_ke`."""
    return pt.log10(cl / ke)


@register_physics
def calc_pk_population_value(mu, beta, weight, wt_ref, omega, eta):
    """One subject's log-coordinate, drawn from the population.

    ``mu + beta*log10(WT/WT_ref) + omega*eta`` -- a typical value, an
    allometric covariate, and this subject's own deviation, all in dex.

    NON-CENTERED, and that is structural rather than stylistic.  The centered
    form gives the per-subject coordinate a scale of ``omega``, so a small
    ``omega`` closes the funnel NUTS is famous for failing on -- and a small
    ``omega`` is exactly what these data produce: the canonical ``nlme`` fit
    of the Theophylline set drives one of the three between-subject SDs to
    zero.  Sampling the standardized deviation keeps every coordinate O(1)
    whatever ``omega`` does.

    ``beta`` is a parameter and not a literal 0.75 so that it can be pinned,
    reported and -- for a study with the weight range to support it --
    estimated, which is what makes the covariate model revisable rather than
    baked in.
    """
    return mu + beta * pt.log10(weight / wt_ref) + omega * eta


@register_physics
def calc_pk_population_typical(mu, beta, weight, wt_ref):
    """The same, for a coordinate with no between-subject variability.

    Every subject takes the typical value for its own weight.  A separate
    function rather than :func:`calc_pk_population_value` with ``omega``
    pinned to zero, because a pinned ``omega`` would leave one free ``eta``
    per subject that no term of the likelihood reads.
    """
    return mu + beta * pt.log10(weight / wt_ref)


@register_physics
def calc_pk_cv_percent(omega):
    """Between-subject variability as the field quotes it: a CV, in percent.

    For a log-normal, ``CV = sqrt(exp(sigma_ln**2) - 1)``.  ``omega`` here is
    the SD of the base-10 logarithm, so ``sigma_ln = omega * ln(10)`` -- the
    factor of 2.3026 between this component's ``omega`` and the one a NONMEM
    or Monolix table prints, which is the whole reason this is computed rather
    than left to the reader.

    ``expm1`` because the interesting limit is a SMALL omega: a between-subject
    SD collapsing toward zero is a real result on these data, and
    ``exp(x) - 1`` loses every significant digit of it.
    """
    return 100.0 * pt.sqrt(pt.expm1(pt.sqr(omega * np.log(10.0))))


@register_physics
def calc_pk_half_life(ke):
    """Terminal half-life ``ln(2) / ke``, in the time unit of ``ke``."""
    return pt.log(2.0) / ke


@register_physics
def calc_pk_auc(dose, ke, v):
    """Area under the curve to infinity for a single dose, ``D / (ke*V)``.

    ``ke * V`` IS the apparent clearance, so this is the textbook ``D / CL``
    written in the two coordinates every parameterization has.  Spelling it
    ``dose / cl`` would read better and is what the design first called for,
    and it is WRONG under TRANS1: there ``cl`` is a REPORTED element (nothing
    in the likelihood consumes it), and nothing may consume a reported element
    -- the consumer would read its pre-patch placeholder.
    ``System._validate_reported_not_consumed`` refuses that manifest outright,
    so the trap is a loud one; this spelling avoids it in both directions
    because ``ke`` and ``v`` are sampled-or-derived in both.  See
    ``subject.Subject.COORD_MODE_TABLE``.
    """
    return dose / (ke * v)


@register_physics
def calc_pk_tmax(ka, ke):
    """Time of peak concentration, ``ln(ka/ke) / (ka - ke)``.

    The SAME removable singularity as the concentration curve (the limit at
    ka == ke is ``1/ka``), and it reuses the same floor rather than growing a
    second one.  Writing ``r = ka/ke`` and ``u = ln(r)/2``::

        ln(r) / (ka - ke) = ln(r) / (ke * (r - 1))

    and ``(r - 1) = exp(2u) - 1 = 2 exp(u) sinh(u)``, so::

        tmax = 2u / (ke * 2 exp(u) sinh(u)) = 1 / (ke * exp(u) * sinh(u)/u)

    which is ``1 / (ke * exp(u) * sinh(u)/u)`` -- the same helper with the
    damping running the other way, ``_damped_sinhc(-u, u)``.  At ka == ke,
    u = 0, the helper is 1 and tmax = 1/ke = 1/ka as required.

    This call does NOT satisfy the helper's ``m >= |s|`` precondition (here
    ``m = -u``), so it owes its own overflow argument: the larger exponent is
    ``y - m -> 2u = ln(ka/ke)``, so the exponential is just ``ka/ke``, and
    that overflows only for a rate ratio around 1e308.  The parameters are
    log-sampled with bounds far inside that.
    """
    u = 0.5 * (pt.log(ka) - pt.log(ke))
    return 1.0 / (ke * _damped_sinhc(-u, u))


@register_physics
def calc_pk_cmax(dose, ka, ke, v):
    """Peak concentration: the curve evaluated at its own tmax."""
    return calc_pk_concentration(calc_pk_tmax(ka, ke), dose, ka, ke, v)


def combined_sigma(conc, sigma_add, sigma_prop):
    """Residual SD of the combined error model.

    ``sqrt(sigma_add**2 + (sigma_prop * C)**2)`` -- the NONMEM/Monolix default
    and the direct analogue of this codebase's jitter-in-quadrature.  It is a
    plain helper rather than registered physics: it is a likelihood term, not
    a parameter expression, and nothing looks it up by ``func_name``.

    Unlike ``Instrument._jitter_floor``'s SIGNED jitter variance, both terms
    here are positive-bounded.  That is not an oversight: this codebase's
    jitter is added in quadrature to a REPORTED per-point error and may
    legitimately go negative to absorb an over-estimated one, whereas these
    data carry no reported error for a negative term to correct.
    """
    return pt.sqrt(pt.sqr(sigma_add) + pt.sqr(sigma_prop * conc))
