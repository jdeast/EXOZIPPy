"""The one-compartment PK forward model (components/pharmacokinetics/physics.py).

See that directory's README.md: the component set is written by an astronomer
and an LLM with no domain reviewer.  These tests pin that the code computes the
model it says it computes -- the textbook closed form, its removable
singularity, and the exact degeneracy the model carries.  They do NOT and
cannot pin that this is the right model for anyone's data.

Two properties carry the design and each caught a real defect while being
written:

* the ``ka == ke`` singularity is removable and must be evaluated with a
  finite gradient, without ``pt.where`` (CLAUDE.md's where-trap) and without
  overflowing at the other end of the range;
* the flip-flop degeneracy is EXACT, which is what makes the two posterior
  modes a real feature of the model rather than a sampler artifact.
"""

import numpy as np
import pytensor
import pytensor.graph.traversal
import pytensor.tensor as pt
import pytest

from exozippy.components.pharmacokinetics import physics as P

# A representative subject: 320 mg oral dose, absorption ~10x faster than
# elimination, 30 L apparent volume.  Roughly theophylline-shaped.
DOSE, KA, KE, V = 320.0, 1.5, 0.1, 30.0


def _fn(func, nargs):
    """Compile ``func`` over ``nargs`` float64 vector inputs."""
    args = [pt.dvector(f"a{i}") for i in range(nargs)]
    return pytensor.function(args, func(*args), on_unused_input="ignore")


@pytest.fixture(scope="module")
def conc():
    return _fn(P.calc_pk_concentration, 5)


@pytest.fixture(scope="module")
def tmax():
    return _fn(P.calc_pk_tmax, 2)


def _c(conc, t, dose=DOSE, ka=KA, ke=KE, v=V):
    """Evaluate the curve with scalars broadcast against ``t``."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ones = np.ones_like(t)
    return conc(t, ones * dose, ones * ka, ones * ke, ones * v)


# ---------------------------------------------------------------------------
# It computes the textbook curve
# ---------------------------------------------------------------------------


def test_matches_the_closed_form_away_from_the_singularity(conc):
    """Given random (t, ka, ke, V), When evaluated, Then it equals the textbook form.

    The NUMBERS, not the code path: 2000 random parameter sets against
    C = D*ka/(V*(ka-ke)) * (exp(-ke t) - exp(-ka t)), the expression the
    implementation deliberately does NOT evaluate directly.
    """
    rng = np.random.default_rng(1)
    n = 2000
    t = rng.uniform(0.1, 25.0, n)
    ka = rng.uniform(0.3, 3.0, n)
    ke = rng.uniform(0.02, 0.25, n)
    v = rng.uniform(20.0, 60.0, n)
    dose = np.full(n, DOSE)

    got = conc(t, dose, ka, ke, v)
    want = P.concentration_reference(t, dose, ka, ke, v)

    assert np.all(np.isfinite(got))
    assert np.abs(got / want - 1.0).max() < 1e-12


def test_a_hand_computed_point(conc):
    """Given one hand-worked case, Then the curve reproduces it.

    A single value carried through the design review by hand, so a refactor
    that changed the model rather than its spelling cannot pass everything
    else by being self-consistent.
    """
    assert _c(conc, 2.0)[0] == pytest.approx(8.787928, abs=1e-5)


def test_the_curve_starts_at_zero_and_decays(conc):
    """Given a single oral dose, Then C(0) = 0 and C decays at long times.

    The qualitative shape of first-order absorption: nothing in the central
    compartment at t = 0, a peak, then elimination.
    """
    assert _c(conc, 0.0)[0] == pytest.approx(0.0, abs=1e-15)
    late = _c(conc, [50.0, 100.0, 200.0])
    assert np.all(np.diff(late) < 0)
    assert np.all(late > 0)


# ---------------------------------------------------------------------------
# The ka == ke singularity is removable, finite, and differentiable
# ---------------------------------------------------------------------------


def test_the_ka_equals_ke_limit_is_the_analytic_one(conc):
    """Given ka == ke exactly, Then C = (D ka / V) t exp(-ka t).

    The closed form is 0/0 here.  The limit is exact, not approached: the
    floor's error is O(KA_KE_FLOOR**2) ~ 1e-13 relative.
    """
    ka = ke = 0.7
    got = _c(conc, 3.0, ka=ka, ke=ke)[0]
    want = (DOSE * ka / V) * 3.0 * np.exp(-ka * 3.0)

    assert got == pytest.approx(want, rel=1e-11)
    assert np.isnan(P.concentration_reference(3.0, DOSE, ka, ke, V)), (
        "the reference is supposed to be 0/0 here -- if it is not, this test "
        "has stopped exercising the singularity"
    )


def test_the_curve_is_continuous_through_the_singularity(conc):
    """Given ka sweeping through ke, Then C varies smoothly with no spike.

    Approached from both sides and straight through, since an even-function
    floor could plausibly be right on one side and wrong on the other.
    """
    ke = 0.7
    offsets = np.array(
        [-1e-3, -1e-5, -1e-7, -1e-9, 0.0, 1e-9, 1e-7, 1e-5, 1e-3]
    )
    vals = conc(
        np.full(offsets.size, 3.0),
        np.full(offsets.size, DOSE),
        ke + offsets,
        np.full(offsets.size, ke),
        np.full(offsets.size, V),
    )
    assert np.all(np.isfinite(vals))
    assert np.ptp(vals) < 1e-3 * vals.mean()


@pytest.mark.parametrize("delta", [0.0, 1e-12, 1e-8, 1e-4, 0.5])
def test_the_gradient_is_finite_at_and_around_the_singularity(delta):
    """Given ka - ke -> 0, When d/dka and d/dke are taken, Then both are finite.

    This is the reason KA_KE_FLOOR is STRICTLY POSITIVE.  A floor AT zero has
    an infinite sqrt derivative on the clamped side and rebuilds the 0 * inf
    the design forbids; NaN in one term poisons the whole gradient vector, so
    a finite value here is a precondition for the model being samplable at all.
    """
    a_ka, a_ke = pt.dscalar("ka"), pt.dscalar("ke")
    curve = P.calc_pk_concentration(
        pt.as_tensor_variable(3.0),
        pt.as_tensor_variable(DOSE),
        a_ka,
        a_ke,
        pt.as_tensor_variable(V),
    )
    grad = pytensor.function([a_ka, a_ke], pt.grad(curve, [a_ka, a_ke]))

    g_ka, g_ke = grad(0.7 + delta, 0.7)
    assert np.isfinite(g_ka)
    assert np.isfinite(g_ke)


def test_no_where_in_the_concentration_graph():
    """Given the compiled curve, Then its graph contains no Switch/Select op.

    The where-trap is a standing invariant (CLAUDE.md), and the singularity
    here is exactly the situation that tempts a branch.  Checking the GRAPH
    rather than the source catches a branch introduced through a helper.
    """
    a = [pt.dscalar(n) for n in ("t", "d", "ka", "ke", "v")]
    graph = P.calc_pk_concentration(*a)
    names = {
        type(node.op).__name__
        for node in pytensor.graph.traversal.ancestors([graph])
        if node.owner is not None
        for node in [node.owner]
    }
    assert not {"Switch", "Select"} & names, sorted(names)


# ---------------------------------------------------------------------------
# Overflow: the defect the first implementation shipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ka, ke, t",
    [
        (1.5, 0.1, 4000.0),  # long time: s = (ka-ke)t/2 ~ 2800
        (800.0, 0.1, 25.0),  # fast absorption
        (0.1, 800.0, 25.0),  # fast elimination (the mirrored branch)
        (1e4, 1e-4, 100.0),  # extreme ratio, both directions
        (1e-4, 1e4, 100.0),
    ],
)
def test_no_overflow_at_large_rate_time_products(conc, ka, ke, t):
    """Given a large |ka - ke| * t, Then the curve is finite.

    The first implementation wrote the damping as exp(-m) * sinh(y)/y, where
    sinh overflows to inf near y = 710 while exp(-m) has underflowed to 0 --
    0 * inf -> NaN, the very gradient poisoning the floor exists to prevent,
    reintroduced at the opposite end of the range.  It is reachable by an
    ordinary sampler excursion, so this is a regression pin and not a
    hypothetical.
    """
    got = _c(conc, t, ka=ka, ke=ke)[0]
    assert np.isfinite(got)
    assert got >= 0.0


@pytest.mark.parametrize("ka", [0.7, 50.0, 500.0])
def test_the_gradient_survives_large_rates(ka):
    """Given a large ka, When the gradient is taken, Then it stays finite.

    The value being finite is not enough: the overflow above produced NaN in
    the gradient first.
    """
    a_ka, a_ke = pt.dscalar("ka"), pt.dscalar("ke")
    curve = P.calc_pk_concentration(
        pt.as_tensor_variable(3.0),
        pt.as_tensor_variable(DOSE),
        a_ka,
        a_ke,
        pt.as_tensor_variable(V),
    )
    grad = pytensor.function([a_ka, a_ke], pt.grad(curve, [a_ka, a_ke]))
    assert all(np.isfinite(g) for g in grad(ka, 0.1))


# ---------------------------------------------------------------------------
# The flip-flop degeneracy
# ---------------------------------------------------------------------------


def test_the_flip_flop_swap_leaves_the_curve_unchanged(conc):
    """Given (ka, ke, V), When swapped to (ke, ka, V*ke/ka), Then C is identical.

    Flip-flop kinetics, and it is EXACT rather than approximate -- which is
    what makes the two posterior modes a property of the model rather than a
    sampler artifact, and what the mode report will be asked to find.

    Derivation: the prefactor D*ka/(V(ka-ke)) and the bracket
    (exp(-ke t) - exp(-ka t)) BOTH change sign under the swap, and the
    residual factor ke/ka is absorbed exactly by V -> V ke/ka.
    """
    t = np.linspace(0.1, 25.0, 500)
    original = _c(conc, t)
    swapped = _c(conc, t, ka=KE, ke=KA, v=V * KE / KA)

    assert np.abs(swapped / original - 1.0).max() < 1e-13


def test_clearance_is_invariant_under_the_flip_flop_but_volume_is_not():
    """Given the flip-flop swap, Then CL is unchanged and V is not.

    The clinically important half of the degeneracy: CL (hence AUC and
    steady-state dosing) is identical in both modes, while V (hence a loading
    dose) differs by ke/ka.  A report that collapsed the two modes would be
    right about one and wrong about the other.
    """
    v_swapped = V * KE / KA

    assert KE * V == pytest.approx(KA * v_swapped, rel=1e-15)
    assert v_swapped != pytest.approx(V, rel=1e-3)


# ---------------------------------------------------------------------------
# Derived / reported quantities
# ---------------------------------------------------------------------------


def test_tmax_matches_its_closed_form(tmax):
    """Given random (ka, ke), Then tmax = ln(ka/ke)/(ka - ke)."""
    rng = np.random.default_rng(2)
    ka = rng.uniform(0.3, 3.0, 1000)
    ke = rng.uniform(0.02, 0.25, 1000)

    got = tmax(ka, ke)
    want = np.log(ka / ke) / (ka - ke)

    assert np.abs(got / want - 1.0).max() < 1e-12


def test_tmax_has_the_same_removable_singularity(tmax):
    """Given ka == ke, Then tmax = 1/ka, via the SAME floor as the curve.

    Reusing the one helper is the point: it is where a reviewer can see the
    floor is a shared primitive rather than a local patch, so this pins that
    the second call site behaves like the first.
    """
    assert tmax(np.array([0.7]), np.array([0.7]))[0] == pytest.approx(
        1.0 / 0.7, rel=1e-11
    )


def test_tmax_actually_maximizes_the_curve(conc, tmax):
    """Given tmax, Then C(tmax) exceeds C just either side of it.

    Pins that tmax is the peak of THIS curve, not an expression that merely
    matches a formula -- a sign error would satisfy the closed-form test.
    """
    peak = tmax(np.array([KA]), np.array([KE]))[0]
    around = _c(conc, [peak - 0.05, peak, peak + 0.05])

    assert around[1] > around[0]
    assert around[1] > around[2]


def test_auc_equals_the_numerically_integrated_curve(conc):
    """Given D/CL, Then it equals the integral of C(t) from 0 to infinity.

    AUC = D/CL is an identity of the model, not a definition, so it is worth
    checking against the curve the model actually produces.
    """
    auc = _fn(P.calc_pk_auc, 2)
    analytic = auc(np.array([DOSE]), np.array([KE * V]))[0]

    grid = np.linspace(0.0, 4000.0, 400_001)[1:]
    numeric = np.trapezoid(_c(conc, grid), grid)

    assert analytic == pytest.approx(numeric, rel=1e-4)


def test_cmax_is_the_curve_at_its_peak(conc):
    """Given cmax, Then it matches the maximum over a fine grid."""
    cmax = _fn(P.calc_pk_cmax, 4)
    got = cmax(
        np.array([DOSE]), np.array([KA]), np.array([KE]), np.array([V])
    )[0]
    grid_max = _c(conc, np.linspace(0.1, 25.0, 20_001)).max()

    assert got == pytest.approx(grid_max, rel=1e-6)


def test_half_life_matches_ln2_over_ke():
    """Given ke, Then t_half = ln(2)/ke."""
    half = _fn(P.calc_pk_half_life, 1)
    assert half(np.array([KE]))[0] == pytest.approx(np.log(2.0) / KE)


def test_the_two_rate_bridges_are_inverses():
    """Given CL = ke V, Then ke = CL/V recovers it, and vice versa.

    The TRANS1 <-> TRANS2 pair.  They are one identity read in two directions,
    and the component picks which one is sampled per instance, so a sign or
    reciprocal slip in either would silently change what the other means.
    """
    ke_from = _fn(P.calc_pk_ke_from_cl, 2)
    cl_from = _fn(P.calc_pk_cl_from_ke, 2)

    cl = cl_from(np.array([KE]), np.array([V]))
    assert ke_from(cl, np.array([V]))[0] == pytest.approx(KE, rel=1e-15)


# ---------------------------------------------------------------------------
# Residual error model
# ---------------------------------------------------------------------------


def test_combined_error_model_is_additive_and_proportional_in_quadrature():
    """Given (sigma_add, sigma_prop), Then sigma = sqrt(add^2 + (prop C)^2).

    The NONMEM/Monolix default.  Pins both limits, because the two terms
    dominate in opposite regimes and a swapped pair would look plausible at
    one concentration.
    """
    sigma = _fn(P.combined_sigma, 3)

    # additive dominates at low concentration, proportional at high
    low = sigma(np.array([0.01]), np.array([0.1]), np.array([0.2]))[0]
    high = sigma(np.array([100.0]), np.array([0.1]), np.array([0.2]))[0]

    assert low == pytest.approx(np.sqrt(0.1**2 + (0.2 * 0.01) ** 2))
    assert high == pytest.approx(np.sqrt(0.1**2 + (0.2 * 100.0) ** 2))
    assert high == pytest.approx(20.0, rel=1e-3)


def test_combined_error_model_is_positive_with_no_data_error_to_absorb():
    """Given a zero concentration, Then sigma is the additive term, not zero.

    Unlike this codebase's SIGNED jitter variance -- which may go negative to
    correct an over-estimated REPORTED error -- these data carry no reported
    error, so there is nothing for a negative term to absorb and sigma must
    stay strictly positive or the likelihood is undefined.
    """
    sigma = _fn(P.combined_sigma, 3)
    assert sigma(np.array([0.0]), np.array([0.1]), np.array([0.2]))[0] == (
        pytest.approx(0.1)
    )
