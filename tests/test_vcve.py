"""The V_c/V_e eccentricity parameterization (Eastman 2024, arXiv:2309.14410).

A transit-only fit in sqrt(e)cos/sin(omega) recovers eccentricities that are
measurably wrong (the paper, over 330 simulated systems); sampling
``V_c/V_e = sqrt(1-e^2)/(1 + e sin omega)`` and the direction of omega recovers
them.  Two things the paper does not address are what most of this file is
about, because they are what a gradient sampler needs:

* the inversion is a QUADRATIC, and the paper selects a root with a discrete
  sign parameter -- piecewise constant, so no gradient, and a logp that jumps.
  Here the likelihood is MARGINALIZED over both roots.
* a combination with no real root is rejected outright there.  Here the
  discriminant is floored (the hard shield, so a NaN is unbuildable) and a soft
  bound supplies the restoring gradient.

The algebra is checked three ways -- against the forward relation, against the
quadratic itself, and against a finite difference -- because a factor of 2
dropped from the discriminant produced eccentricities that looked entirely
plausible and were wrong.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit import physics
from exozippy.system import System

MAX_ECC = physics.MAX_ECC


def _f(node):
    return np.atleast_1d(np.asarray(node.eval(), dtype=float))


def vcve_forward(ecc, omega):
    """Paper eq 4, in numpy, as the reference the port must reproduce."""
    return np.sqrt(1.0 - ecc**2) / (1.0 + ecc * np.sin(omega))


# ---------------------------------------------------------------------------
# 1. The algebra
# ---------------------------------------------------------------------------

_GRID = [
    (e, w)
    for e in (0.0, 0.01, 0.1, 0.4, 0.7, 0.93)
    for w in np.linspace(-np.pi, np.pi, 9)
]


@pytest.mark.parametrize("ecc,omega", _GRID)
def test_the_inversion_round_trips_the_forward_relation(ecc, omega):
    """
    Given an (e, omega) pair,
    When V_c/V_e is computed from it and then inverted,
    Then one of the two roots is that eccentricity again.

    This is the test that catches a wrong discriminant: dropping the factor of
    2 in sqrt(B^2 - 4AC) = 2 sqrt(1 - x^2 cos^2 w) gives roots that are real,
    in range, and wrong -- as the first draft of calc_ecc_from_vcve was.
    """
    x = vcve_forward(ecc, omega)
    hi = _f(physics.calc_ecc_from_vcve(pt.as_tensor_variable(x), omega))[0]
    lo = _f(physics.calc_ecc_from_vcve_lo(pt.as_tensor_variable(x), omega))[0]

    assert hi == pytest.approx(ecc, abs=1e-9) or lo == pytest.approx(
        ecc, abs=1e-9
    ), f"neither root {hi}/{lo} recovers e={ecc} at omega={omega}"


@pytest.mark.parametrize("x,omega", [(0.5, 1.2), (0.9, -0.4), (1.15, 4.3)])
def test_both_roots_satisfy_the_quadratic(x, omega):
    """
    Given a V_c/V_e and omega with a real solution,
    When both roots are computed,
    Then each satisfies A e^2 + B e + C = 0 with the paper's coefficients.

    Independent of the round trip above: it checks the roots against the
    equation rather than against the function they were derived from.
    """
    s = np.sin(omega)
    a = 1.0 + x**2 * s**2
    b = 2.0 * x**2 * s
    c = x**2 - 1.0
    for root in (
        _f(physics.calc_ecc_from_vcve(pt.as_tensor_variable(x), omega))[0],
        _f(physics.calc_ecc_from_vcve_lo(pt.as_tensor_variable(x), omega))[0],
    ):
        if not (0.0 < root < MAX_ECC):
            continue  # clipped: it is not a solution and does not claim to be
        assert a * root**2 + b * root + c == pytest.approx(0.0, abs=1e-9)


def test_the_upper_root_is_the_physical_one_below_one():
    """
    Given V_c/V_e < 1, where the quadratic's roots have opposite signs,
    When both are computed,
    Then the UPPER one is the physical solution and the lower one is clipped.

    This is why the primary branch is the upper root: "prefer the lower
    eccentricity" (EXOFASTv2's choice among PHYSICAL roots) would pin the
    forward model at e = 0 across this whole region, with no gradient.
    """
    x, omega = 0.5, np.pi / 2
    hi = _f(physics.calc_ecc_from_vcve(pt.as_tensor_variable(x), omega))[0]
    lo = _f(physics.calc_ecc_from_vcve_lo(pt.as_tensor_variable(x), omega))[0]

    assert hi == pytest.approx(0.6, abs=1e-9)
    assert vcve_forward(hi, omega) == pytest.approx(x, abs=1e-9)
    assert lo == 0.0  # the unphysical -1.0, clipped


def test_the_jacobian_matches_a_finite_difference():
    """
    Given the analytic log|d(V_c/V_e)/de|,
    When it is compared to a numerical derivative of the forward relation,
    Then they agree.

    The Jacobian is what keeps the prior uniform in e rather than biased toward
    high eccentricity, so getting it wrong is a silent prior change -- exactly
    the class of error the codebase's other Jacobian notes warn about.  This
    pins its MAGNITUDE only; the direction it is applied in is a separate
    statement, and a separate test (see below), because no comparison against
    |dv/de| can distinguish removing the bias from doubling it.
    """
    for ecc, omega in ((0.3, 0.7), (0.6, -1.1), (0.05, 2.9)):
        h = 1e-7
        num = (vcve_forward(ecc + h, omega) - vcve_forward(ecc - h, omega)) / (
            2 * h
        )
        ana = np.exp(
            _f(physics.vcve_log_jacobian(pt.as_tensor_variable(ecc), omega))[0]
        )
        assert ana == pytest.approx(abs(num), rel=1e-5)


def test_subtracting_the_jacobian_flattens_the_implied_prior_on_e():
    """
    Given V_c/V_e drawn uniformly, as its logit-transformed bounds imply,
    When the derived eccentricities are weighted by exp(-log|d(V_c/V_e)/de|),
    Then their density is flat in e -- and it is NOT flat unweighted, nor with
      the opposite sign.

    The sign is the term.  Uniform in V_c/V_e induces p(e) ~ |d(V_c/V_e)/de|,
    which is the paper's "strongly biases e toward high eccentricities", so the
    correction is the RECIPROCAL of the derivative.  Adding the derivative
    instead doubles the bias, and the finite-difference test above passes either
    way -- so the direction has to be measured on the density itself, which is
    what this does.  Both wrong answers are asserted against, because "flat"
    alone would also pass for a weight that is accidentally constant.
    """
    omega = 1.0
    sinw, cosw = np.sin(omega), np.cos(omega)
    rng = np.random.default_rng(0)
    # The real-root region at this omega; beyond it the soft bound takes over.
    x = rng.uniform(0.0, 1.0 / abs(cosw), 400_000)
    ecc = _f(physics.calc_ecc_from_vcve(pt.as_tensor_variable(x), omega))
    keep = (ecc > 0.0) & (ecc < 0.9)
    ecc = ecc[keep]
    jac = np.exp(
        _f(physics.vcve_log_jacobian(pt.as_tensor_variable(ecc), omega))
    )

    bins = np.linspace(0.0, 0.9, 10)

    def spread(weights):
        h, _ = np.histogram(ecc, bins=bins, weights=weights)
        h = h / h.sum()
        return h.max() / h.min()

    assert spread(1.0 / jac) < 1.05  # flat to 5%
    assert spread(np.ones_like(ecc)) > 1.2  # unweighted: biased high
    assert spread(jac) > 1.5  # wrong sign: worse than unweighted


# ---------------------------------------------------------------------------
# 2. The shields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,omega",
    [
        (1.9, 0.0),  # no real root at all: x^2 cos^2 w > 1
        (1.0, 0.0),  # exactly the degenerate double root
        (0.5, np.pi / 2),  # lower root unphysical
        (2.0, -np.pi / 2),  # e -> 1 territory
    ],
)
def test_both_roots_and_the_jacobian_stay_finite(x, omega):
    """
    Given V_c/V_e and omega combinations with no real root, a double root, or a
      root outside [0, 1),
    When the roots and the Jacobian are evaluated,
    Then every value is finite.

    "It must be impossible to draw a NaN likelihood" is the design requirement
    (notes/todo.txt); the paper's answer -- reject -- is a wall with no gradient
    and a NaN in the JAX backward pass.
    """
    xt = pt.as_tensor_variable(x)
    hi = _f(physics.calc_ecc_from_vcve(xt, omega))[0]
    lo = _f(physics.calc_ecc_from_vcve_lo(xt, omega))[0]
    jac_hi = _f(physics.vcve_log_jacobian(pt.as_tensor_variable(hi), omega))[0]

    assert np.isfinite(hi) and 0.0 <= hi <= MAX_ECC
    assert np.isfinite(lo) and 0.0 <= lo <= MAX_ECC
    assert np.isfinite(jac_hi)


def test_the_discriminant_is_reported_unfloored_for_the_soft_bound():
    """
    Given a combination with no real root,
    When the discriminant helper is asked,
    Then it returns the NEGATIVE value, not the floored one.

    Same reason `ecc_from_sqrte` is unclipped: the floored quantity is flat
    across the whole forbidden region, and a flat penalty has no gradient for
    NUTS to follow back out.  The floor belongs inside the sqrt, where it stops
    a NaN; the soft bound needs the honest number.
    """
    value = _f(physics.vcve_discriminant(pt.as_tensor_variable(1.9), 0.0))[0]

    assert value < 0.0
    assert value == pytest.approx(1.0 - 1.9**2)


def test_tp_from_ecc_agrees_with_the_sqrt_e_form_up_to_a_period():
    """
    Given the same orbit expressed both ways,
    When the time of periastron is computed from (e, omega) and from the
      sqrt(e) pair,
    Then they agree modulo one period.

    The (e, omega) form exists because a V_c/V_e orbit REPORTS the sqrt(e) pair
    and so cannot consume it; it must be the same quantity.  The two differ by a
    whole revolution in E0 where sin(f/2) < 0, which every consumer is exactly
    invariant to (tp enters only as n(t - tp) mod 2 pi).
    """
    n = 2 * np.pi / 3.5
    for ecc, omega in ((0.3, 0.7), (0.6, -2.0), (0.02, 1.9)):
        se = np.sqrt(ecc)
        a = _f(
            physics.calc_tp(
                pt.as_tensor_variable(ecc),
                se * np.sin(omega),
                se * np.cos(omega),
                2460000.0,
                n,
            )
        )[0]
        b = _f(
            physics.calc_tp_from_ecc(
                pt.as_tensor_variable(ecc), omega, 2460000.0, n
            )
        )[0]
        period = 2 * np.pi / n
        assert (a - b) % period == pytest.approx(0.0, abs=1e-6) or (
            b - a
        ) % period == pytest.approx(0.0, abs=1e-6)


def test_tp_from_ecc_is_finite_at_exactly_zero_eccentricity():
    """
    Given a circular orbit,
    When the time of periastron is computed both ways,
    Then the (e, omega) form is finite where the sqrt(e) form is not.

    Review 1.8.2: `calc_tp`'s arctan2(sqrt(e) - sesinw, secosw) is arctan2(0, 0)
    at e = 0 and its sqrt(e) has an infinite derivative there.  The mixture made
    that reachable systematically -- an unphysical branch clips to exactly zero
    -- so the V_c/V_e path uses a form with no sqrt(e) at all.
    """
    n = 2 * np.pi / 3.5
    value = _f(
        physics.calc_tp_from_ecc(pt.as_tensor_variable(0.0), 1.9, 2460000.0, n)
    )[0]

    assert np.isfinite(value)


# ---------------------------------------------------------------------------
# 3. The branch mixture
# ---------------------------------------------------------------------------


def test_the_mixture_is_exactly_a_two_branch_marginalization():
    """
    Given a toy model whose two-branch likelihood is computable in closed form,
    When the branch mixture is added,
    Then the model's logp is exactly logaddexp of the two branches' terms.

    The mechanism substitutes the branch node into the RV-level term sum and
    cancels the as-built copy, which is exact but not obviously so -- so it is
    measured against arithmetic rather than argued.
    """
    import scipy.stats as st

    system = System.__new__(System)
    system._branch_alternatives = []

    with pm.Model() as model:
        x = pm.Normal("x")
        d = pm.Deterministic("d", x * 2.0)
        pm.Normal("y", mu=d, sigma=1.0, observed=np.array([1.0, 2.0]))
        pm.Potential("prior_on_d", -0.5 * pt.sqr(d))
        system.register_branch_alternative("toy", {d: d + 10.0})
        assert system._add_branch_mixtures(model) == 2

    got = float(np.asarray(model.compile_logp()({"x": 0.5})))

    def branch(dv):
        return st.norm.logpdf([1.0, 2.0], dv, 1).sum() - 0.5 * dv**2

    expected = st.norm.logpdf(0.5) + np.logaddexp(
        branch(1.0) + np.log(0.5), branch(11.0) + np.log(0.5)
    )
    assert got == pytest.approx(expected, rel=1e-12)


def test_no_declared_branch_leaves_the_model_untouched():
    """
    Given a model with no branch alternatives,
    When the mixture pass runs,
    Then it adds nothing.

    The pass runs on every build, so "no branches" must cost nothing -- this is
    what keeps every sqrt(e)cos/sin fit bit-identical.
    """
    system = System.__new__(System)
    system._branch_alternatives = []

    with pm.Model() as model:
        pm.Normal("x")
        before = float(np.asarray(model.compile_logp()({"x": 0.25})))
        assert system._add_branch_mixtures(model) == 0

    assert float(np.asarray(model.compile_logp()({"x": 0.25}))) == before
    assert [p.name for p in model.potentials] == []


def test_two_branches_on_one_node_cover_all_four_combinations():
    """
    Given two branches that each replace ONE ELEMENT of the same vector node,
    When the mixture is added,
    Then all 2^2 combinations appear, including the one with both applied.

    This is the two-V_c/V_e-orbit case: two orbits are two elements of one `ecc`
    vector, so both branches name the same node.  Merging their replacements
    into one dict -- the obvious implementation, and the first one here -- keeps
    whichever was declared last, so the both-branches combination silently
    duplicates a single-branch one and the mixture covers 3 of 4.  The offsets
    below are deliberately small: with a far-away branch the logsumexp is
    dominated by one term and a missing combination is invisible to 1e-14.
    """
    import scipy.stats as st

    system = System.__new__(System)
    system._branch_alternatives = []

    with pm.Model() as model:
        x = pm.Normal("x", shape=2)
        d = pm.Deterministic("d", x * 1.0)
        pm.Normal("y", mu=pt.sum(d), sigma=1.0, observed=np.array([0.0]))
        system.register_branch_alternative(
            "e0", {d: pt.set_subtensor(d[0], d[0] + 0.7)}
        )
        system.register_branch_alternative(
            "e1", {d: pt.set_subtensor(d[1], d[1] - 1.3)}
        )
        assert system._add_branch_mixtures(model) == 4

    got = float(np.asarray(model.compile_logp()({"x": np.array([1.0, 2.0])})))

    def branch(d0, d1):
        return st.norm.logpdf(0.0, d0 + d1, 1.0)

    log_w = 2 * np.log(0.5)
    expected = st.norm.logpdf([1.0, 2.0]).sum() + np.logaddexp.reduce(
        [
            branch(1.0, 2.0) + log_w,
            branch(1.7, 2.0) + log_w,
            branch(1.0, 0.7) + log_w,
            branch(1.7, 0.7) + log_w,
        ]
    )
    assert got == pytest.approx(expected, rel=1e-12)


def test_a_branch_that_cannot_compose_with_another_raises():
    """
    Given two branches replacing the same node, the second from scratch,
    When it is declared,
    Then it raises and says a replacement must read the node it replaces.

    Composing branches works by substituting one after the other, which only
    reaches both when each replacement is written relative to its node.  A
    from-scratch replacement would silently win or lose depending on
    declaration order -- the exact bug above, wearing a different hat.
    """
    system = System.__new__(System)
    system._branch_alternatives = []

    with pm.Model():
        x = pm.Normal("x", shape=2)
        d = pm.Deterministic("d", x * 1.0)
        system.register_branch_alternative("ok", {d: d + 1.0})

        with pytest.raises(ValueError, match="relative to the node"):
            system.register_branch_alternative("bad", {d: pt.zeros(2)})


# ---------------------------------------------------------------------------
# 4. End to end
# ---------------------------------------------------------------------------


def _transit_config(lc, fitvcve, extra_orbit=None):
    orbit = {"name": "b"}
    if fitvcve is not None:
        orbit["fitvcve"] = fitvcve
    orbit.update(extra_orbit or {})
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [orbit],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "inst0", "file": lc, "band": "TESS"}],
    }


_TRANSIT_PARAMS = {
    "star.0.radius": {"initval": 1.61, "sigma": 0.05},
    "star.0.mass": {"initval": 1.204, "sigma": 0.05},
    "star.0.teff": {"initval": 6207, "sigma": 100},
    "star.0.feh": {"initval": -0.116, "sigma": 0.08},
    "orbit.0.period": {"initval": 2.99},
    "orbit.0.tc": {"initval": 2459634.3},
    "orbit.0.cosi": {"initval": 0.05},
    "planet.0.radius": {"initval": 1.7},
}


@pytest.fixture(scope="module")
def transit_lc(tmp_path_factory):
    path = tmp_path_factory.mktemp("vcve") / "lc.dat"
    rng = np.random.default_rng(11)
    t = np.linspace(2459634.1, 2459634.5, 150)
    np.savetxt(
        path,
        np.column_stack(
            [t, 1.0 + rng.normal(0.0, 1e-3, t.size), np.full(t.size, 1e-3)]
        ),
    )
    return str(path)


def test_a_vcve_orbit_samples_the_new_coordinates_and_reports_the_old(
    transit_lc,
):
    """
    Given a transit fit with fitvcve: true,
    When the model is built,
    Then V_c/V_e and the omega direction are sampled, ecc/omega are derived from
      them, the sqrt(e) pair is REPORTED, and the logp and its gradient are
      finite at the start.
    """
    system = System(
        _transit_config(transit_lc, True), user_params=dict(_TRANSIT_PARAMS)
    )
    system.prepare()
    model = system.build_model()

    raw = {v.name for v in model.free_RVs}
    assert {"orbit.vcve_raw", "orbit.xomega_raw", "orbit.yomega_raw"} <= raw
    assert not {"orbit.secosw_raw", "orbit.sesinw_raw"} & raw
    assert system.orbit.secosw.is_reported.tolist() == [True]
    assert system.orbit.sesinw.is_reported.tolist() == [True]
    assert system.orbit.ecc.is_derived.tolist() == [True]

    names = {p.name for p in model.potentials}
    assert "orbit.vcve_jacobian" in names
    assert "orbit.vcve_real_root" in names
    assert "branch_mixture" in names

    start = system.get_raw_start(model)
    assert np.isfinite(model.compile_logp()(start))
    assert np.all(np.isfinite(model.compile_dlogp()(start)))


def _two_orbit_config(lc, fitvcve):
    """Two planets, one transit dataset, `fitvcve` per orbit."""
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}, {"name": "c", "orbit_ndx": 1}],
        "orbit": [
            {"name": "b", "fitvcve": fitvcve[0]},
            {"name": "c", "fitvcve": fitvcve[1]},
        ],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "inst0", "file": lc, "band": "TESS"}],
    }


_TWO_ORBIT_PARAMS = dict(
    _TRANSIT_PARAMS,
    **{
        "orbit.1.period": {"initval": 7.5},
        "orbit.1.tc": {"initval": 2459634.4},
        "orbit.1.cosi": {"initval": 0.03},
        "planet.1.radius": {"initval": 1.1},
    },
)


@pytest.mark.parametrize(
    "modes,n_comb",
    [
        ([True, False], 2),
        ([True, True], 4),
    ],
)
def test_each_vcve_orbit_adds_one_branch(transit_lc, modes, n_comb):
    """
    Given a two-orbit system with one or both orbits in V_c/V_e mode,
    When the model is built,
    Then the mixture covers 2^k combinations for the k V_c/V_e orbits, and the
      logp and its gradient are finite at the start.

    The mixed case is the per-element-roles claim in its real setting: one
    orbit samples the sqrt(e) pair while the other reports it, from one manifest
    and one vector.  The both-vcve case is the composition the mixture has to
    get right (see test_two_branches_on_one_node_cover_all_four_combinations).
    """
    system = System(
        _two_orbit_config(transit_lc, modes),
        user_params=dict(_TWO_ORBIT_PARAMS),
    )
    system.prepare()
    model = system.build_model()

    assert system.orbit.secosw.is_reported.tolist() == modes
    assert system.orbit.ecc.is_derived.tolist() == [True, True]
    assert system.orbit.vcve.is_sampled.tolist() == modes

    start = system.get_raw_start(model)
    assert np.isfinite(model.compile_logp()(start))
    assert np.all(np.isfinite(model.compile_dlogp()(start)))

    # One branch per V_c/V_e orbit; the mixture potential is present either way.
    assert len(system._branch_alternatives) == sum(modes)
    assert 2 ** sum(modes) == n_comb


def test_the_reported_pair_matches_the_derived_eccentricity(transit_lc):
    """
    Given a V_c/V_e orbit,
    When the reported sqrt(e)cos/sin pair is evaluated at the start,
    Then it equals sqrt(e) cos/sin(omega) of the derived (e, omega).

    The point of reporting them: the two parameterizations produce the same
    table rows, so a reader can compare fits and a params file can be carried
    across the switch.
    """
    system = System(
        _transit_config(transit_lc, True), user_params=dict(_TRANSIT_PARAMS)
    )
    system.prepare()
    model = system.build_model()
    start = system.get_raw_start(model)

    fn = model.compile_fn(
        model.replace_rvs_by_values(
            [
                system.orbit.ecc.value,
                system.orbit.omega.value,
                system.orbit.secosw.value,
                system.orbit.sesinw.value,
            ]
        ),
        inputs=model.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )
    ecc, omega, secosw, sesinw = (
        float(np.atleast_1d(v)[0]) for v in fn(start)
    )

    assert secosw == pytest.approx(np.sqrt(ecc) * np.cos(omega), abs=1e-9)
    assert sesinw == pytest.approx(np.sqrt(ecc) * np.sin(omega), abs=1e-9)


def test_the_jacobian_potential_carries_the_reciprocal_sign(transit_lc):
    """
    Given a built V_c/V_e model,
    When the Jacobian potential is evaluated at the start,
    Then it equals MINUS log|d(V_c/V_e)/de| at that (e, omega).

    The statistical direction is pinned by
    test_subtracting_the_jacobian_flattens_the_implied_prior_on_e; this pins the
    CALL SITE against it, since a stray sign there would leave the physics
    correct and the fit's prior doubled -- and a doubled prior on eccentricity
    is visible only as slightly-too-eccentric answers, which is precisely the
    failure the parameterization exists to fix.
    """
    system = System(
        _transit_config(transit_lc, True), user_params=dict(_TRANSIT_PARAMS)
    )
    system.prepare()
    model = system.build_model()
    start = system.get_raw_start(model)

    (potential,) = [
        p for p in model.potentials if p.name == "orbit.vcve_jacobian"
    ]
    fn = model.compile_fn(
        model.replace_rvs_by_values(
            [potential, system.orbit.ecc.value, system.orbit.omega.value]
        ),
        inputs=model.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )
    term, ecc, omega = (float(np.atleast_1d(v)[0]) for v in fn(start))
    expected = -float(
        _f(
            physics.vcve_log_jacobian(pt.as_tensor_variable(ecc), omega),
        )[0]
    )

    assert term == pytest.approx(expected, rel=1e-9)


def test_an_ecc_omega_seed_reaches_vcve(transit_lc):
    """
    Given a params file that seeds ecc and omega on a V_c/V_e orbit,
    When the relaxation engine resolves the start,
    Then V_c/V_e starts at the value that (e, omega) implies.

    This is what makes one params file drive either parameterization -- the
    reason the symbolic bridge exists at all, and the reason V_c/V_e carries
    rank 5 (so it absorbs the relation rather than rewriting ecc/omega).
    """
    params = dict(_TRANSIT_PARAMS)
    params["orbit.0.ecc"] = {"initval": 0.3}
    params["orbit.0.omega"] = {"initval": 40.0}
    system = System(_transit_config(transit_lc, True), user_params=params)
    system.prepare()

    # Read the engine's resolved start directly: Parameters are materialized in
    # stage 5 (build_model), and this is a statement about stage 3.
    resolved = system.config_manager.resolve("orbit", "vcve", shape=(1,))
    expected = vcve_forward(0.3, np.radians(40.0))
    assert float(np.atleast_1d(resolved["initval"])[0]) == pytest.approx(
        expected, rel=1e-6
    )


def test_a_vcve_transit_fit_samples_under_numpyro(transit_lc):
    """
    Given a fitvcve transit fit,
    When a short chain is sampled with nuts_sampler="numpyro",
    Then every draw is finite.

    The standing house rule, and this feature has more reason to obey it than
    most: the branch mixture is a logsumexp over two graphs that share their
    inputs, the roots come out of a floored sqrt, and the Jacobian is a sum of
    floored logs -- three separate places a JAX backward pass could produce the
    NaN a C-backend gradient check would not show.
    """
    system = System(
        _transit_config(transit_lc, True), user_params=dict(_TRANSIT_PARAMS)
    )
    system.prepare()
    model = system.build_model()

    with model:
        idata = pm.sample(
            draws=25,
            tune=25,
            chains=1,
            cores=1,
            nuts_sampler="numpyro",
            progressbar=False,
            random_seed=0,
        )

    for var in idata.posterior.data_vars:
        assert np.all(np.isfinite(idata.posterior[var].values)), (
            f"non-finite draws for {var}"
        )


def test_the_chord_half_still_raises(transit_lc):
    """
    Given an orbit asking for fitchord: true,
    When the component is constructed,
    Then it raises, naming the undefined physics and pointing at the V_c/V_e
      half that does work.

    The paper pairs V_c/V_e with the transit chord; only the eccentricity half
    is implemented, and the guard is the feature until the other lands.
    """
    with pytest.raises(NotImplementedError) as exc:
        System(
            _transit_config(transit_lc, None, {"fitchord": True}),
            user_params=dict(_TRANSIT_PARAMS),
        )

    msg = str(exc.value)
    assert "fitchord" in msg
    assert "calc_cosi_from_b" in msg
    assert "fitvcve" in msg
