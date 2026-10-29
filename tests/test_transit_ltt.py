"""
Integration tests for the light-travel-time (Roemer delay) wiring in
transit.py's build_likelihood (components/ltt.py is unit-tested on its own
in tests/test_ltt.py; these tests exercise the WIRED path -- real
orbits.a/m_primary/m_total accessors, real units -- rather than calling
ltt.py's functions directly with hand-typed values).
"""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pymc as pm
import pytensor
import pytest
from exoplanet_core.pymc.ops import Kepler

from exozippy.components import ltt
from exozippy.constants import C_LIGHT_RSUN_PER_DAY
from exozippy.system import System

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_TC = 2459000.0
_PERIOD = 365.25  # days; with star.mass ~ 1 Msun this gives a ~ 1 AU


def _write_two_row_lc(path):
    # Only the TIMES matter for the checks below (mu is independent of the
    # flux/err columns); one row at primary conjunction (tc, by definition
    # of the orbit's tc), one at secondary conjunction (tc + P/2, exact for
    # a circular orbit).
    time = np.array([_TC, _TC + _PERIOD / 2.0])
    flux = np.array([1.0, 1.0])
    err = np.array([1.0e-3, 1.0e-3])
    np.savetxt(path, np.column_stack([time, flux, err]))
    return str(path)


def _ltt_wiring_config(lc_file, light_travel_time=None):
    entry = {"name": "inst0", "file": lc_file, "band": "TESS"}
    if light_travel_time is not None:
        entry["light_travel_time"] = light_travel_time
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        # The Claret tie (on by default since #452) seeds the band's limb
        # darkening, which would move the stored pre-LTT golden fixture;
        # these tests are about light travel time, so it is off.
        "band": [
            {
                "name": "TESS",
                "filter": "TESS",
                "ld_law": "quadratic",
                "star_constrains_ld": False,
            }
        ],
        "transit": [entry],
    }


def _ltt_wiring_params(eccentric=False):
    """The shared wiring params.

    `eccentric=True` unpins the sqrt(e) pair off zero, and exists for exactly
    one reason: review 6.8.2 lets a STRUCTURALLY circular orbit -- both
    coordinates pinned at 0, which is what the default here is -- skip the
    Kepler solve outright, since f = E = M there.  Any test that COUNTS
    Kepler-solve Apply nodes as a proxy for "how many times the geometry was
    evaluated" therefore has nothing to count on a circular orbit, and must
    ask on an eccentric one instead.  The numeric tests keep the circular
    default, which is what makes their a/c comparison exact.
    """
    params = {
        "star.0.radius": {
            "initval": 2.0,
            "sigma": 0.05,
        },  # deliberately != 1 Rsun: catches an a/R* (dimensionless)
        # vs physical a [Rsun] mixup, which R* = 1 would hide numerically.
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.0, "sigma": 0.0},  # edge-on, sin_i = 1
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},  # circular, omega=0
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.0},
        # log_q pinned at its floor: mass ratio 1e-9, so the barycentric
        # factor m_primary/m_total is 1 to 9 significant figures -- doesn't
        # perturb the a/c comparison below.
        "planet.0.log_q": {"initval": -9.0, "sigma": 0.0},
    }
    if eccentric:
        params["orbit.0.secosw"] = {"initval": 0.1, "sigma": 0.0}
    return params


def _eval_at_point(node, model, point):
    fn = pytensor.function(
        [],
        node,
        givens=[
            (v, np.asarray(point[v.name]))
            for v in model.free_RVs
            if v.name in point
        ],
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    return fn()


def _count_kepler_solves(node):
    """Number of DISTINCT Kepler-solve Apply nodes among node's ancestors
    (sinf/cosf share one Apply node, so dedupe on the node, not the op)."""
    seen = set()
    for anc in pytensor.graph.ancestors([node]):
        if anc.owner is not None and isinstance(anc.owner.op, Kepler):
            seen.add(anc.owner)
    return len(seen)


def test_wired_ltt_delay_matches_a_over_c_through_real_accessors(tmp_path):
    """
    Given a transit-only system (star mass ~1 Msun, star RADIUS deliberately
    2 Rsun -- not 1, so a physical a [Rsun] and a dimensionless a/R* cannot
    be numerically confused --, period 365.25 d giving a ~ 1 AU via Kepler's
    third law, edge-on (cosi=0) circular (e=0) orbit, negligible planet
    mass) built through the real System/orbit/planet pipeline (NOT hand-
    typed a_rel/factor values),
    When build_likelihood's actual wiring (transit.py's call into
    ltt.retarded_time, using orbits.a/orbits.m_primary/orbits.m_total) is
    exercised by building the model, with a spy capturing the real delay
    tensor it produced,
    Then the delay at primary and secondary conjunction matches
    factor*a_rel/c (the SAME orbits.a/m_primary/m_total nodes the wiring
    itself used, read back independently) to 1e-6 relative -- confirming
    the units/accessor contract end to end -- and separately is close to
    the known ~499 s/AU light time in absolute terms, guarding against a
    shared-formula bug that an internal-consistency check alone would miss.
    Also confirms retarded_time is invoked exactly twice per build_model():
    once from build_likelihood's group loop (one oversample group here,
    ninterp=1) and once from compile_plotters -- the SAME builder
    (Transit._lc_model) called on the data and on the plot grid, so both
    are wired, live, not a dead branch.
    """
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    real_retarded_time = ltt.retarded_time
    calls = []

    def _spy(*args, **kwargs):
        result = real_retarded_time(*args, **kwargs)
        calls.append((args[0], result[1]))  # (t_grid, delay)
        return result

    with mock.patch(
        "exozippy.components.transit.transit.ltt.retarded_time",
        side_effect=_spy,
    ) as spy:
        system = System(
            _ltt_wiring_config(lc), user_params=_ltt_wiring_params()
        )
        system.prepare()
        model = system.build_model()
        # One from build_likelihood (stage 7), one from compile_plotters
        # (which build_model runs afterwards) -- both through _lc_model's
        # group loop, so both carry the 3-D (n_g, k_g, 1) sub-exposure
        # grid; the plot path used to build its own 2-D graph and smear
        # in NumPy afterwards.
        assert spy.call_count == 2

    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))

    assert [t_grid.ndim for t_grid, _ in calls] == [3, 3]
    # The first call is the likelihood's: stage 7 precedes the plotters.
    delay = _eval_at_point(calls[0][1], model, point)
    delay_primary = float(delay[0, 0, 0])
    delay_secondary = float(delay[1, 0, 0])

    orbit_map = system.planet.orbit_map
    a_rel_val = float(
        _eval_at_point(system.orbit.a.value[orbit_map][0], model, point)
    )
    m_primary_val = float(
        _eval_at_point(
            system.orbit.m_primary.value[orbit_map][0], model, point
        )
    )
    m_total_val = float(
        _eval_at_point(system.orbit.m_total.value[orbit_map][0], model, point)
    )
    factor_val = m_primary_val / m_total_val

    expected_amp_days = factor_val * a_rel_val / C_LIGHT_RSUN_PER_DAY

    assert abs(delay_primary) == pytest.approx(expected_amp_days, rel=1e-6)
    assert abs(delay_primary) * 86400.0 == pytest.approx(499.0, abs=2.0)
    assert (delay_secondary - delay_primary) == pytest.approx(
        2.0 * expected_amp_days, rel=1e-6
    )


# A companion massive enough that the occultation role factor is visible:
# (M1 - M2)/M = 0.538 here, against M1/M = 0.769 (EXOFASTv2's emission
# factor) and 1.0 (exoplanet's relative orbit).  At the 1e-9 mass ratio of
# the test above the three agree to 1e-9 and abs() hides the sign.
_MASSIVE_Q = 0.3


def _exact_massive_companion_delays(m_star, m_comp, period, tc, tc_target):
    """Independent reference for the two wired delays.

    a from Kepler's third law (astropy G), c from astropy, the role factor
    from the two component masses, and the retardation condition
    c d = -F z_rel(t - d) solved by fixed-point iteration for the circular
    edge-on orbit z_rel(T) = a cos(n (T - tc_target)) (+ = toward the
    observer, the companion in front at conjunction).  Returns
    (a [R_sun], F, c [R_sun/d], d_primary [d], d_secondary [d]).
    """
    import astropy.constants as ac
    import astropy.units as au

    c = ac.c.to(au.R_sun / au.day).value
    gm = (ac.G * (m_star + m_comp) * au.M_sun).to(au.R_sun**3 / au.day**2)
    a = float(np.cbrt(gm.value * period**2 / (4.0 * np.pi**2)))
    factor = (m_star - m_comp) / (m_star + m_comp)
    n = 2.0 * np.pi / period

    def solve(t):
        d = 0.0
        for _ in range(50):
            d = -factor * a * np.cos(n * ((t - tc_target) - d)) / c
        return d

    return a, factor, c, solve(tc), solve(tc + period / 2.0)


@pytest.mark.slow
def test_wired_occultation_delay_carries_the_mass_difference_factor(tmp_path):
    """
    Given the wiring system above with a 0.3 Msun companion (log_q pinned at
      log10 0.3) instead of a 1e-9 one,
    When the transit's own retarded_time call is captured and evaluated at
      the rows at tc and tc + P/2,
    Then each delay equals, SIGNED, the exact solution of
      c d = -[(M1 - M2)/M] z_rel(t - d) with a from Kepler's third law
      (astropy) -- d(tc) = -0.538 a/c, not -0.769 a/c (M1/M) or -a/c --,
      the observed tc maps onto the orbit's tc_target, and the
      primary-to-secondary offset equals the interval from the exact
      two-body occultation condition (each body at its own retarded time,
      test_ltt._exact_occultation_time).

    Tolerances: 1e-7 relative on the delays -- NOT the 1e-16 the closed
    form could reach: ltt.solve_delay's root (c/az)(sqrt(A^2 + B) - A)
    cancels catastrophically at B ~ 1e-8 and loses ~3e-8 relative (see
    the strict xfail test_solve_delay_root_is_free_of_cancellation below);
    the three role factors differ by O(1), so this still separates them by
    seven orders of magnitude.  2e-9 d on tc_target (the float64 resolution of t ~ 2.46e6 d);
    1e-3 s on the two-body interval, the tolerance test_ltt.py carries for
    the same first-order-in-a/c role factor.
    """
    from test_ltt import _exact_occultation_time

    # Arrange
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    params = _ltt_wiring_params()
    params["planet.0.log_q"] = {"initval": np.log10(_MASSIVE_Q), "sigma": 0.0}
    real_retarded_time = ltt.retarded_time
    calls = []

    def _spy(*args, **kwargs):
        result = real_retarded_time(*args, **kwargs)
        calls.append(result[1])
        return result

    # Act
    with mock.patch(
        "exozippy.components.transit.transit.ltt.retarded_time",
        side_effect=_spy,
    ):
        system = System(_ltt_wiring_config(lc), user_params=params)
        system.prepare()
        model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    # the first call is the likelihood's (stage 7 precedes the plotters)
    delay = _eval_at_point(calls[0], model, point)
    d_primary, d_secondary = float(delay[0, 0, 0]), float(delay[1, 0, 0])
    m_star = float(_eval_at_point(system.star.mass.value[0], model, point))
    m_comp = float(_eval_at_point(system.planet.mass.value[0], model, point))
    period = float(_eval_at_point(system.orbit.period.value[0], model, point))
    tc = float(_eval_at_point(system.orbit.tc.value[0], model, point))
    tc_target = float(
        _eval_at_point(system.orbit.tc_target.value[0], model, point)
    )

    # Assert
    assert m_comp / m_star == pytest.approx(_MASSIVE_Q, rel=1e-12)
    a, factor, c, d_p_ref, d_s_ref = _exact_massive_companion_delays(
        m_star, m_comp, period, tc, tc_target
    )
    assert d_primary == pytest.approx(d_p_ref, rel=1e-7)
    assert d_secondary == pytest.approx(d_s_ref, rel=1e-7)
    assert tc - d_primary == pytest.approx(tc_target, abs=2e-9)
    # teeth: the other two factors are hundreds of seconds away
    for wrong in (m_star / (m_star + m_comp), 1.0):
        assert abs(d_primary + wrong * a / c) * 86400.0 > 100.0

    f_primary = m_star / (m_star + m_comp)
    t_transit = _exact_occultation_time(-period / 4, f_primary, a, period, c)
    t_eclipse = _exact_occultation_time(+period / 4, f_primary, a, period, c)
    exact_offset = (t_eclipse - t_transit) - period / 2.0
    assert (d_secondary - d_primary) * 86400.0 == pytest.approx(
        exact_offset * 86400.0, abs=1e-3
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ltt.solve_delay computes its small root as (c/az)(sqrt(A^2+B) - A), "
        "which cancels catastrophically when B = 2 az (z0 - z)/c^2 ~ 1e-8: "
        "measured 3.1e-8 / 3.8e-8 relative (1.1e-10 / 1.3e-10 d, ~10 us) "
        "at the two conjunctions of a 1 AU, 0.3 Msun-companion orbit. The "
        "algebraically equal (c/az) B / (sqrt(A^2+B) + A) is exact to 1e-16."
    ),
)
def test_solve_delay_root_is_free_of_cancellation():
    """
    Given the line-of-sight kinematics of the massive-companion wiring
      test at its two conjunction rows (circular, edge-on, P = 365.25 d,
      a = 234.68 R_sun, occultation factor 0.538),
    When ltt.solve_delay returns the small root of
      0.5 az d^2 + (c - vz) d + (z - z0) = 0,
    Then it equals that root evaluated in 40-digit mpmath to 1e-13
      relative -- the float64 floor of a well-conditioned root.

    A pure-function pin on the precision loss that sets the 1e-7 tolerance
    of the two massive-companion wiring tests (here and in test_rm_ltt.py).
    """
    import mpmath

    # Arrange
    c = C_LIGHT_RSUN_PER_DAY
    factor, a = (1.0 - 0.3) / 1.3, 234.6816094227646
    n = 2.0 * np.pi / 365.25
    phase = n * np.array([-0.0033940739740563, 182.625 - 0.0033940739740563])
    z = -factor * a * np.cos(phase)
    vz = factor * a * n * np.sin(phase)
    az = factor * a * n * n * np.cos(phase)

    # Act
    got = np.asarray(ltt.solve_delay(z, vz, az, 0.0, c).eval(), dtype=float)

    # Assert
    mpmath.mp.dps = 40
    for gi, zi, vi, ai in zip(got, z, vz, az):
        A = 1 - mpmath.mpf(vi) / c
        B = 2 * mpmath.mpf(ai) * (0 - mpmath.mpf(zi)) / mpmath.mpf(c) ** 2
        ref = (mpmath.mpf(c) / ai) * (mpmath.sqrt(A * A + B) - A)
        assert abs(gi - float(ref)) <= 1e-13 * abs(float(ref)), (gi, ref)


def test_ltt_off_matches_pre_ltt_structure(tmp_path):
    """
    Given the same system, but with light_travel_time: False set on the
    transit file (the OFF branch, reached the real way now -- a per-file
    config key, not a monkeypatched flag),
    When the model is built,
    Then ltt.retarded_time is never called and the transit flux node's
    graph contains exactly ONE Kepler-solve Apply node -- i.e. the OFF
    branch is algebraically the pre-LTT computation (M = (t_grid - tp) * n
    fed straight to the group loop's own, sole, Kepler solve), not a
    second solve that happens to cancel out. The same system built with
    LTT ON (the default -- light_travel_time left unset) is confirmed to
    contain exactly TWO (the delay solve plus the corrected-time geometry
    solve), so this is a real structural difference, not a vacuous count.

    Uses the ECCENTRIC params: review 6.8.2 gives a structurally circular
    orbit a closed-form anomaly and no Kepler op at all, so on the circular
    default both counts would be one lower and the proxy would say nothing
    about LTT.
    """
    lc = _write_two_row_lc(tmp_path / "lc.dat")

    with mock.patch(
        "exozippy.components.transit.transit.ltt.retarded_time"
    ) as off_spy:
        system_off = System(
            _ltt_wiring_config(lc, light_travel_time=False),
            user_params=_ltt_wiring_params(eccentric=True),
        )
        system_off.prepare()
        system_off.build_model()
        off_spy.assert_not_called()

    assert _count_kepler_solves(system_off.transit._model_flux_node) == 1

    system_on = System(
        _ltt_wiring_config(lc), user_params=_ltt_wiring_params(eccentric=True)
    )
    system_on.prepare()
    system_on.build_model()
    assert _count_kepler_solves(system_on.transit._model_flux_node) == 2


def test_ltt_off_build_likelihood_reproduces_pre_ltt_model_values(tmp_path):
    """
    Given a STORED golden fixture (tests/fixtures/transit_ltt_off_reference.
    json) of transit._model_flux_node's output from the actual pre-LTT
    Transit class -- generated once from git commit 3ae5466 (the parent of
    df96bd9, the transit light-travel-time wiring commit; confirmed via grep
    at generation time to have no ltt/LIGHT_TRAVEL references), not a hand
    reimplementation of the (fairly involved) group loop that could itself
    drift from what the real old code did,
    When the SAME config/params (_ltt_wiring_config/_ltt_wiring_params, with
    light_travel_time: False set on the transit file -- the real, per-file
    config-key OFF path, not a monkeypatched flag) build a model with the
    CURRENT Transit,
    Then transit._model_flux_node evaluates to the SAME array the fixture
    recorded, to floating-point precision -- the structural "one Kepler
    solve, ltt never called" check in test_ltt_off_matches_pre_ltt_structure
    confirms the graph SHAPE didn't grow; this confirms the VALUES didn't
    move either, catching a stray bug in surrounding code (e.g. an
    accidental reorder of sin_i/cos_i, a mis-sliced index) that a shape-only
    check would miss. This is the backward-compat guard: with LTT off, the
    model is byte-for-byte the pre-LTT one, config key or not.

    Deliberately does NOT read git history (unlike an earlier version of
    this test, which shelled out to `git show HEAD:...` as its oracle): that
    self-invalidates the moment the LTT commit lands on HEAD, since HEAD
    then stops being the pre-LTT answer -- exactly what happened here. A
    frozen fixture has no such expiry.
    """
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    config = _ltt_wiring_config(lc, light_travel_time=False)
    params = _ltt_wiring_params()

    with open(_FIXTURES_DIR / "transit_ltt_off_reference.json") as f:
        fixture = json.load(f)
    mu_reference = np.asarray(fixture["model_flux"])

    system_off = System(config, user_params=params)
    system_off.prepare()
    model_off = system_off.build_model()
    with model_off:
        point_off = system_off.get_internal_point(
            model_off, system_off.get_raw_start(model_off)
        )
    mu_off = _eval_at_point(
        system_off.transit._model_flux_node, model_off, point_off
    )

    # rtol=1e-12/atol=1e-20: tight enough to catch a real discrepancy, loose
    # enough to absorb last-bit floating-point rounding between platforms
    # (see test_rm_ltt.py's analogous fix -- CI hit ~2.5e-29 absolute there
    # vs. exact equality on macOS, same 15-sig-fig agreement).
    np.testing.assert_allclose(mu_off, mu_reference, rtol=1e-12, atol=1e-20)


def _mixed_group_config(lc0, lc1):
    """Two transit files sharing a band/orbit/planet, both default ninterp=1
    (so _oversample_groups puts their rows in the SAME group) -- inst0 left
    at the light_travel_time default (True), inst1 explicitly False. This is
    the genuinely-mixed case the group loop's pt.where branch exists for.
    """
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "transit": [
            {"name": "inst0", "file": lc0, "band": "TESS"},
            {
                "name": "inst1",
                "file": lc1,
                "band": "TESS",
                "light_travel_time": False,
            },
        ],
    }


@pytest.mark.slow
def test_mixed_group_ltt_gradient_is_finite(tmp_path):
    """
    Given two transit files in the SAME oversample group (both ninterp=1,
    the default) but DIFFERENT light_travel_time settings -- inst0 on,
    inst1 off -- so the group loop must take its pt.where(lt_mask, ...)
    branch (test_ltt_off_matches_pre_ltt_structure's all-on/all-off cases
    never exercise this line at all),
    When the model's logp and its gradient are evaluated at the initial
    point, and a short NUTS chain is sampled with nuts_sampler="numpyro",
    Then both are finite throughout.

    This is a real check, not an assumption: the where-trap risk this
    pt.where COULD carry is that pytensor differentiates BOTH branches even
    where only one is selected, so if either branch had a singularity (like
    solve_delay's az=0 quadratic branch, which needed az_safe specifically
    because of this), the discarded branch's bad gradient can poison the
    selected one. Here both branches are just t_grid_corrected and t_grid --
    ordinary, everywhere-finite time arrays with no division, sqrt, or log
    anywhere near them -- so there is no candidate singularity for either
    branch to carry. This test confirms that holds in practice rather than
    asserting it from the formula alone.
    """
    lc0 = _write_two_row_lc(tmp_path / "lc0.dat")
    lc1 = _write_two_row_lc(tmp_path / "lc1.dat")
    config = _mixed_group_config(lc0, lc1)
    # Eccentric, for the same reason test_ltt_off_matches_pre_ltt_structure
    # is: the Kepler-solve count below is the evidence that the mixed group
    # took the pt.where branch, and review 6.8.2 leaves a structurally
    # circular orbit with no Kepler op to count.
    params = _ltt_wiring_params(eccentric=True)

    with mock.patch(
        "exozippy.components.transit.transit.ltt.retarded_time",
        side_effect=ltt.retarded_time,
    ) as spy:
        system = System(config, user_params=params)
        system.prepare()
        model = system.build_model()
        # One call for the (single, shared) oversample group in
        # build_likelihood, one for compile_plotters -- confirms the mixed
        # group actually took the "any() but not all()" branch and called
        # ltt.retarded_time at all (a bug that skipped the correction
        # entirely for a mixed group would also show 0 calls here).
        assert spy.call_count == 2

    assert _count_kepler_solves(system.transit._model_flux_node) == 2

    with model:
        point = model.initial_point()
        logp_val = model.compile_logp()(point)
        grad_val = model.compile_dlogp()(point)

    assert np.isfinite(logp_val), f"non-finite logp: {logp_val}"
    assert np.all(np.isfinite(grad_val)), (
        f"non-finite gradient in mixed-group pt.where: {grad_val}"
    )

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
