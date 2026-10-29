"""
Tests for fitthermal (PR 1.a: constant secondary-eclipse thermal emission).
Phase-curve variation (BEER: reflection, ellipsoidal, beaming) is not
modeled by this flag -- see PR 1.b.

  - Band: fitthermal config parsing and the per-band manifest opt-in gate
    (thermal pinned at sigma=0 unless a band asks for it, mirroring
    Instrument._register_gp's opt-in pattern).
  - Transit: backward compatibility (thermal pinned -> model bit-for-bit
    unchanged) and the eclipse shape itself (constant bump out of eclipse,
    dropping during secondary eclipse, present but not double-counted
    during primary transit).
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager
from exozippy.components.band.band import Band
from exozippy.components.transit import physics as transit_physics
from exozippy.system import System


def _make_band(config):
    return Band(config, _DummyConfigManager())


# ---------------------------------------------------------------------------
# Band-level unit tests
# ---------------------------------------------------------------------------


def test_load_data_fitthermal_defaults_false():
    """
    Given a band config with no fitthermal key,
    When load_data is called,
    Then fitthermal defaults to False.
    """
    band = _make_band([{}])
    band.load_data(system=None)
    assert band.fitthermal == [False]


def test_load_data_fitthermal_true():
    """
    Given a band config with fitthermal: true,
    When load_data is called,
    Then fitthermal is True for that band.
    """
    band = _make_band([{"fitthermal": True}])
    band.load_data(system=None)
    assert band.fitthermal == [True]


def test_register_parameters_thermal_absent_when_no_band_opts_in():
    """
    Given two bands, neither with fitthermal set,
    When register_parameters is called,
    Then thermal never enters the manifest at all: the parameter (and its
    table row) exists only when some band opts in with fitthermal: true.
    """
    band = _make_band([{}, {}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    assert "thermal" not in band.manifest


def test_register_parameters_thermal_free_only_for_opted_in_band():
    """
    Given two bands, only the second with fitthermal: true,
    When register_parameters is called,
    Then thermal is pinned (sigma=0) for the first band and left alone
    (NaN, so defaults.yaml's free-parameter behavior applies) for the
    second.
    """
    band = _make_band([{}, {"fitthermal": True}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    overrides = band.manifest["thermal"]["overrides"]
    assert overrides["sigma"][0] == 0.0
    assert np.isnan(overrides["sigma"][1])


def test_register_parameters_no_overrides_when_every_band_opts_in():
    """
    Given every band with fitthermal: true,
    When register_parameters is called,
    Then thermal has no overrides at all (free for every band, like u1).
    """
    band = _make_band([{"fitthermal": True}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    assert band.manifest["thermal"] == {}


# ---------------------------------------------------------------------------
# Pure-function tests of the shared occultation helper (no System/model
# needed -- this is what build_likelihood and compile_plotters both call,
# so a bug here is a bug in the actual fitted likelihood, not just plots).
# ---------------------------------------------------------------------------


def _eval_planet_visible(b_p, Z_p, r_p):
    b_t = pt.dvector("b_p")
    z_t = pt.dvector("Z_p")
    out = transit_physics.calc_planet_visible(b_t, z_t, r_p)
    fn = pytensor.function([b_t, z_t], out, on_unused_input="ignore")
    return fn(np.asarray(b_p, dtype=float), np.asarray(Z_p, dtype=float))


def test_planet_visible_is_one_in_front_of_star_regardless_of_b():
    """
    Given Z_p > 0 (planet in front of the star, primary transit),
    When calc_planet_visible is evaluated,
    Then it returns exactly 1.0, even at b values small enough that the
    swapped-geometry occultation calc alone would show a dip -- Z's sign
    must gate it, not b_p.
    """
    r_p = 0.1  # 1/r_p = 10
    b_p = [0.0, 0.05, 0.5, 5.0]  # includes values inside the swapped eclipse
    visible = _eval_planet_visible(b_p, [1.0] * len(b_p), r_p)
    np.testing.assert_allclose(visible, 1.0)


def test_planet_visible_matches_hand_derived_occultation_curve():
    """
    Given Z_p < 0 (planet behind the star) and r_p = 0.1 (so the swapped
    occulter radius is 1/r_p = 10, in planet radii),
    When calc_planet_visible is evaluated at b_p values chosen so that
    b_p / r_p sweeps the known ingress/egress region [r_swap - 1, r_swap + 1]
    = [9, 11],
    Then it matches the independently-verified quad_solution_vector
    uniform-disk occultation curve exactly -- this is the calculation the
    user asked to double check against exofast_tran.pro:105 (b_p must be
    divided by r_p, not just the radius ratio flipped to 1/r_p).
    """
    r_p = 0.1
    b_swap_targets = np.array([0.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0])
    b_p = b_swap_targets * r_p  # b_p = b_swap * r_p, i.e. b_swap = b_p / r_p
    visible = _eval_planet_visible(b_p, [-1.0] * len(b_p), r_p)
    # Independently computed (not via calc_planet_visible) with the star
    # as a uniform-disk occulter of radius 10 in planet radii -- see the
    # numeric check run against exoplanet_core.numpy.ops directly:
    #   b=0.0  -> 0.00000  (fully occulted)
    #   b=9.0  -> 0.00000
    #   b=9.5  -> 0.20266  (egress)
    #   b=10.0 -> 0.51061
    #   b=10.5 -> 0.81114
    #   b=11.0 -> 1.00000  (just clear)
    #   b=12.0 -> 1.00000  (fully visible)
    expected = [0.0, 0.0, 0.20266, 0.51061, 0.81114, 1.0, 1.0]
    np.testing.assert_allclose(visible, expected, atol=2e-5)


def test_planet_visible_bug_regression_unscaled_b_would_fail():
    """
    Given the exact bug flagged during review (dividing only the radius
    ratio by r_p, but forgetting to also divide the impact parameter b_p
    by r_p),
    When the same b_swap_targets are used without rescaling b_p,
    Then the occultation happens at the wrong separations entirely --
    demonstrating why this test suite pins the b_p/r_p scaling explicitly
    rather than only checking the radius ratio.
    """
    r_p = 0.1
    b_swap_targets = np.array([0.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0])
    b_p = b_swap_targets * r_p
    correct = _eval_planet_visible(b_p, [-1.0] * len(b_p), r_p)

    # Reimplement with the flagged bug: b_p used directly instead of b_p/r_p.
    b_t = pt.dvector("b_p")
    z_t = pt.dvector("Z_p")
    from exoplanet_core.pymc import ops as _ops

    r_swap = 1.0 / r_p
    sol_swap = _ops.quad_solution_vector(b_t, r_swap + pt.zeros_like(b_t))
    buggy_visible = pt.where(z_t > 0.0, 1.0, sol_swap[:, 0] / np.pi)
    fn = pytensor.function([b_t, z_t], buggy_visible, on_unused_input="ignore")
    buggy = fn(np.asarray(b_p, dtype=float), np.asarray([-1.0] * len(b_p)))

    # The bug puts the whole ingress/egress transition at b_p ~ [9, 11]
    # instead of b_p ~ [0.9, 1.1] (i.e. b_swap_targets*r_p), so at these
    # b_p values (already scaled by r_p) it reports "fully occulted"
    # (0.0) almost everywhere instead of the correct ingress/egress curve.
    assert not np.allclose(buggy, correct, atol=2e-5)


# ---------------------------------------------------------------------------
# Integration tests: System -> build_model -> evaluate the light curve
# ---------------------------------------------------------------------------


def _write_lc(path, t0=2459634.3, n=120, half_width=0.2):
    """Synthetic flat light curve bracketing one primary transit window."""
    rng = np.random.default_rng(42)
    t = np.linspace(t0 - half_width, t0 + half_width, n)
    flux = 1.0 + rng.normal(0.0, 1e-3, n)
    err = np.full(n, 1e-3)
    np.savetxt(path, np.column_stack([t, flux, err]))
    return str(path)


def _config(lc_file, fitthermal, light_travel_time=True):
    transit = {"name": "inst0", "file": lc_file, "band": "TESS"}
    if not light_travel_time:
        transit["light_travel_time"] = False
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [
            {
                "name": "TESS",
                "filter": "TESS",
                "ld_law": "quadratic",
                "fitthermal": fitthermal,
            }
        ],
        "transit": [transit],
    }


_TC = 2459634.3
_PERIOD = 2.99
_THERMAL_PPM = 5000.0


def _params(thermal_ppm=None):
    p = {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.05},
        # Fixed circular orbit: with e=0, the two conjunctions are exactly
        # half a period apart, so the secondary eclipse midpoint is exactly
        # predictable (tc + period/2) rather than shifted by the small
        # (0.01, 0.01) secosw/sesinw default.
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.7},
    }
    if thermal_ppm is not None:
        # Fixed (sigma=0) at a known value so the expected light-curve
        # bump/dip can be checked exactly rather than at a random initval.
        p["band.TESS.thermal"] = {"initval": thermal_ppm, "sigma": 0.0}
    return p


def _build(fitthermal, thermal_ppm, tmp_path_factory):
    d = tmp_path_factory.mktemp("transit_thermal")
    lc = _write_lc(d / "lc.dat")
    system = System(_config(lc, fitthermal), user_params=_params(thermal_ppm))
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point


def _flux_at(system, point, times):
    """Full model light curve (baseline + transit + thermal) at arbitrary
    times, via the same compiled evaluator plot_data/plot() use."""
    transit = system.transit
    param_values = transit._point_to_plot_params(point, system)
    full, _ = transit._lc_at_times(
        param_values, 0, np.asarray(times, dtype=float)
    )
    return full


@pytest.fixture(scope="module")
def thermal_off_system(tmp_path_factory):
    return _build(
        fitthermal=False, thermal_ppm=None, tmp_path_factory=tmp_path_factory
    )


@pytest.fixture(scope="module")
def thermal_on_system(tmp_path_factory):
    return _build(
        fitthermal=True,
        thermal_ppm=_THERMAL_PPM,
        tmp_path_factory=tmp_path_factory,
    )


def _write_lc_landmarks(path, centers, n_per_landmark=5, half_width=0.01):
    """Synthetic light curve with data points clustered around each of
    `centers` (so build_likelihood's mu, which only exists at the data's
    own times, can be evaluated near specific orbital phases -- e.g. both
    conjunctions -- unlike the plotting path, which accepts any time grid).
    """
    rng = np.random.default_rng(0)
    groups = [
        np.linspace(c - half_width, c + half_width, n_per_landmark)
        for c in centers
    ]
    time = np.concatenate(groups)
    flux = 1.0 + rng.normal(0.0, 1e-3, len(time))
    err = np.full(len(time), 1e-3)
    np.savetxt(path, np.column_stack([time, flux, err]))
    return str(path)


@pytest.fixture(scope="module")
def thermal_on_full_orbit(tmp_path_factory):
    """Same setup as thermal_on_system, but the light curve brackets
    primary transit, secondary eclipse, and a far/plateau phase, so the
    *actual likelihood* mu (not just the plotting path) can be checked at
    each landmark.

    Light-travel time is OFF here so the exact closed-form reference in
    test_thermal_light_curve_matches_the_closed_form_exofast_tran_model
    can phase everything from tc: with no file retarding the orbit,
    tc_target == tc exactly (orbit.md, "Gating"), and the geometry, the
    eclipse and the phase terms share one time base.  The LTT wiring is
    pinned in tests/test_transit_ltt.py."""
    d = tmp_path_factory.mktemp("transit_thermal_full_orbit")
    centers = [_TC, _TC + _PERIOD / 2.0, _TC + _PERIOD * 0.3]
    lc = _write_lc_landmarks(d / "lc.dat", centers)
    system = System(
        _config(lc, fitthermal=True, light_travel_time=False),
        user_params=_params(_THERMAL_PPM),
    )
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point, centers


def _likelihood_mu(system, model, point):
    """The real mu PyTensor node build_likelihood wired into
    pm.Normal("transit_likelihood", mu=..., ...) -- i.e. what the sampler
    actually fits against, as opposed to the separately-compiled plotting
    function (_compiled_full_lc) used elsewhere in this file."""
    rv = model.named_vars["transit_likelihood"]
    mu_node = rv.owner.inputs[2]
    fn = pytensor.function(
        [],
        mu_node,
        givens=[
            (v, np.asarray(point[v.name]))
            for v in model.free_RVs
            if v.name in point
        ],
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    return fn()


def _nearest_index(times, target):
    return int(np.argmin(np.abs(np.asarray(times) - target)))


def test_build_likelihood_mu_shows_eclipse_dip_and_transit_dip(
    thermal_on_full_orbit,
):
    """
    Given the actual build_likelihood mu (the tensor the sampler fits, not
    the plotting path),
    When evaluated at the data points nearest primary transit, secondary
    eclipse, and a far/plateau phase,
    Then: far ~ baseline + thermal ppm; secondary eclipse ~ bare baseline
    (thermal bump removed); primary transit is below the bare baseline
    (real transit depth still subtracted).

    This exercises the code path the mutation check during review found
    untested (compile_plotters and build_likelihood used to duplicate this
    math; they're now one shared helper, but this test still pins
    build_likelihood's mu directly rather than only the plotting output).
    """
    system, model, point, (t_primary, t_secondary, t_far) = (
        thermal_on_full_orbit
    )
    mu = _likelihood_mu(system, model, point)
    baseline = system.transit._point_value(point, system.transit.baseline, 0)
    times = system.transit.time

    mu_far = mu[_nearest_index(times, t_far)]
    mu_secondary = mu[_nearest_index(times, t_secondary)]
    mu_primary = mu[_nearest_index(times, t_primary)]

    assert mu_far == pytest.approx(baseline + _THERMAL_PPM * 1e-6, abs=1e-6)
    assert mu_secondary == pytest.approx(baseline, abs=1e-6)
    assert mu_primary < baseline


# ---------------------------------------------------------------------------
# Exact reference: exofast_tran.pro's light curve, written out in closed
# form and integrated independently of exoplanet-core (review 7.16.1).
# Shared with tests/test_transit_beer.py.
# ---------------------------------------------------------------------------


def _ref_ld_transit_flux(z, p, u1, u2):
    """Quadratic-limb-darkened flux of a star (radius 1) behind an opaque
    disk of radius p at center separation z, by direct 1-D integration.

    Each annulus of radius r loses the arc of it that lies inside the
    planet's disk, 2 arccos((r^2 + z^2 - p^2) / (2 r z)) radians, weighted
    by I(r) = 1 - u1 (1 - mu) - u2 (1 - mu)^2; the total is
    pi (1 - u1/3 - u2/6).  No Mandel-Agol / Agol (2020) algebra is shared
    with exoplanet-core, which is what the model uses.
    """
    from scipy.integrate import quad

    if z >= 1.0 + p:
        return 1.0

    def intensity(r):
        mu = np.sqrt(max(0.0, 1.0 - r * r))
        return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2

    def arc(r):
        if r <= 0.0:
            return 0.0
        if z == 0.0:
            return 2.0 * np.pi if r < p else 0.0
        c = (r * r + z * z - p * p) / (2.0 * r * z)
        if c <= -1.0:
            return 2.0 * np.pi
        if c >= 1.0:
            return 0.0
        return 2.0 * np.arccos(c)

    lo, hi = max(0.0, z - p), min(1.0, z + p)
    kinks = [k for k in (abs(p - z),) if lo < k < hi]
    blocked, _ = quad(
        lambda r: intensity(r) * r * arc(r),
        lo,
        hi,
        points=kinks or None,
        epsabs=1e-15,
        epsrel=1e-13,
        limit=500,
    )
    return 1.0 - blocked / (np.pi * (1.0 - u1 / 3.0 - u2 / 6.0))


def _ref_planet_visible(z, p):
    """Visible fraction of a UNIFORM planetary disk (radius p) behind the
    star (radius 1) at center separation z: one minus the textbook
    circle-circle lens area over pi p^2."""
    if z >= 1.0 + p:
        return 1.0
    if z <= 1.0 - p:
        return 0.0
    a1 = p * p * np.arccos((z * z + p * p - 1.0) / (2.0 * z * p))
    a2 = np.arccos((z * z + 1.0 - p * p) / (2.0 * z))
    a3 = 0.5 * np.sqrt(
        (-z + p + 1.0) * (z + p - 1.0) * (z - p + 1.0) * (z + p + 1.0)
    )
    return 1.0 - (a1 + a2 - a3) / (np.pi * p * p)


def _ref_exofast_tran(
    t,
    tc,
    period,
    ar,
    cosi,
    p,
    u1,
    u2,
    baseline,
    thermal=0.0,
    reflect=0.0,
    ellipsoidal=0.0,
):
    """exofast_tran.pro for a CIRCULAR orbit, in its own order (EXOFASTv2
    lines 97-145): transit flux, + thermal * planetvisible, - reflect/2
    (cos(phase) - 1) * planetvisible, then * (1 - ellipsoidal cos(2 phase)),
    then * f0.  Sky separation z = (a/R*) sqrt(sin^2 phase + cos^2 i
    cos^2 phase); the planet is in front of the star where cos(phase) > 0.
    Amplitudes in ppm.
    """
    t = np.asarray(t, dtype=float)
    phase = 2.0 * np.pi * (t - tc) / period
    z = ar * np.sqrt(np.sin(phase) ** 2 + cosi**2 * np.cos(phase) ** 2)
    front = np.cos(phase) > 0.0
    out = np.empty_like(t)
    for k in range(t.size):
        flux = _ref_ld_transit_flux(z[k], p, u1, u2) if front[k] else 1.0
        visible = 1.0 if front[k] else _ref_planet_visible(z[k], p)
        flux += 1e-6 * thermal * visible
        flux -= 1e-6 * reflect / 2.0 * (np.cos(phase[k]) - 1.0) * visible
        flux *= 1.0 - 1e-6 * ellipsoidal * np.cos(2.0 * phase[k])
        out[k] = baseline * flux
    return out


def _model_values(system, model, point):
    """The internal values the light curve is built from, at ``point``
    (the same givens _likelihood_mu uses)."""
    s = system
    nodes = {
        "tc": s.orbit.tc.value[0],
        "period": s.orbit.period.value[0],
        "ar": s.planet.ar.value[0],
        "cosi": s.orbit.cosi.value[0],
        "ecc": s.orbit.ecc.value[0],
        "p": s.planet.p.value[0],
        "u1": s.band.u1.value[0],
        "u2": s.band.u2.value[0],
        "baseline": s.transit.baseline.value[0],
    }
    fn = pytensor.function(
        [],
        [pt.as_tensor_variable(v) for v in nodes.values()],
        givens=[
            (v, np.asarray(point[v.name]))
            for v in model.free_RVs
            if v.name in point
        ],
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    return {k: float(v) for k, v in zip(nodes, fn())}


def _phase_curve_times(tc, period, ar, p):
    """Times through a whole orbit, dense across both conjunctions and
    their contact points, so the transit's limb-darkened ingress and the
    eclipse's partial phases are both sampled."""
    half = np.arcsin(min(1.0, (1.0 + p) / ar)) * period / (2.0 * np.pi)
    grids = [np.linspace(tc - 0.5 * period, tc + 0.5 * period, 61)]
    for center in (tc, tc + 0.5 * period):
        grids.append(np.linspace(center - 1.3 * half, center + 1.3 * half, 81))
    return np.sort(np.concatenate(grids))


def test_thermal_light_curve_matches_the_closed_form_exofast_tran_model(
    thermal_on_full_orbit,
):
    """
    Given fitthermal on (5000 ppm, fixed) on a circular orbit with light-
    travel time off,
    When the likelihood mu (at the data) and the plotting light curve (on a
    dense grid through both conjunctions) are evaluated,
    Then both equal exofast_tran.pro's model written out in closed form --
    transit flux + 1e-6 thermal planetvisible, times the baseline -- with
    the transit flux and planetvisible integrated independently of
    exoplanet-core, to 1e-10 (quadrature and float precision).

    The margin tests above only check signs at primary transit (flux <
    baseline): a thermal term that was double-counted, gated off, or gated
    by the transit's own flux there, or a primary dip shallower than the
    real Mandel-Agol depth, all pass them.  At mid-transit this pins
    mu = baseline * (F_transit + 1e-6 thermal) exactly.
    """
    # Arrange
    system, model, point, _centers = thermal_on_full_orbit
    v = _model_values(system, model, point)
    assert v["ecc"] == 0.0  # the closed form below is the circular one
    times = _phase_curve_times(v["tc"], v["period"], v["ar"], v["p"])
    ref_kwargs = {
        k: v[k]
        for k in ("tc", "period", "ar", "cosi", "p", "u1", "u2", "baseline")
    }

    # Act
    mu = _likelihood_mu(system, model, point)
    curve = _flux_at(system, point, times)

    # Assert
    np.testing.assert_allclose(
        mu,
        _ref_exofast_tran(
            system.transit.time, thermal=_THERMAL_PPM, **ref_kwargs
        ),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        curve,
        _ref_exofast_tran(times, thermal=_THERMAL_PPM, **ref_kwargs),
        rtol=0,
        atol=1e-10,
    )
    # Non-vacuity: the grid really holds a primary transit point with the
    # thermal term on top of it, and a partially eclipsed point.
    i_tc = _nearest_index(times, v["tc"])
    assert times[i_tc] == pytest.approx(v["tc"], abs=1e-8)
    assert curve[i_tc] < v["baseline"]
    phase = 2 * np.pi * (times - v["tc"]) / v["period"]
    z = v["ar"] * np.hypot(np.sin(phase), v["cosi"] * np.cos(phase))
    vis = [
        _ref_planet_visible(zz, v["p"])
        for zz, ph in zip(z, phase)
        if np.cos(ph) < 0
    ]
    assert any(0.05 < x < 0.95 for x in vis)


def test_thermal_pinned_is_not_a_free_parameter(thermal_off_system):
    """
    Given a band without fitthermal,
    When the model is built,
    Then band.thermal is not among the sampled free variables (sigma=0
    fixed it), and its value is exactly 0.
    """
    system, model, point = thermal_off_system
    assert not any("band.thermal" in n for n in model.named_vars)
    # No fitthermal anywhere: the parameter is not in the manifest at all
    # (no table row), and the transit consumer's gate is False outright.
    assert "thermal" not in system.band.manifest
    assert not hasattr(system.band, "thermal")
    assert system.band.thermal_may_be_nonzero() is False


def _count_quad_solution_ops(node):
    """Number of QuadSolutionVector applies feeding `node` -- one per
    primary-transit evaluation plus one per thermal planetvisible."""
    from pytensor.graph.traversal import io_toposort

    return sum(
        1
        for apply in io_toposort([], [node])
        if type(apply.op).__name__ == "QuadSolutionVector"
    )


def test_thermal_off_builds_no_occultation_graph(thermal_off_system):
    """
    Given a band without fitthermal (thermal pinned at 0),
    When the model is built,
    Then the transit model graph contains exactly one QuadSolutionVector
    per planet (the primary transit) and none for the thermal
    planetvisible -- the gate skips the dead branch outright (matching
    exofast_tran.pro's `if thermal ne 0d0`) instead of relying on the
    compiler to prune a multiply-by-zero.
    """
    system, model, point = thermal_off_system
    assert not system.band.thermal_may_be_nonzero()
    assert _count_quad_solution_ops(system.transit._model_flux_node) == 1


def test_thermal_on_builds_occultation_graph(thermal_on_system):
    """
    Given a band with fitthermal: true,
    When the model is built,
    Then the graph carries the second QuadSolutionVector (the swapped-
    geometry planetvisible) alongside the primary transit's -- the gate
    stays open for any band whose thermal can be nonzero.
    """
    system, model, point = thermal_on_system
    assert system.band.thermal_may_be_nonzero()
    assert _count_quad_solution_ops(system.transit._model_flux_node) == 2


def test_thermal_off_model_is_flat_away_from_transit(thermal_off_system):
    """
    Given fitthermal is off,
    When the light curve is evaluated far from primary transit and at the
    would-be secondary-eclipse time,
    Then both equal the plain baseline (no bump, no dip) -- the
    transit-only model is unchanged.
    """
    system, model, point = thermal_off_system
    baseline = system.transit._point_value(point, system.transit.baseline, 0)
    far = _flux_at(system, point, [_TC + _PERIOD * 0.3])[0]
    secondary = _flux_at(system, point, [_TC + _PERIOD / 2.0])[0]
    assert far == pytest.approx(baseline, abs=1e-9)
    assert secondary == pytest.approx(baseline, abs=1e-9)


def test_thermal_on_adds_constant_bump_away_from_either_conjunction(
    thermal_on_system,
):
    """
    Given fitthermal is on with thermal fixed at 5000 ppm,
    When the light curve is evaluated far from both conjunctions,
    Then flux is baseline + thermal_ppm * 1e-6 (planetvisible == 1: the
    star isn't occulting the planet's disk there).
    """
    system, model, point = thermal_on_system
    baseline = system.transit._point_value(point, system.transit.baseline, 0)
    far = _flux_at(system, point, [_TC + _PERIOD * 0.3])[0]
    assert far == pytest.approx(baseline + _THERMAL_PPM * 1e-6, abs=1e-6)


def test_thermal_on_drops_during_secondary_eclipse(thermal_on_system):
    """
    Given fitthermal is on,
    When the light curve is evaluated at the secondary-eclipse midpoint
    (tc + period/2, exact for this fixed circular orbit),
    Then flux has dropped back down close to the bare baseline (planet
    fully hidden: planetvisible ~ 0), well below the out-of-eclipse
    plateau.
    """
    system, model, point = thermal_on_system
    baseline = system.transit._point_value(point, system.transit.baseline, 0)
    plateau = _flux_at(system, point, [_TC + _PERIOD * 0.3])[0]
    secondary = _flux_at(system, point, [_TC + _PERIOD / 2.0])[0]
    # Should have lost most of the thermal bump...
    assert secondary < plateau - 0.5 * _THERMAL_PPM * 1e-6
    # ...and not overshoot below the bare baseline (planetvisible can't be
    # negative, and the primary-transit dip isn't active here since Z < 0).
    assert secondary == pytest.approx(baseline, abs=1e-6)


def test_thermal_on_primary_transit_is_still_a_dip_below_baseline(
    thermal_on_system,
):
    """
    Given fitthermal is on,
    When the light curve is evaluated at primary-transit center (tc),
    Then flux is below the bare baseline (the real transit depth is
    still subtracted -- the Z > 0 gate for `blocked` is untouched) even
    though the constant thermal term is fully present there
    (planetvisible == 1 in front of the star).
    """
    system, model, point = thermal_on_system
    baseline = system.transit._point_value(point, system.transit.baseline, 0)
    mid_transit = _flux_at(system, point, [_TC])[0]
    assert mid_transit < baseline


@pytest.mark.slow
def test_thermal_eclipse_is_exposure_smeared(tmp_path_factory):
    """
    Given fitthermal on (5000 ppm, fixed) AND exposure smearing
    (exptime=60 min, ninterp=21) on a light curve finely sampling the
    secondary eclipse,
    When the actual likelihood mu is evaluated,
    Then it equals the plotted model at the same times (the same
    expression, _lc_model, compiled on the plot grid -- so the plots show
    the smeared, thermal-included curve the fit scored) and NOT the
    instantaneous model of a sibling system without smearing -- proving
    the thermal term lives inside the oversampling group loop and gets
    smeared with the transit, exactly as EXOFASTv2 averages the full
    model (thermal included) over exofast_chi2v2.pro's grid.
    """
    d = tmp_path_factory.mktemp("thermal_smeared")
    t_sec = _TC + _PERIOD / 2.0
    t = np.linspace(t_sec - 0.1, t_sec + 0.1, 101)
    flux = np.ones_like(t)
    err = np.full_like(t, 1e-3)
    lc = str(d / "lc.dat")
    np.savetxt(lc, np.column_stack([t, flux, err]))

    config = _config(lc, fitthermal=True)
    config["transit"][0]["exptime"] = 60.0
    config["transit"][0]["ninterp"] = 21
    system = System(config, user_params=_params(_THERMAL_PPM))
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))

    mu = _likelihood_mu(system, model, point)

    tr = system.transit
    param_values = tr._point_to_plot_params(point, system)
    smeared, _ = tr._lc_at_times(param_values, 0, t)
    np.testing.assert_allclose(mu, smeared, atol=1e-12)

    # The instantaneous reference: the same light curve with no smearing
    # keys, at the same start point.
    system_instant = System(
        _config(lc, fitthermal=True), user_params=_params(_THERMAL_PPM)
    )
    system_instant.prepare()
    model_instant = system_instant.build_model()
    with model_instant:
        point_instant = system_instant.get_internal_point(
            model_instant, system_instant.get_raw_start(model_instant)
        )
    instantaneous = _likelihood_mu(
        system_instant, model_instant, point_instant
    )
    assert np.max(np.abs(mu - instantaneous)) > 1e-5
