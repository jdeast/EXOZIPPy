"""
Tests for BEER (PR 1.b: Beaming, Ellipsoidal, Reflection phase-curve
variations), building on fitthermal (PR 1.a).

  - Pure-function tests of transit/physics.py's calc_reflect_term,
    calc_ellipsoidal_factor, calc_beam_term: period, phase (peak/zero
    locations), and amplitude, each pinned against exofast_tran.pro's
    formulas (lines 128, 143, 151) -- plus one regression test per term
    encoding a plausible transcription bug, so a future edit that breaks
    the phase/period shape fails loudly rather than silently.
  - planet/physics.py's calc_beam_from_K: unit/magnitude sanity for the
    beam_constrains_mass formula.
  - Band/Planet manifest wiring: fitreflect/fitellip/beam_free/
    beam_constrains_mass parsing and the opt-in pinning gate (mirrors
    fitthermal's).
  - Integration: planetvisible gates thermal/reflect but not ellipsoidal,
    verified through the actual build_likelihood mu at secondary eclipse.
  - Backward compatibility: reflect/ellipsoidal/beam all pinned at exactly
    0 by default. The full-config bit-for-bit regression (kelt4/wasp18
    logp identical before/after this PR) was verified via git stash, not
    as a pytest fixture -- see PR notes.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager
from exozippy.components.band.band import Band
from exozippy.components.planet import physics as planet_physics
from exozippy.components.planet.planet import Planet
from exozippy.components.transit import physics as transit_physics
from exozippy.system import System

_TC = 2459634.3
_PERIOD = 2.99


def _make_band(config):
    return Band(config, _DummyConfigManager())


def _make_planet(config):
    return Planet(config, _DummyConfigManager())


# ---------------------------------------------------------------------------
# Pure-function tests: calc_reflect_term
# ---------------------------------------------------------------------------


def _eval(fn, *arrays):
    """Evaluate a transit_physics term function with dvector inputs."""
    syms = [pt.dvector(f"a{i}") for i in range(len(arrays))]
    out = fn(*syms)
    f = pytensor.function(syms, out, on_unused_input="ignore")
    return f(*[np.asarray(a, dtype=float) for a in arrays])


def test_reflect_zero_at_primary_full_amplitude_at_secondary():
    """
    Given planetvisible == 1 everywhere (isolating the phase shape from
    the eclipse gate),
    When calc_reflect_term is evaluated at primary transit, secondary
    eclipse, and quadrature,
    Then it is exactly 0 at primary transit, full amplitude at secondary
    eclipse, and half amplitude at quadrature -- exofast_tran.pro:128
    algebraically simplifies to reflect*0.5*(1-cos(phase)).
    """
    t = np.array([_TC, _TC + _PERIOD / 2.0, _TC + _PERIOD / 4.0])
    visible = np.ones_like(t)
    term = _eval(
        lambda tt, vv: transit_physics.calc_reflect_term(
            tt, _TC, _PERIOD, 100.0, vv
        ),
        t,
        visible,
    )
    np.testing.assert_allclose(term, [0.0, 100.0e-6, 50.0e-6], atol=1e-12)


def test_reflect_period_is_full_orbital_period():
    """
    Given calc_reflect_term,
    When evaluated one full period after a reference time,
    Then the value repeats exactly (full-period sinusoid, not half).
    """
    t0 = _TC + 0.37 * _PERIOD
    t = np.array([t0, t0 + _PERIOD])
    visible = np.ones_like(t)
    term = _eval(
        lambda tt, vv: transit_physics.calc_reflect_term(
            tt, _TC, _PERIOD, 100.0, vv
        ),
        t,
        visible,
    )
    assert term[0] == pytest.approx(term[1], abs=1e-12)


def test_reflect_planetvisible_gates_the_term():
    """
    Given planetvisible == 0 (deep secondary eclipse),
    When calc_reflect_term is evaluated even at its own peak phase,
    Then the term is exactly 0 -- reflection is gated by planetvisible.
    """
    t = np.array([_TC + _PERIOD / 2.0])
    term = _eval(
        lambda tt, vv: transit_physics.calc_reflect_term(
            tt, _TC, _PERIOD, 100.0, vv
        ),
        t,
        np.zeros_like(t),
    )
    assert term[0] == 0.0


def test_reflect_regression_naive_cosine_would_go_negative_at_transit():
    """
    Given the "non-physical" naive form exofast_tran.pro explicitly
    rejects (commented out at line 125: reflect*cos(phase), without the
    (1-cos)/2 shift),
    When compared against calc_reflect_term at primary transit,
    Then the naive form is negative there while calc_reflect_term is
    exactly 0 -- pinning why the shift matters, not just that it exists.
    """
    t = np.array([_TC])
    visible = np.ones_like(t)
    correct = _eval(
        lambda tt, vv: transit_physics.calc_reflect_term(
            tt, _TC, _PERIOD, 100.0, vv
        ),
        t,
        visible,
    )
    naive = _eval(
        lambda tt, vv: (
            -1e-6 * 100.0 * pt.cos(2.0 * np.pi * (tt - _TC) / _PERIOD) * vv
        ),
        t,
        visible,
    )
    assert correct[0] == pytest.approx(0.0, abs=1e-12)
    assert naive[0] < 0.0


# ---------------------------------------------------------------------------
# Pure-function tests: calc_ellipsoidal_factor
# ---------------------------------------------------------------------------


def test_ellipsoidal_dimmest_at_conjunctions_brightest_at_quadratures():
    """
    Given a positive ellipsoidal amplitude,
    When calc_ellipsoidal_factor is evaluated at both conjunctions and
    both quadratures,
    Then the brightness factor is below 1 at both conjunctions and above
    1 at both quadratures -- the star's tidal bulge presents its
    narrowest silhouette at conjunction, widest at quadrature.
    """
    conjunctions = np.array([_TC, _TC + _PERIOD / 2.0])
    quadratures = np.array([_TC + _PERIOD / 4.0, _TC + 3.0 * _PERIOD / 4.0])
    f_conj = _eval(
        lambda tt: transit_physics.calc_ellipsoidal_factor(
            tt, _TC, _PERIOD, 100.0
        ),
        conjunctions,
    )
    f_quad = _eval(
        lambda tt: transit_physics.calc_ellipsoidal_factor(
            tt, _TC, _PERIOD, 100.0
        ),
        quadratures,
    )
    assert np.all(f_conj < 1.0)
    assert np.all(f_quad > 1.0)
    # Symmetric: both conjunctions equally dim, both quadratures equally bright.
    np.testing.assert_allclose(f_conj[0], f_conj[1], atol=1e-12)
    np.testing.assert_allclose(f_quad[0], f_quad[1], atol=1e-12)


def test_ellipsoidal_period_is_half_the_orbital_period():
    """
    Given calc_ellipsoidal_factor,
    When evaluated half an orbital period after a reference time,
    Then the value repeats exactly (period/2 modulation).
    """
    t0 = _TC + 0.12 * _PERIOD
    t = np.array([t0, t0 + _PERIOD / 2.0])
    factor = _eval(
        lambda tt: transit_physics.calc_ellipsoidal_factor(
            tt, _TC, _PERIOD, 100.0
        ),
        t,
    )
    assert factor[0] == pytest.approx(factor[1], abs=1e-12)


def test_ellipsoidal_regression_full_period_would_give_one_dip_not_two():
    """
    Given the plausible bug of forgetting to halve the period (using the
    full orbital period instead of period/2),
    When compared against calc_ellipsoidal_factor at the two conjunctions
    (t=tc and t=tc+P/2),
    Then the buggy version treats the two conjunctions asymmetrically
    (one bright, one dim -- cos(0)=1 vs cos(pi)=-1 over a full period),
    while the correct (half-period) version is dim at both -- confirming
    the halving is what makes both conjunctions equivalent. (The two
    quadratures are a poor discriminator here: P/4 and 3P/4 of a *full*
    period both land on a cosine zero, so the buggy and correct forms
    coincidentally agree there.)
    """
    conjunctions = np.array([_TC, _TC + _PERIOD / 2.0])
    correct = _eval(
        lambda tt: transit_physics.calc_ellipsoidal_factor(
            tt, _TC, _PERIOD, 100.0
        ),
        conjunctions,
    )
    buggy = _eval(
        lambda tt: (
            1.0 - 1e-6 * 100.0 * pt.cos(2.0 * np.pi * (tt - _TC) / _PERIOD)
        ),
        conjunctions,
    )
    assert correct[0] == pytest.approx(correct[1], abs=1e-12)
    assert buggy[0] != pytest.approx(buggy[1], abs=1e-9)


# ---------------------------------------------------------------------------
# Pure-function tests: calc_beam_term
# ---------------------------------------------------------------------------


def test_beam_zero_at_both_conjunctions_extremal_at_quadratures():
    """
    Given calc_beam_term,
    When evaluated at both conjunctions and both quadratures,
    Then it is exactly 0 at both conjunctions (RV crosses zero there) and
    extremal with opposite sign at the two quadratures (velocity-tracking
    sine, not a cosine).
    """
    conjunctions = np.array([_TC, _TC + _PERIOD / 2.0])
    quadratures = np.array([_TC + _PERIOD / 4.0, _TC + 3.0 * _PERIOD / 4.0])
    at_conj = _eval(
        lambda tt: transit_physics.calc_beam_term(tt, _TC, _PERIOD, 100.0),
        conjunctions,
    )
    at_quad = _eval(
        lambda tt: transit_physics.calc_beam_term(tt, _TC, _PERIOD, 100.0),
        quadratures,
    )
    np.testing.assert_allclose(at_conj, [0.0, 0.0], atol=1e-12)
    assert at_quad[0] == pytest.approx(100.0e-6, abs=1e-12)
    assert at_quad[1] == pytest.approx(-100.0e-6, abs=1e-12)


def test_beam_period_is_full_orbital_period():
    """
    Given calc_beam_term,
    When evaluated one full period after a reference time,
    Then the value repeats exactly.
    """
    t0 = _TC + 0.19 * _PERIOD
    t = np.array([t0, t0 + _PERIOD])
    term = _eval(
        lambda tt: transit_physics.calc_beam_term(tt, _TC, _PERIOD, 100.0),
        t,
    )
    assert term[0] == pytest.approx(term[1], abs=1e-12)


def test_beam_has_no_planetvisible_parameter():
    """
    Given calc_beam_term's signature,
    When inspected,
    Then it takes no planetvisible argument at all -- a structural
    guarantee (not just an empirical one at a sampled point) that beaming
    can never be gated by the planet's occultation state, matching
    exofast_tran.pro:146-152 having no *planetvisible factor.
    """
    import inspect

    params = list(inspect.signature(transit_physics.calc_beam_term).parameters)
    assert "planetvisible" not in params


def test_beam_regression_cosine_would_move_the_zero_crossings():
    """
    Given the plausible transcription bug of using cos instead of sin,
    When compared against calc_beam_term at the conjunctions,
    Then the buggy (cosine) version is at its extremum at conjunction
    (where the correct sine version is exactly 0) -- pinning that this
    must be an odd function of phase, not an even one.
    """
    conjunctions = np.array([_TC, _TC + _PERIOD / 2.0])
    correct = _eval(
        lambda tt: transit_physics.calc_beam_term(tt, _TC, _PERIOD, 100.0),
        conjunctions,
    )
    buggy = _eval(
        lambda tt: 1e-6 * 100.0 * pt.cos(2.0 * np.pi * (tt - _TC) / _PERIOD),
        conjunctions,
    )
    np.testing.assert_allclose(correct, [0.0, 0.0], atol=1e-12)
    assert abs(buggy[0]) == pytest.approx(100.0e-6, abs=1e-12)


# ---------------------------------------------------------------------------
# Pure-function test: calc_beam_from_K (beam_constrains_mass)
# ---------------------------------------------------------------------------


def test_calc_beam_from_k_scales_linearly_and_is_physically_sized():
    """
    Given calc_beam_from_K,
    When evaluated at a WASP-18b-like K (~1800 m/s, one of the largest
    known RV amplitudes) converted to internal units (solRad/d),
    Then the result scales linearly with K and lands in the tens-of-ppm
    range expected for Doppler beaming (not, e.g., off by the
    solRad/day<->m/s conversion factor, which would be ~1e7x too big/small).
    """
    from exozippy.constants import SOLRAD_PER_DAY_TO_MPS

    k_mps = 1800.0
    k_internal = k_mps / SOLRAD_PER_DAY_TO_MPS
    k_sym = pt.dscalar("K")
    beam_ppm = pytensor.function(
        [k_sym], planet_physics.calc_beam_from_K(k_sym)
    )(k_internal)

    assert 1.0 < beam_ppm < 100.0  # sanity range for a very large K
    # Linear in K:
    beam_ppm_2x = pytensor.function(
        [k_sym], planet_physics.calc_beam_from_K(k_sym)
    )(2.0 * k_internal)
    assert beam_ppm_2x == pytest.approx(2.0 * beam_ppm, rel=1e-9)


# ---------------------------------------------------------------------------
# Band manifest wiring: fitreflect / fitellip
# ---------------------------------------------------------------------------


def test_band_fitreflect_fitellip_default_false():
    band = _make_band([{}])
    band.load_data(system=None)
    assert band.fitreflect == [False]
    assert band.fitellip == [False]


def test_band_reflect_ellip_pinned_when_not_opted_in():
    band = _make_band([{}, {"fitreflect": True}, {"fitellip": True}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    reflect_sigma = band.manifest["reflect"]["overrides"]["sigma"]
    ellip_sigma = band.manifest["ellipsoidal"]["overrides"]["sigma"]
    assert reflect_sigma[0] == 0.0
    assert np.isnan(reflect_sigma[1])
    assert reflect_sigma[2] == 0.0
    assert ellip_sigma[0] == 0.0
    assert np.isnan(ellip_sigma[2])
    assert ellip_sigma[1] == 0.0


# ---------------------------------------------------------------------------
# Planet manifest wiring: beam_free / beam_constrains_mass
# ---------------------------------------------------------------------------


def test_planet_beam_free_beam_constrains_mass_default_false():
    planet = _make_planet([{"name": "b"}])
    assert planet.beam_free == [False]
    assert planet.beam_constrains_mass == [False]


def test_planet_beam_absent_when_neither_set():
    """Neither beam flag set anywhere: beam never enters the manifest
    (no parameter, no table row) -- the same opt-in gating as Band's
    thermal/reflect/ellipsoidal."""
    planet = _make_planet([{"name": "b"}])
    planet.register_parameters(system=_FakeSystemNoOrbit())
    assert "beam" not in planet.manifest


def test_planet_beam_free_with_beam_free():
    planet = _make_planet([{"name": "b", "beam_free": True}])
    planet.register_parameters(system=_FakeSystemNoOrbit())
    assert planet.manifest["beam"] == {}


def test_planet_beam_uses_expression_with_beam_constrains_mass():
    planet = _make_planet([{"name": "b", "beam_constrains_mass": True}])
    planet.register_parameters(system=_FakeSystemWithOrbit())
    assert planet.manifest["beam"] == "default"


def test_planet_beam_free_and_beam_constrains_mass_together_derives():
    """
    Given both beam_free and beam_constrains_mass set on the same planet,
    When register_parameters runs,
    Then beam_constrains_mass wins: beam is still derived from K via the
    "default" expression rather than fit freely. Per EXOFASTv2's
    step2pars.pro (~line 256) the two flags are not mutually exclusive --
    beam is computed whenever either is set.
    """
    planet = _make_planet(
        [{"name": "b", "beam_free": True, "beam_constrains_mass": True}]
    )
    planet.register_parameters(system=_FakeSystemWithOrbit())
    assert planet.manifest["beam"] == "default"


def test_beam_constrains_mass_without_orbit_raises():
    planet = _make_planet([{"name": "b", "beam_constrains_mass": True}])
    with pytest.raises(ValueError, match="requires an orbit component"):
        planet.register_parameters(system=_FakeSystemNoOrbit())


class _FakeSystemNoOrbit:
    active_components = {}


class _FakeSystemWithOrbit:
    active_components = {"orbit": None}


def test_beam_off_does_not_require_K_no_orbit_config():
    """
    Given a real System with a star and a planet but no orbit component
    (e.g. a bare mass/radius characterization, or the shape of a
    microlensing lens-companion config -- KMT-2019-BLG-1806 in
    test_sed_flux_constraints.py is exactly this shape) and beam left at
    its default (off),
    When the system is prepared and the model is built,
    Then it builds successfully and planet.beam evaluates to exactly 0 --
    beam must never require K to exist just because K is beam's
    expression dependency in the *beam_constrains_mass* case.

    Regression test for a real bug found in review: Planet.register_
    parameters's off/beam_free manifest entries are dicts with no explicit
    "expr_key" ({} or {"overrides": ...}), which is supposed to mean "no
    expression" -- but graph.py's build-order step had a bug (see
    graph.py's determine_pymc_build_order) that treated any non-None dict
    as implicitly requesting the "default" expression regardless, so it
    tried to require planet.K as a dependency of planet.beam even with
    beam pinned off. This never surfaced for thermal/reflect/ellipsoidal
    because they have no `expressions` block in defaults.yaml at all --
    beam is the first opt-in ppm term that also has a real expression
    (calc_beam_from_K, for beam_constrains_mass), which is what exposed it.
    The Planet-only tests above (test_planet_beam_pinned_when_neither_set
    etc.) use a fake system and call register_parameters directly, so
    they never exercised graph.py's build-order step at all -- this is
    the one that would have caught it.
    """
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
    }
    params = {
        "star.0.radius": {"initval": 1.0, "sigma": 0.05},
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
    }
    system = System(config, user_params=params)
    system.prepare()
    system.build_model()
    # beam is absent from the manifest entirely (opt-in gating); the
    # regression this test guards -- graph.py demanding K for a
    # pinned-off beam -- is covered by the build succeeding at all.
    assert "beam" not in system.planet.manifest


# ---------------------------------------------------------------------------
# Integration: planetvisible gates thermal/reflect but NOT ellipsoidal
# ---------------------------------------------------------------------------


def _write_lc_landmarks(path, centers, n_per_landmark=5, half_width=0.01):
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


def _beer_config(lc_file):
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [
            {
                "name": "TESS",
                "filter": "TESS",
                "ld_law": "quadratic",
                "fitthermal": True,
                "fitreflect": True,
                "fitellip": True,
            }
        ],
        "transit": [{"name": "inst0", "file": lc_file, "band": "TESS"}],
    }


_THERMAL_PPM = 5000.0
_REFLECT_PPM = 2000.0
_ELLIP_PPM = 300.0


def _beer_params():
    return {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.05},
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.7},
        "band.TESS.thermal": {"initval": _THERMAL_PPM, "sigma": 0.0},
        "band.TESS.reflect": {"initval": _REFLECT_PPM, "sigma": 0.0},
        "band.TESS.ellipsoidal": {"initval": _ELLIP_PPM, "sigma": 0.0},
    }


def _likelihood_mu(system, model, point):
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


@pytest.fixture(scope="module")
def beer_system(tmp_path_factory):
    d = tmp_path_factory.mktemp("transit_beer")
    centers = [_TC, _TC + _PERIOD / 2.0, _TC + _PERIOD * 0.3]
    lc = _write_lc_landmarks(d / "lc.dat", centers)
    system = System(_beer_config(lc), user_params=_beer_params())
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point, centers


def test_ellipsoidal_survives_secondary_eclipse_thermal_and_reflect_dont(
    beer_system,
):
    """
    Given thermal, reflect, and ellipsoidal all active on the same band,
    When build_likelihood's actual mu is evaluated at the secondary
    eclipse center (planetvisible ~ 0) versus a far/plateau phase,
    Then thermal+reflect's combined contribution collapses at eclipse
    (both gated by planetvisible), while the *drop* in flux there still
    differs from what a thermal+reflect-only (no ellipsoidal) model would
    predict -- i.e. ellipsoidal's multiplicative dimming at conjunction is
    still present, not gated away by the eclipse.
    """
    system, model, point, (t_primary, t_secondary, t_far) = beer_system
    mu = _likelihood_mu(system, model, point)
    baseline = system.transit._baseline_for(point, 0)
    times = system.transit.time

    mu_far = mu[_nearest_index(times, t_far)]
    mu_secondary = mu[_nearest_index(times, t_secondary)]

    # Far from either conjunction: thermal+reflect near full amplitude,
    # ellipsoidal near 1 (deviation is tiny away from conjunction/quadrature
    # for this phase, dominated here by thermal/reflect).
    assert mu_far > baseline

    # At secondary eclipse center: thermal+reflect have collapsed toward 0
    # added ppm (planetvisible ~ 0), but ellipsoidal's conjunction dimming
    # is a *conjunction*-referenced effect (not eclipse-referenced) and is
    # still multiplying the flux -- so mu_secondary should sit measurably
    # below the bare baseline, not land exactly on it.
    assert mu_secondary < baseline
    # And thermal/reflect's own contribution is negligible there (the
    # eclipse is close to total for this system's geometry):
    assert mu_far - mu_secondary > 0.5 * ((_THERMAL_PPM + _REFLECT_PPM) * 1e-6)


def test_reflect_ellip_beam_are_exactly_zero_by_default(tmp_path_factory):
    """
    Given a band with no fitreflect/fitellip and a planet with no
    beam_free/beam_constrains_mass,
    When the model is built,
    Then band.reflect, band.ellipsoidal, and planet.beam all evaluate to
    exactly 0 -- the same guarantee already established for thermal.
    """
    d = tmp_path_factory.mktemp("transit_beer_off")
    lc = _write_lc_landmarks(d / "lc.dat", [_TC])
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "inst0", "file": lc, "band": "TESS"}],
    }
    params = {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }
    system = System(config, user_params=params)
    system.prepare()
    system.build_model()
    # Opt-in gating: with no fitreflect/fitellip/beam flag anywhere the
    # parameters do not exist at all -- stronger than "evaluates to 0",
    # and no table rows either.
    assert "reflect" not in system.band.manifest
    assert "ellipsoidal" not in system.band.manifest
    assert "beam" not in system.planet.manifest
    assert not hasattr(system.band, "reflect")
    assert not hasattr(system.planet, "beam")


def test_beam_diluted_when_sed_dilution_active(tmp_path_factory):
    """
    Given a two-identical-star system (SED dilution = 0.5, same setup as
    test_sed_deblending_dilutes_transit_depth in test_transit_band.py)
    with beam_free on the planet and thermal/reflect/ellipsoidal left at
    their default off,
    When build_likelihood's actual mu is evaluated at quadrature phase
    (tc + period/4 -- away from any transit/eclipse, where beam is
    extremal and is the only active BEER term),
    Then the beam deviation from baseline equals the UNDILUTED beam term
    times the system's own transit.dilution factor (0.5), not the bare
    undiluted term.

    Regression test for the PR-review fix that folds beam into the
    dil_obs/dil_node scaling (EXOFASTv2 parity: exofast_chi2v2.pro:
    1517/1556 pass beam and dilute together into exofast_tran, which
    dilutes beam at exofast_tran.pro:157, after adding it at :146).
    Before that fix, `actual` below would equal the bare undiluted term
    instead.
    """
    d = tmp_path_factory.mktemp("transit_beer_beam_dilution")
    t_quad = _TC + _PERIOD / 4.0
    lc = _write_lc_landmarks(d / "lc.dat", [t_quad])
    sed_file = d / "two_star.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")

    config = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "planet": [{"name": "b", "beam_free": True}],
        "orbit": [{"name": "b"}],
        "band": [
            {"name": "V", "filter": "V", "ld_law": "quadratic", "star_ndx": 0}
        ],
        "transit": [{"name": "inst0", "file": lc, "band": "V"}],
        "sed": {"file": str(sed_file)},
    }
    _BEAM_PPM = 4000.0
    params = {
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.08},
        "star.B.radius": {"initval": 1.0, "sigma": 0.05},
        "star.B.mass": {"initval": 1.0, "sigma": 0.05},
        "star.B.teff": {"initval": 5800, "sigma": 100},
        "star.B.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.05},
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.7},
        "planet.0.beam": {"initval": _BEAM_PPM, "sigma": 0.0},
    }

    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    assert "transit.dilution" in model.named_vars

    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    mu = _likelihood_mu(system, model, point)
    baseline = system.transit._baseline_for(point, 0)
    times = system.transit.time
    idx = _nearest_index(times, t_quad)
    t_actual = times[idx]

    dil_fn = pytensor.function(
        [],
        model.named_vars["transit.dilution"],
        givens=[
            (v, np.asarray(point[v.name]))
            for v in model.free_RVs
            if v.name in point
        ],
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    dil = float(dil_fn()[0])
    assert dil == pytest.approx(0.5, abs=1e-6)

    phase = 2.0 * np.pi * (t_actual - _TC) / _PERIOD
    undiluted_beam = 1e-6 * _BEAM_PPM * np.sin(phase)
    expected = dil * undiluted_beam

    actual = mu[idx] - baseline
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)
    # Sanity: the undiluted prediction should NOT match -- if it did, the
    # dilution-fold-in fix regressed back to skipping beam.
    assert actual != pytest.approx(undiluted_beam, rel=1e-6)


def test_ellipsoidal_diluted_when_sed_dilution_active(tmp_path_factory):
    """
    Given the same two-identical-star diluted system as the beam-dilution
      test, but with fitellip on and everything else off,
    When build_likelihood's mu is evaluated at secondary conjunction
      (tc + P/2 -- the ellipsoidal extremum, no transit in the data),
    Then the deviation from baseline is the UNDILUTED ellipsoidal
      deviation times the dilution factor (0.5) -- exofast_tran.pro
      applies the dilution to (modelflux - 1) AFTER the ellipsoidal
      factor multiplies in, so ellipsoidal IS diluted there (PR #53
      review finding; the implementation used to let it escape).
    """
    d = tmp_path_factory.mktemp("transit_beer_ellip_dilution")
    t_conj = _TC + _PERIOD / 2.0
    lc = _write_lc_landmarks(d / "lc.dat", [t_conj])
    sed_file = d / "two_star.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")

    config = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [
            {
                "name": "V",
                "filter": "V",
                "ld_law": "quadratic",
                "star_ndx": 0,
                "fitellip": True,
            }
        ],
        "transit": [{"name": "inst0", "file": lc, "band": "V"}],
        "sed": {"file": str(sed_file)},
    }
    _ELLIP = 4000.0
    params = {
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.08},
        "star.B.radius": {"initval": 1.0, "sigma": 0.05},
        "star.B.mass": {"initval": 1.0, "sigma": 0.05},
        "star.B.teff": {"initval": 5800, "sigma": 100},
        "star.B.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.05},
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.7},
        "band.V.ellipsoidal": {"initval": _ELLIP, "sigma": 0.0},
    }

    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    mu = _likelihood_mu(system, model, point)
    baseline = system.transit._baseline_for(point, 0)
    times = system.transit.time
    idx = _nearest_index(times, t_conj)
    t_actual = times[idx]

    # ellip factor - 1 at this exact time (half-period cosine)
    phase = 2.0 * np.pi * (t_actual - _TC) / (_PERIOD / 2.0)
    undiluted_dev = -1e-6 * _ELLIP * np.cos(phase)
    dil = 0.5  # two identical stars
    # Secondary conjunction: no transit/eclipse terms in this config, so
    # mu = baseline * (1 + diluted ellipsoidal deviation).
    expected = baseline * dil * undiluted_dev

    actual = mu[idx] - baseline
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)
    # Sanity: the undiluted prediction must NOT match.
    assert actual != pytest.approx(baseline * undiluted_dev, rel=1e-6)
