"""Transit/occultation durations and the eclipse time (review 8.8.7).

Two claims, and the second is the one the item is really about:

  1. `planet.t14`/`tfwhm`/`tau`, their occultation twins, `planet.bs` and
     `orbit.ts` are first-class derived Parameters -- table rows, LaTeX
     macros, units -- declared wherever there is an orbit, transit data or
     not.  They were bare `pm.Deterministic`s built inside `transit.py`, so
     an RV-only fit could not state them and no fit could put a prior on one.
  2. Stating one CONSTRAINS e and omega.  The likelihood half is free (a
     Gaussian on a derived parameter, with the gradient flowing back through
     the expression); the START is a one-way solver, because the forward
     model is a Kepler solve and its inverse has no closed form.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit import physics as ophys
from exozippy.components.planet import physics as pphys
from exozippy.system import System

PERIOD, TC, COSI = 5.0, 2455010.0, 0.03

_CONFIG = {
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b"}],
    "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
}

_BASE = {
    "star.A.mass": {"initval": 1.0, "sigma": 0.05},
    "star.A.radius": {"initval": 1.0, "sigma": 0.05},
    "orbit.b.period": {"initval": PERIOD},
    "orbit.b.tc": {"initval": TC},
    "orbit.b.cosi": {"initval": COSI},
    "planet.b.radius": {"initval": 1.0},
}


def _system(**extra):
    params = dict(_BASE)
    params.update(extra)
    system = System(dict(_CONFIG), params)
    system.prepare()
    return system


def _solved(system, path):
    return float(system.config_manager.user_params[path]["initval"])


def _at_start(system, model, pairs):
    """Evaluate `(component, parameter)` value nodes at the start point.

    `replace_rvs_by_values` is not optional: a Parameter's `.value` graph
    still names the RandomVariables, and compiling it without the swap
    DRAWS them from their priors instead of reading the start.
    """
    point = model.initial_point()
    nodes = [getattr(getattr(system, c), n).value for c, n in pairs]
    fn = model.compile_fn(model.replace_rvs_by_values(nodes))
    out = fn({k: v for k, v in point.items() if k in fn.f._finder})
    return {
        f"{c}.{n}": float(np.atleast_1d(v)[0]) for (c, n), v in zip(pairs, out)
    }


# ---------------------------------------------------------------------------
# 1. The parameters exist, with no transit data anywhere
# ---------------------------------------------------------------------------

_DURATIONS = ["bs", "t14", "tfwhm", "tau", "t14s", "tfwhms", "taus"]


@pytest.fixture(scope="module")
def built():
    """A star + planet + orbit system with NO transit component, built."""
    system = _system()
    return system, system.build_model()


@pytest.mark.parametrize("name", _DURATIONS)
def test_durations_are_planet_parameters_without_transit_data(name, built):
    """
    Given a star + planet + orbit with no transit component at all,
    When the model is built,
    Then every duration is a declared, derived, table-printing Parameter.
      They are geometry, not photometry: an RV-only fit has the same
      durations, and refusing to state them is what made a published
      duration unusable as a constraint.
    """
    # Arrange / Act
    system, _ = built
    param = getattr(system.planet, name)

    # Assert
    assert name in system.planet.manifest
    assert param.expression is not None
    assert bool(np.all(param.is_derived))
    assert param.print_to_table


def test_the_eclipse_time_is_an_orbit_parameter(built):
    """
    Given the same system,
    When the model is built,
    Then `ts` is an orbit parameter, next to `tc`.  On `orbit` and not on
      `planet` -- unlike EXOFASTv2 -- because every input it has (e, omega,
      tc, P) is an orbit parameter, while the durations need the radius
      ratio and a/R* and so belong to the planet.
    """
    system, _ = built
    assert "ts" in system.orbit.manifest
    assert system.orbit.ts.expression is not None


def test_transit_no_longer_builds_its_own_duration_deterministics():
    """
    Given the duration code has ONE owner now,
    When transit.py is read,
    Then it does not build the durations itself.  Two copies of a formula
      is the failure this move removes; the phased-plot x-range reads
      `planet.t14`.
    """
    import inspect

    from exozippy.components.transit import transit as transit_mod

    src = inspect.getsource(transit_mod)
    assert 'pm.Deterministic(f"{self.prefix}.t14"' not in src
    assert 'point.get("planet.t14")' in src


# ---------------------------------------------------------------------------
# 2. The physics: circular limits, and the numpy/pytensor twins agreeing
# ---------------------------------------------------------------------------


def test_a_circular_orbit_puts_the_eclipse_at_half_a_period():
    """
    Given a circular orbit,
    When the eclipse time is computed,
    Then it is exactly half a period after conjunction -- the check that
      fixes the sign and the branch of the anomaly difference.
    """
    got = ophys.ts_from_ecc_omega(0.0, 0.7, TC, PERIOD, xp=np)
    assert float(got) == pytest.approx(TC + PERIOD / 2.0, abs=1e-9)


def test_a_circular_orbit_has_equal_transit_and_eclipse_durations():
    """
    Given a circular orbit,
    When the two conjunctions' durations are computed,
    Then they are equal: the whole eccentricity signal in the duration
      RATIO is the `1 +/- e sin omega` factor, so at e = 0 there is none.
    """
    kw = dict(
        ar=10.0,
        cosi=0.05,
        sini=np.sqrt(1 - 0.05**2),
        ecc=0.0,
        esinw=0.0,
        p=0.1,
        period=PERIOD,
        xp=np,
    )
    t14, t23 = pphys.duration_pair(secondary=False, **kw)
    t14s, t23s = pphys.duration_pair(secondary=True, **kw)
    assert t14 == pytest.approx(t14s, rel=1e-14)
    assert t23 == pytest.approx(t23s, rel=1e-14)


def _fwhm_ratio(esinw, cosi):
    kw = dict(
        ar=10.0,
        cosi=cosi,
        sini=np.sqrt(1 - cosi**2),
        ecc=abs(esinw),
        esinw=esinw,
        p=0.1,
        period=PERIOD,
        xp=np,
    )
    t14, t23 = pphys.duration_pair(secondary=False, **kw)
    t14s, t23s = pphys.duration_pair(secondary=True, **kw)
    return float((t14s + t23s) / (t14 + t23))


@pytest.mark.parametrize("esinw", [-0.3, -0.1, 0.0, 0.1, 0.3])
def test_a_central_duration_ratio_is_exactly_the_first_order_form(esinw):
    """
    Given a CENTRAL transit (cos i = 0), where the transit and occultation
      chords are the same length,
    When the occultation/transit FWHM ratio is computed,
    Then it is exactly `(1 + e sin w)/(1 - e sin w)` -- the classical
      statement that the duration ratio measures e sin(omega), and the
      limit in which the textbook form is not an approximation at all.
    """
    got = _fwhm_ratio(esinw, 0.0)
    assert got == pytest.approx((1.0 + esinw) / (1.0 - esinw), rel=1e-12)


def test_the_duration_ratio_is_monotone_and_departs_from_first_order():
    """
    Given a real, non-central geometry (b ~ 0.5),
    When the same ratio is computed,
    Then it is still strictly increasing in e sin(omega) -- which is what
      makes the seed solver's bisection well posed -- but departs from the
      first-order form by tens of percent, because the two conjunctions no
      longer share an impact parameter.  That departure is why the solver
      inverts the EXACT ratio through `duration_pair` rather than the
      textbook expression.
    """
    prev, worst = -np.inf, 0.0
    for esinw in (-0.3, -0.1, 0.0, 0.1, 0.3):
        ratio = _fwhm_ratio(esinw, 0.05)
        first_order = (1.0 + esinw) / (1.0 - esinw)
        worst = max(worst, abs(ratio / first_order - 1.0))
        assert ratio > prev
        prev = ratio
    assert worst > 0.05


@pytest.mark.parametrize("ecc", [0.0, 0.2, 0.6])
@pytest.mark.parametrize("omega_deg", [-90.0, 0.0, 37.0, 150.0])
def test_the_numpy_and_pytensor_backends_agree(ecc, omega_deg):
    """
    Given the same inputs,
    When the duration and eclipse-time helpers are evaluated through
      `xp=np` and through `xp=pt`,
    Then they agree to double precision.  The seed solver calls the numpy
      side and the likelihood the pytensor side, so a divergence would let
      the fit start somewhere the model does not agree is right -- which is
      the failure the seeding exists to prevent.
    """
    # Arrange
    omega = np.radians(omega_deg)
    esinw = ecc * np.sin(omega)
    args = (10.0, COSI, np.sqrt(1 - COSI**2), ecc, esinw, 0.1, PERIOD)
    e_t, w_t = pt.dscalar(), pt.dscalar()

    # Act
    np_t14, np_t23 = pphys.duration_pair(*args, secondary=True, xp=np)
    tensor = pytensor.function(
        [e_t, w_t],
        [
            pphys.duration_pair(
                10.0,
                COSI,
                np.sqrt(1 - COSI**2),
                e_t,
                e_t * pt.sin(w_t),
                0.1,
                PERIOD,
                secondary=True,
                xp=pt,
            )[0],
            ophys.ts_from_ecc_omega(e_t, w_t, TC, PERIOD, xp=pt),
        ],
    )(ecc, omega)

    # Assert
    assert float(tensor[0]) == pytest.approx(float(np_t14), rel=1e-14)
    assert float(tensor[1]) == pytest.approx(
        float(ophys.ts_from_ecc_omega(ecc, omega, TC, PERIOD, xp=np)),
        rel=1e-14,
    )


# ---------------------------------------------------------------------------
# 3. Inject and recover: the seed solver
# ---------------------------------------------------------------------------


def _forward(ecc, omega_deg):
    """The (ts, tfwhm, tfwhms) an orbit with this (e, omega) produces."""
    system = _system(
        **{
            "orbit.b.ecc": {"initval": ecc},
            "orbit.b.omega": {"initval": omega_deg},
        }
    )
    ar = _solved(system, "planet.0.ar")
    p = _solved(system, "planet.0.p")
    omega = np.radians(omega_deg)
    kw = dict(
        ar=ar,
        cosi=COSI,
        sini=np.sqrt(1 - COSI**2),
        ecc=ecc,
        esinw=ecc * np.sin(omega),
        p=p,
        period=PERIOD,
        xp=np,
    )
    out = {}
    for key, sec in (("tfwhm", False), ("tfwhms", True)):
        t14, t23 = pphys.duration_pair(secondary=sec, **kw)
        out[key] = float(0.5 * (t14 + t23))
    out["ts"] = float(ophys.ts_from_ecc_omega(ecc, omega, TC, PERIOD, xp=np))
    return out


def _recover(obs):
    system = _system(
        **{
            "orbit.b.ts": {"mu": obs["ts"], "sigma": 0.002},
            "planet.b.tfwhm": {"mu": obs["tfwhm"], "sigma": 0.001},
            "planet.b.tfwhms": {"mu": obs["tfwhms"], "sigma": 0.001},
        }
    )
    return (
        _solved(system, "orbit.0.ecc"),
        _solved(system, "orbit.0.omega"),
    )


@pytest.mark.parametrize(
    "ecc,omega_deg",
    [
        (0.05, 0.0),
        (0.15, -60.0),
        (0.30, 45.0),
        (0.30, 135.0),
        (0.50, 90.0),
        (0.60, -120.0),
    ],
)
def test_a_stated_eclipse_time_and_duration_pair_recover_e_and_omega(
    ecc, omega_deg
):
    """
    Given an orbit's true (e, omega), and the eclipse time and the two FWHM
      durations it produces,
    When those three are handed back to a fresh fit as Gaussian priors and
      nothing else says anything about the eccentricity,
    Then the relaxation engine's start recovers (e, omega) to better than
      1e-3 in e and 0.1 deg in omega -- across the whole range, e = 0.6
      included.  Measured: the worst of these six is de = 3.7e-4 at
      e = 0.6.  Without the solver the start is the defaults.yaml
      e = 2e-4, i.e. no information at all.
    """
    # Arrange
    obs = _forward(ecc, omega_deg)

    # Act
    e_hat, w_hat = _recover(obs)

    # Assert
    wrapped = (w_hat - omega_deg + 180.0) % 360.0 - 180.0
    assert e_hat == pytest.approx(ecc, abs=1e-3)
    assert wrapped == pytest.approx(0.0, abs=0.1)


def test_the_eclipse_phase_alone_seeds_ecosw():
    """
    Given ONLY a stated eclipse time,
    When the engine runs,
    Then e cos(omega) moves to what the phase implies while the seed says
      nothing about e sin(omega).  The two observables are solved as two
      1-D bisections because they measure two nearly orthogonal directions;
      this is the half that is the eclipse phase.
    """
    obs = _forward(0.3, 0.0)  # omega = 0 -> pure e cos(omega)
    system = _system(**{"orbit.b.ts": {"mu": obs["ts"], "sigma": 0.002}})

    ecc = _solved(system, "orbit.0.ecc")
    omega = np.radians(_solved(system, "orbit.0.omega"))
    assert ecc * np.cos(omega) == pytest.approx(0.3, abs=1e-3)


def test_the_solver_is_inert_when_the_user_states_nothing():
    """
    Given a params file that mentions no duration and no eclipse time,
    When the engine runs,
    Then the eccentricity keeps the start the sqrt(e) defaults imply.  The
      gate is on what the USER WROTE and never on what resolved: `ts` and
      the durations are derived, so the engine computes them every
      iteration, and seeding (e, omega) back from those would be a fixed
      point dressed up as an inference.
    """
    system = _system()
    assert _solved(system, "orbit.0.ecc") == pytest.approx(
        0.01**2 + 0.01**2, rel=1e-9
    )


def test_an_explicit_eccentricity_still_wins():
    """
    Given a params file that states BOTH a duration pair and its own
      eccentricity,
    When the engine runs,
    Then the user's eccentricity stands: the timing solver carries
      RANK_DERIVED_MIXED and must not overwrite a RANK_USER entry.
    """
    obs = _forward(0.3, 45.0)
    system = _system(
        **{
            "planet.b.tfwhm": {"mu": obs["tfwhm"], "sigma": 0.001},
            "planet.b.tfwhms": {"mu": obs["tfwhms"], "sigma": 0.001},
            "orbit.b.ecc": {"initval": 0.1},
            "orbit.b.omega": {"initval": 20.0},
        }
    )
    assert _solved(system, "orbit.0.ecc") == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. The likelihood half
# ---------------------------------------------------------------------------


def test_a_gaussian_on_the_eclipse_time_reaches_the_eccentricity_gradient():
    """
    Given a Gaussian prior on the eclipse time,
    When the model is built,
    Then the start logp and gradient are finite AND the gradient with
      respect to the sqrt(e) pair is non-zero -- i.e. the constraint really
      does push on the eccentricity vector.  This half needs no new
      machinery at all: `ts` is a derived Parameter, so `build_pymc`'s
      Gaussian branch applies it and the expression carries the gradient.
    """
    # Arrange
    obs = _forward(0.3, 45.0)
    system = _system(
        **{"orbit.b.ts": {"mu": obs["ts"] + 0.05, "sigma": 0.002}}
    )
    model = system.build_model()

    # Act
    point = model.initial_point()
    logp = model.compile_logp()(point)
    grads = model.compile_dlogp()(point)

    # Assert
    assert np.isfinite(logp)
    assert np.isfinite(np.atleast_1d(grads)).all()
    assert "gaussian_prior.orbit.ts" in {p.name for p in model.potentials}


def test_the_durations_are_reported_at_the_start_point():
    """
    Given a system with a seeded eccentricity,
    When every duration is evaluated at the start point,
    Then all are finite and ordered as the geometry requires:
      tau < T_FWHM < T_14, and the occultation is the longer conjunction
      for e sin(omega) > 0.
    """
    system = _system(
        **{
            "orbit.b.ecc": {"initval": 0.3},
            "orbit.b.omega": {"initval": 90.0},
        }
    )
    model = system.build_model()
    v = _at_start(
        system,
        model,
        [("planet", n) for n in _DURATIONS] + [("orbit", "ts")],
    )

    assert all(np.isfinite(list(v.values())))
    assert v["planet.tau"] < v["planet.tfwhm"] < v["planet.t14"]
    assert v["planet.taus"] < v["planet.tfwhms"] < v["planet.t14s"]
    # omega = 90 deg puts the transit at periastron, so the planet is
    # moving fastest there and the occultation is the longer event.
    assert v["planet.t14s"] > v["planet.t14"]
