"""A time of PERIASTRON seed must reach the fit (review 8.1.1).

`tp` is how the RV literature quotes an eccentric orbit, and until this
landed a `orbit.<n>.tp` entry in a params file reached nothing at all: `tp`
was not in orbit's `get_symbol_map`, so the relaxation engine registered it
as an unmapped two-part leaf and no relation could consume it.  The
tc <-> tp chain runs through Kepler's equation, which is exactly why the
symbolic RELATIONS cannot carry it -- sympy hangs on `M = E - e sin E` -- so
the channel is a one-way standalone solver, the shape 8.1.5 established.

Measured on master with the config below: `orbit.0.tc` stayed at the
defaults.yaml backstop 2460000.0 while `tp` sat resolved at 2455000.0.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit import Orbit, physics
from exozippy.config import PRECEDENCE_DERIVED_MIXED, ConfigManager
from exozippy.system import System

TP = 2455000.0
PERIOD = 10.0
SECOSW, SESINW = 0.3, 0.2
ECC = SECOSW**2 + SESINW**2
OMEGA = np.arctan2(SESINW, SECOSW)

_CONFIG = {
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b"}],
    "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
}


def _params(**extra):
    params = {
        "orbit.b.period": {"initval": PERIOD},
        "orbit.b.secosw": {"initval": SECOSW},
        "orbit.b.sesinw": {"initval": SESINW},
    }
    params.update(extra)
    return params


# ---------------------------------------------------------------------------
# 1. The algebra: the numpy inverse against the pytensor forward model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ecc", [0.0, 0.05, 0.3, 0.7, 0.95])
@pytest.mark.parametrize("omega_deg", [-135.0, -30.0, 0.0, 45.0, 90.0, 179.0])
def test_tc_from_tp_inverts_the_model_kepler_solve(ecc, omega_deg):
    """
    Given any (e, omega) the model can carry,
    When `tc_from_tp` maps a periastron time to a conjunction time and the
      model's own `calc_tp_from_ecc` maps it back,
    Then the round trip returns the original tc.  The numpy helper is the
      engine's copy of the Kepler algebra, so what must be pinned is that it
      cannot drift from the pytensor form the likelihood evaluates.
    """
    # Arrange
    tc, omega = 2455123.456, np.radians(omega_deg)
    n = 2.0 * np.pi / PERIOD
    e_t, w_t, tc_t = pt.dscalar(), pt.dscalar(), pt.dscalar()
    forward = pytensor.function(
        [e_t, w_t, tc_t], physics.calc_tp_from_ecc(e_t, w_t, tc_t, n)
    )

    # Act
    tp = float(forward(ecc, omega, tc))
    back = float(physics.tc_from_tp(tp, ecc, omega, PERIOD))

    # Assert -- tp is only ever consumed modulo the period.
    assert (back - tc) % PERIOD == pytest.approx(0.0, abs=1e-6) or (
        back - tc
    ) % PERIOD == pytest.approx(PERIOD, abs=1e-6)


def test_tc_from_tp_is_vectorized_over_orbits():
    """
    Given per-orbit arrays of (tp, e, omega, P),
    When tc_from_tp is called once,
    Then it returns one tc per orbit -- the stage-3 window needs the whole
      vector, and a scalar-only helper would have to be looped there.
    """
    out = physics.tc_from_tp(
        np.array([TP, TP + 5.0]),
        np.array([0.0, 0.3]),
        np.array([0.0, 1.0]),
        np.array([10.0, 20.0]),
    )
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# 2. The engine: a tp seed resolves tc
# ---------------------------------------------------------------------------


def _resolved_tc(params):
    system = System(dict(_CONFIG), params)
    system.prepare()
    entry = system.config_manager.user_params.get("orbit.0.tc")
    return float(entry["initval"])


def test_a_tp_seed_resolves_the_time_of_conjunction():
    """
    Given a params file that seeds `tp` and not `tc`,
    When the relaxation engine runs,
    Then `orbit.0.tc` is the conjunction time that periastron implies --
      not the defaults.yaml backstop 2460000.0 it stayed at before.
    """
    got = _resolved_tc(_params(**{"orbit.b.tp": {"initval": TP}}))

    want = float(physics.tc_from_tp(TP, ECC, OMEGA, PERIOD))
    assert got == pytest.approx(want, abs=1e-6)
    assert abs(got - 2460000.0) > 1.0


def test_a_user_tc_still_wins_over_a_tp_seed():
    """
    Given a params file that seeds BOTH tc and tp,
    When the engine runs,
    Then tc is the user's own value: the solver carries PRECEDENCE_DERIVED_MIXED
      and a PRECEDENCE_USER entry is exactly what it must not overwrite.
    """
    got = _resolved_tc(
        _params(
            **{
                "orbit.b.tp": {"initval": TP},
                "orbit.b.tc": {"initval": 2455500.0},
            }
        )
    )
    assert got == pytest.approx(2455500.0, abs=1e-9)


def test_no_tp_seed_leaves_tc_at_its_backstop():
    """
    Given a params file that seeds neither tc nor tp,
    When the engine runs,
    Then tc keeps the defaults.yaml backstop -- the solver raises KeyError
      until tp resolves, and nothing resolves it, so the channel is inert.
    """
    assert _resolved_tc(_params()) == pytest.approx(2460000.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. The stage-3 tc window has to move with it
# ---------------------------------------------------------------------------


def _window(user_params):
    """(centre, half-width) of the manifest's tc window, in days."""
    cm = ConfigManager(dict(user_params), system_config=dict(_CONFIG))
    orbit = Orbit(_CONFIG["orbit"], cm)
    orbit.register_parameters(system=None)
    entry = orbit.manifest["tc"]
    lo = np.atleast_1d(entry["lower"])
    up = np.atleast_1d(entry["upper"])
    return (lo + up) / 2.0, (up - lo) / 2.0


def test_the_tc_window_is_centred_on_the_tp_implied_conjunction():
    """
    Given a params file that seeds `tp` and not `tc`,
    When the orbit declares its parameters at stage 3 -- BEFORE the engine
      has solved anything,
    Then the hard tc window is centred on the conjunction tp implies.
      Centring it on the defaults.yaml backstop instead put the solved tc
      ~5000 d outside its own bounds, which is a fatal start-value error
      naming a parameter the user never wrote.
    """
    centre, half = _window(_params(**{"orbit.b.tp": {"initval": TP}}))

    want = float(physics.tc_from_tp(TP, ECC, OMEGA, PERIOD))
    np.testing.assert_allclose(centre, [want], atol=1e-6)
    np.testing.assert_allclose(half, [PERIOD / 2.0])


def test_a_user_tc_still_centres_the_window():
    """
    Given both tc and tp seeded,
    When the window is declared,
    Then it is centred on the user's tc -- the window must agree with the
      solver about which value wins, or the two disagree by a period.
    """
    centre, _ = _window(
        _params(
            **{
                "orbit.b.tp": {"initval": TP},
                "orbit.b.tc": {"initval": 2455500.0},
            }
        )
    )
    np.testing.assert_allclose(centre, [2455500.0])


def test_the_solved_tc_lands_inside_its_own_window():
    """
    Given a tp seed,
    When the whole model is built,
    Then it builds -- which is the real assertion: `build_pymc` raises when a
      start value is outside its hard bounds, so a window and a solver that
      disagreed could not produce a model at all.
    """
    system = System(dict(_CONFIG), _params(**{"orbit.b.tp": {"initval": TP}}))
    system.prepare()
    model = system.build_model()
    assert np.isfinite(model.compile_logp()(model.initial_point()))


# ---------------------------------------------------------------------------
# 4. A tp whose tc lands NEAR the backstop (review 1.1.9)
# ---------------------------------------------------------------------------

# examples/ob170114's periastron.  Its tc is ~1875 d from the 2460000.0
# backstop -- 7.6e-4 RELATIVE, inside the engine's old 1e-3 relative change
# band, so the solve was discarded and the backstop's rank promoted to
# "solved".  Every tp-seeded orbit with tc in ~2457540-2462460 was refused at
# build; TP above sits outside that band, which is why sections 2-3 passed.
TP_IN_BAND = 2457930.059


def _quadrant_params(omega_deg, ecc):
    omega = np.radians(omega_deg)
    return {
        "orbit.b.period": {"initval": PERIOD},
        "orbit.b.secosw": {"initval": np.sqrt(ecc) * np.cos(omega)},
        "orbit.b.sesinw": {"initval": np.sqrt(ecc) * np.sin(omega)},
        "orbit.b.tp": {"initval": TP_IN_BAND},
    }


@pytest.mark.parametrize("omega_deg", [45.0, 135.0, 225.0, 315.0])
def test_a_tp_near_the_backstop_round_trips_into_an_in_window_tc(omega_deg):
    """
    Given a tp seed whose conjunction lies ~1875 d from the tc backstop,
      with omega in each quadrant,
    When the model is built,
    Then tc is the conjunction tp implies (modulo the period, which is all
      tp determines), held at the solver's rank -- and the model BUILDS,
      i.e. the solved tc lies inside the window stage 3 centred on the same
      conjunction.  Before the fix the backstop 2460000 was kept and
      `build_pymc` refused it as outside its bounds.
    """
    ecc = 0.3
    system = System(dict(_CONFIG), _quadrant_params(omega_deg, ecc))
    system.prepare()
    cm = system.config_manager
    got = cm._last_resolved["orbit.0.tc"]

    want = float(
        physics.tc_from_tp(TP_IN_BAND, ecc, np.radians(omega_deg), PERIOD)
    )
    offset = (got - want) % PERIOD
    assert min(offset, PERIOD - offset) == pytest.approx(0.0, abs=1e-6)
    assert abs(got - TP_IN_BAND) < PERIOD
    assert cm._last_provenance["orbit.0.tc"] == PRECEDENCE_DERIVED_MIXED

    model = system.build_model()
    assert np.isfinite(model.compile_logp()(system.get_raw_start(model)))
