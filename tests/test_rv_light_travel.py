"""The star's own light-travel time in CORRECTED radial velocities
(rvinstrument ``data_format: rv``, ``light_travel_time:`` on by default):
``RVInstrument._rv_model`` evaluates every reflex column at the star's
EMISSION time, retarded by its own line-of-sight displacement in that orbit's
subsystem (``_orbit_emission_times``, the raw-redshift mode's machinery).

Checked here against an INDEPENDENT numpy model -- the star's velocity and
line-of-sight position composed by hand from the orbits' start elements
(tests/test_raw_redshift_orbits.py's ``_star_motion``) and its own
fixed-point solve of ``t_emit = t - Z_star(t_emit)/c``:

* a close binary observed in its SECONDARY's light (the companion side);
* two planets, the outer one on a Jacobi orbit, each orbit retarded by
  its own subsystem (``_orbit_emission_times``);
* ``light_travel_time: false`` is the arrival-time model, and the orbit's
  reporting follows the key (``tc_target == tc`` off, the observed-frame
  shift on: Orbit._ltt_reporting_mask);
* an orbit component without physical a/masses warns once and is evaluated
  at the arrival time.
"""

import logging

import numpy as np
import pytest
from test_raw_redshift_orbits import (
    _at,
    _build,
    _elements,
    _star_motion,
    _times,
)

C_MPS = 299792458.0


def _independent_rv(t, orbits, gamma_ms, retard=True, subsystems=None):
    """gamma + the star's line-of-sight reflex (m/s, away = positive), each
    orbit's at its own emission time by fixed-point iteration: retarded by
    the star's displacement in the orbits ``subsystems[k]`` (default orbit
    k alone)."""
    if subsystems is None:
        subsystems = [[k] for k in range(len(orbits))]
    V = 0.0
    for k, o in enumerate(orbits):
        t_emit = np.array(t, dtype=float)
        if retard:
            sub = [orbits[q] for q in subsystems[k]]
            for _ in range(10):
                Z, _ = _star_motion(t_emit, sub)
                t_emit = t - Z / C_MPS / 86400.0
        V = V + _star_motion(t_emit, [o])[1]
    return gamma_ms + V


def _write(path, t, v):
    np.savetxt(path, np.column_stack([t, v, np.full(t.size, 5.0)]))
    return str(path)


def _model_rv(system, model, start):
    rvi = system.rvinstrument
    (mod,) = _at(model, start, [rvi._rv_model_data_node])
    return mod * rvi._rv_factor()


def _binary(tmp_path_factory, ltt):
    d = tmp_path_factory.mktemp(f"rvltt_binary_{ltt}")
    t = _times(span=(2460000.0, 2460030.0))
    entry = {"name": "HPF", "file": _write(d / "B.rv", t, np.zeros(t.size))}
    entry["star_ndx"] = "B"
    if not ltt:
        entry["light_travel_time"] = False
    cfg = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [{"name": "AB", "primary": ["A"], "companion": ["B"]}],
        "rvinstrument": [entry],
    }
    p = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.02},
        "star.A.radius": {"initval": 1.0, "sigma": 0.02},
        "star.B.mass": {"initval": 0.3, "sigma": 0.02},
        "star.B.radius": {"initval": 0.3, "sigma": 0.02},
        "orbit.AB.period": {"initval": 3.0},
        "orbit.AB.tc": {"initval": 2460010.0},
        "orbit.AB.ecc": {"initval": 0.1},
        "orbit.AB.omega": {"initval": 40.0},
        "orbit.AB.cosi": {"initval": 0.2},
        "rvinstrument.HPF.gamma": {"initval": 10.0},
    }
    return _build(cfg, p), t


@pytest.fixture(scope="module")
def binary_on(tmp_path_factory):
    return _binary(tmp_path_factory, True)


@pytest.fixture(scope="module")
def binary_off(tmp_path_factory):
    return _binary(tmp_path_factory, False)


@pytest.mark.slow
def test_a_binary_secondary_is_retarded_by_its_own_light_travel_time(
    binary_on, binary_off
):
    """
    Given corrected RVs of the SECONDARY of a 1.0 + 0.3 Msun, 3-day binary
      (~100 km/s), light_travel_time on (the default) and off,
    When the likelihood's model is evaluated at the start,
    Then on, it equals the independent composition at the secondary's
      emission time (share -M_A/M, its own fixed-point retardation) to
      < 1 mm/s; off, the same composition at the arrival time; the two
      differ by the ~K^2/c a 3-day binary's light-travel time is worth
      (tens of m/s on this secondary); and the orbit reports the observed
      frame only when on (tc_target != tc).
    """
    for (system, model, start), t in (binary_on, binary_off):
        assert system.orbit.star_membership(1) == [(0, "companion")]
    (s_on, m_on, x_on), t = binary_on
    (s_off, m_off, x_off), _ = binary_off
    orb_on = _elements(s_on, m_on, x_on, {0: "companion"})
    orb_off = _elements(s_off, m_off, x_off, {0: "companion"})
    g = s_on.rvinstrument._rv_factor()
    (gamma_on,) = _at(m_on, x_on, [s_on.rvinstrument.gamma.value])
    (gamma_off,) = _at(m_off, x_off, [s_off.rvinstrument.gamma.value])
    mod_on = _model_rv(s_on, m_on, x_on)
    mod_off = _model_rv(s_off, m_off, x_off)
    ind_on = _independent_rv(t, orb_on, float(gamma_on[0]) * g)
    ind_off = _independent_rv(t, orb_off, float(gamma_off[0]) * g, False)
    assert np.max(np.abs(mod_on - ind_on)) < 1e-3
    assert np.max(np.abs(mod_off - ind_off)) < 1e-3
    # The retardation itself, at the same (target-frame) elements.
    unretarded = _independent_rv(t, orb_on, float(gamma_on[0]) * g, False)
    effect = np.max(np.abs(ind_on - unretarded))
    print(
        f"binary secondary: the star's light travel moves v by {effect:.2f} m/s"
    )
    assert effect > 5.0
    tc_on, tct_on = _at(
        m_on, x_on, [s_on.orbit.tc.value, s_on.orbit.tc_target.value]
    )
    tc_off, tct_off = _at(
        m_off, x_off, [s_off.orbit.tc.value, s_off.orbit.tc_target.value]
    )
    assert tct_off[0] == tc_off[0]
    assert abs(tct_on[0] - tc_on[0]) * 86400.0 > 1.0


@pytest.fixture(scope="module")
def two_planet_jacobi(tmp_path_factory):
    d = tmp_path_factory.mktemp("rvltt_jacobi")
    t = _times()
    cfg = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}, {"name": "c"}],
        "orbit": [
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "c", "primary": ["A", "b"], "companion": ["c"]},
        ],
        "rvinstrument": [
            {"name": "HPF", "file": _write(d / "A.rv", t, np.zeros(t.size))}
        ],
    }
    p = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.02},
        "star.A.radius": {"initval": 1.0, "sigma": 0.02},
        "orbit.b.period": {"initval": 12.0},
        "orbit.b.tc": {"initval": 2460300.0},
        "orbit.b.ecc": {"initval": 0.2},
        "orbit.b.omega": {"initval": 70.0},
        "orbit.c.period": {"initval": 400.0},
        "orbit.c.tc": {"initval": 2460200.0},
        "orbit.c.ecc": {"initval": 0.4},
        "orbit.c.omega": {"initval": 200.0},
        "planet.b.K": {"initval": 3000.0},
        "planet.c.K": {"initval": 1500.0},
        "rvinstrument.HPF.gamma": {"initval": 10.0},
    }
    return _build(cfg, p), t


@pytest.mark.slow
def test_each_orbit_is_retarded_by_its_own_subsystem(two_planet_jacobi):
    """
    Given a star in both orbits' primary groups (b around A, c around
      [A, b]; brown-dwarf amplitudes so the term is visible),
    When the corrected-RV model is evaluated at the start,
    Then it equals the independent composition with each orbit's reflex at
      its own emission time -- b retarded by the star's motion in b, c by
      its motion in c AND in b (nested inside c) -- to < 0.1 mm/s; and
      retarding b by c's motion as well (the outer orbit's light-time effect
      on the inner one, which transits do not model) would miss by more
      (RVInstrument._orbit_emission_times).
    """
    (system, model, start), t = two_planet_jacobi
    orbits = _elements(system, model, start, {0: "primary", 1: "primary"})
    (gamma,) = _at(model, start, [system.rvinstrument.gamma.value])
    g = float(gamma[0]) * system.rvinstrument._rv_factor()
    mod = _model_rv(system, model, start)
    ind = _independent_rv(t, orbits, g, subsystems=[[0], [0, 1]])
    assert np.max(np.abs(mod - ind)) < 1e-4
    summed = _independent_rv(t, orbits, g, subsystems=[[0, 1], [0, 1]])
    assert np.max(np.abs(summed - ind)) > 1e-3


def test_an_orbit_without_physical_elements_warns_once_and_is_not_retarded(
    caplog,
):
    """
    Given an orbit component that lacks a/m_primary/m_companion/m_total/
      sini (its bodies did not resolve),
    When the RV instrument asks whether it can retard the reflex,
    Then it answers no and warns ONCE (as rm.py does), naming the missing
      parameters and how to silence it.
    """
    from exozippy.components.rvinstrument.rvinstrument import RVInstrument

    rvi = RVInstrument.__new__(RVInstrument)
    rvi._ltt_unsupported_warned = False

    class NoBodies:
        pass

    with caplog.at_level(logging.WARNING):
        assert rvi._keplerian_ltt_supported(NoBodies()) is False
        assert rvi._keplerian_ltt_supported(NoBodies()) is False
    msgs = [
        r.getMessage()
        for r in caplog.records
        if "light-travel" in r.getMessage()
    ]
    assert len(msgs) == 1
    assert "m_companion" in msgs[0] and "light_travel_time: false" in msgs[0]
