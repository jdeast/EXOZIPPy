"""
Integration tests for the light-travel-time (Roemer delay) wiring in
rm.py's compute_rm_rv (the true-anomaly seam only -- see components/rm.py
and the design doc). components/ltt.py's own unit tests (tests/test_ltt.py)
and transit's wiring tests (tests/test_transit_ltt.py) cover the shared
helper and its transit consumer; these tests exercise the RM consumer's
WIRED path -- real orbits.a/m_primary/m_total accessors, real units --
rather than calling ltt.py's functions directly with hand-typed values.
"""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytensor
import pytest

from exozippy.components import rm
from exozippy.constants import C_LIGHT_RSUN_PER_DAY
from exozippy.system import System

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_TC = 2459000.0
_PERIOD = 365.25  # days; with star.mass ~ 1 Msun this gives a ~ 1 AU


def _write_two_row_rv(path):
    # Only the TIMES matter for compute_rm_rv's own returned tensor (the
    # thing under test here); the rv/err columns just need to be valid data
    # for rvinstrument's likelihood plumbing.
    time = np.array([_TC, _TC + _PERIOD / 2.0])
    rv = np.array([0.0, 0.0])
    err = np.array([10.0, 10.0])
    np.savetxt(path, np.column_stack([time, rv, err]))
    return str(path)


def _rm_wiring_config(rv_file):
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "rvinstrument": [
            {"name": "TRES", "file": rv_file, "rm": "b", "rm_band": "TESS"}
        ],
    }


def _rm_wiring_params():
    return {
        "star.0.radius": {
            "initval": 2.0,
            "sigma": 0.05,
        },  # deliberately != 1 Rsun, same reasoning as test_transit_ltt.py:
        # catches an a/R* (dimensionless) vs physical a [Rsun] mixup.
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.0, "sigma": 0.0},  # edge-on, sin_i = 1
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},  # circular, omega=0
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.0},
        # RM makes this planet RV-mass-constrained, so mass_parameterization
        # defaults to linear (not log_q, unlike the transit-only wiring
        # test) -- pin planet.mass directly, near enough to 0 Mjup that the
        # barycentric factor m_primary/m_total stays ~1 to high precision
        # (1 Mjup / 1 Msun ~ 1e-3; 1e-6 Mjup is ~1e-9 relative).
        "planet.0.mass": {"initval": 1.0e-6, "sigma": 0.0},
    }


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


def _build_with_spy(config, params):
    """Builds the system with compute_rm_rv spied on the module attribute
    (the way rvinstrument.py's local `from ..rm import compute_rm_rv`
    resolves it -- see test_rossiter.py's
    test_rm_is_evaluated_only_on_its_own_instrument_rows for the same
    pattern). Returns (system, model, point, list of (time, result) calls).
    """
    real = rm.compute_rm_rv
    calls = []

    def _spy(system, time, *args, **kwargs):
        result = real(system, time, *args, **kwargs)
        calls.append((time, result))
        return result

    with mock.patch.object(rm, "compute_rm_rv", side_effect=_spy):
        system = System(config, user_params=params)
        system.prepare()
        model = system.build_model()

    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point, calls


def test_wired_rm_ltt_delay_matches_a_over_c_through_real_accessors(tmp_path):
    """
    Given an RM-enabled system (star mass ~1 Msun, star RADIUS deliberately
    2 Rsun, period 365.25 d giving a ~ 1 AU via Kepler's third law, edge-on
    circular orbit, negligible planet mass) built through the real System/
    orbit/planet pipeline,
    When rm.compute_rm_rv's wiring (its call into ltt.retarded_time, using
    orbits.a/m_primary/m_total at orbit_idx) is exercised by building the
    model, with a spy capturing the real delay it produced (via a second,
    direct capture of ltt.retarded_time itself, since compute_rm_rv's
    return value is the RM anomaly, not the delay),
    Then the delay matches factor*a_rel/c (the SAME orbits.a/m_primary/
    m_total nodes the wiring itself used, read back independently) to
    1e-6 relative, and is close to the known ~499 s/AU light time in
    absolute terms.
    """
    rv_file = _write_two_row_rv(tmp_path / "rv.dat")
    real_retarded_time = rm.ltt.retarded_time
    delay_calls = []

    def _ltt_spy(*args, **kwargs):
        result = real_retarded_time(*args, **kwargs)
        delay_calls.append(result[1])
        return result

    with mock.patch.object(rm.ltt, "retarded_time", side_effect=_ltt_spy):
        system = System(
            _rm_wiring_config(rv_file), user_params=_rm_wiring_params()
        )
        system.prepare()
        model = system.build_model()

    assert len(delay_calls) >= 1  # build_likelihood (+ compile_plotters)

    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))

    delay = _eval_at_point(delay_calls[0], model, point)
    delay_primary = float(delay[0])
    delay_secondary = float(delay[1])

    orbit_idx = 0
    a_rel_val = float(
        _eval_at_point(system.orbit.a.value[orbit_idx], model, point)
    )
    m_primary_val = float(
        _eval_at_point(system.orbit.m_primary.value[orbit_idx], model, point)
    )
    m_total_val = float(
        _eval_at_point(system.orbit.m_total.value[orbit_idx], model, point)
    )
    factor_val = m_primary_val / m_total_val

    expected_amp_days = factor_val * a_rel_val / C_LIGHT_RSUN_PER_DAY

    assert abs(delay_primary) == pytest.approx(expected_amp_days, rel=1e-6)
    assert abs(delay_primary) * 86400.0 == pytest.approx(499.0, abs=2.0)
    assert (delay_secondary - delay_primary) == pytest.approx(
        2.0 * expected_amp_days, rel=1e-6
    )


def test_rm_ltt_off_reproduces_pre_ltt_output(tmp_path):
    """
    Given a STORED golden fixture (tests/fixtures/rm_ltt_off_reference.json)
    of compute_rm_rv's output from the actual pre-LTT rm.py -- generated
    once from git HEAD at df96bd9 (rm.py's own light-travel-time wiring was
    still uncommitted at that point; confirmed via grep at generation time
    to have no ltt/LIGHT_TRAVEL references), not a hand re-derivation of the
    geometry that could itself drift from what the real old code did,
    When the SAME config/params (_rm_wiring_config/_rm_wiring_params) build
    a model with the CURRENT compute_rm_rv, rm._LIGHT_TRAVEL_TIME_ACTIVE
    monkeypatched to False,
    Then compute_rm_rv's own returned RM-anomaly tensor evaluates to the
    SAME array the fixture recorded, to floating-point precision --
    confirming the OFF branch is algebraically the pre-LTT computation (raw
    time straight into get_true_anomaly), not a second Kepler solve that
    happens to cancel out, and that no other line in compute_rm_rv shifted
    underneath it.

    Deliberately does NOT read git history (unlike an earlier version of
    this test, which shelled out to `git show HEAD:...`): that pattern
    self-invalidates the moment rm.py's own LTT wiring is committed, since
    HEAD then stops being the pre-LTT answer -- exactly what already
    happened to transit's analogous test once ITS commit landed. A frozen
    fixture has no such expiry.
    """
    rv_file = _write_two_row_rv(tmp_path / "rv.dat")
    config = _rm_wiring_config(rv_file)
    params = _rm_wiring_params()

    with open(_FIXTURES_DIR / "rm_ltt_off_reference.json") as f:
        fixture = json.load(f)
    rv_reference = np.asarray(fixture["rm_rv"])

    with mock.patch.object(rm, "_LIGHT_TRAVEL_TIME_ACTIVE", False):
        _sys_off, model_off, point_off, calls_off = _build_with_spy(
            config, params
        )
    rv_off = _eval_at_point(calls_off[0][1], model_off, point_off)

    np.testing.assert_array_equal(rv_reference, rv_off)
