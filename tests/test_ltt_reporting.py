"""The observed-frame (BJD_TDB) timing twins tc_bjd/ts_bjd/tp_bjd (PR #307,
review 1.8.9): the closed form in orbit/physics.py against the shared
light-travel kernel, the signs, the per-orbit mask in its off / mixed / RM /
two-orbit cases, and the headline-vs-target-frame labels.
"""

import logging

import numpy as np
import pytest
from test_rm_ltt import _rm_wiring_config, _rm_wiring_params, _write_two_row_rv
from test_transit_ltt import (
    _eval_at_point,
    _ltt_wiring_config,
    _ltt_wiring_params,
    _write_two_row_lc,
)

from exozippy.components import ltt
from exozippy.constants import C_LIGHT_RSUN_PER_DAY
from exozippy.system import System

TWINS = ("tc_bjd", "ts_bjd", "tp_bjd")
BASES = ("tc", "ts", "tp")


def _eccentric_inclined(params):
    """An orbit where every term in the closed form is live: e = 0.25,
    omega = 53 deg, sin i != 1 -- so tp's shift differs from tc's and ts's
    is not simply minus tc's."""
    params = dict(params)
    params["orbit.0.secosw"] = {"initval": 0.3, "sigma": 0.0}
    params["orbit.0.sesinw"] = {"initval": 0.4, "sigma": 0.0}
    params["orbit.0.cosi"] = {"initval": 0.1, "sigma": 0.0}
    return params


def _build(config, params):
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point


def _val(system, model, point, node):
    return np.atleast_1d(np.asarray(_eval_at_point(node, model, point), float))


def _exact_shift(orbit, model, point, t_node):
    """-z(t)*factor/c from ltt.line_of_sight_kinematics -- the kernel every
    likelihood retards with -- evaluated at the target-frame event time."""
    factor = (
        orbit.m_primary.value - orbit.m_companion.value
    ) / orbit.m_total.value
    z, _, _ = ltt.line_of_sight_kinematics(
        t_node,
        orbit.tp.value,
        orbit.n.value,
        orbit.ecc.value,
        orbit.sinw.value,
        orbit.cosw.value,
        orbit.sini.value,
        orbit.a.value,
        factor=factor,
        circular=False,
    )
    return _val(orbit, model, point, -z / C_LIGHT_RSUN_PER_DAY)


def test_twins_match_the_kernel_at_all_three_events_and_have_the_right_signs(
    tmp_path,
):
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    system, model, point = _build(
        _ltt_wiring_config(lc), _eccentric_inclined(_ltt_wiring_params())
    )
    o = system.orbit
    assert all(k in o.manifest for k in TWINS)
    assert o._ltt_report_mask.tolist() == [1.0]

    shifts = {}
    for base, twin in zip(BASES, TWINS):
        got = _val(o, model, point, getattr(o, twin).value) - _val(
            o, model, point, getattr(o, base).value
        )
        want = _exact_shift(o, model, point, getattr(o, base).value)
        # The two forms are algebraically identical; the residual is the
        # kernel's numerical Kepler solve at ts/tp (the closed form uses the
        # exact anomaly).  1e-9 d = 86 us on a ~600 s shift, vs the 16 us the
        # PR's original Taylor-solved delay was off by at tc.
        assert got == pytest.approx(want, abs=1e-9), (base, got, want)
        shifts[base] = float(got[0])

    a_over_c = (
        float(_val(o, model, point, o.a.value)[0]) / C_LIGHT_RSUN_PER_DAY
    )
    # Planet in front at conjunction: observed EARLY.  Behind at eclipse:
    # observed LATE.  Both of order a/c (~500 s here, a ~ 1 AU).
    assert shifts["tc"] < 0 < shifts["ts"]
    assert 0.3 * a_over_c < -shifts["tc"] < 1.5 * a_over_c
    assert 0.3 * a_over_c < shifts["ts"] < 1.5 * a_over_c
    # tp_bjd is the observed PERIASTRON (JDE 2026-09-21), so its shift is
    # z(tp)'s, not tc's: on this orbit they differ by tens of seconds.
    assert abs(shifts["tp"] - shifts["tc"]) > 10.0 / 86400.0


def test_ltt_off_everywhere_keeps_the_twins_equal_to_the_target_frame(
    tmp_path,
):
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    system, model, point = _build(
        _ltt_wiring_config(lc, light_travel_time=False),
        _eccentric_inclined(_ltt_wiring_params()),
    )
    o = system.orbit
    assert o._ltt_report_mask.tolist() == [0.0]
    # Declared anyway -- the reader always gets a BJD_TDB row -- and exactly
    # equal to the target-frame value, since nothing retarded this orbit.
    for base, twin in zip(BASES, TWINS):
        assert twin in o.manifest
        got = _val(o, model, point, getattr(o, twin).value)
        want = _val(o, model, point, getattr(o, base).value)
        assert np.array_equal(got, want), (base, got, want)


def test_headline_labels_are_bjd_tdb_and_target_rows_are_relabeled(tmp_path):
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    system, _, _ = _build(_ltt_wiring_config(lc), _ltt_wiring_params())
    o = system.orbit
    assert o.tc_bjd.latex == "T_C"
    assert o.ts_bjd.latex == "T_S"
    assert o.tp_bjd.latex == "T_P"
    assert "BJD_TDB" in o.tc_bjd.description
    assert o.tc.latex == r"T_{C,\rm target}"
    assert o.ts.latex == r"T_{S,\rm target}"
    assert o.tp.latex == r"T_{P,\rm target}"
    assert "target frame" in o.tc.description


def test_mixed_files_warn_once_and_treat_the_orbit_as_retarded(
    tmp_path, caplog
):
    lc_on = _write_two_row_lc(tmp_path / "on.dat")
    lc_off = _write_two_row_lc(tmp_path / "off.dat")
    config = _ltt_wiring_config(lc_on, light_travel_time=True)
    config["transit"].append(
        {
            "name": "inst1",
            "file": lc_off,
            "band": "TESS",
            "light_travel_time": False,
        }
    )
    with caplog.at_level(logging.WARNING):
        system, model, point = _build(config, _ltt_wiring_params())
    o = system.orbit
    assert o._ltt_report_mask.tolist() == [1.0]
    msgs = [
        r.getMessage() for r in caplog.records if "disagree" in r.getMessage()
    ]
    assert len(msgs) == 1
    shift = _val(o, model, point, o.tc_bjd.value - o.tc.value)
    assert shift[0] < 0


def test_rm_alone_retards_the_orbit_it_names(tmp_path):
    rv = _write_two_row_rv(tmp_path / "rv.dat")
    system, model, point = _build(_rm_wiring_config(rv), _rm_wiring_params())
    o = system.orbit
    assert o._ltt_report_mask.tolist() == [1.0]
    shift = _val(o, model, point, o.tc_bjd.value - o.tc.value)
    assert shift[0] < 0

    rv2 = _write_two_row_rv(tmp_path / "rv2.dat")
    system_off, _, _ = _build(
        _rm_wiring_config(rv2, light_travel_time=False), _rm_wiring_params()
    )
    assert system_off.orbit._ltt_report_mask.tolist() == [0.0]


def test_mask_is_per_orbit_a_planetless_orbit_is_not_retarded(tmp_path):
    """Two orbits, one transit file: the planet's orbit is retarded, the
    stellar-companion orbit (no planet, so nothing in the transit model
    touches it) is not, and its twin equals its target-frame value."""
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    config = _ltt_wiring_config(lc)
    config["star"].append({"name": "B", "mist": False})
    config["orbit"] = [
        {"name": "b", "primary": ["A"], "companion": ["b"]},
        {"name": "AB", "primary": ["A"], "companion": ["B"]},
    ]
    params = dict(_ltt_wiring_params())
    params["orbit.1.period"] = {"initval": 3000.0}
    params["orbit.1.tc"] = {"initval": 2459100.0}
    params["star.1.mass"] = {"initval": 0.5, "sigma": 0.05}
    params["star.1.radius"] = {"initval": 0.5}
    params["star.1.teff"] = {"initval": 3800, "sigma": 100}
    params["star.1.feh"] = {"initval": 0.0, "sigma": 0.08}
    system, model, point = _build(config, params)
    o = system.orbit
    assert o._ltt_report_mask.tolist() == [1.0, 0.0]
    shift = _val(o, model, point, o.tc_bjd.value - o.tc.value)
    assert shift[0] < 0
    assert shift[1] == 0.0
