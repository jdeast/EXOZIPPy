"""The two timing frames (PR #307, review 1.8.9): the sampled ``tc`` is the
OBSERVED (BJD_TDB) conjunction, ``tc_target``/``ts_target``/``tp_target`` are
the target-frame quantities every Kepler solve descends from, and ``ts``/``tp``
are the observed eclipse and periastron.  Pins the closed form in
orbit/physics.py against the shared light-travel kernel, the signs, the
per-orbit mask in its off / mixed / RM-only / two-orbit / no-bodies cases,
and the labels.
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

# (observed name, target-frame name)
PAIRS = (("tc", "tc_target"), ("ts", "ts_target"), ("tp", "tp_target"))


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


def _val(model, point, node):
    return np.atleast_1d(np.asarray(_eval_at_point(node, model, point), float))


def _exact_shift(orbit, model, point, t_target_node):
    """-z(t)*factor/c from ltt.line_of_sight_kinematics -- the kernel every
    likelihood retards with -- at the target-frame event time."""
    factor = (
        orbit.m_primary.value - orbit.m_companion.value
    ) / orbit.m_total.value
    z, _, _ = ltt.line_of_sight_kinematics(
        t_target_node,
        orbit.tp_target.value,
        orbit.n.value,
        orbit.ecc.value,
        orbit.sinw.value,
        orbit.cosw.value,
        orbit.sini.value,
        orbit.a.value,
        factor=factor,
        circular=False,
    )
    return _val(model, point, -z / C_LIGHT_RSUN_PER_DAY)


def _shift(orbit, model, point, observed, target):
    return _val(
        model,
        point,
        getattr(orbit, observed).value - getattr(orbit, target).value,
    )


def test_observed_minus_target_matches_the_kernel_at_all_three_events(
    tmp_path,
):
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    system, model, point = _build(
        _ltt_wiring_config(lc), _eccentric_inclined(_ltt_wiring_params())
    )
    o = system.orbit
    assert o._ltt_report_mask.tolist() == [1.0]

    shifts = {}
    for observed, target in PAIRS:
        got = _shift(o, model, point, observed, target)
        want = _exact_shift(o, model, point, getattr(o, target).value)
        # Algebraically identical; the residual is the kernel's numerical
        # Kepler solve at ts/tp (the closed form uses the exact anomaly).
        # 1e-9 d = 86 us on a ~600 s shift.
        assert got == pytest.approx(want, abs=1e-9), (observed, got, want)
        shifts[observed] = float(got[0])

    a_over_c = float(_val(model, point, o.a.value)[0]) / C_LIGHT_RSUN_PER_DAY
    # Planet in front at conjunction: observed EARLY.  Behind at eclipse:
    # observed LATE.  Both of order a/c (~500 s here, a ~ 1 AU).
    assert shifts["tc"] < 0 < shifts["ts"]
    assert 0.3 * a_over_c < -shifts["tc"] < 1.5 * a_over_c
    assert 0.3 * a_over_c < shifts["ts"] < 1.5 * a_over_c
    # tp is the observed PERIASTRON (JDE 2026-09-21), so its shift is
    # z(tp)'s, not tc's: on this orbit they differ by tens of seconds.
    assert abs(shifts["tp"] - shifts["tc"]) > 10.0 / 86400.0


def test_the_kepler_epoch_descends_from_the_target_frame(tmp_path):
    """tp_target is calc_tp of tc_target, not of the sampled tc: the
    Kepler solves see the target frame.  Checked through the values: with
    the shift removed, tp_target - tc_target is the ecc/omega-only offset
    that tp - tc would have had on an un-retarded orbit."""
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    params = _eccentric_inclined(_ltt_wiring_params())
    on, m_on, p_on = _build(_ltt_wiring_config(lc), params)
    off, m_off, p_off = _build(
        _ltt_wiring_config(lc, light_travel_time=False), params
    )
    d_on = _val(
        m_on, p_on, on.orbit.tp_target.value - on.orbit.tc_target.value
    )
    d_off = _val(
        m_off, p_off, off.orbit.tp_target.value - off.orbit.tc_target.value
    )
    assert d_on == pytest.approx(d_off, abs=1e-9)
    # ...whereas the observed pair carries the periastron-vs-conjunction
    # shift difference on the retarded orbit.
    d_obs = _val(m_on, p_on, on.orbit.tp.value - on.orbit.tc.value)
    assert abs(float(d_obs[0] - d_on[0])) > 10.0 / 86400.0


def test_ltt_off_everywhere_makes_the_frames_coincide(tmp_path):
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    system, model, point = _build(
        _ltt_wiring_config(lc, light_travel_time=False),
        _eccentric_inclined(_ltt_wiring_params()),
    )
    o = system.orbit
    assert o._ltt_report_mask.tolist() == [0.0]
    for observed, target in PAIRS:
        assert observed in o.manifest and target in o.manifest
        got = _val(model, point, getattr(o, observed).value)
        want = _val(model, point, getattr(o, target).value)
        assert np.array_equal(got, want), (observed, got, want)


def test_labels_plain_names_are_bjd_tdb_and_target_rows_say_so(tmp_path):
    lc = _write_two_row_lc(tmp_path / "lc.dat")
    system, _, _ = _build(_ltt_wiring_config(lc), _ltt_wiring_params())
    o = system.orbit
    assert o.tc.latex == "T_C"
    assert o.ts.latex == "T_S"
    assert o.tp.latex == "T_P"
    assert o.tc_target.latex == r"T_{C,\rm target}"
    assert o.ts_target.latex == r"T_{S,\rm target}"
    assert o.tp_target.latex == r"T_{P,\rm target}"
    assert "target frame" in o.tc_target.description
    assert "target" not in o.tc.description.lower()
    assert "bjd" not in o.tc_target.description.lower()


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
    assert _shift(o, model, point, "tc", "tc_target")[0] < 0


def test_rm_alone_retards_the_orbit_it_names(tmp_path):
    rv = _write_two_row_rv(tmp_path / "rv.dat")
    system, model, point = _build(_rm_wiring_config(rv), _rm_wiring_params())
    o = system.orbit
    assert o._ltt_report_mask.tolist() == [1.0]
    assert _shift(o, model, point, "tc", "tc_target")[0] < 0

    rv2 = _write_two_row_rv(tmp_path / "rv2.dat")
    system_off, _, _ = _build(
        _rm_wiring_config(rv2, light_travel_time=False), _rm_wiring_params()
    )
    assert system_off.orbit._ltt_report_mask.tolist() == [0.0]


def test_mask_is_per_orbit_a_planetless_orbit_is_not_retarded(tmp_path):
    """Two orbits, one transit file: the planet's orbit is retarded, the
    stellar-companion orbit (no planet, so nothing in the transit model
    touches it) is not, and its frames coincide."""
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
    shift = _shift(o, model, point, "tc", "tc_target")
    assert shift[0] < 0
    assert shift[1] == 0.0


def test_an_orbit_without_masses_still_reports_both_frames_as_one():
    """Geometry-only orbit (its implicit planet does not exist, so no
    a/m_*): the twins take the `no_bodies` identity expression rather than
    disappearing, and no LTT deps are demanded of a component that cannot
    supply them."""
    system = System(
        {"star": [{"name": "A", "mist": False}], "orbit": [{"name": "b"}]},
        user_params={},
    )
    system.prepare()
    o = system.orbit
    assert "a" not in o.manifest
    for key in ("tc_target", "ts", "tp"):
        assert o.manifest[key]["expr_key"] == "no_bodies"
    assert "ts_target" in o.manifest and "tp_target" in o.manifest
