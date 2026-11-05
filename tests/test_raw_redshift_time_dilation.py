"""The raw-redshift time dilation's transverse reflex (review 8.5.10;
rvinstrument.md, "Without astrometry").

``gamma_*`` needs the star's whole speed.  Without astrometry (no
``bigomega``) the model used to carry only the line-of-sight reflex in
``beta^2``, which put a spurious ``K^2/(4c) cos(2 phase)`` on a circular
orbit; it now adds each orbit's transverse SPEED, which needs no node.
Checked on the independent numpy harness of tests/test_raw_redshift_orbits.py:

* a circular edge-on SB1 without astrometry against the full 3-D model at
  three nodes: no 2f residual, only the node-dependent cross term at the
  orbital frequency (which a Keplerian absorbs);
* the same SB1 WITH relative astrometry: the 3-D path never reads the new
  term and matches the 3-D model;
* an eccentric SB1: the added term is (|v|^2 - v_los^2)/2c with |v|^2
  from vis-viva.
"""

import numpy as np
import pytest
from test_raw_redshift_orbits import (
    _at,
    _build,
    _elements,
    _independent_cz,
    _model_cz,
    _raw_entry,
    _star_motion,
    _star_params,
    _star_start,
    _star_velocity3,
    _times,
    _write,
)

from exozippy.constants import C_MPS

pytestmark = pytest.mark.slow

#: An SB1: A (1.0 Msun) observed, B (0.5 Msun) unseen, P = 10 d, so the
#: primary's K ~ 38 km/s; 60 epochs over four orbits.
SB1_SPAN = (2460000.0, 2460040.0)


def _sb1(tmp_path_factory, tag, ecc=None, astrometry_node=None):
    """Build the SB1 above.  ``ecc`` -- None for a circular orbit (the
    sqrt(e) pair pinned at zero, edge-on), else ``(e, omega [deg], cos
    i)``.  ``astrometry_node`` -- None for raw RVs alone (no ``bigomega``),
    else a node [deg] and a three-epoch relative-astrometry file, which is
    there only to make the system model the sky orientation."""
    d = tmp_path_factory.mktemp(f"rawz_sb1_{tag}")
    jd = _times(n=60, span=SB1_SPAN, seed=5)
    path = _write(d / "A.rv", jd, np.full(jd.size, -2.0e4))
    cfg = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [{"name": "AB", "primary": ["A"], "companion": ["B"]}],
        "rvinstrument": [_raw_entry(path, star="A")],
    }
    p = _star_params({"A": 1.0, "B": 0.5})
    p.update(
        {
            "orbit.AB.period": {"initval": 10.0},
            "orbit.AB.tc": {"initval": 2460010.0},
            "rvinstrument.HPF.gamma": {"initval": -19990.0},
        }
    )
    if ecc is None:
        p.update(
            {
                "orbit.AB.secosw": {"initval": 0.0, "sigma": 0.0},
                "orbit.AB.sesinw": {"initval": 0.0, "sigma": 0.0},
                "orbit.AB.cosi": {"initval": 0.0},
            }
        )
    else:
        e, w, ci = ecc
        p.update(
            {
                "orbit.AB.ecc": {"initval": e},
                "orbit.AB.omega": {"initval": w},
                "orbit.AB.cosi": {"initval": ci},
            }
        )
    if astrometry_node is not None:
        rel = d / "AB.rel"
        np.savetxt(
            rel,
            np.column_stack(
                [
                    np.array([2460001.0, 2460004.0, 2460007.0]),
                    np.full(3, 1.4),
                    np.full(3, 0.1),
                    np.array([10.0, 70.0, 130.0]),
                    np.full(3, 5.0),
                ]
            ),
        )
        cfg["astrometryinstrument"] = [
            {"name": "Rel", "file": str(rel), "mode": "rel", "orbit": "AB"}
        ]
        p["orbit.AB.bigomega"] = {"initval": astrometry_node}
    return _build(cfg, p), jd


def _harmonics(system, model, start, resid):
    """Least-squares amplitudes (m/s) of ``resid`` at the orbital frequency
    and its first harmonic, in the star's orbital phase at each epoch
    (with an offset and a slope), and the RMS left after removing them."""
    (t_b,) = _at(model, start, [system.rvinstrument._raw_t_bary_node])
    (P,) = _at(model, start, [system.orbit.period.value])
    (tp,) = _at(model, start, [system.orbit.tp_target.value])
    ph = 2 * np.pi * (t_b - tp[0]) / P[0]
    A = np.column_stack(
        [
            np.ones_like(ph),
            t_b - t_b.mean(),
            np.cos(ph),
            np.sin(ph),
            np.cos(2 * ph),
            np.sin(2 * ph),
        ]
    )
    c, *_ = np.linalg.lstsq(A, resid, rcond=None)
    left = resid - A @ c
    return np.hypot(c[2], c[3]), np.hypot(c[4], c[5]), np.std(left)


def _sb1_inputs(system, model, start):
    orbits = _elements(system, model, start, {0: "primary"})
    (gamma,) = _at(model, start, [system.rvinstrument.gamma.value])
    star = _star_start(system, model, start, 0)
    gamma_ms = float(gamma[0]) * system.rvinstrument._rv_factor()
    return orbits, star, gamma_ms


def _vt_sys(star):
    """The star's systemic transverse speed (m/s) from pm and parallax."""
    _, _, pm_ra, pm_dec, plx, _ = star
    return 4740.47 * np.hypot(pm_ra, pm_dec) / plx


@pytest.fixture(scope="module")
def circular_sb1(tmp_path_factory):
    return _sb1(tmp_path_factory, "circ")


@pytest.fixture(scope="module")
def circular_sb1_astrometry(tmp_path_factory):
    """The same SB1 with relative astrometry, node 60 deg: the 3-D path.
    Built with ``_reflex_transverse_speed2`` made to raise, so the 3-D
    path provably never reads the speed-only term: its graph is the one
    from before review 8.5.10, node for node."""
    from unittest import mock

    from exozippy.components.rvinstrument.rvinstrument import RVInstrument

    def refuse(*_a, **_k):
        raise AssertionError("the 3-D path read the speed-only term")

    with mock.patch.object(RVInstrument, "_reflex_transverse_speed2", refuse):
        return _sb1(tmp_path_factory, "astrom", astrometry_node=60.0)


def test_a_circular_sb1_has_no_spurious_2f_term_without_astrometry(
    circular_sb1,
):
    """
    Given an edge-on circular SB1 (K ~ 38 km/s) with raw redshifts and no
      astrometry (no bigomega),
    When the model is compared with the independent model whose time
      dilation sees the full 3-D reflex velocity, at three nodes,
    Then the residual has no component at twice the orbital frequency
      (< 0.5 mm/s), while the pre-fix model -- the line-of-sight reflex
      alone in gamma_* -- carries K^2/(4c) there (~1.2 m/s); what is left
      is the node-dependent transverse cross term v_sys,perp . v_perp / c
      at the orbital frequency, bounded by v_t,sys (K/sin i) / c (~3 m/s),
      which a Keplerian absorbs (shifts of tc and K) to < 1 mm/s RMS.
      Measured: 2f 0.10-0.24 mm/s; 1f 0.81, 2.54, 3.35 m/s at nodes 0,
      60, 120 deg against a bound of 3.40; 0.3-0.6 mm/s left.
    """
    (system, model, start), jd = circular_sb1
    assert "bigomega" not in system.orbit.manifest
    orbits, star, gamma_ms = _sb1_inputs(system, model, start)
    K = abs(orbits[0]["share"]) * (
        2 * np.pi / (orbits[0]["P"] * 86400.0) * orbits[0]["a"]
    )
    mod = _model_cz(system, model, start)
    speed = _independent_cz(jd, orbits, star, gamma_ms, star[5])
    assert np.max(np.abs(mod - speed)) < 1e-3
    old = _independent_cz(
        jd, orbits, star, gamma_ms, star[5], transverse="none"
    )
    _, old_2f, _ = _harmonics(system, model, start, old - speed)
    print(f"K = {K:.0f} m/s; pre-fix 2f term {old_2f:.3f} m/s")
    assert old_2f == pytest.approx(K**2 / (4 * C_MPS), rel=0.02)
    bound = _vt_sys(star) * K / C_MPS
    for node in (0.0, 60.0, 120.0):
        full = _independent_cz(
            jd,
            orbits,
            star,
            gamma_ms,
            star[5],
            transverse="sky",
            bigomega=[np.radians(node)],
        )
        a1, a2, left = _harmonics(system, model, start, mod - full)
        print(
            f"node {node:5.1f}: 1f {a1:.3f} m/s (bound {bound:.3f}), "
            f"2f {a2 * 1e3:.3f} mm/s, left {left * 1e3:.3f} mm/s"
        )
        assert a2 < 5e-4
        assert a1 < 1.05 * bound
        assert left < 1e-3


def test_the_3d_path_is_unchanged_and_matches_the_full_model(
    circular_sb1, circular_sb1_astrometry
):
    """
    Given the same SB1 with relative astrometry (bigomega = 60 deg), built
      with the speed-only term made to raise,
    When its model is compared with the independent 3-D model at that node
      and with the astrometry-free model,
    Then it matches the 3-D model to < 1 mm/s (the sky wobble of rho_hat
      it carries and the independent model does not is far below that),
      and differs from the astrometry-free one only by the transverse
      cross term: at the orbital frequency, no 2f component.
    """
    (system, model, start), jd = circular_sb1_astrometry
    assert "bigomega" in system.orbit.manifest
    orbits, star, gamma_ms = _sb1_inputs(system, model, start)
    (node,) = _at(model, start, [system.orbit.bigomega.value])
    assert np.degrees(node[0]) == pytest.approx(60.0, abs=1e-6)
    mod = _model_cz(system, model, start)
    full = _independent_cz(
        jd, orbits, star, gamma_ms, star[5], transverse="sky", bigomega=node
    )
    assert np.max(np.abs(mod - full)) < 1e-3
    (s0, m0, st0), _ = circular_sb1
    a1, a2, left = _harmonics(s0, m0, st0, _model_cz(s0, m0, st0) - mod)
    print(f"no-astrometry - 3-D: 1f {a1:.3f} m/s, 2f {a2 * 1e3:.3f} mm/s")
    assert a1 > 0.1
    assert a2 < 5e-4


@pytest.fixture(scope="module")
def eccentric_sb1(tmp_path_factory):
    return _sb1(tmp_path_factory, "ecc", ecc=(0.4, 50.0, 0.3))


def test_an_eccentric_sb1s_time_dilation_is_its_whole_orbital_speed(
    eccentric_sb1,
):
    """
    Given an eccentric (e = 0.4, cos i = 0.3) SB1 without astrometry,
    When the model is compared with the pre-fix (line-of-sight-only) model,
    Then the difference is the transverse share of the time dilation,
      (|v|^2 - v_los^2) / 2c with |v|^2 from vis-viva -- an independent
      formula from the model's phase terms -- to < 1 mm/s, and the whole
      dilation |v|^2 / 2c it completes varies by meters per second over
      the orbit.
    """
    (system, model, start), jd = eccentric_sb1
    orbits, star, gamma_ms = _sb1_inputs(system, model, start)
    mod = _model_cz(system, model, start)
    speed = _independent_cz(jd, orbits, star, gamma_ms, star[5])
    assert np.max(np.abs(mod - speed)) < 1e-3
    old = _independent_cz(
        jd, orbits, star, gamma_ms, star[5], transverse="none"
    )
    # The same quantities at the model's own retarded times, by hand.
    (t_b,) = _at(model, start, [system.rvinstrument._raw_t_bary_node])
    o = orbits[0]
    t_emit = np.array(t_b)
    for _ in range(8):
        Z, _v = _star_motion(t_emit, [o])
        t_emit = t_b - Z / C_MPS / 86400.0
    vz, _, _, vperp2 = _star_velocity3(t_emit, o)
    expected = vperp2 / (2 * C_MPS)
    assert np.max(np.abs((mod - old) - expected)) < 1e-3
    dil = (vperp2 + vz**2) / (2 * C_MPS)
    print(
        f"time dilation |v|^2/2c: ptp {np.ptp(dil):.3f} m/s; its "
        f"transverse share ptp {np.ptp(expected):.3f} m/s"
    )
    assert np.ptp(dil) > 1.0
