"""A lens pair closer than S_MERGE is ONE lens (review 1.6.22; mulensing.md,
"A lens pair closer than S_MERGE is one lens").

A linear (Taylor) lens orbit's s(t) = s_0 + ds_dt (t - t0_par) can cross
zero inside a multi-year light curve.  The vbm_direct Op used to clamp it at
S_FLOOR = 1e-6 and hand VBM two lenses 1e-6 Einstein radii apart, where its
polynomial roots are inaccurate and MultiMag2 returned magnifications wrong
by up to 10% (OGLE-2013-BLG-0341's close seeds: 8,448 of 23,287 calls).  The
Op now merges such a pair -- the summed mass at its center of mass -- before
calling VBM, and Lens.build_likelihood warns when a linear orbit drives a
separation below S_MERGE inside the data span.  Pinned here:

1. A three-lens geometry with the pair at 1e-6 equals the merged two-lens
   BinaryMag2 to VBM's Tol, statically and on the moving path.
2. A binary whose pair collapses is the single lens (ESPL), epoch by epoch.
3. Crossing the threshold is continuous to well inside Tol.
4. The merged path carries the astrometric centroid in the caller's frame.
5. The Op still refuses a gradient on both sides (it never had one), and
   the curve's finite-difference slope in s is finite on both sides.
6. The warning fires for a linear orbit whose s(t) crosses zero inside the
   span, naming the seed, and not for one that does not.
"""

import logging

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import VBMicrolensing

from exozippy.components.mulensing.op import (
    VBMDirectMagOp,
    close_lens_groups,
    merge_lens_groups,
)
from exozippy.components.mulensing.physics import S_MERGE

_V = VBMicrolensing.VBMicrolensing
_COORDS = "268.0d -29.0d"
_T0 = 2456400.0
_RHO = 3e-3
_U1 = 0.6
_TOL = 1e-3  # the Op's default VBM Tol
# planet: companion 0 (s 0.95, q 1.7e-4 of the primary, alpha 175 deg);
# stellar companion: companion 1 (q 2.42 of the primary) -- the
# OGLE-2013-BLG-0341 close layout (Gould et al. 2014), with the stellar pair
# collapsed the way the linear orbit collapses it.
_PLANET = (0.95, 1.7e-4, 175.0)
_STAR_Q, _STAR_ALPHA = 2.42, 190.0


def _times(n=61, half=20.0):
    return np.linspace(_T0 - half, _T0 + half, n)


def _params(s_star, u_0=0.02, with_ld=True):
    p = [_T0, u_0, 30.0, 0.0, 0.0, _RHO]
    p += list(_PLANET) + [s_star, _STAR_Q, _STAR_ALPHA]
    if with_ld:
        p.append(_U1)
    return np.array(p, dtype=float)


def _vbm():
    v = _V()
    v.Tol = _TOL
    v.RelTol = 0.0
    v.SetLDprofile(_V.LDlinear)
    v.a1 = _U1
    v.minannuli = 2
    return v


def _trajectory(times, u_0):
    tau = (times - _T0) / 30.0
    return -tau, np.full_like(tau, -u_0)


def _merged_binary_reference(times, u_0, s_star):
    """The merged two-lens BinaryMag2 computed here from first principles:
    positions and masses as the Op lays them out (C15, center of mass at
    the origin), the stellar pair replaced by its summed mass at its center
    of mass, and the source rotated into VBM's binary frame."""
    m0 = 1.0 / (1.0 + _PLANET[1] + _STAR_Q)
    m = np.array([m0, _PLANET[1] * m0, _STAR_Q * m0])
    al = np.radians([_PLANET[2], _STAR_ALPHA])
    pos = np.array(
        [
            [0.0, 0.0],
            [_PLANET[0] * np.cos(al[0]), -_PLANET[0] * np.sin(al[0])],
            [s_star * np.cos(al[1]), -s_star * np.sin(al[1])],
        ]
    )
    pos -= m @ pos
    pc = (m[0] * pos[0] + m[2] * pos[2]) / (m[0] + m[2])
    d = pos[1] - pc
    s = np.hypot(*d)
    e = d / s
    q = m[1] / (m[0] + m[2])
    v = _vbm()
    x, y = _trajectory(times, u_0)
    out = []
    for xi, yi in zip(x, y):
        r = np.array([xi, yi])  # the merged pair keeps the center of mass
        out.append(
            v.BinaryMag2(s, q, r @ e, r @ np.array([-e[1], e[0]]), _RHO)
        )
    return np.array(out)


def _static_op(n_companions=2, astrometry=False):
    op = VBMDirectMagOp(
        coords=_COORDS,
        n_companions=n_companions,
        use_rho=True,
        bandpass="I",
        astrometry=astrometry,
    )
    p, t, o = pt.dvector("p"), pt.dvector("t"), pt.dmatrix("o")
    return pytensor.function([p, t, o], op(p, t, o))


def _moving_op(n_companions=2):
    op = VBMDirectMagOp(
        coords=_COORDS,
        n_companions=n_companions,
        use_rho=True,
        bandpass="I",
        orbital_motion=True,
    )
    p, t, o = pt.dvector("p"), pt.dvector("t"), pt.dmatrix("o")
    s_m, a_m = pt.dmatrix("s"), pt.dmatrix("a")
    return pytensor.function([p, t, o, s_m, a_m], op(p, t, o, s_m, a_m))


# ---------------------------------------------------------------------------
# 1. three lenses, pair at 1e-6 == the merged binary
# ---------------------------------------------------------------------------


def test_close_lens_groups_single_linkage():
    pos = np.array([[0.0, 0.0], [0.9, 0.1], [4e-4, 0.0], [0.9 + 5e-4, 0.1]])
    assert close_lens_groups(pos) == [[0, 2], [1, 3]]
    assert close_lens_groups(np.array([[0.0, 0.0], [0.5, 0.0]])) is None
    merged_pos, merged_m = merge_lens_groups(
        pos, np.array([0.4, 0.1, 0.3, 0.2]), [[0, 2], [1, 3]]
    )
    np.testing.assert_allclose(merged_m, [0.7, 0.3])
    np.testing.assert_allclose(merged_pos[0], [0.3 * 4e-4 / 0.7, 0.0])


def test_three_lens_pair_at_1e_6_equals_the_merged_binary():
    """
    Given: the OGLE-2013-BLG-0341 close layout with the stellar pair 1e-6
           Einstein radii apart (where the old S_FLOOR clamp put it),
    When:  the static three-lens Op evaluates a peak-crossing trajectory,
    Then:  it returns the merged two-lens BinaryMag2 to within VBM's Tol --
           here to rounding, since it IS that call.
    """
    times = _times()
    obs = np.zeros((times.size, 3))
    got = _static_op()(_params(1e-6), times, obs)
    want = _merged_binary_reference(times, 0.02, 1e-6)
    assert want.max() > 20.0  # the trajectory really crosses the peak
    np.testing.assert_allclose(got, want, rtol=0, atol=_TOL)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_moving_pair_through_zero_equals_the_merged_binary_where_closed():
    """
    Given: the stellar companion on a linear s(t) that crosses zero mid-span
           (negative before, as a linear orbit carries it),
    When:  the moving three-lens Op evaluates the curve,
    Then:  every epoch with s(t) < S_MERGE is the merged binary, and the
           others are the static three-lens Op at that epoch's geometry.
    """
    times = _times()
    obs = np.zeros((times.size, 3))
    s_star = 0.05 * (times - _T0) / 10.0  # 0 at _T0, +/-0.1 at the ends
    S = np.vstack([np.full_like(times, _PLANET[0]), s_star])
    AL = np.vstack(
        [np.full_like(times, _PLANET[2]), np.full_like(times, _STAR_ALPHA)]
    )
    got = _moving_op()(_params(0.0), times, obs, S, AL)
    closed = s_star < S_MERGE
    assert 0 < closed.sum() < times.size
    # merged epochs: the merged binary (whose geometry does not depend on how
    # far below the threshold the pair is)
    want_closed = _merged_binary_reference(times[closed], 0.02, 0.0)
    np.testing.assert_allclose(got[closed], want_closed, rtol=1e-12)
    # open epochs: the static three-lens Op at each epoch's own separation
    f_static = _static_op()
    for i in np.flatnonzero(~closed):
        want = f_static(_params(s_star[i]), times[i : i + 1], obs[i : i + 1])
        np.testing.assert_allclose(got[i], want[0], rtol=1e-12)


def test_binary_pair_collapse_is_the_single_lens():
    """
    Given: a binary (one companion) whose linear s(t) crosses zero,
    When:  the moving binary Op evaluates the curve,
    Then:  the epochs with s(t) < S_MERGE are ESPLMag2 of the whole mass at
           the center of mass, and no epoch is non-finite.
    """
    times = _times()
    obs = np.zeros((times.size, 3))
    S = np.array([0.01 * (times - _T0) + 0.1])  # zero at _T0 - 10 d
    AL = np.full_like(S, 30.0)
    p = np.array([_T0, 0.02, 30.0, 0.0, 0.0, _RHO, 0.1, 0.5, 30.0, _U1])
    got = _moving_op(n_companions=1)(p, times, obs, S, AL)
    closed = S[0] < S_MERGE
    assert closed.sum() >= 10
    assert np.all(np.isfinite(got))
    v = _vbm()
    u = np.hypot((times - _T0) / 30.0, 0.02)
    want = np.array([v.ESPLMag2(ui, _RHO) for ui in u[closed]])
    np.testing.assert_allclose(got[closed], want, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. continuity at the threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_companions", [1, 2], ids=["binary", "triple"])
def test_crossing_the_threshold_is_continuous_within_tol(n_companions):
    """
    Given: the pair just above and just below S_MERGE (static geometry),
    When:  both evaluate one peak-crossing trajectory,
    Then:  the two curves differ by less than VBM's Tol everywhere (and by
           less than 1e-4 of A): the merge error at S_MERGE plus the
           difference between the two VBM integrators is all a sampler
           crossing the threshold sees.  Measured jumps at 1e-3 are a few
           1e-6 of A (mulensing.md).
    """
    times = _times(n=81, half=3.0)
    obs = np.zeros((times.size, 3))
    f = _static_op(n_companions=n_companions)
    if n_companions == 1:

        def params(s):
            return np.array(
                [_T0, 0.02, 30.0, 0.0, 0.0, _RHO, s, _STAR_Q, 30.0, _U1]
            )

    else:
        params = _params
    above = f(params(S_MERGE * (1 + 1e-9)), times, obs)
    below = f(params(S_MERGE * (1 - 1e-9)), times, obs)
    assert above.max() > 20.0
    jump = np.abs(above - below)
    assert jump.max() < _TOL
    assert np.max(jump / below) < 1e-4


# ---------------------------------------------------------------------------
# 3. the centroid on the merged path
# ---------------------------------------------------------------------------


def test_merged_path_carries_the_centroid():
    """
    Given: the astrometric three-lens Op with the pair at 1e-6,
    When:  it returns (A, dtau, dbeta),
    Then:  A is the photometric Op's, and the centroid shift is the merged
           binary's -- VBM's astrox read after the merged call, rotated back
           from the binary frame -- matching the pair at 2e-3 (unmerged,
           where VBM is accurate) to 1e-3 Einstein radii, against shifts of
           a few 1e-2 that a wrong rotation would scramble.
    """
    times = _times()
    obs = np.zeros((times.size, 3))
    A_m, dt_m, db_m = _static_op(astrometry=True)(_params(1e-6), times, obs)
    A_p = _static_op()(_params(1e-6), times, obs)
    np.testing.assert_allclose(A_m, A_p, rtol=1e-12)
    A_o, dt_o, db_o = _static_op(astrometry=True)(_params(2e-3), times, obs)
    np.testing.assert_allclose(dt_m, dt_o, rtol=0, atol=1e-3)
    np.testing.assert_allclose(db_m, db_o, rtol=0, atol=1e-3)
    assert np.max(np.abs(dt_m)) > 1e-2 and np.max(np.abs(db_m)) > 1e-2


# ---------------------------------------------------------------------------
# 4. gradients: none through the Op on either side, finite slope in s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s_star", [2.0 * S_MERGE, 0.5 * S_MERGE])
def test_gradient_is_refused_loudly_on_both_sides(s_star):
    """
    Given: the Op on either side of the threshold,
    When:  a gradient is requested,
    Then:  it raises NotImplementedError, as it always has (mulensing.md,
           "No gradient through the Op path, loudly"): the merge lives in
           numpy inside the Op, so it adds no pt.where branch and no floor
           to any graph a gradient flows through.  What a gradient-free
           sampler sees is the curve, and its finite-difference slope in s
           is finite on both sides.
    """
    op = VBMDirectMagOp(
        coords=_COORDS, n_companions=2, use_rho=True, bandpass="I"
    )
    p, t, o = pt.dvector("p"), pt.dvector("t"), pt.dmatrix("o")
    with pytest.raises(NotImplementedError):
        pytensor.grad(op(p, t, o).sum(), p)

    times = _times(n=21, half=3.0)
    obs = np.zeros((times.size, 3))
    f = _static_op()
    h = 1e-5 * s_star
    slope = (
        f(_params(s_star + h), times, obs) - f(_params(s_star - h), times, obs)
    ) / (2 * h)
    assert np.all(np.isfinite(slope))


# ---------------------------------------------------------------------------
# 5. the warning
# ---------------------------------------------------------------------------

_T0_PAR = 2458554.89


def _write_lc(path, n=120):
    rng = np.random.default_rng(7)
    t = np.linspace(_T0_PAR - 40.0, _T0_PAR + 40.0, n)
    mag = 15.0 - rng.uniform(0, 0.001, n)
    np.savetxt(path, np.column_stack([t, mag, np.full(n, 0.01)]))
    return str(path)


def _linear_system(tmp_path, ds_dt):
    from exozippy.system import System

    lc = _write_lc(tmp_path / "lc.dat")
    config = {
        "star": [{"name": "L1"}, {"name": "L2"}, {"name": "Source"}],
        "mulensevent": [{"finite_source": True, "t0_par": _T0_PAR}],
        "lens": [{"body": "star.L1"}, {"body": "star.L2", "orbit": "L"}],
        "orbit": [
            {
                "name": "L",
                "type": "linear",
                "primary": ["L1"],
                "companion": ["L2"],
            }
        ],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    params = {
        "source.Source.t_0": {"initval": _T0_PAR},
        "source.Source.u_0": {"initval": 0.1},
        "mulensevent.t_E": {"initval": 30.0},
        "source.Source.rho": {"initval": 0.002},
        "lens.L2.s": {"initval": 0.5},
        "lens.L2.alpha": {"initval": 40.0},
        "lens.L2.q": {"initval": 0.3},
        "mulensevent.pi_E_N": {"initval": 0.1},
        "mulensevent.pi_E_E": {"initval": 0.1},
        "orbit.L.ds_dt": {"initval": ds_dt},
        "orbit.L.dalpha_dt": {"initval": 5.0},
        "star.L1.mass": {"initval": 0.5},
        "star.L1.distance": {"initval": 3000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("L1", "L2", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 268.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -29.0, "sigma": 0}
    system = System(config, user_params=params)
    system.prepare()
    return system


@pytest.mark.slow
def test_warning_names_the_seed_whose_linear_orbit_crosses_zero(
    tmp_path, caplog
):
    """
    Given: a binary on a linear orbit with s_0 = 0.5 at t0_par over a +/-40 d
           light curve, and two seeds: ds_dt = 0.1 /yr (s stays near 0.5)
           and 6 /yr (s crosses zero 30 d before t0_par),
    When:  the model is built,
    Then:  ONE warning names the companion, the orbit and seed 1 -- with
           the crossing epoch -- and not seed 0.
    """
    system = _linear_system(tmp_path, [0.1, 6.0])
    with caplog.at_level(logging.WARNING):
        system.build_model()
    msgs = [
        r.getMessage() for r in caplog.records if "LINEAR" in r.getMessage()
    ]
    assert len(msgs) == 1
    msg = msgs[0]
    assert "star.L2" in msg and "orbit 'L'" in msg
    assert "seed 1" in msg and "seed 0" not in msg
    t_cross = _T0_PAR - 0.5 / 6.0 * 365.25
    assert f"through zero at BJD {t_cross:.1f}" in msg


@pytest.mark.slow
def test_no_warning_when_the_separation_stays_open(tmp_path, caplog):
    """
    Given: the same binary with ds_dt = 0.1 /yr only,
    When:  the model is built,
    Then:  no linear-orbit collapse warning.
    """
    system = _linear_system(tmp_path, 0.1)
    with caplog.at_level(logging.WARNING):
        system.build_model()
    assert not any("LINEAR" in r.getMessage() for r in caplog.records)
