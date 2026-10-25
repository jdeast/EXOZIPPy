"""Lens orbital motion for EVERY companion (review 8.6.33; mulensing.md,
"Three or more lens bodies").

Until 2026-10 exactly one companion could move: Lens kept one moving orbit
and raised otherwise, and VBMDirectMagOp took one (s_t, alpha_t) series into
companion 0 and set the N-lens geometry once.  Now each companion follows
its own orbit (linear or keplerian) or stays static, and the N-lens path sets
the geometry per epoch.  Pinned here:

1. The moving N-lens Op equals, epoch by epoch, the STATIC N-lens Op built
   at that epoch's geometry -- an independent reference that shares only
   VBM, with finite and point sources and with the astrometric centroid.
2. A system with one moving and one static companion builds; the static
   companion's row of the geometry series is its constant.
3. A system with two companions, each on its own linear orbit, builds and
   gets one bound-orbit potential per moving companion.
4. Two companions sharing one orbit are refused (a companion GROUP needs the
   hierarchical composition of phase 2).
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from test_lens_orbital_motion import _T0_PAR, _write_binary_lc

from exozippy.components.mulensing.op import VBMDirectMagOp
from exozippy.constants import DAYS_PER_YEAR

pytestmark = pytest.mark.slow

_COORDS = "268.0d -29.0d"
_BASE = [_T0_PAR, 0.05, 40.0, 0.0, 0.0]  # t_0, u_0, t_E, pi_E_N, pi_E_E
_RHO = 0.002
# (s_0, q, alpha_0 [deg], ds_dt [1/yr], dalpha_dt [deg/yr]) per companion:
# a stellar companion and a planet, both moving.
_COMPANIONS = [(0.8, 0.25, 150.0, 0.3, 40.0), (1.3, 3e-3, 60.0, -0.2, -25.0)]


def _times():
    return np.linspace(_T0_PAR - 30.0, _T0_PAR + 30.0, 41)


def _series(times):
    dt = (times - _T0_PAR) / DAYS_PER_YEAR
    S = np.array([s0 + ds * dt for s0, _, _, ds, _ in _COMPANIONS])
    A = np.array([a0 + da * dt for _, _, a0, _, da in _COMPANIONS])
    return S, A


def _params(rho, s_alpha):
    p = list(_BASE) + ([rho] if rho else [])
    for (s, a), (_, q, _, _, _) in zip(s_alpha, _COMPANIONS):
        p += [s, q, a]
    return np.array(p, dtype=float)


@pytest.mark.parametrize("use_rho", [True, False], ids=["finite", "point"])
@pytest.mark.parametrize("astrometry", [False, True], ids=["A", "centroid"])
def test_moving_nlens_op_equals_the_static_op_epoch_by_epoch(
    use_rho, astrometry
):
    times = _times()
    obs = np.zeros((times.size, 3))
    S, A = _series(times)
    rho = _RHO if use_rho else None
    common = dict(
        coords=_COORDS,
        n_companions=len(_COMPANIONS),
        use_rho=use_rho,
        astrometry=astrometry,
    )

    p, t, o = pt.dvector("p"), pt.dvector("t"), pt.dmatrix("o")
    s_m, a_m = pt.dmatrix("s"), pt.dmatrix("a")
    moving = VBMDirectMagOp(orbital_motion=True, **common)
    f_moving = pytensor.function(
        [p, t, o, s_m, a_m], moving(p, t, o, s_m, a_m)
    )
    static = VBMDirectMagOp(**common)
    f_static = pytensor.function([p, t, o], static(p, t, o))

    # The param vector's static entries are the t0_par anchors.
    p0 = _params(rho, [(s0, a0) for s0, _, a0, _, _ in _COMPANIONS])
    got = f_moving(p0, times, obs, S, A)
    want = [
        f_static(
            _params(rho, list(zip(S[:, i], A[:, i]))),
            times[i : i + 1],
            obs[i : i + 1],
        )
        for i in range(times.size)
    ]
    if astrometry:
        for k in range(3):
            np.testing.assert_allclose(
                got[k], np.concatenate([w[k] for w in want]), rtol=1e-12
            )
    else:
        np.testing.assert_allclose(got, np.concatenate(want), rtol=1e-12)
    # and the geometry really moved: the curve differs from the frozen one
    frozen = f_static(p0, times, obs)
    first = got[0] if astrometry else got
    frozen = frozen[0] if astrometry else frozen
    assert np.max(np.abs(first - frozen)) > 1e-4


def _triple_system(tmp_path, orbits, lens_orbit):
    """Three stellar lens bodies; ``lens_orbit`` names each companion's
    orbit (None = static)."""
    from exozippy.system import System

    lc = _write_binary_lc(tmp_path / "lc.dat")
    config = {
        "star": [
            {"name": "L1"},
            {"name": "L2"},
            {"name": "L3"},
            {"name": "Source"},
        ],
        "mulensevent": [{"finite_source": True, "t0_par": _T0_PAR}],
        "lens": [{"body": "star.L1"}]
        + [
            {"body": f"star.{nm}", **({"orbit": o} if o else {})}
            for nm, o in zip(("L2", "L3"), lens_orbit)
        ],
        "orbit": orbits,
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    params = {
        "source.Source.t_0": {"initval": _T0_PAR},
        "source.Source.u_0": {"initval": 0.06},
        "mulensevent.t_E": {"initval": 60.0},
        "source.Source.rho": {"initval": 0.0015},
        "lens.L2.s": {"initval": 0.45},
        "lens.L2.alpha": {"initval": 189.0},
        "lens.L2.q": {"initval": 0.27},
        "lens.L3.s": {"initval": 1.4},
        "lens.L3.alpha": {"initval": 40.0},
        "lens.L3.q": {"initval": 0.01},
        "mulensevent.pi_E_N": {"initval": -0.02},
        "mulensevent.pi_E_E": {"initval": 0.15},
        "star.L1.distance": {"initval": 1100.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for o in orbits:
        params[f"orbit.{o['name']}.ds_dt"] = {"initval": 0.04}
        params[f"orbit.{o['name']}.dalpha_dt"] = {"initval": -100.0}
    for nm in ("L1", "L2", "L3", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 268.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -29.0, "sigma": 0}
    system = System(config, user_params=params)
    system.prepare()
    return system


def _orbit(name, companion):
    return {
        "name": name,
        "type": "linear",
        "primary": ["L1"],
        "companion": [companion],
    }


def _eval(system, model, nodes):
    start = system.get_raw_start(model)
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values(nodes),
        on_unused_input="ignore",
    )
    return fn(*[start[v.name] for v in model.value_vars])


def test_a_static_companion_beside_a_moving_one(tmp_path):
    system = _triple_system(tmp_path, [_orbit("L", "L2")], ["L", None])
    assert system.lens.companion_motion == ["linear", None]
    model = system.build_model()
    times = _times()
    S, A = system.mulensevent._companion_geometry_series(times, system)
    S, A, s3, a3 = _eval(
        system,
        model,
        [S, A, system.lens.s.value[2], system.lens.alpha.value[2]],
    )
    assert S.shape == A.shape == (2, times.size)
    # the moving companion moves; the static one is its constant
    assert np.ptp(S[0]) > 1e-4
    np.testing.assert_allclose(S[1], float(s3), rtol=0, atol=0)
    np.testing.assert_allclose(A[1], np.degrees(float(a3)), rtol=1e-12)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_two_companions_each_on_their_own_orbit(tmp_path):
    system = _triple_system(
        tmp_path, [_orbit("L", "L2"), _orbit("M", "L3")], ["L", "M"]
    )
    assert system.lens.companion_motion == ["linear", "linear"]
    assert system.lens.motion_orbit_idxs == [0, 1]
    model = system.build_model()
    names = {v.name for v in model.potentials}
    assert {"lens.bound_orbit_1", "lens.bound_orbit_2"} <= names
    S, _ = system.mulensevent._companion_geometry_series(_times(), system)
    (S,) = _eval(system, model, [S])
    assert np.ptp(S[0]) > 1e-4 and np.ptp(S[1]) > 1e-4
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_two_companions_on_one_orbit_are_refused(tmp_path):
    orbit = {
        "name": "L",
        "type": "linear",
        "primary": ["L1"],
        "companion": ["L2", "L3"],
    }
    with pytest.raises(NotImplementedError, match="share an orbit"):
        _triple_system(tmp_path, [orbit], ["L", "L"])
