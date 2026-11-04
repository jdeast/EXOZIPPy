"""`type: constant_velocity`: a RELATIVE straight line, and its two readers
(review 8.8.14; orbit.md "Taylor orbits").

A constant velocity is invisible to RVs and absolute astrometry (each
instrument's gamma and the star's proper motion absorb it), so its readers
are relative ones:

  * a microlensing lens companion: the pair's separation moves on the
    straight line ``r(t) = r(t0_par) + v (t - t0_par)`` in the lens frame,
    with ``r(t0_par) = s0 (cos alpha0, -sin alpha0)`` from the LENS (C15)
    and ``v`` the velocity the orbit's (ds_dt, dalpha_dt) define AT t0_par
    -- uniform_rotation's own position and velocity there, so the two
    agree to first order and differ at second (the polar line keeps
    rotating, the straight line does not);
  * a `rel` dataset: the companion's sky offset from the primary,
    ``(rel_dE, rel_dN) + (rel_vE, rel_vN) (t - epoch)`` in mas.

Neither MulensModel (polar-linear or Keplerian lens motion) nor
VBMicrolensing (circular/Keplerian orbital light curves) has a straight-line
lens mode, so the lens reader is validated against the definition (numpy)
for the geometry and against MulensModel STATIC models evaluated epoch by
epoch at that geometry for the magnification.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from test_lens_orbital_motion import (
    _COORDS,
    _MAP,
    _T0_PAR,
    _compile,
    _compile_om,
    _mulensmodel_kernel,
    _om_system,
    _times_and_obs,
)

from exozippy.components.mulensing.op import BinaryLensMagOp, VBMDirectMagOp
from exozippy.constants import DAYS_PER_YEAR
from exozippy.system import System

pytestmark = pytest.mark.slow

_D2R = np.pi / 180.0


def _straight_line(times, p):
    """(s, alpha_deg) of the constant_velocity companion, from the
    definition: uniform_rotation's position and velocity at t0_par, held."""
    dt = (np.asarray(times) - _T0_PAR) / DAYS_PER_YEAR
    a0 = p["alpha"] * _D2R
    dal = p["dalpha_dt"] * _D2R
    vx = p["ds_dt"] * np.cos(a0) - p["s"] * dal * np.sin(a0)
    vy = -(p["ds_dt"] * np.sin(a0) + p["s"] * dal * np.cos(a0))
    x = p["s"] * np.cos(a0) + vx * dt
    y = -p["s"] * np.sin(a0) + vy * dt
    return np.hypot(x, y), np.degrees(np.arctan2(-y, x)), (x, y)


def _polar_line(times, p):
    dt = (np.asarray(times) - _T0_PAR) / DAYS_PER_YEAR
    s = p["s"] + p["ds_dt"] * dt
    a = (p["alpha"] + p["dalpha_dt"] * dt) * _D2R
    return s * np.cos(a), -s * np.sin(a)


def _wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


@pytest.fixture(scope="module")
def lens_pair(tmp_path_factory):
    """The OGLE-2009-BLG-020-like binary of test_lens_orbital_motion on a
    constant_velocity orbit and on a uniform_rotation one, same seeds."""
    cv = _om_system(tmp_path_factory.mktemp("cv"), otype="constant_velocity")
    ur = _om_system(tmp_path_factory.mktemp("ur"), otype="uniform_rotation")
    return cv, ur


def _at_start(system, model, nodes):
    start = system.get_raw_starts(model)[0][0]
    nodes = model.replace_rvs_by_values(nodes)
    fn = pytensor.function(model.value_vars, nodes, on_unused_input="ignore")
    return [
        np.asarray(x) for x in fn(*[start[v.name] for v in model.value_vars])
    ]


# ---------------------------------------------------------------------------
# The lens reader
# ---------------------------------------------------------------------------


def test_the_lens_series_is_the_straight_line(lens_pair):
    """
    Given a lens companion on a constant_velocity orbit seeded at the
      OB09020-like (s0, alpha0, ds_dt, dalpha_dt),
    When the per-epoch geometry the backends receive is evaluated at the
      start over +/-150 d,
    Then it is the straight line through uniform_rotation's position with
      its velocity at t0_par: s to 1e-10 and alpha to 1e-8 deg of the
      numpy definition.
    """
    # Arrange
    (system, model), _ = lens_pair
    times = np.linspace(_T0_PAR - 150.0, _T0_PAR + 150.0, 301)

    # Act
    S, AL = system.mulensevent._companion_geometry_series(times, system)
    s, alpha = _at_start(system, model, [S[0], AL[0]])

    # Assert
    s_ref, a_ref, _ = _straight_line(times, _MAP)
    np.testing.assert_allclose(s, s_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(_wrap_deg(alpha - a_ref), 0.0, atol=1e-8)
    assert system.lens.companion_motion == ["constant_velocity"]


def test_the_two_lens_types_differ_at_second_order(lens_pair):
    """
    Given the same seeds on a constant_velocity and a uniform_rotation
      orbit,
    When the companion's position r(t) = s (cos alpha, -sin alpha) is
      compared between them,
    Then they coincide at t0_par (to 1e-12), the difference grows as dt^2
      (|dr|(2 dt)/|dr|(dt) = 4 to 2% at dt = 5 d) with the leading
      coefficient half the polar line's acceleration at t0_par,
      |(s0 dalpha^2, 2 ds_dt dalpha)| / 2 (to 3%), and at +/-36 d it is
      several source radii --
      the second-order effect a fit can tell apart.
    """
    # Arrange
    (cv, cv_model), (ur, ur_model) = lens_pair
    dts = np.array([0.0, 5.0, 10.0, -36.0, 36.0])
    times = _T0_PAR + dts

    # Act
    rows = []
    for system, model in ((cv, cv_model), (ur, ur_model)):
        S, AL = system.mulensevent._companion_geometry_series(times, system)
        s, a = _at_start(system, model, [S[0], AL[0]])
        rows.append((s * np.cos(a * _D2R), -s * np.sin(a * _D2R)))
    dr = np.hypot(rows[0][0] - rows[1][0], rows[0][1] - rows[1][1])

    # Assert
    assert dr[0] < 1e-12
    assert dr[2] / dr[1] == pytest.approx(4.0, rel=0.02)
    # The polar line's acceleration at t0_par: -s0 dalpha^2 radially,
    # 2 ds_dt dalpha tangentially; the straight line has none.
    dal = _MAP["dalpha_dt"] * _D2R
    acc = np.hypot(_MAP["s"] * dal**2, 2.0 * _MAP["ds_dt"] * dal)
    lead = 0.5 * acc * (5.0 / DAYS_PER_YEAR) ** 2
    assert dr[1] == pytest.approx(lead, rel=0.03)
    # The numpy definitions agree with the model (both types).
    x_cv, y_cv = _straight_line(times, _MAP)[2]
    x_ur, y_ur = _polar_line(times, _MAP)
    np.testing.assert_allclose(
        dr, np.hypot(x_cv - x_ur, y_cv - y_ur), atol=1e-10
    )
    assert min(dr[3], dr[4]) / _MAP["rho"] > 5.0


def test_the_straight_line_magnification_matches_static_mulensmodel(
    lens_pair,
):
    """
    Given random draws about the OB09020-like binary with parallax
      deviations, on the straight-line geometry,
    When the per-epoch vbm_direct Op is fed that geometry, and MulensModel
      is evaluated as a STATIC binary at each epoch's (s, alpha) (it has no
      straight-line mode),
    Then the magnifications agree per point to rtol 1e-8.
    """
    # Arrange
    times, obs = _times_and_obs(n=60)
    vbm_fn = _compile_om(
        _mulensmodel_kernel(
            VBMDirectMagOp(
                coords=_COORDS,
                n_companions=1,
                use_rho=True,
                bandpass="I",
                orbital_motion=True,
            )
        )
    )
    mm_static = _compile(
        BinaryLensMagOp(
            coords=_COORDS, mag_method="VBM", use_rho=True, bandpass="I"
        )
    )
    order = [
        "t_0",
        "u_0",
        "t_E",
        "pi_E_N",
        "pi_E_E",
        "rho",
        "s",
        "q",
        "alpha",
        "u1",
    ]
    rng = np.random.default_rng(20261104)
    for k in range(3):
        p = dict(_MAP)
        p["alpha"] += rng.normal(0, 2.0)
        p["dalpha_dt"] += rng.normal(0, 10.0)
        p["ds_dt"] += rng.normal(0, 0.02)
        s_t, a_t, _ = _straight_line(times, p)

        # Act
        a_vbm = vbm_fn(np.array([p[x] for x in order]), times, obs, s_t, a_t)
        a_mm = np.array(
            [
                mm_static(
                    np.array(
                        [{**p, "s": s_t[i], "alpha": a_t[i]}[x] for x in order]
                    ),
                    times[i : i + 1],
                    obs[i : i + 1],
                )[0]
                for i in range(times.size)
            ]
        )

        # Assert
        np.testing.assert_allclose(a_vbm, a_mm, rtol=1e-8, err_msg=f"draw {k}")


def test_beta_is_shared_and_mulensmodel_is_refused_by_name(lens_pair):
    """
    Given the two lens types at the same seeds,
    When beta (Skowron A19, from the rates at t0_par) and the logp are
      evaluated, and the mulensmodel backend is asked about the straight
      line,
    Then beta is identical (the velocities coincide at t0_par), the logp is
      finite, and MulensModel is refused naming constant_velocity.
    """
    # Arrange
    (cv, cv_model), (ur, ur_model) = lens_pair

    # Act
    (b_cv,) = _at_start(cv, cv_model, [cv.lens.beta.value[1]])
    (b_ur,) = _at_start(ur, ur_model, [ur.lens.beta.value[1]])
    reason = cv.mulensevent.mulensmodel_unsupported(cv)

    # Assert
    assert float(b_cv) == float(b_ur)
    assert np.isfinite(
        cv_model.compile_logp()(cv.get_raw_starts(cv_model)[0][0])
    )
    assert "constant_velocity" in reason


# ---------------------------------------------------------------------------
# The rel reader
# ---------------------------------------------------------------------------

_REL = dict(dE=120.0, dN=-45.0, vE=-8.0, vN=3.5)  # mas, mas/yr
_SEP_ERR = 0.4  # mas


def _write_rel(path, n=24, seed=3, span=(2457000.0, 2459200.0)):
    """Companion-minus-primary sep/PA on the straight line, isotropic 0.4
    mas errors (err_pa = err_sep / sep)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(*span, n))
    t[0], t[-1] = span
    mid = 0.5 * (t[0] + t[-1])
    dt = (t - mid) / DAYS_PER_YEAR
    dE = _REL["dE"] + _REL["vE"] * dt + rng.normal(0, _SEP_ERR, n)
    dN = _REL["dN"] + _REL["vN"] * dt + rng.normal(0, _SEP_ERR, n)
    sep = np.hypot(dE, dN)
    pa = np.degrees(np.arctan2(dE, dN)) % 360.0
    err_pa = np.degrees(_SEP_ERR / sep)
    np.savetxt(
        path,
        np.column_stack([t, sep, np.full(n, _SEP_ERR), pa, err_pa]),
    )
    return t, dE, dN, mid


def _rel_config(tmp, otype="constant_velocity", rv=False):
    t, dE, dN, mid = _write_rel(tmp / "ab.rel")
    config = {
        "name": "relcv",
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [
            {"name": "AB", "type": otype, "primary": ["A"], "companion": ["B"]}
        ],
        "astrometryinstrument": [
            {
                "name": "R",
                "file": str(tmp / "ab.rel"),
                "mode": "rel",
                "orbit": "AB",
            }
        ],
    }
    up = {}
    for s in ("A", "B"):
        up[f"star.{s}.mass"] = {"initval": 1.0, "sigma": 0.05}
        up[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.1}
        up[f"star.{s}.teff"] = {"initval": 5800, "sigma": 100}
        up[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.1}
        up[f"star.{s}.distance"] = {"initval": 50.0}
    if rv:
        rng = np.random.default_rng(5)
        t_rv = np.sort(rng.uniform(2458000.0, 2458600.0, 20))
        np.savetxt(
            tmp / "a.rv",
            np.column_stack(
                [t_rv, 20.0 + rng.normal(0, 3.0, 20), np.full(20, 3.0)]
            ),
        )
        config["rvinstrument"] = [{"name": "RV", "file": str(tmp / "a.rv")}]
    return config, up, (t, dE, dN, mid)


def _max_likelihood(model, start):
    from scipy.optimize import minimize

    logl = model.logp(vars=model.observed_RVs, sum=True)
    values = list(model.value_vars)
    f = pytensor.function(
        values,
        [logl] + pt.grad(logl, values, disconnected_inputs="ignore"),
        on_unused_input="ignore",
    )
    shapes = [np.shape(start[v.name]) for v in values]
    sizes = [int(np.prod(sh)) for sh in shapes]

    def unpack(x):
        out, k = [], 0
        for sh, n in zip(shapes, sizes):
            out.append(np.asarray(x[k : k + n]).reshape(sh))
            k += n
        return out

    def neg(x):
        lg, *g = f(*unpack(x))
        return -float(lg), -np.concatenate([np.ravel(gi) for gi in g])

    x0 = np.concatenate([np.ravel(start[v.name]) for v in values])
    res = minimize(
        neg,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10},
    )
    return {v.name: a for v, a in zip(values, unpack(res.x))}


@pytest.fixture(scope="module")
def rel_fit(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rel")
    config, up, data = _rel_config(tmp)
    system = System(config, user_params=up)
    system.prepare()
    model = system.build_model()
    start = system.get_raw_starts(model)[0][0]
    best = _max_likelihood(model, start)
    return system, model, data, start, best


def test_a_rel_dataset_reads_the_relative_straight_line(rel_fit):
    """
    Given sep/PA epochs of a companion on a straight line relative to its
      primary (0.4 mas isotropic errors), the orbit type: constant_velocity
      and no absolute astrometry,
    When the system is built and its likelihood maximized,
    Then the epoch is the midpoint of the rel data, the four rel_*
      coefficients come back within 4 sigma of the injected line and within
      0.1 sigma of an independent numpy least-squares line through
      (sep sin PA, sep cos PA), and the start (the data seed) is already
      within 5 sigma.
    """
    # Arrange
    system, model, (t, dE, dN, mid), start, best = rel_fit
    orbit = system.orbit
    names = ["rel_dE", "rel_dN", "rel_vE", "rel_vN"]
    nodes = [getattr(orbit, n).value for n in names]

    # Act
    def vals(point):
        nodes_v = model.replace_rvs_by_values(nodes)
        fn = pytensor.function(
            model.value_vars, nodes_v, on_unused_input="ignore"
        )
        return np.array(
            [
                float(np.ravel(x)[0])
                for x in fn(*[point[v.name] for v in model.value_vars])
            ]
        )

    fit, seed = vals(best), vals(start)

    # Assert
    assert orbit.taylor_epoch[0] == pytest.approx(mid, abs=1e-6)
    dt = (t - mid) / DAYS_PER_YEAR
    A = np.column_stack([np.ones_like(dt), dt]) / _SEP_ERR
    cov = np.linalg.inv(A.T @ A)
    lsq_E = np.linalg.lstsq(A, dE / _SEP_ERR, rcond=None)[0]
    lsq_N = np.linalg.lstsq(A, dN / _SEP_ERR, rcond=None)[0]
    lsq = np.array([lsq_E[0], lsq_N[0], lsq_E[1], lsq_N[1]])
    sig = np.sqrt(np.diag(cov))[[0, 0, 1, 1]]
    truth = np.array([_REL["dE"], _REL["dN"], _REL["vE"], _REL["vN"]])
    assert np.all(np.abs(fit - truth) < 4 * sig), (fit, truth, sig)
    assert np.all(np.abs(fit - lsq) < 0.1 * sig), (fit, lsq, sig)
    assert np.all(np.abs(seed - truth) < 5 * sig), (seed, truth, sig)


def test_rel_and_rvs_on_one_straight_line(tmp_path):
    """
    Given the rel dataset plus RVs of the primary on the same
      constant_velocity orbit,
    When the system is built,
    Then it is legal: the RVs see nothing (v0_z is gamma -- no gammadot is
      declared and the orbit's RV column is identically zero), the rel data
      read rel_*, and the logp is finite.
    """
    # Arrange
    config, up, _ = _rel_config(tmp_path, rv=True)

    # Act
    system = System(config, user_params=up)
    system.prepare()
    model = system.build_model()
    vz = system.orbit.taylor_radial_velocity(
        np.linspace(2458000.0, 2458600.0, 7), 0
    )
    (vz,) = _at_start(system, model, [vz])

    # Assert
    assert "gammadot" not in system.orbit.manifest
    assert {"rel_dE", "rel_vN"} <= set(system.orbit.manifest)
    np.testing.assert_array_equal(vz, 0.0)
    assert np.isfinite(
        model.compile_logp()(system.get_raw_starts(model)[0][0])
    )


@pytest.mark.parametrize("otype", ["uniform_rotation"])
def test_a_rel_dataset_of_another_taylor_type_is_refused(tmp_path, otype):
    """
    Given a rel dataset naming a uniform_rotation orbit (the lens's polar
      frame),
    When the system is prepared,
    Then it raises naming the Cartesian types a rel dataset can read
      (constant_acceleration/jerk are read since review 8.8.14's rel
      follow-on, tests/test_taylor_rel.py).
    """
    # Arrange
    config, up, _ = _rel_config(tmp_path, otype=otype)

    # Act / Assert
    with pytest.raises(NotImplementedError, match="constant_velocity"):
        System(config, user_params=up).prepare()
