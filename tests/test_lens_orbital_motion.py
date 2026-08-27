"""
Lens orbital motion, linear mode (conventions.md C24; review 8.6.8).

What is pinned here, and why each pin is the one that matters:

  - The per-epoch vbm_direct construction against MulensModel's LINEAR
    lens-motion branch (the reference implementation of the Skowron+2011
    parameterization; its linear branch is definitional in (ds_dt,
    dalpha_dt), unlike its keplerian branch -- conventions.md section 6).
    This is the SIGN test for the linear mode: a flipped dalpha_dt sense
    would disagree immediately, everywhere off t0_par.
  - Zero rates reproduce the static Op (to VBM's measured ~1e-14
    call-history jitter): the per-epoch layout merge (8.6.8 5c) must not
    move any existing binary fit.
  - The A12/A16 mirror: with no parallax, (u_0, alpha, dalpha_dt) ->
    -(u_0, alpha, dalpha_dt) is an exact symmetry of the light curve --
    the extension of C23's static mirror to orbital motion, and the reason
    a sign for dalpha_dt is only meaningful jointly with sign(u_0).
  - The t0_par anchor: at t = t0_par the moving geometry IS (s_0, alpha_0),
    so the magnification there matches the static model exactly.
  - System level: `orbital_motion: linear` declares ds_dt/dalpha_dt/beta,
    beta evaluates to the A19 formula, and the model builds with a finite
    logp.  Also the mulensmodel backend passes the rates through natively.

The keplerian mode is validated separately against a first-principles
synthetic orbit (NEVER against MulensModel's keplerian branch, which
contradicts its own linear mode by a sign -- conventions.md section 6).
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

pytestmark = pytest.mark.slow

from exozippy.components.mulensing.op import (
    BinaryLensMagOp,
    VBMDirectMagOp,
)
from exozippy.constants import DAYS_PER_YEAR

_COORDS = "268.0d -29.0d"
_T0_PAR = 2458554.89
# A stellar-binary geometry (OGLE-2009-BLG-020-like scale, not planetary:
# orbital motion is measured through the caustic ORIENTATION, so q ~ 0.27
# with a close topology gives the rates real leverage).
_MAP = dict(
    t_0=2458554.89,
    u_0=0.062,
    t_E=76.0,
    pi_E_N=-0.025,
    pi_E_E=0.149,
    rho=0.0015,
    s=0.43,
    q=0.27,
    alpha=189.1,
    ds_dt=0.043,  # gamma_par ~ 0.10/yr at s ~ 0.43
    dalpha_dt=-2.3 * 180.0 / np.pi,  # deg/yr; gamma_perp = +2.3 rad/yr
    u1=0.5,
)
_ORDER_STATIC = [
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
_ORDER_MM_OM = [
    "t_0",
    "u_0",
    "t_E",
    "pi_E_N",
    "pi_E_E",
    "rho",
    "s",
    "q",
    "alpha",
    "ds_dt",
    "dalpha_dt",
    "u1",
]


def _times_and_obs(n=400, span=150.0):
    """Same smooth synthetic deviations as test_vbm_direct_vs_mulensmodel."""
    times = np.linspace(_T0_PAR - span, _T0_PAR + span, n)
    phase = 2.0 * np.pi * (times - _T0_PAR) / DAYS_PER_YEAR
    dev = np.column_stack(
        [
            0.5 * (1.0 - np.cos(phase)),
            0.5 * (np.sin(phase) - phase),
            0.2 * (1.0 - np.cos(phase)),
        ]
    )
    offset = np.array([0.009, 0.004, 0.002])
    return times, dev + offset[None, :]


def _series(times, p):
    """The C24 linear series, alpha in degrees -- the definition itself."""
    dt_yr = (times - _T0_PAR) / DAYS_PER_YEAR
    return p["s"] + p["ds_dt"] * dt_yr, p["alpha"] + p["dalpha_dt"] * dt_yr


def _compile_om(op):
    p = pt.dvector("p")
    t = pt.dvector("t")
    o = pt.dmatrix("o")
    s_t = pt.dvector("s_t")
    a_t = pt.dvector("a_t")
    return pytensor.function([p, t, o, s_t, a_t], op(p, t, o, s_t, a_t))


def _compile(op):
    p = pt.dvector("p")
    t = pt.dvector("t")
    o = pt.dmatrix("o")
    return pytensor.function([p, t, o], op(p, t, o))


def _draw(rng, scale=1.0):
    p = dict(_MAP)
    p["t_0"] += rng.normal(0, 0.5) * scale
    p["u_0"] *= 1 + rng.normal(0, 0.05) * scale
    p["t_E"] *= 1 + rng.normal(0, 0.05) * scale
    p["pi_E_N"] += rng.normal(0, 0.02) * scale
    p["pi_E_E"] += rng.normal(0, 0.02) * scale
    p["rho"] *= 1 + rng.normal(0, 0.1) * scale
    p["s"] *= 1 + rng.normal(0, 0.02) * scale
    p["q"] *= 1 + rng.normal(0, 0.1) * scale
    p["alpha"] += rng.normal(0, 2.0) * scale
    p["ds_dt"] += rng.normal(0, 0.02) * scale
    p["dalpha_dt"] += rng.normal(0, 10.0) * scale
    return p


def test_vbm_per_epoch_matches_mulensmodel_linear_motion():
    """
    Given: random stellar-binary draws with nonzero (ds_dt, dalpha_dt) and
      parallax deviations,
    When: the per-epoch vbm_direct path and MulensModel's native LINEAR
      lens-motion branch evaluate the same models,
    Then: magnifications agree per-point to rtol 1e-8.

    This is the linear-mode SIGN test: MulensModel's linear branch is
    definitional (alpha(t) = alpha_0 + dalpha_dt*(t - t_0_kep)), so a
    flipped rotation sense in the per-epoch construction would disagree
    everywhere off t0_par, not by a tolerance.
    """
    times, obs = _times_and_obs()
    vbm_fn = _compile_om(
        VBMDirectMagOp(
            coords=_COORDS,
            n_companions=1,
            use_rho=True,
            bandpass="I",
            orbital_motion=True,
        )
    )
    mm_fn = _compile(
        BinaryLensMagOp(
            coords=_COORDS,
            mag_method="VBM",
            use_rho=True,
            bandpass="I",
            orbital_motion=True,
            t_0_kep=_T0_PAR,
        )
    )
    rng = np.random.default_rng(20260827)
    for k in range(6):
        p = _draw(rng)
        s_t, a_t = _series(times, p)
        a_vbm = vbm_fn(
            np.array([p[k_] for k_ in _ORDER_STATIC]), times, obs, s_t, a_t
        )
        a_mm = mm_fn(np.array([p[k_] for k_ in _ORDER_MM_OM]), times, obs)
        assert np.all(np.isfinite(a_vbm)), f"draw {k}: non-finite VBM"
        assert np.all(np.isfinite(a_mm)), f"draw {k}: non-finite MM"
        np.testing.assert_allclose(
            a_vbm,
            a_mm,
            rtol=1e-8,
            err_msg=f"draw {k}: per-epoch vbm_direct != MulensModel linear",
        )


def test_zero_rates_reproduce_the_static_op_bitwise():
    """
    Given: ds_dt = dalpha_dt = 0,
    When: the orbital-motion Op and the static Op evaluate the same draw,
    Then: the curves agree to rtol 1e-10 -- the per-epoch layout merge must
      not move any existing fit.  (Not bit-for-bit: VBMicrolensing's
      BinaryMag2 is call-history dependent at the ~1e-14 level -- two
      CONSECUTIVE IDENTICAL calls on one instance differ in the last bits,
      measured 2026-08-27 -- so bitwise equality across Op instances is
      unattainable near a caustic and 1e-10 is far below any physics while
      far above that jitter.)
    """
    times, obs = _times_and_obs()
    p = dict(_MAP)
    p["ds_dt"] = 0.0
    p["dalpha_dt"] = 0.0
    s_t, a_t = _series(times, p)
    pv = np.array([p[k] for k in _ORDER_STATIC])

    om_fn = _compile_om(
        VBMDirectMagOp(
            coords=_COORDS,
            n_companions=1,
            use_rho=True,
            bandpass="I",
            orbital_motion=True,
        )
    )
    static_fn = _compile(
        VBMDirectMagOp(
            coords=_COORDS,
            n_companions=1,
            use_rho=True,
            bandpass="I",
        )
    )
    np.testing.assert_allclose(
        om_fn(pv, times, obs, s_t, a_t),
        static_fn(pv, times, obs),
        rtol=1e-10,
    )


def test_orbital_motion_mirror_is_exact_without_parallax():
    """
    Given: no parallax (zero deviations, zero pi_E),
    When: (u_0, alpha, dalpha_dt) -> -(u_0, alpha, dalpha_dt) with ds_dt
      unchanged,
    Then: the light curve is identical to floating-point noise -- C23's
      exact static mirror (Skowron A12) extended to orbital motion (A16
      with the parallax terms removed).
    """
    times, _ = _times_and_obs()
    obs = np.zeros((len(times), 3))
    fn = _compile_om(
        VBMDirectMagOp(
            coords=_COORDS,
            n_companions=1,
            use_rho=True,
            bandpass="I",
            orbital_motion=True,
        )
    )
    p = dict(_MAP)
    p["pi_E_N"] = 0.0
    p["pi_E_E"] = 0.0
    q = dict(p)
    q["u_0"] = -p["u_0"]
    q["alpha"] = -p["alpha"]
    q["dalpha_dt"] = -p["dalpha_dt"]

    s_t, a_t = _series(times, p)
    s_tm, a_tm = _series(times, q)
    a_plus = fn(np.array([p[k] for k in _ORDER_STATIC]), times, obs, s_t, a_t)
    a_minus = fn(
        np.array([q[k] for k in _ORDER_STATIC]), times, obs, s_tm, a_tm
    )
    np.testing.assert_allclose(a_plus, a_minus, rtol=1e-9)


def test_geometry_at_t0_par_is_the_static_geometry():
    """
    Given: nonzero rates,
    When: the magnification is evaluated AT t0_par,
    Then: it equals the static model's value there -- s_0/alpha_0 are the
      geometry at the anchor (5d), not averages.
    """
    times = np.array([_T0_PAR])
    obs = np.zeros((1, 3))
    p = dict(_MAP)
    s_t, a_t = _series(times, p)
    np.testing.assert_array_equal(s_t, np.array([p["s"]]))
    np.testing.assert_array_equal(a_t, np.array([p["alpha"]]))

    om_fn = _compile_om(
        VBMDirectMagOp(
            coords=_COORDS,
            n_companions=1,
            use_rho=True,
            bandpass="I",
            orbital_motion=True,
        )
    )
    static_fn = _compile(
        VBMDirectMagOp(
            coords=_COORDS, n_companions=1, use_rho=True, bandpass="I"
        )
    )
    pv = np.array([p[k] for k in _ORDER_STATIC])
    # rtol, not bitwise: VBM's BinaryMag2 carries ~1e-14 call-history
    # jitter (see test_zero_rates_reproduce_the_static_op_bitwise).
    np.testing.assert_allclose(
        om_fn(pv, times, obs, s_t, a_t),
        static_fn(pv, times, obs),
        rtol=1e-10,
    )


def test_nan_series_yields_nan_curve():
    """
    Given: a NaN in the per-epoch geometry (a junk proposal),
    When: the Op evaluates,
    Then: every magnification is NaN (logp = -inf; proposal rejected), with
      the warn-once report -- never an exception.
    """
    times, obs = _times_and_obs(n=50)
    fn = _compile_om(
        VBMDirectMagOp(
            coords=_COORDS,
            n_companions=1,
            use_rho=True,
            bandpass="I",
            orbital_motion=True,
        )
    )
    p = dict(_MAP)
    s_t, a_t = _series(times, p)
    s_t = np.asarray(s_t, dtype=float).copy()
    s_t[3] = np.nan
    with pytest.warns(RuntimeWarning, match="orbital"):
        a = fn(np.array([p[k] for k in _ORDER_STATIC]), times, obs, s_t, a_t)
    assert np.all(np.isnan(a))


# ---------------------------------------------------------------------------
# System level
# ---------------------------------------------------------------------------


def _write_binary_lc(path, n=120):
    """Synthetic light curve over the event window (values need not fit --
    these tests probe the model graph, not a fit)."""
    rng = np.random.default_rng(7)
    t = np.linspace(_T0_PAR - 40.0, _T0_PAR + 40.0, n)
    mag = 15.0 - rng.uniform(0, 0.001, n)
    err = np.full(n, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))
    return str(path)


def _om_system(tmp_path, orbital_motion="linear", backend="vbm_direct"):
    from exozippy.system import System

    tmp_path.mkdir(parents=True, exist_ok=True)
    lc = _write_binary_lc(tmp_path / "lc.dat")

    config = {
        "star": [{"name": "L1"}, {"name": "L2"}, {"name": "Source"}],
        "lens": [
            {
                "name": "EV",
                "lenses": ["star.0", "star.1"],
                "sources": ["star.2"],
                "finite_source": True,
                "orbital_motion": orbital_motion,
                "backend": backend,
                "t0_par": _T0_PAR,
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    params = {
        "lens.Source.t_0": {"initval": _MAP["t_0"]},
        "lens.Source.u_0": {"initval": _MAP["u_0"]},
        "lens.Source.t_E": {"initval": _MAP["t_E"]},
        "lens.Source.rho": {"initval": _MAP["rho"]},
        "lens.EV.s": {"initval": _MAP["s"]},
        "lens.EV.alpha": {"initval": _MAP["alpha"]},
        "lens.EV.q": {"initval": _MAP["q"]},
        "lens.Source.pi_E_N": {"initval": _MAP["pi_E_N"]},
        "lens.Source.pi_E_E": {"initval": _MAP["pi_E_E"]},
        "lens.EV.ds_dt": {"initval": _MAP["ds_dt"]},
        "lens.EV.dalpha_dt": {"initval": _MAP["dalpha_dt"]},
        "star.L1.mass": {"initval": 0.84},
        "star.L1.distance": {"initval": 1100.0},
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
    model = system.build_model()
    return system, model


@pytest.fixture(scope="module")
def linear_om_system(tmp_path_factory):
    return _om_system(tmp_path_factory.mktemp("lens_om"))


def test_linear_mode_declares_rates_and_beta(linear_om_system):
    """
    Given: a lens block with orbital_motion: linear,
    When: parameters register,
    Then: ds_dt, dalpha_dt and beta are in the manifest, and the user's
      deg/yr dalpha_dt seed lands as rad/yr internally.
    """
    system, _ = linear_om_system
    lens = system.lens
    for name in ("ds_dt", "dalpha_dt", "beta"):
        assert name in lens.manifest, f"{name} missing from manifest"
    # user unit deg/yr -> internal rad/yr
    got = float(np.atleast_1d(lens.dalpha_dt.initval)[0])
    np.testing.assert_allclose(got, np.radians(_MAP["dalpha_dt"]), rtol=1e-12)


def test_beta_is_the_a19_ratio_and_logp_is_finite(linear_om_system):
    """
    Given: the linear-mode system,
    When: the model builds,
    Then: logp at the start is finite, and lens.beta evaluates to the
      Skowron A19 sky-plane ratio computed by hand from the same start
      values.
    """
    system, model = linear_om_system
    starts, _ = system.get_raw_starts(model)
    logp = float(model.compile_logp(jacobian=False)(starts[0]))
    assert np.isfinite(logp)

    from exozippy.constants import KAPPA

    # Evaluate the graph at the start (raw coordinates are whitened around
    # it, so raw zeros ARE the start -- the test_log_s probe pattern).
    with model:
        f = pytensor.function(
            model.free_RVs,
            [
                system.lens.beta.value,
                system.lens.pi_rel.value,
                system.lens.theta_E.value,
                system.lens.s.value,
                system.lens.ds_dt.value,
                system.lens.dalpha_dt.value,
                system.star.distance.value,
            ],
            on_unused_input="ignore",
        )
        ip = model.initial_point()
        zeros = [
            np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs
        ]
        beta_v, pi_rel, theta_E, s0, ds_dt, dalpha_dt, dist = f(*zeros)

    pi_rel = float(np.atleast_1d(pi_rel)[0])
    theta_E = float(np.atleast_1d(theta_E)[0])
    s0 = float(np.atleast_1d(s0)[0])
    ds_dt = float(np.atleast_1d(ds_dt)[0])
    dalpha_dt = float(np.atleast_1d(dalpha_dt)[0])
    d_s = float(np.atleast_1d(dist)[system.lens.source_map[0]])
    pi_E = pi_rel / theta_E
    gamma_sq = (ds_dt / s0) ** 2 + dalpha_dt**2
    expected = (
        KAPPA
        * pi_E
        * gamma_sq
        * s0**3
        / (8 * np.pi**2 * theta_E * (pi_E + (1000.0 / d_s) / theta_E) ** 3)
    )
    np.testing.assert_allclose(
        float(np.atleast_1d(beta_v)[0]), expected, rtol=1e-6
    )


def test_mulensmodel_backend_gets_native_rates(tmp_path):
    """
    Given: orbital_motion: linear with backend: mulensmodel,
    When: the system builds,
    Then: logp is finite (the native ds_dt/dalpha_dt/t_0_kep pass-through
      path constructs a valid MulensModel).
    """
    system, model = _om_system(tmp_path, backend="mulensmodel")
    starts, _ = system.get_raw_starts(model)
    logp = float(model.compile_logp(jacobian=False)(starts[0]))
    assert np.isfinite(logp)


def test_orbital_motion_requires_a_companion():
    """
    Given: a single-lens block asking for orbital_motion,
    When: the Lens component is constructed,
    Then: it raises naming the problem (no s or alpha to move).
    """
    from exozippy.components.mulensing.lens import Lens
    from exozippy.config import ConfigManager

    with pytest.raises(ValueError, match="companion"):
        Lens(
            [
                {
                    "name": "EV",
                    "lenses": ["star.0"],
                    "sources": ["star.1"],
                    "orbital_motion": "linear",
                }
            ],
            ConfigManager({}),
        )


def test_mulensmodel_backend_applies_finite_source_at_stellar_q():
    """
    Regression for the silent 2-element magnification-methods list
    (mulensing.md "Lens orbital motion", found 2026-08-27):
    ``set_magnification_methods([0.0, "VBM"])`` applies VBM to NOTHING, so
    the MulensModel backend computed POINT-SOURCE magnification for every
    finite-source epoch -- ~1% low exactly at a stellar-binary caustic
    approach, invisible at the planetary geometries the other parity tests
    draw.

    Given: the stellar-binary geometry, static, finite source,
    When: the MulensModel backend and vbm_direct evaluate it,
    Then: they agree to rtol 1e-7 AND both differ from the point-source
      curve by far more than that somewhere (the teeth: were the backend
      still point-source, the first assertion would fail by ~1e-2).
    """
    times, obs = _times_and_obs()
    p = dict(_MAP)
    pv = np.array([p[k] for k in _ORDER_STATIC])

    mm_fn = _compile(
        BinaryLensMagOp(
            coords=_COORDS, mag_method="VBM", use_rho=True, bandpass="I"
        )
    )
    vbm_fn = _compile(
        VBMDirectMagOp(
            coords=_COORDS, n_companions=1, use_rho=True, bandpass="I"
        )
    )
    a_mm = mm_fn(pv, times, obs)
    a_vbm = vbm_fn(pv, times, obs)
    np.testing.assert_allclose(a_mm, a_vbm, rtol=1e-7)

    ps_fn = _compile(
        BinaryLensMagOp(
            coords=_COORDS, mag_method="point_source", use_rho=False
        )
    )
    pv_ps = np.array([p[k] for k in _ORDER_STATIC if k not in ("rho", "u1")])
    a_ps = ps_fn(pv_ps, times, obs)
    assert np.max(np.abs(a_mm - a_ps) / a_ps) > 1e-3, (
        "finite source should matter at this geometry; if it does not, "
        "this test has no teeth"
    )
