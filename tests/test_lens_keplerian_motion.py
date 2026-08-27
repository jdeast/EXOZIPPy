"""
Lens orbital motion, keplerian mode (conventions.md C24; review 8.6.8
5a/5b).

What is pinned here, and why each pin is the one that matters:

  - THE SIGN TEST (the 0a saga; review 3.6.3/3.6.4): alpha(t) must run
    OPPOSITE to the binary axis's own position angle,
    alpha(t) = phi_pi - PA_axis(t), with PA_axis taken from the SAME
    get_sky_position whose absolute orientation is pinned first-principles
    by tests/test_astrometry.py and tests/test_skyframe.py.  A flipped
    sign lowers no chi2 -- it silently reports the reflected orbit
    (Omega -> -Omega, i -> 180 - i; Skowron 5.2) -- so it is pinned
    against a synthetic orbit of KNOWN geometry, never against a fit, and
    NEVER against MulensModel's keplerian branch (which contradicts its
    own linear mode by a sign; conventions.md section 6).
  - sign(cos i) is measurable: flipping cos i flips the alpha rotation
    sense -- the one thing lens orbital motion measures that nothing else
    here can (8.6.8).
  - Local linearity: over a short window the keplerian series matches the
    LINEAR series built from its own finite-difference rates at t0_par.
    The linear mode is pinned against MulensModel's definitional linear
    branch, so this closes the chain keplerian <-> linear <-> reference.
  - No new free parameters (5b): keplerian mode declares NO log_s /
    xalpha / yalpha / ds_dt / dalpha_dt; s and alpha are orbit-derived
    Deterministics whose t0_par values equal the series there.
  - The orbit side: a lens-keplerian-driven orbit declares bigomega (the
    lens rotation sense measures the node) and is NOT node-degenerate.
"""

import numpy as np
import pytensor
import pytest

pytestmark = pytest.mark.slow

from exozippy.constants import DAYS_PER_YEAR
from exozippy.system import System

_T0_PAR = 2458554.89


def _write_lc(path, n=80):
    rng = np.random.default_rng(11)
    t = np.linspace(_T0_PAR - 30.0, _T0_PAR + 30.0, n)
    mag = 15.0 - rng.uniform(0, 0.001, n)
    err = np.full(n, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))
    return str(path)


def _kep_system(tmp_path, cosi=0.5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    lc = _write_lc(tmp_path / "lc.dat")
    config = {
        "star": [{"name": "L1"}, {"name": "L2"}, {"name": "Source"}],
        "orbit": [
            {
                "name": "L",
                "primary": ["star.0"],
                "companion": ["star.1"],
            }
        ],
        "lens": [
            {
                "name": "EV",
                "lenses": ["star.0", "star.1"],
                "sources": ["star.2"],
                "finite_source": True,
                "orbital_motion": "keplerian",
                "orbit": "L",
                "t0_par": _T0_PAR,
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    params = {
        "lens.Source.t_0": {"initval": _T0_PAR},
        "lens.Source.u_0": {"initval": 0.08},
        "lens.Source.t_E": {"initval": 60.0},
        "lens.Source.rho": {"initval": 0.002},
        "lens.Source.pi_E_N": {"initval": -0.05},
        "lens.Source.pi_E_E": {"initval": 0.12},
        "orbit.L.period": {"initval": 300.0},
        "orbit.L.tc": {"initval": _T0_PAR - 40.0},
        "orbit.L.secosw": {"initval": 0.3},
        "orbit.L.sesinw": {"initval": -0.2},
        "orbit.L.cosi": {"initval": cosi},
        "orbit.L.bigomega": {"initval": 40.0},
        "star.L1.mass": {"initval": 0.8},
        "star.L2.mass": {"initval": 0.3},
        "star.L1.distance": {"initval": 1500.0},
        "star.L2.distance": {"initval": 1500.0},
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
def kep_system(tmp_path_factory):
    return _kep_system(tmp_path_factory.mktemp("lens_kep"))


def _eval_at_start(system, model, nodes):
    """Evaluate graph nodes at the start (raw zeros == the start point)."""
    with model:
        f = pytensor.function(model.free_RVs, nodes, on_unused_input="ignore")
        ip = model.initial_point()
        zeros = [
            np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs
        ]
        return f(*zeros)


def _series_and_pa(system, model, times):
    import pytensor.tensor as pt

    s_t, alpha_t_deg = system.lens._companion_geometry_series(times, system)
    omap = np.array([system.lens.kep_orbit_idx])
    dE, dN = system.orbit.get_sky_position(
        pt.as_tensor_variable(times),
        pt.ones(1),
        omap,
        relative=True,
    )
    pa = pt.arctan2(dE[:, 0], dN[:, 0])
    phi_pi = pt.arctan2(
        system.lens.pi_E_E.value[0], system.lens.pi_E_N.value[0]
    )
    out = _eval_at_start(system, model, [s_t, alpha_t_deg, pa, phi_pi])
    return [np.atleast_1d(np.asarray(x, dtype=float)) for x in out]


def _wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def test_alpha_runs_opposite_the_axis_pa(kep_system):
    """
    Given: the keplerian series and the axis PA from get_sky_position
      (whose absolute orientation is pinned first-principles elsewhere),
    When: both are evaluated over an orbital period,
    Then: alpha(t) == phi_pi - PA_axis(t) at every epoch -- C24's minus,
      the sign that chi2 cannot see.
    """
    system, model = kep_system
    times = np.linspace(_T0_PAR - 150.0, _T0_PAR + 150.0, 401)
    s_t, alpha_deg, pa, phi_pi = _series_and_pa(system, model, times)
    expected = np.degrees(phi_pi[0]) - np.degrees(pa)
    np.testing.assert_allclose(_wrap_deg(alpha_deg - expected), 0.0, atol=1e-8)
    # and s(t) is a genuinely moving separation over this window
    assert s_t.max() - s_t.min() > 1e-3


def test_cosi_sign_flips_the_rotation_sense(tmp_path_factory):
    """
    Given: two systems identical except cos i -> -cos i,
    When: the alpha series rate at t0_par is measured,
    Then: the rotation sense flips -- sign(cos i) is measurable, which is
      the entire point of the keplerian mode (8.6.8), and why the example's
      i180 hand-holding can be dropped.
    """
    rates = {}
    for label, cosi in (("pro", 0.5), ("retro", -0.5)):
        system, model = _kep_system(
            tmp_path_factory.mktemp(f"lens_kep_{label}"), cosi=cosi
        )
        dt = 2.0
        times = np.array([_T0_PAR - dt, _T0_PAR + dt])
        _, alpha_deg, _, _ = _series_and_pa(system, model, times)
        rates[label] = _wrap_deg(alpha_deg[1] - alpha_deg[0]) / (2 * dt)
    assert rates["pro"] * rates["retro"] < 0.0, rates


def test_keplerian_series_is_locally_the_linear_series(kep_system):
    """
    Given: the keplerian series' own finite-difference rates at t0_par,
    When: a linear series is built from them over +/-3 days,
    Then: the two agree to second order -- the chain to the linear mode
      (itself pinned against MulensModel's definitional branch) closes.
    """
    system, model = kep_system
    eps = 0.25
    window = np.linspace(_T0_PAR - 1.0, _T0_PAR + 1.0, 17)
    times = np.concatenate([[_T0_PAR - eps, _T0_PAR, _T0_PAR + eps], window])
    s_t, alpha_deg, _, _ = _series_and_pa(system, model, times)
    ds_dt = (s_t[2] - s_t[0]) / (2 * eps)
    dalpha_dt = _wrap_deg(alpha_deg[2] - alpha_deg[0]) / (2 * eps)
    s_lin = s_t[1] + ds_dt * (window - _T0_PAR)
    a_lin = alpha_deg[1] + dalpha_dt * (window - _T0_PAR)
    # Tolerances are the P = 300 d orbit's own quadratic term over the
    # window (~0.5 * rate * (2 pi/P) * dt^2), not floating point.
    np.testing.assert_allclose(s_t[3:], s_lin, atol=5e-5)
    np.testing.assert_allclose(
        _wrap_deg(alpha_deg[3:] - a_lin), 0.0, atol=2e-2
    )


def test_no_new_free_parameters_and_reported_geometry(kep_system):
    """
    Given: the keplerian-mode system,
    Then: no sampled geometry coordinates exist (5b), s/alpha are derived,
      their values equal the series at t0_par, logp is finite, the driven
      orbit declares bigomega, and it is not node-degenerate.
    """
    system, model = kep_system
    lens = system.lens
    for name in ("log_s", "xalpha", "yalpha", "ds_dt", "dalpha_dt", "beta"):
        assert name not in lens.manifest, f"{name} should not exist"
    for name in ("s", "alpha"):
        assert lens.manifest[name]["expr_key"] == "from_orbit"
    free_names = [v.name for v in model.free_RVs]
    assert "lens.log_s_raw" not in free_names
    assert "lens.xalpha_raw" not in free_names
    assert "orbit.xbigomega_raw" in free_names, (
        "a lens-keplerian-driven orbit must sample the node direction"
    )
    assert not bool(np.any(system.orbit.node_degenerate)), (
        "the lens rotation sense identifies the node"
    )

    times = np.array([float(lens.t0_par[0])])
    s_t, alpha_deg, _, _ = _series_and_pa(system, model, times)
    s0, alpha0 = _eval_at_start(
        system, model, [lens.s.value, lens.alpha.value]
    )
    np.testing.assert_allclose(float(np.atleast_1d(s0)[0]), s_t[0], rtol=1e-10)
    np.testing.assert_allclose(
        _wrap_deg(np.degrees(float(np.atleast_1d(alpha0)[0])) - alpha_deg[0]),
        0.0,
        atol=1e-8,
    )

    starts, _ = system.get_raw_starts(model)
    logp = float(model.compile_logp(jacobian=False)(starts[0]))
    assert np.isfinite(logp)


def test_keplerian_config_validation():
    """
    Given: keplerian without an orbit reference, or an unknown name,
    Then: construction raises naming the problem.
    """
    from exozippy.components.mulensing.lens import Lens
    from exozippy.config import ConfigManager

    base = {
        "name": "EV",
        "lenses": ["star.0", "star.1"],
        "sources": ["star.2"],
        "orbital_motion": "keplerian",
    }
    with pytest.raises(ValueError, match="orbit"):
        Lens([dict(base)], ConfigManager({}))

    cm = ConfigManager({})
    cm.system_config = {"orbit": [{"name": "L"}]}
    with pytest.raises(ValueError, match="unknown orbit"):
        Lens([dict(base, orbit="nope")], cm)
