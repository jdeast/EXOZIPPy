"""
Tests for the astrometryinstrument component and Orbit.get_sky_position.

Conventions under test (EXOFASTv2):
  - omega is the argument of periastron of the PRIMARY's orbit (omega_*)
  - bigomega is the position angle of the ascending node, East of North,
    where the ascending node is the node at which the body recedes from
    the observer (consistent with get_radial_velocity)
  - relative=True models the companion (omega_* + 180 deg)

The reference implementation used here builds the sky position via explicit
3D rotations (orbital plane -> inclination about the node line -> node PA),
independent of the Thiele-Innes shortcut used in Orbit.get_sky_position.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit.orbit import Orbit
from exozippy.components.astrometryinstrument import AstrometryInstrument
from exozippy.config import ConfigManager
from exozippy.system import System

from conftest import _DummySystem, _DummyComponent

RAD2MAS = 180.0 / np.pi * 3600e3
RSUN_AU = 0.004650467260962157


# ---------------------------------------------------------------------------
# Independent reference implementation
# ---------------------------------------------------------------------------

def _kepler_E(M, ecc):
    E = np.mod(M, 2 * np.pi)
    for _ in range(100):
        E = E - (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))
    return E


def _true_anomaly(t, P, tp, ecc):
    M = 2 * np.pi * (t - tp) / P
    E = _kepler_E(M, ecc)
    cosf = (np.cos(E) - ecc) / (1 - ecc * np.cos(E))
    sinf = (np.sqrt(1 - ecc ** 2) * np.sin(E)) / (1 - ecc * np.cos(E))
    return np.arctan2(sinf, cosf)


def _sky_pos_reference(t, P, tp, ecc, w, bigom, inc, a_scale):
    """(dE, dN) of the body's own orbit via explicit rotations.

    Orbital frame: x toward periastron, z along the angular momentum.
    x' = r cos(w+f), y' = r sin(w+f) puts x' along the node line; inclining
    about the node line scales y' by cos(i) on the sky (z away from the
    observer, so the node at w+f=0 is ascending: dz/dt > 0 there).  The node
    line is then rotated to PA = bigomega (East of North).
    """
    f = _true_anomaly(t, P, tp, ecc)
    r = a_scale * (1 - ecc ** 2) / (1 + ecc * np.cos(f))
    x = r * np.cos(w + f)
    y_sky = r * np.sin(w + f) * np.cos(inc)
    dN = x * np.cos(bigom) - y_sky * np.sin(bigom)
    dE = x * np.sin(bigom) + y_sky * np.cos(bigom)
    return dE, dN


def _tp_from_tc(tc, P, ecc, w):
    """Time of periastron from time of conjunction (transit at f = pi/2 - w).

    M(tc) = n*(tc - tp), so tp = tc - M_c/n.  atan2 form of the half-angle
    identity tan(f_c/2) = (1 - sin w)/cos w keeps w near -pi/2 finite.
    """
    E_c = 2 * np.arctan2(np.sqrt(1 - ecc) * (1 - np.sin(w)),
                         np.sqrt(1 + ecc) * np.cos(w))
    M_c = E_c - ecc * np.sin(E_c)
    return tc - M_c * P / (2 * np.pi)


# ---------------------------------------------------------------------------
# Unit tests: Orbit.get_sky_position
# ---------------------------------------------------------------------------

_P_DAYS = 10.0
_TC = 2450000.0

np.random.seed(1234)
_CASES = [
    # (omega, ecc, bigomega, cosi)
    # note: exactly e=0 is untestable against tc-based references (tp is
    # convention-dependent there: calc_tp's atan2(0,0) = 0); zero measure
    # for the sampler since secosw/sesinw are sampled continuously.
    (0.3, 0.05, 1.2, 0.4),
    (0.0, 0.3, 0.0, 0.0),
    (np.pi / 2, 0.5, 3.0, -0.6),
    (-1.0, 0.2, 5.5, 0.9),
    (2.5, 0.7, 2.2, -0.95),
    tuple(np.random.uniform([-np.pi, 0.05, 0, -1], [np.pi, 0.8, 2 * np.pi, 1])),
]


@pytest.fixture(scope="module")
def compiled_sky_functions():
    """Compile get_sky_position / get_radial_velocity once for all cases.

    Given: a standalone Orbit whose manifest includes bigomega (astrometry
    active in the system topology).
    """
    user_params = {
        "orbit.0.logP": {"initval": np.log10(_P_DAYS)},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.secosw": {"initval": 0.0},
        "orbit.0.sesinw": {"initval": 0.0},
    }
    dummy_system = _DummySystem()
    dummy_system.config = {"astrometryinstrument": []}

    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        cm = ConfigManager(user_params)
        orbit_comp = Orbit([{"name": "test"}], cm)

        with pm.Model():
            orbit_comp.register_parameters(system=dummy_system)
            assert "xbigomega" in orbit_comp.manifest
            assert "ybigomega" in orbit_comp.manifest
            assert "bigomega" in orbit_comp.manifest
            for param_name in orbit_comp.manifest:
                orbit_comp.add_parameter(model=pm.modelcontext(None),
                                         param_name=param_name, system=dummy_system)

            t_var = pt.vector("t")
            a_var = pt.vector("a_scale")
            K_var = pt.vector("K_int")
            omap = np.array([0])

            dE_star, dN_star = orbit_comp.get_sky_position(t_var, a_var, omap)
            dE_rel, dN_rel = orbit_comp.get_sky_position(t_var, a_var, omap,
                                                         relative=True)
            rv_node = orbit_comp.get_radial_velocity(t_var, K_var, omap)

            free_inputs = [orbit_comp.logP.value, orbit_comp.tc.value,
                           orbit_comp.secosw.value, orbit_comp.sesinw.value,
                           orbit_comp.cosi.value, orbit_comp.xbigomega.value,
                           orbit_comp.ybigomega.value]
            sky_fn = pytensor.function(
                inputs=free_inputs + [t_var, a_var],
                outputs=[dE_star, dN_star, dE_rel, dN_rel],
                on_unused_input="ignore",
            )
            rv_fn = pytensor.function(
                inputs=free_inputs + [t_var, K_var],
                outputs=[rv_node],
                on_unused_input="ignore",
            )
    return sky_fn, rv_fn


def _free_vals(omega, ecc, bigom, cosi):
    return [np.array([np.log10(_P_DAYS)]), np.array([_TC]),
            np.array([np.sqrt(ecc) * np.cos(omega)]),
            np.array([np.sqrt(ecc) * np.sin(omega)]),
            np.array([cosi]),
            np.array([np.cos(bigom)]), np.array([np.sin(bigom)])]


@pytest.mark.parametrize("case", _CASES,
                         ids=[f"w={c[0]:.2f}_e={c[1]:.2f}_O={c[2]:.2f}_ci={c[3]:.2f}"
                              for c in _CASES])
def test_sky_position_matches_rotation_reference(case, compiled_sky_functions):
    """
    Given: random orbital elements
    When: get_sky_position is evaluated over an orbital period
    Then: it matches the independent rotation-matrix implementation for both
          the primary (omega_*) and the relative (omega_* + pi) orbit
    """
    omega, ecc, bigom, cosi = case
    sky_fn, _ = compiled_sky_functions
    t = np.linspace(_TC, _TC + _P_DAYS, 137)
    a_scale = np.array([3.7])  # mas

    dE_s, dN_s, dE_r, dN_r = sky_fn(*_free_vals(omega, ecc, bigom, cosi),
                                    t, a_scale)

    tp = _tp_from_tc(_TC, _P_DAYS, ecc, omega)
    inc = np.arccos(cosi)
    eE_s, eN_s = _sky_pos_reference(t, _P_DAYS, tp, ecc, omega, bigom, inc, 3.7)
    eE_r, eN_r = _sky_pos_reference(t, _P_DAYS, tp, ecc, omega + np.pi, bigom,
                                    inc, 3.7)

    np.testing.assert_allclose(dE_s[:, 0], eE_s, atol=1e-8)
    np.testing.assert_allclose(dN_s[:, 0], eN_s, atol=1e-8)
    np.testing.assert_allclose(dE_r[:, 0], eE_r, atol=1e-8)
    np.testing.assert_allclose(dN_r[:, 0], eN_r, atol=1e-8)


@pytest.mark.parametrize("case", _CASES[:4],
                         ids=[f"w={c[0]:.2f}_e={c[1]:.2f}_O={c[2]:.2f}_ci={c[3]:.2f}"
                              for c in _CASES[:4]])
def test_ascending_node_convention(case, compiled_sky_functions):
    """
    Given: the primary crossing its ascending node (omega_* + f = 0)
    When: the sky position and radial velocity are evaluated there
    Then: the primary sits at PA = bigomega and is receding (RV maximal > 0),
          i.e. the astrometric and RV conventions are mutually consistent
    """
    omega, ecc, bigom, cosi = case
    sky_fn, rv_fn = compiled_sky_functions

    # time of ascending-node crossing: f = -omega
    tp = _tp_from_tc(_TC, _P_DAYS, ecc, omega)
    f_node = -omega
    E_node = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(f_node / 2))
    M_node = E_node - ecc * np.sin(E_node)
    t_node = np.array([tp + M_node * _P_DAYS / (2 * np.pi)])

    vals = _free_vals(omega, ecc, bigom, cosi)
    dE_s, dN_s, _, _ = sky_fn(*vals, t_node, np.array([1.0]))
    (rv,) = rv_fn(*vals, t_node, np.array([1.0]))

    pa = np.arctan2(dE_s[0, 0], dN_s[0, 0])
    assert np.isclose(np.mod(pa - bigom, 2 * np.pi), 0.0, atol=1e-6) or \
        np.isclose(np.mod(pa - bigom, 2 * np.pi), 2 * np.pi, atol=1e-6)
    # RV at the node is the maximum of K*(cos(w+f) + e*cos(w)): strictly positive
    assert rv[0, 0] > 0.0


# ---------------------------------------------------------------------------
# Unit test: parallax factors
# ---------------------------------------------------------------------------

def test_parallax_factors_match_exact_geometry(tmp_path):
    """
    Given: an abs-mode instrument for a star with parallax plx
    When: the load-time parallax factors (P_E, P_N) are scaled by plx
    Then: they match the exact (non-linearized) apparent displacement of the
          star as seen from the moving observer
    """
    # Arrange
    ra0, dec0 = 217.42, -62.68   # deg (alpha Cen-ish: big parallax regime)
    plx = 100.0                  # mas
    t = np.linspace(2457000.0, 2457365.0, 25)

    data = np.column_stack([t, np.full_like(t, ra0), np.full_like(t, dec0),
                            np.ones_like(t), np.ones_like(t)])
    f = tmp_path / "abs.astrom"
    np.savetxt(f, data)

    user_params = {"star.0.ra": {"initval": ra0}, "star.0.dec": {"initval": dec0}}
    cm = ConfigManager(user_params)
    comp = AstrometryInstrument(
        [{"name": "T", "file": str(f), "mode": "abs",
          "observer_location": "earth"}], cm)
    system = _DummySystem()
    system.star = _DummyComponent(1)

    # Act
    comp.load_data(system)
    d = comp.datasets[0]

    # Exact geometry: apparent direction = unit(u * d_AU - b_obs)
    from exozippy.ephemeris import get_observer_position
    xyz = get_observer_position(t, "earth")
    ra_r, dec_r = np.radians(ra0), np.radians(dec0)
    u_hat = np.array([np.cos(dec_r) * np.cos(ra_r),
                      np.cos(dec_r) * np.sin(ra_r),
                      np.sin(dec_r)])
    E_hat = np.array([-np.sin(ra_r), np.cos(ra_r), 0.0])
    N_hat = np.array([-np.sin(dec_r) * np.cos(ra_r),
                      -np.sin(dec_r) * np.sin(ra_r),
                      np.cos(dec_r)])
    d_AU = (RAD2MAS / plx)  # 1/plx[rad] in AU
    vec = u_hat[None, :] * d_AU - xyz
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    exact_dE = (vec @ E_hat) * RAD2MAS
    exact_dN = (vec @ N_hat) * RAD2MAS

    # Assert: first-order parallax factors agree with the exact displacement
    # (second-order terms are ~plx^2/206265 ~ 0.05 mas at plx = 100 mas)
    np.testing.assert_allclose(plx * d["P_E"], exact_dE, atol=0.1)
    np.testing.assert_allclose(plx * d["P_N"], exact_dN, atol=0.1)
    # and the signal is a real annual ellipse, not numerically degenerate
    assert np.ptp(plx * d["P_E"]) > 50.0


# ---------------------------------------------------------------------------
# Unit test: orbit manifest gating
# ---------------------------------------------------------------------------

def test_bigomega_only_registered_with_astrometry():
    """
    Given: identical orbit configs with and without an astrometry component
    When: register_parameters runs
    Then: bigomega and the full cosi range appear only in the astrometry case
    """
    cm = ConfigManager({})

    plain = _DummySystem()
    plain.config = {"orbit": [{}]}
    orbit_plain = Orbit([{"name": "b"}], cm)
    orbit_plain.register_parameters(system=plain)
    assert "bigomega" not in orbit_plain.manifest
    assert "xbigomega" not in orbit_plain.manifest
    assert np.all(np.atleast_1d(orbit_plain.manifest["cosi"]["lower"]) == 0.0)

    astro = _DummySystem()
    astro.config = {"orbit": [{}], "astrometryinstrument": [],
                    "rvinstrument": []}
    orbit_astro = Orbit([{"name": "b"}], cm)
    orbit_astro.register_parameters(system=astro)
    assert "xbigomega" in orbit_astro.manifest
    assert "ybigomega" in orbit_astro.manifest
    assert orbit_astro.manifest["xbigomega"] is None  # RVs: full circle
    assert orbit_astro.manifest["bigomega"] == "default"
    assert np.all(np.atleast_1d(orbit_astro.manifest["cosi"]["lower"]) == -1.0)


def test_bigomega_halfplane_without_rvs():
    """
    Given: an astrometry system with NO RV instrument
    When: the orbit manifest is registered
    Then: ybigomega gets a lower bound of 0 (bigomega in [0, 180]),
          unseeded elements start at bigomega = 90 deg, and omega/bigomega
          carry a table note documenting the artificial boundary
    """
    astro = _DummySystem()
    astro.config = {"orbit": [{}], "astrometryinstrument": []}

    orbit = Orbit([{"name": "b"}], ConfigManager({}))
    orbit.register_parameters(system=astro)

    y_entry = orbit.manifest["ybigomega"]
    assert np.all(np.atleast_1d(y_entry["lower"]) == 0.0)
    assert np.all(np.atleast_1d(y_entry["initval"]) == 1.0)
    assert np.all(np.atleast_1d(orbit.manifest["xbigomega"]["initval"]) == 0.0)
    assert "degenerate" in orbit.manifest["bigomega"]["table_note"]
    assert "degenerate" in orbit.manifest["omega"]["table_note"]

    # The degeneracy is a reflection through the sky plane, invisible to
    # ALL astrometry including relative: the restriction applies to
    # rel-mode-only systems too.
    rel = _DummySystem()
    rel.config = {"orbit": [{}],
                  "astrometryinstrument": [{"mode": "rel"}]}
    orbit_rel = Orbit([{"name": "b"}], ConfigManager({}))
    orbit_rel.register_parameters(system=rel)
    assert np.all(np.atleast_1d(orbit_rel.manifest["ybigomega"]["lower"]) == 0.0)


def test_relative_track_invariant_under_node_flip(compiled_sky_functions):
    """
    Given: the degenerate transformation (bigomega, omega) ->
           (bigomega+180, omega+180) with tp held fixed
    When: the RELATIVE sky track is evaluated
    Then: it is identical -- the transformation is a reflection through
          the sky plane, so no astrometry (absolute or relative) can
          distinguish the two modes; only RVs identify the ascending node
    """
    sky_fn, rv_fn = compiled_sky_functions
    omega, ecc, bigom, cosi = 0.97, 0.35, 3.67, 0.44  # bigomega > 180 deg
    t = np.linspace(_TC, _TC + _P_DAYS, 101)
    a = np.array([10.0])

    # tp fixed: shift tc so calc_tp lands on the same tp with omega+pi
    tp = _tp_from_tc(_TC, _P_DAYS, ecc, omega)
    tc2 = tp + (_TC - _tp_from_tc(_TC, _P_DAYS, ecc, omega + np.pi))

    vals1 = [np.array([np.log10(_P_DAYS)]), np.array([_TC]),
             np.array([np.sqrt(ecc) * np.cos(omega)]),
             np.array([np.sqrt(ecc) * np.sin(omega)]),
             np.array([cosi]),
             np.array([np.cos(bigom)]), np.array([np.sin(bigom)])]
    vals2 = [np.array([np.log10(_P_DAYS)]), np.array([tc2]),
             np.array([-np.sqrt(ecc) * np.cos(omega)]),
             np.array([-np.sqrt(ecc) * np.sin(omega)]),
             np.array([cosi]),
             np.array([-np.cos(bigom)]), np.array([-np.sin(bigom)])]

    _, _, dE1, dN1 = sky_fn(*vals1, t, a)
    _, _, dE2, dN2 = sky_fn(*vals2, t, a)
    np.testing.assert_allclose(dE2, dE1, atol=1e-6)
    np.testing.assert_allclose(dN2, dN1, atol=1e-6)

    # ... while the RVs of the two modes are NOT the same (sign flip of
    # the reflex velocity), which is what actually breaks the degeneracy
    (rv1,) = rv_fn(*vals1, t, np.array([1.0]))
    (rv2,) = rv_fn(*vals2, t, np.array([1.0]))
    assert np.max(np.abs(rv1 - rv2)) > 0.5


def test_bigomega_seed_above_180_remaps_to_degenerate_partner():
    """
    Given: a no-RV astrometry system whose bigomega initval is in (180, 360)
    When: the orbit manifest is registered
    Then: the seed is remapped to the exactly-degenerate solution:
          (x, y) -> (-x, -y), (secosw, sesinw) -> (-secosw, -sesinw), and
          tc shifted so the position-vs-time model is unchanged
    """
    ecc, w, bigom, P, tc = 0.3, np.radians(40.0), np.radians(250.0), 100.0, 2455000.0
    user_params = {
        "orbit.0.logP": {"initval": np.log10(P)},
        "orbit.0.tc": {"initval": tc},
        "orbit.0.secosw": {"initval": np.sqrt(ecc) * np.cos(w)},
        "orbit.0.sesinw": {"initval": np.sqrt(ecc) * np.sin(w)},
        "orbit.0.bigomega": {"initval": np.degrees(bigom)},
    }
    astro = _DummySystem()
    astro.config = {"orbit": [{"name": "b"}], "astrometryinstrument": []}
    cm = ConfigManager(user_params, system_config=astro.config)
    cm.finalize_user_params()
    orbit = Orbit([{"name": "b"}], cm)
    orbit.register_parameters(system=astro)

    x = float(np.atleast_1d(orbit.manifest["xbigomega"]["initval"])[0])
    y = float(np.atleast_1d(orbit.manifest["ybigomega"]["initval"])[0])
    assert y >= 0.0
    bigom_new = np.arctan2(y, x)
    assert np.isclose(np.mod(bigom_new - (bigom - np.pi), 2 * np.pi), 0.0,
                      atol=1e-4) or \
        np.isclose(np.mod(bigom_new - (bigom - np.pi), 2 * np.pi), 2 * np.pi,
                   atol=1e-4)

    sc = float(np.atleast_1d(orbit.manifest["secosw"]["initval"])[0])
    ss = float(np.atleast_1d(orbit.manifest["sesinw"]["initval"])[0])
    assert np.isclose(sc, -np.sqrt(ecc) * np.cos(w), atol=1e-4)
    assert np.isclose(ss, -np.sqrt(ecc) * np.sin(w), atol=1e-4)

    # tc shift preserves the physical orbit: same time of periastron
    tc_new = float(np.atleast_1d(orbit.manifest["tc"]["initval"])[0])
    tp_old = _tp_from_tc(tc, P, ecc, w)
    tp_new = _tp_from_tc(tc_new, P, ecc, w + np.pi)
    assert np.isclose(np.mod(tp_new - tp_old, P), 0.0, atol=1e-6) or \
        np.isclose(np.mod(tp_new - tp_old, P), P, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: full System with gaia + abs + rel instruments
# ---------------------------------------------------------------------------

pytest_slow = pytest.mark.slow

# BH1-like truth
_TRUTH = dict(
    ra0=262.171207, dec0=-0.581091, plx=2.09, pmra=-7.70, pmdec=-25.85,
    P=185.6, ecc=0.451, w=np.radians(12.8), bigom=np.radians(97.8),
    inc=np.radians(126.6), mstar=0.93, mcomp=9.62,
)


def _simulate(tmp_dir):
    """Simulate gaia/abs/rel datasets from the reference implementation."""
    T = _TRUTH
    rng = np.random.default_rng(7)
    mtot = T["mstar"] + T["mcomp"]
    a_AU = (mtot * (T["P"] / 365.25) ** 2) ** (1.0 / 3.0)
    a_star = a_AU * (T["mcomp"] / mtot) * T["plx"]   # photocenter, mas
    a_rel = a_AU * T["plx"]                          # relative, mas

    tp = 2457000.0
    # invert tp(tc): the tc->tp offset is independent of the epoch argument
    tc = 2 * tp - _tp_from_tc(tp, T["P"], T["ecc"], T["w"])

    epoch = 2457400.0
    ra_r, dec_r = np.radians(T["ra0"]), np.radians(T["dec0"])

    from exozippy.ephemeris import get_observer_position

    def linear_terms(t):
        xyz = get_observer_position(t, "earth")
        P_E = xyz[:, 0] * np.sin(ra_r) - xyz[:, 1] * np.cos(ra_r)
        P_N = (xyz[:, 0] * np.cos(ra_r) * np.sin(dec_r)
               + xyz[:, 1] * np.sin(ra_r) * np.sin(dec_r)
               - xyz[:, 2] * np.cos(dec_r))
        dt_yr = (t - epoch) / 365.25
        return (T["pmra"] * dt_yr + T["plx"] * P_E,
                T["pmdec"] * dt_yr + T["plx"] * P_N)

    # gaia mode
    t_g = np.sort(rng.uniform(2456900.0, 2457900.0, 40))
    psi = rng.uniform(0, 2 * np.pi, 40)
    dE_o, dN_o = _sky_pos_reference(t_g, T["P"], tp, T["ecc"], T["w"],
                                    T["bigom"], T["inc"], a_star)
    lE, lN = linear_terms(t_g)
    err_g = np.full(40, 0.1)
    w_al = (lE + dE_o) * np.sin(psi) + (lN + dN_o) * np.cos(psi) \
        + rng.normal(0, err_g)
    np.savetxt(tmp_dir / "sim.gaia.astrom",
               np.column_stack([t_g, w_al, err_g, np.degrees(psi)]))

    # abs mode
    t_a = np.sort(rng.uniform(2456900.0, 2457900.0, 30))
    dE_o, dN_o = _sky_pos_reference(t_a, T["P"], tp, T["ecc"], T["w"],
                                    T["bigom"], T["inc"], a_star)
    lE, lN = linear_terms(t_a)
    err_a = np.full(30, 0.2)
    ra_obs = T["ra0"] + (lE + dE_o + rng.normal(0, err_a)) / RAD2MAS / np.cos(dec_r) * 180 / np.pi
    dec_obs = T["dec0"] + (lN + dN_o + rng.normal(0, err_a)) / RAD2MAS * 180 / np.pi
    np.savetxt(tmp_dir / "sim.abs.astrom",
               np.column_stack([t_a, ra_obs, dec_obs, err_a, err_a]))

    # rel mode (companion relative to host: omega_* + pi)
    t_r = np.sort(rng.uniform(2456900.0, 2457900.0, 20))
    dE_r, dN_r = _sky_pos_reference(t_r, T["P"], tp, T["ecc"], T["w"] + np.pi,
                                    T["bigom"], T["inc"], a_rel)
    err_sep = np.full(20, 0.05)
    err_pa = np.full(20, 0.5)  # deg
    sep = np.hypot(dE_r, dN_r) + rng.normal(0, err_sep)
    pa = np.degrees(np.arctan2(dE_r, dN_r)) + rng.normal(0, err_pa)
    np.savetxt(tmp_dir / "sim.rel.astrom",
               np.column_stack([t_r, sep, err_sep, pa, err_pa]))

    return tc, epoch


@pytest.fixture(scope="module")
def astrometry_system(tmp_path_factory):
    """Build one System with gaia + abs + rel instruments at the truth."""
    tmp_dir = tmp_path_factory.mktemp("astrom")
    tc, epoch = _simulate(tmp_dir)
    T = _TRUTH

    config = {
        "name": "astromtest",
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "astrometryinstrument": [
            {"name": "GaiaSim", "file": str(tmp_dir / "sim.gaia.astrom"),
             "mode": "gaia", "observer_location": "earth", "epoch": epoch},
            {"name": "GroundAbs", "file": str(tmp_dir / "sim.abs.astrom"),
             "mode": "abs", "observer_location": "earth", "epoch": epoch},
            {"name": "GroundRel", "file": str(tmp_dir / "sim.rel.astrom"),
             "mode": "rel"},
        ],
    }
    user_params = {
        "star.A.mass": {"initval": T["mstar"], "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5900, "sigma": 100},
        "star.A.feh": {"initval": -0.2, "sigma": 0.1},
        "star.A.ra": {"initval": T["ra0"]},
        "star.A.dec": {"initval": T["dec0"]},
        "star.A.pm_ra": {"initval": T["pmra"]},
        "star.A.pm_dec": {"initval": T["pmdec"]},
        "star.A.distance": {"initval": 1000.0 / T["plx"]},
        "planet.BH.mass": {"initval": T["mcomp"] * 1047.5655},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": T["P"]},
        "orbit.BH.tc": {"initval": tc},
        "orbit.BH.secosw": {"initval": np.sqrt(T["ecc"]) * np.cos(T["w"])},
        "orbit.BH.sesinw": {"initval": np.sqrt(T["ecc"]) * np.sin(T["w"])},
        "orbit.BH.bigomega": {"initval": np.degrees(T["bigom"])},
        "orbit.BH.cosi": {"initval": np.cos(T["inc"])},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    point = model.initial_point()
    return system, model, point


def _chi2_of(model, point, rv_name, n_obs, errs):
    obs = [v for v in model.observed_RVs if v.name == rv_name]
    assert len(obs) == 1, f"missing observed RV {rv_name}"
    ll = model.compile_logp(vars=obs, sum=True)(point)
    return -2.0 * ll - n_obs * np.log(2 * np.pi) - 2.0 * np.sum(np.log(errs))


@pytest.mark.slow
def test_gaia_mode_chi2_at_truth(astrometry_system):
    """
    Given: simulated Gaia along-scan epoch astrometry at the injected truth
    When: the model likelihood is evaluated at the initial point
    Then: chi2/N is consistent with pure noise (model matches simulation)
    """
    system, model, point = astrometry_system
    d = system.astrometryinstrument.datasets[0]
    chi2 = _chi2_of(model, point, "astrometryinstrument.model_GaiaSim",
                    len(d["w"]), d["err"])
    assert chi2 / len(d["w"]) < 2.0


@pytest.mark.slow
def test_abs_mode_chi2_at_truth(astrometry_system):
    """
    Given: simulated 2-D absolute astrometry at the injected truth
    When: the model likelihood is evaluated at the initial point
    Then: chi2/N is consistent with pure noise in both coordinates
    """
    system, model, point = astrometry_system
    d = system.astrometryinstrument.datasets[1]
    n = len(d["dE_obs"])
    chi2_E = _chi2_of(model, point, "astrometryinstrument.model_GroundAbs_E",
                      n, d["err_E"])
    chi2_N = _chi2_of(model, point, "astrometryinstrument.model_GroundAbs_N",
                      n, d["err_N"])
    assert chi2_E / n < 2.0
    assert chi2_N / n < 2.0


@pytest.mark.slow
def test_rel_mode_chi2_at_truth(astrometry_system):
    """
    Given: simulated relative (sep, PA) astrometry at the injected truth
    When: the model likelihood is evaluated at the initial point
    Then: chi2/N is consistent with pure noise in sep and wrapped PA
    """
    system, model, point = astrometry_system
    d = system.astrometryinstrument.datasets[2]
    n = len(d["sep"])
    chi2_sep = _chi2_of(model, point, "astrometryinstrument.model_GroundRel_sep",
                        n, d["err_sep"])
    sigma_pa = np.sqrt(d["err_pa"] ** 2)  # jitter initval = 0
    chi2_pa = _chi2_of(model, point, "astrometryinstrument.model_GroundRel_pa",
                       n, sigma_pa)
    assert chi2_sep / n < 2.0
    assert chi2_pa / n < 2.0


@pytest.mark.slow
def test_finite_logp_and_gradient(astrometry_system):
    """
    Given: the full gaia+abs+rel model
    When: logp and dlogp are evaluated at the initial point
    Then: both are finite (NUTS-safe: ops.kepler provides gradients)
    """
    system, model, point = astrometry_system
    logp = model.compile_logp()(point)
    assert np.isfinite(logp)
    dlogp = model.compile_dlogp()(point)
    assert np.all(np.isfinite(dlogp))
