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

from conftest import _DummyComponent, _DummySystem
from exozippy.components.astrometryinstrument import AstrometryInstrument
from exozippy.components.orbit.orbit import Orbit
from exozippy.config import ConfigManager
from exozippy.system import System

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
    sinf = (np.sqrt(1 - ecc**2) * np.sin(E)) / (1 - ecc * np.cos(E))
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
    r = a_scale * (1 - ecc**2) / (1 + ecc * np.cos(f))
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
    E_c = 2 * np.arctan2(
        np.sqrt(1 - ecc) * (1 - np.sin(w)), np.sqrt(1 + ecc) * np.cos(w)
    )
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
    tuple(
        np.random.uniform([-np.pi, 0.05, 0, -1], [np.pi, 0.8, 2 * np.pi, 1])
    ),
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
                orbit_comp.add_parameter(
                    model=pm.modelcontext(None),
                    param_name=param_name,
                    system=dummy_system,
                )

            t_var = pt.vector("t")
            a_var = pt.vector("a_scale")
            K_var = pt.vector("K_int")
            omap = np.array([0])

            dE_star, dN_star = orbit_comp.get_sky_position(t_var, a_var, omap)
            dE_rel, dN_rel = orbit_comp.get_sky_position(
                t_var, a_var, omap, relative=True
            )
            rv_node = orbit_comp.get_radial_velocity(t_var, K_var, omap)

            free_inputs = [
                orbit_comp.logP.value,
                orbit_comp.tc.value,
                orbit_comp.secosw.value,
                orbit_comp.sesinw.value,
                orbit_comp.cosi.value,
                orbit_comp.xbigomega.value,
                orbit_comp.ybigomega.value,
            ]
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
    return [
        np.array([np.log10(_P_DAYS)]),
        np.array([_TC]),
        np.array([np.sqrt(ecc) * np.cos(omega)]),
        np.array([np.sqrt(ecc) * np.sin(omega)]),
        np.array([cosi]),
        np.array([np.cos(bigom)]),
        np.array([np.sin(bigom)]),
    ]


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[
        f"w={c[0]:.2f}_e={c[1]:.2f}_O={c[2]:.2f}_ci={c[3]:.2f}" for c in _CASES
    ],
)
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

    dE_s, dN_s, dE_r, dN_r = sky_fn(
        *_free_vals(omega, ecc, bigom, cosi), t, a_scale
    )

    tp = _tp_from_tc(_TC, _P_DAYS, ecc, omega)
    inc = np.arccos(cosi)
    eE_s, eN_s = _sky_pos_reference(
        t, _P_DAYS, tp, ecc, omega, bigom, inc, 3.7
    )
    eE_r, eN_r = _sky_pos_reference(
        t, _P_DAYS, tp, ecc, omega + np.pi, bigom, inc, 3.7
    )

    np.testing.assert_allclose(dE_s[:, 0], eE_s, atol=1e-8)
    np.testing.assert_allclose(dN_s[:, 0], eN_s, atol=1e-8)
    np.testing.assert_allclose(dE_r[:, 0], eE_r, atol=1e-8)
    np.testing.assert_allclose(dN_r[:, 0], eN_r, atol=1e-8)


@pytest.mark.parametrize(
    "case",
    _CASES[:4],
    ids=[
        f"w={c[0]:.2f}_e={c[1]:.2f}_O={c[2]:.2f}_ci={c[3]:.2f}"
        for c in _CASES[:4]
    ],
)
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
    assert np.isclose(
        np.mod(pa - bigom, 2 * np.pi), 0.0, atol=1e-6
    ) or np.isclose(np.mod(pa - bigom, 2 * np.pi), 2 * np.pi, atol=1e-6)
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
    ra0, dec0 = 217.42, -62.68  # deg (alpha Cen-ish: big parallax regime)
    plx = 100.0  # mas
    t = np.linspace(2457000.0, 2457365.0, 25)

    data = np.column_stack(
        [
            t,
            np.full_like(t, ra0),
            np.full_like(t, dec0),
            np.ones_like(t),
            np.ones_like(t),
        ]
    )
    f = tmp_path / "abs.astrom"
    np.savetxt(f, data)

    user_params = {
        "star.0.ra": {"initval": ra0},
        "star.0.dec": {"initval": dec0},
    }
    cm = ConfigManager(user_params)
    comp = AstrometryInstrument(
        [
            {
                "name": "T",
                "file": str(f),
                "mode": "abs",
                "observer_location": "earth",
            }
        ],
        cm,
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)

    # Act
    comp.load_data(system)
    d = comp.datasets[0]

    # Exact geometry: apparent direction = unit(u * d_AU - b_obs)
    from exozippy.ephemeris import get_observer_position

    xyz = get_observer_position(t, "earth")
    ra_r, dec_r = np.radians(ra0), np.radians(dec0)
    u_hat = np.array(
        [
            np.cos(dec_r) * np.cos(ra_r),
            np.cos(dec_r) * np.sin(ra_r),
            np.sin(dec_r),
        ]
    )
    E_hat = np.array([-np.sin(ra_r), np.cos(ra_r), 0.0])
    N_hat = np.array(
        [
            -np.sin(dec_r) * np.cos(ra_r),
            -np.sin(dec_r) * np.sin(ra_r),
            np.cos(dec_r),
        ]
    )
    d_AU = RAD2MAS / plx  # 1/plx[rad] in AU
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
# Unit tests: RA offsets must wrap across the 0/360 branch cut (review 3.8)
# ---------------------------------------------------------------------------


class _ConstParam:
    """Minimal stand-in for a Parameter with a constant value vector."""

    def __init__(self, values):
        self.value = pt.as_tensor_variable(np.asarray(values, dtype=float))


def _abs_dataset_across_ra_zero(tmp_path, ra_ref_deg, ra_obs_deg, dec_deg):
    """An abs-mode instrument whose observed RA sits on the far side of the
    0/360 branch cut from the reference RA.  Returns (component, dataset)."""
    t = np.linspace(2457000.0, 2457365.0, 5)
    data = np.column_stack(
        [
            t,
            np.full_like(t, ra_obs_deg),
            np.full_like(t, dec_deg),
            np.ones_like(t),
            np.ones_like(t),
        ]
    )
    f = tmp_path / "abs_wrap.astrom"
    np.savetxt(f, data)

    cm = ConfigManager(
        {
            "star.0.ra": {"initval": ra_ref_deg},
            "star.0.dec": {"initval": dec_deg},
        }
    )
    comp = AstrometryInstrument(
        [{"name": "T", "file": str(f), "mode": "abs"}], cm
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)
    comp.load_data(system)
    return comp, comp.datasets[0]


def test_abs_mode_wraps_ra_offsets_across_zero(tmp_path):
    """
    Given: an abs-mode target at RA = 0.1 deg with the reference at 359.9 deg
    When:  load_data turns the observed positions into (dE, dN) offsets
    Then:  dE_obs is the short way round (+0.2 deg of RA times cos(dec)),
           not the unwrapped -359.8 deg

    The unwrapped difference is 1.1e9 mas, i.e. ~1e9 sigma from any model at
    the start of the fit -- catastrophic for every target straddling RA = 0.
    """
    # Arrange / Act: dec = 0 so cos(dec) = 1 and the offset IS the RA
    # difference, making the assertion read in degrees of RA.
    _, d = _abs_dataset_across_ra_zero(tmp_path, 359.9, 0.1, 0.0)

    # Assert
    offset_deg = np.degrees(d["dE_obs"] / RAD2MAS)
    np.testing.assert_allclose(offset_deg, 0.2, atol=1e-9)
    assert np.all(np.abs(d["dE_obs"]) < 1e6), (
        "RA offset was not wrapped across the 0/360 branch cut"
    )
    # Dec has no branch cut to cross and must be untouched.
    np.testing.assert_allclose(d["dN_obs"], 0.0, atol=1e-9)


def test_abs_mode_ra_offsets_are_continuous_across_zero(tmp_path):
    """
    Given: an abs-mode dataset with epochs on BOTH sides of RA = 0
    When:  the observed offsets are computed
    Then:  they span the true 0.15 deg of sky, not 360 deg

    This is the half of the bug no single value of star.ra could ever absorb:
    unwrapped, consecutive rows of ONE file differ by 360 deg.
    """
    # Arrange
    t = np.linspace(2457000.0, 2457730.0, 8)
    ra_obs = np.where(np.arange(len(t)) % 2 == 0, 359.95, 0.1)
    data = np.column_stack(
        [t, ra_obs, np.zeros_like(t), np.ones_like(t), np.ones_like(t)]
    )
    f = tmp_path / "abs_straddle.astrom"
    np.savetxt(f, data)
    cm = ConfigManager(
        {"star.0.ra": {"initval": 0.0}, "star.0.dec": {"initval": 0.0}}
    )
    comp = AstrometryInstrument(
        [{"name": "T", "file": str(f), "mode": "abs"}], cm
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)

    # Act
    comp.load_data(system)
    span_deg = np.degrees(np.ptp(comp.datasets[0]["dE_obs"]) / RAD2MAS)

    # Assert
    np.testing.assert_allclose(span_deg, 0.15, atol=1e-9)


def test_absolute_model_wraps_ra_offset_across_zero(tmp_path):
    """
    Given: an abs-mode reference at RA = 359.9 deg and star.ra at 0.1 deg
    When:  the symbolic model term dE is evaluated
    Then:  it reproduces the +0.2 deg offset the (wrapped) data carry

    The model side has to wrap too: star.ra carries hard bounds [0, 360] in
    star/defaults.yaml, so with the reference near 360 an unwrapped model term
    cannot produce a negative RA offset at all -- the fit is unreachable, not
    merely offset.
    """
    # Arrange
    comp, d = _abs_dataset_across_ra_zero(tmp_path, 359.9, 0.1, 0.0)

    star = _DummySystem()
    star.ra = _ConstParam(np.radians([0.1]))
    star.dec = _ConstParam(np.radians([0.0]))
    star.pm_ra = _ConstParam([0.0])
    star.pm_dec = _ConstParam([0.0])
    star.parallax = _ConstParam([0.0])
    system = _DummySystem()
    system.star = star
    comp.epoch = float(d["time"][0])

    # Act
    dE, _dN = comp._absolute_model(system, d, d["time"], None)
    dE_val = np.asarray(dE.eval())

    # Assert: the model matches the wrapped observed offset, to the mas
    np.testing.assert_allclose(np.degrees(dE_val / RAD2MAS), 0.2, atol=1e-9)
    np.testing.assert_allclose(dE_val, d["dE_obs"], atol=1e-6)


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
    astro.config = {
        "orbit": [{}],
        "astrometryinstrument": [],
        "rvinstrument": [],
    }
    orbit_astro = Orbit([{"name": "b"}], cm)
    orbit_astro.register_parameters(system=astro)
    assert "xbigomega" in orbit_astro.manifest
    assert "ybigomega" in orbit_astro.manifest
    assert orbit_astro.manifest["xbigomega"] is None  # RVs: full circle
    assert orbit_astro.manifest["bigomega"] == "default"
    assert np.all(np.atleast_1d(orbit_astro.manifest["cosi"]["lower"]) == -1.0)


# The half-plane TRUNCATION these two tests used to pin is gone (review
# 1.8.3): `ybigomega >= 0` biased a posterior that hugged the boundary, and a
# seed in (180, 360) was silently remapped onto its degenerate partner rather
# than honoured.  Both are replaced by `tests/test_node_degeneracy.py`, which
# asserts the new behaviour end to end (no bound, the seed kept, the labels
# collapsed AFTER sampling by one fold) on a real System rather than on the
# _DummySystem harness -- the predicate is per orbit now, so it reads the live
# astrometry/RV instances and a stub with an empty instrument list correctly
# reports nothing degenerate.


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

    vals1 = [
        np.array([np.log10(_P_DAYS)]),
        np.array([_TC]),
        np.array([np.sqrt(ecc) * np.cos(omega)]),
        np.array([np.sqrt(ecc) * np.sin(omega)]),
        np.array([cosi]),
        np.array([np.cos(bigom)]),
        np.array([np.sin(bigom)]),
    ]
    vals2 = [
        np.array([np.log10(_P_DAYS)]),
        np.array([tc2]),
        np.array([-np.sqrt(ecc) * np.cos(omega)]),
        np.array([-np.sqrt(ecc) * np.sin(omega)]),
        np.array([cosi]),
        np.array([-np.cos(bigom)]),
        np.array([-np.sin(bigom)]),
    ]

    _, _, dE1, dN1 = sky_fn(*vals1, t, a)
    _, _, dE2, dN2 = sky_fn(*vals2, t, a)
    np.testing.assert_allclose(dE2, dE1, atol=1e-6)
    np.testing.assert_allclose(dN2, dN1, atol=1e-6)

    # ... while the RVs of the two modes are NOT the same (sign flip of
    # the reflex velocity), which is what actually breaks the degeneracy
    (rv1,) = rv_fn(*vals1, t, np.array([1.0]))
    (rv2,) = rv_fn(*vals2, t, np.array([1.0]))
    assert np.max(np.abs(rv1 - rv2)) > 0.5


# ---------------------------------------------------------------------------
# Integration: full System with gaia + abs + rel instruments
# ---------------------------------------------------------------------------

pytest_slow = pytest.mark.slow

# BH1-like truth
_TRUTH = dict(
    ra0=262.171207,
    dec0=-0.581091,
    plx=2.09,
    pmra=-7.70,
    pmdec=-25.85,
    P=185.6,
    ecc=0.451,
    w=np.radians(12.8),
    bigom=np.radians(97.8),
    inc=np.radians(126.6),
    mstar=0.93,
    mcomp=9.62,
)


def _simulate(tmp_dir):
    """Simulate gaia/abs/rel datasets from the reference implementation."""
    T = _TRUTH
    rng = np.random.default_rng(7)
    mtot = T["mstar"] + T["mcomp"]
    a_AU = (mtot * (T["P"] / 365.25) ** 2) ** (1.0 / 3.0)
    a_star = a_AU * (T["mcomp"] / mtot) * T["plx"]  # photocenter, mas
    a_rel = a_AU * T["plx"]  # relative, mas

    tp = 2457000.0
    # invert tp(tc): the tc->tp offset is independent of the epoch argument
    tc = 2 * tp - _tp_from_tc(tp, T["P"], T["ecc"], T["w"])

    epoch = 2457400.0
    ra_r, dec_r = np.radians(T["ra0"]), np.radians(T["dec0"])

    from exozippy.ephemeris import get_observer_position

    def linear_terms(t):
        xyz = get_observer_position(t, "earth")
        P_E = xyz[:, 0] * np.sin(ra_r) - xyz[:, 1] * np.cos(ra_r)
        P_N = (
            xyz[:, 0] * np.cos(ra_r) * np.sin(dec_r)
            + xyz[:, 1] * np.sin(ra_r) * np.sin(dec_r)
            - xyz[:, 2] * np.cos(dec_r)
        )
        dt_yr = (t - epoch) / 365.25
        return (
            T["pmra"] * dt_yr + T["plx"] * P_E,
            T["pmdec"] * dt_yr + T["plx"] * P_N,
        )

    # gaia mode
    t_g = np.sort(rng.uniform(2456900.0, 2457900.0, 40))
    psi = rng.uniform(0, 2 * np.pi, 40)
    dE_o, dN_o = _sky_pos_reference(
        t_g, T["P"], tp, T["ecc"], T["w"], T["bigom"], T["inc"], a_star
    )
    lE, lN = linear_terms(t_g)
    err_g = np.full(40, 0.1)
    w_al = (
        (lE + dE_o) * np.sin(psi)
        + (lN + dN_o) * np.cos(psi)
        + rng.normal(0, err_g)
    )
    np.savetxt(
        tmp_dir / "sim.gaia.astrom",
        np.column_stack([t_g, w_al, err_g, np.degrees(psi)]),
    )

    # abs mode
    t_a = np.sort(rng.uniform(2456900.0, 2457900.0, 30))
    dE_o, dN_o = _sky_pos_reference(
        t_a, T["P"], tp, T["ecc"], T["w"], T["bigom"], T["inc"], a_star
    )
    lE, lN = linear_terms(t_a)
    err_a = np.full(30, 0.2)
    ra_obs = (
        T["ra0"]
        + (lE + dE_o + rng.normal(0, err_a))
        / RAD2MAS
        / np.cos(dec_r)
        * 180
        / np.pi
    )
    dec_obs = (
        T["dec0"] + (lN + dN_o + rng.normal(0, err_a)) / RAD2MAS * 180 / np.pi
    )
    np.savetxt(
        tmp_dir / "sim.abs.astrom",
        np.column_stack([t_a, ra_obs, dec_obs, err_a, err_a]),
    )

    # rel mode (companion relative to host: omega_* + pi)
    t_r = np.sort(rng.uniform(2456900.0, 2457900.0, 20))
    dE_r, dN_r = _sky_pos_reference(
        t_r, T["P"], tp, T["ecc"], T["w"] + np.pi, T["bigom"], T["inc"], a_rel
    )
    err_sep = np.full(20, 0.05)
    err_pa = np.full(20, 0.5)  # deg
    sep = np.hypot(dE_r, dN_r) + rng.normal(0, err_sep)
    pa = np.degrees(np.arctan2(dE_r, dN_r)) + rng.normal(0, err_pa)
    np.savetxt(
        tmp_dir / "sim.rel.astrom",
        np.column_stack([t_r, sep, err_sep, pa, err_pa]),
    )

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
            {
                "name": "GaiaSim",
                "file": str(tmp_dir / "sim.gaia.astrom"),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": epoch,
            },
            {
                "name": "GroundAbs",
                "file": str(tmp_dir / "sim.abs.astrom"),
                "mode": "abs",
                "observer_location": "earth",
                "epoch": epoch,
            },
            {
                "name": "GroundRel",
                "file": str(tmp_dir / "sim.rel.astrom"),
                "mode": "rel",
            },
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
    chi2 = _chi2_of(
        model,
        point,
        "astrometryinstrument.model_GaiaSim",
        len(d["w"]),
        d["err"],
    )
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
    chi2_E = _chi2_of(
        model, point, "astrometryinstrument.model_GroundAbs_E", n, d["err_E"]
    )
    chi2_N = _chi2_of(
        model, point, "astrometryinstrument.model_GroundAbs_N", n, d["err_N"]
    )
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
    chi2_sep = _chi2_of(
        model,
        point,
        "astrometryinstrument.model_GroundRel_sep",
        n,
        d["err_sep"],
    )
    sigma_pa = np.sqrt(d["err_pa"] ** 2)  # jitter initval = 0
    chi2_pa = _chi2_of(
        model, point, "astrometryinstrument.model_GroundRel_pa", n, sigma_pa
    )
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


@pytest.mark.slow
def test_jax_gradient_finite_with_massive_companion(astrometry_system):
    """
    Given: a system whose companion is a 9.6 Msun black hole
           (m_total = 10.6 Msun, i.e. 500 * m_total >> 709)
    When: the logp gradient is evaluated through the JAX backend
          (the numpyro/blackjax sampling path)
    Then: every gradient is finite.

    Regression: planet.build_likelihood's log(sigmoid(500*m_total))
    potential NaN'd in the JAX gradient once exp(500*m_total) overflowed
    (m_total > 1.42 Msun) -- an unselected jnp.where branch of pytensor's
    softplus -- silently freezing every numpyro chain at its start while
    the C-backend gradient stayed finite.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    system, model, point = astrometry_system
    logp_fn = get_jaxified_logp(model)
    vals = [point[v.name] for v in model.value_vars]
    assert np.isfinite(float(logp_fn(vals)))
    grads = jax.grad(lambda vs: logp_fn(vs))(vals)
    bad = [
        v.name
        for v, g in zip(model.value_vars, grads)
        if not np.all(np.isfinite(np.asarray(g)))
    ]
    assert not bad, f"non-finite JAX gradients: {bad}"


# ---------------------------------------------------------------------------
# Review 2.6.1: the rel-mode jitter-variance floor must cover BOTH channels
# ---------------------------------------------------------------------------

# Interferometric-style relative astrometry: the TANGENTIAL error
# (err_pa * sep) is far better than the radial one (err_sep).  That is the
# regime in which a jitter legal for the separation channel still makes the
# position-angle variance err_pa^2 + jv/sep^2 negative.
_INTERF = dict(P=300.0, tp=2457000.0, a_rel=50.0, err_sep=0.05, err_pa=0.02)


def _write_interferometric_rel(path):
    """Write a rel-mode file whose PA channel is the tighter constraint."""
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(2456900.0, 2457900.0, 20))
    ph = 2 * np.pi * (t - _INTERF["tp"]) / _INTERF["P"]
    dE = _INTERF["a_rel"] * np.cos(ph)
    dN = _INTERF["a_rel"] * 0.7 * np.sin(ph)
    np.savetxt(
        path,
        np.column_stack(
            [
                t,
                np.hypot(dE, dN),
                np.full_like(t, _INTERF["err_sep"]),
                np.degrees(np.arctan2(dE, dN)),
                np.full_like(t, _INTERF["err_pa"]),
            ]
        ),
    )
    return t


def test_rel_jitter_floor_covers_the_pa_channel(tmp_path):
    """
    Given: rel-mode data whose tangential error (err_pa*sep) is much smaller
           than its separation error
    When: load_data computes the jitter-variance floor
    Then: the floor keeps the PA channel's variance positive as well -- it is
          set by whichever channel is tighter, not by err_sep alone

    Regression (review 2.6.1): the floor was -0.95*min(err_sep)**2, which for
    these data still allows a jitter that drives err_pa**2 + jv/sep**2
    negative -> NaN sigma over a region the sampler may visit.
    """
    # Arrange
    f = tmp_path / "interf.astrom"
    _write_interferometric_rel(f)
    comp = AstrometryInstrument(
        [{"name": "I", "file": str(f), "mode": "rel"}], ConfigManager({})
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)

    # Act
    comp.load_data(system)
    floor = float(np.atleast_1d(comp.jittervar_lower[0])[0])
    d = comp.datasets[0]
    tangential = d["err_pa"] * d["sep"]  # mas

    # Assert: this file really is PA-limited (else the test proves nothing)
    assert np.min(tangential) < np.min(d["err_sep"])
    # every allowed jitter keeps BOTH variances positive
    assert np.all(d["err_sep"] ** 2 + floor > 0.0)
    assert np.all(d["err_pa"] ** 2 + floor / d["sep"] ** 2 > 0.0)
    # and it is the shared -0.95 * min(err)**2 margin over the tighter channel
    assert floor == pytest.approx(-0.95 * np.min(tangential) ** 2)


@pytest.fixture(scope="module")
def interferometric_rel_system(tmp_path_factory):
    """A one-instrument rel-mode System on PA-limited data."""
    tmp_dir = tmp_path_factory.mktemp("interf")
    _write_interferometric_rel(tmp_dir / "interf.astrom")

    config = {
        "name": "interf",
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "astrometryinstrument": [
            {
                "name": "I",
                "file": str(tmp_dir / "interf.astrom"),
                "mode": "rel",
            }
        ],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "star.A.distance": {"initval": 100.0},
        "planet.b.mass": {"initval": 300.0},
        "orbit.b.period": {"initval": _INTERF["P"]},
        "orbit.b.tc": {"initval": _INTERF["tp"]},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    return system, model, model.initial_point()


@pytest.mark.slow
def test_rel_jitter_inside_its_bound_has_finite_logp_and_gradient(
    interferometric_rel_system,
):
    """
    Given: a rel-mode model on PA-limited data
    When: jitter_variance is set to values strictly INSIDE its resolved
          lower bound (the region the sampler is free to visit)
    Then: logp and every dlogp entry are finite there

    Regression (review 2.6.1): with the floor taken from err_sep alone the
    bound resolved to -0.002375 while the PA channel needs jv > -1.5e-4, and
    logp at jitter_variance = -0.0023726 was -inf with a non-finite
    gradient -- a wall of NaN inside the allowed interval.
    """
    system, model, point = interferometric_rel_system
    jv = system.astrometryinstrument.jitter_variance
    lower = float(np.atleast_1d(jv.lower)[0])
    assert lower < 0.0  # the bound under test is the negative floor

    key = f"{jv.label}_raw"
    logp_fn = model.compile_logp()
    dlogp_fn = model.compile_dlogp()

    for frac in (0.999, 0.5, 0.0):  # 0.999*lower is a hair inside the wall
        probe = dict(point)
        probe[key] = jv.raw_from_initval(np.array([frac * lower]))
        logp = logp_fn(probe)
        dlogp = dlogp_fn(probe)
        assert np.isfinite(logp), f"logp = {logp} at jv = {frac * lower:.6g}"
        assert np.all(np.isfinite(dlogp)), (
            f"non-finite dlogp at jv = {frac * lower:.6g}"
        )


# ---------------------------------------------------------------------------
# Reviews 2.6.2 / 2.6.3 / 2.6.4: plots of an orbit-less gaia fit
# ---------------------------------------------------------------------------

# Proper motion is PINNED here (sigma: 0), so it is absent from every draw --
# which is exactly what point.get(label, 0.0) used to turn into zero.
_PM = dict(ra0=262.171207, dec0=-0.581091, pmra=-7.70, pmdec=-25.85, plx=2.09)


@pytest.fixture(scope="module")
def pm_only_gaia_system(tmp_path_factory):
    """A legal gaia fit with NO orbit: proper motion + parallax only."""
    tmp_dir = tmp_path_factory.mktemp("pmonly")
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(2456900.0, 2457900.0, 40))
    psi = rng.uniform(0, 2 * np.pi, 40)
    err = np.full(40, 0.1)
    np.savetxt(
        tmp_dir / "pm.astrom",
        np.column_stack([t, rng.normal(0, err), err, np.degrees(psi)]),
    )

    config = {
        "name": "pmonly",
        "star": [{"name": "A", "mist": False}],
        "astrometryinstrument": [
            {
                "name": "G",
                "file": str(tmp_dir / "pm.astrom"),
                "mode": "gaia",
                "observer_location": "earth",
            }
        ],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "star.A.ra": {"initval": _PM["ra0"]},
        "star.A.dec": {"initval": _PM["dec0"]},
        "star.A.pm_ra": {"initval": _PM["pmra"], "sigma": 0},
        "star.A.pm_dec": {"initval": _PM["pmdec"], "sigma": 0},
        "star.A.distance": {"initval": 1000.0 / _PM["plx"]},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    point = system.get_internal_point(model, model.initial_point())
    system.compile_plotter_functions(model)
    return system, model, point


@pytest.mark.slow
def test_plot_model_uses_pinned_proper_motion(pm_only_gaia_system):
    """
    Given: a fit whose proper motion is PINNED (sigma: 0) and therefore
           absent from the point
    When: the along-scan model trace is built by plot_data
    Then: it carries the pinned proper motion, matching an independent
          pm + parallax evaluation

    Regression (review 2.6.2): _linear_terms read the point with
    point.get(label, 0.0), so every pinned parameter silently plotted as
    zero -- the star stood still in the plots while the likelihood used
    its real proper motion.
    """
    system, _, point = pm_only_gaia_system
    comp = system.astrometryinstrument
    d = comp.datasets[0]

    # Act
    spec = comp.plot_data(system, point)[0]
    model_trace = [tr for tr in spec.traces if tr.role == "model"][0]

    # Assert: independent pm + parallax along-scan reference (no orbit)
    dt_yr = (d["time"] - comp.epoch) / 365.25
    dE = _PM["pmra"] * dt_yr + _PM["plx"] * d["P_E"]
    dN = _PM["pmdec"] * dt_yr + _PM["plx"] * d["P_N"]
    expected = dE * d["sin_psi"] + dN * d["cos_psi"]
    np.testing.assert_allclose(model_trace.y, expected, atol=1e-6)

    # ... and the pm term genuinely dominates, so dropping it is not a
    # rounding-level difference: with pm = 0 the trace would be tiny
    parallax_only = (
        _PM["plx"] * d["P_E"] * d["sin_psi"]
        + _PM["plx"] * d["P_N"] * d["cos_psi"]
    )
    assert np.ptp(expected) > 5 * np.ptp(parallax_only)


@pytest.mark.slow
def test_orbitless_gaia_fit_still_renders_its_plots(
    pm_only_gaia_system, tmp_path
):
    """
    Given: a legal gaia fit with no orbit component
    When: plot_data (data-only and with a point) and plot() are called
    Then: specs come back and the PDFs are written -- neither the data nor
          the pm+parallax model nor the sky plot needs an orbit

    Regression (review 2.6.3): compile_plotters returned early without an
    orbit, leaving _compiled_photo unset, and plot() bailed on that -- an
    orbit-less astrometry fit produced no astrometry PDFs at all.
    """
    system, _, point = pm_only_gaia_system
    comp = system.astrometryinstrument

    # data-only specs are usable without a model at all
    assert len(comp.plot_data(system, None)) == 1
    assert len(comp.plot_data(system, point)) == 1

    prefix = str(tmp_path / "px")
    comp.plot(system, [point], filename_prefix=prefix)
    assert (tmp_path / "px_astrometry_G.pdf").exists()
    assert (tmp_path / "px_astrometry_G_sky.pdf").exists()


@pytest.mark.slow
def test_gaia_param_deps_include_the_linear_terms(pm_only_gaia_system):
    """
    Given: a gaia spec whose model is the numpy pm+parallax _linear_terms
    When: its param_deps are declared
    Then: they name the star's ra/dec/pm/distance labels

    Regression (review 2.6.4): param_deps only ever came from walking the
    photocenter-orbit tensor graph, so the NumPy linear terms contributed
    nothing.  With no orbit the list was empty, the Evaluator's
    changed_label filter skipped the component for every slider, and the
    GUI live chart froze completely.
    """
    system, _, point = pm_only_gaia_system
    spec = system.astrometryinstrument.plot_data(system, point)[0]

    star = system.star
    for par in (star.ra, star.dec, star.pm_ra, star.pm_dec, star.distance):
        assert par.label in spec.param_deps, f"{par.label} missing"
    # every declared dep must be a label the Evaluator can actually send
    known = {p.label for p in system.plot_params}
    assert set(spec.param_deps) <= known


def test_conflicting_per_dataset_epochs_raise(tmp_path):
    """
    Given: two astrometry datasets that each declare a different `epoch:`
    When: load_data resolves the reference epoch for ra/dec/pm
    Then: it raises naming both epochs

    Regression (silent-bandaid audit): `epoch:` is advertised per instrument
    in config_schema, but there is only ONE reference epoch because there is
    only one star.ra/dec/pm to propagate from.  The code took epochs[0] and
    dropped the rest, so the canonical Hipparcos (1991.25) + Gaia (2016.0)
    combination silently propagated the second dataset from the first one's
    epoch -- an offset of pm * 24.75 yr, absorbed by ra/dec/pm.
    """
    # Arrange
    f1 = tmp_path / "a.astrom"
    f2 = tmp_path / "b.astrom"
    _write_interferometric_rel(f1)
    _write_interferometric_rel(f2)
    comp = AstrometryInstrument(
        [
            {
                "name": "A",
                "file": str(f1),
                "mode": "rel",
                "epoch": 2448349.0625,
            },
            {"name": "B", "file": str(f2), "mode": "rel", "epoch": 2457389.0},
        ],
        ConfigManager({}),
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)

    # Act / Assert
    with pytest.raises(ValueError, match=r"conflicting `epoch:` values"):
        comp.load_data(system)


def test_repeated_identical_epochs_are_accepted(tmp_path):
    """
    Given: two datasets that name the SAME epoch
    When: load_data runs
    Then: it is accepted and used -- the guard only rejects disagreement
    """
    # Arrange
    f1 = tmp_path / "a.astrom"
    f2 = tmp_path / "b.astrom"
    _write_interferometric_rel(f1)
    _write_interferometric_rel(f2)
    comp = AstrometryInstrument(
        [
            {"name": "A", "file": str(f1), "mode": "rel", "epoch": 2457389.0},
            {"name": "B", "file": str(f2), "mode": "rel", "epoch": 2457389.0},
        ],
        ConfigManager({}),
    )
    system = _DummySystem()
    system.star = _DummyComponent(1)

    # Act
    comp.load_data(system)

    # Assert
    assert comp.epoch == pytest.approx(2457389.0)


# ---------------------------------------------------------------------------
# build_maps builds star_map and nothing else (review 5.10.1)
# ---------------------------------------------------------------------------


def test_build_maps_does_not_build_a_dead_planet_map(tmp_path):
    """
    Given a rel dataset using the legacy `planet_ndx` alias,
    When build_maps runs,
    Then there is no `planet_map`: rel-orbit resolution lives in __init__
      (self.rel_orbit), nothing read the map, and stage 5 turned every
      `*_map` attribute into an int32 shared variable -- so the dead
      assignment cost one per fit.  star_map stays; it IS read.
    """
    # Arrange
    f = tmp_path / "rel.astrom"
    _write_interferometric_rel(f)
    comp = AstrometryInstrument(
        [
            {
                "name": "R",
                "file": str(f),
                "mode": "rel",
                "planet_ndx": 0,
                "star_ndx": 0,
            }
        ],
        ConfigManager({}),
    )

    # Act
    comp.build_maps()

    # Assert
    assert not hasattr(comp, "planet_map")
    assert list(comp.star_map) == [0]
    # the legacy alias still resolves -- it just does so in __init__
    assert comp.rel_orbit == [0]
