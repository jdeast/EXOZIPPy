"""
Tests for Orbit.state_vectors (review 4.8.2) -- the one full-state accessor
the element-to-observable projections are re-expressed on.

What is pinned here, and why each pin is the one that matters:

  - The velocities really are d/dt of the positions (central differences),
    which is what makes the tuple a STATE vector rather than six related
    numbers.  A sign error in vx_phase/vz_phase cannot raise chi2 anywhere
    (nothing consumed VX/VY before this accessor existed), so only a direct
    derivative check can catch one.
  - The node convention: at the ascending node the body sits at
    PA = bigomega moving AWAY from the observer (Z = 0 crossing upward,
    VZ > 0) -- the same convention test_skyframe.py pins for
    get_sky_position/get_radial_velocity, asserted here on the state tuple
    itself.
  - get_sky_position is the position half, get_radial_velocity is VZ with
    the amplitude collapsed into K -- the two delegations that make the
    existing projections consumers of the kernel rather than parallel
    re-derivations.
  - relative=True is the omega + 180 flip, i.e. exact negation of all six
    components (a Keplerian identity; review 8.8.15 records that it becomes
    a subtraction under an N-body backend).

The conventions themselves (left-handed frame, Thiele-Innes projection,
parallax signs) stay pinned by tests/test_skyframe.py and
tests/test_astrometry.py, which now exercise the same kernel path.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummySystem
from exozippy.components.orbit.orbit import Orbit
from exozippy.config import ConfigManager

_P_DAYS = 10.0
_TC = 2450000.0

_CASES = [
    # (omega, ecc, bigomega, cosi)
    (0.3, 0.05, 1.2, 0.4),
    (0.0, 0.3, 0.0, 0.0),
    (np.pi / 2, 0.5, 3.0, -0.6),
    (-1.0, 0.2, 5.5, 0.9),
    (2.5, 0.7, 2.2, -0.95),
]


@pytest.fixture(scope="module")
def compiled_state_functions():
    """Compile state_vectors / get_sky_position / get_radial_velocity once.

    Given: a standalone Orbit whose manifest includes bigomega (astrometry
    active in the system topology), exactly as tests/test_astrometry.py
    builds one.
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

            state = orbit_comp.state_vectors(t_var, a_var, omap)
            state_rel = orbit_comp.state_vectors(
                t_var, a_var, omap, relative=True
            )
            dE, dN = orbit_comp.get_sky_position(t_var, a_var, omap)
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
            state_fn = pytensor.function(
                inputs=free_inputs + [t_var, a_var],
                outputs=list(state) + list(state_rel) + [dE, dN],
                on_unused_input="ignore",
            )
            rv_fn = pytensor.function(
                inputs=free_inputs + [t_var, K_var],
                outputs=[rv_node],
                on_unused_input="ignore",
            )
    return state_fn, rv_fn


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


def _eval_state(state_fn, case, t, a_scale=1.0):
    out = state_fn(*_free_vals(*case), t, np.array([a_scale], dtype=float))
    out = [np.asarray(o)[:, 0] for o in out]
    return out[:6], out[6:12], (out[12], out[13])


_IDS = [
    f"w={c[0]:.2f}_e={c[1]:.2f}_O={c[2]:.2f}_ci={c[3]:.2f}" for c in _CASES
]


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_velocities_are_derivatives_of_positions(
    case, compiled_state_functions
):
    """
    Given: a dense time grid over one period,
    When: state_vectors is evaluated,
    Then: (VX, VY, VZ) match central differences of (X, Y, Z) at every
          interior point.
    """
    state_fn, _ = compiled_state_functions
    t = np.linspace(_TC, _TC + _P_DAYS, 20001)
    (X, Y, Z, VX, VY, VZ), _, _ = _eval_state(state_fn, case, t)

    # np.gradient's endpoints are one-sided (first order); compare interior.
    for pos, vel, name in ((X, VX, "VX"), (Y, VY, "VY"), (Z, VZ, "VZ")):
        num = np.gradient(pos, t)
        np.testing.assert_allclose(
            vel[1:-1], num[1:-1], atol=5e-6, err_msg=name
        )


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_ascending_node_convention(case, compiled_state_functions):
    """
    Given: the epochs where Z crosses zero upward (the ascending node),
    When: the state there is inspected,
    Then: the body is receding (VZ > 0) and sits at PA = bigomega.
    """
    state_fn, _ = compiled_state_functions
    omega, ecc, bigom, cosi = case
    t = np.linspace(_TC, _TC + _P_DAYS, 200001)
    (X, Y, Z, VX, VY, VZ), _, _ = _eval_state(state_fn, case, t)

    crossings = np.nonzero((Z[:-1] < 0.0) & (Z[1:] >= 0.0))[0]
    assert crossings.size >= 1
    for i in crossings:
        assert VZ[i] > 0.0, "body must recede at its ascending node"
        pa = np.degrees(np.arctan2(Y[i], X[i])) % 360.0
        expected = np.degrees(bigom) % 360.0
        diff = (pa - expected + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05, f"PA at node = {pa}, bigomega = {expected}"


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_projection_delegations(case, compiled_state_functions):
    """
    Given: one set of elements,
    When: get_sky_position and get_radial_velocity are evaluated alongside
          state_vectors,
    Then: (dE, dN) == (Y, X) exactly, and the RV equals VZ rescaled by the
          collapsed amplitude K / (n * a * sin(i) / sqrt(1 - e^2)).
    """
    state_fn, rv_fn = compiled_state_functions
    omega, ecc, bigom, cosi = case
    t = np.linspace(_TC, _TC + _P_DAYS, 101)
    (X, Y, Z, VX, VY, VZ), _, (dE, dN) = _eval_state(state_fn, case, t)

    np.testing.assert_array_equal(dE, Y)
    np.testing.assert_array_equal(dN, X)

    # K chosen so the collapsed amplitude equals state_vectors' vamp*sini
    # with a_scale = 1: the two must then agree to roundoff.
    n_val = 2.0 * np.pi / _P_DAYS
    sini = np.sqrt(1.0 - cosi**2)
    K = n_val * sini / np.sqrt(1.0 - ecc**2)
    (rv,) = rv_fn(*_free_vals(*case), t, np.array([K]))
    np.testing.assert_allclose(rv[:, 0], VZ, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_relative_is_the_exact_negation(case, compiled_state_functions):
    """
    Given: the same elements,
    When: state_vectors is evaluated with relative=False and relative=True,
    Then: all six components negate exactly (the omega + 180 flip).
    """
    state_fn, _ = compiled_state_functions
    t = np.linspace(_TC, _TC + _P_DAYS, 101)
    absolute, rel, _ = _eval_state(state_fn, case, t)
    for a, r, name in zip(absolute, rel, "X Y Z VX VY VZ".split()):
        np.testing.assert_allclose(r, -a, rtol=0, atol=1e-13, err_msg=name)
