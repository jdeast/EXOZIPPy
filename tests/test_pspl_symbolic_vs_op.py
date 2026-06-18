"""
Verify that the symbolic (PyTensor) Paczynski formula and the MulensModel
Op give the same magnification.

Note: get_magnification_op() routes point-source PSPL to the symbolic path
by default (NUTS-friendly, avoids _MagGradOp overhead).  These tests call
MulensMagOp directly to verify the Op implementation itself.
"""
import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt

pytestmark = pytest.mark.slow
import pymc as pm

from exozippy.system import System
from exozippy.components.mulensing.op import MulensMagOp


_BASE_USER_PARAMS = {
    "lens.Lens.t_0": {"initval": 2460025.0},
    "lens.Lens.u_0": {"initval": 0.2},
    "lens.Lens.pi_E_N": {"initval": 0.0, "sigma": 0.0},
    "lens.Lens.pi_E_E": {"initval": 0.0, "sigma": 0.0},
    "star.Lens.distance":  {"initval": 4000.0},
    "star.Source.distance":{"initval": 8000.0},
    "star.Lens.mass":      {"initval": 0.5},
    "star.Lens.pm_ra":     {"initval": 0.0},
    "star.Lens.pm_dec":    {"initval": 0.0},
    "star.Source.pm_ra":   {"initval": 0.0},
    "star.Source.pm_dec":  {"initval": 0.0},
    "star.Source.ra":      {"initval": 266.4168},
    "star.Source.dec":     {"initval": -29.0078},
    "star.Lens.ra":        {"initval": 266.4168},
    "star.Lens.dec":       {"initval": -29.0078},
}
_CONFIG = {
    "star": [{"name": "Lens"}, {"name": "Source"}],
    "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}]
}
_COORDS = "266.4168d -29.0078d"


def _build_system(extra_params=None):
    params = dict(_BASE_USER_PARAMS)
    if extra_params:
        params.update(extra_params)
    system = System(_CONFIG, user_params=params)
    system.prepare()
    model = system.build_model()
    return system, model


def _eval_both(system, model, obs_abs, t_vals):
    """Compile and evaluate symbolic + Op magnification at raw=0 (initvals)."""
    with model:
        A_sym_node = system.lens.get_magnification(t_vals, obs_abs, system, index=0)

        sp = system.lens._get_safe_mm_params(0)
        mag_op = MulensMagOp(coords=_COORDS, mag_method="point_source", use_rho=False)
        A_op_node = mag_op(
            pt.stack([sp['t0'], sp['u0'], sp['tE'], sp['pi_N'], sp['pi_E']]),
            pt.as_tensor_variable(t_vals),
            pt.as_tensor_variable(obs_abs),
        )

        f_sym = pytensor.function(model.free_RVs, A_sym_node, on_unused_input='ignore')
        f_op  = pytensor.function(model.free_RVs, A_op_node,  on_unused_input='ignore')

        ip = model.initial_point()
        zero_in = [np.zeros_like(ip[v.name]).astype("float64") for v in model.free_RVs]
        return f_sym(*zero_in), f_op(*zero_in)


def test_pspl_symbolic_vs_op_no_parallax():
    """
    Given zero observer positions (no satellite, no parallax),
    When symbolic and Op are evaluated,
    Then they agree to < 1e-4 — both reduce to the pure Paczynski formula.
    """
    t_vals = np.linspace(2460000.0, 2460050.0, 200)
    zero_obs = np.zeros((len(t_vals), 3), dtype=np.float64)

    system, model = _build_system()
    m_sym, m_op = _eval_both(system, model, zero_obs, t_vals)

    max_diff = np.max(np.abs(m_sym - m_op))
    assert max_diff < 1e-4, f"max |A_sym - A_op| (no parallax) = {max_diff:.2e}"


def test_pspl_symbolic_vs_op_with_earth_parallax():
    """
    Given a satellite observer (Earth + constant ~0.05 AU displacement) and
      non-zero pi_E so parallax changes the light curve,
    When symbolic and Op are evaluated,
    Then they agree to < 1e-3.

    This test guards against the ephemeris-convention bug where the symbolic
    formula received Skowron+2011 geocentric deviations while MulensModel
    expected absolute barycentric positions.

    Uses a ±1-day window centred on t0: Earth's orbit deviates from a linear
    fit by < 1.5e-4 AU over that interval (far smaller than the 0.05 AU
    satellite offset), so the symbolic linear-Earth approximation and the Op's
    real ephemeris both yield geocentric ≈ satellite_offset.  No mocking needed.
    Earth-ephemeris accuracy error → < 2.6e-5 in u → < 5e-4 in magnification,
    well within the 1e-3 tolerance.
    """
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    from astropy.time import Time
    import astropy.units as u_ast

    solar_system_ephemeris.set('builtin')

    t0 = 2460025.0
    t_vals = np.linspace(t0 - 1.0, t0 + 1.0, 10)

    def _earth(t_arr):
        return (get_body_barycentric('earth', Time(t_arr, format='jd', scale='tdb'))
                .xyz.to(u_ast.au).value.T)   # (N, 3)

    earth_xyz = _earth(t_vals)
    satellite_offset = np.array([0.04, 0.03, 0.01])
    xyz_abs = earth_xyz + satellite_offset[np.newaxis, :]

    dt = 0.5
    earth_pos_ref = _earth(np.array([t0]))[0]
    earth_vel_ref = (_earth(np.array([t0 + dt]))[0]
                     - _earth(np.array([t0 - dt]))[0]) / (2.0 * dt)

    extra = {
        "lens.Lens.pi_E_N": {"initval": 0.3, "sigma": 0.0},
        "lens.Lens.pi_E_E": {"initval": 0.2, "sigma": 0.0},
        "star.Lens.pm_ra":  {"initval": 10.0},
        "star.Lens.pm_dec": {"initval": 5.0},
    }
    system, model = _build_system(extra)

    class _MockInstr:
        pass
    instr = _MockInstr()
    instr._t0_par        = t0
    instr._earth_pos_ref = earth_pos_ref
    instr._earth_vel_ref = earth_vel_ref
    system.mulensinstrument = instr

    m_sym, m_op = _eval_both(system, model, xyz_abs, t_vals)

    max_diff = np.max(np.abs(m_sym - m_op))
    assert max_diff < 1e-3, (
        f"max |A_sym - A_op| with satellite parallax = {max_diff:.2e}\n"
        "Possible ephemeris-convention mismatch: check that both paths "
        "receive absolute barycentric AU and convert consistently."
    )
