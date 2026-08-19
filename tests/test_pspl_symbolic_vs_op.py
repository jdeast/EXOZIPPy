"""
Verify that the symbolic (PyTensor) Paczynski formula and the MulensModel
Op give the same magnification.

Note: get_magnification_op() routes point-source PSPL to the symbolic path
by default (NUTS-friendly, avoids _MagGradOp overhead).  These tests call
MulensMagOp directly to verify the Op implementation itself.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytestmark = pytest.mark.slow
import pymc as pm

from exozippy.components.mulensing.op import MulensMagOp
from exozippy.system import System

_BASE_USER_PARAMS = {
    "lens.Lens.t_0": {"initval": 2460025.0},
    "lens.Lens.u_0": {"initval": 0.2},
    "lens.Lens.pi_E_N": {"initval": 0.0, "sigma": 0.0},
    "lens.Lens.pi_E_E": {"initval": 0.0, "sigma": 0.0},
    "star.Lens.distance": {"initval": 4000.0},
    "star.Source.distance": {"initval": 8000.0},
    "star.Lens.mass": {"initval": 0.5},
    "star.Lens.pm_ra": {"initval": 0.0},
    "star.Lens.pm_dec": {"initval": 0.0},
    "star.Source.pm_ra": {"initval": 0.0},
    "star.Source.pm_dec": {"initval": 0.0},
    "star.Source.ra": {"initval": 266.4168},
    "star.Source.dec": {"initval": -29.0078},
    "star.Lens.ra": {"initval": 266.4168},
    "star.Lens.dec": {"initval": -29.0078},
}
_CONFIG = {
    "star": [{"name": "Lens"}, {"name": "Source"}],
    "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}],
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


def _eval_both(system, model, obs_dev, t_vals):
    """Compile and evaluate symbolic + Op magnification at raw=0 (initvals).

    obs_dev: Skowron+2011 geocentric deviations (AU) -- the single obs_pos
    convention both magnification paths consume.
    """
    with model:
        A_sym_node = system.lens.get_magnification(
            t_vals, obs_dev, system, index=0
        )

        sp = system.lens._get_safe_mm_params(0)
        mag_op = MulensMagOp(
            coords=_COORDS, mag_method="point_source", use_rho=False
        )
        A_op_node = mag_op(
            pt.stack(
                [
                    sp["t_0"],
                    sp["u_0"],
                    sp["t_E"],
                    sp["pi_E_N"],
                    sp["pi_E_E"],
                ]
            ),
            pt.as_tensor_variable(t_vals),
            pt.as_tensor_variable(obs_dev),
        )

        f_sym = pytensor.function(
            model.free_RVs, A_sym_node, on_unused_input="ignore"
        )
        f_op = pytensor.function(
            model.free_RVs, A_op_node, on_unused_input="ignore"
        )

        ip = model.initial_point()
        zero_in = [
            np.zeros_like(ip[v.name]).astype("float64") for v in model.free_RVs
        ]
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
    assert max_diff < 1e-4, (
        f"max |A_sym - A_op| (no parallax) = {max_diff:.2e}"
    )


def _skowron_deviations(t_vals, t0_par, offset=None):
    """Skowron+2011 geocentric deviations for an Earth(+offset) observer.

    delta(t) = earth(t) - [earth(t0_par) + v_earth(t0_par)*(t - t0_par)],
    using astropy's builtin ephemeris (no network).  ``offset`` adds a
    constant satellite displacement in AU.
    """
    import astropy.units as u_ast
    from astropy.coordinates import (
        get_body_barycentric,
        solar_system_ephemeris,
    )
    from astropy.time import Time

    solar_system_ephemeris.set("builtin")

    def _earth(t_arr):
        return (
            get_body_barycentric(
                "earth", Time(t_arr, format="jd", scale="tdb")
            )
            .xyz.to(u_ast.au)
            .value.T
        )  # (N, 3)

    dt = 0.5
    pos_ref = _earth(np.array([t0_par]))[0]
    vel_ref = (
        _earth(np.array([t0_par + dt]))[0] - _earth(np.array([t0_par - dt]))[0]
    ) / (2.0 * dt)
    dev = _earth(t_vals) - (
        pos_ref[None, :] + vel_ref[None, :] * (t_vals - t0_par)[:, None]
    )
    if offset is not None:
        dev = dev + np.asarray(offset)[None, :]
    return dev


def test_pspl_symbolic_vs_op_with_annual_parallax():
    """
    Given a full observing season of ground-based Skowron+2011 geocentric
      deviations (Earth's orbit departs from the linear reference by O(1) AU)
      and non-zero pi_E,
    When symbolic and Op are evaluated on the SAME deviation array,
    Then (a) they agree to < 1e-6 -- both paths consume one obs_pos
      convention -- and (b) the Op's magnification RESPONDS to the
      deviations (max |A - A_no_parallax| > 0.01).

    (b) is the regression for the 2026-08-08 review item 1.1: the Op used to
    subtract the ACTUAL Earth ephemeris from the observer positions, which
    deleted annual parallax entirely (pi_E had zero likelihood response for
    every ground-based Op-path fit).
    """
    t0 = 2460025.0
    t_vals = np.linspace(t0 - 150.0, t0 + 150.0, 200)
    dev = _skowron_deviations(t_vals, t0)

    # pi_E is derived from the pm/mass/distance chain (~0.18 here);
    # initvals on lens.pi_E_* would be ignored (derived parameter).
    extra = {
        "star.Lens.pm_ra": {"initval": 10.0},
        "star.Lens.pm_dec": {"initval": 5.0},
    }
    system, model = _build_system(extra)

    m_sym, m_op = _eval_both(system, model, dev, t_vals)

    max_diff = np.max(np.abs(m_sym - m_op))
    assert max_diff < 1e-6, (
        f"max |A_sym - A_op| with annual parallax = {max_diff:.2e}\n"
        "Possible obs_pos-convention mismatch: both paths must consume "
        "Skowron+2011 geocentric deviations."
    )

    zero_obs = np.zeros_like(dev)
    _, m_op_no_par = _eval_both(system, model, zero_obs, t_vals)
    response = np.max(np.abs(m_op - m_op_no_par))
    assert response > 0.01, (
        f"Op magnification ignores annual parallax: max |dA| = {response:.2e}"
    )


def test_pspl_symbolic_vs_op_with_satellite_offset():
    """
    Given annual deviations plus a constant ~0.05 AU satellite displacement,
    When symbolic and Op are evaluated on the same deviations,
    Then they agree to < 1e-6 and differ from the ground-only curve (the
      satellite term survives on top of the annual term).
    """
    t0 = 2460025.0
    t_vals = np.linspace(t0 - 30.0, t0 + 30.0, 60)
    dev_ground = _skowron_deviations(t_vals, t0)
    dev_sat = dev_ground + np.array([0.04, 0.03, 0.01])[None, :]

    # pm values give a finite t_E (~23 d) so the curve actually varies
    # over the window; pi_E is derived (~0.18) from the same chain.
    extra = {
        "star.Lens.pm_ra": {"initval": 10.0},
        "star.Lens.pm_dec": {"initval": 5.0},
    }
    system, model = _build_system(extra)

    m_sym, m_op = _eval_both(system, model, dev_sat, t_vals)
    max_diff = np.max(np.abs(m_sym - m_op))
    assert max_diff < 1e-6, (
        f"max |A_sym - A_op| with satellite offset = {max_diff:.2e}"
    )

    _, m_op_ground = _eval_both(system, model, dev_ground, t_vals)
    assert np.max(np.abs(m_op - m_op_ground)) > 1e-4, (
        "satellite displacement has no effect on the Op magnification"
    )
