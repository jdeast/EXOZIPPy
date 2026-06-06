import numpy as np
import pytensor
import pymc as pm

from exozippy.system import System

def test_pspl_symbolic_vs_op():
    """
    Verify that the symbolic (PyTensor) Paczynski formula and the MulensModel
    Op give the same magnification for pure PSPL (pi_E = 0, no parallax).

    The two methods use different obs_pos conventions — symbolic takes
    heliocentric deltas from a reference trajectory, while the Op takes
    absolute heliocentric XYZ — so parallax comparisons require careful
    coordinate alignment.  The cleanest shared ground truth is zero parallax:
    both formulas must then reproduce u(t) = sqrt(u0^2 + tau^2) exactly.
    """
    # --- ARRANGE ---
    t_vals = np.linspace(2460000.0, 2460050.0, 200)

    # Pure PSPL: pi_E = 0, so obs_pos zeros give identical trajectories in both
    # implementations regardless of sign/frame conventions.
    user_params = {
        "lens.Lens.t_0": {"initval": 2460025.0},
        "lens.Lens.u_0": {"initval": 0.05},
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

    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}]
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()

    # Zero observer offset -> no parallax for either formula
    zero_obs = np.zeros((len(t_vals), 3), dtype=np.float64)

    # --- ACT ---
    with model:
        A_symbolic_node = system.lens.get_magnification(t_vals, zero_obs, system, index=0)
        A_op_node = system.lens.get_magnification_op(t_vals, zero_obs, system, index=0)

        f_symbolic = pytensor.function(model.free_RVs, A_symbolic_node, on_unused_input='ignore')
        f_op = pytensor.function(model.free_RVs, A_op_node, on_unused_input='ignore')

        init_point = model.initial_point()
        zero_inputs = [np.zeros_like(init_point[v.name]).astype("float64") for v in model.free_RVs]

        m_symbolic = f_symbolic(*zero_inputs)
        m_op = f_op(*zero_inputs)

    # --- ASSERT ---
    # Both implement the same Paczynski formula; they should agree to 1e-4
    # (limited by MulensModel's internal floating-point handling).
    max_difference = np.max(np.abs(m_symbolic - m_op))
    assert max_difference < 1e-4, f"max |A_symbolic - A_op| = {max_difference:.2e}"
