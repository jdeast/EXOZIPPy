import numpy as np
import pytest
import pytensor
import pymc as pm
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris, get_body_barycentric_posvel
import erfa
from astropy.coordinates.builtin_frames.utils import get_jd12

from exozippy.system import System

def test_pspl_symbolic_vs_op():
    # --- ARRANGE ---
    t_vals = np.linspace(2460000.0, 2460050.0, 200)
    ra_rad = 266.4168 * (np.pi / 180.0)
    dec_rad = -29.0078 * (np.pi / 180.0)

    # Set physical parameters to force a high-parallax microlensing event
    user_params = {
        "lens.t_0": {"initval": 2460025.0},
        "lens.u_0": {"initval": 0.05},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.pm_ra": {"initval": -10.0},
        "star.Lens.pm_dec": {"initval": 5.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0},
        "star.Source.ra": {"initval": ra_rad},
        "star.Source.dec": {"initval": dec_rad},
        "star.Lens.ra": {"initval": ra_rad},
        "star.Lens.dec": {"initval": dec_rad},
    }

    # Two distinct stars to prevent dividing by zero!
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"lens_ndx": 0, "source_ndx": 1}]
    }

    system = System(config, user_params=user_params)
    model = system.build_model()

    solar_system_ephemeris.set('jpl')
    t_obj = Time(t_vals, format='jd', scale='tdb')
    earth_bary = get_body_barycentric('earth', t_obj).xyz
    sun_bary = get_body_barycentric('sun', t_obj).xyz
    obs_pos_heliocentric = (earth_bary - sun_bary).to('au').value.T

    # 2. Calculate the 3D Geocentric offset for the symbolic graph
    t0_val = 2460025.0
    t0_obj = Time(t0_val, format='jd', scale='tdb')

    earth_pos_ref, earth_vel_ref = get_body_barycentric_posvel('earth', t0_obj)
    sun_pos_ref, sun_vel_ref = get_body_barycentric_posvel('sun', t0_obj)

    pos_ref_val = (earth_pos_ref.xyz - sun_pos_ref.xyz).to('au').value
    velocity = (earth_vel_ref.xyz - sun_vel_ref.xyz).to('au/d').value  # AU/day

    # Heliocentric offset projection
    delta_s_3d = obs_pos_heliocentric - pos_ref_val - np.outer(t_vals - t0_val, velocity)

    # --- ACT ---
    with model:
        A_symbolic_node = system.lens.get_magnification(t_vals, delta_s_3d, system, index=0)
        A_op_node = system.lens.get_magnification_op(t_vals, np.zeros_like(obs_pos_heliocentric), system, index=0)

        # 1. Compile functions that accept the free parameters
        f_symbolic = pytensor.function(model.free_RVs, A_symbolic_node, on_unused_input='ignore')
        f_op = pytensor.function(model.free_RVs, A_op_node, on_unused_input='ignore')

        # 2. Feed the functions 0.0s to evaluate exactly at the initvals
        init_point = model.initial_point()
        zero_inputs = [np.zeros_like(init_point[v.name]).astype("float64") for v in model.free_RVs]

        m_symbolic = f_symbolic(*zero_inputs)
        m_op = f_op(*zero_inputs)

    # --- ASSERT ---
    max_difference = np.max(np.abs(m_symbolic - m_op))

    # there's a 10^-4 AU difference between my (JPL) ephemeris and MuLensModel's (analytic) ephemeris
    # which manifests as a 0.003 difference in the mag. That's totally fine
    assert max_difference < 1e-2