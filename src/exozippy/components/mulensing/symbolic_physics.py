import sympy as sp
from ...constants import KAPPA, RSUN_TO_AU

# 1. Define all possible symbols
# These MUST match the strings produced by ConfigManager.finalize_user_params
t_0, u_0, t_E = sp.symbols('t_0 u_0 t_E')
theta_E, mu_rel_mag = sp.symbols('theta_E mu_rel_mag')
pi_rel, lens_mass = sp.symbols('pi_rel lens_mass')
lens_distance, source_distance = sp.symbols('lens_distance source_distance')
mu_ra_rel, mu_dec_rel = sp.symbols('mu_ra_rel mu_dec_rel')
lens_pm_ra, source_pm_ra = sp.symbols('lens_pm_ra source_pm_ra')
lens_pm_dec, source_pm_dec = sp.symbols('lens_pm_dec source_pm_dec')
pi_E_N, pi_E_E = sp.symbols('pi_E_N pi_E_E')
rho, source_radius = sp.symbols('rho source_radius')

# src/exozippy/components/mulensing/symbolic_physics.py

SYMBOL_MAP = {
    # Timing & Geometry
    "t_0": "lens.Lens.t_0",
    "u_0": "lens.Lens.u_0",
    "t_E": "lens.Lens.t_E",
    "rho": "lens.Lens.rho",

    # Einstein Radius & Parallax
    "theta_E": "lens.Lens.theta_E",
    "pi_rel": "lens.Lens.pi_rel",
    "pi_E_N": "lens.Lens.pi_E_N",
    "pi_E_E": "lens.Lens.pi_E_E",

    # Physical Properties (Mass/Distance)
    "lens_mass": "star.Lens.mass",
    "lens_distance": "star.Lens.distance",
    "source_distance": "star.Source.distance",
    "source_radius": "star.Source.radius",

    # Proper Motions (Vector Components)
    "lens_pm_ra": "star.Lens.pm_ra",
    "lens_pm_dec": "star.Lens.pm_dec",
    "source_pm_ra": "star.Source.pm_ra",
    "source_pm_dec": "star.Source.pm_dec",

    # Relative Motion (Magnitude and components used by the solver)
    "mu_rel_mag": "lens.Lens.mu_rel_mag",  # Mapping to the lens component parameter
    "mu_ra_rel": "lens.Lens.mu_ra_rel",
    "mu_dec_rel": "lens.Lens.mu_dec_rel"
}

RELATIONS = [
    # Einstein Radius
    sp.Eq(theta_E ** 2, KAPPA * lens_mass * pi_rel),

    # Relative Parallax (dist in pc -> pi in mas)
    sp.Eq(pi_rel, (1000 / lens_distance) - (1000 / source_distance)),

    # Einstein Time (mu in mas/yr -> t_E in days)
    sp.Eq(t_E, theta_E / (mu_rel_mag / 365.25)),

    # Relative Motion Magnitude
    sp.Eq(mu_rel_mag ** 2, mu_ra_rel ** 2 + mu_dec_rel ** 2),

    # Proper Motion Vector Components
    sp.Eq(mu_ra_rel, lens_pm_ra - source_pm_ra),
    sp.Eq(mu_dec_rel, lens_pm_dec - source_pm_dec),

    # Parallax Vector Components
    sp.Eq(pi_E_N, (pi_rel / theta_E) * (mu_dec_rel / mu_rel_mag)),
    sp.Eq(pi_E_E, (pi_rel / theta_E) * (mu_ra_rel / mu_rel_mag)),

    # Finite Source (R_sun to AU, then to mas)
    sp.Eq(rho, ((source_radius * RSUN_TO_AU / source_distance) * 1000.0) / theta_E)
]


def get_solver_paths():
    """
    Optional: Pre-compiles common inversion formulas for speed.
    """
    return RELATIONS