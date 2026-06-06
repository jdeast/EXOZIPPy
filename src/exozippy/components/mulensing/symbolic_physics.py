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

comp_key = "lens"
def get_symbol_map(lens_config_list):
    """
    Dynamically maps SymPy symbols to YAML paths based on
    which star indices are lens/source.
    """
    # Grab the first lens configuration (for PSPL)

    l_idx = lens_config_list.get("lens_ndx", 0)
    s_idx = lens_config_list.get("source_ndx", 1)

    return {
        f"t_0": f"t_0",
        f"u_0": f"u_0",
        f"t_E": f"t_E",
        f"rho": f"rho",

        f"theta_E": f"theta_E",
        f"pi_rel": f"pi_rel",
        f"pi_E_N": f"pi_E_N",
        f"pi_E_E": f"pi_E_E",

        f"mu_rel_mag": f"mu_rel_mag",
        f"mu_ra_rel": f"mu_ra_rel",
        f"mu_dec_rel": f"mu_dec_rel",

        "lens_mass": f"star.{l_idx}.mass",
        "lens_distance": f"star.{l_idx}.distance",
        "lens_pm_ra": f"star.{l_idx}.pm_ra",
        "lens_pm_dec": f"star.{l_idx}.pm_dec",
        "lens_ra": f"star.{l_idx}.ra",
        "lens_dec": f"star.{l_idx}.dec",

        "source_mass": f"star.{s_idx}.mass",
        "source_distance": f"star.{s_idx}.distance",
        "source_pm_ra": f"star.{s_idx}.pm_ra",
        "source_pm_dec": f"star.{s_idx}.pm_dec",
        "source_ra": f"star.{s_idx}.ra",
        "source_dec": f"star.{s_idx}.dec",
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

    # Derived shortcut: pi_rel = kappa * mass * |pi_E|^2
    # (obtained by eliminating theta_E from the Einstein-radius and pi_E-magnitude
    # equations: |pi_E|^2 = (pi_rel/theta_E)^2 and theta_E^2 = kappa*mass*pi_rel).
    # This gives the solver a direct rank-100 path when mass and pi_E are both
    # user-supplied, bypassing the distance hint and avoiding sign ambiguity in
    # the quadratic for mu_ra_rel / mu_dec_rel.
    sp.Eq(pi_rel, KAPPA * lens_mass * (pi_E_N ** 2 + pi_E_E ** 2)),

    # Finite Source (R_sun to AU, then to mas)
    sp.Eq(rho, ((source_radius * RSUN_TO_AU / source_distance) * 1000.0) / theta_E)
]


def get_solver_paths():
    """
    Optional: Pre-compiles common inversion formulas for speed.
    """
    return RELATIONS