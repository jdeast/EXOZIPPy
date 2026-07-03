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
q_lens, companion_mass = sp.symbols('q_lens companion_mass')
alpha, xalpha, yalpha = sp.symbols('alpha xalpha yalpha')

comp_key = "lens"


def get_symbol_map(lens_config_list):
    """
    Dynamically maps SymPy symbols to YAML paths based on lens/source/companion
    body assignments.  Supports both the legacy lens_ndx/source_ndx keys and the
    NLNS lenses:/sources: list syntax.

    companion_mass is only added to the map for binary events (len(lenses) > 1).
    When absent, any RELATION that mentions companion_mass or q_lens will be
    skipped by the relaxation engine (all symbols must be in master_symbol_map).
    """
    companion_mass_path = None

    if "lenses" in lens_config_list:
        lenses = lens_config_list["lenses"]
        l_comp, l_idx = lenses[0].split(".")
        l_idx = int(l_idx)

        sources = lens_config_list.get("sources", ["star.1"])
        s_comp, s_idx = sources[0].split(".")
        s_idx = int(s_idx)

        if len(lenses) > 1:
            c_comp, c_idx = lenses[1].split(".")
            companion_mass_path = f"{c_comp}.{c_idx}.mass"
    else:
        l_idx = int(lens_config_list.get("lens_ndx", 0))
        s_idx = int(lens_config_list.get("source_ndx", 1))

    result = {
        "t_0": "t_0",
        "u_0": "u_0",
        "t_E": "t_E",
        "rho": "rho",
        "q_lens": "q",   # → lens.{i}.q after yaml_key prefix

        "theta_E": "theta_E",
        "pi_rel": "pi_rel",
        "pi_E_N": "pi_E_N",
        "pi_E_E": "pi_E_E",

        "mu_rel_mag": "mu_rel_mag",
        "mu_ra_rel": "mu_ra_rel",
        "mu_dec_rel": "mu_dec_rel",

        "alpha": "alpha",
        "xalpha": "xalpha",
        "yalpha": "yalpha",

        "lens_mass": f"star.{l_idx}.mass",
        "lens_distance": f"star.{l_idx}.distance",
        "lens_pm_ra": f"star.{l_idx}.pm_ra",
        "lens_pm_dec": f"star.{l_idx}.pm_dec",
        "lens_ra": f"star.{l_idx}.ra",
        "lens_dec": f"star.{l_idx}.dec",

        "source_mass": f"star.{s_idx}.mass",
        "source_radius": f"star.{s_idx}.radius",
        "source_distance": f"star.{s_idx}.distance",
        "source_pm_ra": f"star.{s_idx}.pm_ra",
        "source_pm_dec": f"star.{s_idx}.pm_dec",
        "source_ra": f"star.{s_idx}.ra",
        "source_dec": f"star.{s_idx}.dec",
    }

    if companion_mass_path:
        result["companion_mass"] = companion_mass_path

    return result

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
    sp.Eq(rho, ((source_radius * RSUN_TO_AU / source_distance) * 1000.0) / theta_E),

    # Binary lens mass ratio: q = M_companion / M_primary
    # companion_mass is only in the symbol map for binary events, so this relation
    # is automatically inert for PSPL (relaxation engine skips equations with
    # unregistered symbols).  Propagates: user-supplied q → companion mass initval,
    # or known masses → q for diagnostics.
    sp.Eq(q_lens * lens_mass, companion_mass),

    # Source trajectory angle: alpha (radians, internal) → xalpha, yalpha.
    # xalpha = r·cos(alpha), yalpha = r·sin(alpha), where r is a free positive
    # scale sampled from the N(0,1) prior — only the direction arctan2(y,x) matters.
    # Wide bounds (±100) and N(0,1) priors give a uniform marginal prior on alpha;
    # bounding to [-1,1] would break isotropy and bias angles near ±45°.
    # The relaxation engine uses these only forward (alpha → xalpha, yalpha):
    # given alpha, set xalpha=cos(alpha), yalpha=sin(alpha) as unit-circle seeds.
    # mkprior converts the sampled xalpha/yalpha back to alpha via arctan2.
    # alpha itself is not in the symbol map for PSPL events (no xalpha/yalpha
    # registered), so both relations are automatically inert for point-source fits.
    sp.Eq(xalpha, sp.cos(alpha)),
    sp.Eq(yalpha, sp.sin(alpha)),
]


def get_solver_paths():
    """
    Optional: Pre-compiles common inversion formulas for speed.
    """
    return RELATIONS