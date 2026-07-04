import sympy as sp
from ...constants import KAPPA, RSUN_TO_AU

# 1. Define all possible symbols
# These MUST match the strings produced by ConfigManager.finalize_user_params
t_0, u_0, t_E = sp.symbols('t_0 u_0 t_E')
theta_E, mu_rel_mag = sp.symbols('theta_E mu_rel_mag')
pi_rel = sp.symbols('pi_rel')
# lens_mass_total drives theta_E/t_E/rho/pi_E (community convention: binary-lens
# parameters are referenced to the TOTAL lens mass).  For single lenses it maps
# directly to the primary star's mass; for binaries it maps to lens.0.mlens_total
# and the mass-sum relation below ties it to the per-body masses.
lens_mass_total, primary_lens_mass = sp.symbols('lens_mass_total primary_lens_mass')
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

    Returns a LIST of symbol maps, one per source body (NSNL): each source has
    its own trajectory, so the per-source parameter chain (t_0, u_0, rho, t_E,
    theta_E, pi_rel, pi_E_*, mu_*) is instantiated once per source at the
    element-index paths lens.<j>.<param>, where j is the source's slot in the
    ``sources:`` list (matching element j of the lens component's vector
    parameters).  Lens-side and companion symbols are shared across all maps;
    ConfigManager dedupes the resulting identical relation instances.

    companion_mass is only added to the map for binary events (len(lenses) > 1).
    When absent, any RELATION that mentions companion_mass or q_lens will be
    skipped by the relaxation engine (all symbols must be in master_symbol_map).
    """
    companion_mass_path = None
    is_binary_lens = False

    if "lenses" in lens_config_list:
        lenses = lens_config_list["lenses"]
        l_comp, l_idx = lenses[0].split(".")
        l_idx = int(l_idx)

        sources = lens_config_list.get("sources", ["star.1"])

        if len(lenses) > 1:
            is_binary_lens = True
            c_comp, c_idx = lenses[1].split(".")
            companion_mass_path = f"{c_comp}.{c_idx}.mass"
    else:
        l_idx = int(lens_config_list.get("lens_ndx", 0))
        sources = [f"star.{int(lens_config_list.get('source_ndx', 1))}"]

    # theta_E/t_E/rho/pi_E are referenced to the TOTAL lens mass: the primary
    # star's mass for a single lens, the derived lens.0.mlens_total for a
    # binary (tied to the per-body masses by the mass-sum relation).
    if is_binary_lens:
        lens_mass_total_path = "lens.0.mlens_total"
    else:
        lens_mass_total_path = f"star.{l_idx}.mass"

    maps = []
    for j, src in enumerate(sources):
        s_comp, s_idx = src.split(".")
        s_idx = int(s_idx)

        result = {
            # Per-source trajectory chain: element j of the lens vector params.
            # Explicit full paths — only one lens event is allowed, so the
            # element index unambiguously identifies the source slot.
            "t_0": f"lens.{j}.t_0",
            "u_0": f"lens.{j}.u_0",
            "t_E": f"lens.{j}.t_E",
            "rho": f"lens.{j}.rho",

            "theta_E": f"lens.{j}.theta_E",
            "pi_rel": f"lens.{j}.pi_rel",
            "pi_E_N": f"lens.{j}.pi_E_N",
            "pi_E_E": f"lens.{j}.pi_E_E",

            "mu_rel_mag": f"lens.{j}.mu_rel_mag",
            "mu_ra_rel": f"lens.{j}.mu_ra_rel",
            "mu_dec_rel": f"lens.{j}.mu_dec_rel",

            # Shared per-companion geometry (companion slot 0)
            "q_lens": "lens.0.q",
            "alpha": "lens.0.alpha",
            "xalpha": "lens.0.xalpha",
            "yalpha": "lens.0.yalpha",

            "lens_mass_total": lens_mass_total_path,
            "lens_distance": f"star.{l_idx}.distance",
            "lens_pm_ra": f"star.{l_idx}.pm_ra",
            "lens_pm_dec": f"star.{l_idx}.pm_dec",
            "lens_ra": f"star.{l_idx}.ra",
            "lens_dec": f"star.{l_idx}.dec",

            "source_mass": f"{s_comp}.{s_idx}.mass",
            "source_radius": f"{s_comp}.{s_idx}.radius",
            "source_distance": f"{s_comp}.{s_idx}.distance",
            "source_pm_ra": f"{s_comp}.{s_idx}.pm_ra",
            "source_pm_dec": f"{s_comp}.{s_idx}.pm_dec",
            "source_ra": f"{s_comp}.{s_idx}.ra",
            "source_dec": f"{s_comp}.{s_idx}.dec",
        }

        if companion_mass_path:
            result["companion_mass"] = companion_mass_path
            result["primary_lens_mass"] = f"star.{l_idx}.mass"

        maps.append(result)

    return maps

RELATIONS = [
    # Einstein Radius (total lens mass)
    sp.Eq(theta_E ** 2, KAPPA * lens_mass_total * pi_rel),

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
    sp.Eq(pi_rel, KAPPA * lens_mass_total * (pi_E_N ** 2 + pi_E_E ** 2)),

    # Finite Source (R_sun to AU, then to mas)
    sp.Eq(rho, ((source_radius * RSUN_TO_AU / source_distance) * 1000.0) / theta_E),

    # Binary lens mass ratio: q = M_companion / M_primary
    # companion_mass/primary_lens_mass are only in the symbol map for binary
    # events, so these relations are automatically inert for PSPL (relaxation
    # engine skips equations with unregistered symbols).  Propagates:
    # user-supplied q → companion mass initval, or known masses → q.
    sp.Eq(q_lens * primary_lens_mass, companion_mass),

    # Total lens mass = sum of body masses (binary only; inert for PSPL where
    # lens_mass_total maps directly onto the primary star's mass).
    sp.Eq(lens_mass_total, primary_lens_mass + companion_mass),

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