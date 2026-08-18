import numpy as np
import sympy as sp

from ...constants import KEPLER_CONST, G

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# Only symbols that appear in RELATIONS below are declared: ConfigManager
# binds a relation's free_symbols to the paths get_symbol_map names, by
# symbol NAME, so a declared-but-unused symbol is inert.  (The map itself is
# wider than this list on purpose -- every mapped path becomes a leaf symbol
# in the relaxation engine whether or not a relation mentions it.)

# All parameters are strictly real.
# Positivity bounds (mass > 0, radius > 0) are enforced downstream by defaults.yaml
star_radius, star_mass = sp.symbols("star_radius star_mass", real=True)
mass, radius = sp.symbols("mass radius", real=True)
p = sp.symbols("p", real=True)
density = sp.symbols("density", real=True)

# Log parameters
log_q = sp.symbols("log_q", real=True)
ecc = sp.symbols("ecc", real=True)

K, sini, period = sp.symbols("K sini period", real=True)
a, ar, m_total = sp.symbols("a ar m_total", real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the Planet component.

comp_key = "planet"


def get_symbol_map(config):
    # Grab the indices to know WHICH star and orbit this planet belongs to
    star_idx = config.get("star_ndx", 0)
    orbit_idx = config.get("orbit_ndx", 0)

    return {
        "mass": "mass",
        "log_q": "log_q",
        "radius": "radius",
        "density": "density",
        "logg": "logg",
        "p": "p",
        "ar": "ar",
        "K": "K",
        "a": "a",
        "m_total": "m_total",
        # Cross-Component Bridges:
        "sini": f"orbit.{orbit_idx}.sini",
        "period": f"orbit.{orbit_idx}.period",
        "ecc": f"orbit.{orbit_idx}.ecc",
        "star_mass": f"star.{star_idx}.mass",
        "star_radius": f"star.{star_idx}.radius",
    }


# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
ONE = sp.Integer(1)
TWO = sp.Integer(2)
THREE = sp.Integer(3)
Gsym = sp.Rational(int(round(G * 1e10)), 10000000000)

RELATIONS = [
    # Mass ratio to the host star.  In log_q mode this back-solves a user's
    # planet.<i>.mass initval (or the K -> mass custom solver's result) into a
    # log_q start.  In linear mode log_q is never materialized, and because it
    # carries the lowest rank in this relation (planet/defaults.yaml) it always
    # absorbs the residual, so the relation cannot perturb mass or star_mass.
    sp.Eq(mass, (10**log_q) * star_mass),
    # Bulk Density (rho \propto M / R^3)
    sp.Eq(density, mass / (radius**THREE)),
    # Radius ratio
    sp.Eq(p, radius / star_radius),
    # RV semi-amplitude.  planet.logg has no relation here on purpose:
    # star/symbolic_physics.py owns the logg bridge, and the runtime value
    # comes from planet/physics.py's calc_logg_from_mass.
    sp.Eq(
        K,
        (
            ((TWO * sp.pi * Gsym) / (period * (star_mass + mass) ** TWO))
            ** sp.Rational(1, 3)
        )
        * mass
        * sini
        / sp.sqrt(ONE - ecc**TWO),
    ),
    # ---- The scaled semi-major axis, and what it unlocks -------------------
    # These three say what planet/physics.py's calc_m_total, calc_arsun and
    # calc_arstar compute, and they exist because NOTHING resolved a/R* before
    # them.  That gap was worked around twice: Planet._initial_semimajor_axes
    # recomputed the start by hand for the crossing barrier, and the transit
    # chord (below) could not be bridged at all, since its inversion needs
    # a/R*.  One definition in the engine replaces both workarounds.
    #
    # `a`, `ar` and `m_total` carry rank 5 in defaults.yaml so Condition B
    # always rewrites THEM rather than the mass, radius or period they are
    # derived from.  That is not a preference, it is the only correct
    # direction: all three are derived Parameters whose runtime value is their
    # expression, so a seed on one cannot survive into the model anyway -- and
    # letting a stale `planet.ar` entry silently move a period would be the
    # exact inversion the ranking system exists to prevent.
    #
    # They also make the engine FASTER, which is not the reason for them but is
    # worth knowing: resolving a/R* and (through it) esinw early takes an
    # iteration out of the relaxation loop.  Measured on examples/kelt17,
    # prepare() went from 2.8 s to 2.0 s.
    #
    # The transit chord is deliberately NOT bridged by a relation here.  It
    # would be the natural next line -- chord^2 + b^2 = (1 + p)^2, now that
    # a/R* resolves -- and it costs 7.6 s of sympy per System (2.0 s -> 9.6 s
    # on the same example), because the equation is quadratic in the chord and
    # transcendental in omega, and the engine attempts every unresolved symbol
    # in it.  orbit/symbolic_physics.py registers a ten-line solver instead,
    # for the same reason planet.log_q has one: "so the inversion never reaches
    # sp.solve".
    sp.Eq(m_total, mass + star_mass),
    sp.Eq(a**THREE, (KEPLER_CONST**THREE) * m_total * (period**TWO)),
    sp.Eq(ar, a / star_radius),
]


def register_solvers(config_manager):
    def solver_wrapper(resolved, system_config, index):
        # 1. Look up the logical mappings directly from the YAML dictionary
        planet_cfgs = system_config.get("planet", [{}])
        p_cfg = planet_cfgs[index] if index < len(planet_cfgs) else {}

        o_idx = p_cfg.get("orbit_ndx", 0)
        s_idx = p_cfg.get("star_ndx", 0)

        # 2. Fetch the required float values using the exact absolute paths
        deps = {
            "K": resolved.get(f"planet.{index}.K"),
            "ecc": resolved.get(f"orbit.{o_idx}.ecc"),
            "sini": resolved.get(f"orbit.{o_idx}.sini"),
            "period": resolved.get(f"orbit.{o_idx}.period"),
            "primary_mass": resolved.get(f"star.{s_idx}.mass"),
        }

        # 3. If any dependency hasn't been solved yet, abort and try later
        if any(v is None for v in deps.values()):
            raise KeyError("Missing dependencies for companion mass solver")

        return float(solve_companion_mass(**deps))

    config_manager.register_custom_solver("planet.mass", solver_wrapper)

    def log_q_wrapper(resolved, system_config, index):
        """log_q from the planet mass and its host mass.

        Registered so the transcendental inversion of
        Eq(mass, 10**log_q * star_mass) never reaches sp.solve (2 s alarm).
        A non-positive mass is not representable in this coordinate; abort
        (KeyError is the documented "cannot solve yet" signal) and leave the
        defaults.yaml start in place.  That case only arises in linear mode,
        where the value is unused -- a log_q-mode planet with a negative mass
        was already rejected by Planet._reconcile_mass_user_params.
        """
        planet_cfgs = system_config.get("planet", [{}])
        p_cfg = planet_cfgs[index] if index < len(planet_cfgs) else {}
        s_idx = p_cfg.get("star_ndx", 0)

        mass = resolved.get(f"planet.{index}.mass")
        star_mass = resolved.get(f"star.{s_idx}.mass")
        if mass is None or star_mass is None:
            raise KeyError("Missing dependencies for log_q solver")
        if mass <= 0.0 or star_mass <= 0.0:
            raise KeyError(
                f"planet.{index}.mass = {mass} is not representable as log_q"
            )

        return float(np.log10(mass / star_mass))

    config_manager.register_custom_solver("planet.log_q", log_q_wrapper)


def solve_companion_mass(K, ecc, sini, period, primary_mass):
    # K in radSol/day (m/s?)
    # period in days
    # primary_mass in solMass

    # Constants (IDL defaults)
    cubert2 = 1.25992104989487319

    x = period / (2.0 * np.pi * G) * (K * np.sqrt(1.0 - ecc**2) / sini) ** 3
    x2 = x**2
    x3 = x**3
    m12 = primary_mass**2
    m13 = m12 * primary_mass
    m14 = m12**2

    # The IDL analytic solution
    y = (
        27.0 * m12 * x
        + np.sqrt(729.0 * m14 * x2 + 108.0 * m13 * x3)
        + 18.0 * primary_mass * x2
        + 2.0 * x3
    ) ** (1.0 / 3.0)
    companion_mass = (
        y / (3.0 * cubert2)
        - cubert2 * (-6.0 * primary_mass * x - x2) / (3.0 * y)
        + x / 3.0
    )

    return companion_mass  # Return in Msun
