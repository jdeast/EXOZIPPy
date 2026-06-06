import sympy as sp
import numpy as np
from ...constants import LOGG_CONST, KEPLER_CONST, G
import inspect

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (mass > 0, radius > 0) are enforced downstream by defaults.yaml
star_radius, star_mass = sp.symbols('star_radius star_mass', real=True)
mass, radius = sp.symbols('mass radius', real=True)
p, ar = sp.symbols('p ar', real=True)
density = sp.symbols('density', real=True)

# Log parameters
logg = sp.symbols('logg', real=True)
ecc = sp.symbols('ecc', real=True)

K, arsun, sini, period, m_total = sp.symbols('K arsun sini period m_total', real=True)

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
        "radius": "radius",
        "density": "density",
        "logg": "logg",
        "p": "p",
        "ar": "ar",
        "K": "K",
        "arsun": "arsun",
        "m_total": "m_total",

        # Cross-Component Bridges:
        "sini": f"orbit.{orbit_idx}.sini",
        "period": f"orbit.{orbit_idx}.period",
        "ecc": f"orbit.{orbit_idx}.ecc",
        "star_mass": f"star.{star_idx}.mass",
        "star_radius": f"star.{star_idx}.radius"
    }

# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
ONE = sp.Integer(1)
TWO = sp.Integer(2)
THREE = sp.Integer(3)
Gsym = sp.Rational(int(round(G*1e10)),10000000000)

RELATIONS = [
    # Bulk Density (rho \propto M / R^3)
    sp.Eq(density, mass / (radius ** THREE)),

    # Surface Gravity in cgs (g = G * M / R^2)
    #sp.Eq(logg, LOGG_CONST + sp.log(mass, 10) - 2.0 * sp.log(radius, 10)),

    sp.Eq(p, radius / star_radius), # You'll need to define star_radius as a symbol!

    # RV semi-amplitude
    #sp.Eq(m_total, star_mass + mass),
    #sp.Eq(arsun, KEPLER_CONST * (m_total ** (1.0/3.0)) * (period ** (2.0/3.0))),
    #sp.Eq(K, (2.0 * sp.pi * sini * arsun * mass) /
    #             (period * m_total * sp.sqrt(1.0 - ecc ** 2))),
    #sp.Eq(K,((2.0*sp.pi*G)/(period*(star_mass + mass)**2))**(1.0/3.0)*mass*sini/(sp.sqrt(1.0-ecc**2)))
    #sp.Eq(mass, sp.Symbol('mass_check_sentinel')) # this triggers our custom solver for K->mass
    sp.Eq(K, (
        ((TWO * sp.pi * Gsym) / (period * (star_mass + mass)**TWO))**sp.Rational(1, 3)
    ) * mass * sini / sp.sqrt(ONE - ecc**TWO))

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
            "primary_mass": resolved.get(f"star.{s_idx}.mass")
        }

        # 3. If any dependency hasn't been solved yet, abort and try later
        if any(v is None for v in deps.values()):
            raise KeyError("Missing dependencies for companion mass solver")

        return float(solve_companion_mass(**deps))

    config_manager.register_custom_solver("planet.mass", solver_wrapper)

def solve_companion_mass(K, ecc, sini, period, primary_mass):
    # K in radSol/day (m/s?)
    # period in days
    # primary_mass in solMass

    # Constants (IDL defaults)
    cubert2 = 1.25992104989487319

    x = period / (2.0 * np.pi * G) * (K * np.sqrt(1.0 - ecc ** 2) / sini) ** 3
    x2 = x ** 2
    x3 = x ** 3
    m12 = primary_mass ** 2
    m13 = m12 * primary_mass
    m14 = m12 ** 2

    # The IDL analytic solution
    y = (27.0 * m12 * x + np.sqrt(729.0 * m14 * x2 + 108.0 * m13 * x3) + 18.0 * primary_mass * x2 + 2.0 * x3) ** (1.0 / 3.0)
    companion_mass = y / (3.0 * cubert2) - cubert2 * (-6.0 * primary_mass * x - x2) / (3.0 * y) + x / 3.0

    return companion_mass  # Return in Msun

def get_solver_paths():
    """
    Returns the equations defining the physical state of a Planet.
    """
    return RELATIONS