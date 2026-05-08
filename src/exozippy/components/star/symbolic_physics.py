import sympy as sp

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (e.g., mass > 0, teff > 0) are enforced downstream by defaults.yaml
mass, radius, luminosity = sp.symbols('mass radius luminosity', real=True)
teff, density = sp.symbols('teff density', real=True)
distance, parallax = sp.symbols('distance parallax', real=True)

# Log parameters
logmass, logradius, loglum = sp.symbols('logmass logradius loglum', real=True)

# Astrometry
pm_ra, pm_dec = sp.symbols('pm_ra pm_dec', real=True)
ra, dec = sp.symbols('ra dec', real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the Star component.

SYMBOL_MAP = {
    # Mass
    "mass": "mass",
    "logmass": "logmass",

    # Radius & Density
    "radius": "radius",
    "logradius": "logradius",
    "density": "density",

    # Luminosity & Temp
    "luminosity": "luminosity",
    "loglum": "loglum",
    "teff": "teff",

    # Astrometry
    "distance": "distance",
    "parallax": "parallax",
    "pm_ra": "pm_ra",
    "pm_dec": "pm_dec",
    "ra": "ra",
    "dec": "dec"
}

# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
# Standard unit assumptions:
# Mass/Radius/Lum/Density in Solar Units.
# Teff in Kelvin.
# Distance in pc, Parallax in mas.

RELATIONS = [
    # Reparameterization Bridges (Base-10)
    sp.Eq(mass, 10 ** logmass),
    sp.Eq(radius, 10 ** logradius),
    sp.Eq(luminosity, 10 ** loglum),

    # Stellar Density (Solar units: rho_sun = 1.0)
    # rho = M / R^3
    sp.Eq(density, mass / (radius ** 3)),

    # Stefan-Boltzmann Law (Solar units: T_sun = 5772.0 K)
    # L = R^2 * (T / T_sun)^4
    sp.Eq(luminosity, (radius ** 2) * ((teff / 5772.0) ** 4)),

    # Astrometric Bridge
    sp.Eq(parallax, 1000.0 / distance)
]


def get_solver_paths():
    """
    Returns the equations defining the physical state of a Star.
    """
    return RELATIONS