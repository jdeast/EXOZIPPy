import sympy as sp

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (e.g., mass > 0) are enforced downstream by defaults.yaml
mass, radius = sp.symbols('mass radius', real=True)
density, surface_gravity = sp.symbols('density surface_gravity', real=True)

logmass, logradius = sp.symbols('logmass logradius', real=True)
logg = sp.symbols('logg', real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the component.

SYMBOL_MAP = {
    # Mass
    "mass": "mass",
    "logmass": "logmass",

    # Radius
    "radius": "radius",
    "logradius": "logradius",

    # Density & Gravity
    "density": "density",
    "surface_gravity": "surface_gravity",
    "logg": "logg"
}

# ---------------------------------------------------------
# 3. Physics Relations
# ---------------------------------------------------------
# NOTE ON UNITS:
# This assumes Mass and Radius are in Solar Units, and Density is relative to Solar.
# Surface gravity is calculated in cgs (cm/s^2) where G_sun ~ 27420 cm/s^2.
# If your celestial body uses Earth or Jupiter units, update the G_CGS constant!

G_CGS_SCALAR = 27420.0  # Conversion for Solar Mass / Solar Radius -> cgs

RELATIONS = [
    # Reparameterization Bridges (Base-10)
    sp.Eq(mass, 10 ** logmass),
    sp.Eq(radius, 10 ** logradius),

    # Bulk Density (rho \propto M / R^3)
    sp.Eq(density, mass / (radius ** 3)),

    # Surface Gravity in cgs (g = G * M / R^2)
    sp.Eq(surface_gravity, G_CGS_SCALAR * mass / (radius ** 2)),

    # Logg Bridge
    sp.Eq(surface_gravity, 10 ** logg)
]


def get_solver_paths():
    """
    Returns the equations defining the physical state of a generic Celestial Body.
    """
    return RELATIONS