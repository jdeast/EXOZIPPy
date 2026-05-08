import sympy as sp

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (mass > 0, radius > 0) are enforced downstream by defaults.yaml
mass, radius = sp.symbols('mass radius', real=True)
density, surface_gravity = sp.symbols('density surface_gravity', real=True)

# Log parameters
logmass, logradius = sp.symbols('logmass logradius', real=True)
logg = sp.symbols('logg', real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the Planet component.

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
# You will need to set G_CGS_SCALAR based on the global units your framework uses for planets.
# If Mass and Radius are in Earth units: G_CGS_SCALAR ~ 982.0
# If Mass and Radius are in Jupiter units: G_CGS_SCALAR ~ 2479.0
# If Mass and Radius are in Solar units: G_CGS_SCALAR ~ 27420.0

G_CGS_SCALAR = 982.0  # Defaulting to Earth units for the scalar

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
    Returns the equations defining the physical state of a Planet.
    """
    return RELATIONS