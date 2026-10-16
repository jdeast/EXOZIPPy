import sympy as sp

from ...constants import LOGG_CONST

# ---------------------------------------------------------
# 1. Define Symbols
# ---------------------------------------------------------

# All parameters are strictly real.
# Positivity bounds (e.g., mass > 0, teff > 0) are enforced downstream by defaults.yaml
mass, radius, luminosity = sp.symbols("mass radius luminosity", real=True)
teff, density, logg, feh = sp.symbols("teff density logg feh", real=True)
distance, parallax = sp.symbols("distance parallax", real=True)

# Log parameters
logmass = sp.symbols("logmass", real=True)

# Astrometry
pm_ra, pm_dec = sp.symbols("pm_ra pm_dec", real=True)
ra, dec = sp.symbols("ra dec", real=True)

# ---------------------------------------------------------
# 2. Symbol Map
# ---------------------------------------------------------
# Maps SymPy symbols back to the local parameter keys inside the Star component.

comp_key = "star"


def get_symbol_map(star_config_list, system_config):
    return {
        f"logmass": f"logmass",
        f"mass": f"mass",
        f"radius": f"radius",
        f"density": f"density",
        f"logg": "logg",
        f"luminosity": f"luminosity",
        f"teff": f"teff",
        f"feh": f"feh",
        f"distance": f"distance",
        f"parallax": f"parallax",
        f"rv": f"rv",
        f"ra": f"ra",
        f"dec": f"dec",
        f"pm_ra": f"pm_ra",
        f"pm_dec": f"pm_dec",
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
    sp.Eq(mass, 10**logmass),
    # Stellar density in the internal solMass/solRad3: the MEAN density of a
    # sphere, rho = 3 M / (4 pi R^3), exactly as physics.calc_density (and
    # so the model) computes it.  This read M / R^3 until review 3.14.22, so
    # a pinned density reached the start 4 pi / 3 = 4.19x off (its first
    # test, hat3_mist with a pinned density, measured 0.62 g/cm3 built for
    # 2.6 requested, on master and on the new engine alike).
    sp.Eq(density, sp.Rational(3, 4) / sp.pi * mass / (radius**3)),
    # Surface Gravity in cgs (g = G * M / R^2)
    sp.Eq(logg, LOGG_CONST + logmass - 2.0 * sp.log(radius, 10)),
    # Stefan-Boltzmann Law (Solar units: T_sun = 5772.0 K)
    # L = R^2 * (T / T_sun)^4
    sp.Eq(luminosity, (radius**2) * ((teff / 5772.0) ** 4)),
    # Astrometric Bridge
    sp.Eq(parallax, 1000.0 / distance),
]
