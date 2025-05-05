import numpy as np

# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def mkconstants():
    """
    Returns a dictionary of physical constants in CGS units, consistent with EXOFASTv2.
    Values are based on IAU 2012/B2/B3 resolutions.

    Returns
    -------
    constants : dict
        Dictionary containing physical constants in CGS units.
    """
    pi = np.pi

    constants = {
        # Fundamental constants
        'G': 6.67408e-8,             # Gravitational constant (cm^3/g/s^2)
        'c': 2.99792458e10,          # Speed of light in vacuum (cm/s)
        'sigmab': 5.670367e-5,       # Stefan-Boltzmann constant (erg/s/cm^2/K^4)

        # Stellar and planetary constants
        'RSun': 6.957e10,            # Solar radius (cm)
        'LSun': 3.828e33,            # Solar luminosity (erg/s)
        'GMsun': 1.3271244e26,       # GM of Sun (cm^3/s^2)
        'GMearth': 3.986004e20,      # GM of Earth (cm^3/s^2)
        'GMjupiter': 1.2668653e23,   # GM of Jupiter (cm^3/s^2)
        'REarth': 6.3781e8,          # Earth equatorial radius (cm)
        'RJupiter': 7.1492e9,        # Jupiter equatorial radius (cm)

        # Unit conversions
        'AU': 1.495978707e13,        # Astronomical unit (cm)
        'day': 86400.0,              # Day (s)
        'meter': 100.0,              # Meter (cm)
    }

    # Derived constants
    constants['RhoSun'] = (
        3.0 * constants['GMsun'] / constants['G'] / constants['RSun']**3 / (4.0 * pi)
    )
    constants['GravitySun'] = constants['GMsun'] / constants['RSun']**2
    constants['pc'] = constants['AU'] * 3600.0 * 180.0 / pi

    return constants
