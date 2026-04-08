import astropy.constants as const
import astropy.units as u
import numpy as np
import math

# --- 1. PHYSICAL CONSTANTS (Internal Float System: R_sun, M_sun, Day) ---
G = const.G.to(u.R_sun**3 / (u.M_sun * u.day**2)).value

KEPLER_CONST = (G/ (4.0 * np.pi ** 2))**(1.0/3.0)
LOGG_CONST = np.log10(const.GM_sun.cgs.value/const.R_sun.cgs.value**2) # cgs
LUM_CONST = 1.0/((const.L_sun/const.sigma_sb/const.R_sun**2).cgs.value/(4.0*np.pi)) # K^-4
FBOL_CONST = 1.0/(4.0 * np.pi * (const.pc/const.R_sun)** 2.0)
DENSITY_CONST = 3.0 / (4.0 * np.pi)

# --- 2. MATHEMATICAL CONSTANTS ---
PI = np.pi
TWOPI = 2.0 * np.pi


# --- 3. STATISTICAL CONSTANTS (For the Back-End) ---
# Used for 68% confidence intervals in tables and corner plots
SIGMA_1 = math.erf(1.0 / math.sqrt(2.0))
SIGMA_1_LOW = 0.5 - SIGMA_1 / 2.0
SIGMA_1_HIGH = 0.5 + SIGMA_1 / 2.0