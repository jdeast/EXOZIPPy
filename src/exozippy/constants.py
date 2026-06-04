import astropy.constants as const
import astropy.units as u
import numpy as np
import math

# --- 1. PHYSICAL CONSTANTS (Internal Float System: R_sun, M_sun, Day) ---
G = const.G.to(u.R_sun**3 / (u.M_sun * u.day**2)).value

RSUN_TO_AU = (1.0 * u.R_sun).to(u.au).value
KEPLER_CONST = (G/ (4.0 * np.pi ** 2))**(1.0/3.0)
LOGG_CONST = np.log10(const.GM_sun.cgs.value/const.R_sun.cgs.value**2) # cgs
LUM_CONST = 1.0/((const.L_sun/const.sigma_sb/const.R_sun**2).cgs.value/(4.0*np.pi)) # K^-4
FBOL_CONST = 1.0/(4.0 * np.pi * (const.pc/const.R_sun)** 2.0)
DENSITY_CONST = 3.0 / (4.0 * np.pi)
FROM_PM_D_TO_V = u.au.to(u.km) / u.yr.to(u.s) # = 4.74, for unit conversion:
# multiply it by proper motion [mas/yr] and distance [kpc] to get velocity [km/s]
FROM_V_D_TO_PM = 1. / FROM_PM_D_TO_V  # = 0.211 - opposite unit conversion:
# multiply it by velocity [km/s] and divide by distance [kpc] to get proper motion [mas/yr]

PC_TO_RSUN_CONST = u.pc.to(u.R_sun)
ANG_TO_MICRON_CONST = u.Angstrom.to(u.micron)

# --- 4. MICROLENSING CONSTANTS ---
# Kappa: 4G / (c^2 * au) in units of mas / M_sun
KAPPA = (4.0 * const.G * const.M_sun / (const.c**2 * const.au)).to(u.mas, equivalencies=u.dimensionless_angles()).value

# The standard proper motion conversion factor (m/s per mas/yr * pc)
K_VEL_CONVERSION = (const.au / u.yr).to(u.km / u.s).value

# --- 2. MATHEMATICAL CONSTANTS ---
PI = np.pi
TWOPI = 2.0 * np.pi


# --- 3. STATISTICAL CONSTANTS (For the Back-End) ---
# Used for 68% confidence intervals in tables and corner plots
SIGMA_1 = math.erf(1.0 / math.sqrt(2.0))
SIGMA_1_LOW = 0.5 - SIGMA_1 / 2.0
SIGMA_1_HIGH = 0.5 + SIGMA_1 / 2.0

# --- 5. BULGE CONSTANTS ---
BULGE_BAR_ANGLE = np.radians(25.)  # bar axis relative to Sun direction
BULGE_DENSITY_X_0 = 1.590  # in kpc, bulge density axis X in kpc from Zhu+17
BULGE_DENSITY_Y_0 = 0.424  # in kpc, bulge density axis Y in kpc from Zhu+17
BULGE_DENSITY_Z_0 = 0.424  # in kpc, bulge density axis Z in kpc from Zhu+17
BULGE_GAMMA = -2. # see Koshimoto and Bennett 2020 Sec. 3.4
BULGE_VELOCITY_SIGMA_1 = 120.  # in km/s, basedon on Koshimoto & Bennett 2020 tab. 1
BULGE_VELOCITY_SIGMA_2 = 100.  # in km/s, basedon on Koshimoto & Bennett 2020 tab. 1
BULGE_VELOCITY_SIGMA_3 = 80.  # in km/s, basedon on Koshimoto & Bennett 2020 tab. 1
BULGE_ROTATION_ANGULAR_VELOCITY = 50. # in km/s/kpc, basedon on Koshimoto & Bennett 2020 tab. 1

# --- 6. DISK CONSTANTS ---
DISK_SCALE_LENGTH = 3.5  # disk density scale length from Koshimoto & Bennett 2020; in kpc
DISK_SCALE_HEIGHT = 0.325  # disk density scale height from Koshimoto & Bennett 2020; in kpc
DISK_ROTATION_VELOCITY = 220.  # in km/s
DISK_VELOCITY_SIGMA_U = 30.  # in km/s, rough guess
DISK_VELOCITY_SIGMA_V = 30.  # in km/s, rough guess
DISK_VELOCITY_SIGMA_W = 30.  # in km/s, rough guess
KROUPA_IMF_SLOPE = -1.3  # Kroupa IMF (mass range typical for lenses)
SALPETER_IMF_SLOPE = -2.35  # Salpeter IMF

# --- 7. SUN CONSTANTS ---
SUN_GC_DISTANCE = 8.3
SUN_VELOCITY_X = -12.7  # in km/s
SUN_VELOCITY_Y = 24.0 + DISK_ROTATION_VELOCITY  # in km/s
SUN_VELOCITY_Z = 7.25   # in km/s

# IAU 2015, Resolution B2 zero point values
LSUN = 1 * u.Lsun
M_BOL_SUN = LSUN.to(u.M_bol).value
L0 = LSUN/10**(-0.4*M_BOL_SUN)
LOG_L0_CONST = 2.5 * np.log10(L0.value)