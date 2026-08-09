import math

import astropy.constants as const
import astropy.units as u
import numpy as np

# --- 1. PHYSICAL CONSTANTS (Internal Float System: R_sun, M_sun, Day) ---
G = const.G.to(u.R_sun**3 / (u.M_sun * u.day**2)).value

RSUN_TO_AU = (1.0 * u.R_sun).to(u.au).value
KEPLER_CONST = (G / (4.0 * np.pi**2)) ** (1.0 / 3.0)
LOGG_CONST = np.log10(const.GM_sun.cgs.value / const.R_sun.cgs.value**2)  # cgs
LUM_CONST = 1.0 / (
    (const.L_sun / const.sigma_sb / const.R_sun**2).cgs.value / (4.0 * np.pi)
)  # K^-4
FBOL_CONST = 1.0 / (4.0 * np.pi * (const.pc / const.R_sun) ** 2.0)
DENSITY_CONST = 3.0 / (4.0 * np.pi)
FROM_PM_D_TO_V = u.au.to(u.km) / u.yr.to(u.s)  # = 4.74, for unit conversion:
# multiply it by proper motion [mas/yr] and distance [kpc] to get velocity [km/s]
FROM_V_D_TO_PM = 1.0 / FROM_PM_D_TO_V  # = 0.211 - opposite unit conversion:
# multiply it by velocity [km/s] and divide by distance [kpc] to get proper motion [mas/yr]

PC_TO_RSUN_CONST = u.pc.to(u.R_sun)
ANG_TO_MICRON_CONST = u.Angstrom.to(u.micron)

# Internal planet units are solar; the Chen & Kipping relation is in Earth
# units.  IAU 2015 Resolution B3 nominal values via astropy.
MSUN_TO_MEARTH = (1.0 * u.M_sun).to(u.M_earth).value
RSUN_TO_REARTH = (1.0 * u.R_sun).to(u.R_earth).value

# --- 4. MICROLENSING CONSTANTS ---
# Kappa: 4G / (c^2 * au) in units of mas / M_sun
KAPPA = (
    (4.0 * const.G * const.M_sun / (const.c**2 * const.au))
    .to(u.mas, equivalencies=u.dimensionless_angles())
    .value
)

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
BULGE_BAR_ANGLE = np.radians(25.0)  # bar axis relative to Sun direction
BULGE_DENSITY_X_0 = 1.590  # in kpc, bulge density axis X in kpc from Zhu+17
BULGE_DENSITY_Y_0 = 0.424  # in kpc, bulge density axis Y in kpc from Zhu+17
BULGE_DENSITY_Z_0 = 0.424  # in kpc, bulge density axis Z in kpc from Zhu+17
BULGE_GAMMA = -2.0  # see Koshimoto and Bennett 2020 Sec. 3.4
# Outer cylindrical cutoff of the bar (genulens: Rc, srob).  Beyond
# R = BULGE_RC the bar density is multiplied by a Gaussian of width
# BULGE_RC_WIDTH in R -- without it the shallow exp(-r_s/2) profile
# leaks bulge stars all the way to the Sun (0.9% of central at R0).
BULGE_RC = 2.632  # in kpc, Koshimoto+ 2021 E-model Rc (genulens rc)
BULGE_RC_WIDTH = 0.5  # in kpc (genulens srob)
BULGE_VELOCITY_SIGMA_1 = (
    120.0  # in km/s, basedon on Koshimoto & Bennett 2020 tab. 1
)
BULGE_VELOCITY_SIGMA_2 = (
    100.0  # in km/s, basedon on Koshimoto & Bennett 2020 tab. 1
)
BULGE_VELOCITY_SIGMA_3 = (
    80.0  # in km/s, basedon on Koshimoto & Bennett 2020 tab. 1
)
BULGE_ROTATION_ANGULAR_VELOCITY = (
    50.0  # in km/s/kpc, basedon on Koshimoto & Bennett 2020 tab. 1
)

# --- 6. DISK CONSTANTS ---
# Thin disk structure from Koshimoto, Bennett & Suzuki 2021 (genulens):
# Rd = 2.6 kpc with the density held CONSTANT inside R = 5.3 kpc (their
# DISK=2 "hole"/plateau -- the inner disk does not keep rising toward
# the GC), vertical exp with 325 pc (their age bins span 61-445 pc
# sech^2; one exp layer is our simplification).
DISK_SCALE_LENGTH = 2.6  # in kpc, thin disk Rd (genulens Rd[1])
DISK_RDBREAK = 5.3  # in kpc, density flat inside this R (genulens Rdbreak)
DISK_SCALE_HEIGHT = 0.325  # in kpc, thin disk vertical scale
# Thick disk (genulens Rd[2], zd[7]): exp in both R (same plateau) and z.
THICK_DISK_SCALE_LENGTH = 2.2  # in kpc
THICK_DISK_SCALE_HEIGHT = 0.903  # in kpc
# Disk kinematics: the analytic (non-Shu) branch genulens itself provides
# (its B14disk mode, Bennett et al. 2014): mean rotation and fixed
# dispersions per component.  These replace the old "rough guess"
# (220; 30,30,30).
DISK_ROTATION_VELOCITY = 218.0  # in km/s, thin disk mean v_phi
DISK_VELOCITY_SIGMA_U = 39.9  # in km/s, thin disk radial
DISK_VELOCITY_SIGMA_V = 27.9  # in km/s, thin disk azimuthal
DISK_VELOCITY_SIGMA_W = 19.1  # in km/s, thin disk vertical
THICK_DISK_ROTATION_VELOCITY = 170.0  # in km/s (asymmetric drift included)
THICK_DISK_VELOCITY_SIGMA_U = 67.0  # in km/s
THICK_DISK_VELOCITY_SIGMA_V = 51.0  # in km/s
THICK_DISK_VELOCITY_SIGMA_W = 42.0  # in km/s
KROUPA_IMF_SLOPE = -1.3  # Kroupa IMF (mass range typical for lenses)
SALPETER_IMF_SLOPE = -2.35  # Salpeter IMF

# --- 7. SUN CONSTANTS ---
SUN_GC_DISTANCE = 8.16  # in kpc (genulens/Koshimoto+21 R0 = 8160 pc)
SUN_Z_OFFSET = 0.025  # in kpc, Sun's height above the plane (genulens zsun)
# Solar velocity in the galactocentric frame, genulens convention
# (vxsun toward the GC, vysun in the rotation direction, vzsun up):
SUN_GALCEN_V = (10.0, 243.0, 7.0)  # in km/s
SUN_VELOCITY_X = -12.7  # in km/s (legacy, rp.py convention)
SUN_VELOCITY_Y = 24.0 + DISK_ROTATION_VELOCITY  # in km/s (legacy)
SUN_VELOCITY_Z = 7.25  # in km/s (legacy)

# --- 8. GALACTIC POPULATION NUMBER DENSITIES (stars/pc^3, MS+BD) ---
# Branch weights for the disk/thick/bulge mixture in
# components/galacticmodel.  These are genulens's own NUMBER-density
# channel (n0MS* in its run_context.hpp) -- the channel it uses to decide
# which population a star belongs to -- so no mean-stellar-mass caveat
# applies.  Only ratios matter for the mixture; the absolute scale is
# arbitrary but kept physical for auditability.
#
# Thin disk local (R0, midplane): sum of the 7 thin-disk age bins' n0MSd.
DISK_LOCAL_NUMBER_DENSITY = 0.1633
# Thick disk local: n0MSd[7].
THICK_DISK_LOCAL_NUMBER_DENSITY = 7.91e-3
# Bar central: derived with genulens's VVV-box budget applied to OUR bar
# profile (exp(-r_s/2), Zhu+17 axes, BULGE_RC cutoff):
#   rho0b = (frho0b * M_VVV(P17) - M_disk_in_box) / int_box(profile)
#         = (0.839015 * 1.32e10 - 1.312e9) / 9.307e9 pc^3
#         = 1.049 Msun/pc^3   (Portail+ 2017 box |xb|<2.2,|yb|<1.4,|z|<1.2)
#   n0   = rho0b * fb_MS * m2nb_MS = rho0b * (1.62/2.07) / 0.227943
# Pinned by tests/test_galactic_model.py, which recomputes the integral.
BULGE_CENTRAL_NUMBER_DENSITY = 3.60

# IAU 2015, Resolution B2 zero point values
LSUN = 1 * u.Lsun
M_BOL_SUN = LSUN.to(u.M_bol).value
L0 = LSUN / 10 ** (-0.4 * M_BOL_SUN)
LOG_L0_CONST = 2.5 * np.log10(L0.value)
