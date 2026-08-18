import math

import astropy.constants as const
import astropy.units as u
import numpy as np

# --- 1. PHYSICAL CONSTANTS (Internal Float System: R_sun, M_sun, Day) ---
G = const.G.to(u.R_sun**3 / (u.M_sun * u.day**2)).value

RSUN_TO_AU = (1.0 * u.R_sun).to(u.au).value
# Speed of light in EXOZIPPy's native internal unit system (R_sun, M_sun,
# day) -- NOT AU/day like exoplanet's own c_light, and NOT AU/s like
# EXOFASTv2 -- so light-travel-time delays (components/ltt.py) combine
# directly with orbit.arsun (already in R_sun) with no AU round-trip.
C_LIGHT_RSUN_PER_DAY = const.c.to(u.R_sun / u.day).value
KEPLER_CONST = (G / (4.0 * np.pi**2)) ** (1.0 / 3.0)
C_MPS = const.c.to(u.m / u.s).value
SOLRAD_PER_DAY_TO_MPS = (1.0 * u.R_sun / u.day).to(u.m / u.s).value
LOGG_CONST = np.log10(const.GM_sun.cgs.value / const.R_sun.cgs.value**2)  # cgs
LUM_CONST = 1.0 / (
    (const.L_sun / const.sigma_sb / const.R_sun**2).cgs.value / (4.0 * np.pi)
)  # K^-4
FBOL_CONST = 1.0 / (4.0 * np.pi * (const.pc / const.R_sun) ** 2.0)
DENSITY_CONST = 3.0 / (4.0 * np.pi)

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
# (PI = np.pi lived here with zero consumers repo-wide -- an alias for a name
# every module already imports.  Deleted, review 5.2.2.  TWOPI stays: it has
# real callers and spells something np.pi does not.)
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
# Hydrogen-burning minimum mass at ~solar composition (Chabrier & Baraffe
# 2000, ARA&A 38, 337).  Composition-dependent -- ~0.072 Msun at solar
# metallicity, rising toward ~0.09 Msun in metal-poor material -- and 0.075 is
# the round value the low-mass literature uses.  It is also already this
# codebase's stellar low-mass boundary (components/mann applies Mann+2015/2019
# over 0.075-0.7 Msun), so one number serves both.  Below it an object is a
# brown dwarf, not a star, and a STELLAR IMF has no claim on it.
HYDROGEN_BURNING_LIMIT = 0.075  # solMass

# Free-floating-planet mass function, Sumi et al. 2023 (AJ 166, 108;
# arXiv:2303.08280), from the MOA-II 9-year survey toward the Galactic bulge.
# Their planetary-mass population is quoted already as a density in LOG mass:
#     dN_4/dlog M = 2.18(+0.52/-1.40) * (M / 8 M_Earth)^-alpha_4
#                   dex^-1 star^-1,   alpha_4 = 0.96(+0.47/-0.27),
# measured over 0.33 < M/M_Earth < 6660 (i.e. 1e-6 < M/Msun < 0.02).
#
# Only the SLOPE survives into a mass prior: the pivot (8 M_Earth) and the
# amplitude (2.18 dex^-1 star^-1) are both absorbed by normalizing the density
# over the sampled support.  The amplitude -- equivalently f = 21(+23/-13)
# FFPs per star over the fitted range -- is an ABUNDANCE, and would only
# matter to a model that weighed the FFP and stellar lens populations against
# each other.  See galacticmodel.ffp_logmass_logp.
#
# The fitted range is NOT imposed as a bound anywhere: sub-stellar masses are
# this relation's domain, so where to cut its support off is the user's prior
# choice (star.<name>.logmass's `lower`), not ours.  The number below is
# carried only so the warning in Star._warn_ffp_logmass_bound can quote it as
# a concrete candidate.
FFP_MASS_FUNCTION_SLOPE = 0.96  # alpha_4, the exponent of dN/dlog M ~ M^-alpha
FFP_MASS_FUNCTION_MIN_MEARTH = 0.33  # lowest mass Sumi+2023 fit, in M_Earth

# --- 7. SUN CONSTANTS ---
SUN_GC_DISTANCE = 8.16  # in kpc (genulens/Koshimoto+21 R0 = 8160 pc)
SUN_Z_OFFSET = 0.025  # in kpc, Sun's height above the plane (genulens zsun)
# Solar velocity in the galactocentric frame, genulens convention
# (vxsun toward the GC, vysun in the rotation direction, vzsun up):
SUN_GALCEN_V = (10.0, 243.0, 7.0)  # in km/s

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
