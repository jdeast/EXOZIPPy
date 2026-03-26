import astropy.constants as const
import numpy as np

# constants easier to digest variable names
G = const.GM_sun.value / const.R_sun.value ** 3 * 86400.0 ** 2
AU = const.au.value / const.R_sun.value
mjup = const.GM_jup.value / const.GM_sun.value
rjup = const.R_jup.cgs.value / const.R_sun.cgs.value  # rsun/rjup
pc = const.pc.cgs.value  # cm/pc
rsun = const.R_sun.cgs.value  # cm/r_sun
msun = const.M_sun.cgs.value  # g/m_sun
sigmasb = const.sigma_sb.cgs.value
Gmsun = const.GM_sun.cgs.value
Gmjup = const.GM_jup.cgs.value
Gmearth = const.GM_earth.cgs.value
meter = 100.0  # cm/m
lsun = 4.0*np.pi*rsun**2*sigmasb*5778**4
