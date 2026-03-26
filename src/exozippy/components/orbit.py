import numpy as np
import astropy.units as u

# pymc/exoplanet imports
import pytensor.tensor as pt
import pymc as pm
from exoplanet_core.pymc import ops as ops

# local imports
from exozippy.constants import G, AU, mjup, rjup, pc, rsun, msun, sigmasb, Gmsun, meter
from .parameter import Parameter
from .component import Component

# debugging imports
import ipdb

class Orbit(Component):
    def __init__(self, config, user_params):
        self.label = "Orbital Parameters"
        self.config = config
        self.user_params = user_params
        self.norbits = len(config)

        self.tref = self.config[0].get("tref",2460000.0)

        self.names = [c.get("name") for c in self.config]

        self.primary = [c.get("primary","star.0") for c in self.config] # star zero is the host by default
        self.companion = [c.get("companion", f"planet.{i}") for i, c in enumerate(self.config)] # the orbits are aligned, one per planet
        self.i180 = [c.get("i180",False) for c in self.config]
        self.fitvcve = [c.get("fitvcve",False) for c in self.config]

    def build_parameters(self, model):

        prefix = f"orbit"

        # physical range is -1 <= cosi <=1
        # but unless we have astrometry or mutual transits, bound should be 0 <= cosi <=1
        if self.i180:
            cosilower = -1.0
        else:
            cosilower = 0.0

        self.logP = Parameter(f"{prefix}.logP",
                              lower=-2, upper=13.7,
                              initval=1.0, init_scale=-1e-5,
                              unit=u.dex(u.d), internal_unit=u.dex(u.d),
                              latex=r"\log{P}", description='log(Period/day)', print_to_table=False,
                              user_params=self.user_params, shape=(self.norbits,))
        self.logP.build_pymc()

        # force the node. it's required for the model anyway, and it makes plotting easier
        self.period = Parameter(f"{prefix}.period",
                                force_node=True,
                                expression=lambda: 10**self.logP.value,
                                unit=u.d, internal_unit=u.d,
                                latex="P", description='Period',
                                user_params=self.user_params)
        self.period.build_pymc()

        pi2 = 2.0 * np.pi
        self.n = Parameter(f"{prefix}.n",
                           expression=lambda: pi2 / self.period.value,
                           unit=u.day**(-1.0), internal_unit=u.day**(-1.0),
                           latex="n", description='', print_to_table=False,
                           user_params=self.user_params)
        self.n.build_pymc()

        # this scale should be dataset dependent
        self.tc_base = Parameter(f"{prefix}.tc_base",
                                 lower=-1e6, upper=1e6,
                                 initval=0.0, init_scale=0.001,
                                 unit=u.d, internal_unit=u.d,unit_latex=r"\bjdtdb",
                                 latex='T_C', description='Time of conjunction (base)', print_to_table=True,
                                 user_params=self.user_params, shape=(self.norbits,))
        self.tc_base.build_pymc()

        # force the node. it's required for the model anyway, and it makes plotting easier
        self.tc = Parameter(f"{prefix}.tc",
                            force_node=True,
                            expression=lambda: self.tc_base.value + self.tref,
                            unit=u.d, internal_unit=u.d, unit_latex=r"\bjdtdb",
                            latex='T_C', description='Time of conjunction',
                            user_params=self.user_params)
        self.tc.build_pymc()

        # if we're fitting vcve, we need the planet first, then the ecc initialization goes in build_dependent_parameters
        # this gets complicated if there's a mix of fitvcve and not... punt for now
        if self.fitvcve[0]: return

        self.cosi = Parameter(f"{prefix}.cosi",
                              lower=cosilower, upper=1.0,
                              initval=1e-7, init_scale=0.01,
                              unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                              latex=r"\cos{i}", description='cos of inc',
                              user_params=self.user_params, shape=(self.norbits,))
        self.cosi.build_pymc()

        # 2. an adaption of Dan Foreman-Mackey's unit_disk
        # Ensures e < 1 and handles the omega periodicity
        self.secosw = Parameter(f"{prefix}.secosw",
                                lower=-1.0, upper=1.0,
                                initval=0.01, init_scale=0.2,
                                unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                latex=r"e^{1/2}\cos{\omega_*}", description='secosw', print_to_table=False,
                                user_params=self.user_params, shape=(self.norbits,))
        self.secosw.build_pymc()

        # if this were the actual sqrt(e)sin(omega), it would draw e>=1
        self.sesinw_raw = Parameter(f"{prefix}.sesinw_raw",
                                    lower=-1.0, upper=1.0,
                                    initval=0.01, init_scale=0.2,
                                    unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                    latex=r"e^{1/2}\cos{\omega_*}_{raw}", description='sesinw_raw', print_to_table=False,
                                    user_params=self.user_params, shape=(self.norbits,))
        self.sesinw_raw.build_pymc()

        # correct the prior so it is uniform in e/omega
        norm = pt.sqrt(1.0 - self.secosw.value ** 2)
        pm.Potential(f"{prefix}.unit_disk_jacobian", pt.log(norm))

        # this version of sesinw enforces e < 1
        self.sesinw = Parameter(f"{prefix}.sesinw",
                                expression=lambda: self.sesinw_raw.value * norm,
                                unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                latex=r"e^{1/2}\sin{\omega_*}", description='sesinw', print_to_table=False,
                                user_params=self.user_params)
        self.sesinw.build_pymc()

        self.ecc = Parameter(f"{prefix}.ecc",
                             expression=lambda: self.secosw.value ** 2 + self.sesinw.value ** 2,
                             unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                             latex='e', description='eccentricity',
                             user_params=self.user_params)
        self.ecc.build_pymc()

        self.omega = Parameter(f"{prefix}.omega",
                                 expression=lambda: pt.arctan2(self.sesinw.value, self.secosw.value),
                                 unit=u.deg, internal_unit=u.rad,
                                 latex=r"\omega_*", description='Arg of periastron',
                                 user_params=self.user_params)
        self.omega.build_pymc()

        self.sinw = Parameter(f"{prefix}.sinw",
                                 expression=lambda: pt.sin(self.omega.value),
                                 unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                 latex=r"\sin{\omega_*}", description='sin of arg of periastron', print_to_table=False,
                                 user_params=self.user_params)
        self.sinw.build_pymc()

        self.cosw = Parameter(f"{prefix}.cosw",
                                 expression=lambda: pt.cos(self.omega.value),
                                 unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                                 latex=r"\cos{\omega_*}", description='cos of arg of periastron', print_to_table=False,
                                 user_params=self.user_params)
        self.cosw.build_pymc()

        self.esinw = Parameter(f"{prefix}.esinw",
                               expression=self.ecc.value * self.sinw.value,
                               unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                               latex=r"e\sin{\omega_*}", description='e times sin of arg of periastron',
                               user_params=self.user_params)
        self.esinw.build_pymc()

        self.ecosw = Parameter(f"{prefix}.ecosw",
                               expression=lambda: self.ecc.value * self.cosw.value,
                               unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                               latex=r"e\cos{\omega_*}", description='e times cos of arg of periastron',
                               user_params=self.user_params)
        self.ecosw.build_pymc()

        self.inc = Parameter(f"{prefix}.inc",
                             expression=lambda: pt.arccos(self.cosi.value),
                             unit=u.deg, internal_unit=u.rad,
                             latex='i', description='inclination',
                             user_params=self.user_params)
        self.inc.build_pymc()

        self.sini = Parameter(f"{prefix}.sini",
                              expression=lambda: pt.sin(self.inc.value),
                              unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                              latex=r"\sin{i}", description='sin(inc)', print_to_table=False,
                              user_params=self.user_params)
        self.sini.build_pymc()

        # Stable Tc -> Tp logic
        self.E0 = Parameter(f"{prefix}.E0",
                            expression=lambda: 2.0 * pt.arctan2(pt.sqrt(1.0 - self.ecc.value) * self.cosw.value, pt.sqrt(1.0 + self.ecc.value) * (1.0 + self.sinw.value)),
                            unit=u.deg, internal_unit=u.rad,
                            latex='E0', description='Eccentric Anomaly', print_to_table=False,
                            user_params=self.user_params)
        self.E0.build_pymc()

        self.tp = Parameter(f"{prefix}.tp",
                            expression=lambda: self.tc.value - (self.E0.value - self.ecc.value * pt.sin(self.E0.value)) / self.n.value,
                            unit=u.d, internal_unit=u.d, unit_latex= r"\bjdtdb",
                            latex='T_P', description='Time of Periastron',
                            user_params=self.user_params)
        self.tp.build_pymc()

    def build_dependent_parameters(self, model, star, planet, user_params=None):
        # TODO: handle the vcve parameterization
        return
        # transit only parameterization (Eastman, 2024)
        # vcve, sin(omega), cos(omega), sign, chord
        self.vcve = Parameter(f"{prefix}.vcve", lower=0.0, initval=1.0,
                              latex='V_c/V_e', description='Scaled velocity',
                              latex_unit='', user_params=self.user_params)

        self.cosw = Parameter(f"{prefix}.cosw", lower=-1.0, upper=1.0, initval=0.0,
                              latex=r"\cos{\omega_*}", description='Cos of arg of periastron',
                              latex_unit='', user_params=self.user_params)

        self.sinw = Parameter(f"{prefix}.sinw", lower=-1.0, upper=1.0, initval=0.0,
                              latex=r"\sin{\omega_*}", description='Sin of arg of periastron',
                              latex_unit='', user_params=self.user_params)

        # bound omega
        L = self.cosw.value ** 2 + self.sinw.value ** 2

        # ensures uniform omega distribution
        wbound = pm.Potential(f"{prefix}.wbound", pt.switch(L > 1.0, -np.inf, 0.0))

        self.omega = Parameter(f"{prefix}.omega", unit=u.rad,
                               expression=pt.arctan2(self.sinw, self.cosw),
                               latex=r"\omega_*", description='Arg of periastron',
                               latex_unit='rad', user_params=self.user_params)

        self.sign = Parameter(f"{prefix}.sign", lower=-1.0, upper=1.0,  # boolean?
                              latex='Sign', description='Sign of quadratic solution',
                              latex_unit='', user_params=self.user_params)

        # solve quadratic for e (Eastman 2024, eq 5)
        a = self.vcve.value ** 2 * self.sinw.value ** 2 + 1.0
        b = 2.0 * self.vcve.value ** 2 * self.sinw.value
        c = self.vcve.value ** 2 - 1.0

        self.ecc = Parameter(f"{prefix}.e",
                             expression=-b + pt.sign(self.sign.value) * pt.sqrt(b ** 2 - 4.0 * a * c) / (2.0 * a),
                             latex='e', description='eccentricity',
                             latex_unit='', user_params=self.user_params)

        self.esinw = Parameter(f"{prefix}.esinw",
                               expression=self.ecc.value * self.sinw.value,
                               latex=r"e\sin{\omega_*}", description='e times sin of arg of periastron',
                               latex_unit='', user_params=self.user_params)

        jacobian = 1.0

        self.chord = Parameter(f"{prefix}.chord", lower=0.0,
                               latex='chord', description='transit chord',
                               latex_unit='', user_params=self.user_params)

        self.cosi = Parameter(f"{prefix}.cosi",
                              expression=planet.b.value /
                                         (planet.ar.value * (1.0 - self.ecc.value ** 2) / (
                                                 1.0 + self.esinw.value)),
                              latex=r"\cos{i}", description='cos of inclination',
                              latex_unit='', user_params=self.user_params)

        jacobian *= self.b.value ** 2 / (
                self.cosi.value * self.chord.value)  # d(chord)/d(cosi)

        # correct the prior to be uniform in e/omega (Eastman 2024, eq 6)
        jacobian *= (self.ecc.value + self.sinw.value) / (
                pt.sqrt(1.0 - self.ecc.value ** 2) * (1.0 + self.esinw.value) ** 2)  # d(vcve)/d(e)

        self.jacobian = pm.Potential(f"{prefix}.jacobian", pt.abs(jacobian))

    def get_true_anomaly(self, t):
        """Returns the true anomaly f for all planets at all times."""
        t_grid = t[:, None]
        tp = self.tp.value[None, :]
        n = self.n.value[None, :]
        ecc = self.ecc.value[None, :]

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        return pt.arctan2(sinf, cosf)

    def get_radial_velocity(self, t, K, orbit_map):
        """
        The optimized vectorized reflex RV signal.
        t: (N_obs,) vector of times
        K: (N_planets,) vector of semi-amplitudes
        """
        # 1. Broadcast time and orbital parameters into 2D grids
        # Shape: (N_obs, N_planets)
        t_grid = t[:, None]
        tp = self.tp.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]
        K_grid = K[None, :]

        # 2. Calculate Mean Anomaly (M)
        # M = n * (t - tp)
        M = (t_grid - tp) * n

        # 3. Solve Kepler's Equation
        # ops.kepler handles the (N_obs, N_planets) grid efficiently
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        # 4. Calculate RV per planet
        # Using the identity: cos(w + f) = cos(w)cos(f) - sin(w)sin(f)
        rv_matrix = K_grid * (cosw*cosf - sinw*sinf + ecc*cosw)

        return rv_matrix

        # 5. Sum across planets (axis 1) to get the total signal at each time
        # Result Shape: (N_obs,)
        return pt.sum(rv_matrix, axis=1)