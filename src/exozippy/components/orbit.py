import numpy as np
import astropy.units as u

# pymc/exoplanet imports
import pytensor.tensor as pt
import pymc as pm
from exoplanet_core.pymc import ops as ops

# local imports
from exozippy.constants import TWOPI
from ..physics import calc_tp

from .parameter import Parameter
from .component import Component

# debugging imports
import ipdb

class Orbit(Component):
    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)

        self.label = "Orbital Parameters"

        self.primary = [c.get("primary","star.0") for c in self.config] # star zero is the host by default
        self.companion = [c.get("companion", f"planet.{i}") for i, c in enumerate(self.config)] # the orbits are aligned, one per planet
        self.i180 = [c.get("i180",False) for c in self.config]
        self.fitvcve = [c.get("fitvcve",False) for c in self.config]

    def build_parameters(self, model):
        prefix = "orbit"
        shape = (self.n_elements,)

        # 1. PEER INTO THE CONFIG (Pre-flight)
        # We grab the resolved dictionaries to peek at the initial values
        logP_cfg = self.config_manager.resolve(prefix, "logP", shape=shape)
        tc_cfg = self.config_manager.resolve(prefix, "tc", shape=shape)

        # 2. CALCULATE WINDOWS (Domain Intelligence)
        # Convert logP (dex days) to period (days) to find the half-period
        # Note: resolve() ensures these are numpy arrays or scalars based on shape
        logP_init = np.atleast_1d(logP_cfg["initval"])
        tc_init = np.atleast_1d(tc_cfg["initval"])

        half_period = (10 ** logP_init) / 2.0

        # Set the windowed bounds
        parameters = {
            "logP": None,
            "period": {"force_node": True, "expr_key": "default"},
            "n": "default",
            "tc": {
                "force_node": True,
                "lower": tc_init - half_period,
                "upper": tc_init + half_period
            }
        }

        fitvcve_mask = np.atleast_1d(getattr(self, 'fitvcve', False)).astype(bool)
        hk_mask = ~fitvcve_mask

        # --- Eccentricity / Omega Geometry Swap ---
        if any(self.fitvcve):
            # Fit using VC/VE (or VCVE as a single vector depending on your math)
            raise NotImplementedError()
            parameters["vcve"] = None
            parameters["cosw"] = None
            parameters["sinw"] = None
            parameters["chord"] = None
            parameters["ecc"] = "from_vcve"
            parameters["omega"] = "from_vcve"
            parameters["cosi"] = "from_chord"
            #parameters["b"] = "default"
        else:
            parameters.update({
                "secosw": {"mask": hk_mask},
                "sesinw": {"mask": hk_mask},
                "cosi": {"mask": hk_mask},
                "ecc": "default",
                "omega": "default",
                "inc": "default",
                "sini": "default",
                "sinw": "default",
                "cosw": "default",
                "esinw": "default",
                "ecosw": "default",
                "tp": "default",
            })

        # Create a 'lower' array: 0.0 where NOT i180, -1.0 where i180
        i180_arr = np.atleast_1d(getattr(self, 'i180', False))
        derived_lowers = np.where(i180_arr, -1.0, 0.0)
        parameters["cosi"] = {"lower": derived_lowers}

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=prefix)

    def load_data(self):
        pass

    def build_dependent_parameters(self, model, system):
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

    def build_likelihood(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

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