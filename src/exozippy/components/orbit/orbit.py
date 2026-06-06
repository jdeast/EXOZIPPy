import numpy as np
import astropy.units as u

# pymc/exoplanet imports
import pytensor.tensor as pt
import pymc as pm
from exoplanet_core.pymc import ops as ops

# local imports
#from exozippy.components.parameter import Parameter
from exozippy.components.component import Component
# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics

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

    @property
    def prefix(self):
        return "orbit"

    def register_parameters(self, system):
        """Stage 2: Calculate window constraints and declare the manifest."""
        shape = (self.n_elements,)

        # 1. Peer into the config (Pre-flight windows)
        logP_cfg = self.config_manager.resolve(self.prefix, "logP", shape=shape, names=self.names)
        tc_cfg = self.config_manager.resolve(self.prefix, "tc", shape=shape, names=self.names)

        logP_init = np.atleast_1d(logP_cfg["initval"])
        tc_init = np.atleast_1d(tc_cfg["initval"])
        half_period = (10 ** logP_init) / 2.0

        self.manifest = {
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

        if any(self.fitvcve):
            raise NotImplementedError("VCVE parameterization not yet migrated to manifest.")
        else:
            self.manifest.update({
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

        i180_arr = np.atleast_1d(getattr(self, 'i180', False))
        derived_lowers = np.where(i180_arr, -1.0, 0.0)
        self.manifest["cosi"] = {"lower": derived_lowers}

    def build_likelihood(self, model, system):
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