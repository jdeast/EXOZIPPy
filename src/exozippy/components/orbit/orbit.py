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

        # Astrometry constrains the longitude of the ascending node and
        # breaks the i <-> 180-i degeneracy, so sample bigomega and allow
        # the full inclination range when an astrometry component is active.
        topology_keys = []
        if hasattr(system, 'config') and hasattr(system.config, 'keys'):
            topology_keys = list(system.config.keys())
        has_astrometry = (hasattr(system, 'astrometryinstrument')
                          or 'astrometryinstrument' in topology_keys)
        if has_astrometry:
            self.manifest["bigomega"] = None

        i180_arr = np.atleast_1d(getattr(self, 'i180', False)) | has_astrometry
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

    def get_sky_position(self, t, a_scale, orbit_map, relative=False):
        """
        Vectorized sky-plane offsets of an orbiting body.

        t: (N_obs,) vector of times [BJD_TDB]
        a_scale: (N_planets,) amplitude scaling, e.g. the photocenter or
                 relative semimajor axis in mas; sets the output units
        orbit_map: integer map from planet slots to orbit elements
        relative: False -> the primary/photocenter orbit around the
                  barycenter (uses omega_*); True -> the companion's orbit
                  relative to the primary (omega_* + 180 deg)

        Returns (dE, dN), each (N_obs, N_planets): offsets toward East and
        North in the units of a_scale.

        Conventions (EXOFASTv2): omega is the argument of periastron of the
        PRIMARY's orbit (omega_*). bigomega is the position angle of the
        ascending node, measured East of North, where the ascending node is
        the node at which the body recedes from the observer -- consistent
        with the sign of get_radial_velocity (the primary crosses its
        ascending node at omega_* + f = 0, where its RV is maximal).
        Without RV data, (bigomega, omega) and (bigomega+180, omega+180)
        are degenerate.
        """
        t_grid = t[:, None]
        tp = self.tp.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]
        cosi = self.cosi.value[orbit_map][None, :]
        bigomega = self.bigomega.value[orbit_map][None, :]
        cosO = pt.cos(bigomega)
        sinO = pt.sin(bigomega)

        if relative:
            # The companion's argument of periastron is omega_* + pi
            cosw = -cosw
            sinw = -sinw

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        # Separation from the barycenter (or primary) in units of a_scale
        r = a_scale[None, :] * (1.0 - ecc ** 2) / (1.0 + ecc * cosf)

        # cos/sin(omega + f)
        coswf = cosw * cosf - sinw * sinf
        sinwf = sinw * cosf + cosw * sinf

        # Thiele-Innes projection (North, East), PA measured East of North:
        # at omega + f = 0 (ascending node) the body sits at PA = bigomega.
        dN = r * (cosO * coswf - sinO * sinwf * cosi)
        dE = r * (sinO * coswf + cosO * sinwf * cosi)
        return dE, dN

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