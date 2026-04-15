from pathlib import Path

# local imports
from exozippy.components.component import Component

# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics

# debugging imports
import ipdb

from exozippy.components.celestial_body.celestial_body import CelestialBody


class Star(CelestialBody):
    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)

        self.label = "Stellar Parameters"

        # this belongs at the system level (probably)
        #self.sedfile = Path(self.config.get("sedfile", None))
        #self.teffsedfloor = self.config.get("teffsedfloor", 0.020)
        #self.fbolsedfloor = self.config.get("fbolsedfloor", 0.024)

        self.mannmass = [c.get("mannmass",False) for c in self.config]
        self.mannrad = [c.get("mannrad",False) for c in self.config]
        self.mist = [c.get("mist",True) for c in self.config]
        self.parsec = [c.get("parsec",False) for c in self.config]
        if isinstance(self.config, list):
            self.sedfile = self.config[0].get("sedfile")
        else:
            self.sedfile = self.config.get("sedfile")

    @property
    def prefix(self):
        return "star"

    def build_parameters(self, model):

        # 1. Build the shared DNA (mass, radius, density, logg)
        self.build_core_parameters(model)

        # 2. Build Star-specific parameters
        parameters = {
            "teff": None,
            "feh": None,
            "luminosity": "default",
        }

        self.build_pars_from_dict(parameters,shape=(self.n_elements,))

        '''
        this belongs in a mann component, build_likelihood function, but keep it here until we do that
            mann_mstar, mann_rstar, mann_sigma_mstar, mann_sigma_rstar = massradius_mann(self.ks0.value, self.feh.value,
                                                                                             distance=self.distance.value)
            if self.mannmass:
                mannmass_prior = pm.Potential(f"{prefix}."mannmass_prior", -0.5 *
                                              ((self.mass.value - mann_mstar) / mann_sigma_mstar) ** 2)
            if self.mannrad:
                mannrad_prior = pm.Potential(f"{prefix}.mannrad_prior", -0.5 *
                                             ((star.radius.value - mann_rstar) / mann_sigma_rstar) ** 2)
        '''
    def load_data(self):
        # no data to load
        pass

    def build_dependent_parameters(self, model, system):

        parameters = {}

        # for each relevant component, we'll include additional parameters (both sampled and derived)
        # It's ok if we declare variables multiple times, they'll only be added once.

        # SED model
        if hasattr(system, 'sed'):
            parameters["distance"] = None
            parameters["av"] = None
            parameters["radiussed"] = None
            parameters["teffsed"] = None
            parameters["luminositysed"] = "default"
            parameters["fbolsed"] = "default"

        # evolutionary model
        if hasattr(system, "evolutionary_model"):
            parameters["age"] = {"mask": self.mist | self.parsec}
            parameters["initfeh"] = {"mask": self.mist | self.parsec}

        # mann model
        if hasattr(system, 'mann'):
            parameters["distance"] = None
            parameters["appks"] = None
            parameters["absks"] = "default"

        # microlensing model
        if hasattr(system, 'lens'):
            parameters["pm_ra"] = None
            parameters["pm_dec"] = None
            parameters["distance"] = None

        # astrometric model
        if hasattr(system, 'astrometry'):
            parameters["ra"] = None
            parameters["dec"] = None
            parameters["pm_ra"] = None
            parameters["pm_dec"] = None
            parameters["distance"] = None

        if "distance" in parameters.keys():
            parameters["parallax"] = "default"
            parameters["fbol"] = "default"

        self.build_pars_from_dict(parameters, shape=(self.n_elements,))

    def build_likelihood(self, model, system):
        # no likelihood (yet?)
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass