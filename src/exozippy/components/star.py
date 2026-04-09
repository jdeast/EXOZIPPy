from pathlib import Path

# local imports
from .component import Component

# debugging imports
import ipdb

class Star(Component):
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

    def build_parameters(self, model):
        prefix = "star"

        parameters = {
            "mass" : None,
            "radius": None,
            "teff": None,
            "feh": None,
            "density": "default",
            "logg":"default",
            "luminosity": "default",
        }

        # Conditional additions
        # evolutionary model
        if any(self.mist) or any(self.parsec):
            parameters["age"] = {"mask": self.mist | self.parsec}
            parameters["initfeh"] = {"mask": self.mist | self.parsec}

        # SED model
        if getattr(self, 'sedfile', None):
            parameters["fbol"] = "default"
            parameters["av"] = None
            parameters["distance"] = None
            parameters["radiussed"] = None
            parameters["teffsed"] = None
            parameters["luminositysed"] = "default"
            parameters["fbolsed"] = "default"

        self.build_pars_from_dict(parameters,shape=(self.n_elements,), prefix=prefix)

        '''
        self.parallax = Parameter(f"{prefix}.parallax",
                                  expression= lambda: calc_parallax(distance),
                                  unit=u.mas, internal_unit=u.mas,
                                  latex=r"\varpi", description='Parallax',
                                  user_params=self.user_params, names=self.names,
                                  shape=(self.nstars,))
        self.parallax.build_pymc()

        if self.mannmass or self.mannrad:
            self.appks = Parameter(f"{prefix}.appks",
                                   lower=-30,upper=30,
                                   unit=u.mag, internal_unit=u.mag,
                                   latex='ks', description="Apparent ks mag",
                                   user_params=self.user_params, names=self.names,
                                   shape=(self.nstars,))
            self.appks.build_pymc()

            # revisit units when we uncomment this!!!
            self.absks = Parameter(f"{prefix}.absks",
                                   expression= lambda: calc_absmag(self.appks.value, self.distance.value),
                                   unit=u.mag, internal_unit=u.mag,
                                   latex='Ks', description="Absolute Ks mag",
                                   user_params=self.user_params, names=self.names,
                                   shape=(self.nstars,))
            self.absks.build_pymc()

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
        # no dependent parameters necessary
        # build SED parameters here?
        pass

    def build_likelihood(self, model, system):
        # no likelihood (yet?)
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass