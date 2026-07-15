from exozippy.components.component import Component
from . import physics

class Star(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Stellar Parameters"
        self.mist = [c.get("mist", True) for c in self.config]
        self.parsec = [c.get("parsec", False) for c in self.config]

        if isinstance(self.config, list):
            self.sedfile = self.config[0].get("sedfile")
        else:
            self.sedfile = self.config.get("sedfile")

    @property
    def prefix(self):
        return "star"

    def register_parameters(self, system):
        """Stage 2: Declare the manifest and push to ConfigManager."""

        # 1. Get the stellar parameters we always want
        self.manifest = {
            "logmass": None,
            "radius": None,
            "mass": "default",
            "density": "default",
            "logg": "default"
        }

        # 2. these should require evolutionary model, empirical relation,
        # limb darkening, sed, or maybe microlensing (baseline flux)
        # but for now, we'll always initialize them
        self.manifest.update({
            "teff": None,
            "feh": None,
            "luminosity": "default",
        })

        # Helper to check if a component is in the system topology,
        # even if it hasn't been instantiated as an attribute yet.
        topology_keys = []
        if hasattr(system, 'config'):
            topology_keys = list(system.config.keys())
        elif hasattr(system, 'config_manager') and hasattr(system.config_manager, 'system_config'):
            if system.config_manager.system_config:
                topology_keys = list(system.config_manager.system_config.keys())

        def in_system(comp_name):
            return hasattr(system, comp_name) or comp_name in topology_keys

        # 3. Add system-dependent parameters
        if in_system('sed'):
            self.manifest.update({
                "distance": None,
                "av": None,
                "radiussed": None,
                "teffsed": None,
                "luminositysed": "default",
                "fbolsed": "default"
            })

        if in_system("evolutionary_model"):
            mask = [m or p for m, p in zip(self.mist, self.parsec)]
            self.manifest.update({
                "age": {"mask": mask},
                "initfeh": {"mask": mask}
            })

        # The Mann relations key on absolute Ks, so they need the distance
        # modulus. The apparent/absolute Ks themselves live on the mann
        # component, which derives them from its own non-centered latent --
        # a free star.appks would be an unconstrained nuisance whenever the
        # Ks comes from the SED.
        if in_system('mann'):
            self.manifest.update({"distance": None})

        # Absolute astrometry (gaia/abs modes) constrains the reference
        # position and proper motion; rel-mode data are differential and
        # need only the parallax scale (distance), so those instruments do
        # not add the ra/dec/pm parameters.
        astrom_comp = getattr(system, 'astrometryinstrument', None)
        if astrom_comp is not None:
            astrom_modes = astrom_comp.modes
        else:
            astrom_cfgs = (getattr(self.config_manager, 'system_config', None)
                           or {}).get('astrometryinstrument') or []
            astrom_modes = [(c or {}).get("mode", "gaia") for c in astrom_cfgs]
        has_abs_astrom = any(m in ("gaia", "abs") for m in astrom_modes)

        if in_system('lens') or in_system('galacticmodel') or has_abs_astrom:
            self.manifest.update({
                "ra": None, "dec": None, "pm_ra": None,
                "pm_dec": None, "distance": None
            })
        elif astrom_modes:
            self.manifest.setdefault("distance", None)

        if in_system('galacticmodel'):
            self.manifest["rv"] = None

        if "distance" in self.manifest:
            self.manifest.update({"parallax": "default", "fbol": "default"})

    def build_likelihood(self, model, system):
        # Explicit pass-through!
        pass