import numpy as np

from exozippy.components.component import Component

from . import physics


def _microlensing_only_star_indices(system):
    """Star indices that are exclusively a microlensing source body.

    Nothing in the mulensing physics reads a source star's mass/teff/feh/
    radius (only the lens-side bodies' masses feed t_E; see
    mulensing/symbolic_physics.py's dead `source_mass`/`source_radius`
    symbol-map entries -- declared, never used in a RELATIONS equation), so
    these are dynamically irrelevant whenever a star is a source and never
    also a lens body.
    """
    lens = getattr(system, "lens", None)
    if lens is None or not getattr(lens, "source_bodies", None):
        return set()

    source_idx = {
        idx
        for event in lens.source_bodies
        for (ctype, idx) in event
        if ctype == "star"
    }
    lens_idx = {
        idx
        for event in lens.lens_bodies
        for (ctype, idx) in event
        if ctype == "star"
    }
    return source_idx - lens_idx


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
            "logg": "default",
        }

        # 2. these should require evolutionary model, empirical relation,
        # limb darkening, sed, or maybe microlensing (baseline flux)
        # but for now, we'll always initialize them
        self.manifest.update(
            {
                "teff": None,
                "feh": None,
                "luminosity": "default",
            }
        )

        # Helper to check if a component is in the system topology,
        # even if it hasn't been instantiated as an attribute yet.
        topology_keys = []
        if hasattr(system, "config"):
            topology_keys = list(system.config.keys())
        elif hasattr(system, "config_manager") and hasattr(
            system.config_manager, "system_config"
        ):
            if system.config_manager.system_config:
                topology_keys = list(
                    system.config_manager.system_config.keys()
                )

        def in_system(comp_name):
            return hasattr(system, comp_name) or comp_name in topology_keys

        # 3. Add system-dependent parameters
        if in_system("sed"):
            self.manifest.update(
                {
                    "distance": None,
                    "av": None,
                    "radiussed": None,
                    "teffsed": None,
                    "luminositysed": "default",
                    "fbolsed": "default",
                }
            )

        if in_system("evolutionary_model"):
            mask = [m or p for m, p in zip(self.mist, self.parsec)]
            self.manifest.update(
                {"age": {"mask": mask}, "initfeh": {"mask": mask}}
            )

        # The Mann relations key on absolute Ks, so they need the distance
        # modulus. The apparent/absolute Ks themselves live on the mann
        # component, which derives them from its own non-centered latent --
        # a free star.appks would be an unconstrained nuisance whenever the
        # Ks comes from the SED.
        if in_system("mann"):
            self.manifest.update({"distance": None})

        # Rossiter-McLaughlin: the shared line-broadening terms (macro/beta/
        # micro) live on the star; vsini + lambda live on orbit (they are
        # coupled by the sqrt(vsini)cos/sin(lambda) reparameterization).
        from ..rm import rm_enabled

        if rm_enabled(system):
            self.manifest.update(
                {"vmacro": None, "vbeta": None, "vmicro": None}
            )

        # Absolute astrometry (gaia/abs modes) constrains the reference
        # position and proper motion; rel-mode data are differential and
        # need only the parallax scale (distance), so those instruments do
        # not add the ra/dec/pm parameters.
        astrom_comp = getattr(system, "astrometryinstrument", None)
        if astrom_comp is not None:
            astrom_modes = astrom_comp.modes
        else:
            astrom_cfgs = (
                getattr(self.config_manager, "system_config", None) or {}
            ).get("astrometryinstrument") or []
            astrom_modes = [(c or {}).get("mode", "gaia") for c in astrom_cfgs]
        has_abs_astrom = any(m in ("gaia", "abs") for m in astrom_modes)

        if in_system("lens") or in_system("galacticmodel") or has_abs_astrom:
            self.manifest.update(
                {
                    "ra": None,
                    "dec": None,
                    "pm_ra": None,
                    "pm_dec": None,
                    "distance": None,
                }
            )
        elif astrom_modes:
            self.manifest.setdefault("distance", None)

        if in_system("galacticmodel"):
            self.manifest["rv"] = None

        if "distance" in self.manifest:
            self.manifest.update({"parallax": "default", "fbol": "default"})

        # Pure microlensing-source stars: pin the parameters nothing in this
        # topology consumes, instead of requiring every microlensing
        # params.yaml to fix them by hand (see run_event.py's old
        # build_user_params, which did exactly this per-event).
        ml_source_idx = _microlensing_only_star_indices(system)
        if ml_source_idx:
            relation_idx = set()
            for relation in ("mann", "torres"):
                comp = getattr(system, relation, None)
                if comp is not None:
                    relation_idx |= set(comp.star_indices)

            sed_idx = set()
            sed = getattr(system, "sed", None)
            blend_matrix = getattr(sed, "blend_matrix", None)
            if blend_matrix is not None:
                sed_idx = set(np.nonzero((blend_matrix != 0).any(axis=0))[0])

            abs_astrom_idx = set()
            astrom = getattr(system, "astrometryinstrument", None)
            if astrom is not None:
                modes = getattr(astrom, "modes", None)
                star_map = getattr(astrom, "star_map", None)
                if modes is not None and star_map is not None:
                    abs_astrom_idx = {
                        int(star_map[i])
                        for i, m in enumerate(modes)
                        if m in ("gaia", "abs")
                    }

            def _pin_sigma(param_name, skip_idx):
                idx_list = sorted(ml_source_idx - skip_idx)
                if not idx_list or param_name not in self.manifest:
                    return
                entry = self.manifest[param_name]
                entry = dict(entry) if isinstance(entry, dict) else {}
                pin = np.full(self.n_elements, np.nan)
                pin[idx_list] = 0.0
                overrides = dict(entry.get("overrides", {}))
                overrides["sigma"] = pin.tolist()
                entry["overrides"] = overrides
                self.manifest[param_name] = entry

            _pin_sigma("logmass", relation_idx)
            _pin_sigma("teff", relation_idx | sed_idx)
            _pin_sigma("feh", relation_idx | sed_idx)
            _pin_sigma("radius", relation_idx | sed_idx)
            _pin_sigma("ra", abs_astrom_idx)
            _pin_sigma("dec", abs_astrom_idx)

    def build_likelihood(self, model, system):
        # Explicit pass-through!
        pass
