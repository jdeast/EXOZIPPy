from ..component import Component
# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics

class CelestialBody(Component):
    """Middle-tier class for anything with mass and radius."""

    def build_core_parameters(self, model, overrides=None):
        core_params = {
            "logmass": None,
            "radius": None,
            "mass": "default",
            "density": "default",
            "logg": "default"
        }

        # Apply any dynamic data-driven heuristics (like RV K -> Mass)
        if overrides:
            for k, v in overrides.items():
                if k in core_params:
                    # If it's currently None, just assign the override dict
                    if core_params[k] is None:
                        core_params[k] = v
                    elif isinstance(core_params[k], dict):
                        core_params[k].update(v)

        self.build_pars_from_dict(core_params, shape=(self.n_elements,))
