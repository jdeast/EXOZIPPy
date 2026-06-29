import logging
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component

logger = logging.getLogger(__name__)


class Band(Component):
    """Photometric band with limb-darkening coefficients.

    One Band instance per filter. Instruments reference a band by name.
    Supports linear (sample u1) and quadratic Kipping (sample q1/q2, derive u1/u2) laws.
    """

    yaml_key = "band"

    @property
    def prefix(self):
        return "band"

    def load_data(self, system):
        self.filter_names = [c.get("filter", "") for c in self.config]
        self.star_indices = [c.get("star_ndx", 0) for c in self.config]
        self.ld_laws = [c.get("ld_law", "quadratic") for c in self.config]

    def build_maps(self):
        self.star_map = np.array(self.star_indices, dtype=int)

    def register_parameters(self, system):
        has_quadratic = any(law != "linear" for law in self.ld_laws)
        if has_quadratic:
            self.manifest = {
                "q1": None,
                "q2": None,
                "u1": "default",
                "u2": "default",
            }
        else:
            self.manifest = {
                "u1": None,
            }

    def build_likelihood(self, model, system):
        pass

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
