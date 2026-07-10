import logging
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.sed.bc_grid import (
    resolve_filter_name,
    facility_from_svo_name,
    _load_alias_table,
)

logger = logging.getLogger(__name__)


class Band(Component):
    """Photometric band with limb-darkening coefficients.

    One Band instance per filter. Instruments reference a band by name.
    Supports linear (sample u1) and quadratic Kipping (sample q1/q2, derive u1/u2) laws.

    Band is the single carrier of filter identity for instruments: each
    element's user-facing `filter:` string is resolved through the SED
    component's alias table (filters/filternames.txt) into canonical
    MIST (`filter_mist`) and SVO (`filter_svo`) names, which the SED
    flux-prediction hooks (mulensing f_source constraint, transit
    deblending, astrometry fluxfrac) key on.
    """

    yaml_key = "band"

    @property
    def prefix(self):
        return "band"

    def load_data(self, system):
        self.filter_names = [c.get("filter", "") for c in self.config]
        self.star_indices = [c.get("star_ndx", 0) for c in self.config]
        self.ld_laws = [c.get("ld_law", "quadratic") for c in self.config]

        # Canonical filter identities via the SED alias table. An
        # unknown name passes through unchanged (the user may already be
        # supplying a canonical column name), but gets a warning so
        # typos surface early.
        alias_df = _load_alias_table()
        self.filter_mist = []
        self.filter_svo = []
        for band_name, filt in zip(self.names, self.filter_names):
            self.filter_mist.append(
                resolve_filter_name(filt, alias_df, alias="MIST") if filt else None)
            self.filter_svo.append(
                resolve_filter_name(filt, alias_df, alias="SVO") if filt else None)
            if filt and alias_df is not None and not alias_df.eq(filt).any(axis=1).any():
                logger.warning(
                    f"Band '{band_name}': filter '{filt}' is not in the "
                    f"filter alias table (components/sed/filters/"
                    f"filternames.txt); assuming it is already a canonical "
                    f"BC-table/SVO name."
                )

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
