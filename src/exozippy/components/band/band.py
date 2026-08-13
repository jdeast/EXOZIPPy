import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.sed.bc_grid import (
    _load_alias_table,
    facility_from_svo_name,
    resolve_filter_name,
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

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "filter",
                "kind": "option",
                "accepts": None,
                "required": True,
                "doc": (
                    "Filter/bandpass name; resolved through the SED filter "
                    "alias table (components/sed/filters/filternames.txt) "
                    "into a canonical name at load time."
                ),
            },
            {
                "key": "star_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": (
                    "Index or name of the star whose limb darkening this "
                    "band models. Default 0."
                ),
            },
            {
                "key": "ld_law",
                "kind": "option",
                "accepts": ["quadratic", "linear"],
                "required": False,
                "doc": "Limb-darkening law. Default 'quadratic'.",
            },
            {
                "key": "fitthermal",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fit a constant secondary-eclipse thermal-emission "
                    "depth (ppm) for this band. Default False, which pins "
                    "thermal at 0 (transit-only model, unchanged). "
                    "Phase-curve variation (BEER) is not modeled by this "
                    "flag; see fitreflect/fitellip (PR 1.b)."
                ),
            },
            {
                "key": "fitreflect",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fit a reflected-light phase-curve amplitude (ppm) for "
                    "this band, peaking at secondary eclipse and zero at "
                    "primary transit. Default False, which pins reflect at "
                    "0. Set alongside fitthermal only with real full-orbit "
                    "phase coverage -- with eclipse-only data the two are "
                    "degenerate and only their sum is measurable."
                ),
            },
            {
                "key": "fitellip",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fit an ellipsoidal-variation amplitude (ppm) for this "
                    "band: a half-orbital-period brightness modulation from "
                    "the star's tidal distortion, dimmest at both "
                    "conjunctions and brightest at both quadratures. "
                    "Default False, which pins ellipsoidal at 0."
                ),
            },
        ]

    def load_data(self, system):
        self.filter_names = [c.get("filter", "") for c in self.config]
        self.star_indices = [c.get("star_ndx", 0) for c in self.config]
        self.ld_laws = [c.get("ld_law", "quadratic") for c in self.config]
        self.fitthermal = [
            bool(c.get("fitthermal", False)) for c in self.config
        ]
        self.fitreflect = [
            bool(c.get("fitreflect", False)) for c in self.config
        ]
        self.fitellip = [bool(c.get("fitellip", False)) for c in self.config]

        # Canonical filter identities via the SED alias table. An
        # unknown name passes through unchanged (the user may already be
        # supplying a canonical column name), but gets a warning so
        # typos surface early.
        alias_df = _load_alias_table()
        self.filter_mist = []
        self.filter_svo = []
        for band_name, filt in zip(self.names, self.filter_names):
            self.filter_mist.append(
                resolve_filter_name(filt, alias_df, alias="MIST")
                if filt
                else None
            )
            self.filter_svo.append(
                resolve_filter_name(filt, alias_df, alias="SVO")
                if filt
                else None
            )
            if (
                filt
                and alias_df is not None
                and not alias_df.eq(filt).any(axis=1).any()
            ):
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

        # thermal/reflect/ellipsoidal (all ppm) are opt-in per band via
        # fitthermal/fitreflect/fitellip. Bands that don't opt in are pinned
        # at sigma=0 (same "overrides" pattern Instrument._register_gp uses
        # for terms an element didn't ask for), so the parameter's value is
        # exactly 0 and the transit model is unchanged unless a band
        # explicitly asks for it.
        self.manifest["thermal"] = self._pinned_manifest_entry(self.fitthermal)
        self.manifest["reflect"] = self._pinned_manifest_entry(self.fitreflect)
        self.manifest["ellipsoidal"] = self._pinned_manifest_entry(
            self.fitellip
        )

    def _pinned_manifest_entry(self, opt_in_flags):
        """A manifest entry that's free where opt_in_flags is True, and
        pinned to sigma=0 (fixed at its default initval, 0) elsewhere.
        Shared by thermal/reflect/ellipsoidal's identical opt-in gating.
        """
        off = [i for i in range(self.n_elements) if not opt_in_flags[i]]
        entry = {}
        if off:
            pin = np.full(self.n_elements, np.nan)
            pin[off] = 0.0
            entry["overrides"] = {"sigma": pin.tolist()}
        return entry

    def _may_be_nonzero(self, param):
        """True unless every element of `param` is pinned (sigma == 0) at
        exactly 0 -- the RESOLVED parameter state, after user params.

        Consumers (transit) use this to skip building an expensive graph
        (thermal: a second quad_solution_vector per planet; reflect: a
        planetvisible evaluation -- both as expensive as the transit
        itself) entirely when it can only ever evaluate to zero.
        Deliberately not `any(self.fit<x>)`: the manifest "overrides" pin
        is layered UNDER user params, so params.yaml can free the
        parameter (sigma > 0) or fix it at a nonzero value without the
        fit<x> flag, and the gate must stay open for those. Anything
        non-numeric (a linked expression, no sigma at all -> uniform
        prior) counts as active.
        """

        def _vec(value, fill):
            arr = np.atleast_1d(value if value is not None else fill)
            out = np.full(self.n_elements, np.nan)
            for i in range(self.n_elements):
                v = arr[i % len(arr)] if len(arr) else fill
                try:
                    out[i] = fill if v is None else float(v)
                except (TypeError, ValueError):
                    return None  # non-numeric (e.g. link) -> active
            return out

        sigmas = _vec(param.sigma, np.nan)
        inits = _vec(param.initval, 0.0)
        if sigmas is None or inits is None:
            return True
        pinned_at_zero = (sigmas == 0.0) & (inits == 0.0)
        return not bool(np.all(pinned_at_zero))

    def thermal_may_be_nonzero(self):
        """True unless every band's thermal is pinned at exactly 0 -- see
        _may_be_nonzero."""
        return self._may_be_nonzero(self.thermal)

    def reflect_may_be_nonzero(self):
        """True unless every band's reflect is pinned at exactly 0 -- see
        _may_be_nonzero."""
        return self._may_be_nonzero(self.reflect)

    def build_likelihood(self, model, system):
        pass

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
