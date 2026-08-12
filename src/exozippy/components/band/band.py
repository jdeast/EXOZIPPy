import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.sed.bc_grid import (
    AMBIGUOUS_FILTER_ALIASES,
    _load_alias_table,
    facility_from_svo_name,
    resolve_filter_name,
)

logger = logging.getLogger(__name__)


class Band(Component):
    """Photometric band with limb-darkening coefficients.

    One Band instance per filter. Instruments reference a band by name.
    Supports linear (sample u1) and quadratic Kipping (sample q1/q2, derive u1/u2) laws.
    All bands must declare the same law: the limb-darkening manifest is
    shared by the whole band vector, so the two laws cannot coexist (see
    `_parse_ld_laws`).

    Band is the single carrier of filter identity for instruments: each
    element's user-facing `filter:` string is resolved through the shared
    filter alias table (exozippy/filters/filternames.txt) into canonical
    MIST (`filter_mist`) and SVO (`filter_svo`) names, which the SED
    flux-prediction hooks (mulensing f_source constraint, transit
    deblending, astrometry fluxfrac) key on.
    """

    yaml_key = "band"

    # Accepted `ld_law:` spellings. An unrecognized value raises rather than
    # falling through to the quadratic branch: a silently ignored law key is
    # the same bug class as `IMF: Salpeter` (PR #82), and here it would also
    # silently change the sampled parameter set (q1/q2 instead of u1).
    LD_LAWS = ("quadratic", "linear")

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
                    "Filter/bandpass name; resolved through the shared "
                    "filter alias table (exozippy/filters/filternames.txt) "
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
                "accepts": list(cls.LD_LAWS),
                "required": False,
                "doc": (
                    "Limb-darkening law. Default 'quadratic' (Kipping "
                    "q1/q2, deriving u1/u2); 'linear' samples u1 directly. "
                    "Every band must declare the same law -- the "
                    "limb-darkening manifest is shared by the whole band "
                    "vector. An unrecognized value, or a mix of laws, "
                    "raises."
                ),
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
                    "flag; see PR 1.b."
                ),
            },
        ]

    def load_data(self, system):
        self.filter_names = [c.get("filter", "") for c in self.config]
        self.star_indices = [c.get("star_ndx", 0) for c in self.config]
        self.ld_laws = self._parse_ld_laws()
        self.fitthermal = [
            bool(c.get("fitthermal", False)) for c in self.config
        ]

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
                    f"filter alias table (exozippy/filters/"
                    f"filternames.txt); assuming it is already a canonical "
                    f"BC-table/SVO name."
                )
            elif alias_df is not None and filt in AMBIGUOUS_FILTER_ALIASES:
                # INFO, not a warning, and deliberately so.  The label names
                # two rows of the alias table, so SOMETHING has to choose --
                # but the choice is now a documented project convention
                # (AMBIGUOUS_FILTER_ALIASES) rather than the row order it
                # used to be, and "I" is how essentially every ground-based
                # microlensing survey spells its band.  A warning on every
                # such fit would be noise on a config that is doing nothing
                # wrong, and the standing lesson from the FFP logmass bound
                # is that warnings people cannot act on teach them to ignore
                # warnings.  Stating the resolution once per band is enough
                # for a user who meant the other one to notice.
                logger.info(
                    f"Band '{band_name}': filter '{filt}' names more than "
                    f"one row of the filter alias table; resolving it to "
                    f"'{self.filter_svo[-1]}' by the project convention "
                    f"that a bare '{filt}' is the Cousins band. Write the "
                    f"filter out (e.g. 'Generic/Bessell.{filt}' or "
                    f"'Generic/Cousins.{filt}') to say so explicitly."
                )

    def _parse_ld_laws(self):
        """Per-band ``ld_law:``, validated, and required to be uniform.

        Two things are deliberately hard errors here rather than defaults:

        * **An unrecognized law.** ``ld_law: quadratik`` used to satisfy
          ``law != "linear"`` and be modelled as quadratic -- a typo silently
          selecting a different sampled parameter set.
        * **A mix of laws across bands.** The manifest is per *parameter*, not
          per element: ``Parameter.build_pymc`` derives ``is_derived`` from
          ``expression is not None`` for the whole vector, so one Band cannot
          hold a derived ``u1`` (Kipping, quadratic) for some elements and a
          sampled ``u1`` for others. The old ``any(law != "linear")`` picked
          the quadratic manifest for everyone, which handed every band a free
          ``u2`` -- silently modelling a user's declared-linear band as
          quadratic. Per-element derivation would need the manifest ``mask``
          field (declared in ``Parameter``, not yet consumed); until that
          exists, raising is the only non-silent option.
        """
        laws = []
        for i, c in enumerate(self.config):
            law = c.get("ld_law", "quadratic")
            norm = str(law).strip().lower()
            if norm not in self.LD_LAWS:
                raise ValueError(
                    f"{self.prefix}.{self.names[i]}: unknown ld_law "
                    f"{law!r}. Accepted: {', '.join(self.LD_LAWS)} "
                    f"('quadratic' = Kipping q1/q2, the default; 'linear' = "
                    f"sample u1 directly)."
                )
            laws.append(norm)

        if len(set(laws)) > 1:
            detail = ", ".join(
                f"{name}: {law}" for name, law in zip(self.names, laws)
            )
            raise ValueError(
                f"[{self.prefix}] all bands must use the same ld_law; got "
                f"{detail}. A mixed-law system is not supported: the "
                f"limb-darkening manifest is shared by every band element, so "
                f"one law has to be chosen for the whole vector, and the "
                f"quadratic choice would give the 'linear' bands a free u2 "
                f"(silently modelling them as quadratic). Use one law "
                f"everywhere -- 'quadratic' with the linear bands' q2 pinned "
                f"at 0.5 in the params file reproduces a linear law exactly "
                f"(u2 = sqrt(q1)*(1 - 2*q2) = 0), at the cost of a prior "
                f"uniform in q1 rather than in u1."
            )
        return laws

    def build_maps(self):
        self.star_map = np.array(self.star_indices, dtype=int)

    def register_parameters(self, system):
        # Uniform by construction (_parse_ld_laws raises on a mix), so the
        # first element's law is the system's law.
        if self.ld_laws and self.ld_laws[0] == "linear":
            self.manifest = {
                "u1": None,
            }
        else:
            self.manifest = {
                "q1": None,
                "q2": None,
                "u1": "default",
                "u2": "default",
            }

        # thermal (secondary-eclipse depth, ppm) is opt-in per band via
        # fitthermal: true. Bands that don't opt in are pinned at sigma=0
        # (same "overrides" pattern Instrument._register_gp uses for terms
        # an element didn't ask for), so thermal.value is exactly 0 and the
        # transit model is unchanged unless a band explicitly asks for it.
        off = [i for i in range(self.n_elements) if not self.fitthermal[i]]
        thermal_entry = {}
        if off:
            pin = np.full(self.n_elements, np.nan)
            pin[off] = 0.0
            thermal_entry["overrides"] = {"sigma": pin.tolist()}
        self.manifest["thermal"] = thermal_entry

    def thermal_may_be_nonzero(self):
        """True unless every band's thermal is pinned (sigma == 0) at
        exactly 0 -- the RESOLVED parameter state, after user params.

        Consumers (transit) use this to skip building the thermal graph
        entirely (a second quad_solution_vector per planet, as expensive
        as the transit itself) when it can only ever evaluate to zero.
        Deliberately not `any(self.fitthermal)`: the manifest "overrides"
        pin is layered UNDER user params, so params.yaml can free thermal
        (sigma > 0) or fix it at a nonzero value without fitthermal, and
        the gate must stay open for those. Anything non-numeric (a linked
        expression, no sigma at all -> uniform prior) counts as active.
        """
        param = self.thermal

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

    def build_likelihood(self, model, system):
        pass

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
