import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.parameterization import (
    merge_overrides,
    mode_manifest,
    pin_unselected,
)
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

    # What each law's parameters ARE, as a parameterization mode table (see
    # components/parameterization.py): the quadratic law samples the Kipping
    # pair and derives (u1, u2) from it; the linear law samples u1 itself and
    # has no Kipping coordinates and no second coefficient at all.  Bands may
    # differ -- q1/q2 and u2 are then INACTIVE on the linear bands (not
    # parameters of theirs: pinned for bookkeeping, no potential, no table row)
    # and u1 is derived on the quadratic ones and sampled on the linear ones.
    LD_MODE_TABLE = {
        "quadratic": {
            "q1": None,
            "q2": None,
            "u1": "default",
            "u2": "default",
        },
        "linear": {"u1": None},
    }

    # The coordinate each law actually SAMPLES.  Read by the unread-band autopin
    # (a sigma on a derived element is a silent no-op, so pinning u1 on a
    # quadratic band would pin nothing) and by anything else that needs to know
    # which knob a band's limb darkening turns.
    LD_SAMPLED_PARAMS = {"quadratic": ("q1", "q2"), "linear": ("u1",)}

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
                    "Bands may declare different laws: a linear band has no "
                    "Kipping coordinates and no u2 (its second coefficient "
                    "is exactly 0). An unrecognized value raises."
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
        self.ld_laws = self._parse_ld_laws()
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
        """Per-band ``ld_law:``, validated.  Bands may differ.

        An unrecognized law is a hard error rather than a default:
        ``ld_law: quadratik`` used to satisfy ``law != "linear"`` and be
        modelled as quadratic -- a typo silently selecting a different sampled
        parameter set.

        A MIX of laws across bands used to be an error too, because
        ``Parameter.build_pymc`` derived ``is_derived`` for a whole vector, so
        one Band could not hold a Kipping-derived ``u1`` for some elements and a
        sampled ``u1`` for others; the workaround was quadratic everywhere with
        the linear bands' ``q2`` pinned at 0.5 (``u2 = sqrt(q1)(1 - 2 q2) = 0``),
        at the cost of a prior uniform in ``q1`` rather than in ``u1``.  Roles
        are per element now (see the manifest vocabulary), so ``register_
        parameters`` expresses the mix directly and this only validates.
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
        return laws

    def build_maps(self):
        self.star_map = np.array(self.star_indices, dtype=int)

    # Every place in the codebase that reads a Band's limb darkening, and the
    # condition under which it does.  Keep this list in sync with the grep for
    # `band.u1` / `band.u2` -- a new consumer that is not represented here
    # would have its band's LD silently pinned.
    #
    #   transit/transit.py       band.u1/u2[obs_band_map]  -- unconditional
    #   mulensing/mulensinstrument.py  band.u1[band_idx]   -- finite_source only
    #   rm.py (via rvinstrument `rm:`)  band.u1/u2[band_idx]
    #
    # astrometryinstrument's optional `band:` is deliberately absent: it uses
    # the band only for its filter identity (the SED photocenter fluxfrac),
    # never its limb darkening.
    def _ld_consumer_indices(self, system):
        """Band indices whose limb darkening something in this topology reads.

        Read from each consumer's raw ``config`` rather than from its parsed
        band map: ``MulensInstrument.band_map`` is built in *stage 2*
        (``register_parameters``), so whether it exists yet depends on
        component ordering, while ``Component.config`` is set in ``__init__``
        and is always available here.
        """
        name_to_idx = {name: i for i, name in enumerate(self.names)}
        consumers = set()

        def _mark(idx):
            if idx is not None and 0 <= idx < self.n_elements:
                consumers.add(idx)

        def _cfg(comp_name):
            comp = getattr(system, comp_name, None)
            return list(getattr(comp, "config", None) or [])

        # Transit: the occultation model cannot be computed without limb
        # darkening, so any transit referencing a band reads it.
        for c in _cfg("transit"):
            _mark(name_to_idx.get(c.get("band")))

        # Microlensing: the magnification only takes u1 when the source is
        # resolved.  `any` over the lens elements, not `[0]`, because that is
        # the conservative direction -- MulensInstrument.build_likelihood
        # currently gates on finite_source[0].
        finite_source = any(
            bool(c.get("finite_source", False)) for c in _cfg("lens")
        )
        if finite_source:
            # Every band a light curve references is marked, not just the
            # lowest-indexed one build_likelihood actually passes down (it
            # warns and uses the first).  Pinning on that tie-break would make
            # the pin an artifact of an acknowledged limitation.
            for c in _cfg("mulensinstrument"):
                _mark(name_to_idx.get(c.get("band")))

        # Rossiter-McLaughlin: rvinstrument `rm:` reads the `rm_band` band, or
        # band 0 when unset (see rm.resolve_rm_indices).
        for c in _cfg("rvinstrument"):
            if not c.get("rm"):
                continue
            rm_band = c.get("rm_band")
            _mark(0 if rm_band is None else name_to_idx.get(rm_band))

        return consumers

    def _pin_unread_limb_darkening(self, system, consumers=None):
        """Pin the LD parameters of the bands nothing in the topology reads.

        Only reached when SOME band is consumed: with no consumer at all
        the LD parameters are omitted from the manifest entirely
        (register_parameters), so a filter-identity-only band contributes
        no table rows.  This pin covers the mixed case -- the manifest is
        per parameter, so one consumed band forces the whole vector to
        exist, and the unread elements are fixed here.

        Which coordinate gets pinned is PER BAND, because which coordinate a
        band samples is: a quadratic band samples the Kipping pair (its u1/u2
        are derived from it, and a sigma on a derived element is a silent
        no-op), while a linear band samples u1 itself.  A single pin list would
        therefore either miss a linear band's only free parameter or write a
        no-op onto a quadratic band's derived one.

        The pin goes through the manifest "overrides" channel, which layers
        UNDER the params file (`sigma` takes apply_value's last-writer-wins
        `else` branch, and user params are applied after internal_overrides),
        so an explicit `band.<name>.q1: {sigma: 0.1}` still frees it -- someone
        may deliberately want a free LD to put a prior on.
        """
        if consumers is None:
            consumers = self._ld_consumer_indices(system)
        unread = [i for i in range(self.n_elements) if i not in consumers]
        if not unread:
            return

        # Per parameter, the elements that BOTH sample it and are unread; the
        # same opt-in pin the BEER terms and Instrument's GP/robust
        # registrations use (components/parameterization.py), merged into
        # whatever options the entry already carries.
        for param_name in ("q1", "q2", "u1", "u2"):
            if param_name not in self.manifest:
                continue
            samplers = [
                i
                for i in range(self.n_elements)
                if param_name in self.LD_SAMPLED_PARAMS[self.ld_laws[i]]
            ]
            keep = [i for i in samplers if i in consumers]
            if len(keep) == len(samplers):
                continue  # every band that samples this one is read
            pin = pin_unselected(self.n_elements, keep)
            # merge_overrides reads the entry through the manifest interpreter,
            # so adding an option cannot drop a bare-string expr_key and turn a
            # derived parameter into a sampled one (review 4.5.3).
            self.manifest[param_name] = merge_overrides(
                self.manifest.get(param_name), pin.get("overrides", {})
            )

        for i in unread:
            joined = "/".join(self.LD_SAMPLED_PARAMS[self.ld_laws[i]])
            first = self.LD_SAMPLED_PARAMS[self.ld_laws[i]][0]
            logger.info(
                f"[{self.prefix}] pinning {self.prefix}.{self.names[i]}."
                f"{joined} (sigma=0): nothing in this topology reads this "
                f"band's limb darkening. Only a transit, a finite-source "
                f"microlensing light curve, or an rvinstrument 'rm:' block "
                f"does; astrometry's 'band:' uses the filter identity only. "
                f"Give {self.prefix}.{self.names[i]}.{first} an entry "
                f"in the params file to sample it anyway."
            )

    def register_parameters(self, system):
        self.manifest = {}

        # Limb darkening enters the manifest ONLY when something in the
        # topology reads it (a transit, a finite-source microlensing light
        # curve, or an rvinstrument rm: block).  A band on a point-source
        # microlensing fit declares filter identity and nothing else --
        # such a fit used to carry pinned-Fixed LD rows in every table.
        # With SOME bands consumed, the whole vector must exist (the
        # manifest is per parameter); the unread elements are pinned by
        # _pin_unread_limb_darkening.
        consumers = self._ld_consumer_indices(system)
        if consumers:
            # Per band: a quadratic band samples the Kipping pair and derives
            # (u1, u2) from it; a linear band samples u1 directly and has no
            # Kipping coordinates and no u2 at all.  mode_manifest turns the
            # per-band laws into the masks that says so -- and for a system
            # that made ONE choice it returns exactly the manifest this code
            # used to write by hand, so those systems build an identical graph.
            self.manifest.update(
                mode_manifest(
                    self.ld_laws,
                    self.LD_MODE_TABLE,
                    n_elements=self.n_elements,
                    # A linear law's second coefficient is exactly zero, not
                    # "whatever the quadratic default was".
                    options={"u2": {"inactive_value": 0.0}},
                    where=f"{self.prefix}.ld_law",
                )
            )
            self._pin_unread_limb_darkening(system, consumers=consumers)
        else:
            logger.info(
                f"[{self.prefix}] no limb-darkening parameters: nothing in "
                f"this topology reads any band's limb darkening (only a "
                f"transit, a finite-source microlensing light curve, or an "
                f"rvinstrument 'rm:' block does)."
            )

        # thermal/reflect/ellipsoidal (all ppm, opt-in per band via
        # fitthermal/fitreflect/fitellip) enter the manifest ONLY when some
        # band opts in; in a mixed set the opted-out bands are pinned at
        # sigma=0 (the "overrides" pattern Instrument._register_gp uses),
        # so their value is exactly 0 and the transit model is unchanged.
        # With no opt-in anywhere the parameter does not exist at all --
        # no table row, and <x>_may_be_nonzero() is False by construction.
        # A params-file entry on band.<name>.<x> therefore requires the
        # fit<x> flag on that band (the entry alone used to free it).
        for name, flags in (
            ("thermal", self.fitthermal),
            ("reflect", self.fitreflect),
            ("ellipsoidal", self.fitellip),
        ):
            if any(flags):
                # pin_unselected is the ONE opt-in pin (see
                # components/parameterization.py); Instrument's GP and robust
                # registrations are the other two callers, and this was the
                # third line-for-line copy of it.  Routing through it also
                # retires the hand-written manifest read this used to do as a
                # WRITER (review 4.5.3): only reached when some band opts in,
                # so the entry it replaces is always a plain options dict.
                self.manifest[name] = pin_unselected(
                    self.n_elements, [i for i, on in enumerate(flags) if on]
                )

    def _may_be_nonzero(self, name):
        """True unless every element of parameter ``name`` is pinned
        (sigma == 0) at exactly 0 -- the RESOLVED state, after user params.

        Consumers (transit) use this to skip building an expensive graph
        (thermal: a second quad_solution_vector per planet; reflect: a
        planetvisible evaluation -- both as expensive as the transit
        itself) entirely when it can only ever evaluate to zero.
        With no fit<x> anywhere the parameter is not even in the manifest
        (register_parameters) and this is False outright; with a mixed set
        it is deliberately not `any(self.fit<x>)`: the manifest
        "overrides" pin is layered UNDER user params, so params.yaml can
        free an opted-out band's element (sigma > 0) or fix it at a
        nonzero value, and the gate must stay open for those. Anything
        non-numeric (a linked expression, no sigma at all -> uniform
        prior) counts as active.
        """
        if name not in self.manifest:
            return False
        param = getattr(self, name)

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
        return self._may_be_nonzero("thermal")

    def reflect_may_be_nonzero(self):
        """True unless every band's reflect is pinned at exactly 0 -- see
        _may_be_nonzero."""
        return self._may_be_nonzero("reflect")

    def ellipsoidal_may_be_nonzero(self):
        """True unless every band's ellipsoidal is pinned at exactly 0 --
        see _may_be_nonzero."""
        return self._may_be_nonzero("ellipsoidal")

    def build_likelihood(self, model, system):
        pass

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
