"""The red clump of a microlensing sight line as a reference object
(review 8.9.11; redclump.md).

One instance stands for the CENTROID of the field red clump on the event's
sight line: a population object with a known intrinsic color, (V-I)_0 =
1.06 +/- 0.06 (Bensby et al. 2013), and luminosity, M_I = -0.12 (Nataf et
al. 2013), at the galactic model's bar distance.  It is not a Star (no
mass, no IMF, no tracks, no kinematics); the SED evaluates it exactly as it
evaluates a star through the flux-source seam (``Component.
sed_flux_sources``), so an SED row may name it -- the calibrated I_RC and
(V-I)_RC of a CMD, or the source-minus-clump offsets Delta I and
Delta(V-I) -- and its extinction is a forward-model quantity, integrated
through each passband at the fit's own law.
"""

import logging

import numpy as np
import pytensor.tensor as pt

from exozippy.components.component import Component, SEDFluxSource
from exozippy.config import user_entry
from exozippy.constants import LOG_L0_CONST, LUM_CONST
from exozippy.outputs.prose import get_collector

from . import physics  # noqa: F401 -- registers the clump's expressions

logger = logging.getLogger(__name__)

#: The clump's intrinsic-color filters: OGLE's V is Johnson, its I Cousins.
V_FILTER = "Generic/Johnson.V"
I_FILTER = "Generic/Cousins.I"

#: The clump's R_V parameter (sed.RV_DUST; not imported at module level,
#: the SED imports the component layer).
RV_DUST_KEY = "rv_dust"

#: Off the bulge there is no clump (background.py's bulge box).
BULGE_BOX_DEG = 10.0

#: The bar-distance prior's width, in magnitudes of distance modulus: the
#: systematic of the clump CENTROID (galacticmodel.md, "The sight-line
#: queries": the port's bar and genulens's coded clump formula agree to
#: <= 0.08 mag), not the bar's depth, which is the spread of single stars.
BAR_DISTANCE_SIGMA_MAG = 0.1

#: |port bar mean - genulens's red-clump formula| above which the two
#: definitions of the clump distance disagree enough to say so.
BAR_FORMULA_TOLERANCE_MAG = 0.15

#: Start A_I mismatch with galacticmodel's static source_ai_rc to warn on.
SOURCE_AI_RC_TOLERANCE_MAG = 0.2

#: Default parameter citations, set as manifest options unless the user
#: states their own (a defaults.yaml cannot carry one: config.CITATION_KEYS).
CITATIONS = {
    "feh": ("Hill:2011",),
    "mi": ("Nataf:2013",),
    "vi0": ("Bensby:2013",),
    "av_ratio": ("Nataf:2013",),
}


class RedClump(Component):
    """The red-clump reference of one sight line (redclump.md)."""

    CONFIG_KEYS = ("name", "clump_constrains_source_av", "bands", "colors")
    context_dep_names = frozenset(
        {
            "bc_v0",
            "bc_i0",
            "ext_band",
            "ext_color",
            "profile_av",
            "profile_rv_dust",
            "sightline_rv_dust",
        }
    )
    aligned_context_deps = context_dep_names
    per_build_caches = ("_context_cache",)

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Red Clump"
        self._context_cache = None
        if self.n_elements != 1:
            raise ValueError(
                f"redclump takes exactly one block, got {self.n_elements}: "
                f"one clump describes one sight line."
            )
        cfg = self.config[0] or {}
        where = f"[{self.prefix}.{self.names[0]}]"
        unknown = sorted(set(cfg) - set(self.CONFIG_KEYS))
        if unknown:
            raise ValueError(
                f"{where} unknown key(s) {unknown}; a redclump block takes "
                f"{list(self.CONFIG_KEYS)}."
            )
        sys_cfg = getattr(config_manager, "system_config", None) or {}
        for need, why in (
            ("sed", "the clump is evaluated, and read, only through the SED"),
            (
                "galacticmodel",
                "the sight line is the galactic model's anchor and the "
                "clump's distance prior is its bar",
            ),
        ):
            if need not in sys_cfg:
                raise ValueError(
                    f"{where} needs a '{need}:' block: {why} (redclump.md)."
                )
        flag = cfg.get("clump_constrains_source_av")
        if flag is not None and not isinstance(flag, bool):
            raise ValueError(
                f"{where} clump_constrains_source_av must be true or false "
                f"(a YAML boolean); got {flag!r}."
            )
        self.tie_requested = flag
        band_names = [
            (b or {}).get("name", str(i))
            for i, b in enumerate(sys_cfg.get("band") or [])
        ]
        bands = cfg.get("bands") or []
        if not isinstance(bands, list) or not all(
            isinstance(b, str) for b in bands
        ):
            raise ValueError(
                f"{where} bands: must be a list of band-block names; got "
                f"{bands!r}."
            )
        if len(set(bands)) != len(bands):
            raise ValueError(f"{where} bands: names a band twice: {bands}.")
        colors = cfg.get("colors") or []
        if not isinstance(colors, list) or not all(
            isinstance(c, list)
            and len(c) == 2
            and all(isinstance(x, str) for x in c)
            and c[0] != c[1]
            for c in colors
        ):
            raise ValueError(
                f"{where} colors: must be a list of [blue, red] pairs of "
                f"two different band-block names; got {colors!r}."
            )
        for b in bands + [x for c in colors for x in c]:
            if b not in band_names:
                raise ValueError(
                    f"{where} names band {b!r}, which no band: block "
                    f"declares (band blocks: {band_names}).  A band carries "
                    f"the filter identity (sed.md, 'Bands, filters')."
                )
        self.bands = list(bands)
        self.colors = [tuple(c) for c in colors]
        self.tied = []
        self.tied_map = np.zeros(0, dtype=int)

    @property
    def prefix(self):
        return "redclump"

    @property
    def source_name(self):
        """The spelling an SED row's photType uses for the clump."""
        return self.names[0]

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "name",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "The clump's name, which SED rows use in photType "
                    "(must not be a star's name)."
                ),
            },
            {
                "key": "clump_constrains_source_av",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "The interim soft tie A_V(source) = A_V(clump) "
                    "(1 +/- 9%, Nataf et al. 2013) on every microlensing "
                    "source. Default true; false for a source in front of "
                    "the bar."
                ),
            },
            {
                "key": "bands",
                "kind": "ref",
                "accepts": ["band"],
                "required": False,
                "doc": (
                    "Band blocks whose clump extinction is reported "
                    "(redclump.<name>.extinction_<k>); a published clump "
                    "extinction is a prior on that element."
                ),
            },
            {
                "key": "colors",
                "kind": "ref",
                "accepts": ["band"],
                "required": False,
                "doc": (
                    "[blue, red] band pairs whose clump color excess is "
                    "reported (redclump.<name>.color_excess_<k>)."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # The SED seams (construction-time answers, asked at the SED's stage 1)
    # ------------------------------------------------------------------
    def _path(self, param):
        return f"{self.prefix}.0.{param}"

    def sed_flux_sources(self, system):
        from exozippy.components.sed.sed import RV_DUST, sed_fits_rv_dust

        paths = {
            "teffsed": self._path("teff"),
            "loggsed": self._path("logg"),
            "feh": self._path("feh"),
            "av": self._path("av"),
            "radiussed": self._path("radius"),
            "distance": self._path("distance"),
        }
        if sed_fits_rv_dust(system):
            # The clump's own R_V: the sight line's sed.rv_dust, or under
            # the dust profile's near/far split the mix in front of it.
            paths[RV_DUST] = self._path(RV_DUST)
        # teff, feh, av and one flux scale (mi + distance): a star's count.
        # The clump's distance is where the sight-line dust profile is
        # anchored: its extinction is the measured one (sed.md).
        return [
            SEDFluxSource(
                self.source_name,
                paths,
                photometric_dof=4,
                extinction_anchor=True,
            )
        ]

    def sed_flux_filters(self, system):
        return {V_FILTER, I_FILTER}

    def sed_flux_bands(self, system):
        return set(self.bands) | {x for c in self.colors for x in c}

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    def register_parameters(self, system):
        sed = system.sed
        gm = system.galacticmodel
        cm = self.config_manager
        where = f"[{self.prefix}.{self.names[0]}]"

        # The sight line, and the bar on it (galacticmodel's public queries).
        l_deg, b_deg = gm.sight_line(system)
        l_signed = (l_deg + 180.0) % 360.0 - 180.0
        if abs(l_signed) >= BULGE_BOX_DEG or abs(b_deg) >= BULGE_BOX_DEG:
            raise ValueError(
                f"{where} the sight line (l, b) = ({l_signed:.2f}, "
                f"{b_deg:.2f}) deg is outside the bulge box (|l|, |b| < "
                f"{BULGE_BOX_DEG:g} deg): there is no red clump there."
            )
        mean_mu, sd_mu, mode_mu = gm.bar_distance_moments(system)
        from exozippy.components.galacticmodel import koshimoto21 as k21

        formula = k21.red_clump_distance_modulus(l_deg, b_deg)
        if abs(mean_mu - formula) > BAR_FORMULA_TOLERANCE_MAG:
            logger.warning(
                f"{where} the galactic model's bar puts the clump at "
                f"distance modulus {mean_mu:.3f}, genulens's red-clump "
                f"formula at {formula:.3f} ({mean_mu - formula:+.3f} mag): "
                f"more than {BAR_FORMULA_TOLERANCE_MAG} apart on this sight "
                f"line.  The prior uses the bar's; state "
                f"redclump.distance {{mu, sigma}} to override it."
            )
        d_bar = 10.0 ** (0.2 * mean_mu + 1.0)
        sig_d = BAR_DISTANCE_SIGMA_MAG * np.log(10.0) / 5.0 * d_bar
        self.bar_moments = (mean_mu, sd_mu, mode_mu)
        logger.info(
            f"{where} sight line (l, b) = ({l_signed:.3f}, {b_deg:.3f}) deg; "
            f"bar distance modulus mean {mean_mu:.3f} (mode {mode_mu:.3f}, "
            f"depth {sd_mu:.3f} mag of single stars): clump distance prior "
            f"N({d_bar:.0f}, {sig_d:.0f}) pc."
        )

        self.manifest = {
            "teff": None,
            "logg": None,
            "feh": None,
            "mi": None,
            "distance": {
                "overrides": {"mu": [d_bar], "sigma": [sig_d]},
            },
            "av": None,
            "radius": "default",
            "vi0": "default",
        }
        # Under the SED's sight-line dust profile the clump's A_V is the
        # profile at its distance -- the anchor -- times its field-scale
        # granularity, 1 + av_frac (Nataf et al. 2013's 9%: the clump
        # CENTROID measures the pencil-beam profile to that; sed.md).
        self.dust_plan = sed.dust_plan(system)
        if self.dust_plan.on:
            self.manifest["av"] = {
                "expr_key": "from_dust_profile",
                "lower": [
                    float("nan")
                    if self.user_wrote_field("av", "lower")[0]
                    else -np.inf
                ],
            }
            self.manifest["av_frac"] = None
        if sed.fit_rv_dust:
            self.manifest[RV_DUST_KEY] = (
                {"expr_key": "from_dust_profile", "print_to_table": True}
                if self.dust_plan.on and self.dust_plan.split
                else "from_sightline"
            )
        cm.add_hint(self._path("distance"), d_bar)
        n_b, n_c = len(self.bands), len(self.colors)
        owner = self.names[0]
        if n_b:
            self.manifest["extinction"] = {
                "expr_key": "default",
                "shape": (n_b,),
                "element_owners": [owner] * n_b,
            }
        if n_c:
            self.manifest["color_excess"] = {
                "expr_key": "default",
                "shape": (n_c,),
                "element_owners": [owner] * n_c,
            }
        self._register_tie(system, where)
        for param, keys in CITATIONS.items():
            if (
                param in self.manifest
                and not self.user_wrote_field(param, "citation").any()
            ):
                self.manifest[param] = self._with_option(
                    self.manifest[param], citation=keys
                )

        # The coordinates the clump's rows pin decide at
        # PRECEDENCE_LIKELIHOOD_BACKED (components.md, item 5).
        if not self.dust_plan.on:
            cm.declare_likelihood_backed(self._path("av"))
        cm.declare_likelihood_backed(self._path("teff"))

        self._check_read(system, sed, where)
        self._seed_av(system, sed, where)
        if not sed.fit_rv_dust:
            # JDE 2026-11-09: a warning, kept short (redclump.md says why).
            logger.warning(
                f"{where} redclump without sed.fit_rv_dust: a fixed R_V may "
                f"bias Teff/R_star (redclump.md)."
            )

    @staticmethod
    def _with_option(entry, **opts):
        if entry is None:
            return dict(opts)
        if isinstance(entry, str):
            return {"expr_key": entry, **opts}
        return {**entry, **opts}

    def _register_tie(self, system, where):
        """The interim soft av tie (redclump.md; PR 3's dust profile
        replaces it): which stars' A_V is tied to the clump's."""
        sed = system.sed
        flag = self.tie_requested
        self.tied = []
        if self.dust_plan.on:
            if flag is True:
                raise ValueError(
                    f"{where} clump_constrains_source_av: true is the interim "
                    f"tie the SED's sight-line dust profile replaces (every "
                    f"star's A_V is A(d) anchored at the clump).  Remove the "
                    f"key, or set 'sed: {{distance_constrains_av: false}}' "
                    f"to use the tie instead of the profile."
                )
            logger.info(
                f"{where} the sight-line dust profile ties every object's "
                f"A_V to the clump's anchor; no interim av_ratio tie."
            )
            return
        if flag is False:
            logger.info(
                f"{where} clump_constrains_source_av: false -- no source's "
                f"A_V is tied to the clump's."
            )
            return
        selected = set()
        for comp in system.active_components.values():
            selected |= {int(k) for k in comp.selected_sources(system)}
        star = system.star
        n = int(star.n_elements)
        # A star whose distance is HARD-linked to a selected source (a 1L2S
        # second source) is on the same sight line at the same distance.
        cm = self.config_manager
        for k in range(n):
            link = (cm.links.get(f"star.{k}.distance") or {}).get("initval")
            entry = user_entry(cm.user_params or {}, f"star.{k}.distance")
            if (
                link is not None
                and (entry or {}).get("sigma") == 0
                and len(link.dep_paths) == 1
            ):
                parts = link.dep_paths[0].split(".")
                if (
                    len(parts) == 3
                    and parts[0] == "star"
                    and parts[2] == "distance"
                    and parts[1].isdigit()
                    and int(parts[1]) in selected
                ):
                    selected.add(k)
        # An orbit-group member's av IS its anchor's: tie the anchor only.
        members = set(star.orbit_anchors(system))
        seen = sed.seen_star_mask(system)
        tied, unread = [], []
        for k in sorted(selected - members):
            (tied if seen[k] else unread).append(k)
        if unread:
            logger.info(
                f"{where} not tying {[star.names[k] for k in unread]}: "
                f"nothing reads their SED flux, so their A_V is pinned."
            )
        if not tied:
            if flag is True:
                raise ValueError(
                    f"{where} clump_constrains_source_av: true, but no "
                    f"component selected a source whose SED flux is read "
                    f"(a microlensing source with an SED row or a stated "
                    f"zeropoint tie).  Remove the key, or read the source."
                )
            logger.info(
                f"{where} no selected source to tie to the clump's A_V; "
                f"the tie is off."
            )
            return
        self.tied = tied
        self.tied_map = np.asarray(tied, dtype=int)
        self.manifest["av_ratio"] = {
            "expr_key": "default",
            "shape": (len(tied),),
            "element_owners": [self.names[0]] * len(tied),
        }
        logger.info(
            f"{where} A_V of {[star.names[k] for k in tied]} tied to the "
            f"clump's: av_ratio ~ N(1, 0.09) (Nataf et al. 2013's "
            f"extinction granularity; clump_constrains_source_av: false "
            f"turns it off)."
        )

    def _extinction_priors(self):
        """Params-file priors (mu/sigma) on the clump's extinction or
        color-excess elements -- a published clump extinction."""
        out = []
        prefix = f"{self.prefix}."
        for key, entry in (self.config_manager.user_params or {}).items():
            if not (isinstance(entry, dict) and key.startswith(prefix)):
                continue
            param = key.split(".")[-1]
            base = param.rsplit("_", 1)[0]
            if base in ("extinction", "color_excess") and (
                "mu" in entry or "sigma" in entry
            ):
                out.append(key)
        for key in getattr(self.config_manager, "links", {}) or {}:
            param = key.split(".")[-1]
            if key.startswith(prefix) and param.rsplit("_", 1)[0] in (
                "extinction",
                "color_excess",
            ):
                out.append(key)
        return out

    def _clump_rows(self, sed):
        """Indices of the SED rows naming the clump (either side)."""
        col = int(sed.nstars) + [p.name for p in sed.providers].index(
            self.source_name
        )
        bm = np.asarray(sed.pos_matrix) | np.asarray(sed.neg_matrix)
        return [int(i) for i in np.flatnonzero(bm[:, col])], col

    def _check_read(self, system, sed, where):
        """SOMETHING must read the clump: an SED row naming it, or a prior
        on one of its extinction elements.  And say when nothing CALIBRATES
        it: differential rows alone fix the source relative to the clump,
        not the extinction (redclump.md, "What the clump measures")."""
        rows, col = self._clump_rows(sed)
        priors = self._extinction_priors()
        if not rows and not priors:
            raise ValueError(
                f"{where} the red clump is read by no SED row and no prior: "
                f"name it ({self.source_name!r}) in a .sed row's photType, "
                f"or state a published clump extinction as a prior on "
                f"redclump.{self.names[0]}.extinction_<k> (bands:), or "
                f"remove the block."
            )
        absolute = [
            i
            for i in rows
            if not np.asarray(sed.neg_matrix)[i].any() or sed.color_rows[i]
        ]
        if not absolute and not priors:
            logger.info(
                f"{where} only differential rows read the clump: they fix "
                f"the source's color and magnitude RELATIVE to it, not the "
                f"extinction, so the clump's A_V is constrained only by "
                f"what calibrates the source (a stated zeropoint tie, an "
                f"absolute source row) and the av tie.  A calibrated "
                f"I_RC / (V-I)_RC row naming the clump measures it."
            )

    # ------------------------------------------------------------------
    # The A_V start, from the clump's own calibrated rows
    # ------------------------------------------------------------------
    def _start(self, param):
        cfg = self.config_manager.resolve(self.prefix, param, element=0)
        return float(np.atleast_1d(cfg["initval"])[0])

    def _seed_av(self, system, sed, where):
        """Start A_V where the clump's calibrated rows put it: a color row
        naming the clump on both sides (its (V-I)_RC: distance-free), else
        an absolute row naming only the clump.  A component hint, so a user
        initval wins.  Numeric, through the SED's own grid (``bc_np``)."""
        from scipy.optimize import brentq

        if self.user_wrote_field("av", "initval").any():
            return
        if self.dust_plan.on and (
            sed.user_wrote_field("dust_anchor", "initval").any()
            # a user's A_V start on a star already seeded the anchor (a
            # higher-precedence hint; the last add_hint would win)
            or getattr(sed, "_dust_seed", None)
        ):
            return
        rows, col = self._clump_rows(sed)
        pos, neg = np.asarray(sed.pos_matrix), np.asarray(sed.neg_matrix)
        only = np.zeros(pos.shape[1], dtype=int)
        only[col] = 1
        colour = [
            i
            for i in rows
            if sed.color_rows[i]
            and (pos[i] == only).all()
            and (neg[i] == only).all()
        ]
        absolute = [
            i for i in rows if (pos[i] == only).all() and not neg[i].any()
        ]
        if not colour and not absolute:
            return
        teff, logg, feh = (self._start(p) for p in ("teff", "logg", "feh"))
        rv = None
        if sed.fit_rv_dust:
            from exozippy.components.sed.sed import RV_DUST

            rv = float(
                np.atleast_1d(
                    self.config_manager.resolve("sed", RV_DUST, element=0)[
                        "initval"
                    ]
                )[0]
            )
        lo = self._start_bound("av", "lower")
        hi = self._start_bound("av", "upper")
        if sed.grid_axes is not None and isinstance(sed.grid_axes, dict):
            hi = min(hi, float(sed.grid_axes["av_pts"].max()))
        if colour:
            i = colour[0]
            a = sed.row_filters[i]
            b = sed.row_filters[int(sed.neg_col[i])]

            def resid(av):
                return (
                    sed.bc_np(b, teff, logg, feh, av, rv)
                    - sed.bc_np(a, teff, logg, feh, av, rv)
                    - sed.mag[i]
                )

            what = f"its color row {i} ({a} - {b})"
        else:
            i = absolute[0]
            mi = self._start("mi")
            dist = self._start("distance")
            bc_i0 = sed.bc_np(I_FILTER, teff, logg, feh, 0.0, rv)
            lum = 10.0 ** (-0.4 * (mi + bc_i0 - LOG_L0_CONST))
            radius = np.sqrt(lum / (LUM_CONST * teff**4))

            def resid(av):
                return (
                    sed.appmag_np(
                        sed.filters[i], teff, logg, feh, av, radius, dist, rv
                    )
                    - sed.mag[i]
                )

            what = f"its absolute row {i} ({sed.filters[i]})"
        f_lo, f_hi = resid(lo), resid(hi)
        if f_lo * f_hi > 0:
            logger.warning(
                f"{where} no A_V in [{lo:g}, {hi:g}] reproduces {what} at "
                f"the clump's start (Teff {teff:.0f} K); A_V starts at its "
                f"default."
            )
            return
        av = float(brentq(resid, lo, hi, xtol=1e-6))
        if self.dust_plan.on:
            # The clump's A_V is derived from the profile's anchor, which IS
            # A(d_clump) (av_frac starts at 0): seed the anchor, in its band.
            band = self.dust_plan.anchor_band
            value = (
                av
                if band is None
                else sed.reference_band_extinction_np(band, av, system, rv)
            )
            self.config_manager.add_hint("sed.dust_anchor", value)
            logger.info(
                f"{where} the dust anchor starts at {value:.3f} mag (clump "
                f"A_V {av:.3f}), from {what}."
            )
            return
        self.config_manager.add_hint(self._path("av"), av)
        logger.info(f"{where} A_V starts at {av:.3f} mag, from {what}.")

    def _start_bound(self, param, field):
        cfg = self.config_manager.resolve(self.prefix, param, element=0)
        return float(np.atleast_1d(cfg[field])[0])

    # ------------------------------------------------------------------
    # Stage 6: the context nodes off the SED's grid
    # ------------------------------------------------------------------
    def add_parameter(self, model, param_name, system, context_nodes=None):
        """Inject the clump's bolometric corrections -- the SED's one grid
        evaluation (``SED.bc_node``) at the clump's own coordinates: V and I
        at A_V = 0 for (V-I)_0 and the radius, and every ``bands:`` filter
        at A_V = 0 and at its A_V for the extinctions."""
        if param_name in ("radius", "vi0", "extinction", "color_excess"):
            if not context_nodes:
                context_nodes = self._context(model, system)
        if not context_nodes and param_name in ("av", RV_DUST_KEY):
            context_nodes = self._sight_line_context(model, system, param_name)
        return super().add_parameter(model, param_name, system, context_nodes)

    def _sight_line_context(self, model, system, param_name):
        """The clump's A_V and R_V from the SED's sight line: under the dust
        profile A(d_clump) (``profile_av``) and, under its near/far split,
        the R_V mixed in front of the clump; else sed.rv_dust itself."""
        sed = system.sed
        if param_name == "av":
            if not self.dust_plan.on:
                return None
            d = self._local_node(model, system, "distance")
            return {"profile_av": sed.profile_av_node(system, d)}
        if self.dust_plan.on and self.dust_plan.split:
            d = self._local_node(model, system, "distance")
            return {"profile_rv_dust": sed.profile_rv_node(system, d)}
        return {
            "sightline_rv_dust": sed.source_input_node(
                system, f"sed.0.{RV_DUST_KEY}"
            )[None]
        }

    def _local_node(self, model, system, name):
        if not Component._parameter_is_current(self, name, model):
            self.add_parameter(model, name, system)
        return getattr(self, name).value

    def _band_filter(self, system, band):
        b = system.band
        return b.filter_mist[list(b.names).index(band)]

    def _context(self, model, system):
        cached = self._context_cache
        if cached is not None and cached[0] is model:
            return cached[1]
        sed = system.sed
        teff, logg, feh = (
            self._local_node(model, system, p) for p in ("teff", "logg", "feh")
        )
        rv = None
        if sed.fit_rv_dust:
            rv = self._local_node(model, system, RV_DUST_KEY)
        bc0 = sed.bc_node(teff, logg, feh, pt.zeros_like(teff), rv)[0]
        out = {
            "bc_v0": bc0[sed.filter_column(V_FILTER)][None],
            "bc_i0": bc0[sed.filter_column(I_FILTER)][None],
        }
        names = sorted(set(self.bands) | {x for c in self.colors for x in c})
        if names:
            av = self._local_node(model, system, "av")
            bc_av = sed.bc_node(teff, logg, feh, av, rv)[0]
            ext = {
                b: bc0[sed.filter_column(self._band_filter(system, b))]
                - bc_av[sed.filter_column(self._band_filter(system, b))]
                for b in names
            }
            if self.bands:
                out["ext_band"] = pt.stack([ext[b] for b in self.bands])
            if self.colors:
                out["ext_color"] = pt.stack(
                    [ext[a] - ext[b] for a, b in self.colors]
                )
        self._context_cache = (model, out)
        return out

    # ------------------------------------------------------------------
    # Stage 7
    # ------------------------------------------------------------------
    def build_likelihood(self, model, system):
        """No likelihood of its own: the SED rows naming it, the derived
        Gaussians (vi0, av_ratio, any extinction prior) and the distance
        prior are the clump's terms.  Here: the luminosity-function
        mismatch check and the draft's sentences."""
        self._ai_check = self._uses_static_ai_rc(system)
        self._check_source_ai_rc(system)
        self._add_prose(system)

    def _uses_static_ai_rc(self, system):
        gm = getattr(system, "galacticmodel", None)
        return getattr(gm, "source_selection", None) == "luminosity_function"

    def _warn_ai_rc(self, system, a_i, when):
        """galacticmodel's luminosity-function source selection carries a
        STATIC A_I to the red clump (``source_ai_rc``); with a clump in the
        fit that is also a fitted quantity.  The selection stays static
        (JDE 2026-11-09); the remedy is the user's -- update it and rerun --
        so the warning gives the clump's own A_I as the value to use."""
        gm = system.galacticmodel
        if abs(a_i - gm.source_ai_rc) <= SOURCE_AI_RC_TOLERANCE_MAG:
            return
        logger.warning(
            f"[{self.prefix}] the clump's {when} A_I = {a_i:.3f} differs from "
            f"galacticmodel.source_ai_rc = {gm.source_ai_rc:.3f} by "
            f"{a_i - gm.source_ai_rc:+.3f} mag (> "
            f"{SOURCE_AI_RC_TOLERANCE_MAG} mag).  The luminosity-function "
            f"source selection uses the static value: set "
            f"galacticmodel.source_ai_rc: {a_i:.3f} and rerun."
        )

    def _check_source_ai_rc(self, system):
        """The start check (``_warn_ai_rc``); the posterior-median one is
        ``check_at_point``."""
        if not self._uses_static_ai_rc(system):
            return
        sed = system.sed
        teff, logg, feh = (
            float(np.atleast_1d(getattr(self, p).initval)[0])
            for p in ("teff", "logg", "feh")
        )
        rv = None
        if sed.fit_rv_dust:
            # The sight line's start R_V (the clump's own is derived; under
            # the split it differs from this by the near mix -- a start
            # check, warn-only).
            rv = float(np.atleast_1d(sed.rv_dust.initval)[0])
        if self.dust_plan.on:
            # Derived: A(d_clump) = the anchor at the start (av_frac 0).
            anchor = float(np.atleast_1d(sed.dust_anchor.initval)[0])
            band = self.dust_plan.anchor_band
            if band is None:
                av = anchor
            else:
                from scipy.optimize import brentq

                av = float(
                    brentq(
                        lambda x: (
                            sed.reference_band_extinction_np(
                                band, x, system, rv
                            )
                            - anchor
                        ),
                        1e-6,
                        float(sed.grid_axes["av_pts"].max()),
                    )
                )
        else:
            av = float(np.atleast_1d(self.av.initval)[0])
        a_i = sed.bc_np(I_FILTER, teff, logg, feh, 0.0, rv) - sed.bc_np(
            I_FILTER, teff, logg, feh, av, rv
        )
        self._warn_ai_rc(system, a_i, "start")

    def checks_at_point(self):
        return bool(getattr(self, "_ai_check", False))

    def check_at_point(self, model, system, point, where):
        """At the posterior median, the source_ai_rc check again with the
        FITTED clump A_I (warn only; nothing is rebuilt)."""
        if where != "median" or not self.checks_at_point():
            return False
        import pytensor

        sed = system.sed
        rv = None
        if sed.fit_rv_dust:
            rv = getattr(self, RV_DUST_KEY).value
        teff, logg, feh, av = (
            getattr(self, p).value for p in ("teff", "logg", "feh", "av")
        )
        col = sed.filter_column(I_FILTER)
        a_i = (
            sed.bc_node(teff, logg, feh, pt.zeros_like(teff), rv)[0, col]
            - sed.bc_node(teff, logg, feh, av, rv)[0, col]
        )
        fn = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values([a_i]),
            on_unused_input="ignore",
        )
        val = float(fn(*[point[v.name] for v in model.value_vars])[0])
        self._warn_ai_rc(system, val, "fitted (posterior median)")
        return False

    def _add_prose(self, system):
        mean_mu, sd_mu, _mode = self.bar_moments
        prose = get_collector(system)
        prose.add(
            r"The red clump of the field is modeled as a reference object "
            r"evaluated through the same bolometric corrections as the "
            r"stars: its intrinsic color $(V-I)_0 = 1.06 \pm 0.06$ "
            r"\citep{Bensby:2013} is a prior on the color computed from "
            r"its model atmosphere, its absolute magnitude $M_I = -0.12$ "
            r"\citep{Nataf:2013} a prior on its luminosity, and its distance "
            rf"is the Galactic bar's on this sight line (distance modulus "
            rf"${mean_mu:.2f} \pm {BAR_DISTANCE_SIGMA_MAG:.2f}$, the bar's "
            rf"depth of {sd_mu:.2f}~mag being the spread of individual "
            r"clump stars) in the model of \citet{Koshimoto:2021}. ",
            section="stellar",
            key=f"{self.prefix}.reference",
            rank=20.6,
        )
        if self.tied:
            prose.add(
                r"The extinction to the source is tied to the clump's, "
                r"$A_V/A_{V,\rm RC} = 1.00 \pm 0.09$, the field-scale "
                r"granularity of bulge extinction \citep{Nataf:2013}. ",
                section="priors",
                key=f"{self.prefix}.av_tie",
                rank=20.7,
            )
        if not system.sed.fit_rv_dust:
            prose.add(
                r"The bolometric corrections assume a single extinction "
                r"law, whose $A_I/E(V-I) \approx 1.7$ for the clump is "
                r"steeper than the $\approx 1.2$ measured toward the bulge "
                r"\citep{Nataf:2013}. ",
                section="priors",
                key=f"{self.prefix}.table_law",
                rank=20.8,
            )
