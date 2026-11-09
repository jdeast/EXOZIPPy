"""
Galactic prior: genulens's Galactic model (Koshimoto, Baba & Bennett 2021,
ApJ 917, 78, "K21"; https://github.com/nkoshimoto/genulens), ported
faithfully in koshimoto21.py, enforcing the density, kinematics and
present-day mass function of the Galaxy on every modeled star.  Cite K21
when you use it.

This module is the component: the sight line, the mixture over genulens's
branches (seven thin-disk age bins, the thick disk, the E+E_X bar, the
stellar halo and, inside its box, the nuclear stellar disk), linked stars,
the free-floating-planet mass function, the population age / [Fe/H] priors
and the evidence normalizer.  The physics, function by function against
genulens, is koshimoto21.py; galacticmodel.md says what is ported and where
it deviates.  The earlier 3-population model (``model: simplified``) was
removed in review 1.7.6.  The event-rate selection factor (theta_E *
mu_rel) is deliberately NOT here -- it lives in the lens component, so this
prior stays microlensing-agnostic.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component

# One owner for "read a Parameter's hard support" -- star.py's volume prior
# normalizes over star.distance's bounds exactly the way the FFP mass
# function below normalizes over star.logmass's.
from exozippy.components.parameter import sampled_bounds as _sampled_bounds

from . import koshimoto21 as k21
from . import populations
from .pdmf import GRID_DEX as PDMF_GRID_DEX
from .pdmf import interp_table

logger = logging.getLogger(__name__)

#: defaults.yaml's ra/dec initvals (degrees): a star still sitting exactly here
#: with no user or data provenance was never positioned.
ANCHOR_PLACEHOLDER_RA_DEG = 180.0
ANCHOR_PLACEHOLDER_DEC_DEG = 0.0


def _unnormalized_warning():
    logger.warning(
        "[galacticmodel] Cannot read finite log10-mass bounds; the FFP "
        "mass prior is left unnormalized (harmless for sampling, but its "
        "logp is offset by an unknown constant and is not comparable with "
        "the stellar mass function's)."
    )
    return 0.0


def _power_law_log_norm(k, param):
    """log of the normalizer of p(x) ~ 10^(k x) over x in [lower, upper].

    ``param`` is the sampled log10-mass Parameter; its hard bounds are the
    support, so

        Z = int_lo^hi 10^(k x) dx = (10^(k hi) - 10^(k lo)) / (k ln10)

    which is finite for any k as long as both bounds are.  Evaluated as
    logdiffexp(., .) - log(|k| ln10) so the exponentials never overflow
    (10^(k*lo) is 1e12 already at the defaults.yaml floor of -9 dex).

    ``k`` may be a scalar (one slope shared by every star) or
    a per-star array (the FFP slope is per star and user-settable, and each
    star carries its own bounds).  Both branches of the k == 0 split are
    always evaluated and then selected, which is safe because neither can
    produce a NaN: upper > lower always, and the power-law branch substitutes
    k = 1 where k is zero.
    """
    bounds = _sampled_bounds(param)
    if bounds is None:
        return _unnormalized_warning()
    lower, upper = bounds

    ln10 = np.log(10.0)
    k = np.asarray(k, dtype=float)
    is_flat = k == 0.0  # uniform in log10 M
    k_safe = np.where(is_flat, 1.0, k)

    a, b = k_safe * upper * ln10, k_safe * lower * ln10
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    power_law = hi + np.log1p(-np.exp(lo - hi)) - np.log(np.abs(k_safe) * ln10)
    return np.where(is_flat, np.log(upper - lower), power_law)


def ffp_logmass_logp(logmass, alpha, param):
    """Free-floating-planet mass prior, as a density in x = log10(M/Msun).

    THIS IS THE SEAM.  If you have your own free-floating-planet mass
    function, replace the two lines below the comment banner and nothing else:
    every caller (``GalacticModel.build_likelihood``), the per-star selection
    (``star:``'s ``mass_function: ffp``), the floor, and the tests reach the
    functional form only through here.

    Functional form
    ---------------
    Sumi et al. 2023 (AJ 166, 108; arXiv:2303.08280), MOA-II 9-year survey:

        dN/dlog M = Z * (M / M_norm)^(-alpha)  dex^-1 star^-1

    with alpha = 0.96 (+0.47/-0.27), Z = 2.18 (+0.52/-1.40) dex^-1 star^-1 and
    M_norm = 8 M_Earth, fit over 0.33 < M/M_Earth < 6660 (1e-6 < M/Msun <
    0.02).  ``alpha`` here is that exponent, POSITIVE for the observed
    rising-toward-low-mass slope; it is per star and user-settable (``star:``
    key ``mass_function: {kind: ffp, alpha: ...}``) because this measurement
    is uncertain and Roman will revise it.

    Change of variables
    -------------------
    None -- and that is the whole point of preferring this parameterization.
    The measurement is ALREADY a density in log mass, and x = log10(M) is
    already the sampled coordinate (``star.logmass``), so

        log p(x) = -alpha * ln10 * x + const,

    with no Jacobian.  Contrast a mass function quoted as dN/dM (Salpeter's
    power law, say), which picks up |dM/dx| = M ln10, turning its exponent
    from -alpha into (1 - alpha).  If you substitute a mass
    function quoted as dN/dM, you must put that Jacobian back.

    Normalization
    -------------
    From the sampled support, not from the paper.  ``param`` is the
    ``star.logmass`` Parameter; ``_power_law_log_norm`` integrates 10^(k x)
    over its hard [lower, upper] bounds, which makes this a proper density
    directly comparable with the stellar mass function, which koshimoto21
    normalizes over the same support.  Both Z
    and M_norm are therefore unusable here: each contributes only an additive
    constant that the normalizer cancels exactly.  Z is an abundance (a rate
    per star), and would matter only to a model that weighed the FFP and
    stellar lens populations against each other -- this prior conditions on
    the lens being an FFP, so it cannot see a rate.

    The support's lower end is a real modeling choice -- a rising density
    piles prior mass against it, 90% of it within 1/alpha dex -- and it is
    deliberately left to the user, as ``star.<name>.logmass``'s ordinary
    ``lower`` bound.  Nothing clamps it: sub-stellar masses are this
    relation's domain, so a floor imposed here would gate exactly the models
    it exists to express.  ``Star._warn_ffp_logmass_bound`` says so out loud
    when the bound is still the default.
    """
    # ---- substitute your own mass function here -------------------------
    # k is the exponent of the density in the SAMPLED coordinate, p(x) ~
    # 10^(k x); the normalizer is the integral of that over the support.
    k = -np.asarray(alpha, dtype=float)
    return k * np.log(10.0) * logmass - _power_law_log_norm(k, param)
    # ---------------------------------------------------------------------


# ----------------------------------------------------------------------
# The kinematic prior's bookkeeping.
#
# The linked-star analysis (linked_star_structure) is static -- it reads
# which star.* elements are HARD-linked -- and decides which stars carry a
# position term of their own.  The rulings are in components/star/star.md,
# "Linked stars: one position, one density term".
# ----------------------------------------------------------------------

#: log of the rv column of the velocity Jacobian: the velocity densities are
#: in km/s, the sampled rv is in m/s.
_LOG_KMS_PER_MS = float(np.log(1e-3))

#: The three Galactic POPULATIONS the age / [Fe/H] priors are written for
#: (populations.py); koshimoto21.POPULATION maps each of genulens's branches
#: onto one of them.
BRANCHES = ("thin_disk", "thick_disk", "bulge")

#: The velocity coordinates of a star, in the column order of the
#: (pm_ra, pm_dec, rv) -> galactocentric map.
VELOCITY_COORDS = ("pm_ra", "pm_dec", "rv")


def _hard_links(param):
    """{element: link spec} of a Parameter's HARD links (sigma: 0 initval
    links), or {} -- a mock or an unlinked parameter has none."""
    links = getattr(param, "element_links", None) or {}
    return links.get("hard", {}) or {}


def linked_star_structure(stars, n_star, prefix="star"):
    """Which stars carry a position of their own, and which velocity
    coordinates of each star are hard-linked.

    Returns ``(root_of, vel_linked, refused)``:

    - ``root_of[k]`` is the star whose position star ``k`` shares.  A star
      whose ``distance`` is HARD-linked to exactly one other star's
      distance, and to nothing else, joins that star's group (chains are
      followed to the end); every other star is its own root.
    - ``vel_linked[k]`` is the sorted tuple of column indices (into
      ``VELOCITY_COORDS``) of star ``k``'s hard-linked velocity
      coordinates, for every star that has any.
    - ``refused`` lists ``(k, dep_paths)`` for a hard-linked distance that
      reads anything other than ONE other star's distance: that star keeps
      its own position term (and the caller warns), because there is no
      single partner whose position it is.
    """
    parent = list(range(n_star))
    refused = []
    for k, spec in sorted(_hard_links(stars.distance).items()):
        k = int(k)
        intra = set(spec.get("intra_deps", ()) or ())
        paths = tuple(spec.get("dep_paths", ()) or ())
        foreign = [
            p
            for p in paths
            if not (
                len(p.split(".")) == 3
                and p.split(".")[0] == prefix
                and p.split(".")[2] == "distance"
            )
        ]
        if len(intra) == 1 and not foreign and k not in intra:
            parent[k] = int(next(iter(intra)))
        else:
            refused.append((k, paths))

    root_of = np.arange(n_star)
    for k in range(n_star):
        j, seen = k, set()
        while parent[j] != j:
            if j in seen:  # a cycle the link engine should have refused
                raise ValueError(
                    f"galacticmodel: the hard links on {prefix}.distance "
                    f"form a cycle through element {k}; a cycle of hard "
                    f"links has no value."
                )
            seen.add(j)
            j = parent[j]
        root_of[k] = j

    vel_linked = {}
    for c, name in enumerate(VELOCITY_COORDS):
        param = getattr(stars, name, None)
        if param is None:
            continue
        for k in _hard_links(param):
            vel_linked.setdefault(int(k), set()).add(c)
    vel_linked = {k: tuple(sorted(v)) for k, v in vel_linked.items()}
    return root_of, vel_linked, refused


class GalacticModel(Component):
    """The Galactic prior on a line of sight: the genulens port (Koshimoto,
    Baba & Bennett 2021, ApJ 917, 78; koshimoto21.py).  Cite K21 when you
    use it; galacticmodel.md says where the port departs from genulens."""

    #: the galactic model (a MODEL choice, so an enum; one spelling).
    #: koshimoto21 = the genulens port (koshimoto21.py), the only model since
    #: review 1.7.6 removed the 3-population ``simplified`` one.  The key
    #: stays so a config that spells the default out keeps working.
    SUPPORTED_MODELS = ("koshimoto21",)

    #: Keys of the removed ``model: simplified``, each with what replaces it.
    #: They RAISE: a removed key silently ignored is a prior the user asked
    #: for and did not get (CLAUDE.md, no fallbacks).
    REMOVED_KEYS = {
        "IMF": (
            "model: koshimoto21 carries K21's broken-power-law IMF, which "
            "its branch normalizations are derived from"
        ),
        "present_day": (
            "model: koshimoto21 always applies genulens's present-day mass "
            "function; 'remnants: false' drops the dark remnants (genulens's "
            "REMNANT 0)"
        ),
    }

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Galactic Prior"
        if self.n_elements != 1:
            # Only config[0] is ever read (anchor_idx and the options), and
            # the extra blocks used to leak into the likelihood as a
            # broadcast: with 2 blocks and 1 star the whole kinematic prior
            # was counted TWICE, and with 2 blocks and 3 stars it raised a
            # bare shape error.  One galactic prior describes one line of
            # sight.
            raise ValueError(
                f"galacticmodel takes exactly one config block, got "
                f"{self.n_elements}.  It is a single prior on the line of "
                f"sight shared by every star; use 'anchor_idx' to choose "
                f"which star's (ra, dec) defines it."
            )
        cfg = self.config[0]
        model = cfg.get("model", "koshimoto21")
        if model == "simplified":
            raise ValueError(
                "galacticmodel model: simplified was removed (review 1.7.6). "
                "The earlier 3-population Galactic model (thin disk, thick "
                "disk, bar) is gone; model: koshimoto21, the port of genulens "
                "(Koshimoto, Baba & Bennett 2021), is the Galactic model. "
                "Delete the 'model' key, or write 'model: koshimoto21'."
            )
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"galacticmodel model {model!r} is not one of "
                f"{', '.join(self.SUPPORTED_MODELS)} (case-sensitive): "
                f"'koshimoto21' is the port of genulens (Koshimoto, Baba & "
                f"Bennett 2021)."
            )
        self.model = model
        removed = [k for k in self.REMOVED_KEYS if k in cfg]
        if removed:
            raise ValueError(
                "galacticmodel "
                + "; ".join(
                    f"key '{k}' belonged to the removed model: simplified "
                    f"(review 1.7.6) -- {self.REMOVED_KEYS[k]}"
                    for k in removed
                )
                + ".  Delete "
                + ("it" if len(removed) == 1 else "them")
                + "; model: koshimoto21 is the Galactic model."
            )
        # genulens's REMNANT switch (default on, JDE 2026-10-30).
        remnants = cfg.get("remnants", True)
        if not isinstance(remnants, bool):
            raise ValueError(
                f"galacticmodel remnants must be true or false, got "
                f"{remnants!r}."
            )
        self.remnants = remnants
        binaries = cfg.get("binaries", False)
        if not isinstance(binaries, bool):
            raise ValueError(
                f"galacticmodel binaries must be true or false, got "
                f"{binaries!r}."
            )
        self.binaries = binaries
        self._parse_source_selection()
        self.anchor_idx = self.config[0].get("anchor_idx", 0)
        # Population-conditional age and [Fe/H] priors (review 1.7.5): two
        # independent booleans (JDE's n-way ruling), default on.  They act
        # only on a star whose age / initfeh is SAMPLED (a star on an
        # evolutionary track); every other star is untouched.
        for key in ("population_age", "population_feh"):
            val = self.config[0].get(key, True)
            if not isinstance(val, bool):
                raise ValueError(
                    f"galacticmodel {key} must be true or false, got {val!r}."
                )
            setattr(self, key, val)

    SOURCE_SELECTIONS = ("sqrt_distance", "luminosity_function")

    def _parse_source_selection(self):
        """koshimoto21's source weighting (genulens's rhoD_S), an n-way MODEL
        choice: sqrt_distance (genulens's default, gammaDs = 0.5: n_MS
        sqrt(D / 8 kpc), no D^2) or luminosity_function (genulens's AIrc
        mode: n_MS D^2 fLF(D), sources with apparent I in source_i_range
        given A_I = source_ai_rc at the red clump)."""
        cfg = self.config[0]
        keys = (
            "source_selection",
            "source_i_range",
            "source_ai_rc",
            "source_dm_rc",
        )
        sel = cfg.get("source_selection", "sqrt_distance")
        if sel not in self.SOURCE_SELECTIONS:
            raise ValueError(
                f"galacticmodel source_selection {sel!r} is not one of "
                f"{', '.join(self.SOURCE_SELECTIONS)} (case-sensitive)."
            )
        self.source_selection = sel
        if sel == "luminosity_function":
            if "source_ai_rc" not in cfg:
                raise ValueError(
                    "galacticmodel source_selection: luminosity_function "
                    "needs source_ai_rc (the I-band extinction to the red "
                    "clump, genulens's AIrc)."
                )
            i_range = cfg.get("source_i_range", [14.0, 21.0])
            if len(i_range) != 2 or not i_range[0] < i_range[1]:
                raise ValueError(
                    f"galacticmodel source_i_range must be [I_min, I_max] "
                    f"with I_min < I_max, got {i_range!r}."
                )
            self.source_i_range = (float(i_range[0]), float(i_range[1]))
            self.source_ai_rc = float(cfg["source_ai_rc"])
            dm = cfg.get("source_dm_rc")
            self.source_dm_rc = None if dm is None else float(dm)
        else:
            extra = [k for k in keys[1:] if k in cfg]
            if extra:
                raise ValueError(
                    f"galacticmodel {', '.join(extra)} apply to "
                    f"source_selection: luminosity_function only."
                )

    @property
    def prefix(self):
        return "galacticmodel"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "population_age",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Population-conditional AGE prior (default true): a "
                    "star whose age is sampled (a star on an evolutionary "
                    "track) takes its age from the thin-disk (star "
                    "formation over 0-10 Gyr), thick-disk (12 +/- 1 Gyr) "
                    "or bulge (9 +/- 1 Gyr) age distribution, mixed by the "
                    "same density x kinematics weights as its position "
                    "(Koshimoto+2021), instead of uniform on its bounds.  "
                    "false = the uniform prior."
                ),
            },
            {
                "key": "population_feh",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Population-conditional INITIAL-METALLICITY prior "
                    "(default true) on a star whose initfeh is sampled: "
                    "thin disk 0.0 +/- 0.2 dex at R0 with a -0.06 dex/kpc "
                    "radial gradient (Genovali+2014), thick disk -0.78 +/- "
                    "0.30 (Robin+2003), bulge 45% at -0.30 +/- 0.25 plus "
                    "55% at +0.32 +/- 0.11 (Hill+2011), mixed like "
                    "population_age.  false = the uniform prior."
                ),
            },
            {
                "key": "model",
                "kind": "option",
                "accepts": list(cls.SUPPORTED_MODELS),
                "required": False,
                "doc": (
                    "The Galactic model (default and only value "
                    "'koshimoto21'): a faithful port "
                    "of genulens (Koshimoto, Baba & Bennett 2021, ApJ 917, "
                    "78; github.com/nkoshimoto/genulens) -- seven thin-disk "
                    "age bins, thick disk, the E+E_X bar and the stellar "
                    "halo, with genulens's Shu-distribution disk "
                    "kinematics, bar streaming and dispersion gradients, "
                    "and its present-day mass function.  The earlier "
                    "3-population 'simplified' model was removed (review "
                    "1.7.6) and raises, as do its IMF and present_day keys.  "
                    "koshimoto21's age / [Fe/H] priors are still the "
                    "3-population ones (galacticmodel.md)."
                ),
            },
            {
                "key": "remnants",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "genulens's REMNANT (default "
                    "true): stars above their branch's PARSEC turnoff become "
                    "white dwarfs (Kalirai+2008), neutron stars or black "
                    "holes (Raithel+2018), neutron stars and black holes "
                    "with genulens's natal kicks (350 / 100 km/s).  false = "
                    "genulens's REMNANT 0: dead stars are discarded."
                ),
            },
            {
                "key": "binaries",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "genulens's BINARY (default "
                    "false, genulens's default): the lens of a single-lens "
                    "event may be an unresolved binary -- genulens's "
                    "multiplicity, mass ratios and separations; a companion "
                    "resolved wide leaves one body as the lens, one closer "
                    "than the event can resolve makes the lens their total "
                    "mass, and a detectable binary is excluded."
                ),
            },
            {
                "key": "source_selection",
                "kind": "option",
                "accepts": ["sqrt_distance", "luminosity_function"],
                "required": False,
                "doc": (
                    "How a survey-selected SOURCE "
                    "star (Component.selected_sources: a microlensing "
                    "event's source) is weighted, as genulens does it.  "
                    "'sqrt_distance' (default, genulens's default): its "
                    "main-sequence count times sqrt(D / 8 kpc), with no "
                    "D^2 volume element.  'luminosity_function' (genulens's "
                    "AIrc mode): main-sequence count times D^2 times the "
                    "fraction of each population's luminosity function with "
                    "apparent I in source_i_range (default [14, 21]), given "
                    "source_ai_rc (required: A_I at the red clump) and "
                    "optionally source_dm_rc (default Nataf+16 as genulens "
                    "codes it)."
                ),
            },
            {
                "key": "source_i_range",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "[I_min, I_max] for source_selection: luminosity_function.",
            },
            {
                "key": "source_ai_rc",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "A_I to the red clump (genulens AIrc), for luminosity_function.",
            },
            {
                "key": "source_dm_rc",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "Red-clump distance modulus (genulens DMrc), optional.",
            },
            {
                "key": "anchor_idx",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Index of the star whose (ra, dec) defines the line of "
                    "sight for the density/kinematic prior (default 0). One "
                    "galacticmodel block describes one sight line."
                ),
            },
        ]

    def register_parameters(self, system):
        """No parameters to sample! Just an empty manifest."""
        self.manifest = {}

    # ------------------------------------------------------------------
    # Public sight-line queries (numpy, stage 1-3 safe; galacticmodel.md,
    # "The sight-line queries").  Not part of the port: what a component
    # outside the galactic prior (the red clump) needs to know about the
    # sight line, computed from the port's own functions so there is ONE
    # definition of "the sight line" and of "the bar".
    # ------------------------------------------------------------------
    def sight_line(self, system):
        """``(l_deg, b_deg)`` of the galactic model's line of sight: the
        anchor star's POSITION ROOT (a lens whose position derives from its
        source reads the source's), at its informed start, through the
        port's own ``line_of_sight_basis``.  Read the stage 1-3 way
        (``require_star_radec``: the params file's position, never a probed
        placeholder), and through the pure root query
        ``star.position_root_of``, so the answer does not depend on whether
        Star's stage 3 has run."""
        from exozippy.components.star.position import require_star_radec
        from exozippy.components.star.star import position_root_of

        root = position_root_of(system, int(self.anchor_idx))
        ra, dec = require_star_radec(
            self.config_manager,
            root,
            f"[{self.prefix}] the line of sight (anchor star {self.anchor_idx})",
        )
        l_deg, b_deg, _m, _v0 = k21.line_of_sight_basis(ra, dec)
        return float(l_deg), float(b_deg)

    def bar_distance_moments(self, system):
        """``(mean_mu, sd_mu, mode_mu)`` of the bar's stars on the sight
        line (``koshimoto21.bar_distance_moments``)."""
        return k21.bar_distance_moments(*self.sight_line(system))

    # ------------------------------------------------------------------
    # Reporting: tell the tables what these potentials are.
    #
    # Both potentials below act on parameters this component does not own
    # and does not sample -- they are `pm.Potential`s over star.logmass and
    # over (star.distance, pm_ra, pm_dec, rv).  `Parameter.get_prior_str`
    # can only see a parameter's OWN fields, so without these declarations
    # every one of them is reported as "Uniform" (a bounded element with no
    # sigma), which is exactly the prior these terms replace.  See
    # parameter.PriorContribution.
    # ------------------------------------------------------------------

    def _mass_function_mask(self, stars, ffp_mask):
        """Which stars take genulens's present-day mass function (inside the
        branch mixture): every star except those whose ``mass_function:``
        opted onto the FFP mass function."""
        n_star = int(np.atleast_1d(stars.logmass.initval).size)
        if ffp_mask.size:
            return ~ffp_mask
        return np.ones(n_star, dtype=bool)

    # ------------------------------------------------------------------
    # Population-conditional age and [Fe/H] priors (review 1.7.5;
    # populations.py holds the distributions).
    # ------------------------------------------------------------------

    def _population_mask(self, stars, name, enabled):
        """Per star: does star.<name> take the population prior?  Only a
        SAMPLED element (a free coordinate: not pinned, not derived, not
        hard-linked -- an orbit member's age is its anchor's, counted once
        there) and only with the toggle on.  A star with no track has no
        age/initfeh element at all; one an evolutionarymodel does not name
        has them pinned at a bookkeeping value (``_pin_unmodeled_stars``),
        which must not steer its population."""
        n_star = int(np.atleast_1d(stars.distance.initval).size)
        param = getattr(stars, name, None)
        if not enabled or param is None:
            return np.zeros(n_star, dtype=bool)
        return np.array(
            [param.element_is_sampled(k) for k in range(n_star)], dtype=bool
        )

    def _support(self, param, mask):
        """Static (lower, upper) per star of a Parameter that must have
        them; masked-off stars get a wide dummy support (never read, but
        kept finite so 0 * term stays 0)."""
        bounds = _sampled_bounds(param)
        if bounds is None:
            raise ValueError(
                f"[{self.prefix}] the population prior on {param.label} "
                f"needs finite static lower/upper bounds; none could be read."
            )
        n = mask.size
        lo = np.broadcast_to(np.atleast_1d(bounds[0]), (n,)).astype(float)
        hi = np.broadcast_to(np.atleast_1d(bounds[1]), (n,)).astype(float)
        if getattr(param, "element_links", None) and (
            param.element_links.get("lower")
            or param.element_links.get("upper")
        ):
            logger.warning(
                f"[{self.prefix}] {param.label} carries a linked bound; the "
                f"population prior is normalized over its STATIC bounds."
            )
        return lo, hi

    def _population_branch_terms(self, stars, r, age_mask, feh_mask):
        """Per branch, per star: log g_b(age) + log p_b(initfeh | R), each
        normalized over the star's own support, times the 0/1 masks.

        Normalized per branch, so adding them inside the mixture leaves the
        branch weights the physical number densities: the joint over
        (position, velocity, mass, age, [Fe/H]) stays a normalized density
        and ``evidence_log_norm`` is unchanged.  The thin disk's [Fe/H]
        normalizer moves with R (its center does), so it is in the graph;
        the rest are build-time constants.  Every term is finite on every
        element (a masked-off star gets a wide dummy support), so the 0/1
        weight is plain arithmetic, never a selection.
        """
        n_b = len(BRANCHES)
        terms = [0.0] * n_b
        if age_mask.any():
            w = age_mask.astype(float)
            lo, hi = self._support(stars.age, age_mask)
            lo = np.where(age_mask, lo, 0.0)
            hi = np.where(age_mask, hi, populations.AGE_MAX_GYR)
            t = stars.age.value
            for b, pop in enumerate(BRANCHES):
                log_g = populations.log_age_density(pop, t, xp=pt)
                terms[b] = terms[b] + w * (
                    log_g - populations.log_age_norm(pop, lo, hi)
                )
        if feh_mask.any():
            w = feh_mask.astype(float)
            lo, hi = self._support(stars.initfeh, feh_mask)
            populations.check_thin_disk_feh_support(
                lo[feh_mask], hi[feh_mask], f"[{self.prefix}] star.initfeh"
            )
            lo = np.where(feh_mask, lo, -10.0)
            hi = np.where(feh_mask, hi, 10.0)
            f = stars.initfeh.value
            for b, pop in enumerate(BRANCHES):
                log_p = populations.log_feh_density(pop, f, r_kpc=r, xp=pt)
                if pop == "thin_disk":
                    log_z = populations.thin_disk_log_feh_norm(
                        r, lo, hi, xp=pt
                    )
                else:
                    log_z = populations.log_feh_norm(pop, lo, hi)
                terms[b] = terms[b] + w * (log_p - log_z)
        return terms

    def _declare_population_priors(self, system, stars, age_mask, feh_mask):
        """Prior-column declarations and the draft sentence for the
        population age / [Fe/H] priors (declare-at-site rule)."""
        from ...outputs.prose import get_collector

        if age_mask.any():
            stars.age.add_prior_contribution(
                latex=(
                    r"Galactic-population mixture: thin disk SFH "
                    r"$\propto e^{-(10-T)/7}$ on 0--10\,Gyr, thick disk "
                    r"$12 \pm 1$\,Gyr, bulge $9 \pm 1$\,Gyr"
                ),
                text=(
                    "Galactic-population mixture: thin disk SFH "
                    "exp(-(10-T)/7) on 0-10 Gyr, thick disk 12+/-1 Gyr, "
                    "bulge 9+/-1 Gyr"
                ),
                elements=np.nonzero(age_mask)[0],
                supersedes_bounds=True,
            )
        if feh_mask.any():
            stars.initfeh.add_prior_contribution(
                latex=(
                    r"Galactic-population mixture: thin disk "
                    r"$\mathcal{N}(-0.06\,(R-R_0), 0.2)$, thick disk "
                    r"$\mathcal{N}(-0.78, 0.3)$, bulge "
                    r"$0.45\,\mathcal{N}(-0.30, 0.25) + "
                    r"0.55\,\mathcal{N}(0.32, 0.11)$"
                ),
                text=(
                    "Galactic-population mixture: thin disk "
                    "N(-0.06 (R-R0), 0.2), thick disk N(-0.78, 0.3), bulge "
                    "0.45 N(-0.30, 0.25) + 0.55 N(0.32, 0.11)"
                ),
                elements=np.nonzero(feh_mask)[0],
                supersedes_bounds=True,
            )
        if not (age_mask.any() or feh_mask.any()):
            return
        what = " and ".join(
            w
            for w, m in (("age", age_mask), ("initial metallicity", feh_mask))
            if m.any()
        )
        get_collector(system).add(
            rf"The {what} prior of each star on an evolutionary track "
            r"follows its Galactic population, mixed with the same "
            r"density and kinematic weights as its position: ages from "
            r"a thin-disk star-formation history "
            r"$\propto e^{-(10\,{\rm Gyr}-T)/7\,{\rm Gyr}}$ over "
            r"0--10\,Gyr, a $12 \pm 1$\,Gyr thick disk and a "
            r"$9 \pm 1$\,Gyr bulge \citep{Koshimoto:2021}; metallicities "
            r"from a thin disk centered on solar at the Sun with the "
            r"Cepheid gradient of $-0.06$\,dex\,kpc$^{-1}$ "
            r"\citep{Genovali:2014} and a 0.2\,dex width, a thick disk at "
            r"$-0.78 \pm 0.30$ \citep{Robin:2003}, and the two-component "
            r"bulge distribution of \citet{Hill:2011}.",
            section="priors",
            key=f"{self.prefix}.population_age_feh",
            rank=21,
        )

    def _birth_logmass(self, system, stars):
        """The ``stars.logmass`` vector with each star's BIRTH mass in place.

        A star whose present-day mass differs from its initial one has a
        component that says so: ``Component.initial_logmass`` returns
        ``{star index: log10 initial mass}`` (the evolutionary model, which
        also adds the Jacobian of that change of coordinate).  Two
        components claiming one star would be two different birth masses,
        and raise.  Assembled with ``set_subtensor``, never ``pt.where``.
        """
        claims = {}
        for comp in system.active_components.values():
            for si, value in comp.initial_logmass(system).items():
                if si in claims:
                    raise ValueError(
                        f"[{self.prefix}] star {si} has an initial mass from "
                        f"both '{claims[si][0]}' and '{comp.prefix}'; a star "
                        f"has one birth mass."
                    )
                claims[si] = (comp.prefix, value)
        logmass = stars.logmass.value
        if not claims:
            return logmass
        idx = sorted(claims)
        return pt.set_subtensor(
            logmass[np.asarray(idx, dtype=int)],
            pt.stack([claims[i][1] for i in idx]),
        )

    def _declare_mass_prior(self, stars, ffp_mask):
        """Describe the mass prior, per star.

        Two descriptions, because the prior really is two densities: genulens's
        present-day mass function (inside the kinematic mixture) on the stars
        on the stellar mass function, and the FFP mass function (the
        ``imf_prior`` potential) on the stars whose ``mass_function:`` opted
        out of it.  Both are normalized over star.logmass's own support, so
        both supersede the uniform.
        """
        logmass = getattr(stars, "logmass", None)
        if logmass is None:
            return

        imf_latex = (
            r"Koshimoto et al. (2021) present-day mass function "
            r"(genulens: K21 IMF, PARSEC turnoff"
            + (r", WD/NS/BH remnants)" if self.remnants else ")")
        )
        imf_text = (
            "Koshimoto+2021 present-day mass function (genulens: K21 "
            "IMF, PARSEC turnoff"
            + (", WD/NS/BH remnants)" if self.remnants else ")")
        )
        imf_elements = np.nonzero(~ffp_mask)[0] if ffp_mask.size else None
        if imf_elements is None or len(imf_elements):
            logmass.add_prior_contribution(
                latex=imf_latex,
                text=imf_text,
                elements=imf_elements,
                supersedes_bounds=True,
            )

        # The FFP slope is per star, so a system with two differently-sloped
        # FFPs gets two contributions, each naming its own elements.
        alphas = np.atleast_1d(getattr(stars, "ffp_alpha", []))
        for i in np.nonzero(ffp_mask)[0]:
            alpha = float(alphas[i]) if i < alphas.size else float("nan")
            logmass.add_prior_contribution(
                latex=(
                    r"Sumi et al. (2023) FFP mass function "
                    rf"($\alpha = {alpha:g}$)"
                ),
                text=f"Sumi+2023 FFP mass function (alpha = {alpha:g})",
                elements=[int(i)],
                supersedes_bounds=True,
            )

    def _add_mass_prior_prose(self, system, stars, ffp_mask):
        """Modeling-draft sentences for the mass prior.

        Mirrors ``_declare_mass_prior``'s per-star split: one sentence for
        the stars on genulens's present-day mass function, and each FFP star
        gets the Sumi slope actually applied.  Declared next to the
        potential (declare-at-site rule).
        """
        from ...outputs.prose import get_collector
        from ...outputs.texutils import latex_escape

        prose = get_collector(system)

        if getattr(stars, "logmass", None) is None:
            return
        # Read the star inventory defensively: unit tests drive this
        # potential through minimal mock star components.
        star_names = [str(nm) for nm in (getattr(stars, "names", None) or [])]
        n = max(
            int(getattr(stars, "n_elements", len(star_names))),
            len(star_names),
            int(ffp_mask.size),
        )
        if len(star_names) < n:
            star_names += [str(i) for i in range(len(star_names), n)]
        mask = ffp_mask if ffp_mask.size else np.zeros(n, dtype=bool)
        if not mask.all():
            prose.add(
                r"The mass of each star on the stellar mass function "
                r"followed the present-day mass function of "
                r"\citet{Koshimoto:2021} as implemented in genulens: their "
                r"broken-power-law initial mass function, truncated at each "
                r"population's main-sequence turnoff from the PARSEC "
                r"isochrones \citep{Bressan:2012}"
                + (
                    r", with the dead stars turned into white dwarfs "
                    r"\citep{Kalirai:2008} and neutron stars and black holes "
                    r"\citep{Raithel:2018} that received natal kicks of 350 "
                    r"and 100\,km\,s$^{-1}$"
                    if self.remnants
                    else ""
                )
                + r", inside the Galactic-population mixture.",
                section="priors",
                key=f"{self.prefix}.imf",
                rank=10,
            )
        alphas = np.atleast_1d(getattr(stars, "ffp_alpha", []))
        for i in np.nonzero(mask)[0]:
            alpha = float(alphas[i]) if i < alphas.size else float("nan")
            prose.add(
                f"For {latex_escape(star_names[i])}, we instead adopted "
                r"the free-floating-planet mass function of "
                r"\citet{Sumi:2023}, "
                r"$dN/d\log M \propto M^{-\alpha}$ with "
                rf"$\alpha = {alpha:g}$.",
                section="priors",
                key=f"{self.prefix}.ffp.{i}",
                rank=11,
            )

    def _declare_kinematic_prior(self, stars):
        """Describe the kinematic_prior potential.

        One joint density over distance and the velocity coordinates, so the
        same term is declared against each of them; distance additionally
        carries the ``2 log d`` volume element, which is why
        ``Star.build_likelihood`` stands down where this component exists.
        """
        distance = getattr(stars, "distance", None)
        if distance is not None:
            distance.add_prior_contribution(
                latex=(
                    r"Galactic density $\times$ kinematics "
                    r"(incl. $d^{2}$ volume element)"
                ),
                text="Galactic density x kinematics (incl. d^2 volume)",
                supersedes_bounds=True,
            )
        for name in ("pm_ra", "pm_dec", "rv"):
            param = getattr(stars, name, None)
            if param is not None:
                param.add_prior_contribution(
                    latex="Galactic kinematic model",
                    supersedes_bounds=True,
                )

    def _require_anchor_position(self, stars, anchor):
        """Refuse to build the prior on an anchor star nobody positioned.

        The whole prior -- density, d^2 volume, velocity frame, the three
        branch Gaussians, for EVERY star -- is a function of one line of
        sight, the anchor's.  A star left at defaults.yaml's ra/dec (180, 0)
        is Galactic (l 276, b +60), and a sight line leaving the plane at
        60 deg makes the density fall exponentially with distance: the
        2026-09 DC2018 sweep ran every event that way (the configs pinned the
        Lens's coordinates and anchored on the Source) and reported source
        distances and radii at half their true values for a week before the
        cause was found (notes 2026-09-24).  Nobody wants a Galactic prior
        without saying where it is, so this is an error, not a warning.

        "Set" means the provenance ledger says the start came from the user
        or from data (a hint pushed by an instrument).  A Parameter with no
        ledger at all (tests, scripts) is left alone: the ledger is the only
        witness that distinguishes "left at the default" from "unknown".
        """
        # A coordinate is "unset" when the provenance ledger does not credit it
        # to the user or to data AND its start value is still defaults.yaml's
        # placeholder.  Both halves matter: tests position stars by writing
        # `star.ra.initval = ...` straight past the ledger (provenance says
        # "default", value says otherwise -- positioned), and a mock Parameter
        # has no ledger at all (getattr, not a method call).  The placeholders
        # are pinned by tests/test_galactic_anchor_position.py.
        placeholder = {
            "ra": np.radians(ANCHOR_PLACEHOLDER_RA_DEG),
            "dec": np.radians(ANCHOR_PLACEHOLDER_DEC_DEG),
        }
        unset = []
        for name, param in (("ra", stars.ra), ("dec", stars.dec)):
            src = getattr(param, "element_initval_source", lambda i: None)(
                anchor
            )
            if src in ("user", "data", "solved"):
                continue
            if np.isclose(
                float(param.element_start(anchor)),
                placeholder[name],
                rtol=0.0,
                atol=1e-12,
            ):
                unset.append(name)
        if not unset:
            return
        names = getattr(stars, "names", None)
        label = names[anchor] if names and anchor < len(names) else str(anchor)
        missing = " and ".join(f"star.{label}.{n}" for n in unset)
        raise ValueError(
            f"[{self.prefix}] the galactic model's anchor star '{label}' "
            f"(star index {anchor}) has no sky position: {missing} "
            f"still at the defaults.yaml value. The whole density and "
            f"kinematic prior is evaluated along this star's line of sight, "
            f"so set star.{label}.ra and star.{label}.dec in the params file "
            f"(or point 'anchor_idx' at a star that has them)."
        )

    def _warn_if_anchor_coords_sampled(self, stars, ra_rad, dec_rad, anchor):
        """Warn once if the anchor star's ra/dec are SAMPLED, since the
        line-of-sight basis below is built from their start values and frozen.

        Same policy as ``Lens._frozen_op_coords_deg``: keep the freeze, say so
        -- but only when it could conceivably matter, i.e. when the sampler
        actually moves the coordinates (galacticmodel + gaia/abs astrometry).

        The freeze is safe here too, and separately measured rather than
        assumed by analogy with the microlensing Op: shifting the sight line
        by 1 arcsec moves the anchor's galactocentric velocity by 1.3e-3 km/s
        (4e-5 of a 30 km/s thin-disk dispersion) and its galactic position by
        5e-5 kpc, against kpc-scale density gradients.  It stays negligible
        out to ~1 arcmin (0.08 km/s, 3 pc) and only reaches 0.16 sigma at a
        full degree -- five orders of magnitude beyond the mas-scale posterior
        width astrometry gives ra/dec.
        """
        moving = [
            name
            for name, param in (("ra", stars.ra), ("dec", stars.dec))
            if param.element_is_sampled(anchor)
        ]
        if not moving:
            return
        logger.warning(
            f"[{self.prefix}] star.{'/'.join(moving)} of the anchor star "
            f"{anchor} is sampled, but the galactic model's line of "
            f"sight is FROZEN at the start value (ra="
            f"{np.degrees(ra_rad):.6f} deg, dec={np.degrees(dec_rad):.6f} "
            f"deg) for the whole fit. This is safe -- 1 arcsec of coordinate "
            f"error moves the galactocentric velocity by ~1e-3 km/s and the "
            f"galactic position by ~0.05 pc -- but the sampled ra/dec do NOT "
            f"feed the density or kinematic prior."
        )

    def build_likelihood(self, model, system):
        stars = system.star

        # The sight line, ONCE, from the anchor star's initial RA/Dec.  This
        # is a property of the line of sight, not of a component element:
        # the loop this replaced ran over self.n_elements (the number of
        # galacticmodel BLOCKS) while indexing self.anchor_idx every time,
        # so it stacked n_elements identical copies and then relied on those
        # copies broadcasting against the (n_star,) sampled vectors -- which
        # silently doubled the whole prior for 2 blocks and 1 star, and
        # raised a bare shape error whenever n_blocks and n_stars disagreed
        # and neither was 1.  The velocity map stays 2-D so it broadcasts
        # over any number of stars by construction.
        # A lens whose position is DERIVED from its source's (review 8.10.4)
        # has no start of its own -- its resolved initval is a placeholder --
        # so the sight line is read off the star it derives from, which it
        # matches to the lens-source separation (mas; the freeze below is
        # safe out to ~1 arcmin).
        anchor = stars.position_root(self.anchor_idx)
        if anchor != self.anchor_idx:
            logger.info(
                f"[{self.prefix}] anchor star {self.anchor_idx}'s position is "
                f"derived from star {anchor}'s (the lens track); the line of "
                f"sight is read from star {anchor}."
            )
        self._require_anchor_position(stars, anchor)
        ra_rad = stars.ra.element_start(anchor)
        dec_rad = stars.dec.element_start(anchor)
        self._warn_if_anchor_coords_sampled(stars, ra_rad, dec_rad, anchor)
        self._k21_sight_line(system, stars, ra_rad, dec_rad)

        # A mass function is a density in the BIRTH mass, in the SAMPLED
        # coordinate x = log10(M).  For most stars that is star.logmass
        # itself; a star whose mass loss is modeled (a MIST track) samples
        # its PRESENT-DAY mass, and the component that models it supplies
        # the initial mass along with the Jacobian that makes this a density
        # in it (Component.initial_logmass).
        birth_logmass = self._birth_logmass(system, stars)

        # Per-star opt-out of the stellar mass function.  A microlensing lens
        # that is a free-floating planet MUST be declared as a star
        # (Lens._validate_bodies rejects a non-star primary), and everything
        # else that makes it a star is right -- the galactic density and
        # kinematic priors below are mass-independent.  Only the mass prior
        # is wrong: K21's mass function ends at 0.001 Msun, so a 3 Mjup body
        # would be charged for being what the user said it is.  The choice
        # has to be PER STAR (a stellar source with an FFP lens is the
        # realistic system), which is why it lives on the star block and not
        # here; see Star._parse_mass_functions.  The FFP stars' mass prior is
        # the imf_prior potential; every other star takes genulens's
        # present-day mass function inside the branch mixture below and
        # contributes 0 here.
        ffp_mask = getattr(stars, "ffp_mask", None)
        ffp_mask = (
            np.zeros(0, dtype=bool)
            if ffp_mask is None
            else np.asarray(ffp_mask, dtype=bool)
        )
        if ffp_mask.any():
            # A 0/1 weight on a term finite everywhere on the support: plain
            # arithmetic rather than a selection, so a clean gradient on
            # every backend.
            imf_logp = pt.sum(
                ffp_mask.astype(float)
                * ffp_logmass_logp(
                    birth_logmass, stars.ffp_alpha, stars.logmass
                )
            )
        else:
            imf_logp = pt.constant(0.0, dtype="float64")
        pm.Potential(f"{self.prefix}.imf_prior", imf_logp)

        pdmf_mask = self._mass_function_mask(stars, ffp_mask)
        self._k21_mf = (
            self._build_k21_mass_function(stars) if pdmf_mask.any() else None
        )
        self._declare_mass_prior(stars, ffp_mask)
        self._add_mass_prior_prose(system, stars, ffp_mask)

        # even though non-physical values will be rejected
        # we still need to be able to calculate a likelihood,
        distance = pt.maximum(stars.distance.value, 1e-3) / 1e3  # kpc

        # The genulens port (koshimoto21.py): its branches with their own
        # NORMALIZED velocity densities, in genulens's position and velocity
        # frames.  Each branch carries its own normalization -- constants
        # cancel within a single Potential but NOT across the logsumexp --
        # and the number densities (genulens's lens density n0, every
        # object) set the physical branch ratios.  The per-population inputs
        # not yet ported per branch (age/[Fe/H]) are mapped from each
        # branch's population (k21.POPULATION).
        log_dens, log_vel, r = self._k21_branch_terms(stars)
        parent = tuple(k21.POPULATION[b] for b in self._k21_branches)

        # Population-conditional age / [Fe/H] priors (review 1.7.5): per
        # population, per star, inside the same mixture (populations.py).
        age_mask = self._population_mask(stars, "age", self.population_age)
        feh_mask = self._population_mask(stars, "initfeh", self.population_feh)
        pop_terms = None
        if age_mask.any() or feh_mask.any():
            pop_terms = self._population_branch_terms(
                stars, r, age_mask, feh_mask
            )
            pop_terms = [pop_terms[j] for j in parent]
        if self._k21_mf is not None:
            # genulens's present-day mass function per branch, with the
            # neutron stars' and black holes' kicked velocity densities: the
            # mass and the velocity are not separable.
            log_vel = self._k21_mass_velocity_terms(
                birth_logmass, pdmf_mask, age_mask, log_vel
            )
            self._add_k21_remnant_probability(
                birth_logmass, log_dens, log_vel, pdmf_mask, age_mask
            )
        # the per-branch population terms, kept apart for the linked path (a
        # member's velocity term is replaced, these are not)
        self._k21_pop = [
            pt.zeros_like(lv) + (0.0 if pop_terms is None else pop_terms[j])
            for j, lv in enumerate(log_vel)
        ]
        branch_logp = [dens + lv for dens, lv in zip(log_dens, log_vel)]
        if pop_terms is not None:
            # Per-star terms (a linked member's entries are 0: its
            # age/initfeh are not sampled).
            branch_logp = [bl + pt_ for bl, pt_ in zip(branch_logp, pop_terms)]
        self._declare_population_priors(system, stars, age_mask, feh_mask)

        volume_element = 2.0 * pt.log(distance * 1000.0)

        # The velocity densities are evaluated in km/s while the sampled
        # coordinates are (pm_ra, pm_dec, rv); the change of variables
        # v = v0 + M @ (K*pm_ra*d, K*pm_dec*d, rv/1e3) needs
        # |det dv/d(pm_ra, pm_dec, rv)| = (K*d)^2 * 1e-3 (M is orthogonal,
        # |det| = 1; the constant 1e-3 from m/s -> km/s is dropped here and
        # reported by evidence_log_norm).  Without the +2*log(d) the intended
        # prior rho * d^2 * f(v) * (K*d)^2 was applied as rho * d^2 * f(v),
        # under-weighting large distances by d^2.
        velocity_jacobian = 2.0 * pt.log(k21.K_KMS * distance)

        # Linked stars (star.md, "Linked stars: one position, one density
        # term").  A star whose distance is HARD-linked to another star's
        # has no position of its own: the system it belongs to has one
        # position along the line of sight, so it gets one density and one
        # d^2 volume element, inside one branch (one population for the
        # whole system).
        n_star = int(
            getattr(stars, "n_elements", 0)
            or np.atleast_1d(stars.distance.initval).size
        )
        root_of, vel_linked, refused = linked_star_structure(
            stars, n_star, prefix=getattr(stars, "prefix", "star")
        )
        self._linked_star_report(stars, root_of, vel_linked, refused)
        roots = sorted(set(int(j) for j in root_of))
        members = [k for k in range(n_star) if root_of[k] != k]

        if members or vel_linked:
            kinematic_penalty = self._k21_grouped_logp(
                log_dens,
                branch_logp,
                volume_element,
                velocity_jacobian,
                root_of,
                vel_linked,
            )
        else:
            # Every star independent: one mixture per star.
            kinematic_penalty = pt.sum(
                pm.math.logsumexp(pt.stack(branch_logp), axis=0)
                + volume_element
                + velocity_jacobian
            )

        self._evidence_log_norm = self._kinematic_evidence_log_norm(
            stars, roots, vel_linked
        )
        pm.Potential(f"{self.prefix}.kinematic_prior", kinematic_penalty)

        self._declare_kinematic_prior(stars)
        from ...outputs.prose import get_collector

        get_collector(system).add(
            r"We imposed Galactic density and kinematic priors on the "
            r"distance, proper motion, and radial velocity of each modeled "
            r"star, following the Galactic model of "
            r"\citet{Koshimoto:2021} as implemented in genulens.",
            section="priors",
            key=f"{self.prefix}.kinematic",
            rank=20,
        )

    def _k21_sight_line(self, system, stars, ra_rad, dec_rad):
        """The genulens port's sight line: (l, b), the velocity map, the
        components genulens simulates there (the NSD inside |l| < 5,
        |b| < 2), and which stars are survey-selected sources
        (Component.selected_sources) with their weighting."""
        l_deg, b_deg, m_np, v0_np = k21.line_of_sight_basis(ra_rad, dec_rad)
        self._k21_lb = (l_deg, b_deg)
        self._k21_basis = (m_np, v0_np)
        self._k21_branches = k21.branches_for(l_deg, b_deg)
        n_star = int(np.atleast_1d(stars.distance.initval).size)
        src = np.zeros(n_star, dtype=bool)
        for comp in system.active_components.values():
            for k in comp.selected_sources(system):
                if not 0 <= int(k) < n_star:
                    raise ValueError(
                        f"[{self.prefix}] '{comp.prefix}' selected source "
                        f"star {k}, but there are {n_star} stars."
                    )
                src[int(k)] = True
        self._k21_src = src
        self._k21_binary = {}
        if self.binaries:
            for comp in system.active_components.values():
                self._k21_binary.update(comp.unresolved_binary_lenses(system))
        if self.source_selection == "sqrt_distance":
            self._k21_selection = ("sqrt_distance",)
        else:
            ai0, h = k21.extinction_scale(
                l_deg, b_deg, self.source_ai_rc, self.source_dm_rc
            )
            self._k21_selection = (
                "luminosity_function",
                *self.source_i_range,
                float(ai0),
                float(h),
            )
        sel = self._k21_selection
        self._k21_root_selection = lambda k: (
            l_deg,
            b_deg,
            sel if src[k] else None,
        )

    def _k21_branch_terms(self, stars):
        """koshimoto21's per-branch log number densities and NORMALIZED log
        velocity densities, one per-star vector each, in the sight line's
        branch order, plus the cylindrical radius (kpc) the population
        priors read.  Positions are genulens's Dlb2xyz and velocities
        genulens's (l, b) projection (koshimoto21.line_of_sight_basis).  A
        survey-selected source takes genulens's SOURCE density instead
        (main-sequence count times the source weight w_S, which replaces
        the D^2 volume element: k21.log_source_weights)."""
        l_deg, b_deg = self._k21_lb
        m_np, v0_np = self._k21_basis
        br = self._k21_branches
        d_pc = pt.maximum(stars.distance.value, 1e-3)
        d_kpc = d_pc / 1e3
        x, y, z = k21.dlb_to_xyz(d_pc, l_deg, b_deg, xp=pt)
        m = pt.as_tensor_variable(m_np)
        cols = pt.stack(
            [
                k21.K_KMS * stars.pm_ra.value * d_kpc,
                k21.K_KMS * stars.pm_dec.value * d_kpc,
                stars.rv.value / 1e3,
            ],
            axis=1,
        )
        vel = pt.as_tensor_variable(v0_np) + (m @ cols[:, :, None]).squeeze(-1)
        dens = k21.log_number_densities(x, y, z, xp=pt)
        velo = k21.log_velocity_densities(
            x, y, z, vel[:, 0], vel[:, 1], vel[:, 2], xp=pt
        )
        self._k21_state = (x, y, z, vel)
        self._k21_distance = d_pc
        r_kpc = pt.sqrt(x * x + y * y + 1.0) / 1e3
        log_dens = [dens[i] for i in br]
        if self._k21_src.any():
            s = self._k21_src.astype(float)
            dens_ms = k21.log_number_densities(
                x, y, z, xp=pt, luminous_only=True
            )
            w_s = k21.log_source_weights(br, d_pc, self._k21_selection, xp=pt)
            log_dens = [
                (1.0 - s) * d + s * (dens_ms[i] + w)
                for d, i, w in zip(log_dens, br, w_s)
            ]
        return log_dens, [velo[i] for i in br], r_kpc

    def _build_k21_mass_function(self, stars):
        bounds = _sampled_bounds(stars.logmass)
        if bounds is None:
            raise ValueError(
                f"[{self.prefix}] model: koshimoto21 needs finite "
                f"star.logmass lower/upper bounds to normalize its mass "
                f"function; none could be read."
            )
        n_star = int(np.atleast_1d(stars.logmass.initval).size)
        lower = np.broadcast_to(np.atleast_1d(bounds[0]), (n_star,))
        upper = np.broadcast_to(np.atleast_1d(bounds[1]), (n_star,))
        return k21.MassFunction(
            lower, upper, remnants=self.remnants, branches=self._k21_branches
        )

    def _k21_mass_velocity_terms(self, x_mass, pdmf_mask, age_mask, log_vel):
        """Per branch, per star: log [PDMF x velocity density] of the
        genulens port, normalized over the star's logmass support:

            log[ phi_still(x) p_b(v) + phi_NS(x) p_NS,b(v)
                 + phi_BH(x) p_BH,b(v) ] - log Z_b

        (``still`` = luminous stars and white dwarfs, which keep the
        branch's velocity distribution; neutron stars and black holes carry
        genulens's natal kicks, koshimoto21.log_kicked_velocity).  A
        survey-selected SOURCE is a luminous star (genulens draws sources
        from n0MS): the luminous PDMF alone, normalized over the luminous
        objects.  A star whose AGE is sampled is luminous by its track:
        K21's IMF alone (its alive condition is the track's).  A star off
        the stellar IMF (an FFP) keeps the bare velocity density.  Also
        stores, per branch, the MASS-ONLY term (``_k21_mass_only``) a
        member star whose whole velocity is its root's takes.  Arithmetic
        0/1 weights on finite terms -- no selection."""
        mf = self._k21_mf
        x, y, z, vel = self._k21_state
        w = pdmf_mask.astype(float)
        a = (age_mask & pdmf_mask).astype(float)
        src = self._k21_src.astype(float)
        bases = k21.kick_bases(x, y, z, xp=pt, branches=self._k21_branches)
        log_imf = k21.log_imf(x_mass, xp=pt) - mf.log_imf_peak
        out, mass_only = [], []
        for j in range(len(self._k21_branches)):
            tab = {
                g: interp_table(
                    mf.log_table[g][j], mf.x0, PDMF_GRID_DEX, x_mass
                )
                for g in mf.GROUPS
            }
            log_lum = interp_table(mf.log_lum[j], mf.x0, PDMF_GRID_DEX, x_mass)
            if self._k21_binary:
                tab["still"] = self._k21_binary_still(j, tab["still"], x_mass)
            mean, sig = bases[j]
            kicked = [
                tab[g]
                + k21.log_kicked_velocity(
                    vel[:, 0], vel[:, 1], vel[:, 2], mean, sig, speed, xp=pt
                )
                for g, speed in (("NS", k21.KICK_NS), ("BH", k21.KICK_BH))
            ]
            mix = pm.math.logsumexp(
                pt.stack([tab["still"] + log_vel[j], *kicked]), axis=0
            )
            total = pm.math.logsumexp(
                pt.stack([tab[g] for g in mf.GROUPS]), axis=0
            )
            field = (
                (1.0 - a) * (mix - log_vel[j]) + a * log_imf - mf.log_norm[j]
            )
            field_m = (1.0 - a) * total + a * log_imf - mf.log_norm[j]
            source = (1.0 - a) * log_lum + a * log_imf - mf.log_norm_lum[j]
            out.append(log_vel[j] + w * ((1.0 - src) * field + src * source))
            mass_only.append(w * ((1.0 - src) * field_m + src * source))
        self._k21_mass_only = mass_only
        return out

    def _k21_binary_still(self, j, still, x_mass):
        """For an unresolved-binary lens (binaries: true), branch j's
        luminous-plus-white-dwarf term with the luminous part replaced by
        genulens's effective binary-lens density at the event's scale r0 =
        sqrt(kappa pi_rel) D_L (koshimoto21.log_binary_lens_density; white
        dwarfs get no companion in genulens).  Other stars keep ``still``;
        assembled with set_subtensor."""
        mf = self._k21_mf
        stars_d = self._k21_distance
        b = self._k21_branches[j]
        log_wd = interp_table(mf.log_wd[j], mf.x0, PDMF_GRID_DEX, x_mass)
        out = still
        for lens, src in sorted(self._k21_binary.items()):
            lr0 = k21.log_r0(stars_d[lens], stars_d[src], xp=pt)
            log_bin = (
                k21.log_binary_lens_density(b, x_mass[lens], lr0, xp=pt)
                - mf.log_imf_peak
            )
            out = pt.set_subtensor(
                out[lens], pt.logaddexp(log_bin, log_wd[lens])
            )
        return out

    def _k21_grouped_logp(
        self,
        log_dens,
        branch_logp,
        volume_element,
        velocity_jacobian,
        root_of,
        vel_linked,
    ):
        """The genulens port with linked stars: per group (a root and the
        members sharing its position), ONE density at the root's position,
        ONE volume element and ONE branch for the whole system:

            log p_g = logsumexp_b [ log n_b(d_r) + sum_k T_b,k ] + 2 log d_r

        A member whose velocity is entirely its root's (all three
        coordinates hard-linked: a binary companion) contributes its MASS
        term only; one with its own velocity contributes its full
        (normalized) mass x velocity term and its own velocity Jacobian,
        with the rv column's m/s -> km/s owed in-model so a member
        integrates to exactly 1.  A partly linked velocity raises: the
        disk's v_phi density is not Gaussian, so the conditional density of
        the free coordinates given the linked ones has no closed form."""
        n_star = len(root_of)
        roots = sorted(set(int(j) for j in root_of))
        members = [k for k in range(n_star) if root_of[k] != k]
        partial = {k: c for k, c in vel_linked.items() if len(c) != 3}
        if partial or any(int(root_of[k]) == k for k in vel_linked):
            raise NotImplementedError(
                f"[{self.prefix}] model: koshimoto21 supports a linked star "
                f"whose velocity is entirely its root's, or entirely its own; "
                f"star(s) {sorted(partial) or sorted(vel_linked)} have a "
                f"partly linked (or root-linked) velocity, whose conditional "
                f"density has no closed form under genulens's non-Gaussian "
                f"disk kinematics."
            )
        dens = [d for d in log_dens]
        T = [bl - d for bl, d in zip(branch_logp, dens)]
        J = velocity_jacobian
        mass_only = getattr(self, "_k21_mass_only", None)
        for k in members:
            if k in vel_linked:
                for b in range(len(T)):
                    mo = 0.0 if mass_only is None else mass_only[b][k]
                    # population terms (age/[Fe/H]) ride in T - the
                    # velocity part; keep them, drop the velocity
                    T[b] = pt.set_subtensor(T[b][k], mo + self._k21_pop[b][k])
                J = pt.set_subtensor(J[k], 0.0)
            else:
                J = pt.set_subtensor(J[k], J[k] + _LOG_KMS_PER_MS)
        W = np.zeros((len(roots), n_star))
        for k in range(n_star):
            W[roots.index(int(root_of[k])), k] = 1.0
        W = pt.as_tensor_variable(W)
        roots_idx = np.asarray(roots, dtype=np.int64)
        group_branch = [d[roots_idx] + pt.dot(W, t) for d, t in zip(dens, T)]
        return (
            pt.sum(pm.math.logsumexp(pt.stack(group_branch), axis=0))
            + pt.sum(volume_element[roots_idx])
            + pt.sum(J)
        )

    def _add_k21_remnant_probability(
        self, x_mass, log_dens, log_terms, pdmf_mask, age_mask
    ):
        """``galacticmodel.p_remnant`` for the genulens port: the branch
        responsibilities (density x mass x velocity) times each branch's
        dark fraction at the star's mass.  Reporting only."""
        idx = np.nonzero(pdmf_mask)[0]
        mf = self._k21_mf
        stacked = pt.stack([d + t for d, t in zip(log_dens, log_terms)])[
            :, idx
        ]
        resp = pt.exp(stacked - pm.math.logsumexp(stacked, axis=0))
        lum_only = age_mask[idx].astype(float)
        dark = (1.0 - lum_only) * pt.stack(
            [
                interp_table(
                    mf.dark_frac[j], mf.x0, PDMF_GRID_DEX, x_mass[idx]
                )
                for j in range(len(self._k21_branches))
            ]
        )
        pm.Deterministic(
            f"{self.prefix}.p_remnant", pt.sum(resp * dark, axis=0)
        )

    # ------------------------------------------------------------------
    # Linked stars and the normalization (star.md, "Linked stars: one
    # position, one density term").
    # ------------------------------------------------------------------

    def _linked_star_report(self, stars, root_of, vel_linked, refused):
        """Say, once per build, which stars share a position and what each
        linked star still contributes -- and warn for a hard-linked
        distance the rule cannot place."""
        names = list(getattr(stars, "names", None) or [])

        def label(k):
            return names[k] if k < len(names) else str(k)

        for k, j in enumerate(root_of):
            if j == k:
                continue
            linked = vel_linked.get(k, ())
            free = [
                VELOCITY_COORDS[c] for c in range(3) if c not in set(linked)
            ]
            what = (
                "nothing further (its velocity is hard-linked too)"
                if not free
                else "the density of its free velocity coordinate(s) "
                + ", ".join(free)
                + (
                    " conditional on the linked "
                    + ", ".join(VELOCITY_COORDS[c] for c in linked)
                    if linked
                    else ""
                )
                + ", in the same Galactic population"
            )
            logger.info(
                f"[{self.prefix}] star '{label(k)}' shares the position of "
                f"star '{label(int(j))}' (hard-linked distance): one "
                f"position, one density and d^2 volume term; '{label(k)}' "
                f"contributes {what}."
            )
        for k, paths in refused:
            logger.warning(
                f"[{self.prefix}] star '{label(k)}' has a hard-linked "
                f"distance that does not name exactly one other star's "
                f"distance (it reads {list(paths) or 'nothing'}), so there "
                f"is no single partner whose position it shares. It keeps "
                f"its own Galactic density term, which is then a factor on "
                f"the link's sources rather than a density of a coordinate "
                f"of its own. Link it as 'initval: star.<other>.distance' "
                f"with 'sigma: 0' to share that star's position instead."
            )

    def _kinematic_evidence_log_norm(self, stars, roots, vel_linked):
        """The constant to ADD to a log evidence for the kinematic prior to
        be a normalized density (outputs/evidence's sign convention: the
        bridge integrates exp(model logp), and lnZ_norm = lnZ + this).

        Per ROOT (a star with a position of its own), the term in the model
        is the normalized density over (distance, pm_ra, pm_dec, rv) plus

            log Z_pos - log(1e-3)

        where Z_pos = int sum_b n_b(d) w_b(d) dd over the root's static
        distance bounds (koshimoto21.log_position_norm, with the root's own
        weighting: the lens density times d^2, or a survey-selected
        source's genulens source weight) -- the stars per steradian in the
        cone -- and the last is the rv column's m/s -> km/s Jacobian (the
        velocity densities are normalized in the model, their (2 pi)^(3/2)
        included).  A root whose own velocity is hard-linked owes no
        Jacobian constant.  A root whose distance is PINNED owes log of its
        weighted density at the pin instead of Z_pos (the remaining velocity
        mixture is then normalized).  A member owes nothing: its term is
        exact in the model.  Velocity bounds are treated as infinite (pm
        +/- 1e5 mas/yr is > 400 km/s beyond 1 pc).
        """
        bounds = _sampled_bounds(stars.distance)
        is_derived = getattr(stars.distance, "element_is_derived", None)
        links = getattr(stars.distance, "element_links", None) or {}
        if links.get("lower") or links.get("upper"):
            logger.warning(
                f"[{self.prefix}] star.distance carries a linked lower/upper "
                f"bound; the kinematic prior's position normalizer uses the "
                f"STATIC bounds, so evidence_log_norm is approximate there."
            )
        total = 0.0
        cache = {}
        for k in roots:
            los = self._k21_root_selection(k)
            sampled = stars.distance.element_is_sampled(k) or bool(
                is_derived(k) if callable(is_derived) else False
            )
            if sampled:
                if bounds is None:
                    logger.warning(
                        f"[{self.prefix}] cannot read finite star.distance "
                        f"bounds; the kinematic prior's position normalizer "
                        f"is omitted from evidence_log_norm."
                    )
                    c = 0.0
                else:
                    lo = float(
                        np.atleast_1d(bounds[0])[
                            min(k, np.size(bounds[0]) - 1)
                        ]
                    )
                    hi = float(
                        np.atleast_1d(bounds[1])[
                            min(k, np.size(bounds[1]) - 1)
                        ]
                    )
                    key = (lo, hi, los)
                    if key not in cache:
                        cache[key] = k21.log_position_norm(lo, hi, *los)
                    c = cache[key]
            else:
                d0 = float(stars.distance.element_start(k))
                log_n = k21.log_total_number_density(d0, *los)
                c = float(log_n) + 2.0 * np.log(max(d0, 1e-3))
            if k not in vel_linked:
                c -= _LOG_KMS_PER_MS
            total -= c
        return float(total)

    def evidence_log_norm(self):
        """Constant that makes this component's potentials normalized priors
        in a log evidence (ADD it to lnZ; see _kinematic_evidence_log_norm).

        Covers the kinematic prior only -- the FFP potential (imf_prior) is already
        normalized over star.logmass's support.  Available after
        build_likelihood; 0.0 before.
        """
        return float(getattr(self, "_evidence_log_norm", 0.0))

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
