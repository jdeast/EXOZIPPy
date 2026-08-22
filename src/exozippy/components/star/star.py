import logging
from collections import namedtuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component, in_topology
from exozippy.components.parameter import sampled_bounds
from exozippy.components.parameterization import (
    merge_options,
    merge_overrides,
)
from exozippy.constants import (
    FFP_MASS_FUNCTION_MIN_MEARTH,
    FFP_MASS_FUNCTION_SLOPE,
    HYDROGEN_BURNING_LIMIT,
    MSUN_TO_MEARTH,
)
from exozippy.outputs.prose import get_collector

from . import physics

logger = logging.getLogger(__name__)

# Lowest mass Sumi et al. 2023 actually fit, in dex(solMass): 0.33 M_Earth,
# -6.004 dex.  NOT a bound -- nothing in this module clamps the FFP support.
# It is quoted in the warning as one concrete candidate for the lower bound
# the USER has to choose.  See Star._warn_ffp_logmass_bound.
FFP_LOGMASS_CALIBRATION_MIN = float(
    np.log10(FFP_MASS_FUNCTION_MIN_MEARTH / MSUN_TO_MEARTH)
)


def _microlensing_only_star_indices(system):
    """Star indices that are exclusively a microlensing source body.

    The mulensing physics reads a source star's DISTANCE, proper motion and
    sky position, and its MASS only through the IMF prior -- never its
    teff or feh, and its radius only under ``finite_source`` (rho's deps
    include ``star.radius[source_map]``; see mulensing/defaults.yaml).  A
    source-only star's mass therefore has no likelihood term of its own,
    which is what the pins below are for.

    NOTE this is deliberately no longer the predicate for radius/teff/feh.
    It used to claim mulensing "never reads a source star's mass/teff/feh/
    radius", which is false under finite_source (review 3.8.1) -- and being
    a microlensing fact it could only ever answer for microlensing
    topologies, so it pinned the SOURCE star's radius while leaving the
    LENS star's, equally unread, free.  Readership is a property of the
    whole topology, so it is answered by ``Star.structure_consumers``, for
    every star at once.
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


# One reader of one star's structural parameter.  ``param`` is the star
# parameter read, ``star`` the index of the star it belongs to, and ``label``
# names the reader for the log line and the degeneracy warning.
StarConsumer = namedtuple("StarConsumer", "label param star")


class Star(Component):
    # Which mass prior each star draws from.  "imf" (the default, and what
    # every config written before this key existed gets) means the stellar
    # initial mass function chosen by the galacticmodel block's `IMF:` key;
    # "ffp" means the free-floating-planet mass function, i.e.
    # galacticmodel.ffp_logmass_logp.
    #
    # Why PER STAR and not a galacticmodel-level key: the IMF potential is a
    # plain sum over the whole star.logmass vector, lens and source alike, and
    # the system this exists for -- an FFP lens crossing a bulge SOURCE star
    # -- needs a different mass prior on each.  Putting the choice on the star
    # block keeps it next to the mass it describes, next to the other
    # per-star model switches (mist:, parsec:), and resolvable by the star's
    # own name; a galacticmodel key listing star names/indices would be a
    # second place the star topology is spelled out, and would have to
    # re-implement name resolution that Component already does.
    MASS_FUNCTIONS = ("imf", "ffp")

    # Sub-keys of the dict form, `mass_function: {kind: ffp, alpha: ...}`.
    # Deliberately short.  The support's lower edge is NOT here: it is
    # star.<name>.logmass's ordinary `lower` bound, and a second way to set
    # the same number would be a second thing to keep consistent.  The pivot
    # mass and the abundance are not here either -- see
    # galacticmodel.ffp_logmass_logp for why neither can affect this prior.
    FFP_OPTION_KEYS = ("kind", "alpha")

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Stellar Parameters"
        self.mist = [c.get("mist", True) for c in self.config]
        self.parsec = [c.get("parsec", False) for c in self.config]

        self._parse_mass_functions()

    @property
    def prefix(self):
        return "star"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "mass_function",
                "kind": "option",
                "accepts": list(cls.MASS_FUNCTIONS),
                "required": False,
                "doc": (
                    "Which mass prior this star draws from (default 'imf'). "
                    "'imf' is the stellar initial mass function selected by "
                    "the galacticmodel block's IMF: key.  'ffp' is the "
                    "free-floating-planet mass function of Sumi et al. 2023 "
                    "(a power law in log mass, dN/dlogM ~ M^-0.96), for a "
                    "microlensing lens that is a free-floating planet -- "
                    "such a lens must be declared as a 'star' block with a "
                    "low logmass, and this is what keeps it from also being "
                    "charged the stellar IMF.  The choice is per star, so a "
                    "stellar source and an FFP lens can each get the right "
                    "prior.  The slope is tunable with the dict form, "
                    "'mass_function: {kind: ffp, alpha: 0.96}' (alpha is the "
                    "positive exponent of dN/dlogM ~ M^-alpha), because the "
                    "measurement is uncertain and will be revised.  Selecting "
                    "it warns about star.<name>.logmass's lower bound: the "
                    "density rises toward low mass, so that bound is a real "
                    "prior choice and the default (-9 dex) is not a decision "
                    "anyone made.  To implement a different functional form "
                    "entirely, edit galacticmodel.ffp_logmass_logp -- it is "
                    "the only place the form appears.  Requires a "
                    "galacticmodel block; without one no mass prior is "
                    "applied at all."
                ),
            },
            {
                "key": "mist",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Constrain this star with the MIST evolutionary model "
                    "(default true).  Only consulted when an "
                    "evolutionarymodel block exists."
                ),
            },
            {
                "key": "parsec",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Constrain this star with the PARSEC evolutionary model "
                    "(default false).  Only consulted when an "
                    "evolutionarymodel block exists."
                ),
            },
        ]

    def _parse_mass_functions(self):
        """Read each star's ``mass_function:`` key into per-star arrays.

        Sets three parallel, ``n_elements``-long attributes that the
        galacticmodel reads at stage 7:

          ``mass_functions`` -- "imf" or "ffp", one per star
          ``ffp_mask``       -- the same thing as a boolean array
          ``ffp_alpha``      -- the FFP slope per star (the published default
                                on every star, including the "imf" ones,
                                where it is simply unused)

        Two spellings are accepted.  The string form, ``mass_function: ffp``,
        is the whole story for a user taking the published measurement as-is.
        The dict form, ``mass_function: {kind: ffp, alpha: 1.2}``, exists
        because this measurement is uncertain and will be revised -- nobody
        should have to edit the source to track a new fit.  Unknown kinds and
        unknown sub-keys both raise: a silently ignored mass-function key is
        precisely the bug PR #82 fixed for ``IMF: Salpeter``.
        """
        self.mass_functions = []
        self.ffp_alpha = np.full(self.n_elements, FFP_MASS_FUNCTION_SLOPE)

        for i, c in enumerate(self.config):
            spec = c.get("mass_function")
            name = f"{self.prefix}.{self.names[i]}"

            if spec is None:
                self.mass_functions.append("imf")
                continue

            if isinstance(spec, dict):
                unknown = set(spec) - set(self.FFP_OPTION_KEYS)
                if unknown:
                    raise ValueError(
                        f"{name}: unknown mass_function key(s) "
                        f"{sorted(unknown)}.  Accepted: "
                        f"{', '.join(self.FFP_OPTION_KEYS)}."
                    )
                kind = spec.get("kind")
                if kind is None:
                    raise ValueError(
                        f"{name}: mass_function given as a dict must name a "
                        f"'kind', e.g. mass_function: {{kind: ffp, alpha: "
                        f"0.96}}."
                    )
            else:
                kind, spec = spec, {}

            kind = str(kind).lower()
            if kind not in self.MASS_FUNCTIONS:
                raise ValueError(
                    f"{name}: mass_function '{kind}' is not implemented.  "
                    f"Supported: {', '.join(self.MASS_FUNCTIONS)} ('imf' = "
                    f"the stellar IMF chosen by the galacticmodel block's "
                    f"IMF: key, the default; 'ffp' = the free-floating-"
                    f"planet mass function of Sumi et al. 2023, a power law "
                    f"in log mass).  To add another, implement it alongside "
                    f"galacticmodel.ffp_logmass_logp."
                )

            if kind != "ffp" and set(spec) - {"kind"}:
                raise ValueError(
                    f"{name}: mass_function '{kind}' takes no options; "
                    f"{sorted(set(spec) - {'kind'})} apply only to 'ffp'."
                )

            if "alpha" in spec:
                self.ffp_alpha[i] = float(spec["alpha"])

            self.mass_functions.append(kind)

        self.ffp_mask = np.array(
            [m == "ffp" for m in self.mass_functions], dtype=bool
        )

    def _user_set_logmass_lower(self, i):
        """Has the user named a lower bound for star i's logmass?

        True for a numeric entry in the params file and for a link expression
        (``extract_links`` strips those out of ``user_params``, so both places
        have to be checked).  All three address spellings resolve.
        """
        cm = self.config_manager
        user_params = getattr(cm, "user_params", None) or {}
        links = getattr(cm, "links", None) or {}
        for key in (
            f"{self.prefix}.logmass",
            f"{self.prefix}.{i}.logmass",
            f"{self.prefix}.{self.names[i]}.logmass",
        ):
            entry = user_params.get(key)
            if isinstance(entry, dict) and entry.get("lower") is not None:
                return True
            if "lower" in (links.get(key) or {}):
                return True
        return False

    def _warn_ffp_logmass_bound(self):
        """Tell the user that the FFP support's lower edge is their call.

        NOT a floor.  The Salpeter branch raises star.logmass's bound because
        a *stellar* IMF below the hydrogen-burning limit is outside its domain
        of validity -- that is a correctness fix.  Here the sub-stellar range
        IS the domain, so clamping it would gate the very models this mass
        function exists to express.  The support stays at defaults.yaml's
        [-9, 2.5] dex and the user gets the information to choose.

        Fired only where the bound is still the default: a user who has set
        ``lower`` has already made the decision, and warning them anyway is
        how a codebase teaches people to ignore its warnings.

        The UPPER edge needs no such warning.  The density falls toward high
        mass, so nothing accumulates there.
        """
        if not self.ffp_mask.any():
            return

        # The bound as it stands (defaults, merged with any user entry), in
        # user units -- logmass's user and internal units are both
        # dex(solMass), so no conversion is involved.
        cfg = self.config_manager.resolve(
            self.prefix,
            "logmass",
            shape=(self.n_elements,),
            names=self.names,
        )
        current = cfg.get("lower")

        for i in np.nonzero(self.ffp_mask)[0]:
            if self._user_set_logmass_lower(i):
                continue
            lower = (
                None if current is None else float(np.atleast_1d(current)[i])
            )
            if lower is None:
                continue

            alpha = float(self.ffp_alpha[i])
            nats_per_dex = alpha * np.log(10.0)
            # p(x) ~ 10^(-alpha x) on [lower, upper]: exactly 90% of the mass
            # lies within 1/alpha dex of the lower edge (1 - 10^-1 = 0.9),
            # whatever the upper edge is.  A tidy, exact way to say "the
            # bound is the answer".
            ninety = 1.0 / alpha if alpha > 0 else float("inf")
            logger.warning(
                f"[{self.prefix}] mass_function: ffp on "
                f"{self.prefix}.{self.names[i]}.logmass -- its lower bound is "
                f"still the default {lower:g} dex ({10.0**lower:.3g} "
                f"solMass), which is a prior decision nobody made.  The FFP "
                f"mass function RISES toward low mass (dN/dlogM ~ M^-{alpha:g}"
                f", {nats_per_dex:.2f} nats per dex), so 90% of the prior "
                f"mass sits within {ninety:.2f} dex of whatever that bound "
                f"is: a mass the data do not pin down will end up floor-"
                f"dominated.  {lower:g} dex is about "
                f"{10.0**lower / 4.72e-10:.2g} Ceres masses, far below "
                f"anything a microlensing survey can detect.  Set "
                f"'{self.prefix}.{self.names[i]}.logmass: {{lower: ...}}' in "
                f"your params file to your survey's detection limit or your "
                f"own prior belief -- e.g. "
                f"{FFP_LOGMASS_CALIBRATION_MIN:.4f} dex "
                f"({FFP_MASS_FUNCTION_MIN_MEARTH:g} M_Earth), the lowest mass "
                f"Sumi+2023 fit, below which the relation is extrapolation.  "
                f"Note bounds may only be TIGHTENED, so raise it there; it "
                f"cannot later be loosened back below the default."
            )

    # ------------------------------------------------------------------
    # Which stellar structure parameters this topology actually reads
    # ------------------------------------------------------------------
    #
    # radius/teff/feh are declared for every star unconditionally, because
    # they are inputs to the reporting chain every star carries (logg,
    # density, luminosity, fbol).  Whether anything in the LIKELIHOOD reads
    # them is a property of the topology, and in a point-source microlensing
    # fit with no SED, no evolutionary model and no empirical relation the
    # answer is nothing at all: they enter no potential and no observation,
    # so free they refill their prior and pinned they change nothing.  Both
    # shipped microlensing params files say so in a comment and pin all six
    # by hand ("these don't impact the likelihood ... eventually, they won't
    # even be sampled for such fits" -- examples/ob08092), which is exactly
    # the state review 3.8.1 asked to end.
    #
    # So they are declared INACTIVE per star where nothing reads them (role 4
    # -- no raw coordinate, no potential, no table row), and left FREE where
    # something does.  The mask is computed from the CONFIG, so adding an
    # `sed:` block flips them back on with no user action, and never pins a
    # value in code for a parameter the model reads: an element is either not
    # a parameter of this configuration at all, or it is free.
    STRUCTURE_PARAMS = ("radius", "teff", "feh")

    def structure_consumers(self, system):
        """Every reader of a star's radius/teff/feh in this topology.

        THE single readership predicate for those three, in the shape
        ``Band.ld_consumers`` established: one function answering for every
        (star, parameter) pair, so a new consumer is remembered in one place
        rather than in a mask and a warning separately.

        Read from each consumer's raw ``config`` and from attributes set in
        ``__init__``/``build_maps`` (stage 2), never from anything built in
        stage 3, since component order within stage 3 is not guaranteed.

        What counts as a consumer is a term in the LIKELIHOOD, not an
        expression that merely mentions the parameter.  ``star.luminosity``
        reads radius and teff, and ``star.logg``/``star.density`` read
        radius, for every star always -- but nothing reads THEM back, so
        counting them would make the predicate trivially true everywhere and
        answer the wrong question.  (Their table rows are the known residual:
        an inactive star still reports a luminosity computed from its
        bookkeeping pin.  A derived element cannot itself be made inactive --
        ``manifest.py`` refuses that combination -- so suppressing those rows
        is a separate change.)

        The complete list, each entry naming the code that reads it:

        * **sed** -- ``SED.build_likelihood``'s teffsed and fbolsed floor
          potentials are ``pt.sum`` over the WHOLE star vector with no mask,
          and fbol is ``calc_fbol(luminosity(radius, teff), distance)``, so
          an SED reads every star's radius and teff.
          ``_predicted_appmag_node`` reads the whole ``star.feh`` vector.
          This is the "if the user supplies an SED they all become
          constrained -- weakly, and then it is useful to find out how weak"
          case.
        * **evolutionarymodel** -- a track indexes (initfeh, eep) and returns
          the present-day structure, so it reads all three of any star that
          opted in via ``mist:``/``parsec:``.  No such component ships; the
          branch fires on the config key, exactly as the age/initfeh/eep
          declaration does, so a premature block does not silently deactivate
          what it is about to want.
        * **mulensinstrument/lens** -- ``rho``'s deps are
          ``star.radius[source_map]``, and ONLY under ``finite_source``.
          This is review 3.8.1's actual defect: the old blanket pin fixed
          exactly this radius at an untouched 1.0 solRad default.
        * **planet** -- ``planet.p`` (radius/star.radius) and ``planet.ar``
          (a/star.radius) are declared only when an orbit exists
          (``Planet.register_parameters``'s ``has_orbit``), and read the
          hosts named by ``star_ndx``.  A transit or an RV orbit reaches
          star.radius through them, not directly.
        * **mann** -- the radius penalty reads ``star.radius`` and the
          relations' [Fe/H] term reads ``star.feh``, for its target stars.
        * **torres** -- reads teff, feh and radius (and logg, hence radius
          again) of its target stars.
        """
        out = []

        def _mark(label, param, stars):
            # `stars` may be a numpy index map (planet.star_map).  Never
            # `stars or []`: bool() of a 1-element array reads the ELEMENT,
            # so a lone planet around star 0 -- array([0]) -- tests False and
            # the whole consumer vanishes.  It did, for one commit: hd80606
            # and kelt17 deactivated a star radius that planet.p reads.
            for s in [] if stars is None else list(stars):
                s = int(s)
                if 0 <= s < self.n_elements:
                    out.append(StarConsumer(label, param, s))

        all_stars = range(self.n_elements)

        def _in_topology(name):
            return in_topology(system, name) is not None

        if _in_topology("sed"):
            for param in self.STRUCTURE_PARAMS:
                _mark("sed", param, all_stars)

        if _in_topology("evolutionarymodel"):
            opted = [
                i
                for i, (m, p) in enumerate(zip(self.mist, self.parsec))
                if m or p
            ]
            for param in self.STRUCTURE_PARAMS:
                _mark("evolutionarymodel", param, opted)

        lens = getattr(system, "lens", None)
        if lens is not None and any(getattr(lens, "finite_source", [])):
            # `any`, not `[0]`, and every source body rather than
            # source_map[0]: the conservative direction, matching
            # Band.ld_consumers' reasoning about the same flag.
            sources = {
                idx
                for event in getattr(lens, "source_bodies", None) or []
                for (ctype, idx) in event
                if ctype == "star"
            }
            _mark("lens(finite_source)", "radius", sorted(sources))

        planet = getattr(system, "planet", None)
        if planet is not None and _in_topology("orbit"):
            _mark("planet(p, ar)", "radius", getattr(planet, "star_map", None))

        for name, params in (
            ("mann", ("radius", "feh")),
            ("torres", ("radius", "teff", "feh")),
        ):
            comp = getattr(system, name, None)
            if comp is None:
                continue
            for param in params:
                _mark(name, param, getattr(comp, "star_indices", None))

        for param in self.STRUCTURE_PARAMS:
            _mark("user prior", param, self._user_prior_stars(param))

        return out

    # The fields that state a posterior term or a support, exactly as
    # Parameter._user_constraint_fields defines them -- and deliberately NOT
    # `initval`, which is a start value and cannot move a posterior.  A star
    # whose params file carries only an initval for its radius has said where
    # the number is, not that anything should fit it.
    USER_CONSTRAINT_FIELDS = ("mu", "sigma", "lower", "upper")

    def _user_prior_stars(self, param):
        """Stars whose ``param`` the USER constrained in the params file.

        A ``mu``/``sigma`` is a term in the logp -- the user asserting a
        spectroscopic measurement -- so it makes the parameter read by
        definition, and the element has to stay free for it to apply to.
        Five shipped examples do exactly this (kelt17's 7454 +/- 75 K and
        [Fe/H] 0.21 +/- 0.08, hd80606, GaiaBH1, HIP1349, kelt4):
        deactivating those would DROP a real constraint and stop reporting a
        quantity the user measured, which is the opposite of the point.  It
        is also why readership cannot be answered from the component topology
        alone.

        ``lower``/``upper`` count too, and that is a deliberate widening
        rather than an oversight.  It is arguable that a bound on a quantity
        nothing reads restrains nothing -- but the user has stated an opinion
        about this parameter, and the cost of ignoring it is concrete:
        ``solve_api._bounds_diagnostics`` skips inactive elements, so a bound
        that excludes its own initval stopped being reported to the GUI at
        all (caught by
        ``test_solve_api.py::test_bounds_excluding_initval_yields_diagnostic``,
        which sets exactly ``star.0.teff: {initval: 6207, lower: 7000}``).
        Silently dropping a user's input is the failure mode this whole item
        is about; do not narrow this back.

        ``sigma: 0`` is the ONE exception, and it overrides the rest of the
        entry.  That is a pin, and the inactive role subsumes it exactly --
        the value is held, nothing is sampled, no prior applies -- so
        honoring it here would make review 3.8.1's fix a no-op on the very
        files it exists for (ob08092 and DC2018_128 pinned all six that way
        by hand).  Those entries become redundant, and ``build_pymc`` says so
        per element.

        Read from ``user_params`` (standardized to the index spelling at
        ConfigManager construction, i.e. before this runs) and never from the
        resolved vectors -- every parameter has bounds and many have a sigma
        from defaults.yaml, so a resolved value says nothing about who asked
        for it.  Most specific spelling wins, as everywhere else: the index
        and name forms are checked before the broadcast.  Any lookup fault
        degrades to "the user wrote nothing", which is the conservative
        direction here only in the sense that it matches the old behavior;
        it cannot happen for a well-formed params file.
        """

        def is_constrained(entry):
            sigma = entry.get("sigma")
            if isinstance(sigma, (int, float)) and float(sigma) == 0.0:
                return False  # an explicit pin, which inactive subsumes
            return any(f in entry for f in self.USER_CONSTRAINT_FIELDS)

        return self._user_entry_stars(param, is_constrained)

    def _user_pinned_stars(self, param):
        """Stars whose ``param`` the user pinned outright (``sigma: 0``)."""

        def is_pin(entry):
            sigma = entry.get("sigma")
            return isinstance(sigma, (int, float)) and float(sigma) == 0.0

        return self._user_entry_stars(param, is_pin)

    def _user_entry_stars(self, param, predicate):
        """Stars whose most specific params-file entry satisfies ``predicate``.

        The one lookup behind ``_user_prior_stars`` and
        ``_user_pinned_stars``, so the two cannot disagree about which entry
        they are reading.  Most specific spelling wins, as everywhere else,
        which is why this stops at the first hit rather than unioning the
        three (contrast ``Parameter._user_constraint_fields``, whose union is
        a warning heuristic and does not have to adjudicate).
        """
        params = getattr(self.config_manager, "user_params", None) or {}
        if not params:
            return []
        out = []
        for i in range(self.n_elements):
            keys = [f"{self.prefix}.{i}.{param}"]
            if i < len(self.names):
                keys.append(f"{self.prefix}.{self.names[i]}.{param}")
            keys.append(f"{self.prefix}.{param}")
            for key in keys:
                entry = params.get(key)
                if not isinstance(entry, dict):
                    continue
                if predicate(entry):
                    out.append(i)
                break  # most specific spelling wins
        return out

    def _apply_structure_activity(self, system):
        """Mark radius/teff/feh inactive on the stars nothing reads them for.

        Called at the end of ``register_parameters``, when the manifest is
        complete.  A parameter no star uses at all is still DECLARED (wholly
        inactive) rather than dropped, unlike ``mode_manifest``'s rule:
        ``star.logg``/``density``/``luminosity`` name radius and teff in their
        own ``deps``, so dropping either would be a build-graph dependency
        error rather than a saving.
        """
        consumers = self.structure_consumers(system)
        read = {p: set() for p in self.STRUCTURE_PARAMS}
        for c in consumers:
            if c.param in read:
                read[c.param].add(c.star)

        deactivated = {}
        for param in self.STRUCTURE_PARAMS:
            if param not in self.manifest:
                continue
            active = [i in read[param] for i in range(self.n_elements)]
            if all(active):
                continue
            self.manifest[param] = merge_options(
                self.manifest[param], mask=active
            )
            deactivated[param] = [
                self.names[i] for i, on in enumerate(active) if not on
            ]

        if deactivated:
            detail = "; ".join(
                f"{p} ({', '.join(names)})"
                for p, names in sorted(deactivated.items())
            )
            logger.info(
                f"[{self.prefix}] nothing in this topology reads {detail}, so "
                f"those are not parameters of this fit: they are held at "
                f"their resolved values, sample nothing and are not reported. "
                f"Add an 'sed:' block (or mann/torres, or an evolutionary "
                f"model) and they become free and constrained automatically."
            )

        self._warn_finite_source_radius(system, read["radius"])

    def _warn_finite_source_radius(self, system, radius_readers):
        """Rope, not gates: a finite-source source radius is free but
        degenerate unless something else pins the angular source size.

        ``rho = theta_star/theta_E`` with ``theta_star = R_S/d_S``, so the
        light curve measures rho and NOT the radius: theta_E and d_S absorb
        any rescaling of it.  That degeneracy is exactly why the blanket pin
        existed, and it is the wrong answer -- pinning it at an untouched
        1.0 solRad default is a modeling choice made silently by the code,
        and a finite-source non-detection genuinely bounds rho, hence the
        radius, from ABOVE.  So it is free, and the degeneracy is named.
        """
        lens = getattr(system, "lens", None)
        if lens is None or not any(getattr(lens, "finite_source", [])):
            return

        # Only the stars whose radius is read SOLELY because of the finite
        # source: an SED, mann or torres already supplies the missing
        # constraint, and a user prior IS the constraint, so warning in
        # either case would be noise.
        others = {
            c.star
            for c in self.structure_consumers(system)
            if c.param == "radius" and c.label != "lens(finite_source)"
        }
        # A user's `sigma: 0` is not a prior (so it left the element read
        # only by the finite source) but it IS a decision -- they pinned the
        # radius themselves.  Telling them it is degenerate and offering to
        # let them constrain it is exactly the "warning people learn to
        # ignore" this codebase avoids elsewhere.
        others |= set(self._user_pinned_stars("radius"))
        degenerate = sorted(radius_readers - others)
        if not degenerate:
            return

        names = ", ".join(self.names[i] for i in degenerate)
        logger.warning(
            f"[{self.prefix}] finite_source is on, so the source radius of "
            f"{names} IS read (rho = theta_star/theta_E) -- but it is not "
            f"separately identifiable: theta_E and the source distance "
            f"absorb any rescaling of it, so it is sampled with only the "
            f"light curve's UPPER limit on rho constraining it.  It is left "
            f"free deliberately (that upper limit is real information, and "
            f"pinning it would be this code choosing a radius for you).  To "
            f"break the degeneracy supply the angular source size another "
            f"way: an 'sed:' block, a mann/torres relation on that star, or "
            f"an explicit prior -- 'star.{self.names[degenerate[0]]}.radius: "
            f"{{mu: ..., sigma: ...}}'."
        )

    def _galactic_imf(self, system):
        """(galacticmodel present?, its IMF name) as (bool, str or None).

        ``in_topology`` prefers the instantiated component and falls back to
        the raw config, so the answer does not depend on whether
        galacticmodel happens to have been built before the stars -- a missed
        lookup here would silently drop a mass-prior floor, which is the one
        failure mode the floors exist to prevent.  What is left here is the
        only part that is this caller's own: reading ``IMF:`` off whichever
        of the two shapes came back.
        """
        found = in_topology(system, "galacticmodel")
        if found is None:
            return False, None

        imf = getattr(found, "imf", None)
        if imf is None:
            # A raw config block: a list of instances, or (defensively) one
            # bare dict.  GalacticModel itself rejects more than one, so
            # config[0] is the whole story.  An EMPTY list is treated as no
            # galacticmodel, which is the historical answer here and the
            # conservative one -- there is no instance to draw an IMF from.
            # (Contrast the plain topology question, where an empty block is
            # a real answer; a premature `evolutionarymodel: {}` is tested.)
            if isinstance(found, (list, tuple)):
                if not found:
                    return False, None
                found = found[0]
            imf = (found or {}).get("IMF", "chabrier")
        return True, str(imf).lower()

    def _salpeter_logmass_floor(self, imf):
        """Lower bound to impose on logmass under a power-law IMF, or None.

        defaults.yaml keeps logmass's floor at an unphysical -9 dex on
        purpose: a planetary-mass LENS has to be declared as a star (see
        Lens._validate_bodies, whose two guards advertise exactly that
        workaround).  Under the default Chabrier lognormal that floor is
        inert -- the prior density at -9 dex is ~exp(-107), so the bound is a
        safety rail nothing ever touches.

        Under IMF: Salpeter it stops being a rail and becomes the answer.
        The power law rises toward low mass without limit, at
        (1 - 2.35) * ln10 = 3.11 nats per dex, so it accumulates ~26 nats of
        preference between the Chabrier peak and -9 dex; whatever the data
        allow, the posterior piles against the floor.  A stellar IMF also
        simply does not apply below the hydrogen-burning limit -- that is a
        brown dwarf, with its own (poorly constrained) mass function.

        So under Salpeter the floor is raised to the hydrogen-burning limit.
        NOTE this is not a claim that a single Salpeter power law is accurate
        down to there: Salpeter (1955) fit ~0.4-10 Msun, and the real IMF
        flattens below ~0.5 Msun, which is precisely why Kroupa and Chabrier
        exist.  Everything below ~0.5 Msun is extrapolation.  The
        hydrogen-burning limit is the honest floor for a *stellar* IMF, and
        it keeps the M-dwarf population that dominates real microlensing
        lenses inside the prior -- truncating at Salpeter's own 0.4 Msun
        calibration limit would exclude most actual lenses and make the prior
        actively wrong for the science case it is there to serve.

        Contrast the FFP mass function, which gets NO automatic floor (see
        _warn_ffp_logmass_bound).  The distinction is domain of validity, not
        taste: a stellar IMF below the hydrogen-burning limit is being applied
        outside its domain, which is a correctness problem, whereas the
        sub-stellar range IS the FFP relation's domain and its lower cutoff is
        an ordinary prior choice that belongs to the user.
        """
        if imf != "salpeter":
            return None
        return float(np.log10(HYDROGEN_BURNING_LIMIT))

    def _logmass_manifest_entry(self, system):
        """Manifest entry for logmass: the power-law IMF floor, where it
        applies, plus the FFP advisory.

        Returns ``None`` (a plain free parameter, byte-for-byte the
        pre-2026-08 model) unless some star draws a power-law stellar IMF.

        The floor goes through the manifest's "overrides" channel, NOT its
        ordinary options.  Options are merged OVER the resolved config
        (``{**cfg, **options}``), which would replace a user's tighter bound
        with the looser floor -- i.e. silently LOWER a bound the user raised.
        Overrides instead go through ConfigManager.resolve's ``apply_value``,
        which for the "lower" key keeps ``max(current, new)`` (and ``min``
        for "upper").  The user's own params go through that same
        ``apply_value``, so the result is exactly ``max(user_lower, floor)``
        regardless of which is applied first: a user bound above the floor
        survives untouched, and a user bound below it is raised to the floor.
        That last case is deliberate -- asking for Salpeter and for support
        below the hydrogen-burning limit is incoherent, and "bounds may only
        be tightened" is the house rule the same max() enforces everywhere.

        The override is per element (NaN = "leave this one alone"), because
        the mass function is per star: a star that opted into the FFP mass
        function is not drawing the stellar IMF and must not inherit its
        hydrogen-burning floor -- that is the whole point of selecting it.
        """
        has_gm, imf = self._galactic_imf(system)

        if self.ffp_mask.any() and not has_gm:
            names = ", ".join(
                f"{self.prefix}.{self.names[i]}"
                for i in np.nonzero(self.ffp_mask)[0]
            )
            logger.warning(
                f"[{self.prefix}] mass_function: ffp is set on {names} but "
                f"the config has no 'galacticmodel' block, so NO mass prior "
                f"is applied to any star and the key does nothing.  Add a "
                f"galacticmodel block to get the free-floating-planet mass "
                f"function (and the galactic density and kinematic priors "
                f"that go with a microlensing lens)."
            )

        if has_gm:
            self._warn_ffp_logmass_bound()

        floor = self._salpeter_logmass_floor(imf) if has_gm else None
        if floor is None:
            return None

        # NaN leaves an element alone; only the stars actually drawing the
        # stellar IMF get the floor.
        floors = np.where(self.ffp_mask, np.nan, floor)
        if not np.any(np.isfinite(floors)):
            return None

        # Warn only where the floor actually moves the bound: resolve() gives
        # the bound as it stands now (defaults, already merged with any user
        # entry), in user units -- logmass's user and internal units are both
        # dex(solMass), so no conversion is involved.
        cfg = self.config_manager.resolve(
            self.prefix,
            "logmass",
            shape=(self.n_elements,),
            names=self.names,
        )
        current = cfg.get("lower")
        for i in range(self.n_elements):
            if not np.isfinite(floors[i]):
                continue
            old = None if current is None else float(np.atleast_1d(current)[i])
            if old is not None and old >= floor:
                continue
            old_txt = (
                "unbounded"
                if old is None
                else f"{old:g} dex ({10.0**old:.3g} solMass)"
            )
            logger.warning(
                f"[{self.prefix}] IMF: Salpeter -- raising the lower bound on "
                f"{self.prefix}.{self.names[i]}.logmass from {old_txt} to "
                f"{floor:.4f} dex ({HYDROGEN_BURNING_LIMIT:g} solMass, the "
                f"hydrogen-burning limit), because a power-law IMF rises "
                f"toward low mass without limit (+3.11 nats/dex) so an "
                f"unphysically low floor becomes the answer rather than a "
                f"safety rail, and a stellar IMF does not apply to "
                f"sub-stellar objects.  To avoid this, use the default "
                f"IMF: chabrier (a lognormal, calibrated across the "
                f"sub-stellar range), set a HIGHER lower bound yourself "
                f"if you want a different floor -- this bound can be "
                f"tightened but not loosened -- or, for a genuinely "
                f"sub-stellar lens, give that star "
                f"'mass_function: ffp', whose support is not clamped."
            )
        return {"overrides": {"lower": floors.tolist()}}

    def _pm_manifest_entry(self, system):
        """pm_ra/pm_dec manifest entry: sampled everywhere, except that a
        lens with `fitmurel: true` flips its PRIMARY lens star's element to
        derived (pm = pm_source + mu_rel; the sampled coordinate is the
        LC-measured relative pm on the lens component).  Deriving the LENS
        element rather than the source's is load-bearing: the source pm
        carries the tight bulge prior, so deriving the source would turn
        that prior into a difference constraint and recreate the ridge the
        swap removes.  Reads only raw lens config and stage-1/2 attributes
        (lens_bodies/source_bodies) -- component order within stage 3 is
        not guaranteed.  Also builds the two index maps the expression's
        deps name: murel_source_map (per star element, the index of the
        lens's first source star) and murel_traj_map (trajectory 0).
        """
        lens = getattr(system, "lens", None)
        if lens is None or not getattr(lens, "lens_bodies", None):
            return None
        if not bool(lens.config[0].get("fitmurel", False)):
            return None
        l_type, l_idx = lens.lens_bodies[0][0]
        s_type, s_idx = lens.source_bodies[0][0]
        if l_type != "star" or s_type != "star":
            return None
        n = self.n_elements
        self.murel_source_map = np.full(n, int(s_idx), dtype=int)
        self.murel_traj_map = np.zeros(n, dtype=int)
        return {"expr_key": {"from_mulens_murel": [int(l_idx)]}}

    def _distance_manifest_entry(self, system):
        """distance entry: sampled everywhere, except that a single-source
        lens with `fitpirel: true` flips its PRIMARY lens star's element to
        derived, D_l = 1000/(pi_rel + 1000/D_s) -- swap 2 of the surgical
        coordinate plan.  Same staging and maps as _pm_manifest_entry; the
        map builder there runs first when both flags are set.
        """
        lens = getattr(system, "lens", None)
        if lens is None or not getattr(lens, "lens_bodies", None):
            return None
        if not bool(lens.config[0].get("fitpirel", False)):
            return None
        if int(getattr(lens, "n_sources", 1)) > 1:
            return None  # lens.py warns; the flag is ignored there too
        l_type, l_idx = lens.lens_bodies[0][0]
        s_type, s_idx = lens.source_bodies[0][0]
        if l_type != "star" or s_type != "star":
            return None
        n = self.n_elements
        self.murel_source_map = np.full(n, int(s_idx), dtype=int)
        self.murel_traj_map = np.zeros(n, dtype=int)
        return {"expr_key": {"from_mulens_pirel": [int(l_idx)]}}

    def register_parameters(self, system):
        """Stage 3: Declare the manifest and push to ConfigManager."""

        # 1. Get the stellar parameters we always want.  logmass may carry a
        # power-law-IMF floor (None otherwise, i.e. a plain free parameter).
        self.manifest = {
            "logmass": self._logmass_manifest_entry(system),
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

        # Is a component in the system topology, even if it has not been
        # instantiated as an attribute yet?  One implementation, in
        # component.py -- this used to walk its own holder chain with an
        # `elif` that skipped config_manager.system_config whenever
        # system.config existed at all (review 4.8.1).
        def in_system(comp_name):
            return in_topology(system, comp_name) is not None

        # 3. Add system-dependent parameters
        if in_system("sed"):
            self.manifest.update(
                {
                    "distance": None,
                    "av": None,
                    "radiussed": None,
                    "teffsed": None,
                    "loggsed": "default",
                    "luminositysed": "default",
                    "fbolsed": "default",
                }
            )

        # An evolutionary model indexes a track by (initial metallicity, EEP)
        # and reads off the present-day age; all three are declared here, per
        # star, masked to the stars that opted into a model.  The mask is a
        # real per-element role now (a star with no track has no EEP: it is
        # held at a bookkeeping value, sampled by nothing, and reported
        # nowhere), so every piece the component needs from the star side is in
        # place and landing one requires no edit here.
        #
        # The branch is driven by the CONFIG KEY, so it also fires for a
        # premature `evolutionarymodel:` block that no component backs -- which
        # is why the claim this comment used to make ("today the branch never
        # fires") was false (review 3.8.2).  With no component to read them the
        # track coordinates of an opted-in star are sampled with nothing
        # constraining them, so say so rather than leaving it to be noticed in
        # a posterior.
        if in_system("evolutionarymodel"):
            mask = [m or p for m, p in zip(self.mist, self.parsec)]
            if any(mask):
                self.manifest.update(
                    {
                        "age": {"mask": mask},
                        "initfeh": {"mask": mask},
                        "eep": {"mask": mask},
                    }
                )
                if not hasattr(system, "evolutionarymodel"):
                    opted = [nm for nm, on in zip(self.names, mask) if on]
                    logger.warning(
                        f"[{self.prefix}] the config names an "
                        f"'evolutionarymodel' block but no such component is "
                        f"registered, so the track coordinates of "
                        f"{', '.join(opted)} (initfeh, eep) and the age they "
                        f"index are sampled with NOTHING reading them -- "
                        f"likelihood-free dimensions that only widen the "
                        f"posterior. Remove the block, or set 'mist: False' on "
                        f"those stars, until the component exists."
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
        #
        # A fourth holder chain used to be written out here too; it goes
        # through in_topology now, and what is left is this caller's own
        # question -- which MODES those instruments are in, off whichever of
        # the two shapes came back.
        astrom = in_topology(system, "astrometryinstrument")
        astrom_modes = getattr(astrom, "modes", None)
        if astrom_modes is None:
            astrom_modes = [
                (c or {}).get("mode", "gaia") for c in (astrom or [])
            ]
        has_abs_astrom = any(m in ("gaia", "abs") for m in astrom_modes)

        if in_system("lens") or in_system("galacticmodel") or has_abs_astrom:
            self.manifest.update(
                {
                    "ra": None,
                    "dec": None,
                    "pm_ra": self._pm_manifest_entry(system),
                    "pm_dec": self._pm_manifest_entry(system),
                    "distance": self._distance_manifest_entry(system),
                }
            )
        elif astrom_modes:
            self.manifest.setdefault("distance", None)

        if in_system("galacticmodel"):
            self.manifest["rv"] = None

        if "distance" in self.manifest:
            self.manifest.update({"parallax": "default", "fbol": "default"})

        # Pure microlensing-source stars: pin the parameters this topology
        # reads but supplies no likelihood term for, instead of requiring
        # every microlensing params.yaml to fix them by hand (see
        # run_event.py's old build_user_params, which did exactly this
        # per-event).
        #
        # Only logmass, ra and dec remain here, and they are tier-2 pins ON
        # PURPOSE: all three ARE read (logmass by galacticmodel's imf_prior,
        # a pt.sum over the whole star vector; ra/dec by the lens trajectory
        # geometry), so they go through the "overrides" channel, which layers
        # UNDER the params file and lets a user free them again.  radius,
        # teff and feh left this block in 2026-08 (review 3.8.1): those are
        # not read at all in such a topology, which is a stronger statement
        # and gets the structural treatment -- see _apply_structure_activity
        # below and the tier reasoning on STRUCTURE_PARAMS.
        ml_source_idx = _microlensing_only_star_indices(system)
        if ml_source_idx:
            relation_idx = set()
            for relation in ("mann", "torres"):
                comp = getattr(system, relation, None)
                if comp is not None:
                    relation_idx |= set(comp.star_indices)

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
                pin = np.full(self.n_elements, np.nan)
                pin[idx_list] = 0.0
                # merge_overrides, not a hand-written
                # `dict(entry) if isinstance(entry, dict) else {}`: that
                # spelling reads the manifest vocabulary as a writer and
                # silently DROPS a bare-string expr_key, turning a derived
                # parameter into a sampled one with no message (review
                # 4.5.3, the same defect Band's autopin carried).  Latent
                # here -- none of the three parameters below is a bare string
                # today -- and unrepresentable now.
                self.manifest[param_name] = merge_overrides(
                    self.manifest[param_name], {"sigma": pin.tolist()}
                )

            _pin_sigma("logmass", relation_idx)
            _pin_sigma("ra", abs_astrom_idx)
            _pin_sigma("dec", abs_astrom_idx)

        # Last: the manifest is complete, so readership can be answered for
        # every parameter at once.
        self._apply_structure_activity(system)

    # Floor inside the volume prior's log, in pc.  The same clip
    # galacticmodel applies before its own volume element
    # (``pt.maximum(stars.distance.value, 1e-3)``), and equal to
    # defaults.yaml's lower bound on star.distance, so it is inert for any
    # ordinary fit: the logit transform already keeps a sampled element
    # strictly inside its bounds.  It only matters if a hard link ever
    # drives an element to zero, where a bare log(0) would be a -inf wall
    # with no gradient for the sampler to follow.
    DISTANCE_FLOOR_PC = 1.0e-3

    def _volume_prior_log_norm(self):
        """log Z of p(d) ~ d^2 over star.distance's hard support.

        Z = int_lower^upper d^2 dd = (upper^3 - lower^3) / 3, per element,
        which is finite (and positive) for any finite bounds.  Evaluated as
        ``3 log(upper) + log1p(-(lower/upper)^3) - log 3`` so the cube never
        has to be formed at the far end of a wide support.

        WHY NORMALIZE, when the galacticmodel-or-not choice is topology-
        driven rather than user-selected (so the PR #82/#86 argument -- make
        two user-selectable IMFs comparable -- does not apply)?  Three
        reasons, none of which is comparability across topologies:

        1. The prior this REPLACES is normalized.  A bounded, no-sigma
           element's logit reparameterization gives exactly U(lower, upper),
           an honest density integrating to 1 (see parameter.py section 5b's
           note on why no extra -log(span) belongs there).  Dropping the
           normalizer here would quietly demote star.distance's prior from a
           density to an unnormalized reweighting -- a regression in a
           property the code currently has.
        2. The bounds ARE user-settable even though the prior choice is not.
           Today, tightening `star.distance: {upper: ...}` moves logp by
           exactly the -log(span) that the tightening is worth; unnormalized,
           it would move it by an arbitrary offset instead.
        3. It is one closed-form constant with no runtime cost and no
           gradient.

        Returns 0.0 (unnormalized, with a warning) when the bounds cannot be
        read as finite floats, matching the IMF normalizers: a constant can
        never change the sampling, so it is not worth failing a fit over.
        """
        bounds = sampled_bounds(self.distance)
        if bounds is None:
            logger.warning(
                f"[{self.prefix}] Cannot read finite bounds on "
                f"{self.prefix}.distance; the d^2 volume prior is left "
                f"unnormalized (harmless for sampling, but its logp is "
                f"offset by an unknown constant)."
            )
            return 0.0
        lower, upper = bounds

        # Bounds may only be tightened and defaults.yaml's are (1e-3, 1e5)
        # pc, so this is belt and braces.
        bad = (upper <= 0.0) | (lower < 0.0) | (upper <= lower)
        if np.any(bad):
            logger.warning(
                f"[{self.prefix}] Non-positive or inverted bounds on "
                f"{self.prefix}.distance; the d^2 volume prior is left "
                f"unnormalized."
            )
            return 0.0

        # A dynamic (linked) bound is re-mapped inside build_pymc, so the
        # static bounds read above are not that element's real support and
        # this constant does not normalize it.  Say so rather than pretend.
        links = getattr(self.distance, "element_links", None) or {}
        if links.get("lower") or links.get("upper"):
            logger.warning(
                f"[{self.prefix}] {self.prefix}.distance carries a linked "
                f"lower/upper bound, so its support is dynamic; the d^2 "
                f"volume prior is normalized over the STATIC bounds and is "
                f"therefore an unnormalized reweighting inside the dynamic "
                f"interval."
            )

        return (
            3.0 * np.log(upper)
            + np.log1p(-((lower / upper) ** 3))
            - np.log(3.0)
        )

    def build_likelihood(self, model, system):
        """Stage 7: the constant-space-density (volume) prior on distance.

        A bounded element with no sigma is sampled UNIFORM in its own
        coordinate -- parameter.py's logit transform implies exactly
        U(lower, upper) -- so ``star.distance`` defaulted to uniform in d
        over defaults.yaml's [1e-3, 1e5] pc.  Nobody chose that: it is what
        fell out of the default machinery.  Transformed, it is
        p(plx) ~ plx^-2, whereas an object drawn from a locally constant
        space density gives p(d) ~ d^2, i.e. p(plx) ~ plx^-4 -- the volume
        of the shell it could have come from.  The two disagree by exactly
        d^2, which is negligible for a well-measured parallax and dominant
        for a poor one (plx/sigma below ~10), the Lutz-Kelker regime.

        DEFER TO galacticmodel WHERE ONE EXISTS.  Its ``kinematic_prior``
        already carries ``volume_element = 2*log(d)`` over the SAME full
        ``star.distance`` vector this covers -- it reads ``system.star`` and
        sums with no mask, lens and source and every other star alike -- so
        adding a second copy would apply d^4.  A galactic model is also a
        strictly stronger statement than constant space density (it has the
        disk/bulge density profiles along the actual sight line), so where
        one exists it wins outright and this term stays out of the way.

        The term applies to every element of the vector, including pinned
        and hard-linked ones, exactly as the galacticmodel priors do.  For a
        pinned element it is a constant and cannot change the posterior.

        Note this applies on top of a user's Gaussian ``sigma`` on distance,
        and that is deliberate: such a constraint is a parallax MEASUREMENT,
        and multiplying it by the volume prior is the standard treatment.
        The shift is ~2*(sigma/d)^2 in fractional distance -- 2e-4 for a
        1% parallax, 18% for a 30% one, which is precisely the regime where
        the volume prior is the correct thing to do.
        """
        if "distance" not in self.manifest:
            # No distance parameter in this topology at all (no sed, mann,
            # lens, galacticmodel or astrometry) -- nothing to weight.
            return

        has_galacticmodel, _ = self._galactic_imf(system)
        if has_galacticmodel:
            logger.debug(
                f"[{self.prefix}] galacticmodel present -- deferring the "
                f"distance prior to its density/kinematic mixture, which "
                f"already carries the d^2 volume element."
            )
            return

        distance = pt.maximum(self.distance.value, self.DISTANCE_FLOOR_PC)
        pm.Potential(
            f"{self.prefix}.volume_prior",
            pt.sum(2.0 * pt.log(distance) - self._volume_prior_log_norm()),
        )

        # Tell the reported tables what was just added.  Without this the
        # Prior column reads whatever star.distance's own fields imply --
        # "Uniform" for a bounded element with no sigma -- which is exactly
        # the prior this potential replaces.  supersedes_bounds: the term IS
        # a normalized density over star.distance's own support, so the
        # rendered text is "p(d) propto d^2 on [lower, upper]"; a user's
        # Gaussian sigma (a parallax measurement) is kept alongside it,
        # matching the "applies on top of" note above.
        self.distance.add_prior_contribution(
            latex=r"$p(d) \propto d^{2}$",
            text="p(d) propto d^2",
            supersedes_bounds=True,
        )
        get_collector(system).add(
            r"We applied a constant-space-density (volume) prior, "
            r"$p(d) \propto d^{2}$, to each modeled star's distance, the "
            r"appropriate weighting for a poorly measured parallax "
            r"\citep{Lutz:1973}.",
            section="priors",
            key=f"{self.prefix}.volume_prior",
            rank=15,
        )
