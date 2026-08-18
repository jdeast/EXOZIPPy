import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.parameter import sampled_bounds
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

    def _galactic_imf(self, system):
        """(galacticmodel present?, its IMF name) as (bool, str or None).

        Prefers the instantiated component and falls back to the raw config,
        so the answer does not depend on whether galacticmodel happens to
        have been built before the stars -- a missed lookup here would
        silently drop a mass-prior floor, which is the one failure mode the
        floors exist to prevent.
        """
        gm = None
        if hasattr(system, "active_components"):
            gm = system.active_components.get("galacticmodel")
        if gm is None:
            gm = getattr(system, "galacticmodel", None)
        if gm is not None:
            return True, str(getattr(gm, "imf", "chabrier")).lower()

        cfgs = None
        for holder in (
            getattr(system, "config", None),
            getattr(
                getattr(system, "config_manager", None), "system_config", None
            ),
        ):
            if isinstance(holder, dict) and holder.get("galacticmodel"):
                cfgs = holder["galacticmodel"]
                break
        if not cfgs:
            return False, None

        first = cfgs[0] if isinstance(cfgs, (list, tuple)) else cfgs
        return True, str((first or {}).get("IMF", "chabrier")).lower()

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
