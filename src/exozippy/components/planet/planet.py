import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.parameterization import mode_manifest
from exozippy.constants import KEPLER_CONST, MSUN_TO_MEARTH, RSUN_TO_REARTH
from exozippy.outputs.prose import get_collector, join_names
from exozippy.outputs.texutils import latex_escape
from exozippy.potentials import soft_lower_bound

from ..orbit.orbit import amplitude_constrained_orbits
from . import physics


def _has_positive_sigma(param):
    """True where a Parameter carries a user-stated Gaussian width.

    `sigma` is None where nobody stated one, a scalar where a broadcast
    entry did, and a per-element vector with NaN holes where one element's
    entry did -- so all three shapes have to answer.
    """
    if param is None or param.sigma is None:
        return False
    arr = np.atleast_1d(np.asarray(param.sigma, dtype=float))
    return bool(np.any(np.nan_to_num(arr, nan=0.0) > 0.0))


logger = logging.getLogger(__name__)


class Planet(Component):
    # The two mass coordinates, as a parameterization mode table (see
    # components/parameterization.py).  `linear` samples planet.mass itself over
    # a range including negatives -- RV and astrometric amplitudes flip phase
    # through zero, so crossing zero is what avoids the Lucy-Sweeney bias on a
    # marginal detection -- while `log_q` samples log10(m_p / m_host) and derives
    # the mass from it.  Planets may differ: log_q is not a parameter of a linear
    # planet at all, and vice versa.
    MASS_MODE_TABLE = {
        "linear": {"mass": None},
        "log_q": {
            "mass": {"expr_key": "default", "force_node": True},
            "log_q": None,
        },
    }

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"
        # BEER (PR 1.b): Doppler beaming amplitude. Per-planet, not
        # per-band (EXOFASTv2 declares it ss.planet[i].beam, unlike
        # thermal/reflect/ellipsoidal which are ss.band[i].*).
        self.fitbeam = [bool(c.get("fitbeam", False)) for c in self.config]
        self.beam_constrains_mass = [
            bool(c.get("beam_constrains_mass", False)) for c in self.config
        ]
        # EXOFASTv2's `alloworbitcrossing`, per planet (review 8.8.9).  Per
        # planet and not per fit because the exception it exists for is a
        # PAIR of resonant or co-orbital bodies, and naming one of them is
        # how a user says which pair they mean; a pair is exempt when either
        # member opts out.
        self.allow_orbit_crossing = [
            bool(c.get("allow_orbit_crossing", False)) for c in self.config
        ]
        # How many MUTUAL Hill radii a pair must be separated by (review
        # 8.8.9).  Per planet for the same reason as the switch above, and
        # a pair takes the LARGER of its two members' values: the number is
        # a statement about how much room that planet needs, so the
        # stricter statement wins.  1.0 is EXOFASTv2's one-Hill pad.
        self.min_hill_separation = [
            float(c.get("min_hill_separation", 1.0)) for c in self.config
        ]

    @property
    def prefix(self):
        return "planet"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "fitbeam",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fit a Doppler beaming amplitude (ppm) for this planet "
                    "directly from the photometry, independent of the RV "
                    "semi-amplitude K -- does not constrain planet mass. "
                    "Default False, which pins beam at 0. If "
                    "beam_constrains_mass is also set, beam_constrains_mass "
                    "wins and beam is derived from K instead of fit freely."
                ),
            },
            {
                "key": "beam_constrains_mass",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Compute this planet's Doppler beaming amplitude from "
                    "the RV semi-amplitude K (Faigler & Mazeh 2011 eq. 1, "
                    "bolometric approximation) instead of fitting it "
                    "freely -- ties the photometric beaming signal to the "
                    "same mass/K driving the RV model. Default False."
                ),
            },
            {
                "key": "allow_orbit_crossing",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Exempt this planet from the orbit-crossing / Hill-"
                    "sphere barrier (EXOFASTv2's alloworbitcrossing). "
                    "Default False. Crossing orbits ARE stable when "
                    "protected by a mean-motion resonance (Neptune-Pluto's "
                    "3:2) and co-orbital pairs live inside their mutual "
                    "Hill sphere, so a fit of such a system has to set "
                    "this; a pair is exempt when either member does."
                ),
            },
            {
                "key": "min_hill_separation",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Minimum separation between this planet's orbit and "
                    "every other planet's, in MUTUAL Hill radii "
                    "R_H = ((m1+m2)/(3 M_*))^(1/3) (a1+a2)/2. Default 1.0, "
                    "EXOFASTv2's one-Hill pad -- a crossing criterion, NOT "
                    "a stability claim. The empirical long-term stability "
                    "threshold for multiplanet systems is ~8-12 mutual "
                    "Hill radii (Chambers, Wetherill & Boss 1996; Pu & Wu "
                    "2015; Obertas, Van Laerhoven & Tamayo 2017), so set "
                    "it there for a real stability prior. A pair takes the "
                    "larger of its two members' values."
                ),
            },
            {
                "key": "chen",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Apply the Chen & Kipping (2017) mass-radius relation "
                    "to this planet. Default: on when exactly one side is "
                    "data-constrained -- transits measure the radius (the "
                    "relation then constrains the mass), RVs/astrometry "
                    "measure the mass (the relation then constrains the "
                    "radius); off when both or neither are fit."
                ),
            },
            {
                "key": "mass_parameterization",
                "kind": "option",
                "accepts": ["linear", "log_q"],
                "required": False,
                "doc": (
                    "Sampled mass coordinate. 'linear' samples planet.mass "
                    "itself, with a negative lower bound so an RV or "
                    "astrometric amplitude can flip sign (avoids the "
                    "Lucy-Sweeney positive-definite bias). 'log_q' samples "
                    "log10(m_planet / m_host) and derives the mass. Default: "
                    "'linear' when RV or astrometric data measure this "
                    "planet's orbit, 'log_q' otherwise (including every "
                    "microlensing lens body, where q <= 0 is meaningless)."
                ),
            },
            {
                "key": "star_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": "Index of the host star (default 0).",
            },
            {
                "key": "orbit_ndx",
                "kind": "ref",
                "accepts": ["orbit"],
                "required": False,
                "doc": "Index of the orbit this planet is on (default 0).",
            },
        ]

    def build_maps(self):
        """Stage 2: Define logical Numpy arrays. (Base class auto-converts to Tensors)."""
        self.star_map = np.array([p.get("star_ndx", 0) for p in self.config])
        self.orbit_map = np.array([p.get("orbit_ndx", 0) for p in self.config])

    def register_parameters(self, system):
        """Stage 3: Auto-estimates and Manifest declaration."""
        has_orbit = "orbit" in system.active_components

        self._resolve_mass_parameterization(system)

        # The mass coordinate, per planet: a `log_q` planet derives its mass
        # from log10(m_p/m_host) and a `linear` one samples the (signed) mass
        # itself, and `log_q` is not a parameter of a linear planet at all.  A
        # system where every planet agrees expands to exactly the manifest this
        # used to write by hand.
        mass_entries = mode_manifest(
            self.mass_parameterizations,
            self.MASS_MODE_TABLE,
            n_elements=self.n_elements,
            where=f"{self.prefix}.mass_parameterization",
        )

        # Insertion order is load-bearing, so it is preserved exactly: graph.py
        # registers the build-order nodes in manifest order, and that order is
        # the order the PyMC nodes -- and so the terms of the summed logp -- get
        # created in.  `mass` first, `log_q` (when any planet uses it) after
        # `m_total`, as the hand-written manifest had them.
        self.manifest = {
            "mass": mass_entries["mass"],
            "radius": None,
            "density": "default",
            "logg": "default",
            "m_total": "default",
        }
        if "log_q" in mass_entries:
            self.manifest["log_q"] = mass_entries["log_q"]

        if has_orbit:
            self.manifest.update(
                {
                    "p": "default",
                    "a": "default",
                    "ar": "default",
                    # force_node: `b` keeps its historical manifest
                    # POSITION (insertion order is the build order) and
                    # gains the pm.Deterministic that `transit.b` used to
                    # be -- the Winn-2010 parity test reads it out of the
                    # model, and a derived Parameter that is not a node is
                    # absent from a posterior point.
                    "b": {"expr_key": "default", "force_node": True},
                    "K": "default",
                    "max_ecc": "default",
                }
            )
            # Transit/occultation durations (review 8.8.7).  Declared
            # whenever there is an orbit, with or without transit data:
            # they are geometry, and a Gaussian on one is how a published
            # duration or eclipse time constrains e and omega.  They used
            # to be bare Deterministics inside transit.py, so an RV-only
            # fit could not state them and a transit fit could not put a
            # prior on them.
            # force_node so each is a pm.Deterministic in the trace, as
            # the transit.py versions were: the phased-plot x-range reads
            # `planet.t14` out of a posterior point, and a derived
            # Parameter that is not a node is absent from one.
            self.manifest.update(
                {
                    name: {"expr_key": "default", "force_node": True}
                    for name in (
                        "bs",
                        "t14",
                        "tfwhm",
                        "tau",
                        "t14s",
                        "tfwhms",
                        "taus",
                    )
                }
            )

        # BEER (PR 1.b): beam is either (a) derived from K for every planet
        # (beam_constrains_mass), (b) free-fit for every planet (fitbeam),
        # or (c) pinned at 0 everywhere (neither set -- transit model
        # unchanged). The two flags are not mutually exclusive: per
        # EXOFASTv2's step2pars.pro (~line 256), beam is computed whenever
        # either is set, and beam_constrains_mass takes priority -- when
        # both are set, beam is still derived from K, not fit freely.
        # Unlike thermal/reflect/ellipsoidal's per-band opt-in, this is a
        # single mode for the whole component: Component.add_parameter
        # resolves one manifest entry as either a whole-component expression
        # ("default") or a whole-component free/fixed tensor (via per-element
        # sigma), never a mix of the two within one parameter -- so a
        # per-planet mix of derived/free/off beam isn't supported yet.
        any_beam_constrains_mass = any(self.beam_constrains_mass)
        any_fitbeam = any(self.fitbeam)
        if any_beam_constrains_mass and not has_orbit:
            raise ValueError(
                f"[{self.prefix}] beam_constrains_mass requires an orbit "
                f"component (beam is derived from K, which requires the "
                f"orbital elements)."
            )
        if (
            len(set(self.fitbeam)) > 1
            or len(set(self.beam_constrains_mass)) > 1
        ):
            logger.warning(
                f"[{self.prefix}] fitbeam/beam_constrains_mass differ "
                f"across planets (fitbeam={self.fitbeam}, "
                f"beam_constrains_mass={self.beam_constrains_mass}); beam "
                f"is a whole-component mode, not per-planet (see the "
                f"comment above), so the resolved mode applies to every "
                f"planet -- derived-from-K if any planet set "
                f"beam_constrains_mass, else free-fit if any set "
                f"fitbeam, else pinned at 0."
            )
        if any_beam_constrains_mass:
            self.manifest["beam"] = "default"
        elif any_fitbeam:
            off = [i for i in range(self.n_elements) if not self.fitbeam[i]]
            entry = {}
            if off:
                pin = np.full(self.n_elements, np.nan)
                pin[off] = 0.0
                entry["overrides"] = {"sigma": pin.tolist()}
            self.manifest["beam"] = entry
        # Neither flag set anywhere: beam does not enter the manifest at
        # all (no parameter, no table row), matching Band's opt-in gating
        # for thermal/reflect/ellipsoidal.  Consumers guard on
        # `"beam" in planets.manifest`.

        # Data-driven estimate: Initialize 'K' directly from the RV data variance
        rv_comps = [
            c
            for c in system.active_components.values()
            if hasattr(c, "k_init")
        ]
        if rv_comps and has_orbit:
            k_ms_guess = rv_comps[0].k_init / np.sqrt(self.n_elements)
            for i in range(self.n_elements):
                self.config_manager.add_hint(f"planet.{i}.K", k_ms_guess)

        self._resolve_chen(system)

    def _resolve_mass_parameterization(self, system):
        """Pick the sampled mass coordinate (stage 3, topology known).

        'linear' samples planet.mass directly over a range that includes
        negative values: RV and astrometric amplitudes flip phase through
        zero, so letting the mass cross zero is what avoids the Lucy-Sweeney
        positive-definite bias at a marginal detection.

        Everywhere else a negative mass is meaningless and actively harmful:
        microlensing clips q to [1e-9, 100] inside the magnification model
        (so the negative half is likelihood-flat) and Chen & Kipping clips at
        1e-10 Mearth (so the negative half is a flat plateau with no
        restoring force).  Those planets sample log_q = log10(m_p / m_host)
        instead, which also turns the microlensing constraint into a shear of
        the sampled pair (star.logmass, planet.log_q) rather than a diagonal
        ridge -- worth a measured ~4.5x on the whitened 2x2 curvature block of
        examples/DC2018_128, not the order of magnitude one might expect (see
        CLAUDE.md).  Removing the unreachable-but-samplable negative region is
        the larger win.

        Default: 'linear' iff the planet is mass-constrained by RV or
        astrometry AND is not a microlensing lens body; 'log_q' otherwise.  A
        per-planet ``mass_parameterization:`` key overrides.
        """
        mass_sides = self._mass_constrained(system)

        lens = system.active_components.get("lens")
        lens_bodies = set(lens.lens_bodies[0]) if lens is not None else set()

        modes, reasons = [], []
        for p, (c, nm) in enumerate(zip(self.config, self.names)):
            is_lens_body = ("planet", p) in lens_bodies
            user = c.get("mass_parameterization")
            if user is not None:
                if user not in ("linear", "log_q"):
                    raise ValueError(
                        f"planet '{nm}': 'mass_parameterization:' must be "
                        f"'linear' or 'log_q', got {user!r}."
                    )
                modes.append(user)
                reasons.append("set on the planet")
                continue

            # A lens body outranks the RV/astrometry signal: microlensing
            # hard-forbids q <= 0, so a signed mass there is unusable no
            # matter what else measures the orbit.
            if is_lens_body:
                modes.append("log_q")
                reasons.append("it is a microlensing lens body")
            elif mass_sides[p]:
                modes.append("linear")
                reasons.append("RV/astrometry data measure its orbit")
            else:
                modes.append("log_q")
                reasons.append("no signed observable constrains its mass")

        # Per planet, and mixing is fine: roles are per element (see the
        # manifest vocabulary), so one planet can derive its mass from log_q
        # while another samples a signed linear mass.  This used to be a hard
        # error for an explicit disagreement and a silent fall back to
        # all-linear for an implicit one, because Parameter.build_pymc derived a
        # whole vector or none of it.
        detail = ", ".join(
            f"'{nm}' -> {m} ({r})"
            for nm, m, r in zip(self.names, modes, reasons)
        )
        if any(m == "log_q" for m in modes):
            logger.info(
                f"sampling log10(m_planet/m_host) where it applies: {detail} "
                "(override per planet with 'mass_parameterization: linear')."
            )

        self.mass_parameterizations = modes
        # The whole-component answer, kept for the readers that only need to
        # know whether ANY planet uses the ratio coordinate; per-planet callers
        # read `mass_parameterizations`.
        self.mass_parameterization = (
            "log_q" if modes and all(m == "log_q" for m in modes) else "linear"
        )
        self._reconcile_mass_user_params()

    def _reconcile_mass_user_params(self):
        """Move a user's ``sigma: 0`` from planet.mass onto log_q, and reject
        a stale log_q entry in linear mode.

        In log_q mode planet.mass is derived, and sigma=0 on a derived
        parameter is a no-op that only emits a warning -- so a user pinning a
        planet's mass would silently get a free one.  Fixing the sampled
        coordinate instead is exactly equivalent.  Bounds are deliberately
        NOT translated: they are absolute masses and the host mass is not
        known here, and a derived parameter's bounds already become soft
        barriers on the physical value, which is the semantics we want.
        """
        up = self.config_manager.user_params
        for i in range(self.n_elements):
            mass_key, log_q_key = f"planet.{i}.mass", f"planet.{i}.log_q"

            # Per planet, because the coordinate is: a stale log_q entry is
            # only stale for the planets that sample a linear mass.
            if self.mass_parameterizations[i] == "linear":
                if log_q_key in up:
                    raise ValueError(
                        f"'{log_q_key}' is set but planet '{self.names[i]}' "
                        "samples a linear mass. This usually means a params "
                        "file written by mkparam for a log_q fit is being "
                        "reused after the data topology changed. Remove the "
                        "entry, or set 'mass_parameterization: log_q' on "
                        "that planet."
                    )
                continue

            entry = up.get(mass_key)
            if not isinstance(entry, dict):
                continue

            # A negative mass is unrepresentable as log_q.  Catch it here
            # rather than letting the relaxation engine quietly fall back to
            # the defaults.yaml start and derive a positive mass instead.
            initval = entry.get("initval")
            if initval is not None and np.min(np.atleast_1d(initval)) <= 0.0:
                raise ValueError(
                    f"'{mass_key}' initval {initval} is not positive, but "
                    f"planet '{self.names[i]}' samples log10(m_p/m_host), "
                    "which cannot represent it. A negative mass only makes "
                    "sense where an RV or astrometric amplitude can flip "
                    "sign; set 'mass_parameterization: linear' on every "
                    "planet if that is what you want."
                )

            if entry.get("sigma") is None:
                continue
            # sigma > 0 is a real prior and needs no help: a derived
            # parameter gets its Gaussian potential on the physical value.
            if float(entry["sigma"]) != 0.0:
                continue

            entry.pop("sigma")
            log_entry = up.setdefault(log_q_key, {})
            log_entry["sigma"] = 0.0
            logger.info(
                f"'{mass_key}' sigma=0: holding the planet mass fixed by "
                f"fixing the sampled '{log_q_key}' instead."
            )

    def _mass_constrained(self, system):
        """Per planet: does RV or astrometric data measure its orbit?

        These are the signed observables -- an RV or astrometric amplitude
        flips phase through zero -- so this is both the Chen mass-side
        predicate and the set of planets for which a signed linear mass is
        meaningful.  Cached: both callers run in stage 3.
        """
        if getattr(self, "_mass_side", None) is not None:
            return self._mass_side

        orbit = system.active_components.get("orbit")

        # Orbits whose motion is measured by RV or astrometric data.  A
        # planet is mass-constrained if it is a body of one of them.  Asked of
        # the ORBIT component, which owns the predicate: the same question
        # decides whether an orbit is transit-only, and the two answers must
        # not drift apart (Orbit.amplitude_constrained_orbits).
        mass_orbits = (
            amplitude_constrained_orbits(system, orbit)
            if orbit is not None
            else set()
        )

        self._mass_side = [
            orbit is not None
            and any(("planet", p) in orbit.bodies(o) for o in mass_orbits)
            for p in range(self.n_elements)
        ]
        return self._mass_side

    def _resolve_chen(self, system):
        """Decide, per planet, whether the Chen & Kipping (2017) mass-radius
        relation applies (stage 3: every component's constructor and
        load_data have run, so the data topology is known).

        Default (EXOFASTv2 mkss.pro's ``chen = fittran xor fitrv``, extended
        to astrometry): on when exactly one side of the relation is
        data-constrained.  Transits measure the radius, so the relation
        constrains the mass; RVs and astrometry measure the mass, so the
        relation constrains the radius.  With both or neither fit it is off
        by default.  A per-planet ``chen: true/false`` config key overrides.
        """
        # Every transit light curve models every planet, so any transit
        # data set puts every planet on the radius-constrained side.
        radius_side = "transit" in system.active_components
        mass_sides = self._mass_constrained(system)

        self.chen = []
        # Per planet, the quantities the relation (not data) determines --
        # drives the "Derived from \citet{Chen:2017}" table note.
        self.chen_derives = []
        for p, (c, nm) in enumerate(zip(self.config, self.names)):
            mass_side = mass_sides[p]
            user = c.get("chen")
            if user is not None:
                if not isinstance(user, bool):
                    raise ValueError(
                        f"planet '{nm}': 'chen:' must be true or false, "
                        f"got {user!r}."
                    )
                self.chen.append(user)
            else:
                self.chen.append(radius_side != mass_side)
                if self.chen[-1]:
                    constrained = (
                        "mass from radius"
                        if radius_side
                        else "radius from mass"
                    )
                    logger.info(
                        f"planet '{nm}': applying the Chen & Kipping (2017) "
                        f"mass-radius relation to constrain the {constrained} "
                        f"(override with 'chen: false' on the planet)."
                    )
            derives = set()
            if self.chen[-1]:
                if not mass_side:
                    derives.add("mass")
                if not radius_side:
                    derives.add("radius")
            self.chen_derives.append(derives)

    def build_likelihood(self, model, system):
        # Soft barriers via the shared clipped log-sigmoid helpers.
        # scale * softness = 0.0088 reproduces the historical steepness of
        # 500 nats/unit.  The raw pm.math.log(pt.sigmoid(z)) form used here
        # previously NaNs in the JAX gradient once z > ~709 (exp(z) overflows
        # inside an unselected jnp.where branch of pytensor's softplus): any
        # system with m_total > 1.42 Msun silently froze every numpyro chain
        # at its starting point.  See potentials.py.
        #
        # The barrier is applied to the UNCLIPPED sum, not to self.m_total:
        # calc_m_total clips at 1e-9, so feeding the clipped node here made
        # the whole m_total < 0 region a constant log(sigmoid(~0)) = -0.693
        # with exactly zero gradient -- a flat plateau with no restoring
        # force, which is precisely the pathology the log_q notes warn about
        # and is reachable in 'linear' mass mode (planet.mass's lower bound
        # is -1000 Mjup, more than a solar mass below zero).
        star = system.active_components["star"]
        m_total_unclipped = star.mass.value[self.star_map] + self.mass.value
        pm.Potential(
            f"{self.prefix}.m_pos_constraint",
            soft_lower_bound(m_total_unclipped, 0.0, scale=0.88),
        )

        self._add_chen_potential()
        self._annotate_chen_table_notes(system)
        # Modeling-draft prose, declared next to the potential it describes
        # (outputs/prose.py's declare-at-site rule).
        enabled = [nm for nm, on in zip(self.names, self.chen) if on]
        if enabled:
            noun = "planet" if len(enabled) == 1 else "planets"
            get_collector(system).add(
                r"We imposed the \citet{Chen:2017} probabilistic "
                rf"mass--radius relation on {noun} "
                + join_names(latex_escape(n) for n in enabled)
                + ", constraining whichever of the mass and radius the "
                "data do not.",
                section="planetary",
                key=f"{self.prefix}.chen",
            )

        if "orbit" not in system.active_components:
            return

        orbits = system.orbit
        self._add_duration_prose(system, orbits)
        # The eccentricity barrier used to live here, on
        # orbits.ecc.value[self.orbit_map] -- i.e. on the node calc_ecc
        # clips at 0.9999, which gave it zero gradient over 21.5% of the
        # sampled (secosw, sesinw) square, and gave a planet-free orbit no
        # bound at all.  It now lives in Orbit._add_eccentricity_bound,
        # which bounds the unclipped sum and folds self.max_ecc in as the
        # per-orbit threshold.  Exactly one potential per orbit; do not add
        # a second one here.

        if self.n_elements >= 2:
            self._add_crossing_potential(system, orbits)

    def _initial_semimajor_axes(self):
        """Per-planet starting semi-major axis, solRad, or NaN.

        The relaxation engine resolves `planet.a` (Kepler's third law and the
        m_total sum are relations in planet/symbolic_physics.py), so this reads
        the start it solved.  It used to RECOMPUTE it -- a hand copy of
        `KEPLER_CONST m_total^(1/3) P^(2/3)` living in this file, assembled from
        the period, the planet mass and the host mass -- because nothing
        resolved `a` at all; that gap is what the relations closed, and closing
        it deleted the copy.  A relation cannot drift from `physics.calc_arsun`
        the way a second implementation can.

        NaN where the engine could not solve it (an incomplete harness system);
        the caller already sorts those last and refuses them as a barrier
        scale.
        """
        n = self.n_elements
        try:
            out = np.asarray(
                np.atleast_1d(self.a.initval), dtype=float
            ) * np.ones(n)
        except (TypeError, ValueError):
            return np.full(n, np.nan)
        return out

    def _add_duration_prose(self, system, orbits):
        """Modeling-draft sentences for the durations and eclipse time.

        Declared here because this is where they become part of the model:
        the durations and `orbit.ts` are derived Parameters, so a user's
        `mu`/`sigma` on one is a Gaussian in the logp and the gradient
        reaches the eccentricity vector through the expression -- there is no
        separate potential to hang the sentence off, which is precisely why
        the "declare it next to the pm.Potential" rule needs reading as
        "declare it where the feature is".
        """
        prose = get_collector(system)
        prose.add(
            r"We report the transit and occultation durations "
            r"($T_{14}$, $T_{\rm FWHM}$, $\tau$ and their eclipse "
            r"counterparts) and the time of secondary eclipse computed "
            r"from the orbital elements following \citet{Winn:2010}.",
            section="orbits",
            key=f"{self.prefix}.durations",
        )

        constrained = [
            name
            for name, param in (
                ("transit duration", getattr(self, "t14", None)),
                ("FWHM transit duration", getattr(self, "tfwhm", None)),
                ("ingress/egress duration", getattr(self, "tau", None)),
                ("occultation duration", getattr(self, "t14s", None)),
                ("FWHM occultation duration", getattr(self, "tfwhms", None)),
                ("time of secondary eclipse", getattr(orbits, "ts", None)),
            )
            if _has_positive_sigma(param)
        ]
        if not constrained:
            return
        prose.add(
            "We applied Gaussian priors to the "
            + join_names(constrained)
            + ", which constrain the eccentricity and argument of "
            r"periastron: the phase of the secondary eclipse measures "
            r"$e\cos{\omega_*}$ and the ratio of the occultation to the "
            r"transit duration measures $e\sin{\omega_*}$ "
            r"\citep{Winn:2010,Charbonneau:2005}.",
            section="orbits",
            key=f"{self.prefix}.duration_constraints",
        )

    def _add_crossing_potential(self, system, orbits):
        """Soft orbit-crossing / Hill-sphere barrier between planet pairs.

        Descended from EXOFASTv2's `exofast_chi2v2.pro`:445-468 (review
        8.8.9).  Two planets whose orbits approach within a Hill radius are
        unstable on timescales far shorter than the age of any system we
        fit, so the posterior is bounded by requiring the radial gap between
        the inner planet's apastron and the outer's periastron to be at
        least `k` MUTUAL Hill radii:

            a_out (1 - e_out) - a_in (1 + e_in)  >=  k R_H,
            R_H = ((m_in + m_out) / 3 M_*)^(1/3) (a_in + a_out) / 2

        `k` is `min_hill_separation`, per planet, defaulting to **1.0** --
        EXOFASTv2's one-Hill pad, so the formulation below differs from the
        port while the default STRICTNESS does not.  A pair takes the larger
        of its two members' `k`.

        What the earlier versions had: first the bare apsidal test with no
        Hill pad, on ADJACENT pairs only (before that, a never-executed
        `-inf` block that walked attributes deleted in the vectorized
        refactor, so any two-planet system raised AttributeError); then the
        faithful port, which pads each side by that planet's OWN Hill radius
        `h_i = (1 - e_i) a_i (m_i / 3 M_*)^(1/3)`.

        Five things about it, all deliberate:

        - **Soft, not `+infinity` chi2.**  EXOFASTv2 rejects outright; a
          gradient sampler cannot be given a wall with no gradient to follow
          out of (and the JAX backward pass NaNs on one).  It is the same
          clipped log-sigmoid barrier as every other soft constraint here,
          with a transition width of 1% of the inner planet's STARTING
          semi-major axis, so the barrier is scaled to the orbit it guards.
        - **All pairs**, as EXOFASTv2 does.  Adjacent-only was an
          approximation with nothing to recommend it: three planets can have
          a widely eccentric outer one crossing the innermost while missing
          its neighbour.
        - **Planets are ordered ONCE**, by their starting semi-major axes.
          The ordering is a topology choice, not a sampled quantity:
          re-deriving it per draw would make the potential discontinuous
          wherever two orbits swap places.
        - **The Hill radius is the MUTUAL one, not each planet's own.**  It
          is the length the dynamics literature states separations in, so
          `min_hill_separation` is a number a user can look up rather than
          one only this file's algebra defines; it is symmetric in the pair
          by construction; and it depends on the two masses only through
          their SUM, so a marginal secondary mass no longer moves the
          threshold on its own.  Everything it needs is a node that already
          exists.

          How far this moves the boundary, since "the default strictness
          does not change" is true only of the number `k`, not of the pad
          it buys.  For an equal-mass pair the mutual pad at `k = 1` is
          `2^(1/3)/2 = 0.630` of the summed per-planet pads at `e = 0`, so
          the barrier admits 37% closer packing there.  But the old pads
          carried a `(1 - e_i)` factor and the mutual radius carries none,
          so the ratio is `k 2^(1/3) / [(1-e_i) + (1-e_j)]` and it passes
          through 1 at `e = 1 - 2^(1/3)/2 = 0.370`: BELOW that the mutual
          form is looser, ABOVE it stricter (26% stricter at `e = 0.5`).
          No single `k` reproduces the old boundary, because the ratio
          depends on both eccentricities and, through the cube root, on
          the mass ratio.

          And the shift is visible far inside the forbidden zone, not just
          at the boundary, which the hard-rejection original never was.
          `log sigmoid(x) -> x` for `x << 0`, so the penalty is LINEAR in
          the deficit with no clip on that side (`_MAX_ARG` bounds only the
          allowed side).  Changing the threshold therefore shifts logp by a
          rigid `4.4 / (0.01 a_in) * Delta_pad`, the same offset whether the
          pair is marginally crossing or completely interpenetrating --
          about 23 nats for a Jupiter-mass pair, 32 for 3 M_J.  On the
          allowed side both forms saturate to ~0 within `0.01 a_in` of
          their own threshold, so a comfortably separated pair sees no
          difference at all.  `M_*` is the INNER planet's host: for the ordinary case
          of two planets round one star that is simply the star, and where
          the hosts differ the criterion is not a stability criterion anyway
          (see the nested-orbit limitation below).
        - **1 R_H is a crossing criterion, NOT a stability claim, and the
          default is deliberately permissive.**  The criterion is crude in
          both directions and EXOFASTv2's own comment says so.  *Too
          strict*: **resonant protection makes crossing orbits stable** --
          Neptune and Pluto cross in radius and are protected by their 3:2
          mean-motion resonance, and co-orbital / Trojan pairs live INSIDE
          their mutual Hill sphere, i.e. at a separation of ZERO by this
          measure.  Mutual inclination, which also separates orbits that
          overlap in radius, is not modeled either.  A fit of such a system
          must set `allow_orbit_crossing: true` on a member of the pair (or
          lower `min_hill_separation`) -- otherwise the barrier quietly
          penalizes the very configuration being measured.  *Too loose*: the
          empirical long-term threshold for multiplanet systems is ~8-12
          mutual Hill radii (Chambers, Wetherill & Boss 1996; Pu & Wu 2015;
          Obertas, Van Laerhoven & Tamayo 2017), so a user who wants a real
          stability prior sets `min_hill_separation` in that range.  Real
          stability is an N-body / SPOCK-style analysis, a follow-up rather
          than a per-step penalty.

        Limitation, unchanged and deliberate: only pairs on distinct orbits
        are constrained.  For nested hierarchical orbits (a planet orbiting
        a body group) this compares semi-major axes about different centers,
        which is the right first-order condition but not a stability
        criterion.
        """
        # Semi-major axis (solRad) and eccentricity, per planet.
        a = self.a.value
        ecc = orbits.ecc.value[self.orbit_map]
        star = system.active_components["star"]
        m_star = star.mass.value[self.star_map]

        inner_edge = (1.0 - ecc) * a
        outer_edge = (1.0 + ecc) * a

        a_init = self._initial_semimajor_axes()
        # A missing start sorts last but must not become the barrier scale.
        order = np.argsort(np.where(np.isfinite(a_init), a_init, np.inf))

        for k, i_raw in enumerate(order[:-1]):
            i = int(i_raw)
            for j_raw in order[k + 1 :]:
                j = int(j_raw)
                if self.orbit_map[i] == self.orbit_map[j]:
                    # One orbit cannot cross itself; two planets sharing an
                    # orbit index are co-orbital by construction.
                    continue
                if (
                    self.allow_orbit_crossing[i]
                    or self.allow_orbit_crossing[j]
                ):
                    logger.info(
                        "[%s] orbit-crossing barrier skipped for the "
                        "%s/%s pair (allow_orbit_crossing).",
                        self.prefix,
                        self.names[i],
                        self.names[j],
                    )
                    continue

                scale = a_init[i] if np.isfinite(a_init[i]) else 0.0
                scale = float(max(abs(scale), 1e-6))

                # Mutual Hill radius of the pair.  The cube root's ARGUMENT
                # is floored, never its result: `planet.mass` may be
                # negative in `linear` mass mode, `x**(1/3)` is NaN there,
                # and flooring after the root would multiply its infinite
                # derivative at zero by pt.maximum's zero gradient and give
                # NaN again -- the calc_theta_E / calc_jitter idiom.  1e-30
                # leaves R_H at ~1e-10 a, i.e. no pad at all, which is the
                # right answer for a pair the data say has no mass.
                mass_ratio = pt.maximum(
                    (self.mass.value[i] + self.mass.value[j])
                    / (3.0 * m_star[i]),
                    1e-30,
                )
                r_hill = mass_ratio ** (1.0 / 3.0) * 0.5 * (a[i] + a[j])
                k = max(
                    self.min_hill_separation[i], self.min_hill_separation[j]
                )

                pm.Potential(
                    f"{self.prefix}.crossing_bound_"
                    f"{self.names[i]}_{self.names[j]}",
                    soft_lower_bound(
                        inner_edge[j] - outer_edge[i] - k * r_hill,
                        0.0,
                        scale=scale,
                    ),
                )

        self._add_crossing_prose(system, order)

    def _add_crossing_prose(self, system, order):
        """The modeling-draft sentence for the barrier above.

        Declared next to the `pm.Potential` it describes, the
        outputs/prose.py rule.
        """
        exempt = [
            nm for nm, off in zip(self.names, self.allow_orbit_crossing) if off
        ]
        constrained = [
            self.names[int(i)]
            for i in order
            if not self.allow_orbit_crossing[int(i)]
        ]
        if len(constrained) < 2:
            return
        seps = sorted(
            {
                self.min_hill_separation[int(i)]
                for i in order
                if not self.allow_orbit_crossing[int(i)]
            }
        )
        # A config fact, so it may be interpolated (outputs/prose.py).  One
        # number when every planet agrees, a range when they do not -- the
        # pair threshold is the larger of its two members'.
        if len(seps) == 1:
            sep_txt = f"{seps[0]:g}"
            unit = "radius" if seps[0] == 1.0 else "radii"
        else:
            sep_txt = f"{seps[0]:g}--{seps[-1]:g}"
            unit = "radii"
        prose = get_collector(system)
        text = (
            "We required the orbits of "
            + join_names(latex_escape(n) for n in constrained)
            + f" to stay separated by at least {sep_txt} mutual Hill "
            + unit
            + r", $R_{\rm H} = [(m_1 + m_2)/3 M_\star]^{1/3} "
            r"(a_1 + a_2)/2$, as a soft penalty rather than the hard "
            r"rejection of \citet{Eastman:2019}; the criterion neglects "
            "mutual inclination and resonant protection, and one mutual "
            "Hill radius is a crossing criterion rather than a long-term "
            "stability requirement."
        )
        if exempt:
            noun = "Planet" if len(exempt) == 1 else "Planets"
            text += (
                f"  {noun} "
                + join_names(latex_escape(n) for n in exempt)
                + " were exempted from it."
            )
        prose.add(text, section="orbits", key=f"{self.prefix}.orbit_crossing")

    def _add_chen_potential(self):
        """Gaussian tie between each chen-enabled planet's mass and radius.

        Ported from EXOFASTv2 (exofast_chi2v2.pro): the relation predicts
        the radius from the mass and the penalty is Gaussian in linear
        Earth radii, so whichever side the data pin, the other follows.
        The mass is clipped at 1e-10 Mearth exactly as EXOFASTv2 does --
        planet.mass's lower bound is negative (to assess RV significance)
        and a non-integer power of a negative mass is NaN.  Unlike
        EXOFASTv2's chi2-only accumulation, the -log(sigma) normalization
        is kept: sigma scales with the predicted radius, so it is not a
        constant (same reasoning as components/mann).
        """
        mask = np.array(self.chen, dtype=bool)
        if not mask.any():
            return

        mpearth = pt.maximum(self.mass.value * MSUN_TO_MEARTH, 1e-10)
        rpearth = self.radius.value * RSUN_TO_REARTH

        pred = physics.calc_chen_radius(mpearth)
        sigma = physics.calc_chen_radius_sigma(mpearth)

        # Reported in internal units (solRad), comparable to planet.radius.
        pm.Deterministic(
            f"{self.prefix}.chen_radius_pred", pred / RSUN_TO_REARTH
        )

        logp = -0.5 * pt.sqr((rpearth - pred) / sigma) - pt.log(sigma)
        pm.Potential(
            f"{self.prefix}.chen_prior",
            pt.sum(pt.where(pt.as_tensor_variable(mask), logp, 0.0)),
        )

    CHEN_TABLE_NOTE = r"Derived from \citet{Chen:2017}"

    def _annotate_chen_table_notes(self, system):
        """Mark every parameter the Chen & Kipping relation flows into.

        When the relation, rather than data, determines a planet's mass
        (transit-only fit) or radius (RV/astrometry-only fit), that quantity
        and everything computed from it belong to the relation.  Dependence
        is read off the built tensor graph: any Parameter whose value node
        has planet.mass/planet.radius among its ancestors gets the
        "Derived from \\citet{Chen:2017}" table note (stage 7, so every
        component's nodes exist).  Granularity is per parameter, not per
        element -- one chen-enabled planet marks the whole row.
        """
        from pytensor.graph.traversal import ancestors

        from ..parameter import Parameter

        derived = set()
        for on, d in zip(self.chen, self.chen_derives):
            if on:
                derived |= d
        if not derived:
            return

        seeds = []
        if "mass" in derived:
            seeds.append(self.mass.value)
        if "radius" in derived:
            seeds.append(self.radius.value)
        seed_ids = {id(s) for s in seeds}

        for comp in system.get_all_components():
            for p in comp.__dict__.values():
                if not isinstance(p, Parameter):
                    continue
                node = getattr(p, "value", None)
                if node is None:
                    continue
                node_ids = {id(a) for a in ancestors([node])} | {id(node)}
                if not (node_ids & seed_ids):
                    continue
                if p.table_note is None:
                    p.table_note = self.CHEN_TABLE_NOTE
                elif self.CHEN_TABLE_NOTE not in p.table_note:
                    p.table_note = p.table_note + "; " + self.CHEN_TABLE_NOTE

    def plot(self, system, points, filename_prefix="debug"):
        pass
