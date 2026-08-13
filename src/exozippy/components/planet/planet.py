import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.constants import KEPLER_CONST, MSUN_TO_MEARTH, RSUN_TO_REARTH
from exozippy.potentials import soft_lower_bound

from . import physics

logger = logging.getLogger(__name__)


class Planet(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"
        # BEER (PR 1.b): Doppler beaming amplitude. Per-planet, not
        # per-band (EXOFASTv2 declares it ss.planet[i].beam, unlike
        # thermal/reflect/ellipsoidal which are ss.band[i].*).
        self.beam_free = [bool(c.get("beam_free", False)) for c in self.config]
        self.beam_constrains_mass = [
            bool(c.get("beam_constrains_mass", False)) for c in self.config
        ]

    @property
    def prefix(self):
        return "planet"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "beam_free",
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
        """Stage 1b: Define logical Numpy arrays. (Base class auto-converts to Tensors)."""
        self.star_map = np.array([p.get("star_ndx", 0) for p in self.config])
        self.orbit_map = np.array([p.get("orbit_ndx", 0) for p in self.config])

    def register_parameters(self, system):
        """Stage 2: Auto-estimates and Manifest declaration."""
        has_orbit = "orbit" in system.active_components

        self._resolve_mass_parameterization(system)

        if self.mass_parameterization == "log_q":
            mass_entry = {"expr_key": "default", "force_node": True}
        else:
            mass_entry = None

        self.manifest = {
            "mass": mass_entry,
            "radius": None,
            "density": "default",
            "logg": "default",
            "m_total": "default",
        }

        if self.mass_parameterization == "log_q":
            self.manifest["log_q"] = None

        if has_orbit:
            self.manifest.update(
                {
                    "p": "default",
                    "arsun": "default",
                    "ar": "default",
                    "b": "default",
                    "K": "default",
                    "max_ecc": "default",
                }
            )

        # BEER (PR 1.b): beam is either (a) derived from K for every planet
        # (beam_constrains_mass), (b) free-fit for every planet (beam_free),
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
        any_beam_free = any(self.beam_free)
        if any_beam_constrains_mass and not has_orbit:
            raise ValueError(
                f"[{self.prefix}] beam_constrains_mass requires an orbit "
                f"component (beam is derived from K, which requires the "
                f"orbital elements)."
            )
        if (
            len(set(self.beam_free)) > 1
            or len(set(self.beam_constrains_mass)) > 1
        ):
            logger.warning(
                f"[{self.prefix}] beam_free/beam_constrains_mass differ "
                f"across planets (beam_free={self.beam_free}, "
                f"beam_constrains_mass={self.beam_constrains_mass}); beam "
                f"is a whole-component mode, not per-planet (see the "
                f"comment above), so the resolved mode applies to every "
                f"planet -- derived-from-K if any planet set "
                f"beam_constrains_mass, else free-fit if any set "
                f"beam_free, else pinned at 0."
            )
        if any_beam_constrains_mass:
            self.manifest["beam"] = "default"
        elif any_beam_free:
            off = [i for i in range(self.n_elements) if not self.beam_free[i]]
            entry = {}
            if off:
                pin = np.full(self.n_elements, np.nan)
                pin[off] = 0.0
                entry["overrides"] = {"sigma": pin.tolist()}
            self.manifest["beam"] = entry
        else:
            self.manifest["beam"] = {
                "overrides": {"sigma": [0.0] * self.n_elements}
            }

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
        """Pick the sampled mass coordinate (stage 2, topology known).

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

        # One mode per component: Parameter.build_pymc derives a whole vector
        # or none of it (is_derived is set from `expression is not None`), so
        # a mixed system cannot be built.  Explicit overrides that disagree
        # are a user error; a mixed *default* silently falls back to linear,
        # which is the historical behavior.
        detail = ", ".join(
            f"'{nm}' -> {m} ({r})"
            for nm, m, r in zip(self.names, modes, reasons)
        )
        if len(set(modes)) > 1:
            explicit = any(
                c.get("mass_parameterization") is not None for c in self.config
            )
            if explicit:
                raise ValueError(
                    "All planets must share one 'mass_parameterization' "
                    f"(got {detail}). Mixing linear and log_q planets in one "
                    "system is not yet supported; set the same value on "
                    "every planet."
                )
            logger.info(
                f"planets disagree on the mass coordinate ({detail}); "
                "falling back to 'linear' for all of them, since mixing is "
                "not yet supported. Set 'mass_parameterization: log_q' on "
                "every planet to sample the mass ratio instead."
            )
            modes = ["linear"] * len(modes)
        elif modes and modes[0] == "log_q":
            logger.info(
                f"sampling log10(m_planet/m_host): {detail} "
                "(override with 'mass_parameterization: linear')."
            )

        self.mass_parameterization = modes[0] if modes else "linear"
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

            if self.mass_parameterization == "linear":
                if log_q_key in up:
                    raise ValueError(
                        f"'{log_q_key}' is set but planet '{self.names[i]}' "
                        "samples a linear mass. This usually means a params "
                        "file written by mkparam for a log_q fit is being "
                        "reused after the data topology changed. Remove the "
                        "entry, or set 'mass_parameterization: log_q' on "
                        "every planet."
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
        meaningful.  Cached: both callers run in stage 2.
        """
        if getattr(self, "_mass_side", None) is not None:
            return self._mass_side

        orbit = system.active_components.get("orbit")

        # Orbits whose motion is measured by RV or astrometric data.  A
        # planet is mass-constrained if it is a body of one of them.
        mass_orbits = set()
        rv = system.active_components.get("rvinstrument")
        if rv is not None and orbit is not None:
            for s in set(rv.star_ndx):
                mass_orbits.update(o for o, _ in orbit.star_membership(s))
        ast = system.active_components.get("astrometryinstrument")
        if ast is not None and orbit is not None:
            for i, mode in enumerate(ast.modes):
                if mode == "rel":
                    if ast.rel_orbit[i] is not None:
                        mass_orbits.add(ast.rel_orbit[i])
                else:
                    # gaia/abs photocenter wobble sums the orbits whose
                    # primary group contains the target star.
                    s = int(ast.config[i].get("star_ndx", 0))
                    mass_orbits.update(
                        o
                        for o, role in orbit.star_membership(s)
                        if role == "primary"
                    )

        self._mass_side = [
            orbit is not None
            and any(("planet", p) in orbit.bodies(o) for o in mass_orbits)
            for p in range(self.n_elements)
        ]
        return self._mass_side

    def _resolve_chen(self, system):
        """Decide, per planet, whether the Chen & Kipping (2017) mass-radius
        relation applies (stage 2: every component's constructor and
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

        if "orbit" not in system.active_components:
            return

        orbits = system.orbit
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

    def _initial_semimajor_axes(self, system, orbits):
        """Per-planet starting semi-major axis, solRad.

        ``planet.arsun`` is a derived Parameter and carries no ``initval``
        of its own, so the start is recomputed here from the same relation
        (``physics.calc_arsun``) using the start values the relaxation
        engine did resolve.  Returns NaN wherever an input is missing; the
        caller falls back to the config order there.
        """

        def vec(x, n):
            try:
                out = np.asarray(np.atleast_1d(x), dtype=float) * np.ones(n)
            except (TypeError, ValueError):
                return np.full(n, np.nan)
            return out

        n = self.n_elements
        period = vec(orbits.period.initval, orbits.n_elements)[self.orbit_map]
        m_planet = vec(self.mass.initval, n)
        star = system.active_components["star"]
        m_star = vec(star.mass.initval, star.n_elements)[self.star_map]

        m_total = np.maximum(m_star + m_planet, 1e-9)
        with np.errstate(invalid="ignore"):
            return (
                KEPLER_CONST * m_total ** (1.0 / 3.0) * period ** (2.0 / 3.0)
            )

    def _add_crossing_potential(self, system, orbits):
        """Soft non-crossing barrier between neighboring planets' orbits.

        Two planets whose orbits intersect are dynamically unstable on
        timescales far shorter than the age of any system we fit, so the
        posterior is bounded by keeping the outer planet's periastron,
        a_out (1 - e_out), outside the inner planet's apastron,
        a_in (1 + e_in).  This is the constraint the original (never
        executed) block described; three things about it are new:

        - It reads the component's own vectors.  The old code walked a
          ``self.planets`` list of per-planet objects with ``.orbit.a`` /
          ``.orbit.a_val`` attributes, none of which have existed since the
          vectorized refactor -- so ANY system with two or more planets
          raised AttributeError here.  A planet's semi-major axis is
          ``planet.arsun`` (solRad, derived from m_total and the orbit's
          period) and its eccentricity is its orbit's, via ``orbit_map``.
        - The wall is a soft bound, not ``pt.switch(..., 0, -inf)``.  A -inf
          gives NUTS no gradient to follow out of the forbidden region (and
          NaNs the JAX backward pass), so this uses the same clipped
          log-sigmoid barrier as every other soft constraint here.  The
          transition width is 1% of the inner planet's starting semi-major
          axis, i.e. the barrier is scaled to the orbit it guards.
        - Planets are ordered ONCE, by their starting semi-major axes.  The
          ordering is a topology choice, not a sampled quantity: re-deriving
          it per draw would make the potential discontinuous wherever two
          orbits swap places.

        Limitation, deliberately not modeled: only adjacent pairs are
        constrained, and only pairs on distinct orbits.  For nested
        hierarchical orbits (a planet orbiting a body group) this compares
        semi-major axes about different centers, which is the right
        first-order condition but not a stability criterion.
        """
        # Semi-major axis (solRad) and eccentricity, per planet.
        a = self.arsun.value
        ecc = orbits.ecc.value[self.orbit_map]

        a_init = self._initial_semimajor_axes(system, orbits)
        # A missing start sorts last but must not become the barrier scale.
        order = np.argsort(np.where(np.isfinite(a_init), a_init, np.inf))

        for k in range(len(order) - 1):
            i, j = int(order[k]), int(order[k + 1])
            if self.orbit_map[i] == self.orbit_map[j]:
                # One orbit cannot cross itself; two planets sharing an
                # orbit index are co-orbital by construction.
                continue

            inner_apastron = a[i] * (1.0 + ecc[i])
            outer_periastron = a[j] * (1.0 - ecc[j])

            scale = a_init[i] if np.isfinite(a_init[i]) else 0.0
            scale = float(max(abs(scale), 1e-6))

            pm.Potential(
                f"{self.prefix}.crossing_bound_"
                f"{self.names[i]}_{self.names[j]}",
                soft_lower_bound(
                    outer_periastron - inner_apastron, 0.0, scale=scale
                ),
            )

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
        "Derived from \\citet{Chen:2017}" table note (stage 6, so every
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
