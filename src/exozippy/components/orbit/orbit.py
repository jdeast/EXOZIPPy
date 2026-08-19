import logging

import astropy.units as u
import numpy as np

logger = logging.getLogger(__name__)

import pymc as pm
import pytensor.tensor as pt
from exoplanet_core.pymc import ops as ops

from exozippy.components.component import Component, in_topology
from exozippy.components.parameter import Parameter
from exozippy.components.parameterization import merge_options, mode_manifest
from exozippy.outputs.prose import get_collector, join_names
from exozippy.potentials import soft_lower_bound, soft_upper_bound

# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics
from .bodies import parse_orbit_bodies


def amplitude_constrained_orbits(system, orbit):
    """Orbits whose motion an RV or astrometric dataset measures.

    The SIGNED observables: an RV or astrometric amplitude flips phase through
    zero, so these are the data that pin down a mass and an inclination sign,
    as opposed to a transit, which measures a depth and a duration and is blind
    to both.

    Two callers want it for different reasons -- `Planet._mass_constrained`
    asks which planets have a signed mass (the Chen mass-side predicate and the
    `linear` vs `log_q` choice), and `Orbit._transit_only` asks which orbits a
    transit measures ALONE, the topology Eastman (2024)'s parameterization is
    for.  One implementation, because the two must never disagree about what
    "measured by RVs" means.

    A module function rather than a method, and taking the orbit as an
    argument: it is a fact about the (system, orbit) PAIR, it needs nothing
    from an Orbit but `star_membership`, and that keeps it usable by anything
    holding a membership map -- including the test doubles that stand in for a
    full Orbit.
    """
    components = getattr(system, "active_components", None) or {}
    constrained = set()
    rv = components.get("rvinstrument")
    if rv is not None:
        for s in set(rv.star_ndx):
            constrained.update(o for o, _ in orbit.star_membership(s))
    ast = components.get("astrometryinstrument")
    if ast is not None:
        for i, mode in enumerate(ast.modes):
            if mode == "rel":
                if ast.rel_orbit[i] is not None:
                    constrained.add(ast.rel_orbit[i])
            else:
                # gaia/abs photocenter wobble sums the orbits whose primary
                # group contains the target star.
                s = int(ast.config[i].get("star_ndx", 0))
                constrained.update(
                    o
                    for o, role in orbit.star_membership(s)
                    if role == "primary"
                )
    return constrained


class Orbit(Component):
    """
    Two-body Keplerian orbit between a primary and a companion body group
    (see bodies.py for the group syntax).  Alongside the timing/geometry
    elements, each orbit derives its own physical scale -- m_primary,
    m_companion, m_total, a, K -- from the masses of its member bodies,
    so hierarchical systems (e.g. B orbits C, B+C orbits A, planet b orbits
    A) stay mass-consistent automatically: every orbit touching a body
    reads the same star.mass/planet.mass nodes.  Each group is treated as a
    point mass at its barycenter (standard hierarchical approximation).
    """

    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)
        self.label = "Orbital Parameters"

        self.primary_bodies, self.companion_bodies = parse_orbit_bodies(
            self.config, getattr(config_manager, "system_config", None)
        )
        self.i180 = [c.get("i180", False) for c in self.config]
        self._parse_ecc_parameterization()

        self._reject_wip_parameterizations()

    # The two eccentricity parameterizations, as a mode table (see
    # components/parameterization.py).  `hk` samples the sqrt(e)cos/sin(omega)
    # pair and derives (ecc, omega) from it; `vcve` samples V_c/V_e and an omega
    # direction vector, derives (ecc, omega) from those, and REPORTS the
    # sqrt(e)cos/sin pair (role 3) so both parameterizations produce the same
    # table rows and a user's prior on either survives the switch.  V_c/V_e
    # itself is reported on an `hk` orbit, for the same reason.  Orbits may
    # differ: element roles are per instance.
    ECC_MODE_TABLE = {
        "hk": {
            "secosw": None,
            "sesinw": None,
            "ecc": "default",
            "omega": "default",
            "tp": "default",
            "esinw": "default",
            "ecosw": "default",
            "vcve": {"output_expr_key": "from_ecc"},
        },
        "vcve": {
            "vcve": None,
            "xomega": None,
            "yomega": None,
            "ecc": {"expr_key": "from_vcve", "force_node": True},
            "omega": {"expr_key": "from_xy", "force_node": True},
            # tp must come from (e, omega) here: this orbit does not sample the
            # sqrt(e) pair, it REPORTS it, and a reported element is consumed by
            # nothing -- reading it would read its pre-patch placeholder.
            "tp": "from_ecc",
            # ...and for the same reason: these reach e sin/cos(omega) through
            # the sqrt(e) pair, which this orbit reports rather than samples.
            "esinw": "from_ecc",
            "ecosw": "from_ecc",
            "secosw": {"output_expr_key": "from_ecc"},
            "sesinw": {"output_expr_key": "from_ecc"},
        },
    }

    # The two INCLINATION parameterizations, per orbit (see
    # components/parameterization.py).  `cosi` samples the cosine of the
    # inclination, which is what an isotropic prior is uniform in; `chord`
    # samples the transit chord instead and derives cos i from it, which is
    # what a transit DURATION constrains (Eastman 2024).  The third mode is
    # not a parameterization at all: an orbit with no single transiting planet
    # has no radius ratio, so it has no chord, and `nochord` leaves the
    # parameter INACTIVE there -- pinned, no potential, no table row.
    #
    # Disjoint from ECC_MODE_TABLE by construction (that one owns the
    # eccentricity coordinates, this one the geometric ones), so the two
    # expansions merge into one manifest without either knowing about the
    # other.  That is also what lets a user turn on either half alone.
    INC_MODE_TABLE = {
        "cosi": {
            "cosi": None,
            "chord": {"output_expr_key": "from_cosi"},
        },
        "chord": {
            "chord": None,
            "cosi": {"expr_key": "from_chord", "force_node": True},
        },
        "nochord": {
            "cosi": None,
        },
    }

    # Parameters whose being PINNED means the user has already decided the
    # quantity a parameterization would reparameterize: {switch: params}.
    _PIN_BLOCKS_DEFAULT = {
        "fitvcve": ("secosw", "sesinw", "ecc"),
        "fitchord": ("cosi", "inc"),
    }

    def _user_pinned(self, index, params):
        """Which of `params` the user pinned (sigma = 0) on orbit `index`.

        Reads the user's own entries, in the two spellings that survive
        `standardize_param_names` (the indexed one, and the broadcast
        `orbit.<param>` that covers every element).  Values, not resolved
        config: a defaults.yaml sigma is not a decision anybody made.
        """
        user = getattr(self.config_manager, "user_params", None) or {}
        pinned = []
        for param in params:
            for key in (
                f"{self.prefix}.{index}.{param}",
                f"{self.prefix}.{param}",
            ):
                entry = user.get(key)
                if isinstance(entry, dict) and entry.get("sigma") == 0:
                    pinned.append(param)
                    break
        return pinned

    def _pin_blocks_default(self, index, switch):
        """True if a pin means this orbit must keep the conventional
        coordinates.

        A parameterization that is ON BY DEFAULT must not throw away a
        constraint the user wrote, and `sigma: 0` on an element that the flip
        makes DERIVED is dropped (with a warning) rather than honored -- the
        one lossy case in an otherwise constraint-preserving switch.  So a user
        who pinned the very quantity being reparameterized keeps their
        coordinates.

        For V_c/V_e there is a second, sharper reason, and it is why this is
        not merely polite: pinning `secosw`/`sesinw` at zero IS a circular
        orbit, and a circular orbit is exactly where the V_c/V_e inversion is
        SINGULAR (V_c/V_e = 1 at omega = 0 is the double root, where the two
        branches merge and de/d(V_c/V_e) is infinite).  The parameterization
        cannot express the fit that config asks for.

        An explicit `fitvcve: true` still wins -- the user asked -- and
        build_pymc warns per element about the dropped fields.
        """
        pinned = self._user_pinned(index, self._PIN_BLOCKS_DEFAULT[switch])
        if not pinned:
            return False
        name = self.names[index] if index < len(self.names) else index
        logger.info(
            "[%s.%s] keeping the conventional coordinates: %s %s pinned "
            "(sigma: 0), and the %s parameterization would make %s derived, "
            "which drops the pin.  Set '%s: true' explicitly to override.",
            self.prefix,
            name,
            ", ".join(pinned),
            "is" if len(pinned) == 1 else "are",
            "V_c/V_e" if switch == "fitvcve" else "transit-chord",
            "it" if len(pinned) == 1 else "them",
            switch,
        )
        return True

    def _transit_only(self, system):
        """Per orbit: is a transit the ONLY thing measuring this orbit?

        The topology Eastman (2024) is about, and the condition under which
        both halves of it default ON.  A transit measures a duration, which
        `V_c/V_e` and the chord carry directly; where an RV or astrometric
        amplitude also measures the orbit, the conventional coordinates are
        well constrained and the paper's argument does not apply.

        Every transit light curve models every planet (the same assumption
        `Planet._resolve_chen` makes for the radius side), so "has transit
        data" is a property of the SYSTEM, while "is otherwise constrained"
        is per orbit.
        """
        if in_topology(system, "transit") is None:
            return [False] * self.n_elements
        constrained = amplitude_constrained_orbits(system, self)
        return [i not in constrained for i in range(self.n_elements)]

    def _chord_planet_indices(self):
        """Per orbit: the index of its one transiting planet, or -1.

        A chord is `sqrt((1 + p)^2 - b^2)`, so it needs a radius ratio -- one
        radius ratio.  An orbit whose companion group holds no planet (a
        stellar binary) has none, and one holding SEVERAL has no single answer:
        two planets sharing an orbit have two different chords, and asking
        which one `orbit.chord` means is a question with no correct answer.
        Both are `nochord`, and `_parse_inc_parameterization` refuses an
        explicit `fitchord: true` on them rather than picking a planet.
        """
        out = []
        for i in range(self.n_elements):
            planets = [
                idx for (t, idx) in self.companion_bodies[i] if t == "planet"
            ]
            out.append(planets[0] if len(planets) == 1 else -1)
        return out

    def _log_parameterization_choices(self, system):
        """Say which orbits the topology moved off the conventional
        coordinates.

        A default that changes the sampled coordinates is exactly the kind of
        thing that should not be discovered by reading a table of unfamiliar
        parameter names, so it is logged per orbit, with the reason and the
        key that turns it off.
        """
        if system is None:
            return
        flipped = [
            self.names[i] if i < len(self.names) else str(i)
            for i in range(self.n_elements)
            if self.fitvcve[i]
            and not (self.config[i] or {}).get("fitvcve", False)
        ]
        if not flipped:
            return
        logger.info(
            "[%s] %s measured by transits alone: sampling V_c/V_e and the "
            "transit chord instead of sqrt(e)cos(omega)/sqrt(e)sin(omega) and "
            "cos i (Eastman 2024), which is what a transit duration "
            "constrains.  secosw/sesinw/cosi are still reported.  Set "
            "'fitvcve: false' on the orbit to fit the conventional "
            "coordinates.",
            self.prefix,
            ", ".join(flipped),
        )

    def _parse_inc_parameterization(self, system=None):
        """Read `fitchord:` into per-orbit mode names.

        Called from register_parameters (stage 3) rather than __init__,
        because both questions it asks are about topology -- whether the orbit
        has a single planet, and whether the system has any transit data --
        and neither is answerable before the components exist.

        `nochord` (the chord is INACTIVE: pinned, no potential, no table row)
        covers two cases, and the second is the reason it takes `system`.  An
        orbit with no single planet has no radius ratio and so no chord at
        all.  An orbit in a system with NO TRANSIT DATA has one arithmetically
        and it means nothing: `sqrt((1 + p)^2 - b^2)` for a companion that
        never crosses the disc is zero, and reporting a column of zeros in
        every RV-only fit is worse than not reporting it.  examples/GaiaBH1 is
        the case that makes this concrete -- it models a BLACK HOLE as a
        `planet` block, so "has a planet" is true and "could transit" is
        emphatically not.

        An explicit `fitchord: true` still samples the chord in a
        transit-free system: that is a reparameterization, not a claim about
        data, and gating it would be a gate where a warning belongs.
        """
        self._chord_planet = self._chord_planet_indices()
        # in_topology, not a bare active_components lookup: this file asked
        # the same question three ways and they disagreed about whether a
        # config-only system counts (review 4.8.1).  It does -- the local
        # _topology helper this replaced said so, and a topology-driven
        # DEFAULT must not depend on whether the component happens to be
        # built yet.
        has_transit = in_topology(system, "transit") is not None
        self.inc_modes = []
        for i, on in enumerate(self.fitchord):
            name = self.names[i] if i < len(self.names) else i
            if self._chord_planet[i] >= 0 and not (on or has_transit):
                # A real planet, but nothing that could see a transit and no
                # request to sample it: the chord is arithmetic, not a result.
                self.inc_modes.append("nochord")
                continue
            if self._chord_planet[i] < 0:
                if bool((self.config[i] or {}).get("fitchord", False)):
                    n_planets = sum(
                        1
                        for (t, _) in self.companion_bodies[i]
                        if t == "planet"
                    )
                    raise ValueError(
                        f"[{self.prefix}.{name}] 'fitchord: true' needs "
                        f"exactly one planet on the orbit -- the chord is "
                        f"sqrt((1 + R_P/R_*)^2 - b^2), so it is defined by a "
                        f"radius ratio -- and this orbit's companion group "
                        f"holds {n_planets}.  Sample 'cosi' here (drop the "
                        f"key), or split the bodies onto their own orbits."
                    )
                self.inc_modes.append("nochord")
            else:
                self.inc_modes.append("chord" if on else "cosi")

    def _parse_ecc_parameterization(self, system=None):
        """Read `fitvcve:`/`fitchord:` into per-orbit mode lists.

        BOTH DEFAULT ON FOR A TRANSIT-ONLY ORBIT (`_transit_only`), which is
        the topology Eastman (2024) measured: over 330 simulated systems, a
        transit-only fit in sqrt(e)cos/sin(omega) recovers eccentricities that
        are measurably wrong, while `V_c/V_e` and the transit chord -- the two
        coordinates a duration constrains -- recover them.  Anywhere an RV or
        astrometric amplitude also measures the orbit, the conventional
        coordinates stay: the paper's argument is about what transits alone
        can and cannot see.  The two halves flip TOGETHER because that is the
        pair the paper validated; turning on one alone is supported but is a
        deliberate act.

        The coupling rule is the user's: `fitvcve: false` forces
        `fitchord: false` unless fitchord was asked for explicitly.  It falls
        out of `fitchord` defaulting to whatever `fitvcve` resolved to, which
        is also what makes `fitvcve: true` alone turn both on.

        `system` is None from __init__, where the data topology is not known
        yet; register_parameters re-parses with it, and that pass is the one
        that decides.  Everything before it is a placeholder, and nothing
        reads the modes in between.
        """
        default_on = (
            self._transit_only(system)
            if system is not None
            else [False] * self.n_elements
        )
        self.fitvcve = []
        for i, c in enumerate(self.config):
            asked = c.get("fitvcve")
            if asked is not None:
                self.fitvcve.append(bool(asked))
                continue
            on = default_on[i] and not self._pin_blocks_default(i, "fitvcve")
            self.fitvcve.append(on)
        self.fitchord = []
        for i, c in enumerate(self.config):
            asked = c.get("fitchord")
            if asked is not None:
                self.fitchord.append(bool(asked))
                continue
            # Follows fitvcve, which is what "unless separately set" means --
            # and is subject to its own pin check, since a fixed inclination is
            # a decision the chord would drop just as surely.
            on = self.fitvcve[i] and not self._pin_blocks_default(
                i, "fitchord"
            )
            self.fitchord.append(on)
        self.ecc_modes = ["vcve" if on else "hk" for on in self.fitvcve]

    # ------------------------------------------------------------------
    # WIP parameterizations (review 5.11)
    # ------------------------------------------------------------------
    # Physics functions the WIP defaults.yaml expression keys name and which
    # nothing defines -- neither orbit/physics.py nor anywhere else, so
    # PHYSICS_REGISTRY has no entry to look up and selecting one of these
    # expression keys cannot build a node at all.
    WIP_PHYSICS = {}

    # Orbit parameter names a params file may reasonably reach for and which
    # this component does not have, as {param: why}.  Not "work in progress"
    # any more -- both parameterizations are built -- but an unknown parameter
    # path is otherwise SILENTLY IGNORED, which is the failure mode
    # `config._reject_renamed_arsun` exists to prevent, so the entry stays and
    # names where the quantity really lives.
    WIP_PARAMS = {
        "b": (
            (),
            "The impact parameter lives on the PLANET, because it is defined "
            "by one -- 'planet.<name>.b', derived from the orbit's cos i and "
            "the planet's a/R*.  The orbit's own semi-major axis is 'a' (AU); "
            "the scaled a/R* is 'planet.<name>.ar'.  If you meant to "
            "constrain the transit geometry from this side, note that "
            "'fitchord: true' samples 'orbit.<name>.chord', the transit "
            "chord sqrt((1 + R_P/R_*)^2 - b^2), which fixes b exactly.",
        ),
    }

    @classmethod
    def _missing_physics(cls, expr_keys):
        """Name the undefined physics functions the given expr_keys call."""
        return ", ".join(
            f"{cls.WIP_PHYSICS[k]}() (defaults.yaml {k})"
            for k in sorted(expr_keys)
        )

    def _reject_wip_parameterizations(self):
        """Refuse a params entry naming an orbit parameter that does not exist.

        Both halves of Eastman (2024) are built now -- `fitvcve:` samples
        V_c/V_e and `fitchord:` samples the transit chord -- so nothing here
        rejects a parameterization any more.  What survives is the one job the
        guard always did independently of that: a params-file key naming
        `orbit.<name>.b` is otherwise SILENTLY IGNORED (the failure mode
        `config._reject_renamed_arsun` exists to prevent), and the impact
        parameter genuinely lives on the planet.  See WIP_PARAMS for the
        message it raises.
        """
        wip = []

        user_params = getattr(self.config_manager, "user_params", None) or {}
        for key in user_params:
            parts = str(key).split(".")
            if len(parts) < 2 or parts[0] != self.prefix:
                continue
            param = parts[-1]
            if param in self.WIP_PARAMS:
                expr_keys, why = self.WIP_PARAMS[param]
                extra = (
                    f"  The defaults.yaml expressions that would consume it "
                    f"call undefined physics functions: "
                    f"{self._missing_physics(expr_keys)}."
                    if expr_keys
                    else ""
                )
                wip.append(
                    f"'{key}': the orbit has no parameter '{param}'.  {why}"
                    f"{extra}"
                )

        if wip:
            raise NotImplementedError("\n".join(wip))

    @property
    def prefix(self):
        return "orbit"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "primary",
                "kind": "ref",
                "accepts": ["star", "planet"],
                "required": False,
                "doc": (
                    "Body group forming the primary of this two-body "
                    "Keplerian arc: a list of star/planet instance names or "
                    "star.X/planet.X paths (a multi-body group is treated as "
                    "a point mass at its barycenter). Omit both primary and "
                    "companion to use the legacy implicit host/planet "
                    "topology."
                ),
            },
            {
                "key": "companion",
                "kind": "ref",
                "accepts": ["star", "planet"],
                "required": False,
                "doc": (
                    "Body group forming the companion of this two-body "
                    "Keplerian arc (see primary)."
                ),
            },
            {
                "key": "i180",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Reflect the inclination about 90 deg (retrograde branch "
                    "of the transit/RV inclination degeneracy). Default false."
                ),
            },
            {
                # Read by Transit/RVInstrument through
                # components/globalsearch.search_mode; it lives on the orbit
                # block because that is the thing being seeded, exactly as
                # 'mmexofast:' lives on the lens block.  The orbit component
                # itself never reads it.
                "key": "global_search",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Blind period search (BLS on transit photometry, "
                    "Lomb-Scargle on radial velocities) to seed this orbit's "
                    "period and conjunction time. Default: run it only when "
                    "the relaxation engine cannot derive them from the params "
                    "file. true forces it; false opts out. Single-orbit "
                    "systems only -- a periodogram peak names no orbit."
                ),
            },
            {
                "key": "fitvcve",
                "kind": "option",
                "accepts": [False],
                "required": False,
                "doc": (
                    "WIP -- 'fitvcve: true' RAISES NotImplementedError.  It "
                    "would parametrize eccentricity via V_c/V_e instead of "
                    "sqrt(e)cos(omega)/sqrt(e)sin(omega), but the from_vcve "
                    "physics functions are undefined.  The per-orbit switch "
                    "itself is no longer a blocker: element roles are per "
                    "instance now."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Body groups
    # ------------------------------------------------------------------
    def bodies(self, i):
        """All (comp_type, index) bodies of orbit i (both groups)."""
        return self.primary_bodies[i] + self.companion_bodies[i]

    def star_membership(self, star_idx):
        """
        Orbits containing star star_idx, as [(orbit_index, role), ...] with
        role 'primary' or 'companion'.  Used by instruments to decide which
        orbits move (or blend with) a given star.
        """
        out = []
        key = ("star", int(star_idx))
        for i in range(self.n_elements):
            if key in self.primary_bodies[i]:
                out.append((i, "primary"))
            elif key in self.companion_bodies[i]:
                out.append((i, "companion"))
        return out

    def build_maps(self):
        """Stage 2: 0/1 weight matrices mapping body masses into groups.

        _group_w[side][comp_type] is an (n_orbits, n_<comp_type>) float
        matrix; the group mass is its product with the component's mass
        vector.  Matrices are built for every component type referenced by
        at least one group.
        """
        sys_cfg = getattr(self.config_manager, "system_config", None) or {}
        self._group_w = {"primary": {}, "companion": {}}
        for side, groups in (
            ("primary", self.primary_bodies),
            ("companion", self.companion_bodies),
        ):
            types = {t for g in groups for (t, _) in g}
            # Sorted: _group_w is a plain dict populated in THIS order, and
            # add_parameter walks it to lazily materialize each referenced
            # component's `mass` -- so an unsorted walk decides whether
            # star.mass or planet.mass becomes a PyMC RV first.
            # model.free_RVs order is the compiled input signature
            # (system.py) and the gradient-vector layout (polish.py),
            # neither of which may depend on PYTHONHASHSEED.  Inert while one side
            # references a single body type; live for a hierarchical group
            # that mixes stars and planets.
            for ctype in sorted(types):
                section = sys_cfg.get(ctype) or []
                if not isinstance(section, list):
                    section = [section]
                n_cols = max(
                    [len(section)]
                    + [idx + 1 for g in groups for (t, idx) in g if t == ctype]
                )
                W = np.zeros((self.n_elements, n_cols))
                for i, g in enumerate(groups):
                    for t, idx in g:
                        if t == ctype:
                            W[i, idx] = 1.0
                self._group_w[side][ctype] = W

    def _resolve_initval(self, name, shape):
        """This orbit's stage-2 initval for ``name``, NaN where unseeded.

        Stage 3 runs BEFORE the relaxation engine, so this sees only what
        the user wrote (plus component hints and defaults.yaml) -- nothing
        the engine will later derive.  Values are in the parameter's own
        user unit; the caller converts.
        """
        n_el = int(np.prod(shape))
        val = self.config_manager.resolve(
            self.prefix, name, shape=shape, names=self.names
        )["initval"]
        if val is None:
            return np.full(n_el, np.nan)
        return np.atleast_1d(val).astype(float).copy()

    def _seeded_period(self, shape):
        """The per-orbit period in days implied by the stage-2 seeds.

        BOTH spellings are legal in a params file, and the relaxation
        engine (stage 4) is what normally reconciles them -- but it has not
        run yet, so a user-supplied ``period:`` has NOT been propagated
        into ``logP``.  Reading ``logP`` alone therefore returns its
        defaults.yaml initval (1.0 -> 10 d) for every fit that seeds
        ``period:``.  Prefer the directly seeded ``period`` and fall back
        to ``10**logP``.

        Still not covered, because only the engine can get there: a period
        implied by ``a`` plus the member masses.  Seed ``logP`` (or
        ``period``) directly when that is how the orbit is specified.
        """
        period_user = self._resolve_initval("period", shape)
        logP = self._resolve_initval("logP", shape)
        return np.where(np.isnan(period_user), 10.0**logP, period_user)

    def register_parameters(self, system):
        """Stage 3: Calculate window constraints and declare the manifest."""
        shape = (self.n_elements,)

        # 1. Peer into the config (Pre-flight windows)
        tc_cfg = self.config_manager.resolve(
            self.prefix, "tc", shape=shape, names=self.names
        )

        tc_init = np.atleast_1d(tc_cfg["initval"])
        # tc is periodic (tc and tc + P are the same solution), so one full
        # period is the right hard window -- but it must be the period the
        # user actually seeded, in either spelling.  See _seeded_period.
        half_period = self._seeded_period(shape) / 2.0

        self.manifest = {
            "logP": None,
            "period": {"force_node": True, "expr_key": "default"},
            "n": "default",
            "tc": {
                "force_node": True,
                "lower": tc_init - half_period,
                "upper": tc_init + half_period,
            },
        }

        # Re-read the switches, NOW with the system: this is the pass that
        # decides, because the transit-only default is a question about the
        # data topology and __init__ cannot see it.  (It is also a re-read for
        # the older reason: `fitvcve` is a plain attribute anyone could set
        # between construction and stage 3.)
        self._parse_ecc_parameterization(system)
        self._reject_wip_parameterizations()
        self._log_parameterization_choices(system)

        # The eccentricity parameterization, per orbit (see ECC_MODE_TABLE).
        # An all-`hk` system -- every shipped example -- gets exactly the
        # entries this used to write by hand, plus the `vcve` it now REPORTS.
        ecc_entries = mode_manifest(
            self.ecc_modes,
            self.ECC_MODE_TABLE,
            n_elements=self.n_elements,
            where=f"{self.prefix}.fitvcve",
        )

        # Insertion order is load-bearing and preserved exactly: graph.py
        # registers its build-order nodes in manifest order, and that order is
        # the order the PyMC nodes -- and so the terms of the summed logp -- get
        # created in.  The historical keys keep their historical positions
        # (`cosi` between the sqrt(e) pair and `ecc`, where it has always been,
        # even though the i180 block below replaces its entry), and the new ones
        # are appended.
        # The inclination parameterization, per orbit (see INC_MODE_TABLE).
        # An all-`cosi` system -- every shipped example -- keeps the entry
        # `cosi` has always had, plus the `chord` it now REPORTS wherever the
        # orbit has a planet to define one.
        self._parse_inc_parameterization(system)
        inc_entries = mode_manifest(
            self.inc_modes,
            self.INC_MODE_TABLE,
            n_elements=self.n_elements,
            where=f"{self.prefix}.fitchord",
        )

        for key in ("secosw", "sesinw"):
            if key in ecc_entries:
                self.manifest[key] = ecc_entries[key]
        self.manifest["cosi"] = inc_entries["cosi"]
        for key in ("ecc", "omega"):
            self.manifest[key] = ecc_entries[key]
        self.manifest.update(
            {
                "inc": "default",
                "sini": "default",
                "sinw": "default",
                "cosw": "default",
            }
        )
        for key in ("esinw", "ecosw", "tp"):
            self.manifest[key] = ecc_entries[key]
        for key in ("vcve", "xomega", "yomega"):
            if key in ecc_entries:
                self.manifest[key] = ecc_entries[key]
        if "chord" in inc_entries:
            self.manifest["chord"] = inc_entries["chord"]

        # Physical scale of every orbit, from the member bodies' masses
        # (see class docstring).  Group-mass deps name the mass vectors of
        # whichever component types the groups actually reference; the
        # weighted per-group sums are injected in add_parameter below.
        # Bare orbits whose implicit default bodies do not exist (test
        # harnesses, geometry-only systems) skip the scale parameters.
        if self._validate_bodies(system):
            body_types = sorted(
                {
                    t
                    for i in range(self.n_elements)
                    for (t, _) in self.bodies(i)
                }
            )
            group_deps = [f"{t}.mass" for t in body_types]
            self.manifest.update(
                {
                    "m_primary": {"expr_key": "default", "deps": group_deps},
                    "m_companion": {"expr_key": "default", "deps": group_deps},
                    "m_total": "default",
                    "a": "default",
                    "K": "default",
                }
            )

        # Rossiter-McLaughlin: declare the spin-orbit params only when some
        # rvinstrument enables `rm:`. Samples the decorrelated
        # sqrt(vsini)cos/sin(lambda) pair and derives vsini/lam from them
        # (mirrors the secosw/sesinw -> ecc/omega idiom above).
        from ..rm import rm_enabled

        if rm_enabled(system):
            self.manifest.update(
                {
                    "svcoslam": None,
                    "svsinlam": None,
                    "vsini": {"expr_key": "from_sv"},
                    "lam": {"expr_key": "from_sv"},
                }
            )

        # Astrometry constrains the longitude of the ascending node and
        # breaks the i <-> 180-i degeneracy, so sample the node direction
        # vector (xbigomega, ybigomega; each N(0,1) -> uniform marginal on
        # bigomega, like the microlensing trajectory angle alpha) and allow
        # the full inclination range when an astrometry component is active.
        has_astrometry = (
            in_topology(system, "astrometryinstrument") is not None
        )
        if has_astrometry:
            self.manifest["xbigomega"] = None
            self.manifest["ybigomega"] = None
            self.manifest["bigomega"] = "default"

            # The (bigomega, omega) <-> (bigomega+180, omega+180)
            # transformation is a reflection through the sky plane
            # (z -> -z): invisible to ANY astrometry, absolute or relative.
            # Only radial information (RVs) identifies the ascending node.
            has_rv = in_topology(system, "rvinstrument") is not None
            if not has_rv:
                self._restrict_bigomega_halfplane(shape)

        i180_arr = np.atleast_1d(getattr(self, "i180", False)) | has_astrometry
        derived_lowers = np.where(i180_arr, -1.0, 0.0)
        # merge_options, not a fresh dict: on a `fitchord` orbit this entry
        # carries an expr_key, and overwriting it would silently turn the
        # derived cos i back into a sampled one (review 4.5.3).  The bound
        # keeps its meaning either way -- hard support where cos i is sampled,
        # a soft barrier where it is derived.
        self.manifest["cosi"] = merge_options(
            self.manifest.get("cosi"), lower=derived_lowers
        )
        # The sign the chord parameterization cannot see: a transit at i and
        # at 180 - i are the same transit, so `calc_cosi_from_chord` is handed
        # this as a context node rather than trying to recover it.  It follows
        # `i180:` ALONE and not i180_arr above: astrometry widens cos i's bound
        # to [-1, 1] because astrometry MEASURES the sign, which is the one
        # thing a chord cannot express -- so where both are asked for, the
        # chord orbit keeps the +1 branch and says so.
        own_i180 = np.atleast_1d(getattr(self, "i180", False)).astype(bool)
        if own_i180.size != self.n_elements:
            own_i180 = np.zeros(self.n_elements, dtype=bool)
        self._chord_sign = np.where(own_i180, -1.0, 1.0)
        if has_astrometry:
            chord_orbits = [
                self.names[i] if i < len(self.names) else str(i)
                for i, m in enumerate(getattr(self, "inc_modes", []))
                if m == "chord"
            ]
            if chord_orbits:
                logger.warning(
                    "[%s] 'fitchord: true' on %s, but this system has "
                    "astrometry, which measures the SIGN of cos i -- and the "
                    "transit chord is even in it, so the fit is restricted to "
                    "the %s branch (set 'i180: true' to select the other). "
                    "Sample 'cosi' instead to let the astrometry choose.",
                    self.prefix,
                    ", ".join(chord_orbits),
                    "i > 90 deg" if own_i180.any() else "i < 90 deg",
                )

    def _restrict_bigomega_halfplane(self, shape):
        """Astrometry without RVs: restrict bigomega to [0, 180] deg.

        (bigomega, omega_*) and (bigomega+180, omega_*+180) is a reflection
        through the sky plane, so it produces identical astrometry of every
        kind (absolute, epoch, and relative); only RVs identify which node
        is ascending.  Bounding ybigomega >= 0 selects the bigomega in
        [0, 180] mode.  Seeds in (180, 360) are remapped to the equivalent
        solution -- which flips (xbigomega, ybigomega) AND (secosw,
        sesinw), and shifts tc so the orbit's position-vs-time is
        unchanged.  A table note documents the artificial boundary on
        omega_* and bigomega.
        """
        note = (
            r"With astrometry but no RVs, $(\Omega, \omega_*)$ and "
            r"$(\Omega+180^\circ, \omega_*+180^\circ)$ are exactly "
            r"degenerate (which node is ascending is unknown); "
            r"$\Omega$ is artificially restricted to "
            r"$[0^\circ, 180^\circ]$ to select one mode."
        )

        # NOTE: this runs at stage 3, BEFORE the relaxation engine, so only
        # user-provided initvals (and defaults) are visible here.  The x/y
        # direction vector is therefore derived directly from the user's
        # bigomega initval; the manifest initvals set below override the
        # relaxation-engine seeds at build time.
        cm = self.config_manager

        def rslv(name):
            return self._resolve_initval(name, shape)

        factor_bo = cm.get_conversion_factor(self.prefix, "bigomega") or 1.0
        bo = rslv("bigomega") * factor_bo  # rad; NaN where unseeded

        # Unseeded elements start at bigomega = 90 deg (center of the
        # allowed half-plane; y = 0 would sit exactly on the new bound).
        x_init = np.where(np.isnan(bo), 0.0, np.cos(bo))
        y_init = np.where(np.isnan(bo), 1.0, np.sin(bo))

        # Seeds with bigomega in (180, 360): remap to the degenerate
        # partner (bigomega - 180, omega_* + 180, and tc shifted so the
        # position-vs-time model is unchanged).
        flip = y_init < 0.0
        if np.any(flip):
            logger.warning(
                f"[{self.prefix}] bigomega initval(s) in (180, 360) deg but no "
                f"RVs are present; remapping element(s) {np.where(flip)[0]} to "
                f"the degenerate (bigomega-180, omega+180) solution."
            )

            # Orientation in the user's own terms: prefer explicit ecc+omega
            # initvals; otherwise secosw/sesinw (user or defaults).
            sc0 = rslv("secosw")
            ss0 = rslv("sesinw")
            factor_om = cm.get_conversion_factor(self.prefix, "omega") or 1.0
            om = rslv("omega") * factor_om
            e_u = rslv("ecc")
            have_ew = ~np.isnan(om) & ~np.isnan(e_u)
            sc0 = np.where(have_ew, np.sqrt(np.abs(e_u)) * np.cos(om), sc0)
            ss0 = np.where(have_ew, np.sqrt(np.abs(e_u)) * np.sin(om), ss0)

            tc0 = rslv("tc")
            period = self._seeded_period(shape)

            def _M_c(ecc, w):
                E_c = 2.0 * np.arctan2(
                    np.sqrt(1.0 - ecc) * (1.0 - np.sin(w)),
                    np.sqrt(1.0 + ecc) * np.cos(w),
                )
                return E_c - ecc * np.sin(E_c)

            # Same ceiling calc_ecc applies, for the same reason: this is a
            # forward-model evaluation (a Kepler solve), not a bound.
            ecc0 = np.clip(sc0**2 + ss0**2, 0.0, physics.MAX_ECC)
            w0 = np.arctan2(ss0, sc0)
            n_mm = 2.0 * np.pi / period
            tp = tc0 - _M_c(ecc0, w0) / n_mm
            tc_new = tp + _M_c(ecc0, w0 + np.pi) / n_mm

            x_init = np.where(flip, -x_init, x_init)
            y_init = np.where(flip, -y_init, y_init)
            sc_init = np.where(flip, -sc0, sc0)
            ss_init = np.where(flip, -ss0, ss0)
            tc_init = np.where(flip, tc_new, tc0)

            # merge_options, not `{**entry, ...}`: a manifest entry may be None
            # (a free parameter, which is how mode_manifest spells the
            # sqrt(e) pair) or a bare string naming an expression, and both
            # spellings break a splat -- the first with a TypeError, the second
            # by silently dropping the expr_key (review 4.5.3).
            self.manifest["secosw"] = merge_options(
                self.manifest.get("secosw"), initval=sc_init
            )
            self.manifest["sesinw"] = merge_options(
                self.manifest.get("sesinw"), initval=ss_init
            )
            half_period = period / 2.0
            self.manifest["tc"] = merge_options(
                self.manifest.get("tc"),
                initval=tc_init,
                lower=tc_init - half_period,
                upper=tc_init + half_period,
            )

        # Keep seeded boundary values (bigomega exactly 0 or 180) strictly
        # inside the ybigomega >= 0 bound.
        y_init = np.maximum(y_init, 1e-6)

        self.manifest["xbigomega"] = {"initval": x_init}
        self.manifest["ybigomega"] = {"initval": y_init, "lower": 0.0}
        self.manifest["bigomega"] = {"expr_key": "default", "table_note": note}
        self.manifest["omega"] = {"expr_key": "default", "table_note": note}

    def _validate_bodies(self, system):
        """Check body references against the live system topology.

        Returns True when every body resolves to an active component
        element, so the mass/scale parameters can be built.  Unresolvable
        bodies raise if the user declared the groups explicitly; implicit
        defaults (bare orbit in a test harness or geometry-only system)
        just disable the scale parameters.
        """
        if not hasattr(system, "active_components"):
            return False
        for i in range(self.n_elements):
            explicit = (
                "primary" in self.config[i] or "companion" in self.config[i]
            )
            for ctype, idx in self.bodies(i):
                comp = getattr(system, ctype, None)
                bad = (
                    comp is None
                    or not isinstance(comp, Component)
                    or idx >= comp.n_elements
                )
                if bad and explicit:
                    n = comp.n_elements if isinstance(comp, Component) else 0
                    raise ValueError(
                        f"[{self.prefix}.{self.names[i]}] references body "
                        f"'{ctype}.{idx}', but the active system has only "
                        f"{n} '{ctype}' instance(s)."
                    )
                if bad:
                    logger.info(
                        f"[{self.prefix}.{self.names[i]}] implicit body "
                        f"'{ctype}.{idx}' is not in the system; orbit "
                        f"mass/scale parameters (m_total, a, K) are "
                        f"disabled."
                    )
                    return False
        # A planet in a companion group should point its orbit_ndx here
        # (transit/planet geometry reads the orbit through that map).
        planet_cfgs = (
            getattr(self.config_manager, "system_config", None) or {}
        ).get("planet") or []
        for i in range(self.n_elements):
            for ctype, idx in self.companion_bodies[i]:
                if ctype != "planet" or idx >= len(planet_cfgs):
                    continue
                o_ndx = int((planet_cfgs[idx] or {}).get("orbit_ndx", 0))
                if o_ndx != i:
                    logger.warning(
                        f"[{self.prefix}.{self.names[i]}] companion planet."
                        f"{idx} has orbit_ndx={o_ndx}, not {i}; the planet's "
                        f"transit/RV geometry will follow orbit {o_ndx} "
                        f"while its mass moves this orbit."
                    )
        return True

    _GROUP_MASS_SIDE = {"m_primary": "primary", "m_companion": "companion"}

    # The chord expressions' deps that are NOT orbit parameters: the
    # transiting planet's geometry and the i180 sign, injected as context
    # nodes by _chord_context below.  Declaring them here is what keeps
    # graph.py from looking for an `orbit.p` (the group masses avoid this by
    # naming `planet.mass`, a real parameter of a real component; there is no
    # such parameter for `chord_sign` at all, and `p`/`ar` are per PLANET, so
    # the orbit could not consume them elementwise anyway).
    context_dep_names = frozenset({"p", "ar", "chord_sign"})

    # ...and all three are built per ORBIT, so Component._element_expression
    # may slice them to a per-element mask.
    aligned_context_deps = frozenset({"p", "ar", "chord_sign"})

    # Parameters whose expressions consume the transiting planet's geometry.
    _CHORD_PARAMS = ("cosi", "chord")

    def _chord_context(self, model, system):
        """`p`, `ar` and `chord_sign` for every orbit, as context nodes.

        The chord is defined by the orbit's transiting PLANET, and the orbit
        has no map naming another component's parameters -- so these travel
        the same channel the group masses do (see add_parameter below): the
        component builds them itself and hands them to the generic machinery
        under the dep names the expression asks for.

        Every vector is length n_elements, indexed by ORBIT, which is what
        makes them safe to slice per element (`aligned_context_deps`).  An
        orbit with no single planet reads planet 0 and contributes nothing:
        `chord` is INACTIVE there and `cosi` is sampled, so no expression this
        feeds is evaluated on those elements.  Filling them with a real
        planet's numbers rather than NaN is deliberate -- a NaN would ride
        through `pt.set_subtensor`'s unselected half into the gradient.
        """
        # `system` is None in standalone use -- a bare Orbit built by a test
        # harness, which every geometry test does -- and `cosi` is exactly the
        # parameter those build.  There is nothing to read and nothing that
        # needs it (a standalone orbit is never in chord mode, so no expression
        # these feed is selected), but the dep parser still wants the names.
        planet = None
        if system is not None:
            planet = getattr(system, "active_components", {}).get("planet")
        idx = np.asarray(getattr(self, "_chord_planet", []), dtype=int)
        if idx.size != self.n_elements:
            idx = np.full(self.n_elements, -1, dtype=int)
        sign = np.asarray(
            getattr(self, "_chord_sign", np.ones(self.n_elements)),
            dtype=float,
        )
        if sign.size != self.n_elements:
            sign = np.ones(self.n_elements)

        ctx = {"chord_sign": pt.as_tensor_variable(sign)}
        # Stashed for _add_chord_terms, which needs the same two vectors to
        # build the Jacobian and the geometry bound at stage 7 and must not
        # build a SECOND copy of them: the barrier has to restrain the very
        # node the model was built from.
        self._chord_geometry = ctx
        if planet is None or planet.n_elements == 0:
            # No planet component at all: only reachable with `chord` absent
            # from the manifest (every orbit is `nochord`), so these are
            # placeholders that keep the dep parser happy.
            ctx["p"] = pt.zeros((self.n_elements,))
            ctx["ar"] = pt.zeros((self.n_elements,))
            return ctx

        safe = np.where(idx < 0, 0, idx).astype("int32")
        take = pt.as_tensor_variable(safe)
        for name in ("p", "ar"):
            if not isinstance(getattr(planet, name, None), Parameter):
                planet.add_parameter(model, name, system)
            ctx[name] = getattr(planet, name).value[take]
        return ctx

    def finalize_reported(self, model, system, context_nodes=None):
        """The deferred pass after stage 7, with the planet geometry the reported `chord` needs.

        `System.build_model` calls this with no context nodes -- it cannot
        know what a component's deferred expressions consume -- so the orbit
        supplies its own, exactly as add_parameter does below.
        """
        ctx = dict(context_nodes or {})
        if getattr(self, "_pending_reported", None):
            ctx.update(self._chord_context(model, system))
        return super().finalize_reported(model, system, ctx)

    def add_parameter(self, model, param_name, system, context_nodes=None):
        """
        The group masses are weighted sums over other components' mass
        vectors -- a matrix product the generic dep parser cannot express.
        Intercept them here: build each referenced component's mass node,
        pre-compute the per-group weighted sums, and hand them to the
        generic machinery as context nodes keyed by the dep names.
        """
        side = self._GROUP_MASS_SIDE.get(param_name)
        if side is not None and not context_nodes:
            if not hasattr(self, "_group_w"):
                self.build_maps()  # standalone use outside the lifecycle
            context_nodes = dict(context_nodes or {})
            for ctype, W in self._group_w[side].items():
                comp = getattr(system, ctype, None)
                if comp is None:
                    # Standalone harness (validated systems raised at stage
                    # 2): absent components contribute zero mass.
                    context_nodes[f"{ctype}.mass"] = pt.zeros(
                        (self.n_elements,)
                    )
                    continue
                if not isinstance(getattr(comp, "mass", None), Parameter):
                    comp.add_parameter(model, "mass", system)
                context_nodes[f"{ctype}.mass"] = pt.dot(
                    pt.as_tensor_variable(W), comp.mass.value
                )
            # A group side may reference only a subset of the body types
            # named in the shared deps list; the missing type contributes
            # zero mass.
            for ctype in ("star", "planet"):
                dep = f"{ctype}.mass"
                if dep not in context_nodes and any(
                    d == dep for d in self.manifest[param_name]["deps"]
                ):
                    context_nodes[dep] = pt.zeros((self.n_elements,))
        if param_name in self._CHORD_PARAMS:
            context_nodes = dict(context_nodes or {})
            for dep, node in self._chord_context(model, system).items():
                context_nodes.setdefault(dep, node)
        return super().add_parameter(model, param_name, system, context_nodes)

    def build_likelihood(self, model, system):
        self._add_eccentricity_bound(system)
        self._add_vcve_terms(system)
        self._add_chord_terms(system)

    def _chord_indices(self):
        """Indices of the orbits sampling the chord (empty for every cos i
        system)."""
        modes = list(getattr(self, "inc_modes", []))
        return [i for i, m in enumerate(modes) if m == "chord"]

    def _add_chord_terms(self, system):
        """The two terms a chord orbit owes: the Jacobian and the shield.

        The geometric half of what `_add_vcve_terms` does for the
        eccentricity, and deliberately a separate potential rather than a
        joint one: the paper's eq 6 is the determinant of BOTH
        reparameterizations at once, but applying them independently is what
        lets a user turn on either half alone, and the product of the two
        factors is that determinant.

        THE JACOBIAN keeps the prior on the inclination isotropic.  Sampling
        cos i uniformly is what `p(cos i) = const` means, and it is what this
        component does by default; sampling the chord instead induces
        `p(cos i) = |d(chord)/d(cos i)|`, which is NOT uniform -- it vanishes
        at a central transit and diverges at a grazing one.  Flattening it
        means adding `log|d(cos i)/d(chord)|`, i.e. SUBTRACTING what
        `physics.chord_log_jacobian` returns.  The sign is the term, exactly
        as it is for V_c/V_e; the direction is measured (the implied density
        on cos i is checked for flatness) rather than argued, because a
        finite-difference check of the derivative passes under either sign.

        THE SHIELD is the soft half of the pair.  `chord_radicand` is floored
        inside `calc_cosi_from_chord`'s sqrt so a NaN is unbuildable, which
        leaves that whole region flat -- so the penalty here reads the
        UNFLOORED radicand, where it has a gradient pointing back to a chord
        that a transit could actually produce.  Same argument, and the same
        helper, as the eccentricity bound and the V_c/V_e real-root bound.
        """
        idx = self._chord_indices()
        if not idx:
            return
        if not (
            isinstance(getattr(self, "chord", None), Parameter)
            and isinstance(getattr(self, "ecc", None), Parameter)
            and isinstance(getattr(self, "esinw", None), Parameter)
        ):
            return
        geom = getattr(self, "_chord_geometry", None)
        if geom is None:
            return

        take = np.asarray(idx, dtype="int32")
        chord = self.chord.value[take]
        ecc = self.ecc.value[take]
        esinw = self.esinw.value[take]
        p_ratio = geom["p"][take]
        ar = geom["ar"][take]

        # MINUS the derivative -- see the docstring, and vcve_log_jacobian's,
        # for why the sign is the whole content of this term.
        pm.Potential(
            f"{self.prefix}.chord_jacobian",
            -pt.sum(
                physics.chord_log_jacobian(chord, p_ratio, ar, ecc, esinw)
            ),
        )
        # scale = 1.0: the radicand is (1 + p)^2 - chord^2, an O(1) quantity
        # in units of R_*, so the default 1% softness is a 0.01-wide
        # transition -- the same order of steepness as the eccentricity
        # bound's, and 4.4 nats one width past the last transiting geometry.
        pm.Potential(
            f"{self.prefix}.chord_geometry",
            soft_lower_bound(
                physics.chord_radicand(chord, p_ratio), 0.0, scale=1.0
            ),
        )

        names = [self.names[i] if i < len(self.names) else str(i) for i in idx]
        self.chord.add_prior_contribution(
            latex=r"$\propto |\partial \cos{i} / \partial \rm chord|$",
            text="uniform in cos i (Jacobian applied)",
            elements=idx,
            supersedes_bounds=True,
            support_phrase="whose chord support is",
        )
        collector = get_collector(system)
        if collector is not None:
            collector.add(
                "The transit geometry of "
                f"{join_names(names)} was parametrized by the transit chord "
                r"rather than $\cos{i}$ \citep{Eastman:2024}, multiplied by "
                r"$|\partial \cos{i} / \partial \rm chord|$ so that the "
                r"prior on the inclination remains isotropic.",
                section="orbits",
                key="orbit.chord",
            )

    def _vcve_indices(self):
        """Indices of the orbits sampling V_c/V_e (empty for every hk system)."""
        modes = list(getattr(self, "ecc_modes", []))
        return [i for i, m in enumerate(modes) if m == "vcve"]

    def _add_vcve_terms(self, system):
        """The two terms a V_c/V_e orbit owes: the Jacobian and the shield.

        THE JACOBIAN keeps the prior uniform in eccentricity.  A uniform step
        in V_c/V_e "imposes a non-physical prior that strongly biases e toward
        high eccentricities" (Eastman 2024, section 3), so the likelihood
        carries `log|de/d(V_c/V_e)|` -- MINUS what `physics.vcve_log_jacobian`
        returns; see the sign comment below, which is the difference between
        removing that bias and doubling it.  Applied per orbit, from the
        `ecc`/`omega` NODES, which is what makes the branch mixture replicate it
        per branch automatically: each root then carries its own weight, and
        that is exactly right, because the Jacobian differs between the two
        roots.

        THE SHIELD is the soft half of the pair that keeps an imaginary
        eccentricity from being a wall.  `_vcve_quadratic` floors the
        discriminant at zero (the hard half, so no NaN can ever be built), which
        leaves that whole region flat -- so the penalty here is applied to the
        UNFLOORED discriminant, where it has a gradient pointing back into the
        region where a real eccentricity exists.  Same argument, and the same
        `soft_lower_bound` helper, as the eccentricity bound above.

        The chord half's own independent Jacobian (`|d(chord)/d(cos i)|`) lands
        with the chord half; the paper's eq 6 is the joint determinant of the
        two, and JDE's design applies them independently so either half can be
        switched on alone.
        """
        idx = self._vcve_indices()
        if not idx:
            return
        if not (
            isinstance(getattr(self, "vcve", None), Parameter)
            and isinstance(getattr(self, "ecc", None), Parameter)
            and isinstance(getattr(self, "omega", None), Parameter)
        ):
            return

        take = np.asarray(idx, dtype="int32")
        ecc = self.ecc.value[take]
        omega = self.omega.value[take]
        vcve = self.vcve.value[take]

        # MINUS the derivative, and the sign is the term.  V_c/V_e is the
        # sampled coordinate, so the eccentricity it derives inherits the
        # density p(e) ~ |d(V_c/V_e)/de|, which diverges as e -> 1 -- the bias
        # the paper reports.  Flattening it means adding log|de/d(V_c/V_e)|,
        # i.e. subtracting what vcve_log_jacobian returns.  Adding it would
        # double the bias, and no check of the derivative's MAGNITUDE can tell
        # the two apart, so the direction is pinned by measuring the implied
        # prior on e for flatness (tests/test_vcve.py).
        pm.Potential(
            f"{self.prefix}.vcve_jacobian",
            -pt.sum(physics.vcve_log_jacobian(ecc, omega)),
        )
        # Declare the OTHER root, so the likelihood is marginalized over both
        # instead of one being chosen (System.register_branch_alternative).  One
        # declaration per V_c/V_e orbit: two orbits are four combinations, which
        # is why the mixture warns past two.  Substituting BOTH the clipped
        # `ecc` node and the unclipped one the collision barrier reads is what
        # makes that barrier a per-branch weight rather than a term evaluated
        # only at the primary root.
        register = getattr(system, "register_branch_alternative", None)
        if callable(register):
            unclipped = getattr(self, "_vcve_unclipped_nodes", {})
            for i in idx:
                alt_ecc = pt.set_subtensor(
                    self.ecc.value[i],
                    physics.calc_ecc_from_vcve_lo(
                        self.vcve.value[i], self.omega.value[i]
                    ),
                )
                replacements = {self.ecc.value: alt_ecc}
                node = unclipped.get(i)
                if node is not None:
                    alt_unclipped = pt.set_subtensor(
                        node[i],
                        physics.ecc_from_vcve_unclipped(
                            self.vcve.value[i],
                            self.omega.value[i],
                            upper=False,
                        ),
                    )
                    replacements[node] = alt_unclipped
                name = self.names[i] if i < len(self.names) else str(i)
                register(
                    f"{self.prefix}.{name}: lower V_c/V_e root",
                    replacements,
                )
        # scale = 1.0 because the discriminant 1 - (V_c/V_e)^2 cos^2 omega is
        # dimensionless and at most 1 by construction, so the default 1%
        # softness is a 0.01-wide transition: ~440 nats per unit, the same
        # order of steepness as the collision bound's 500 (see
        # _add_eccentricity_bound), and 4.4 nats one transition width past the
        # fold.
        pm.Potential(
            f"{self.prefix}.vcve_real_root",
            soft_lower_bound(
                physics.vcve_discriminant(vcve, omega), 0.0, scale=1.0
            ),
        )

        # The Jacobian is not a prior the parameters state themselves, and it
        # REPLACES what the sampled bounds imply, so the tables must say so
        # rather than reporting "Uniform" on vcve (see "Reporting
        # component-added priors").
        names = [self.names[i] if i < len(self.names) else str(i) for i in idx]
        self.vcve.add_prior_contribution(
            latex=r"$\propto |\partial e / \partial (V_c/V_e)|$",
            text="uniform in e (Jacobian applied)",
            elements=idx,
            supersedes_bounds=True,
            support_phrase="whose V_c/V_e support is",
        )
        collector = get_collector(system)
        if collector is not None:
            collector.add(
                "The eccentricity and argument of periastron of "
                f"{join_names(names)} were parametrized by "
                r"$V_c/V_e$ and the direction of $\omega_*$ "
                r"\citep{Eastman:2024}, with the likelihood marginalized over "
                r"both roots of the $V_c/V_e$ inversion and multiplied by "
                r"$|\partial e / \partial (V_c/V_e)|$ so that the prior on the "
                r"eccentricity remains uniform.",
                section="orbits",
                key="orbit.vcve",
            )

    def _add_eccentricity_bound(self, system):
        """Soft upper bound on every orbit's eccentricity.

        The barrier is applied to the UNCLIPPED sum secosw^2 + sesinw^2, not
        to self.ecc: calc_ecc clips at MAX_ECC = 0.9999, so feeding the
        clipped node here froze the penalty at a constant on the whole
        e > 0.9999 region -- a flat plateau with exactly zero gradient, and
        no restoring force for NUTS to follow back out.  That region is not
        a corner case: secosw and sesinw are each uniform on [-1, 1], so the
        clipped part of the sampled square has area 4 - pi * 0.9999, i.e.
        21.5% of the prior volume.  This is the identical mistake documented
        (and already fixed) for m_total in Planet.build_likelihood.

        The bound lives here, not on the planet component, because
        eccentricity is a property of the ORBIT: a stellar binary with no
        planet at all used to get no eccentricity bound whatsoever.  Where a
        planet does orbit, its collision limit (planet.max_ecc, the
        eccentricity at which periastron reaches the stellar surface) is the
        tighter constraint, so the per-orbit threshold is the minimum of
        MAX_ECC and the max_ecc of every planet mapped to that orbit.  One
        potential per orbit, planet or no planet.

        scale = 0.88 with the default 1% softness gives the historical
        steepness of 4.4 / 0.0088 = 500 nats per unit eccentricity, matching
        the barrier this replaces (and Planet's mass barrier).
        """
        e_unclipped = self._unclipped_ecc()
        if e_unclipped is None:
            return

        threshold = pt.as_tensor_variable(
            np.full(self.n_elements, physics.MAX_ECC)
        )
        planets = system.active_components.get("planet")
        if isinstance(getattr(planets, "max_ecc", None), Parameter):
            for p, o in enumerate(np.atleast_1d(planets.orbit_map)):
                o = int(o)
                threshold = pt.set_subtensor(
                    threshold[o],
                    pt.minimum(threshold[o], planets.max_ecc.value[p]),
                )

        pm.Potential(
            f"{self.prefix}.e_collision_bound",
            soft_upper_bound(e_unclipped, threshold, scale=0.88),
        )

    def _unclipped_ecc(self):
        """The unclipped eccentricity of every orbit, or None if unavailable.

        What a soft bound must see: `calc_ecc` (and `calc_ecc_from_vcve`) clip
        at MAX_ECC, and a flat penalty has no gradient for NUTS to follow.
        Per orbit, because the coordinate the eccentricity is built from is per
        orbit: `secosw^2 + sesinw^2` on a sqrt(e)cos/sin orbit, the unclipped
        V_c/V_e root on a V_c/V_e one.  A vector of both is assembled here, so
        the collision bound above stays one potential over all orbits whatever
        each of them samples.
        """
        vcve_mask = np.atleast_1d(
            np.asarray(getattr(self, "ecc_modes", []), dtype=object) == "vcve"
        )
        if vcve_mask.size != self.n_elements:
            vcve_mask = np.zeros(self.n_elements, dtype=bool)

        hk = None
        secosw = getattr(self, "secosw", None)
        sesinw = getattr(self, "sesinw", None)
        if isinstance(secosw, Parameter) and isinstance(sesinw, Parameter):
            hk = physics.ecc_from_sqrte(secosw.value, sesinw.value)

        vcve = getattr(self, "vcve", None)
        omega = getattr(self, "omega", None)
        vc = None
        if isinstance(vcve, Parameter) and isinstance(omega, Parameter):
            vc = physics.ecc_from_vcve_unclipped(vcve.value, omega.value)

        if not vcve_mask.any():
            if hk is None:
                logger.debug(
                    "[orbit] secosw/sesinw are not built; skipping the "
                    "eccentricity bound."
                )
            return hk
        if vc is None:
            logger.debug(
                "[orbit] vcve/omega are not built; skipping the eccentricity "
                "bound."
            )
            return None
        if hk is None or bool(vcve_mask.all()):
            self._vcve_unclipped_nodes = {
                int(i): vc for i in np.nonzero(vcve_mask)[0]
            }
            return vc
        # The elements this REPLACES are exactly the ones whose secosw/sesinw
        # are reported, i.e. whose `hk` entries are phase-1 placeholders at this
        # point (build_likelihood runs before finalize_reported).  So the
        # substitution is not a preference between two live values -- it is what
        # keeps a placeholder out of the bound.  set_subtensor and not a
        # pt.where over the two vectors for the house reason as well: a
        # discarded branch's value never enters the graph, since where's VJP
        # multiplies it by zero and 0*NaN poisons the whole vector's gradient
        # (see Parameter._patch_elements).
        idx = np.nonzero(vcve_mask)[0].astype("int32")
        mixed = pt.set_subtensor(hk[idx], vc[idx])
        self._vcve_unclipped_nodes = {int(i): mixed for i in idx}
        return mixed

    def get_true_anomaly(self, t):
        """Returns the true anomaly f for all planets at all times."""
        t_grid = t[:, None]
        tp = self.tp.value[None, :]
        n = self.n.value[None, :]
        ecc = self.ecc.value[None, :]

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        return pt.arctan2(sinf, cosf)

    def get_sky_position(self, t, a_scale, orbit_map, relative=False):
        """
        Vectorized sky-plane offsets of an orbiting body.

        t: (N_obs,) vector of times [BJD_TDB]
        a_scale: (N_planets,) amplitude scaling, e.g. the photocenter or
                 relative semimajor axis in mas; sets the output units
        orbit_map: integer map from planet slots to orbit elements
        relative: False -> the primary/photocenter orbit around the
                  barycenter (uses omega_*); True -> the companion's orbit
                  relative to the primary (omega_* + 180 deg)

        Returns (dE, dN), each (N_obs, N_planets): offsets toward East and
        North in the units of a_scale.

        Conventions (EXOFASTv2): omega is the argument of periastron of the
        PRIMARY's orbit (omega_*). bigomega is the position angle of the
        ascending node, measured East of North, where the ascending node is
        the node at which the body recedes from the observer -- consistent
        with the sign of get_radial_velocity (the primary crosses its
        ascending node at omega_* + f = 0, where its RV is maximal).
        Without RVs, (bigomega, omega) and (bigomega+180, omega+180) are
        exactly degenerate for astrometry of every kind (a reflection
        through the sky plane); see _restrict_bigomega_halfplane.
        """
        t_grid = t[:, None]
        tp = self.tp.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]
        cosi = self.cosi.value[orbit_map][None, :]
        bigomega = self.bigomega.value[orbit_map][None, :]
        cosO = pt.cos(bigomega)
        sinO = pt.sin(bigomega)

        if relative:
            # The companion's argument of periastron is omega_* + pi
            cosw = -cosw
            sinw = -sinw

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        # Separation from the barycenter (or primary) in units of a_scale
        r = a_scale[None, :] * (1.0 - ecc**2) / (1.0 + ecc * cosf)

        # cos/sin(omega + f)
        coswf = cosw * cosf - sinw * sinf
        sinwf = sinw * cosf + cosw * sinf

        # Thiele-Innes projection (North, East), PA measured East of North:
        # at omega + f = 0 (ascending node) the body sits at PA = bigomega.
        dN = r * (cosO * coswf - sinO * sinwf * cosi)
        dE = r * (sinO * coswf + cosO * sinwf * cosi)
        return dE, dN

    def get_radial_velocity(self, t, K, orbit_map):
        """
        The optimized vectorized reflex RV signal.
        t: (N_obs,) vector of times
        K: (N_planets,) vector of semi-amplitudes
        """
        # 1. Broadcast time and orbital parameters into 2D grids
        # Shape: (N_obs, N_planets)
        t_grid = t[:, None]
        tp = self.tp.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]
        K_grid = K[None, :]

        # 2. Calculate Mean Anomaly (M)
        # M = n * (t - tp)
        M = (t_grid - tp) * n

        # 3. Solve Kepler's Equation
        # ops.kepler handles the (N_obs, N_planets) grid efficiently
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        # 4. Calculate RV per planet
        # Using the identity: cos(w + f) = cos(w)cos(f) - sin(w)sin(f)
        rv_matrix = K_grid * (cosw * cosf - sinw * sinf + ecc * cosw)

        return rv_matrix
