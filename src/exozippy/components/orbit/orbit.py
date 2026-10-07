import logging

import astropy.units as u
import numpy as np

logger = logging.getLogger(__name__)

import pymc as pm
import pytensor.tensor as pt
from exoplanet_core.pymc import ops as ops

from exozippy.components.component import Component, in_topology
from exozippy.components.parameter import Parameter
from exozippy.components.parameterization import (
    merge_options,
    mode_manifest,
    restrict_active,
)
from exozippy.config import user_entry
from exozippy.outputs.prose import get_collector, join_names
from exozippy.potentials import soft_lower_bound, soft_upper_bound

# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics
from .bodies import TAYLOR_TYPES, component_instance_names, parse_orbit_bodies

# Every legal `type:` of an orbit block.  `keplerian` is the default; the
# Taylor types are the low-order expansions of an orbit too long to resolve
# (orbit.md "Taylor orbits"); `nbody` is reserved for the integrator backend
# review 8.8.15 describes and is refused until it exists.
ORBIT_TYPES = ("keplerian",) + TAYLOR_TYPES + ("nbody",)


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
    constrained |= _astrometric_orbits(system, orbit)
    return constrained


def _astrometric_orbits(system, orbit):
    """Orbits whose sky motion an astrometric dataset measures.

    The astrometric half of `amplitude_constrained_orbits`, split out because
    `inclination_constrained_orbits` asks the same question: a sky-projected
    orbit measures the inclination as well as the amplitude, so both
    predicates must name the same orbits.
    """
    components = getattr(system, "active_components", None) or {}
    out = set()
    ast = components.get("astrometryinstrument")
    if ast is None:
        return out
    for i, mode in enumerate(ast.modes):
        if mode == "rel":
            if ast.rel_orbit[i] is not None:
                out.add(ast.rel_orbit[i])
        else:
            # gaia/abs photocenter wobble sums the orbits whose primary
            # group contains the target star.
            s = int(ast.config[i].get("star_ndx", 0))
            out.update(
                o for o, role in orbit.star_membership(s) if role == "primary"
            )
    return out


def inclination_constrained_orbits(system, orbit):
    """Orbits whose INCLINATION some dataset measures.

    The sibling of `amplitude_constrained_orbits`, and asked by one caller for
    one reason: `Planet._resolve_mass_parameterization` samples
    `(m sin i, cos i)` by default (`fitmsini`, review 2.14.9) exactly where
    the mass is measured but the inclination is not -- an RV-only orbit,
    where the data constrain `m sin i` and the mass is `m sin i / sin i`
    under the isotropic prior.  A module function beside the amplitude
    predicate so the two answers are kept together and cannot drift.

    What measures an inclination:

    * **a transit light curve** -- every transit file models every planet
      (the assumption `Orbit._transit_only` and `Planet._resolve_chen` make),
      so any `transit:` block names every orbit with a planet among its
      bodies.  This also covers the photometric phase-curve terms (thermal,
      reflection, ellipsoidal, beaming), which live in the transit model;
      `beam_constrains_mass` ties the beaming amplitude to `K`, i.e. to
      `m sin i`, so it adds nothing about `i` by itself.
    * **a Rossiter-McLaughlin (`rvinstrument` `rm:`) or Doppler-tomography
      (`dopptom` `orbit:`) dataset** -- both model the transit chord, which
      fixes the impact parameter;
    * **astrometry** of any mode (gaia/abs/rel) -- a sky-projected orbit;
    * **microlensing**: a lens companion's `orbital_motion: keplerian` and
      the event's `source_orbital_motion: keplerian` (xallarap) both consume
      the orbit's sky geometry per epoch;
    * **the user**: `sigma: 0` on the orbit's `cosi` or `inc` states the
      inclination outright (`Orbit._user_pinned`, the pin `fitchord`'s
      default already defers to).  `examples/hd80606` pins its published
      transit inclination this way; sampling m sin i there would only
      rescale the mass by a constant.

    Returns a set of orbit indices.
    """
    from ..dopptom.dopptom import dt_orbits_in_system
    from ..rm import rm_orbits_in_system

    constrained = set()
    if in_topology(system, "transit") is not None:
        constrained.update(
            i
            for i in range(orbit.n_elements)
            if any(t == "planet" for t, _ in orbit.bodies(i))
        )
    names = list(orbit.names)
    for ref in rm_orbits_in_system(system) | dt_orbits_in_system(system):
        if ref not in names:
            raise ValueError(
                f"[{orbit.prefix}] rm:/dopptom orbit reference {ref!r} names "
                f"no orbit block; defined orbits: {names}."
            )
        constrained.add(names.index(ref))
    constrained |= _astrometric_orbits(system, orbit)
    constrained |= orbit._lens_keplerian_orbits(system)
    constrained |= orbit._lens_xallarap_orbits(system)
    constrained.update(
        i
        for i in range(orbit.n_elements)
        if orbit._user_pinned(i, ("cosi", "inc"))
    )
    return constrained


def occultation_datasets(system, orbit):
    """Per orbit: the light curves that model its planet's OCCULTATION.

    The explicit switch is the BAND's `fitthermal:` / `fitreflect:` -- it
    gives the planet a free emission, so every transit file in that band
    fits the secondary eclipse, whose timing measures e cos omega and whose
    duration e sin omega.  Every transit light curve models every planet
    (the assumption `Orbit._transit_only` already makes), so such a file
    names every orbit with a planet in its companion group.

    Returns {orbit index: [dataset description, ...]} (empty lists where
    none).  A module function beside `amplitude_constrained_orbits` for the
    same reason that one is: it is a fact about the (system, orbit) pair.
    """
    components = getattr(system, "active_components", None) or {}
    out = {i: [] for i in range(orbit.n_elements)}
    transit = components.get("transit")
    band = components.get("band")
    if transit is None or band is None:
        return out
    emitting = {
        name
        for name, th, rf in zip(band.names, band.fitthermal, band.fitreflect)
        if th or rf
    }
    occ = [
        f"transit '{n}' (band '{b}' fits the occultation)"
        for n, b in zip(transit.names, transit.band_names)
        if b in emitting
    ]
    for i in range(orbit.n_elements):
        if any(t == "planet" for t, _ in orbit.companion_bodies[i]):
            out[i] = list(occ)
    return out


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

    # Stage-6 nodes stashed for stage 7 (see Component.per_build_caches):
    # cleared at the top of every build, so a REBUILD can never hand stage 7
    # the previous model's chord geometry or unclipped V_c/V_e root.  Being
    # None at stage 7 on an orbit that needs one is a bookkeeping bug and
    # raises (review 2.8.6).
    per_build_caches = ("_chord_geometry", "_vcve_unclipped_nodes")
    _chord_geometry = None
    _vcve_unclipped_nodes = None

    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)
        self.label = "Orbital Parameters"

        self.types = self._parse_types()
        self.primary_bodies, self.companion_bodies = parse_orbit_bodies(
            self.config, getattr(config_manager, "system_config", None)
        )
        self.i180 = [c.get("i180", False) for c in self.config]
        self._parse_ecc_parameterization()

        self._reject_wip_parameterizations()

    # Orbit-block keys that only mean something for a Keplerian orbit.  On a
    # Taylor orbit they would be silently inert, so they raise instead.
    _KEPLERIAN_ONLY_KEYS = ("fitvcve", "fitchord", "i180", "global_search")

    def _parse_types(self):
        """Read and validate each orbit block's `type:` (default keplerian).

        A Taylor orbit (`linear`, `quadratic`) is a different MODEL, not a
        coordinate choice -- it has a different parameter set -- so `type:`
        is a value, the way `band.ld_law` is, rather than a fit<x> boolean
        (components.md "Config flag vocabulary").  `epoch:` is the Taylor
        expansion's reference time and means nothing on a Keplerian orbit,
        whose epoch is `tc`; the Keplerian-only switches mean nothing on a
        Taylor one.  Either way round the key would be silently inert, so
        both raise.
        """
        types = []
        for i, c in enumerate(self.config):
            c = c or {}
            name = c.get("name", str(i))
            t = c.get("type", "keplerian")
            if t not in ORBIT_TYPES:
                raise ValueError(
                    f"[{self.prefix}.{name}] type: {t!r} is not an orbit "
                    f"type; expected one of {list(ORBIT_TYPES)} (default "
                    f"keplerian).  Orbit types are case-sensitive."
                )
            if t == "nbody":
                raise NotImplementedError(
                    f"[{self.prefix}.{name}] type: nbody is reserved for an "
                    f"N-body integrator backend that does not exist yet "
                    f"(review 8.8.15).  Use keplerian, or linear/quadratic "
                    f"for an orbit too long for the data to resolve."
                )
            if t in TAYLOR_TYPES:
                bad = [k for k in self._KEPLERIAN_ONLY_KEYS if k in c]
                if bad:
                    raise ValueError(
                        f"[{self.prefix}.{name}] {bad} only apply to a "
                        f"Keplerian orbit; a type: {t} orbit has no "
                        f"eccentricity, inclination or period to set them "
                        f"on.  Remove them."
                    )
            elif "epoch" in c:
                raise ValueError(
                    f"[{self.prefix}.{name}] 'epoch:' is the reference epoch "
                    f"of a Taylor orbit (type: linear or quadratic); a "
                    f"Keplerian orbit's epoch is its tc.  Remove the key or "
                    f"set the orbit's type."
                )
            types.append(t)
        return types

    @property
    def is_taylor(self):
        """Per orbit: is this a Taylor orbit (type linear or quadratic)?"""
        return np.array([t in TAYLOR_TYPES for t in self.types], dtype=bool)

    # The two eccentricity parameterizations, as a mode table (see
    # components/parameterization.py).  `hk` samples the sqrt(e)cos/sin(omega)
    # pair and derives (ecc, omega) from it; `vcve` samples V_c/V_e and an omega
    # direction vector, derives (ecc, omega) from those, and REPORTS the
    # sqrt(e)cos/sin pair (REPORTED) so both parameterizations produce the same
    # table rows and a user's prior on either survives the switch.  V_c/V_e
    # itself is reported on an `hk` orbit, for the same reason.  Orbits may
    # differ: element roles are per instance.
    ECC_MODE_TABLE = {
        "hk": {
            "secosw": None,
            "sesinw": None,
            "ecc": "default",
            "omega": "default",
            "tp_target": "default",
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
            "tp_target": "from_ecc",
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
                entry = user_entry(user, key)
                if entry is not None and entry.get("sigma") == 0:
                    pinned.append(param)
                    break
        return pinned

    def _pin_blocks_default(self, index, switch, log=True):
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
        if not log:
            return True
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
        # ONE rule, two triggers: an orbit something other than a transit
        # duration measures (e, omega) of keeps the conventional coordinates.
        # An RV/astrometric amplitude is the first; a fitted OCCULTATION (the
        # band's fitthermal/fitreflect) is the second, because its timing
        # pins e cos omega directly (JDE 2026-10-01: examples/gj1214 forced
        # onto V_c/V_e mixed slowly along the ridge that makes).
        constrained = set(amplitude_constrained_orbits(system, self))
        constrained |= {
            i for i, ds in occultation_datasets(system, self).items() if ds
        }
        return [i not in constrained for i in range(self.n_elements)]

    def _companion_planets(self, i):
        """The planet indices in orbit `i`'s companion group that EXIST.

        Existence is checked against the planet blocks the system config
        declares, because `parse_orbit_bodies`' implicit pairing names
        `planet.i` for an orbit no planet points at whether or not there is
        such a planet (review 2.8.8).  An explicit group naming a missing
        planet raises in `_validate_bodies`; the implicit one only disables the
        orbit's masses -- and before review 2.8.6 it also handed `fitchord` a
        planet that did not exist, so the chord was built from the all-zero
        `p`/`ar` placeholders of `_chord_context` and its Jacobian and
        geometry bound were evaluated on them, with no error.
        """
        sys_cfg = getattr(self.config_manager, "system_config", None) or {}
        n_planets = len(component_instance_names(sys_cfg, "planet"))
        return [
            idx
            for (t, idx) in self.companion_bodies[i]
            if t == "planet" and idx < n_planets
        ]

    def _chord_planet_indices(self):
        """Per orbit: the index of its one transiting planet, or -1.

        A chord is `sqrt((1 + p)^2 - b^2)`, so it needs a radius ratio -- one
        radius ratio.  An orbit whose companion group holds no planet (a
        stellar binary, or an implicit orbit no planet block points at) has
        none, and one holding SEVERAL has no single answer: two planets sharing
        an orbit have two different chords, and asking which one `orbit.chord`
        means is a question with no correct answer.  Both are `nochord`, and
        `_parse_inc_parameterization` refuses an explicit `fitchord: true` on
        them rather than picking a planet.
        """
        out = []
        for i in range(self.n_elements):
            planets = self._companion_planets(i)
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

    def _e_omega_datasets(self, system):
        """Per orbit: the datasets that measure (e, omega) more directly than
        a transit duration does.  {orbit index: [dataset description, ...]}.

        Two kinds.  An RV or astrometric amplitude (the same
        `amplitude_constrained_orbits` predicate the transit-only default
        asks, so the two can never disagree), which measures e cos/sin omega
        through the shape of the curve.  And an OCCULTATION: a transit file
        whose band fits a planetary emission (`fitthermal`/`fitreflect`)
        models the secondary eclipse, whose timing measures e cos omega and
        whose duration e sin omega directly.  Every transit light curve
        models every planet, so that dataset names every orbit with a
        planet.  Whether the file really COVERS the eclipse cannot be known
        before the fit (JDE 2026-10-01), which is why this is a warning and
        not a default.
        """
        components = getattr(system, "active_components", None) or {}
        out = {i: [] for i in range(self.n_elements)}
        rv = components.get("rvinstrument")
        if rv is not None:
            for k, s in enumerate(rv.star_ndx):
                for o, _ in self.star_membership(s):
                    out[o].append(f"rvinstrument '{rv.names[k]}'")
        ast = components.get("astrometryinstrument")
        if ast is not None:
            ast_orbits = amplitude_constrained_orbits(system, self)
            if rv is not None:
                rv_orbits = {
                    o for s in rv.star_ndx for o, _ in self.star_membership(s)
                }
                ast_orbits -= rv_orbits
            for o in ast_orbits:
                out[o].append(
                    "astrometryinstrument "
                    + ", ".join(f"'{n}'" for n in ast.names)
                )
        for i, ds in occultation_datasets(system, self).items():
            out[i].extend(ds)
        return out

    def _warn_vcve_where_e_is_measured(self, system):
        """WARN when an orbit samples V_c/V_e but other data measure e/omega.

        V_c/V_e is the right coordinate when a transit DURATION is what
        constrains the eccentricity (Eastman 2024).  When an occultation's
        timing or an RV curve measures e cos omega / e sin omega directly,
        V_c/V_e is the wrong one: the posterior is a narrow ridge in
        (V_c/V_e, omega) and the sampler mixes slowly along it --
        examples/gj1214 with `fitvcve: true` (its JWST eclipse pins e cos w)
        reached r_hat 1.3 on omega at 4 x (400 + 400) where sqrt(e)cos/sin
        omega converges in 3 minutes (review 1.8.14, JDE 2026-10-01).  The
        same datasets keep the transit-only DEFAULT off (`_transit_only`), so
        this fires only for an explicit `fitvcve: true`, which is honored --
        once per orbit, naming the data.
        """
        if system is None:
            return
        datasets = self._e_omega_datasets(system)
        for i in range(self.n_elements):
            if not self.fitvcve[i] or not datasets[i]:
                continue
            logger.warning(
                "[%s] orbit '%s' samples V_c/V_e (fitvcve), but %s also "
                "measure(s) its eccentricity and argument of periastron "
                "directly.  V_c/V_e is the parameterization for a transit "
                "duration alone; with these data sqrt(e)cos(omega)/"
                "sqrt(e)sin(omega) is the better one -- set 'fitvcve: false' "
                "on the orbit (it mixes slowly along the (V_c/V_e, omega) "
                "ridge otherwise).",
                self.prefix,
                self.names[i],
                "; ".join(datasets[i]),
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
        self.inc_modes = self._inc_modes_for(self.fitchord, system)

    def _inc_modes_for(self, fitchord, system):
        """Per-orbit inclination mode names for the given `fitchord` list.

        The body of `_parse_inc_parameterization`, side-effect free so that
        `inclination_modes` can answer the same question for another
        component without touching this one's state.
        """
        chord_planet = self._chord_planet_indices()
        # in_topology, not a bare active_components lookup: this file asked
        # the same question three ways and they disagreed about whether a
        # config-only system counts (review 4.8.1).  It does -- the local
        # _topology helper this replaced said so, and a topology-driven
        # DEFAULT must not depend on whether the component happens to be
        # built yet.
        has_transit = in_topology(system, "transit") is not None
        modes = []
        for i, on in enumerate(fitchord):
            name = self.names[i] if i < len(self.names) else i
            if chord_planet[i] >= 0 and not (on or has_transit):
                # A real planet, but nothing that could see a transit and no
                # request to sample it: the chord is arithmetic, not a result.
                modes.append("nochord")
                continue
            if chord_planet[i] < 0:
                if bool((self.config[i] or {}).get("fitchord", False)):
                    n_planets = len(self._companion_planets(i))
                    raise ValueError(
                        f"[{self.prefix}.{name}] 'fitchord: true' needs "
                        f"exactly one planet on the orbit -- the chord is "
                        f"sqrt((1 + R_P/R_*)^2 - b^2), so it is defined by a "
                        f"radius ratio -- and this orbit's companion group "
                        f"holds {n_planets}"
                        + (
                            " (no planet block's orbit_ndx points at it)"
                            if n_planets == 0
                            and "companion" not in (self.config[i] or {})
                            else ""
                        )
                        + ".  Sample 'cosi' here (drop the key), or split "
                        "the bodies onto their own orbits."
                    )
                modes.append("nochord")
            else:
                modes.append("chord" if on else "cosi")
        return modes

    def inclination_modes(self, system):
        """Per orbit, the inclination mode stage 3 resolves (`cosi`, `chord`
        or `nochord`), computed without side effects.

        For a component that has to know the answer at ITS stage 3, which may
        run before this orbit's (stage 3 follows the user's config key
        order): `Planet._resolve_mass_parameterization` refuses `fitmsini` on
        a chord orbit, where `m = msini / sin i` and `cos i(chord, a/R*)`
        would derive each other.  The same two functions this orbit's own
        stage 3 calls, on the same inputs, so the two answers cannot differ;
        only the logging is suppressed (the orbit logs its own choice once).
        """
        _, fitchord = self._resolve_switches(system, log=False)
        return self._inc_modes_for(fitchord, system)

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
        self.fitvcve, self.fitchord = self._resolve_switches(system)
        self.ecc_modes = ["vcve" if on else "hk" for on in self.fitvcve]

    def _resolve_switches(self, system, log=True):
        """``(fitvcve, fitchord)`` per orbit -- the body of
        `_parse_ecc_parameterization`, side-effect free apart from the
        `_pin_blocks_default` log line (suppressed with ``log=False``)."""
        default_on = (
            self._transit_only(system)
            if system is not None
            else [False] * self.n_elements
        )
        # A Taylor orbit has no eccentricity at all: its (inactive) Keplerian
        # entries stay on the plain sqrt(e) mode, so no V_c/V_e Jacobian or
        # root mixture is ever built for it.
        default_on = [
            on and not taylor for on, taylor in zip(default_on, self.is_taylor)
        ]
        fitvcve = []
        for i, c in enumerate(self.config):
            asked = c.get("fitvcve")
            if asked is not None:
                fitvcve.append(bool(asked))
                continue
            on = default_on[i] and not self._pin_blocks_default(
                i, "fitvcve", log=log
            )
            fitvcve.append(on)
        fitchord = []
        for i, c in enumerate(self.config):
            asked = c.get("fitchord")
            if asked is not None:
                fitchord.append(bool(asked))
                continue
            # Follows fitvcve, which is what "unless separately set" means --
            # and is subject to its own pin check, since a fixed inclination is
            # a decision the chord would drop just as surely.
            on = fitvcve[i] and not self._pin_blocks_default(
                i, "fitchord", log=log
            )
            fitchord.append(on)
        return fitvcve, fitchord

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
                "key": "type",
                "kind": "option",
                "accepts": list(ORBIT_TYPES),
                "required": False,
                "doc": (
                    "What kind of orbit this is.  'keplerian' (default): a "
                    "two-body Keplerian arc.  'linear' / 'quadratic': an "
                    "orbit too long for the data to resolve, written as the "
                    "first (second) time derivatives of what its consumers "
                    "measure about 'epoch:' -- the primary star's RV slope "
                    "gammadot (and curvature gammaddot), or a lens "
                    "companion's ds_dt/dalpha_dt -- with no period, "
                    "eccentricity, mass or K.  Needs an explicit 'primary:'; "
                    "'companion:' may be omitted for an unseen body.  "
                    "'nbody' is reserved and not implemented."
                ),
            },
            {
                "key": "epoch",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Taylor orbits only: the BJD_TDB the expansion is "
                    "about.  Default: the midpoint of the primary star's RV "
                    "span (EXOFASTv2's RVEPOCH); a lens-read orbit is always "
                    "anchored at the event's t0_par."
                ),
            },
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
                # 'peak_find:' lives on the mulensevent block.  The orbit component
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
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Sample V_c/V_e and the direction of omega instead of "
                    "sqrt(e)cos(omega)/sqrt(e)sin(omega) (Eastman 2024), "
                    "with the likelihood marginalized over both roots of the "
                    "inversion and a Jacobian keeping the prior uniform in e "
                    "and omega.  Default: on for an orbit that only transits "
                    "measure (together with fitchord), off otherwise.  "
                    "'fitvcve: false' also turns fitchord off unless fitchord "
                    "is set explicitly."
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
        KEPLERIAN orbits containing star star_idx, as [(orbit_index, role),
        ...] with role 'primary' or 'companion'.  Used by instruments to
        decide which orbits move (or blend with) a given star.

        Taylor orbits are deliberately NOT here: every caller of this
        projects Keplerian elements, which a Taylor orbit does not have.  A
        consumer that can use one asks `taylor_membership`, and one that
        cannot is refused at stage 3 by `_taylor_consumers` -- never left to
        drop the orbit in silence.
        """
        return self._membership(star_idx, taylor=False)

    def taylor_membership(self, star_idx):
        """`star_membership`'s twin over the TAYLOR orbits."""
        return self._membership(star_idx, taylor=True)

    def _membership(self, star_idx, taylor):
        out = []
        key = ("star", int(star_idx))
        is_taylor = self.is_taylor
        for i in range(self.n_elements):
            if bool(is_taylor[i]) != taylor:
                continue
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

    @property
    def circular_orbits(self):
        """Per orbit: is the eccentricity STRUCTURALLY zero? (review 6.8.2)

        True only where the sqrt(e) pair is PINNED at zero -- `sigma: 0` with
        `initval: 0` on both `secosw` and `sesinw`, which is how a circular
        fit is written (`examples/kelt17`).  Deliberately not "e is small at
        the start": an unpinned eccentricity can move, and treating it as
        circular would silently give the whole run the wrong RV phase.

        Conservative everywhere it cannot be sure -- a V_c/V_e orbit (whose
        circular case is a different pin, on a coordinate whose inversion is
        singular exactly there), a parameter that has not been resolved, or
        anything unpinned -- because the cost of being wrong is a wrong
        model and the cost of being conservative is the Kepler solve we
        were paying anyway.
        """
        n_el = self.n_elements
        circ = np.ones(n_el, dtype=bool)
        for name in ("secosw", "sesinw"):
            if name not in self.manifest:
                return np.zeros(n_el, dtype=bool)
            cfg = self.config_manager.resolve(
                self.prefix, name, shape=(n_el,), names=self.names
            )
            sigma = np.atleast_1d(np.asarray(cfg.get("sigma"), dtype=float))
            initval = np.atleast_1d(
                np.asarray(cfg.get("initval"), dtype=float)
            )
            if sigma.size != n_el or initval.size != n_el:
                return np.zeros(n_el, dtype=bool)
            circ &= (sigma == 0.0) & (initval == 0.0)
        modes = np.atleast_1d(
            np.asarray(getattr(self, "ecc_modes", []), dtype=object)
        )
        if modes.size == n_el:
            circ &= modes == "hk"
        return circ

    def _all_circular(self, orbit_map=None):
        """True when EVERY orbit `solve_kepler` is about to see is circular.

        All-or-nothing, deliberately: the saving comes from not BUILDING the
        Newton solve, and a mixed vector still has to build it for the
        eccentric columns.  Splitting the vector, solving the eccentric
        subset and reassembling with `set_subtensor` would save arithmetic
        and cost a graph that no longer matches the simple one -- not worth
        it until someone measures a system where it matters.
        """
        circ = self.circular_orbits
        if orbit_map is not None:
            try:
                idx = np.atleast_1d(np.asarray(orbit_map, dtype=int))
            except (TypeError, ValueError):
                # A SYMBOLIC map. Which orbits it selects is not knowable at
                # graph-build time, so the structural claim cannot be made and
                # the honest answer is "build the full solve". Declining the
                # fast path is always correct -- it is an optimization, not a
                # semantic -- whereas guessing would solve an eccentric orbit
                # as if it were circular, silently and with a wrong RV phase.
                # Production does not take this branch (rvinstrument's
                # _orbit_rv_terms returns a numpy array), but the signature
                # permits a symbolic map and callers pass one.
                return False
            if idx.size == 0:
                return False
            circ = circ[idx]
        return bool(np.all(circ)) and circ.size > 0

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

    def _user_seeded_initval(self, param):
        """Per orbit: did the USER write an `initval` for `param`?

        `_resolve_initval` cannot answer this -- it returns the defaults.yaml
        value too, and for `tc` that backstop is always a number.  Reads the
        user's own entries in the two spellings that survive
        `standardize_param_names`, exactly as `_user_pinned` does.
        """
        user = getattr(self.config_manager, "user_params", None) or {}
        seeded = np.zeros(self.n_elements, dtype=bool)
        entry = user_entry(user, f"{self.prefix}.{param}")
        if entry is not None and entry.get("initval") is not None:
            seeded[:] = True
        for i in range(self.n_elements):
            entry = user_entry(user, f"{self.prefix}.{i}.{param}")
            if entry is not None and entry.get("initval") is not None:
                seeded[i] = True
        return seeded

    def _seeded_sqrte_pair(self, shape):
        """(secosw, sesinw) per orbit implied by the stage-3 seeds.

        Prefers an explicit `ecc` + `omega` pair -- the spelling the RV
        literature uses -- and falls back to the sqrt(e) pair itself (user
        entry or defaults.yaml).  Stage 3 again: the relaxation engine, which
        normally reconciles the two spellings, has not run, so this is the
        same job `_seeded_period` does for the period.
        """
        sc0 = self._resolve_initval("secosw", shape)
        ss0 = self._resolve_initval("sesinw", shape)
        factor_om = (
            self.config_manager.get_conversion_factor(self.prefix, "omega")
            or 1.0
        )
        om = self._resolve_initval("omega", shape) * factor_om
        e_u = self._resolve_initval("ecc", shape)
        have_ew = ~np.isnan(om) & ~np.isnan(e_u)
        sc0 = np.where(have_ew, np.sqrt(np.abs(e_u)) * np.cos(om), sc0)
        ss0 = np.where(have_ew, np.sqrt(np.abs(e_u)) * np.sin(om), ss0)
        return sc0, ss0

    def _seeded_ecc_omega(self, shape):
        """(e, omega in radians) per orbit from `_seeded_sqrte_pair`.

        Same ceiling `calc_ecc` applies, for the same reason: both callers
        feed the result to a Kepler solve (a forward-model evaluation), not
        to a bound.
        """
        sc0, ss0 = self._seeded_sqrte_pair(shape)
        ecc0 = np.clip(sc0**2 + ss0**2, 0.0, physics.MAX_ECC)
        return ecc0, np.arctan2(ss0, sc0)

    def _seeded_tc(self, shape):
        """Per-orbit time of conjunction in days implied by the stage-3 seeds.

        `tc`'s hard window is `tc_init +/- P/2` and it is declared HERE, at
        stage 3 -- before the relaxation engine can turn a user's time of
        PERIASTRON into a time of conjunction (the one-way solver of review
        8.1.1).  Reading tc's own resolved initval alone therefore returns the
        defaults.yaml backstop 2460000 for a params file that seeds `tp:`
        instead, and the tc the engine goes on to solve lands hundreds of
        thousands of days outside its own window: a fatal out-of-bounds start,
        naming a parameter the user never wrote.  Exactly the shape of the
        `_seeded_period` bug `tests/test_orbit_tc_window.py` covers.

        Only where the user did not seed `tc` themselves: an explicit `tc` is
        PRECEDENCE_USER, the solver stands down for it, and so must the window.
        """
        tc = self._resolve_initval("tc", shape)
        tp = self._resolve_initval("tp", shape)
        use_tp = ~np.isnan(tp) & ~self._user_seeded_initval("tc")
        if not use_tp.any():
            return tc
        ecc0, w0 = self._seeded_ecc_omega(shape)
        implied = physics.tc_from_tp(tp, ecc0, w0, self._seeded_period(shape))
        return np.where(use_tp, implied, tc)

    # Which conjunction the sampler moves, per orbit (orbit.md, "tc is
    # SAMPLED near the data").  `user`: the one the user's seed names, which
    # is already within half a period of the data's time center -- exactly
    # the graph every orbit had before (JDE 2026-10-07).  `shifted`: the
    # conjunction `tc_epoch` whole periods later, near the data, with `tc` at
    # the user's epoch derived from it.
    EPOCH_MODE_TABLE = {
        "user": {"tc": None},
        "shifted": {
            "tc_sampled": None,
            "tc": {"expr_key": "from_sampled", "force_node": True},
        },
    }

    def _user_fixes_tc_support(self):
        """Per orbit: did the user pin, bound or link `tc` itself?

        Each of those is a statement about the SAMPLED coordinate -- `sigma:
        0` holds it, `lower`/`upper` are its hard support, a link ties its
        value -- and each would silently lose its meaning on a derived `tc`
        (a pin on a derived element is dropped, a bound becomes a soft
        barrier, a link refuses an expression).  Such an orbit keeps the
        user's epoch as the sampled one.  A Gaussian `mu`/`sigma` is not in
        this list: it means the same thing on the derived `tc`.
        """
        user = getattr(self.config_manager, "user_params", None) or {}
        fixed = np.zeros(self.n_elements, dtype=bool)
        for i in range(self.n_elements):
            for key in (f"{self.prefix}.{i}.tc", f"{self.prefix}.tc"):
                entry = user_entry(user, key)
                if entry is None:
                    continue
                if (
                    entry.get("sigma") == 0
                    or entry.get("lower") is not None
                    or entry.get("upper") is not None
                ):
                    fixed[i] = True
        for per_elem in self.config_manager.get_element_links(
            self.prefix, "tc"
        ).values():
            for i in per_elem:
                fixed[int(i)] = True
        return fixed

    def _sampling_epochs(self, system, tc_seed, period):
        """Per orbit: how many whole periods to move the seeded `tc` to
        reach the data, and the data's time center.

        The center is the weighted mean of the epochs the components' data
        put on the orbit (`Component.epochs_constraining`, built on the same
        membership maps as `amplitude_constrained_orbits`) -- those of the
        SHARPEST timing rank present only (`epoch_timing_rank`: eclipses over
        RVs and astrometry).  A seed outside the span of those epochs (by
        more than half a period) is shifted by `round((center - tc_seed) /
        period)`, so the sampled conjunction lies within half a period of
        the center, where `tc` and the period are far less correlated than
        at the seed.  Zero for a seed inside the span (the optimal epoch is
        inside it too, and the center is no better a guess than the user's
        own), for an orbit no dataset times (nothing to be near) and for one
        whose `tc` the user pinned, bounded or linked
        (`_user_fixes_tc_support`).

        Returns ``(epochs, centers)``: an int array and a float array (NaN
        where no data), one entry per orbit.
        """
        n = self.n_elements
        times = [[] for _ in range(n)]
        weights = [[] for _ in range(n)]
        ranks = np.full(n, -np.inf)
        components = getattr(system, "active_components", None) or {}
        for comp in components.values():
            rank = comp.epoch_timing_rank
            for o, pairs in comp.epochs_constraining(system, self).items():
                if rank < ranks[o]:
                    continue
                if rank > ranks[o]:
                    ranks[o] = rank
                    times[o], weights[o] = [], []
                for t, w in pairs:
                    times[o].append(np.asarray(t, dtype=float))
                    weights[o].append(np.asarray(w, dtype=float))
        centers = np.full(n, np.nan)
        spans = np.full((n, 2), np.nan)
        for o in range(n):
            if times[o]:
                t = np.concatenate(times[o])
                w = np.concatenate(weights[o])
                centers[o] = float(np.sum(w * t) / np.sum(w))
                spans[o] = (float(t.min()), float(t.max()))
        epochs = np.zeros(n, dtype=int)
        fixed = self._user_fixes_tc_support()
        for o in range(n):
            if np.isnan(centers[o]) or fixed[o]:
                continue
            # A seed INSIDE the span of the epochs that time the orbit (to
            # within half a period) stays where the user put it: the optimal
            # epoch lies inside that span too, and the equal-weight center
            # is only a guess at it -- measured, a worse one than an in-span
            # seed on examples/kelt17 (corr(tc, P) -0.70 at the center
            # against -0.41 at the seed) and examples/gj1214 (-0.20 against
            # -0.06).  Only a seed OUTSIDE the span, beyond the optimum by
            # construction, is moved -- to the center, which for a seed far
            # outside (kelt4's TESS-era seed on its RVs) is far closer.
            half = 0.5 * period[o]
            if spans[o, 0] - half <= tc_seed[o] <= spans[o, 1] + half:
                continue
            epochs[o] = int(np.round((centers[o] - tc_seed[o]) / period[o]))
        return epochs, centers

    def register_parameters(self, system):
        """Stage 3: Calculate window constraints and declare the manifest."""
        shape = (self.n_elements,)

        # Taylor orbits first: which consumer reads each one decides its
        # parameters, and an unsupported consumer must fail before anything
        # is built against an orbit it cannot read.
        self._taylor_consumer_map = self._taylor_consumers(system)

        # 1. Peer into the config (Pre-flight windows)
        # tc is periodic (tc and tc + P are the same solution), so one full
        # period is the right hard window -- but it must be centred on the tc
        # the user actually seeded and scaled by the period they actually
        # seeded, in every legal spelling of each.  See _seeded_tc (which
        # covers a `tp:` seed) and _seeded_period.
        tc_init = self._seeded_tc(shape)
        period_init = self._seeded_period(shape)
        half_period = period_init / 2.0

        # WHICH conjunction is sampled (JDE 2026-10-07): the one nearest the
        # data's time center, `tc_epoch` whole periods from the user's seed.
        # An orbit already within half a period of it (tc_epoch == 0) keeps
        # exactly the manifest it always had; a shifted one samples
        # `tc_sampled` in the same one-period window around the moved seed,
        # and derives `tc` at the user's epoch from it.  orbit.md, "tc is
        # SAMPLED near the data".
        self.tc_epoch, self.data_time_center = self._sampling_epochs(
            system, tc_init, period_init
        )
        shifted = self.tc_epoch != 0
        tc_sampled_init = tc_init + self.tc_epoch * period_init
        self.epoch_modes = ["shifted" if s else "user" for s in shifted]
        epoch_entries = mode_manifest(
            self.epoch_modes,
            self.EPOCH_MODE_TABLE,
            n_elements=self.n_elements,
            options={
                # -inf/+inf on a DERIVED element: no barrier (a bound nobody
                # stated is no bound), where NaN would fall through to the
                # defaults.yaml range and add one.
                "tc": {
                    "force_node": True,
                    "lower": np.where(shifted, -np.inf, tc_init - half_period),
                    "upper": np.where(shifted, np.inf, tc_init + half_period),
                },
                "tc_sampled": {
                    "lower": tc_sampled_init - half_period,
                    "upper": tc_sampled_init + half_period,
                },
            },
            where=f"{self.prefix} sampled epoch",
        )
        for i in np.flatnonzero(shifted):
            # The start of the moved conjunction: the same physical orbit as
            # the seed (tc_sampled - tc_epoch * P == tc_seed at the seeded
            # P), so the build start is the model the user seeded.
            self.config_manager.add_hint(
                f"{self.prefix}.{i}.tc_sampled", float(tc_sampled_init[i])
            )
            logger.info(
                "[%s.%s] tc is sampled %d period(s) from the seeded epoch, "
                "at %.6f (the data's time center is %.6f); tc is reported "
                "at the seeded epoch and t0 at the epoch the posterior "
                "prefers.",
                self.prefix,
                self.names[i],
                int(self.tc_epoch[i]),
                float(tc_sampled_init[i]),
                float(self.data_time_center[i]),
            )

        self.manifest = {
            "logP": None,
            "period": {"force_node": True, "expr_key": "default"},
            "n": "default",
        }
        if "tc_sampled" in epoch_entries:
            self.manifest["tc_sampled"] = epoch_entries["tc_sampled"]
        self.manifest["tc"] = epoch_entries["tc"]

        # Re-read the switches, NOW with the system: this is the pass that
        # decides, because the transit-only default is a question about the
        # data topology and __init__ cannot see it.  (It is also a re-read for
        # the older reason: `fitvcve` is a plain attribute anyone could set
        # between construction and stage 3.)
        self._parse_ecc_parameterization(system)
        self._reject_wip_parameterizations()
        self._log_parameterization_choices(system)
        self._warn_vcve_where_e_is_measured(system)

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
        for key in ("esinw", "ecosw", "tp_target"):
            self.manifest[key] = ecc_entries[key]
        # The occultation time (review 8.8.7).  Here rather than on `planet`
        # because every input is an orbit parameter and `tc` is one of them;
        # after `tp_target` because it is the other Kepler-timing output.
        self.manifest["ts_target"] = {
            "expr_key": "default",
            "force_node": True,
        }
        # The frame twins (see defaults.yaml): `tc_target` (target-frame
        # conjunction, the epoch every Kepler solve descends from) and the
        # OBSERVED `ts`/`tp`.  Declared here with the `no_bodies` identity
        # expression; the masses branch below upgrades them to the real
        # light-travel shift once it knows the orbit has an `a`.
        for key in self._LTT_REPORT_PARAMS:
            self.manifest[key] = {"expr_key": "no_bodies", "force_node": True}
        # EXOFASTv2's T_0, the conjunction at the epoch least correlated with
        # the period: built at the SAMPLED epoch, moved after sampling to the
        # one the posterior prefers (defaults.yaml `optimal_epoch:`).
        self.manifest["t0"] = "default"
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

            # The frame twins get their real expression: the closed-form
            # light-travel shift -z(t_event)*factor/c (physics.calc_tc_target
            # and friends) needs a/m_primary/m_companion/m_total, so only an
            # orbit with masses can compute it; `_ltt_reporting_mask` then
            # decides PER ORBIT whether any consumer actually retards this
            # orbit's geometry -- where none does the shift is zero and the
            # twin equals its partner exactly.  JDE 2026-09-21/22: report
            # both frames, BJD_TDB is the headline, the plain names (tc, ts,
            # tp) are BJD_TDB and the target frame is `*_target`.
            self._ltt_report_mask = self._ltt_reporting_mask(system)
            for key in self._LTT_REPORT_PARAMS:
                self.manifest[key] = {
                    "expr_key": "default",
                    "force_node": True,
                }

        # Rossiter-McLaughlin / Doppler tomography: declare the spin-orbit
        # params only on the orbit ELEMENTS some rvinstrument `rm:` key or
        # dopptom dataset actually targets -- a system-wide switch would
        # hand every other orbit a likelihood-free sampled pair.  Samples
        # the decorrelated sqrt(vsini)cos/sin(lambda) pair and derives
        # vsini/lam from them (mirrors the secosw/sesinw -> ecc/omega
        # idiom above).  With every orbit targeted (the common single-
        # orbit case) mode_manifest returns exactly the plain entries this
        # block used to hand-write, so those graphs are unchanged.
        from ..dopptom.dopptom import dt_orbits_in_system
        from ..rm import rm_orbits_in_system

        spin_targets = rm_orbits_in_system(system) | dt_orbits_in_system(
            system
        )
        taylor_targets = sorted(
            nm
            for nm in spin_targets
            if nm in self.names and self.is_taylor[list(self.names).index(nm)]
        )
        if taylor_targets:
            raise NotImplementedError(
                f"[{self.prefix}] rm:/dopptom name Taylor orbit(s) "
                f"{taylor_targets}; the Rossiter-McLaughlin and Doppler "
                f"tomography models need a transit geometry, which a "
                f"type: linear/quadratic orbit does not have."
            )
        unknown_targets = spin_targets - set(self.names)
        if unknown_targets:
            raise ValueError(
                f"[{self.prefix}] rm:/dopptom orbit reference(s) "
                f"{sorted(unknown_targets)} name no orbit block; defined "
                f"orbits: {list(self.names)}."
            )
        if spin_targets:
            self.manifest.update(
                mode_manifest(
                    [
                        "spinorbit" if nm in spin_targets else "plain"
                        for nm in self.names
                    ],
                    {
                        "spinorbit": {
                            "svcoslam": None,
                            "svsinlam": None,
                            "vsini": "from_sv",
                            "lam": "from_sv",
                        },
                        "plain": {},
                    },
                    # force_node ONLY on the partial-active path: there
                    # the selector machinery does not track a derived
                    # parameter as a Deterministic by default, so
                    # vsini/lam dropped out of the point dict and every
                    # RM plot silently fell back to lam = 0 (caught by
                    # test_model_builder_parity's rm_split tests).  On
                    # the every-orbit-targeted path the default tracking
                    # already builds the nodes, and forcing them there
                    # would ADD trace data_vars master never wrote --
                    # gated so those graphs and traces stay identical.
                    options=(
                        {
                            "vsini": {"force_node": True},
                            "lam": {"force_node": True},
                        }
                        if len(spin_targets) < len(self.names)
                        else None
                    ),
                    where="orbit spin-orbit (rm/dopptom targets)",
                )
            )

        # Astrometry constrains the longitude of the ascending node and
        # breaks the i <-> 180-i degeneracy, so sample the node direction
        # vector (xbigomega, ybigomega; each N(0,1) -> uniform marginal on
        # bigomega, like the microlensing trajectory angle alpha) and allow
        # the full inclination range when an astrometry component is active.
        # A lens block driving its geometry from an orbit (orbital_motion:
        # keplerian, conventions.md C24) measures BOTH the same way: the
        # sky rotation sense of the binary axis is sign(cos i) once the
        # node is fixed, and the axis's position angle is the node -- it is
        # the ONLY effect here that measures them for a lens binary
        # (review 8.6.8 5e).
        # A xallarap orbit (a lens block's `source_orbit:`, C25) is
        # astrometry-like too: the source's sky track enters the
        # trajectory, so bigomega is measurable -- but unlike the
        # lens-geometry case it stays node-DEGENERATE (the track, like all
        # astrometry, is invariant under the sky-plane reflection); see
        # _node_degenerate_orbits.
        has_astrometry = (
            in_topology(system, "astrometryinstrument") is not None
            or bool(self._lens_keplerian_orbits(system))
            or bool(self._lens_xallarap_orbits(system))
        )
        if has_astrometry:
            self.manifest["xbigomega"] = None
            self.manifest["ybigomega"] = None
            self.manifest["bigomega"] = "default"

            # The (bigomega, omega) <-> (bigomega+180, omega+180)
            # transformation is a reflection through the sky plane
            # (z -> -z): invisible to ANY astrometry, absolute or relative.
            # Only radial information (RVs) identifies the ascending node.
            # PER ORBIT, not system-wide: an RV-constrained orbit in a mixed
            # system is not degenerate at all, and treating it as if it were
            # cost it a table note it did not deserve.
            self.node_degenerate = self._node_degenerate_orbits(system)
            if self.node_degenerate.any():
                self._declare_node_degeneracy()

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
        # One entry per orbit block by construction (__init__), so read
        # directly: a length mismatch here used to be papered over with
        # all-False, silently flipping a chord orbit's cos i sign (2.8.6).
        own_i180 = np.asarray(self.i180, dtype=bool)
        self._chord_sign = np.where(own_i180, -1.0, 1.0)
        if has_astrometry:
            chord_orbits = [
                self.names[i]
                for i, m in enumerate(self.inc_modes)
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

        # Taylor orbits (type: linear | quadratic).  Every Keplerian entry
        # above is INACTIVE on them -- not derived from anything, not
        # reported: a Taylor orbit must self-declare that it has no period,
        # eccentricity, mass or K (review 8.8.14, "inactive must never be
        # reported") -- and the Taylor coefficients its consumers read are
        # active on them alone.  A system with no Taylor orbit takes neither
        # branch, so its manifest is exactly what it always was.
        if self.is_taylor.any():
            keplerian = ~self.is_taylor
            if not keplerian.any():
                # Nothing Keplerian at all (an RV trend alone): a parameter
                # no instance has is not a parameter of this system, so it
                # is omitted rather than declared wholly inactive -- the
                # rule mode_manifest applies, and the reason every
                # Keplerian consumer below has nothing to read.
                self.manifest = {}
            for key in list(self.manifest):
                self.manifest[key] = restrict_active(
                    self.manifest[key], keplerian, self.n_elements
                )
            self.taylor_epoch = self._taylor_epochs(
                system, self._taylor_consumer_map
            )
            self.manifest.update(
                self._taylor_manifest(self._taylor_consumer_map)
            )
            self._hint_rv_trends(system, self._taylor_consumer_map)
        else:
            self.taylor_epoch = np.full(self.n_elements, np.nan)

    # Which Taylor coefficients each consumer reads, per orbit type.  A
    # coefficient is active on an orbit exactly where some consumer of that
    # orbit reads it, the way bigomega exists only where astrometry does.
    TAYLOR_COEFFS = {
        "rv": {
            "linear": ("gammadot",),
            "quadratic": ("gammadot", "gammaddot"),
        },
        "lens": {"linear": ("ds_dt", "dalpha_dt")},
    }

    def _taylor_consumers(self, system):
        """{taylor orbit index: set of consumer kinds} -- "rv" and/or "lens".

        RAISES, naming the orbit, for every consumer that would otherwise
        read a Taylor orbit as if it were Keplerian or drop it in silence:
        astrometry, transits (a planet whose orbit_ndx points at one),
        Rossiter-McLaughlin/Doppler tomography (refused where the spin
        targets are read), xallarap, an RV star in the COMPANION group (its
        trend is the primary's scaled by a mass ratio no Taylor orbit has),
        and a quadratic lens orbit (the magnification backends take first
        derivatives only).  A Taylor orbit that NOTHING reads raises too: its
        coefficients would be free dimensions no likelihood term constrains.
        """
        consumers = {int(o): set() for o in np.nonzero(self.is_taylor)[0]}
        if not consumers:
            return consumers
        components = getattr(system, "active_components", None) or {}

        def name(o):
            return self.names[o]

        rv = components.get("rvinstrument")
        if rv is not None:
            for s in sorted(set(rv.star_ndx)):
                for o, role in self.taylor_membership(s):
                    if role != "primary":
                        raise NotImplementedError(
                            f"[{self.prefix}.{name(o)}] the RV star (star "
                            f"{s}) is in this Taylor orbit's COMPANION "
                            f"group.  Its trend would be the primary's "
                            f"scaled by -m_primary/m_companion, a mass ratio "
                            f"a type: {self.types[o]} orbit does not have; "
                            f"put the observed star in the primary group."
                        )
                    consumers[o].add("rv")

        for o in self._lens_taylor_orbits(system):
            if self.types[o] not in self.TAYLOR_COEFFS["lens"]:
                raise NotImplementedError(
                    f"[{self.prefix}.{name(o)}] a lens companion moves on "
                    f"this type: {self.types[o]} orbit, but the "
                    f"magnification backends take first derivatives only "
                    f"(ds_dt, dalpha_dt; conventions.md C24).  Use type: "
                    f"linear, or keplerian."
                )
            consumers[o].add("lens")

        for o in self._lens_xallarap_orbits(system):
            if o in consumers:
                raise NotImplementedError(
                    f"[{self.prefix}.{name(o)}] the event's source_orbit is "
                    f"a type: {self.types[o]} orbit; linear xallarap is "
                    f"deliberately not supported (conventions.md C25, "
                    f"review 8.6.9) -- a linear source drift is absorbed by "
                    f"t_E/t_0/u_0/alpha.  Use a keplerian source orbit."
                )

        planet_cfgs = (
            getattr(self.config_manager, "system_config", None) or {}
        ).get("planet") or []
        if not isinstance(planet_cfgs, list):
            planet_cfgs = [planet_cfgs]
        for j, pc in enumerate(planet_cfgs):
            o = int((pc or {}).get("orbit_ndx", 0))
            if o in consumers:
                raise ValueError(
                    f"[{self.prefix}.{name(o)}] planet "
                    f"{(pc or {}).get('name', j)!r} has orbit_ndx {o}, a "
                    f"type: {self.types[o]} orbit.  A planet's transit and "
                    f"derived geometry need a Keplerian orbit; point its "
                    f"orbit_ndx at one."
                )

        ast = components.get("astrometryinstrument")
        if ast is not None:
            touched = set()
            for i, mode in enumerate(ast.modes):
                if mode == "rel":
                    o = ast.rel_orbit[i]
                    if o is None:
                        continue
                    if o in consumers:
                        touched.add(o)
                    group = self.primary_bodies[o] + self.companion_bodies[o]
                    stars = {idx for t, idx in group if t == "star"}
                else:
                    stars = {int(ast.config[i].get("star_ndx", 0))}
                for t in consumers:
                    if stars & {
                        idx for ty, idx in self.bodies(t) if ty == "star"
                    }:
                        touched.add(t)
            if touched:
                raise NotImplementedError(
                    f"[{self.prefix}] astrometry measures a star on Taylor "
                    f"orbit(s) {[name(o) for o in sorted(touched)]}; the "
                    f"sky-plane Taylor terms are not implemented for "
                    f"astrometryinstrument yet (review 8.8.14)."
                )

        unread = [name(o) for o, c in consumers.items() if not c]
        if unread:
            raise ValueError(
                f"[{self.prefix}] nothing in the model reads Taylor orbit(s) "
                f"{unread}: a type: linear/quadratic orbit is measured by "
                f"the RVs of a star in its PRIMARY group, or by a lens "
                f"companion whose `orbit:` names it.  Its coefficients would "
                f"be free dimensions no likelihood term constrains."
            )
        return consumers

    def _taylor_epochs(self, system, consumers):
        """Per orbit: the Taylor expansion's reference epoch (BJD_TDB), NaN
        on a Keplerian orbit.

        A lens-read orbit is anchored at the event's t0_par -- the epoch the
        parallax and the lens rates already share (C24), and the reason the
        two effects compose -- so an `epoch:` there must be t0_par or
        absent.  Otherwise `epoch:` if given, else EXOFASTv2's RVEPOCH
        default (mkss.pro): the midpoint of the observed span, (min + max)/2
        over every RV of a star in the orbit's primary group.
        """
        epochs = np.full(self.n_elements, np.nan)
        rv = (getattr(system, "active_components", None) or {}).get(
            "rvinstrument"
        )
        for o, kinds in consumers.items():
            user = (self.config[o] or {}).get("epoch")
            if "lens" in kinds:
                t0_par = float(system.mulensevent.t0_par[0])
                if user is not None and float(user) != t0_par:
                    raise ValueError(
                        f"[{self.prefix}.{self.names[o]}] epoch: {user} "
                        f"differs from the event's t0_par {t0_par}.  A lens "
                        f"orbit's rates are anchored at t0_par (C24); set "
                        f"t0_par on the mulensevent block instead, or drop "
                        f"'epoch:'."
                    )
                epochs[o] = t0_par
            elif user is not None:
                epochs[o] = float(user)
            else:
                stars = {
                    idx for t, idx in self.primary_bodies[o] if t == "star"
                }
                times = np.concatenate(
                    [
                        np.asarray(rv.time[rv.rows(i)], dtype=float)
                        for i in range(rv.n_elements)
                        if rv.star_ndx[i] in stars
                    ]
                )
                epochs[o] = 0.5 * (times.min() + times.max())
                logger.info(
                    "[%s] Taylor orbit '%s': reference epoch %.6f BJD_TDB, "
                    "the midpoint of its RVs (set 'epoch:' to choose "
                    "another).",
                    self.prefix,
                    self.names[o],
                    epochs[o],
                )
        return epochs

    def _hint_rv_trends(self, system, consumers):
        """Seed each RV-read Taylor orbit's coefficients from the data.

        EXOFASTv2's start (mkss.pro): a polynomial of the orbit's order fit
        to the velocities about the reference epoch, each instrument's own
        mean removed first (the offsets BETWEEN instruments are not a
        trend).  A ranked data hint, so a user's `initval` still wins.  The
        Keplerian signal is in the fit too; this is a start value, and the
        sampler does the rest.
        """
        rv_read = [o for o, kinds in consumers.items() if "rv" in kinds]
        if not rv_read:
            return  # only lens-read Taylor orbits: no RVs to seed from
        rv = system.rvinstrument
        to_ms = float((u.solRad / u.d).to(u.m / u.s))
        gammas = np.asarray(rv.gamma_init, dtype=float)
        for o in rv_read:
            stars = {idx for t, idx in self.primary_bodies[o] if t == "star"}
            rows = np.isin(
                rv.inst_map,
                [i for i in range(rv.n_elements) if rv.star_ndx[i] in stars],
            )
            dt = np.asarray(rv.time, dtype=float)[rows] - self.taylor_epoch[o]
            resid = (rv.rv * to_ms - gammas[rv.inst_map])[rows]
            order = 2 if self.types[o] == "quadratic" else 1
            if dt.size <= order or np.ptp(dt) == 0.0:
                continue  # too few epochs to say anything: keep the default
            coeffs = np.polyfit(dt, resid, order)  # highest power first
            self.config_manager.add_hint(
                f"{self.prefix}.{o}.gammadot", float(coeffs[-2])
            )
            if order == 2:
                # polyfit's coefficient of dt**2 is HALF the derivative.
                self.config_manager.add_hint(
                    f"{self.prefix}.{o}.gammaddot", float(2.0 * coeffs[0])
                )

    def _taylor_manifest(self, consumers):
        """Manifest entries for the Taylor coefficients, each active exactly
        on the Taylor orbits some consumer reads it on."""
        masks = {}
        for o, kinds in consumers.items():
            for kind in kinds:
                for coeff in self.TAYLOR_COEFFS[kind].get(self.types[o], ()):
                    masks.setdefault(
                        coeff, np.zeros(self.n_elements, dtype=bool)
                    )[o] = True
        out = {}
        for coeff in ("gammadot", "gammaddot", "ds_dt", "dalpha_dt"):
            if coeff not in masks:
                continue
            mask = masks[coeff]
            entry = {}
            if not mask.all():
                entry = {"mask": mask, "inactive_value": 0.0}
            if coeff in ("gammadot", "gammaddot"):
                epochs = ", ".join(
                    (f"{self.names[o]}: " if mask.sum() > 1 else "")
                    + f"{self.taylor_epoch[o]:.6f}"
                    for o in np.nonzero(mask)[0]
                )
                entry["table_note"] = (
                    r"Reference epoch $t_{\rm ref}$ = " + epochs + r" \bjdtdb"
                )
            out[coeff] = entry or None
        return out

    def taylor_radial_velocity(self, t, orbit_idx):
        """The line-of-sight velocity trend of Taylor orbit ``orbit_idx``'s
        PRIMARY group on times ``t``, in the internal RV unit (solRad/d):
        ``gammadot (t - epoch) [+ gammaddot (t - epoch)**2 / 2]``.

        No constant term: a velocity offset at the epoch is exactly
        degenerate with each instrument's gamma, which already carries it.
        """
        o = int(orbit_idx)
        if not self.is_taylor[o]:
            raise ValueError(
                f"[{self.prefix}.{self.names[o]}] taylor_radial_velocity on "
                f"a {self.types[o]} orbit; it has no Taylor coefficients."
            )
        dt = t - float(self.taylor_epoch[o])
        v = self.gammadot.value[o] * dt
        if self.types[o] == "quadratic":
            v = v + 0.5 * self.gammaddot.value[o] * dt**2
        return v

    def _lens_orbit_refs(self, system, comp_name, idx_attr, ref_key):
        """Orbit indices the ``comp_name`` microlensing component references
        through ``ref_key`` -- of ANY orbit type; the callers filter by
        `self.types`.  Reads the built component's resolved ``idx_attr``
        when there is one, else the raw config blocks (the components may
        not be constructed yet in a partial harness).
        """
        comp = in_topology(system, comp_name)
        if comp is None:
            return set()
        idx = getattr(comp, idx_attr, None)
        if idx is not None:
            return {int(idx)}
        blocks = comp if isinstance(comp, list) else [comp]
        out = set()
        for b in blocks:
            if isinstance(b, dict):
                ref = b.get(ref_key)
                if isinstance(ref, int) or str(ref).isdigit():
                    out.add(int(ref))
                elif ref is not None and ref in list(self.names or []):
                    out.add(list(self.names).index(ref))
        return out

    def _lens_companion_orbits(self, system):
        """Orbits a lens COMPANION entry moves on (its ``orbit:`` key)."""
        return self._lens_orbit_refs(
            system, "lens", "motion_orbit_idx", "orbit"
        )

    def _lens_keplerian_orbits(self, system):
        """Keplerian orbits a lens companion's geometry is DERIVED from
        (C24's keplerian mode)."""
        return {
            o
            for o in self._lens_companion_orbits(system)
            if self.types[o] == "keplerian"
        }

    def _lens_taylor_orbits(self, system):
        """Taylor orbits a lens companion's rates are read from (C24's
        linear mode)."""
        return {
            o
            for o in self._lens_companion_orbits(system)
            if self.types[o] in TAYLOR_TYPES
        }

    def _lens_xallarap_orbits(self, system):
        """Orbits the event's SOURCE moves on (``source_orbit:`` on the
        mulensevent block, C25)."""
        return self._lens_orbit_refs(
            system, "mulensevent", "xal_orbit_idx", "source_orbit"
        )

    def _node_degenerate_orbits(self, system):
        """Per orbit: is the ascending node unidentifiable? (review 1.8.3)

        True where astrometry constrains the orbit and nothing radial does.
        `(bigomega, omega_*) -> (bigomega + 180, omega_* + 180)` with the
        matching shift of `tc` is a reflection through the sky plane, so it
        produces identical astrometry of every kind; only an RV says which
        node is ascending.

        PER ORBIT, which is what the shared `amplitude_constrained_orbits`
        predicate makes cheap: it already answers "which orbits does an RV
        measure" and "which does astrometry measure" separately, and the two
        must never disagree with `Planet._mass_constrained` or
        `Orbit._transit_only` about either.  The old test was system-wide --
        any astrometry and no rvinstrument anywhere -- so a mixed system
        truncated an RV-constrained orbit for nothing.
        """
        components = getattr(system, "active_components", None) or {}
        astrometric = set()
        ast = components.get("astrometryinstrument")
        if ast is not None:
            for i, mode in enumerate(ast.modes):
                if mode == "rel":
                    if ast.rel_orbit[i] is not None:
                        astrometric.add(ast.rel_orbit[i])
                else:
                    s_idx = int(ast.config[i].get("star_ndx", 0))
                    astrometric.update(
                        o
                        for o, role in self.star_membership(s_idx)
                        if role == "primary"
                    )
        # A xallarap orbit is astrometry-LIKE for this predicate: the
        # source's sky track enters the trajectory, and a sky track of any
        # kind is invariant under the sky-plane reflection -- so it is
        # node-degenerate exactly as astrometry is, unless something radial
        # breaks it.
        astrometric |= self._lens_xallarap_orbits(system)
        radial = set()
        rv = components.get("rvinstrument")
        if rv is not None:
            for s_idx in set(rv.star_ndx):
                radial.update(o for o, _ in self.star_membership(s_idx))
        # An orbit driving a lens's keplerian geometry is NOT node
        # degenerate: the sky rotation sense of the binary axis in the
        # magnification is exactly what the reflection flips (C24; Skowron
        # Section 5.2's gamma_perp -> -gamma_perp is Omega -> -Omega,
        # i -> 180 - i), so the light curve identifies the node even with
        # no radial data.
        lens_driven = self._lens_keplerian_orbits(system)
        return np.array(
            [
                (i in astrometric)
                and (i not in radial)
                and (i not in lens_driven)
                for i in range(self.n_elements)
            ],
            dtype=bool,
        )

    def _declare_node_degeneracy(self):
        """Seed the node direction vector and annotate the degenerate orbits.

        What this does NOT do any more, and why (review 1.8.3): it used to
        bound `ybigomega >= 0`, truncating `bigomega` to `[0, 180]` to select
        one of the two exactly degenerate modes, and remap a seed in
        `(180, 360)` onto the surviving one.  That HARD truncation biases a
        posterior that hugs the boundary -- measured on `examples/HIP1349`,
        which gave `Omega = 176.5 +/- 2.7` against DMSA's `172.6 +/- 3.4`
        with ZERO origin-crossings in 16k draws.

        The bound is unnecessary, and what it actually cost is worth stating
        precisely.  A posterior centred near the wall extends PAST it, and
        the mass past 180 deg is not represented anywhere else in the
        truncated support -- `(182, omega, tc)` is not the same solution as
        `(2, omega, tc)`, only as `(2, omega+180, tc')`.  So the wall deleted
        real posterior mass and piled the rest against itself, which is the
        HIP1349 measurement.  Without it the chain moves through 180 deg
        continuously and the bias is gone.

        What removing it does NOT buy is migration between the two labels.
        The reflection is a THREE-coordinate move -- the node, omega and tc
        together -- so a straight line through the origin in
        `(xbigomega, ybigomega)` is not along the degeneracy at all; and
        measured, the partner sits ~1e5 raw units away in tc (its raw
        coordinate is scaled to the timing precision, while the two labels
        are half a period apart).  So `fold_node_degeneracy` exists for
        chains or SEEDS that start in different labels -- the case that makes
        Rhat lie -- and not to tidy up after a chain that crossed.

        `src/exozippy/config.md` lists this bound among the manifest's
        deliberate structural values; that entry now names the fold instead.
        """
        note = (
            r"With astrometry but no RVs, $(\Omega, \omega_*)$ and "
            r"$(\Omega+180^\circ, \omega_*+180^\circ)$ are exactly "
            r"degenerate (which node is ascending is unknown); the "
            r"posterior is folded onto $\Omega \in [0^\circ, "
            r"180^\circ)$ after sampling, so the reported interval is a "
            r"fold of a chain that explored both."
        )

        # The direction-vector SEED is not set here any more, and that is a
        # deletion rather than an omission.  It existed to carry the REMAP --
        # a seed in (180, 360) had to be flipped onto the surviving label,
        # and the flip had to beat the relaxation engine, so it was a
        # manifest option.  With no remap there is nothing to beat: the
        # engine has `Eq(xbigomega, cos(bigomega))` and its twin and seeds
        # the pair from a user `bigomega` itself, which is exactly what the
        # astrometry-WITH-RVs branch has always relied on.  Setting it here
        # anyway would apply a shape-wide option to every orbit, degenerate
        # or not, and so move the start of an orbit this has no business
        # touching -- measured on examples/kelt4, whose planet orbit `b` has
        # no bigomega seed and would have been moved from 0 to 90 deg.
        self.manifest["bigomega"] = {"expr_key": "default", "table_note": note}
        self.manifest["omega"] = merge_options(
            self.manifest.get("omega"), table_note=note
        )

    # Sampled coordinates the reflection through the sky plane moves.  The
    # first four flip sign; `tc` moves to the other conjunction.  Every other
    # affected quantity -- omega, esinw, ecosw, tp, ts, sinw, cosw, vcve,
    # bigomega itself -- is DERIVED from these, which is why the fold rewrites
    # only these five and then has PyMC recompute the rest: a hand-written
    # list of derived variables to flip is a list that goes stale silently the
    # next time one is added.
    _FOLD_FLIP = ("xbigomega", "ybigomega", "secosw", "sesinw")

    def fold_node_degeneracy(self, posterior, verbose=True):
        """Collapse the ascending-node degeneracy in a posterior, in place.

        `(bigomega, omega_*, tc)` and `(bigomega + 180, omega_* + 180, tc')`
        describe the SAME physical solution wherever `node_degenerate` is set
        (review 1.8.3), so once the hard half-plane bound is gone nothing
        stops two CHAINS -- or two multi-seed starts -- from occupying
        different labels, and every diagnostic computed on the unfolded
        coordinate then reports two clusters, or non-convergence, for chains
        that agree exactly.  (A single chain will not migrate between them;
        see `_declare_node_degeneracy` for why, and for what removing the
        bound does buy.)  This is that collapse, and it happens ONCE, at
        the single point in `run.py` where sampling ends and post-processing
        begins, so the convergence check, the mode reporter, the seed ledger,
        the tables and the plots all see the same folded draws.  Doing it per
        consumer is how they come to disagree.

        It rewrites the RAW sampled coordinates and nothing else; the caller
        regenerates every Deterministic from them (`pm.compute_deterministics`
        in `System.fold_degenerate_draws`).  That is the whole reason it is
        safe: the alternative -- flipping the physical variables one by one --
        needs a list of every derived quantity that moves, and such a list
        goes stale the next time somebody adds one, silently reporting a
        parameter that no longer agrees with the draws it was computed from.

        Returns True when anything moved.

        Called BEFORE the unit conversion, so the physical values decoded
        here are in internal units (radians, days).
        """
        degenerate = np.atleast_1d(
            np.asarray(getattr(self, "node_degenerate", []), dtype=bool)
        )
        if degenerate.size != self.n_elements or not degenerate.any():
            return False

        moved = False
        for i in np.where(degenerate)[0]:
            # The conjunction that moves is the SAMPLED one: `tc` itself on
            # an orbit sampled at the user's epoch, `tc_sampled` on one
            # sampled nearer the data (tc_epoch != 0; `tc` is then derived
            # and regenerated with every other Deterministic).
            tc_name = self._sampled_tc_name(i)
            params = {}
            skip = None
            for name in self._FOLD_FLIP + (tc_name, "logP"):
                param = getattr(self, name, None)
                key = f"{self.prefix}.{name}_raw"
                if param is None or key not in posterior:
                    skip = name
                    break
                params["tc" if name == tc_name else name] = (param, key)
            if skip is not None:
                logger.warning(
                    "[%s.%s] the ascending-node fold needs %s to be sampled; "
                    "leaving this orbit's draws unfolded, so its Rhat and any "
                    "mode count are computed on the unfolded coordinate.",
                    self.prefix,
                    self.names[i] if i < len(self.names) else i,
                    skip,
                )
                continue

            try:
                phys = {
                    name: p.element_phys_from_raw(
                        i, posterior[key].values[..., self._raw_slot(p, i)]
                    )
                    for name, (p, key) in params.items()
                }
            except ValueError as exc:  # an element that is not sampled
                logger.warning(
                    "[%s.%s] ascending-node fold skipped: %s",
                    self.prefix,
                    self.names[i] if i < len(self.names) else i,
                    exc,
                )
                continue

            # WHICH half-plane to fold ONTO is chosen from the draws, not
            # fixed at [0, 180).  A fixed cut manufactures bimodality the
            # moment a posterior straddles it -- a chain centred on 178 deg
            # with a 3 deg width would be split into a lump at 179 and a lump
            # at 1, which is the review item's own complaint in a new form.
            # `bigomega` is 180-periodic here, so the right centre is the
            # AXIAL mean direction (the circular mean of 2 bigomega, halved),
            # and a draw folds only when it is more than 90 deg from it.  The
            # item offers this as "rotate the cut to bigomega_init +/- 90";
            # taking the centre from the draws rather than from the seed is
            # the same idea without trusting a start value.
            bigomega = np.arctan2(phys["ybigomega"], phys["xbigomega"])
            # `% pi` picks the [0, 180) representative of the axis, which
            # is the conventional half-plane -- arctan2's own principal value
            # would give (-90, 90] and report a perfectly ordinary node as a
            # negative angle.  Which representative the AXIS gets is a
            # labelling choice; which half-plane the draws fold onto is not,
            # and that is set by the centre either way.
            centre = np.mod(
                0.5
                * np.arctan2(
                    np.sin(2.0 * bigomega).mean(),
                    np.cos(2.0 * bigomega).mean(),
                ),
                np.pi,
            )
            flip = np.cos(bigomega - centre) < 0.0
            if not flip.any():
                continue
            moved = True

            ecc = np.clip(
                phys["secosw"] ** 2 + phys["sesinw"] ** 2,
                0.0,
                physics.MAX_ECC,
            )
            omega = np.arctan2(phys["sesinw"], phys["secosw"])
            period = 10.0 ** phys["logP"]
            # The reflected orbit transits at the OTHER conjunction, so tc
            # moves by the difference of the two mean anomalies -- the same
            # `mean_anomaly_at_conjunction` the tp seed solver uses, at omega
            # and at omega + pi.  Then wrapped by whole periods back into tc's
            # own window, which is exactly one period wide: tc and tc + P are
            # the same solution, so there is always exactly one
            # representative inside and the folded draw stays encodable.
            delta = (
                (
                    physics.mean_anomaly_at_conjunction(ecc, omega + np.pi)
                    - physics.mean_anomaly_at_conjunction(ecc, omega)
                )
                * period
                / (2.0 * np.pi)
            )
            tc_param = params["tc"][0]
            lower, upper = self._tc_window(tc_param, i)
            tc_new = phys["tc"] + delta
            if np.isfinite(lower) and np.isfinite(upper):
                span = upper - lower
                tc_new = lower + np.mod(tc_new - lower, span)

            new_phys = {
                name: np.where(flip, -phys[name], phys[name])
                for name in self._FOLD_FLIP
            }
            new_phys["tc"] = np.where(flip, tc_new, phys["tc"])

            # Only the FLIPPED draws are re-encoded.  A decode/encode round
            # trip is the identity only to ~1e-15, and an unflipped draw has
            # no business moving at all -- a fold that perturbs the draws it
            # decided to leave alone is a fold nobody can check.
            for name, values in new_phys.items():
                param, key = params[name]
                slot = self._raw_slot(param, i)
                old_raw = posterior[key].values[..., slot]
                posterior[key].values[..., slot] = np.where(
                    flip, param.element_raw_from_phys(i, values), old_raw
                )

            if verbose:
                logger.info(
                    "[%s.%s] ascending-node fold: %.1f%% of draws mapped to "
                    "the degenerate (bigomega-180, omega+180) partner.",
                    self.prefix,
                    self.names[i] if i < len(self.names) else i,
                    100.0 * float(flip.mean()),
                )
        return moved

    def _sampled_tc_name(self, index):
        """The parameter holding orbit `index`'s SAMPLED conjunction."""
        return "tc_sampled" if self.tc_epoch[index] != 0 else "tc"

    @staticmethod
    def _raw_slot(param, index):
        """Position of element `index` within `param`'s RAW vector.

        Not the element index: only SAMPLED elements get a raw coordinate, so
        a vector with a pinned element ahead of this one is shorter and
        shifted.  Reading the raw array at the element index instead is the
        kind of off-by-one that silently folds the wrong orbit.
        """
        tf = getattr(param, "_raw_transform", None)
        if tf is None:
            raise ValueError(f"[{param.label}] has no sampled elements")
        idx = list(tf["sampled_idx"])
        if index not in idx:
            raise ValueError(f"[{param.label}] element {index} is not sampled")
        return idx.index(index)

    def _tc_window(self, tc_param, index):
        """`tc`'s hard bounds for element `index`, in internal units."""
        tf = getattr(tc_param, "_raw_transform", None)
        if tf is None or not tf["use_logit"][index]:
            return np.nan, np.nan
        return float(tf["lowers"][index]), float(tf["uppers"][index])

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

    def _ltt_reporting_mask(self, system):
        """Per-orbit 0.0/1.0 float array: does some consumer's model
        actually retard THIS orbit's geometry, so `tc`/`tp` need the
        light-travel shift (`tc_target`, and the observed `ts`/`tp`) to relate the observed
        (BJD_TDB) frame rather than the target frame `ltt.py` evaluates
        the Kepler solve in?

        Read from raw config, like `rm.rm_orbits_in_system` -- stage 3
        makes no promise that a sibling component has run its OWN stage 3
        yet, only that every component's `build_maps` (stage 2) has, which
        is what `planet.orbit_map` needs.

        A transit file has no per-planet selection: build_likelihood
        models every planet (hence every orbit with one) in every active
        file's likelihood (see transit.py), so "this orbit is retarded by
        transit" reduces to "this orbit has >=1 planet" AND "some transit
        file has light_travel_time on" -- the per-file default there is
        True, matching `Transit._light_travel_time_active`.  RM is
        per-orbit already, through its own `rm:` key.

        Mixing `light_travel_time` across files for the SAME orbit is a
        pre-existing ambiguity in the model itself (that orbit's own tc/tp
        posterior is then pulled toward the target frame by only some of
        its data) -- not one this mask can resolve, so it warns once and
        treats the orbit as active (closer to correct than leaving it at
        the target-frame value outright).
        """
        mask = np.zeros(self.n_elements, dtype=float)
        cfg = getattr(system, "config", None) or {}

        transit_cfg = cfg.get("transit", []) or []
        transit_flags = [
            bool(c.get("light_travel_time", True)) for c in transit_cfg
        ]
        if any(transit_flags):
            orbit_map = getattr(
                getattr(system, "planet", None), "orbit_map", None
            )
            if orbit_map is not None:
                for o_idx in np.asarray(orbit_map, dtype=int):
                    if 0 <= o_idx < self.n_elements:
                        mask[o_idx] = 1.0
            if len(set(transit_flags)) > 1:
                logger.warning(
                    "orbit: transit files disagree on light_travel_time; "
                    "tc_target and the observed ts/tp treat every orbit touched by an active "
                    "file as fully retarded, an approximation where they "
                    "mix on the same orbit's data."
                )

        rv_cfg = cfg.get("rvinstrument", []) or []
        name_to_idx = {n: i for i, n in enumerate(self.names)}
        for entry in rv_cfg:
            o_name = entry.get("rm")
            if o_name and bool(entry.get("light_travel_time", True)):
                o_idx = name_to_idx.get(o_name)
                if o_idx is not None:
                    mask[o_idx] = 1.0

        return mask

    _GROUP_MASS_SIDE = {"m_primary": "primary", "m_companion": "companion"}

    # The chord expressions' deps that are NOT orbit parameters: the
    # transiting planet's geometry and the i180 sign, injected as context
    # nodes by _chord_context below.  Declaring them here is what keeps
    # graph.py from looking for an `orbit.p` (the group masses avoid this by
    # naming `planet.mass`, a real parameter of a real component; there is no
    # such parameter for `chord_sign` at all, and `p`/`ar` are per PLANET, so
    # the orbit could not consume them elementwise anyway).  `_ltt_mask` is
    # the same idiom for a different reason: it is a CONFIG fact (which
    # orbits some consumer retards, see _ltt_reporting_mask), not a
    # parameter of any component (see _ltt_mask_context).  `_tc_epoch` is a
    # config fact too: the integer number of periods between the user's
    # conjunction and the sampled one, fixed at stage 3 (_sampling_epochs).
    context_dep_names = frozenset(
        {"p", "ar", "chord_sign", "_ltt_mask", "_tc_epoch"}
    )

    # ...and these are built per ORBIT, so Component._element_expression may
    # slice them to a per-element mask (`tc` is derived on the shifted orbits
    # only).  The group masses (W @ mass, one row per orbit) and `_ltt_mask`
    # are per orbit too, which a Taylor orbit's restriction of every
    # Keplerian entry needs: those expressions are then sliced to the
    # Keplerian orbits.
    aligned_context_deps = frozenset(
        {
            "p",
            "ar",
            "chord_sign",
            "_tc_epoch",
            "star.mass",
            "planet.mass",
            "_ltt_mask",
        }
    )

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
        # `_chord_planet`, `_chord_sign` and `inc_modes` are written by
        # register_parameters (stage 3), which add_parameter already requires
        # (no manifest otherwise): read directly, never defaulted (review
        # 2.8.6 -- a defaulted mode list here is how a chord orbit could reach
        # stage 7 with a placeholder geometry).
        idx = np.asarray(self._chord_planet, dtype=int)
        sign = np.asarray(self._chord_sign, dtype=float)
        if idx.size != self.n_elements or sign.size != self.n_elements:
            raise RuntimeError(
                f"[{self.prefix}] chord bookkeeping is {idx.size} planet "
                f"index(es) and {sign.size} sign(s) for {self.n_elements} "
                f"orbit(s) {list(self.names)}; register_parameters writes "
                f"both, one per orbit."
            )

        ctx = {"chord_sign": pt.as_tensor_variable(sign)}
        # Stashed for _add_chord_terms, which needs the same two vectors to
        # build the Jacobian and the geometry bound at stage 7 and must not
        # build a SECOND copy of them: the barrier has to restrain the very
        # node the model was built from.  A per-build cache (see
        # per_build_caches): a REBUILD must never hand stage 7 the previous
        # model's geometry.
        self._chord_geometry = ctx
        # The orbits whose expressions READ the planet geometry: a `chord`
        # orbit derives cos i from it and a `cosi` orbit reports its chord
        # from it.  Only `nochord` orbits read nothing -- and stage 3 makes an
        # orbit `nochord` exactly when it has no single EXISTING planet
        # (`_chord_planet_indices`), so a reader with no planet behind it is a
        # bookkeeping bug, not a configuration.
        readers = [i for i, m in enumerate(self.inc_modes) if m != "nochord"]
        if not readers:
            # Every orbit is `nochord` -- a planet-free system, or a bare
            # Orbit in a test harness (system None): `chord` is not in the
            # manifest and `cosi` is sampled, so nothing evaluates these.
            # They exist only because the dep parser wants the names.
            ctx["p"] = pt.zeros((self.n_elements,))
            ctx["ar"] = pt.zeros((self.n_elements,))
            return ctx

        planet = (
            None if system is None else system.active_components.get("planet")
        )
        n_planets = 0 if planet is None else planet.n_elements
        stray = [i for i in readers if not 0 <= idx[i] < n_planets]
        if stray:
            raise RuntimeError(
                f"[{self.prefix}] orbit(s) "
                f"{[self.names[i] for i in stray]} are in "
                f"{[self.inc_modes[i] for i in stray]} mode, which reads the "
                f"transiting planet's p and a/R*, but name planet index(es) "
                f"{[int(idx[i]) for i in stray]} and the system has "
                f"{n_planets} planet(s).  Stage 3 makes an orbit with no "
                f"existing planet `nochord`."
            )

        safe = np.where(idx < 0, 0, idx).astype("int32")
        take = pt.as_tensor_variable(safe)
        for name in ("p", "ar"):
            # The shared build-time predicate, not a local isinstance: a
            # Parameter left over from an EARLIER model must be rebuilt, or
            # the chord expression consumes the previous build's nodes and
            # the model cannot compile its logp (review 3.14.12).
            if not Component._parameter_is_current(planet, name, model):
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
                if not Component._parameter_is_current(comp, "mass", model):
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
        if param_name in self._LTT_REPORT_PARAMS:
            context_nodes = dict(context_nodes or {})
            context_nodes.setdefault("_ltt_mask", self._ltt_mask_context())
        if param_name in self._EPOCH_PARAMS:
            context_nodes = dict(context_nodes or {})
            context_nodes.setdefault(
                "_tc_epoch",
                pt.as_tensor_variable(
                    np.asarray(self.tc_epoch, dtype="float64")
                ),
            )
        return super().add_parameter(model, param_name, system, context_nodes)

    # The two expressions that read `_tc_epoch`: `tc` derived from the
    # sampled conjunction (calc_tc_from_sampled) and `t0` built at the
    # sampled epoch (calc_t0).
    _EPOCH_PARAMS = ("tc", "t0")

    # tc_target, and the observed ts/tp (defaults.yaml), all consume the one
    # `_ltt_mask` context node -- see _ltt_mask_context and
    # physics.calc_tc_target.  ts_target/tp_target need no mask: they are
    # plain Kepler timing from tc_target.
    _LTT_REPORT_PARAMS = ("tc_target", "ts", "tp")

    def _ltt_mask_context(self):
        """The `_ltt_mask` context node the frame twins consume: the per-orbit
        0.0/1.0 array from `_ltt_reporting_mask` (stage 3) as a constant
        tensor.  All the physics is in physics.calc_tc_target and friends -- the
        closed-form ``-z(t_event)*factor/c`` in the elements -- so unlike the
        chord context nothing here needs a lazy same-component build."""
        mask = getattr(self, "_ltt_report_mask", np.zeros(self.n_elements))
        return pt.as_tensor_variable(np.asarray(mask, dtype="float64"))

    def build_likelihood(self, model, system):
        if self.is_taylor.all():
            # Only Taylor orbits: no eccentricity, chord or retarded
            # geometry exists to bound or describe (their manifest has none
            # of those parameters -- register_parameters).
            return
        self._add_eccentricity_bound(system)
        self._add_vcve_terms(system)
        self._add_chord_terms(system)
        self._add_ltt_frame_prose(system)

    def _add_ltt_frame_prose(self, system):
        """One sentence, only when some orbit is actually retarded: the
        timing rows come in two frames, and which is which."""
        mask = getattr(self, "_ltt_report_mask", None)
        if mask is None or not np.any(mask):
            return
        collector = get_collector(system)
        if collector is None:
            return
        names = [self.names[i] for i in np.flatnonzero(mask)]
        collector.add(
            "The light curves and Rossiter-McLaughlin data of "
            f"{join_names(names)} were modeled at the retarded time, so "
            "the reported $T_C$, $T_S$ and $T_P$ are the observed "
            r"($\rm BJD_{TDB}$) times \citep{Eastman:2010}, and the "
            "target-frame conjunction, eclipse and periastron times the "
            "Keplerian model is evaluated at are given alongside "
            "(subscript ``target'').",
            section="orbits",
            key="orbit.ltt_frame",
        )

    def _chord_indices(self):
        """Indices of the orbits sampling the chord (empty for every cos i
        system)."""
        return [i for i, m in enumerate(self.inc_modes) if m == "chord"]

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
        inside `calc_cosi_from_chord`'s sqrt -- at a STRICTLY POSITIVE floor,
        `physics.CHORD_RADICAND_FLOOR`, and that qualifier is the whole of
        review 1.8.5: with the floor at 0.0 the shield produced a NaN gradient
        instead of preventing one, and because one NaN poisons the whole
        gradient VECTOR, the penalty below could never act.  The shield being
        correct is a PRECONDITION for this potential doing anything at all.
        The floor leaves that whole region flat -- so the penalty here reads
        the UNFLOORED radicand, where it has a gradient pointing back to a
        chord that a transit could actually produce.  Same argument, and the same
        helper, as the eccentricity bound and the V_c/V_e real-root bound.
        """
        idx = self._chord_indices()
        if not idx:
            return
        # Every `chord` orbit has all three in its manifest (INC_MODE_TABLE;
        # ecc/esinw on every orbit) and stage 6 builds every manifest entry,
        # and building `chord` or `cosi` stashes the geometry -- so a miss
        # here is a bookkeeping bug.  It used to RETURN, which would drop the
        # Jacobian (a non-isotropic cos i prior) and the geometry bound with
        # no error (review 2.8.6).
        self._require_built(("chord", "ecc", "esinw"), idx, "fitchord")
        geom = self._chord_geometry
        if geom is None:
            raise RuntimeError(
                f"[{self.prefix}] fitchord orbit(s) "
                f"{[self.names[i] for i in idx]} reached stage 7 with no "
                f"planet geometry for this build: _chord_context must run "
                f"when stage 6 builds 'chord'/'cosi'."
            )

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

        names = [self.names[i] for i in idx]
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
        return [i for i, m in enumerate(self.ecc_modes) if m == "vcve"]

    def _add_vcve_terms(self, system):
        """The terms a V_c/V_e orbit owes: Jacobian, root existence, shield.

        THE JACOBIAN keeps the prior uniform in eccentricity.  A uniform step
        in V_c/V_e "imposes a non-physical prior that strongly biases e toward
        high eccentricities" (Eastman 2024, section 3), so the likelihood
        carries `log|de/d(V_c/V_e)|` -- MINUS what
        `physics.vcve_branch_log_jacobian` returns; see the sign comment below,
        which is the difference between removing that bias and doubling it.
        Applied per orbit and per BRANCH: the lower root substitutes its own
        entry of the Jacobian vector, so each root carries its own weight, and
        that is exactly right, because the Jacobian differs between the two
        roots.

        THE EXISTENCE WEIGHT (review 1.8.14) is a soft lower bound at 0 on
        each branch's unclipped root.  The mixture sums the branches' weights,
        so a branch with no physical orbit behind it must weigh ~0 -- not the
        0.5/|dv/de| it kept on its clipped e = 0, which with the Jacobian and
        the shield below was a prior 1.4% physical, 35.7% at e = 0 and 62.9%
        on the forbidden side of the fold (NUTS chains froze on both).  With
        it, the implied prior over the whole (V_c/V_e, omega) plane is flat in
        e and uniform in omega to the quadrature's resolution
        (tests/test_vcve.py, "the implied prior over the whole plane").

        THE SHIELD is the soft half of the pair that keeps an imaginary
        eccentricity from being a wall.  `_vcve_quadratic` floors the
        discriminant at the strictly positive `VCVE_DISCRIMINANT_FLOOR` (the
        hard half: a floor of exactly 0.0 left the value finite but the
        gradient NaN across the whole region, review 1.8.10), which leaves
        that whole region flat -- so the penalty here is applied to the
        UNFLOORED discriminant, where it has a gradient pointing back into the
        region where a real eccentricity exists.  Same argument, and the same
        `soft_lower_bound` helper, as the eccentricity bound above.  It is
        still load-bearing next to the existence weight: with sin(omega) < 0
        the forbidden side's (mirror) roots are POSITIVE, so only this term
        says no orbit is there.  Being a log-sigmoid it is -log 2 at d = 0, so
        it takes mass off the real side of the fold and leaks the same amount
        onto the forbidden side (1.3% of the prior, measured); the mirror
        continuation in `_vcve_quadratic` is what makes the two cancel, so
        omega stays uniform.

        The chord half's own independent Jacobian (`|d(chord)/d(cos i)|`) lands
        with the chord half; the paper's eq 6 is the joint determinant of the
        two, and JDE's design applies them independently so either half can be
        switched on alone.
        """
        idx = self._vcve_indices()
        if not idx:
            return
        # ECC_MODE_TABLE puts all three in every `vcve` orbit's manifest and
        # stage 6 builds every manifest entry, so a miss is a bookkeeping bug.
        # It used to RETURN, dropping the Jacobian (e biased high -- the
        # paper's own finding), the root mixture and both shields with no
        # error (review 2.8.6).
        self._require_built(("vcve", "ecc", "omega"), idx, "fitvcve")

        take = np.asarray(idx, dtype="int32")
        omega = self.omega.value[take]
        vcve = self.vcve.value[take]
        # The unclipped root vector `_add_eccentricity_bound` built and the
        # collision bound reads (build_likelihood runs that first).  One node
        # for every V_c/V_e orbit; a missing entry is a bookkeeping bug.
        unclipped = self._vcve_unclipped_nodes
        missing = [i for i in idx if i not in (unclipped or {})]
        if missing:
            raise RuntimeError(
                f"[orbit] {self.prefix}: no unclipped V_c/V_e root node for "
                f"orbit(s) {[self.names[i] for i in missing]}; "
                "_add_eccentricity_bound must build it before _add_vcve_terms."
            )
        e_unclipped = unclipped[idx[0]]

        # MINUS the derivative, and the sign is the term.  V_c/V_e is the
        # sampled coordinate, so the eccentricity it derives inherits the
        # density p(e) ~ |d(V_c/V_e)/de|, which diverges as e -> 1 -- the bias
        # the paper reports.  Flattening it means adding log|de/d(V_c/V_e)|,
        # i.e. subtracting what vcve_branch_log_jacobian returns.  Adding it
        # would double the bias, and no check of the derivative's MAGNITUDE can
        # tell the two apart, so the direction is pinned by measuring the
        # implied prior on e for flatness (tests/test_vcve.py).
        #
        # Built from (vcve, omega) per branch, NOT from the clipped `ecc` node
        # (review 1.8.14): on a root the inversion shielded, |e + sin w| of
        # the clipped value is not the map's derivative, and the two places
        # that happens -- a negative root clipped to 0, and the forbidden side
        # of the fold -- carried ~98% of the prior's mass.  `jac` is the
        # UPPER branch's vector; the lower branch substitutes its own entry
        # below, exactly as it substitutes its `ecc`.
        jac = physics.vcve_branch_log_jacobian(vcve, omega, upper=True)
        pm.Potential(f"{self.prefix}.vcve_jacobian", -pt.sum(jac))
        # The root-EXISTENCE weight (review 1.8.14): a branch whose root is
        # negative has no orbit behind it -- `calc_ecc_from_vcve*` clip it to
        # e = 0, where it used to keep its full weight 0.5/|dv/de| and pile
        # 36% of the prior onto e = 0 (divergent at omega = 0, 180 deg).  The
        # soft bound on the UNCLIPPED root is that weight with a gradient,
        # and reading the same node the collision bound reads is what makes
        # the mixture substitute it per branch.  scale = 0.88 for the
        # collision bound's own reason: 500 nats per unit e, so the two ends
        # of the eccentricity range are equally sharp.  Where no real root
        # exists the root is the mirror one (physics._vcve_quadratic), so this
        # also kills the mirror of a negative root -- which is what keeps the
        # real-root shield's leak and its deficit equal, i.e. omega uniform.
        pm.Potential(
            f"{self.prefix}.vcve_root_exists",
            pt.sum(soft_lower_bound(e_unclipped[take], 0.0, scale=0.88)),
        )
        # Declare the OTHER root, so the likelihood is marginalized over both
        # instead of one being chosen (System.register_branch_alternative).  One
        # declaration per V_c/V_e orbit: two orbits are four combinations, which
        # is why the mixture warns past two.  Substituting the clipped `ecc`
        # node, the unclipped one the collision and existence bounds read, and
        # the Jacobian vector is what makes all three a per-branch weight
        # rather than a term evaluated only at the primary root.
        for j, i in enumerate(idx):
            replacements = {
                self.ecc.value: pt.set_subtensor(
                    self.ecc.value[i],
                    physics.calc_ecc_from_vcve_lo(
                        self.vcve.value[i], self.omega.value[i]
                    ),
                ),
                e_unclipped: pt.set_subtensor(
                    e_unclipped[i],
                    physics.ecc_from_vcve_unclipped(
                        self.vcve.value[i],
                        self.omega.value[i],
                        upper=False,
                    ),
                ),
                jac: pt.set_subtensor(
                    jac[j],
                    physics.vcve_branch_log_jacobian(
                        self.vcve.value[i], self.omega.value[i], upper=False
                    ),
                ),
            }
            system.register_branch_alternative(
                f"{self.prefix}.{self.names[i]}: lower V_c/V_e root",
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
        names = [self.names[i] for i in idx]
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

        threshold = pt.as_tensor_variable(
            np.full(self.n_elements, physics.MAX_ECC)
        )
        # A planet-free system (a stellar binary) legitimately has no planet
        # component; one that HAS planets alongside an orbit always declares
        # `max_ecc` (Planet.register_parameters, `has_orbit`), so it is read
        # directly -- a probe here would silently drop the collision limit.
        planets = system.active_components.get("planet")
        if planets is not None:
            if not isinstance(getattr(planets, "max_ecc", None), Parameter):
                raise RuntimeError(
                    f"[{self.prefix}] the collision bound needs planet."
                    f"max_ecc, which stage 6 did not build although the "
                    f"system has an orbit; Planet declares it whenever one "
                    f"exists."
                )
            for p, o in enumerate(np.atleast_1d(planets.orbit_map)):
                o = int(o)
                threshold = pt.set_subtensor(
                    threshold[o],
                    pt.minimum(threshold[o], planets.max_ecc.value[p]),
                )

        if self.is_taylor.any():
            # A Taylor orbit has no eccentricity: its secosw/sesinw are
            # inactive bookkeeping pins and must not enter a potential.
            kep = pt.as_tensor_variable(
                np.nonzero(~self.is_taylor)[0].astype("int32")
            )
            e_unclipped = e_unclipped[kep]
            threshold = threshold[kep]
        pm.Potential(
            f"{self.prefix}.e_collision_bound",
            soft_upper_bound(e_unclipped, threshold, scale=0.88),
        )

    def _unclipped_ecc(self):
        """The unclipped eccentricity of every orbit.

        What a soft bound must see: `calc_ecc` (and `calc_ecc_from_vcve`) clip
        at MAX_ECC, and a flat penalty has no gradient for NUTS to follow.
        Per orbit, because the coordinate the eccentricity is built from is per
        orbit: `secosw^2 + sesinw^2` on a sqrt(e)cos/sin orbit, the unclipped
        V_c/V_e root on a V_c/V_e one.  A vector of both is assembled here, so
        the collision bound above stays one potential over all orbits whatever
        each of them samples.

        Never None (review 2.8.6).  Both modes of ECC_MODE_TABLE name
        secosw/sesinw AND vcve/omega -- sampled, derived or reported -- so all
        four are built by stage 7 on every orbit, an all-circular system
        included (a circular orbit PINS the sqrt(e) pair; it is still built).
        This used to return None with a DEBUG line when a node was missing,
        and `_add_eccentricity_bound` then added no collision bound at all.
        """
        vcve_mask = np.asarray(self.ecc_modes, dtype=object) == "vcve"
        if vcve_mask.size != self.n_elements:
            raise RuntimeError(
                f"[{self.prefix}] {vcve_mask.size} eccentricity mode(s) "
                f"{list(self.ecc_modes)} for {self.n_elements} orbit(s) "
                f"{list(self.names)}; _parse_ecc_parameterization writes one "
                f"per orbit block."
            )
        hk_idx = [int(i) for i in np.nonzero(~vcve_mask)[0]]
        vc_idx = [int(i) for i in np.nonzero(vcve_mask)[0]]
        if hk_idx:
            self._require_built(
                ("secosw", "sesinw"), hk_idx, "the collision bound"
            )
            hk = physics.ecc_from_sqrte(self.secosw.value, self.sesinw.value)
            if not vc_idx:
                return hk
        self._require_built(("vcve", "omega"), vc_idx, "fitvcve")
        vc = physics.ecc_from_vcve_unclipped(self.vcve.value, self.omega.value)
        if not hk_idx:
            self._vcve_unclipped_nodes = {i: vc for i in vc_idx}
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
        idx = np.asarray(vc_idx, dtype="int32")
        mixed = pt.set_subtensor(hk[idx], vc[idx])
        self._vcve_unclipped_nodes = {i: mixed for i in vc_idx}
        return mixed

    def _require_built(self, params, orbits, needed_by):
        """Raise unless every one of `params` is a Parameter at stage 7.

        Not a probe with a fallback: the manifest declares each of these on the
        orbits that reach here and stage 6 builds every manifest entry, so a
        miss is an internal bookkeeping bug -- named, with the orbits and what
        needed the node (review 2.8.6).
        """
        missing = [
            p
            for p in params
            if not isinstance(getattr(self, p, None), Parameter)
        ]
        if missing:
            raise RuntimeError(
                f"[{self.prefix}] {needed_by} on orbit(s) "
                f"{[self.names[i] for i in orbits]} needs {missing}, which "
                f"stage 6 did not build.  Every orbit's manifest declares "
                f"them (ECC_MODE_TABLE / INC_MODE_TABLE), so this is a "
                f"bookkeeping bug, not a configuration."
            )

    def get_true_anomaly(self, t, orbit_idx=None):
        """True anomaly f at times `t`.

        `(N_times, N_orbits)` by default; `(N_times,)` when `orbit_idx` names
        ONE orbit, and then the Kepler solve is done on that orbit alone
        (review 6.8.1).  Slicing the answer instead -- which is what `rm.py`
        did -- solves Kepler's equation on the whole grid and throws every
        other column away, so an N-planet system paid N times over for one
        Rossiter-McLaughlin curve.  `get_sky_position` already indexes its
        inputs before the solve; this is the same pattern.
        """
        if orbit_idx is None:
            t_grid = t[:, None]
            tp = self.tp_target.value[None, :]
            n = self.n.value[None, :]
            ecc = self.ecc.value[None, :]
        else:
            t_grid = t
            tp = self.tp_target.value[orbit_idx]
            n = self.n.value[orbit_idx]
            ecc = self.ecc.value[orbit_idx]

        terms = physics.state_vector_terms(
            t_grid,
            tp,
            n,
            ecc,
            circular=self._all_circular(
                None if orbit_idx is None else [orbit_idx]
            ),
        )

        return pt.arctan2(terms.sinf, terms.cosf)

    def state_vectors(self, t, a_scale, orbit_map, relative=False):
        """Full sky-frame state of an orbiting body:
        (X, Y, Z, VX, VY, VZ), each (N_obs, N_planets).

        t: (N_obs,) vector of times [BJD_TDB]
        a_scale: (N_planets,) amplitude scaling, e.g. the photocenter or
                 relative semimajor axis in mas; positions come out in
                 units of a_scale and velocities in a_scale/day
        orbit_map: integer map from planet slots to orbit elements
        relative: False -> the primary/photocenter orbit around the
                  barycenter (uses omega_*); True -> the companion's orbit
                  relative to the primary (omega_* + 180 deg)

        Axes are skyframe.md's: X = North, Y = East, Z = distance growing
        AWAY from the observer, so dZ/dt carries the radial-velocity sign
        (positive = receding).

        Conventions (EXOFASTv2): omega is the argument of periastron of the
        PRIMARY's orbit (omega_*). bigomega is the position angle of the
        ascending node, measured East of North, where the ascending node is
        the node at which the body recedes from the observer -- consistent
        with the sign of get_radial_velocity (the primary crosses its
        ascending node at omega_* + f = 0, where its RV is maximal).
        Without RVs, (bigomega, omega) and (bigomega+180, omega+180) are
        exactly degenerate for astrometry of every kind (a reflection
        through the sky plane); see _node_degenerate_orbits, which declares
        that per orbit, and System.fold_degenerate_draws, which folds the two
        labels together for the convergence check, seed ledger and mode
        reporter (review 1.8.3).

        This is the accessor an N-body backend replaces (reviews 4.8.2,
        8.8.15): a consumer that can take a full state vector should take it
        from here rather than re-projecting the elements itself.  The
        `a_scale` factoring (the Keplerian's self-similarity) and the
        `relative` omega-flip are Keplerian conveniences that will NOT
        survive that swap -- an integrator emits physical units and
        `relative` becomes a subtraction -- so new consumers should treat
        both as this method's business, never re-derive them.
        """
        t_grid = t[:, None]
        tp = self.tp_target.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]
        cosi = self.cosi.value[orbit_map][None, :]
        sini = self.sini.value[orbit_map][None, :]
        bigomega = self.bigomega.value[orbit_map][None, :]
        cosO = pt.cos(bigomega)
        sinO = pt.sin(bigomega)

        if relative:
            # The companion's argument of periastron is omega_* + pi
            cosw = -cosw
            sinw = -sinw

        terms = physics.state_vector_terms(
            t_grid,
            tp,
            n,
            ecc,
            sinw=sinw,
            cosw=cosw,
            circular=self._all_circular(orbit_map),
        )

        # Separation from the barycenter (or primary) in units of a_scale
        r = a_scale[None, :] * terms.r_over_a

        # Thiele-Innes projection (North, East), PA measured East of North:
        # at omega + f = 0 (ascending node) the body sits at PA = bigomega.
        # One owner (physics.thiele_innes_xy) -- the microlensing keplerian
        # mode projects the same way in Einstein units.
        X, Y = physics.thiele_innes_xy(
            r, terms.coswf, terms.sinwf, cosi, bigomega
        )
        Z = r * terms.sinwf * sini

        # d/dt of the above: vamp * the kernel's velocity phase terms.
        vamp = n * a_scale[None, :] / terms.ecc_factor
        VX = vamp * (cosO * terms.vx_phase - sinO * terms.vz_phase * cosi)
        VY = vamp * (sinO * terms.vx_phase + cosO * terms.vz_phase * cosi)
        VZ = vamp * terms.vz_phase * sini

        return X, Y, Z, VX, VY, VZ

    def get_sky_position(self, t, a_scale, orbit_map, relative=False):
        """
        Vectorized sky-plane offsets of an orbiting body -- the position
        half of `state_vectors` (whose docstring carries the conventions).

        Returns (dE, dN), each (N_obs, N_planets): offsets toward East and
        North in the units of a_scale.
        """
        X, Y, _, _, _, _ = self.state_vectors(
            t, a_scale, orbit_map, relative=relative
        )
        return Y, X

    def get_radial_velocity(self, t, K, orbit_map):
        """
        The optimized vectorized reflex RV signal.
        t: (N_obs,) vector of times
        K: (N_planets,) vector of semi-amplitudes

        This is `state_vectors`' VZ with the amplitude collapsed: K already
        carries the barycentric fraction, sin(i) and the n*a/sqrt(1-e^2)
        velocity scale, so only the kernel's phase term remains.
        """
        # Broadcast time and orbital parameters into (N_obs, N_planets)
        # grids; the kernel does the Kepler solve (review 6.8.2 forwarding).
        t_grid = t[:, None]
        tp = self.tp_target.value[orbit_map][None, :]
        n = self.n.value[orbit_map][None, :]
        ecc = self.ecc.value[orbit_map][None, :]
        cosw = self.cosw.value[orbit_map][None, :]
        sinw = self.sinw.value[orbit_map][None, :]

        terms = physics.state_vector_terms(
            t_grid,
            tp,
            n,
            ecc,
            sinw=sinw,
            cosw=cosw,
            circular=self._all_circular(orbit_map),
        )

        # vz_phase = cos(w + f) + e cos(w)
        return K[None, :] * terms.vz_phase
