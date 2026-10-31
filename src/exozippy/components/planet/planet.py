import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component, in_topology
from exozippy.components.parameterization import (
    merge_overrides,
    mode_manifest,
    pin_unselected,
)
from exozippy.config import user_entry
from exozippy.constants import (
    KEPLER_CONST,
    MSUN_TO_MEARTH,
    RJUP_TO_REARTH,
    RSUN_TO_REARTH,
)
from exozippy.outputs.prose import get_collector, join_names
from exozippy.outputs.texutils import latex_escape
from exozippy.potentials import soft_lower_bound

from ..orbit.bodies import (
    REMOVED_PLANET_KEYS,
    component_instance_names,
    orbit_types,
    planet_membership,
)
from ..orbit.orbit import (
    amplitude_constrained_orbits,
    inclination_constrained_orbits,
)
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


# The mass coordinate is an n-way choice spelled as one BOOLEAN per
# alternative, at most one true (components.md "Config flag vocabulary"; JDE
# 2026-09-12, "consistency is important").  Neither set is the signed linear
# mass.  Each flag is tri-state at the config layer, exactly as orbit's
# fitvcve is: unset -> the topology default
# (Planet._resolve_mass_parameterization), true/false -> honored.
# {flag: MASS_MODE_TABLE key}.
MASS_FLAGS = {"fitlogq": "log_q", "fitmsini": "msini"}


def _mass_parameterization_removed(value):
    translation = {
        "log_q": "'fitlogq: true'",
        "linear": "'fitlogq: false' and 'fitmsini: false'",
    }.get(value)
    which = (
        f"For 'mass_parameterization: {value}' write {translation}.  "
        if translation is not None
        else ""
    )
    return (
        "'mass_parameterization:' was replaced by per-planet booleans "
        "(review 2.14.9; one boolean per coordinate, at most one true): "
        "'fitlogq: true' samples log10(m_planet/m_host), 'fitmsini: true' "
        "samples (m sin i, cos i), and neither samples the signed linear "
        "mass.  Each defaults per planet from the data topology when left "
        f"out.  {which}Delete the 'mass_parameterization:' key."
    )


# Keys that WERE planet config keys and are gone, refused with their own
# message (the mulensing bodies.REMOVED_KEYS pattern): a config naming one was
# written for a spelling that no longer exists, and the user needs the
# replacement, not silence.  {key: message(value)}.
# `orbit_ndx:` and `star_ndx:` are the second spelling of the orbit groups
# (owned by orbit/bodies.py, whose `planet_membership` is the one reader).
REMOVED_KEYS = {
    "mass_parameterization": _mass_parameterization_removed,
    **REMOVED_PLANET_KEYS,
}


def parse_mass_flags(entry, where):
    """``{flag: True | False | None}`` for one planet config entry.

    The one reader of `MASS_FLAGS`, so every consumer (the planet's own
    resolution, and the microlensing readers that need to know whether a lens
    companion samples log_q before the planet has resolved) agrees on the
    spelling, on the retired key and on the "more than one true" refusal.
    """
    entry = entry or {}
    for key, message in REMOVED_KEYS.items():
        if key in entry:
            raise ValueError(f"[{where}] {message(entry[key])}")
    flags = {}
    for flag in MASS_FLAGS:
        raw = entry.get(flag)
        if raw is not None and not isinstance(raw, bool):
            raise ValueError(
                f"[{where}] '{flag}:' must be true or false, got {raw!r}."
            )
        flags[flag] = raw
    on = [f for f, v in flags.items() if v]
    if len(on) > 1:
        raise ValueError(
            f"[{where}] {' and '.join(on)} are both true, and they name "
            "different mass coordinates -- a planet is sampled in one.  Set "
            "at most one (leave both out for the topology default; set both "
            "false for the signed linear mass)."
        )
    return flags


def lens_companion_samples_log_q(entry, where):
    """Does a microlensing lens-companion planet sample log_q?

    A lens body defaults to log_q (Planet._resolve_mass_parameterization), so
    it does unless its entry turns log_q off or asks for msini.  For the
    star/mulensevent readers that decide at THEIR stage 3, which may run
    before the planet's.
    """
    flags = parse_mass_flags(entry, where)
    return not (flags["fitlogq"] is False or flags["fitmsini"])


class Planet(Component):
    # The two mass coordinates, as a parameterization mode table (see
    # components/parameterization.py).  `linear` samples planet.mass itself over
    # a range including negatives -- RV and astrometric amplitudes flip phase
    # through zero, so crossing zero is what avoids the Lucy-Sweeney bias on a
    # marginal detection -- while `log_q` samples log10(m_p / m_host) and derives
    # the mass from it.  Planets may differ: log_q is not a parameter of a linear
    # planet at all, and vice versa.
    #
    # The third, `msini` (fitmsini, review 2.14.9), samples the minimum mass
    # m sin i and derives mass = msini / sin i -- what an RV-only orbit
    # measures, so the (mass, cos i) banana becomes a well-conditioned
    # (msini, cos i) pair; Planet.build_likelihood adds the -log sin i
    # Jacobian that keeps the prior uniform in (mass, cos i).  Everywhere else
    # msini is REPORTED (derived, built late, consumed by nothing): that is
    # what lets mass read msini on one planet while msini reads mass on
    # another without a per-parameter cycle (parameter.md, the element roles).
    MASS_MODE_TABLE = {
        "linear": {"mass": None, "msini": {"output_expr_key": "default"}},
        "log_q": {
            "mass": {"expr_key": "default", "force_node": True},
            "log_q": None,
            "msini": {"output_expr_key": "default"},
        },
        "msini": {
            "mass": {"expr_key": "from_msini", "force_node": True},
            "msini": None,
        },
    }

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"
        # The mass-coordinate booleans, validated at the config boundary (a
        # retired 'mass_parameterization:' raises here, with its migration).
        self.mass_flags = [
            parse_mass_flags(c, f"{self.prefix}.{nm}")
            for c, nm in zip(self.config, self.names)
        ]
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
                    "same mass/K driving the RV model. Default False. "
                    "A tie is a physics LINK, not a one-way assignment: "
                    "information flows toward whichever side is less "
                    "constrained elsewhere (components.md, 'Config flag "
                    "vocabulary')."
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
                "key": "fitlogq",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Sample log10(m_planet / m_host) and derive the mass "
                    "(the prior is then uniform in log q). Default: on when "
                    "no RV or astrometric data measure this planet's orbit, "
                    "including every microlensing lens body, where q <= 0 "
                    "is meaningless. At most one of fitlogq/fitmsini may be "
                    "true; with neither, planet.mass itself is sampled, with "
                    "a negative lower bound so an RV or astrometric "
                    "amplitude can flip sign (avoids the Lucy-Sweeney "
                    "positive-definite bias)."
                ),
            },
            {
                "key": "fitmsini",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Sample the minimum mass m sin i (with the orbit's "
                    "cos i) and derive mass = msini / sin i, with the "
                    "Jacobian that keeps the prior uniform in mass and "
                    "cos i -- the same posterior as sampling the mass, in "
                    "the coordinates an RV curve measures. Default: on when "
                    "RVs measure this planet's mass and nothing (transit, "
                    "RM, Doppler tomography, astrometry, keplerian lens or "
                    "source orbital motion) measures its orbit's "
                    "inclination, unless planet.mass is pinned (sigma: 0). "
                    "Cannot be combined with the orbit's fitchord (cos i "
                    "would then depend on the mass); independent of "
                    "fitvcve. At most one of fitlogq/fitmsini may be true."
                ),
            },
        ]

    def build_maps(self):
        """Stage 2: the planet -> host star and planet -> orbit index maps.

        Both are DERIVED from the orbit blocks' primary:/companion: groups
        (`orbit.bodies.planet_membership`, the one reader): a planet carries
        no key of its own for either, so the two cannot disagree.

        `star_map` always exists -- every planet has a host.  `orbit_map`
        exists only when EVERY planet is named by an orbit block; a planet
        paired implicitly with no orbit (a static lens companion with no
        orbit block) has none, and `orbit_ndx` (the per-planet list, None
        there) is what to ask.  No sentinel index is invented: a reader that
        needs an orbit for an orbit-less planet fails loudly instead of
        indexing orbit -1.
        """
        membership = planet_membership(self.config_manager.system_config)
        if len(membership) != self.n_elements:
            raise RuntimeError(
                f"[{self.prefix}] the system config declares "
                f"{len(membership)} planet(s), but this component has "
                f"{self.n_elements}; planet membership is read off the "
                f"system config, which must hold this component's block."
            )
        self.orbit_ndx = [o for o, _ in membership]
        self.star_map = np.array([h for _, h in membership], dtype=int)
        if all(o is not None for o in self.orbit_ndx):
            self.orbit_map = np.array(self.orbit_ndx, dtype=int)

    #: The summary figure's header, per planet: the Parameter label and the
    #: element it is read at -- the planet's own index, or the index of the
    #: orbit it sits on.
    SUMMARY_HEADER = (
        ("orbit.period", "orbit"),
        ("planet.radius", "planet"),
        ("planet.mass", "planet"),
        ("orbit.ecc", "orbit"),
    )

    def summary_header(self, system):
        """One header line per planet: P, R_P, M_P and e from the posterior.

        Read off the distributed posterior, each value formatted by
        ``summary_plot.format_value`` with the Parameter's own table symbol
        and unit.  A quantity this fit does not report -- no such
        Parameter, or one with no posterior -- is left out of the line
        rather than shown as a blank.  With more than one planet each line
        is prefixed by the planet's name.  See ``Component.summary_header``.
        """
        from ...outputs.summary_plot import format_value, reported_summary

        lookup = system.get_parameter_lookup()
        lines = []
        for p_idx, pname in enumerate(self.names):
            o_idx = self.orbit_ndx[p_idx]
            parts = []
            for label, owner in self.SUMMARY_HEADER:
                if owner == "orbit" and o_idx is None:
                    continue  # no orbit, so no period or eccentricity
                param = lookup.get(label)
                index = o_idx if owner == "orbit" else p_idx
                summ = reported_summary(param, index)
                if summ is None:
                    continue
                value = format_value(summ)
                if value is None:
                    continue
                unit = (param.unit_latex or "").replace("$", "")
                unit_text = rf"\,{unit}" if unit else ""
                parts.append(f"${param.latex} = {value}{unit_text}$")
            if parts:
                prefix = f"{pname}:  " if len(self.names) > 1 else ""
                lines.append(prefix + "   |   ".join(parts))
        return lines

    def keplerian_geometry(self, system):
        """Do this system's planets have Keplerian orbital GEOMETRY?

        Everything Planet derives from an orbit -- p, a, a/R*, b, K, msini,
        teq, the transit durations, the radius prior and the msini/crossing
        potentials -- is Keplerian.  A planet on a Taylor orbit (`type:
        linear|quadratic`: a lens companion's linear orbital motion, or an
        unresolved trend), on a `type: static` orbit (a pairing with no
        dynamics), or on no orbit at all (the implicit static pairing of
        `orbit.bodies.planet_membership`) has none of it, and reading a
        non-Keplerian orbit's inactive bookkeeping as an orbit would be a
        silently wrong number.  So the question is per planet, not "is
        there an orbit component" (#485).

        True when every planet's orbit is Keplerian, False when none is.  A
        MIX raises: the manifest entries are whole-component, so one
        planet's geometry cannot be declared while another's is absent
        (yet).  A TRANSIT block with any planet off a Keplerian orbit raises
        first: every transit file models every planet's transit geometry
        (p, b, a/R*, the durations).
        """
        sys_cfg = self.config_manager.system_config
        types = orbit_types(sys_cfg)
        orbit_names = component_instance_names(sys_cfg, "orbit")
        kinds = [None if o is None else types[o] for o in self.orbit_ndx]
        keplerian = [k == "keplerian" for k in kinds]
        if in_topology(system, "transit") is not None and not all(keplerian):
            off = [
                f"'{n}' ("
                + (
                    "in no orbit block"
                    if k is None
                    else f"on the type: {k} orbit '{orbit_names[o]}'"
                )
                + ")"
                for n, k, o, kep in zip(
                    self.names, kinds, self.orbit_ndx, keplerian
                )
                if not kep
            ]
            raise ValueError(
                f"[{self.prefix}] planet(s) {', '.join(off)} have no "
                f"Keplerian orbit, and this system has a transit block, "
                f"which models every planet's Keplerian transit geometry.  "
                f"Name each planet in the companion group of a Keplerian "
                f"orbit block (type: keplerian, the default)."
            )
        if all(keplerian):
            return True
        if not any(keplerian):
            return False
        raise NotImplementedError(
            f"[{self.prefix}] planets "
            f"{[n for n, kep in zip(self.names, keplerian) if not kep]} have "
            f"no Keplerian orbit and "
            f"{[n for n, kep in zip(self.names, keplerian) if kep]} do; a mix "
            f"is not supported yet (the planet's orbit-derived parameters are "
            f"declared for every planet or none)."
        )

    def register_parameters(self, system):
        """Stage 3: Auto-estimates and Manifest declaration."""
        # Keplerian GEOMETRY, not merely "an orbit component exists": a lens
        # planet on a linear lens orbit has no Keplerian elements to derive
        # p, a, K, ... from (keplerian_geometry).
        has_orbit = self.keplerian_geometry(system)

        self._resolve_mass_parameterization(system)

        # The mass coordinate, per planet: a `log_q` planet derives its mass
        # from log10(m_p/m_host), an `msini` one from m sin i and the orbit's
        # cos i, and a `linear` one samples the (signed) mass itself; `log_q`
        # is not a parameter of a non-log_q planet at all.  msini needs an
        # orbit (sin i), so without one the table carries no msini, exactly
        # as the hand-written manifest declared it only under has_orbit.
        table = self.MASS_MODE_TABLE
        if not has_orbit:
            table = {
                mode: {k: v for k, v in spec.items() if k != "msini"}
                for mode, spec in table.items()
                if mode != "msini"
            }
        mass_entries = mode_manifest(
            self.mass_parameterizations,
            table,
            n_elements=self.n_elements,
            where=f"{self.prefix}.fitlogq/fitmsini",
        )
        if "msini" in mass_entries:
            mass_entries["msini"] = self._msini_bounds_override(
                mass_entries["msini"]
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
        # The linear mass ratio, whichever coordinate was sampled: a
        # linear-mode planet has no log_q, so this is its only q.  Needs
        # no orbit (mass and the host mass), unlike everything below.
        self.manifest["q"] = "default"

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
                    # Sampled on an msini-mode planet, reported elsewhere
                    # (MASS_MODE_TABLE); same manifest position as ever.
                    "msini": mass_entries["msini"],
                    "max_ecc": "default",
                    # EXOFASTv2's delta, tcirc, omegagr (derivepars.pro).
                    "delta": "default",
                    "tcirc": "default",
                    "omegagr": "default",
                    # Irradiation (EXOFASTv2's teq and fave): both need
                    # a/R*, so they exist exactly when the orbit does.
                    "teq": "default",
                    "fave": "default",
                    # A priori transit/eclipse probabilities (Winn 2010
                    # eq 9), grazing-inclusive and full-event.
                    "ptg": "default",
                    "pt": "default",
                    "psg": "default",
                    "ps": "default",
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
            # The OPT-IN pin: `sigma: 0` on the planets that did not set
            # fitbeam, through the manifest "overrides" channel so a user who
            # writes their own sigma still wins, and `{}` (a plain free
            # parameter) when every planet opted in.  That is exactly what the
            # hand-written loop here used to build; it goes through the one
            # helper now so the pattern has a single implementation
            # (review 3.14.11).
            self.manifest["beam"] = pin_unselected(
                self.n_elements, self.fitbeam
            )
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
            self._push_rv_mass_hints(system, k_ms_guess)

        self._resolve_chen(system)

    # The keys that state a planet's mass, in any of its coordinates.  A user
    # value on one of them and RVInstrument's semi-amplitude estimate are two
    # values of ONE quantity (K is the mass at the orbit's P, e, sin i and
    # the host mass), so the user's supersedes the estimate outright.
    _MASS_KEYS = ("mass", "msini", "log_q", "K")

    def _push_rv_mass_hints(self, system, k_ms):
        """Seed each RV planet's mass from the RV semi-amplitude estimate
        `k_ms` (m/s), at the data tier -- and only its mass.

        K and the mass are one quantity as far as the starts go (JDE
        2026-10-14).  The estimate is a hint on the DERIVED planet.K, and
        under the precedence rule (review 3.14.22) the relaxation engine
        reconciles it with the model by moving the LOWEST-tier seeds K
        reads.  The sampled mass coordinate is DEFAULT_WEAK (planet.mass in
        linear coordinates, planet.msini in m sin i ones, both in
        defaults.yaml), below the orbit's cos i, e/omega and the host mass,
        so the mass alone absorbs it, at the P, e, sin i and host mass of
        the final solve -- the exact conversion, done by the engine, with no
        stage-3 guess at those starts.  (Converting here instead was tried:
        ``probe_start`` at stage 3 does not know which elements are derived,
        and its e moved off the default -- to 1.000006 under the init_scale
        noise test -- so the converted mass was not the stage-4 one.)

        Where that ordering cannot hold the estimate is not pushed:

        * a user value on any mass coordinate, or on K, SUPERSEDES it, as a
          higher tier does on the same element -- otherwise the only seeds
          left to move would be the orbit's angles (a face-on or eccentric
          start: harmless for an RV fit, catastrophic for a transit fit,
          which then no longer transits).  Recorded in the solve
          diagnostics (``supersede_hint``).
        * a log_q planet (``fitlogq: true`` beside RVs) samples a DEFAULT
          log_q, which would share the correction with cos i; it is logged.

        Only planets on Keplerian orbits carry the K relation.
        """
        cm = self.config_manager
        orbit = system.active_components["orbit"]
        up = cm.user_params
        for i in range(self.n_elements):
            o = int(self.orbit_map[i])
            if orbit.types[o] != "keplerian":
                continue
            nm = self.names[i]
            stated = [
                key
                for key in self._MASS_KEYS
                for spelled in (
                    f"{self.prefix}.{i}.{key}",
                    f"{self.prefix}.{key}",
                )
                # A `unit:`-only entry is how the user READS the quantity,
                # not a value for it, so it supersedes nothing (1.5.10's K
                # twin: `planet.b.K: {unit: km/s}` used to drop the
                # estimate as though the user had stated a K).
                if set(user_entry(up, spelled) or ()) - {"unit"}
            ]
            if stated:
                cm.supersede_hint(
                    f"{self.prefix}.{i}.K",
                    k_ms,
                    f"{self.prefix}.{nm}.{stated[0]}",
                    "the RV semi-amplitude estimate and the planet's mass "
                    "are one quantity; the user's value is used",
                )
                continue
            if self.mass_parameterizations[i] == "log_q":
                logger.info(
                    f"[{self.prefix}.{nm}] the RV semi-amplitude estimate "
                    f"({k_ms:.4g} m/s) is not used as a start: this planet "
                    "samples log_q, whose default start would share the "
                    "correction with the orbit's cos i.  Seed planet.mass "
                    "or planet.log_q to start from the RV amplitude."
                )
                continue
            # k_ms is in m/s whatever unit the user reads K in (1.5.10).
            cm.add_hint(f"{self.prefix}.{i}.K", k_ms, unit="m/s")

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
        ridge.  Measured on examples/DC2018_128: planet.mass ESS 337 -> 3230
        and star.logmass 363 -> 2865, i.e. the mass stops being the
        convergence bottleneck (see "Planet mass parametrization" in
        src/exozippy/components/star/star.md, which also RETIRES the
        "~4.5x on the whitened 2x2 curvature block" figure this comment used
        to quote -- it measured a local block at the start point).  Removing
        the unreachable-but-samplable negative region is the other half.

        'msini' (review 2.14.9) is for the planet whose mass RVs measure and
        whose inclination nothing does: the data constrain m sin i, so the
        (mass, cos i) posterior is a banana whose tail (P(M > X) ~ (2/pi)
        msini/X) a sampler in those coordinates under-explores.  Sampling
        (msini, cos i) with the -log sin i Jacobian is the same posterior in
        coordinates the data measure (star.md, "Planet mass parametrization").

        Default, per planet, in order: 'log_q' for a microlensing lens body
        or where no RV/astrometry measures the orbit; 'msini' where the orbit's
        inclination is unmeasured (`inclination_constrained_orbits`), the orbit
        does not sample the transit chord, and the user did not pin
        planet.mass; 'linear' otherwise.  ``fitlogq:`` / ``fitmsini:`` on the
        planet override it: true selects that coordinate, false rules it out.
        """
        mass_sides = self._mass_constrained(system)
        # Only a Keplerian orbit carries an inclination (keplerian_geometry).
        orbit = (
            system.active_components.get("orbit")
            if self.keplerian_geometry(system)
            else None
        )
        if orbit is not None:
            inc_constrained = inclination_constrained_orbits(system, orbit)
            inc_modes = orbit.inclination_modes(system)
        else:
            inc_constrained, inc_modes = set(), []

        # The lens component is one instance per lens BODY (8.6.17 split);
        # its `bodies` list is [(comp_type, index), ...], primary first.
        lens = system.active_components.get("lens")
        lens_bodies = (
            set(getattr(lens, "bodies", []) or [])
            if lens is not None
            else set()
        )

        # ANY chord orbit rules msini out for EVERY planet, not just the
        # planets on that orbit.  The build graph orders PARAMETERS, not
        # elements: a chord orbit's cos i element reads planet.ar (through
        # a/R* -> m_total -> planet.mass), so orbit.cosi as a whole depends on
        # planet.mass -- and an msini planet's mass reads orbit.cosi.  The
        # value graph is acyclic per element and the parameter graph is not
        # (measured: two RV orbits, one fitvcve -> fitchord, recursed in
        # Component.add_parameter).
        chord_orbits = [
            orbit.names[i] for i, m in enumerate(inc_modes) if m == "chord"
        ]

        modes, reasons = [], []
        for p, (flags, nm) in enumerate(zip(self.mass_flags, self.names)):
            o = self.orbit_ndx[p]
            has_orbit = orbit is not None
            chord = bool(chord_orbits)

            if flags["fitlogq"]:
                modes.append("log_q")
                reasons.append("fitlogq: true")
                continue
            if flags["fitmsini"]:
                if not has_orbit:
                    raise ValueError(
                        f"[{self.prefix}.{nm}] 'fitmsini: true' needs an "
                        "orbit: the mass is m sin i / sin i, and sin i is the "
                        "orbit's.  Add the orbit, or drop the key."
                    )
                if chord:
                    raise ValueError(
                        f"[{self.prefix}.{nm}] 'fitmsini: true' cannot be "
                        f"combined with the transit chord, which orbit(s) "
                        f"{chord_orbits} sample (fitchord, explicit or its "
                        "default, which follows fitvcve): the chord derives "
                        "cos i from a/R*, which depends on the planet mass, "
                        "while fitmsini derives the mass from cos i -- a "
                        "dependency cycle (the build orders the cos i and "
                        "mass VECTORS, so this holds across orbits too).  "
                        "Set 'fitchord: false' on those orbits, or drop "
                        "'fitmsini'."
                    )
                modes.append("msini")
                reasons.append("fitmsini: true")
                continue

            # The topology default.  A lens body outranks the RV/astrometry
            # signal: microlensing hard-forbids q <= 0, so a signed mass there
            # is unusable no matter what else measures the orbit.
            if ("planet", p) in lens_bodies:
                mode, why = "log_q", "it is a microlensing lens body"
            elif not mass_sides[p]:
                mode, why = "log_q", "no signed observable constrains its mass"
            elif not has_orbit or o in inc_constrained:
                mode, why = "linear", "RV/astrometry data measure its orbit"
            elif chord:
                mode, why = (
                    "linear",
                    "RVs measure its mass, but an orbit samples the chord",
                )
            elif self._user_pinned_mass(p):
                mode, why = (
                    "linear",
                    "RVs measure its mass and planet.mass is pinned "
                    "(sigma: 0), which m sin i coordinates would drop",
                )
            else:
                mode, why = (
                    "msini",
                    "RVs measure m sin i and nothing measures the inclination",
                )
            # An explicit false rules its coordinate out of the default; the
            # signed linear mass is what is left.
            if mode == "log_q" and flags["fitlogq"] is False:
                mode, why = "linear", "fitlogq: false"
            elif mode == "msini" and flags["fitmsini"] is False:
                mode, why = "linear", "fitmsini: false"
            modes.append(mode)
            reasons.append(why)

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
        if any(m != "linear" for m in modes):
            logger.info(
                f"[{self.prefix}] mass coordinate per planet: {detail} "
                "(override per planet with 'fitlogq:' / 'fitmsini:'; both "
                "false samples the signed linear mass)."
            )

        self.mass_parameterizations = modes
        self._reconcile_mass_user_params()

    def _user_pinned_mass(self, index):
        """Did the user pin planet `index`'s mass (sigma: 0)?

        A coordinate that is ON BY DEFAULT must not throw away a constraint
        the user wrote, and `sigma: 0` on an element the flip makes DERIVED
        cannot hold (it raises, `_raise_derived_mass_pin`) -- the orbit's
        `_pin_blocks_default` rule, for the same reason.  Read in the two spellings that survive
        `standardize_param_names` (indexed, and the broadcast that covers
        every element), as `Orbit._user_pinned` does.
        """
        up = self.config_manager.user_params
        for key in (f"{self.prefix}.{index}.mass", f"{self.prefix}.mass"):
            entry = user_entry(up, key)
            if entry is not None and entry.get("sigma") == 0:
                return True
        return False

    def _msini_bounds_override(self, entry):
        """Give the SAMPLED msini elements planet.mass's hard bounds.

        msini carries no lower/upper in defaults.yaml on purpose: a derived or
        reported element with a numeric bound gets a soft barrier, which would
        add a potential to every linear/log_q planet.  The sampled elements
        take mass's bounds instead, per element through the manifest
        "overrides" channel (NaN leaves an element alone), so a user's own
        msini bound still layers on top.  |m sin i| <= |m|, so the mass range
        is the natural support; the mass itself keeps its bounds as soft
        barriers on the derived value, as in log_q mode.
        """
        sampled = np.array(
            [m == "msini" for m in self.mass_parameterizations], dtype=bool
        )
        if not sampled.any():
            return entry
        base = self.config_manager.base_defaults["planet"]
        mass_def, msini_def = base["mass"], base["msini"]
        if mass_def["unit"] != msini_def["unit"]:
            raise ValueError(
                "planet/defaults.yaml: msini takes planet.mass's bounds, so "
                f"the two must share a unit; mass is in {mass_def['unit']!r} "
                f"and msini in {msini_def['unit']!r}."
            )
        lower = np.where(sampled, float(mass_def["lower"]), np.nan)
        upper = np.where(sampled, float(mass_def["upper"]), np.nan)
        return merge_overrides(
            entry, {"lower": lower.tolist(), "upper": upper.tolist()}
        )

    def _reconcile_mass_user_params(self):
        """Check the user's planet.mass / log_q entries against the coordinate
        each planet samples.  READ-ONLY on ``user_params`` (CLAUDE.md).

        * ``sigma: 0`` on planet.mass where the mass is DERIVED (log_q or
          m sin i is sampled) RAISES, naming the sampled coordinate to pin.
          A pin on a derived parameter cannot hold it fixed, and moving the
          user's pin onto another parameter for them -- what this method did
          until review 2.1.26(e) -- rewrote their params behind their back:
          the moved log_q pin then read as a user entry in the ledger,
          ``export_solution`` and the startup table (JDE 2026-10-31: "we
          should never move a user pin").  A sigma > 0 is a real prior and
          works on the derived node as a potential, so it is left alone.
          Bounds are not a pin either: on a derived mass they become soft
          barriers on the physical value, the semantics wanted.
        * A non-positive mass initval in log_q mode raises (unrepresentable).
        * A ``planet.<i>.log_q`` entry on a planet that does not sample log_q
          raises (a stale mkparam restart file).
        """
        up = self.config_manager.user_params
        for i in range(self.n_elements):
            mass_key, log_q_key = f"planet.{i}.mass", f"planet.{i}.log_q"
            mode = self.mass_parameterizations[i]
            if mode != "linear" and self._user_pinned_mass(i):
                self._raise_derived_mass_pin(i, mode)

            # Per planet, because the coordinate is: a stale log_q entry is
            # only stale for the planets that do not sample log_q.
            if mode != "log_q":
                if log_q_key in up:
                    raise ValueError(
                        f"'{log_q_key}' is set but planet '{self.names[i]}' "
                        f"samples {'a linear mass' if mode == 'linear' else 'm sin i'}"
                        ", which has no log_q. This usually means a params "
                        "file written by mkparam for a log_q fit is being "
                        "reused after the data topology changed. Remove the "
                        "entry, or set 'fitlogq: true' on that planet."
                    )
                continue

            entry = user_entry(up, mass_key)
            if entry is None:
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
                    "sign; set 'fitlogq: false' on the planet if that is "
                    "what you want."
                )

    def _raise_derived_mass_pin(self, index, mode):
        """RAISE for a user ``sigma: 0`` on planet ``index``'s DERIVED mass.

        Names the parameter and the remedy; never moves the pin (see
        ``_reconcile_mass_user_params``).  The default coordinate already
        avoids m sin i when the mass is pinned (``_resolve_mass_parameterization``),
        so reaching here means log_q (explicit, a lens body, or no signed
        observable) or an explicit ``fitmsini: true``.
        """
        name = self.names[index]
        if mode == "log_q":
            what = "log10(m_p/m_host) ('log_q')"
            remedy = (
                f"Pin the sampled coordinate instead -- "
                f"'{self.prefix}.{name}.log_q: {{initval: <log10(m_p/m_host)>, "
                f"sigma: 0}}', both masses in one unit -- or set "
                f"'fitlogq: false' on planet '{name}' to sample the linear "
                "mass, and pin that."
            )
        elif mode == "msini":
            what = "m sin i ('msini')"
            remedy = (
                f"Drop 'fitmsini: true' (or set it false) on planet '{name}' "
                "to sample the linear mass, and pin that.  Pinning msini "
                "would not fix the mass, which is msini / sin i."
            )
        else:
            raise ValueError(
                f"[{self.prefix}.{name}] _raise_derived_mass_pin called for "
                f"mass coordinate {mode!r}, which samples the mass itself."
            )
        raise ValueError(
            f"'{self.prefix}.{name}.mass' has 'sigma: 0', but planet "
            f"'{name}' samples {what} and DERIVES its mass, so a pin on "
            f"planet.mass cannot hold it fixed. {remedy}"
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
        self._warn_chen_only_mass(system)
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

        if not self.keplerian_geometry(system):
            return

        orbits = system.orbit
        self._add_radius_prior(system)
        self._add_msini_jacobian(system, orbits)
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

    def _radius_prior_indices(self):
        """Indices of the planets whose `radius` element is SAMPLED.

        `Parameter.element_is_sampled` is the authoritative per-element
        answer (stage 6 wrote the mask): a user's `sigma: 0` pin and a hard
        link (sigma 0 by construction) are excluded, and so would be a
        derived element, though `planet.radius` has no expression today.
        """
        return [
            i
            for i in range(self.n_elements)
            if self.radius.element_is_sampled(i)
        ]

    def _add_radius_prior(self, system):
        """The base prior on each sampled planet radius: uniform in R_P/R_*.

        EXOFASTv2 steps in p = R_P/R_* with a flat prior; EXOZIPPy samples
        `planet.radius`, whose logit transform is exactly uniform in R_P
        (`Parameter.build_pymc`'s section-C correction).  Uniform in R_P is
        `R_*` times uniform in p on the joint (p, R_*) density, which tilts a
        transit fit toward a larger, less dense star: on HAT-3b EXOZIPPy's p
        came out +0.15 sigma above EXOFASTv2's, and reweighting the posterior
        by 1/R_* closed about half the gap.  At fixed R_*,
        `|dp/dR_P| = 1/R_*`, so this ADDS `-log R_*` of the planet's host
        (`star_map`) and the prior becomes uniform in p.  The term itself is
        `physics.radius_base_prior_logp` -- the one place the base prior is
        chosen (JDE 2026-10-16: match EXOFASTv2 for now; whether uniform in
        log R_P is more physical is open), and the three strings below
        describe it, so a change there changes them.

        WHICH PLANETS: every planet whose radius element is SAMPLED
        (`_radius_prior_indices`).  A pinned radius (`sigma: 0`) gets none:
        the term would be a function of R_* alone, i.e. a spurious prior on
        the star.  Only with an orbit (the caller returns first without one):
        p and every other quantity that reads R_P/R_* exist only then, and
        without an orbit nothing reads the host radius through the planet.

        A user prior on R_P (`mu`/`sigma`) and the Chen & Kipping relation
        (`_add_chen_potential`, a Gaussian in R_P from the mass) are separate
        potentials that MULTIPLY this base prior; they do not replace it --
        EXOFASTv2 likewise applies both as penalties on the derived rp on top
        of its flat-in-p stepping.  So they keep the Jacobian, and the Prior
        cell shows the user's Gaussian AND this term (`get_prior_str` never
        drops an explicit sigma).

        The base prior is against R_P's hard bounds, not p's: where nothing
        constrains R_P (no transit, chen off, no user prior) the 1/R_* factor
        integrates over the fixed R_P interval to a 1/R_* prior on the host,
        which EXOFASTv2's p-bounded stepping does not have.  Such a radius is
        not a measurement of anything; with chen on (an RV- or
        astrometry-only fit) EXOFASTv2 has the same 1/R_* marginal, since the
        relation pins R_P = p R_* at a mass-given value.

        R_* is floored strictly positive inside the log
        (`physics.STAR_RADIUS_LOG_FLOOR`, the CLAUDE.md where-trap rule).
        The property test (the implied prior on (p, R_*) is flat, and is off
        by exactly +log R_* without the term) is `tests/test_radius_prior.py`.
        """
        idx = self._radius_prior_indices()
        if not idx:
            return
        take = np.asarray(idx, dtype="int32")
        star = system.active_components["star"]
        rstar = star.radius.value[self.star_map][take]
        pm.Potential(
            f"{self.prefix}.radius_jacobian",
            pt.sum(physics.radius_base_prior_logp(rstar)),
        )

        names = [self.names[i] for i in idx]
        self.radius.add_prior_contribution(
            latex=r"$\propto 1/R_*$",
            text="uniform in Rp/R* (1/R* Jacobian applied)",
            elements=idx,
            supersedes_bounds=True,
            support_phrase="whose radius support is",
        )
        collector = get_collector(system)
        if collector is not None:
            noun = "planet" if len(names) == 1 else "planets"
            collector.add(
                f"For {noun} "
                + join_names(latex_escape(n) for n in names)
                + r" we sampled $R_P$ and multiplied by $1/R_*$, so that "
                r"the prior is uniform in $R_P/R_*$ rather than in $R_P$, "
                r"as in EXOFASTv2 \citep{Eastman:2019}.",
                section="planetary",
                key=f"{self.prefix}.radius_prior",
            )

    def _msini_indices(self):
        """Indices of the planets sampling m sin i (empty for most systems)."""
        return [
            i
            for i, m in enumerate(self.mass_parameterizations)
            if m == "msini"
        ]

    def _add_msini_jacobian(self, system, orbits):
        """The Jacobian an `msini` planet owes (review 2.14.9).

        Sampling `msini` instead of `mass` at fixed cos i is the change of
        variables `mass = msini / sin i`, `|d mass / d msini| = 1 / sin i`.
        The `linear` coordinate's prior is uniform in (mass, cos i); in
        (msini, cos i) that same prior is `1 / sin i`, so this ADDS
        `-log sin i` and the switch is answer-preserving -- the same
        posterior, sampled in the coordinates an RV curve measures.  Without
        it the prior would be uniform in m sin i, i.e. `p(mass | i) ~ sin i`,
        which down-weights exactly the low-inclination, high-mass tail the
        coordinate change exists to sample.  The direction is checked as a
        property (the implied prior on (mass, cos i) is flat), not argued:
        `tests/test_fitmsini.py`.

        The radicand `1 - cos^2 i` is floored STRICTLY POSITIVE inside
        `physics.msini_log_jacobian`, as it is in the mass expression -- see
        `physics.SINI_RADICAND_FLOOR` and the CLAUDE.md where-trap rule.
        """
        idx = self._msini_indices()
        if not idx:
            return
        take = np.asarray(idx, dtype="int32")
        cosi = orbits.cosi.value[self.orbit_map][take]
        # MINUS log sin i -- see the docstring for why the sign is the term.
        pm.Potential(
            f"{self.prefix}.msini_jacobian",
            -pt.sum(physics.msini_log_jacobian(cosi)),
        )

        names = [self.names[i] for i in idx]
        self.msini.add_prior_contribution(
            latex=r"$\propto 1/\sin{i}$",
            text="uniform in mass and cos i (1/sin i Jacobian applied)",
            elements=idx,
            supersedes_bounds=True,
            support_phrase="whose m sin i support is",
        )
        collector = get_collector(system)
        if collector is not None:
            noun = "planet" if len(names) == 1 else "planets"
            collector.add(
                "The radial velocities constrain the minimum mass "
                r"$M_P\sin{i}$ but not the inclination, so for "
                f"{noun} "
                + join_names(latex_escape(n) for n in names)
                + r" we sampled $M_P\sin{i}$ and $\cos{i}$ and derived "
                r"$M_P$, multiplied by $1/\sin{i}$ so that the prior "
                r"remains uniform in $M_P$ and $\cos{i}$.",
                section="planetary",
                key=f"{self.prefix}.msini",
            )

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

    def chen_only_mass_planets(self, system):
        """Planets whose mass ONLY the Chen relation constrains and whose
        radius start reaches the relation's degenerate Jovian regime.

        Returns ``{index: radius start in Earth radii}``.  Review 2.8.18: in
        a transit-only fit the mass is set by the radius through Chen &
        Kipping, which at and above ~1 RJ is nearly flat in radius with
        kinks at its breaks, so the mass spans decades and NUTS does not mix
        it (SBC: 218 of 300 transit replicates failed convergence on log_q;
        296 of 300 converged with log_q pinned).  The relation is NOT
        smoothed and the coordinate is NOT changed (JDE 2026-10-30); the
        user is told instead.

        Excluded: a planet whose mass something else reads -- RV or
        astrometry (then Chen does not derive the mass), a microlensing
        lens body, or the user's own pin, prior or link on any mass
        coordinate (``planet.mass``/``log_q``/``msini``).  The radius is
        the stage-4 start (``probe_start``), the best a build-time check
        has; the threshold is ``physics.CHEN_DEGENERATE_RPEARTH``.
        """
        lens = system.active_components.get("lens")
        lens_bodies = set(lens.bodies) if lens is not None else set()
        candidates = []
        for p in range(self.n_elements):
            if not self.chen[p] or "mass" not in self.chen_derives[p]:
                continue
            if ("planet", p) in lens_bodies:
                continue
            user_says = False
            for coord in ("mass", "log_q", "msini"):
                for field in ("sigma", "mu"):
                    user_says |= bool(self.user_wrote_field(coord, field)[p])
                for per_elem in self.config_manager.get_element_links(
                    self.prefix, coord
                ).values():
                    user_says |= p in {int(i) for i in per_elem}
            if not user_says:
                candidates.append(p)
        if not candidates:
            return {}
        paths = [f"{self.prefix}.{p}.radius" for p in candidates]
        probed = self.config_manager.probe_start(paths)
        out = {}
        for p, path in zip(candidates, paths):
            start = probed[path]
            if start.value is None:
                raise ValueError(
                    f"[{self.prefix}.{self.names[p]}] the radius has no "
                    f"start at build time ({start.source}); the Chen-only "
                    "mass check needs one."
                )
            rpearth = float(start.value) * RSUN_TO_REARTH
            if rpearth >= physics.CHEN_DEGENERATE_RPEARTH:
                out[p] = rpearth
        return out

    def _warn_chen_only_mass(self, system):
        """Warn per planet found by :meth:`chen_only_mass_planets`."""
        for p, rpearth in self.chen_only_mass_planets(system).items():
            nm = self.names[p]
            logger.warning(
                f"[{self.prefix}.{nm}] the mass is constrained ONLY by the "
                f"Chen & Kipping (2017) mass-radius relation (no RV, "
                f"astrometry or prior measures it), and the radius start "
                f"({rpearth / RJUP_TO_REARTH:.2f} RJ) reaches the relation's "
                f"degenerate Jovian regime (>= "
                f"{physics.CHEN_DEGENERATE_RPEARTH / RJUP_TO_REARTH:.2f} RJ), "
                f"where nearly every mass from Saturn to the hydrogen-burning "
                f"limit predicts the same radius and the regime breaks are "
                f"kinks.  The mass, and everything derived from it, is "
                f"unlikely to mix under NUTS.  {self._chen_pin_advice(p)} "
                f"or give the mass a prior (mu/sigma)."
            )

    def _chen_pin_advice(self, index):
        """The pin to recommend: on the SAMPLED mass coordinate, because a
        sigma: 0 on a derived planet.mass raises (2.1.26(e))."""
        nm = self.names[index]
        mode = self.mass_parameterizations[index]
        if mode == "linear":
            return f"Pin it ({self.prefix}.{nm}.mass: {{initval: <value>, sigma: 0}})"
        if mode == "log_q":
            return (
                f"Pin its sampled coordinate ({self.prefix}.{nm}.log_q: "
                f"{{initval: <log10(m_p/m_host)>, sigma: 0}}; or set "
                f"'fitlogq: false' and pin {self.prefix}.{nm}.mass)"
            )
        return (
            f"Pin its sampled coordinate ({self.prefix}.{nm}.msini: "
            f"{{initval: <value>, sigma: 0}})"
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
