"""The microlensing EVENT component (8.6.17 stage 1).

One instance, one event: the encounter between one co-moving lens system and
one co-moving source system (ruling R1).  Everything that is a property of
the ENCOUNTER -- the relative parallax, proper motion, Einstein radius and
crossing time, the parallax vector, the total lens mass -- lives here with
one element each; the per-body geometry lives on the `lens` component (one
entry per lens body, primary first) and the per-source trajectory offsets on
the `source` component (one entry per source body).

This component also owns everything event-scoped that is not a parameter:
the magnification dispatcher (`get_magnification`/`get_magnification_op`,
one call per source trajectory over the shared lens bodies), the event-level
config keys (finite_source, t0_par, backend, mag_method, use_op, the fit*
coordinate flags, peak_find), the event potentials
(event rate, singularity guards, source-behind-lens, the fitpirel
Jacobian), the seeding hints, and the sampler-compatibility declaration.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.orbit.bodies import (
    STATIC_TYPE,
    orbit_types,
    parse_orbit_bodies,
)
from exozippy.components.star.position import probe_star_radec
from exozippy.config import (
    PRECEDENCE_DERIVED_DATA,
    PRECEDENCE_DERIVED_MIXED,
    PRECEDENCE_MULENS_LENS_DISTANCE,
    PRECEDENCE_MULENS_SOURCE_DISTANCE,
    user_entry,
)
from exozippy.constants import DAYS_PER_YEAR
from exozippy.corner_utils import (
    collect_parameter_corner_samples,
    save_corner_plot,
)
from exozippy.outputs.prose import get_collector
from exozippy.potentials import soft_lower_bound
from exozippy.skyframe import observer_sky_offset, parallax_factors, sky_basis

from ..galacticmodel.physics import expected_proper_motion
from .bodies import body_entries, resolve_orbit_ref, validate_event_config
from .op import BinaryLensMagOp, MulensMagOp, VBMDirectMagOp
from .physics import (
    MU_REL_FLOOR,
    T_E_FLOOR,
    THETA_E_FLOOR,
    THETA_E_LENSING_MIN,
    apply_u_0_floor,
    clip_q,
    host_offset_from_com,
    lens_geometry_from_orbit,
    source_offset_from_orbit,
    xallarap_trajectory_shift,
)

logger = logging.getLogger(__name__)

# Central-difference step (days) for the Earth's velocity at t0_par, which
# spans the Skowron frame (conventions.md C5).  A CONVENTION, not a numerical
# detail: the velocity cancels exactly in u(t) but sets the reported t_E and
# the direction of pi_E, so a (t_0, u_0) is comparable across codes only in
# a frame built with the same step.  (omega h)^2/6 = 1.2e-7 relative here;
# it was 0.5 d (1.2e-5) before review 1.10.13.  Module-level so a test can
# reproduce the pre-1.10.13 frame.
FRAME_VELOCITY_STEP_DAYS = 0.05

# alpha is stored in radians (lens.alpha's internal_unit) and consumed in
# degrees by both magnification backends.
_RAD_TO_DEG = 180.0 / np.pi


class MulensEvent(Component):
    """Event-level microlensing parameters and configuration.

    YAML shape (the event options that lived on the old lens block)::

        mulensevent:
          - finite_source: false
            t0_par: 2455379.571
        lens:
          - body: star.Lens          # primary first; companions may be
          - body: planet.b           #   planets or stars
        source:
          - body: star.Source        # star-only

    Under ruling R1 (co-moving systems only, indefinitely) there is exactly
    one relative proper-motion vector per event, so `mu_rel`, `t_E`,
    `theta_E`, `pi_E` and `pi_rel` are event-level scalars (shape (1,), the
    natural shape of a one-instance component).  The event-level source
    distance/pm representative is SOURCE BODY 0's star (design 11.3);
    likewise the lens side resolves through the primary lens star.
    """

    # Microlensing degeneracies are structural (see the note on Lens);
    # samplers consult any component that declares this.
    expects_suppressed_modes = True

    # Earth's velocity at t0_par, projected E/N -- constants injected in
    # add_parameter for the mu_rel helio -> geo conversion (Gould 2004).
    # The lens_offset_e/n quartet (review 8.10.4) likewise: the astrometric
    # reference epoch, t0_par, and the Earth's parallax factors at t0_par.
    context_dep_names = frozenset(
        {
            "earth_vperp_e",
            "earth_vperp_n",
            "track_epoch",
            "track_t0_par",
            "track_earth_pe",
            "track_earth_pn",
        }
    )

    def __init__(self, config, config_manager):
        # Refuse unknown/misplaced keys BEFORE anything reads the block: a
        # per-source flag (fitu0te, star_constrains_rho) or a per-companion
        # key (orbital_motion, orbit) left here would otherwise be silently
        # ignored while the user believes it is in effect (R3: hard breaks
        # fail loudly and instructively).
        validate_event_config(config)
        super().__init__(config, config_manager)
        self.label = "Microlensing Event"

        if self.n_elements > 1:
            raise ValueError(
                "Only one lensing event may be modeled at a time: define a "
                "single mulensevent block.  Multiple lens or source BODIES "
                "are entries of the 'lens:'/'source:' component lists."
            )

        sys_cfg = getattr(config_manager, "system_config", None) or {}
        if "lens" not in sys_cfg or "source" not in sys_cfg:
            missing = [k for k in ("lens", "source") if k not in sys_cfg]
            raise ValueError(
                f"mulensevent: the config must also declare "
                f"{' and '.join(repr(k) for k in missing)} -- one entry per "
                f"body, e.g. lens: [{{body: star.Lens}}], "
                f"source: [{{body: star.Source}}]."
            )
        # Raw-config reads, not component attributes: component order within
        # a stage is not guaranteed (the star.py discipline), and __init__
        # order is the user's config key order.
        self.lens_bodies = body_entries(sys_cfg["lens"], "lens", sys_cfg)
        self.source_bodies = body_entries(sys_cfg["source"], "source", sys_cfg)
        if not self.lens_bodies or not self.source_bodies:
            raise ValueError(
                "mulensevent: 'lens:' and 'source:' need at least one body "
                "entry each."
            )
        self.n_companions = len(self.lens_bodies) - 1
        self.n_sources = len(self.source_bodies)

        ev = self.config[0]
        self.finite_source = bool(ev.get("finite_source", False))

        # Preliminary t0_par; MulensInstrument re-resolves the final value
        # at stage 1 (the peak finder's seeds arrive after this snapshot) and writes
        # it back into t0_par[0] -- a length-1 list precisely so that write
        # is visible to every later reader of the same object.
        self.t0_par = [self._resolve_t0_par(ev, config_manager)]
        # The source's line of sight (ra, dec in radians), set by
        # MulensInstrument.load_data before the geocentric frame is first
        # anchored: the frame's Earth is placed at the instant it received
        # the light tagged t0_par, which depends on the direction.
        self._line_of_sight_rad = None

        # One magnification method per source; all sources start from the
        # event-level config value, and resolve_auto_vbbl refines each slot.
        event_method = ev.get(
            "mag_method",
            (
                "auto_vbbl"
                if (self.finite_source or self.n_companions >= 1)
                else "point_source"
            ),
        )
        self.mag_method = [event_method] * self.n_sources

        self.use_op = bool(ev.get("use_op", False))

        self.backend = ev.get("backend", "vbm_direct")
        self._warned_linear_ld_backend = False
        if self.backend not in ("vbm_direct", "mulensmodel"):
            raise ValueError(
                f"mulensevent.backend must be 'vbm_direct' or "
                f"'mulensmodel', got '{self.backend}'."
            )

        # Source orbital motion -- xallarap (C25), PER SOURCE (review 8.6.9,
        # the source twin of 8.6.33): each source entry's `orbit:` names the
        # orbit it moves on, exactly as a lens companion's does; a source
        # without one is static beside moving ones.  The orbit's `type:` says
        # HOW it moves, and only keplerian is offered: a linear source drift
        # is degenerate with (t_E, t_0, u_0, alpha) in the light curve alone
        # (C25) -- physics, not plumbing.  A source sits ALONE on one side of
        # its orbit; on the primary side its offset is the orbit's primary
        # track, a1 = a m_companion / m_total, on the companion side the
        # opposite track, a2 = -a m_primary / m_total (source_offset_from_
        # orbit is linear in its amplitude).  Two sources on ONE orbit, one
        # per side, are the linked binary source.  A source inside a group
        # of bodies is a hierarchical system (8.6.33 phase 2) and raises.
        self.source_orbit_idx = []  # per source slot: orbit index or None
        self.source_orbit_side = []  # +1 primary side, -1 companion side
        src_block = sys_cfg.get("source") or []
        refs = [
            e.get("orbit") if isinstance(e, dict) else None for e in src_block
        ]
        types = orbit_types(sys_cfg)
        prim = comp = None
        if any(r is not None for r in refs):
            prim, comp = parse_orbit_bodies(sys_cfg.get("orbit"), sys_cfg)
        for j, ref in enumerate(refs):
            if ref is None:
                self.source_orbit_idx.append(None)
                self.source_orbit_side.append(None)
                continue
            o = resolve_orbit_ref(config_manager, ref, f"source.{j}")
            t = types[o] if o < len(types) else "keplerian"
            if t == STATIC_TYPE:
                raise ValueError(
                    f"source.{j}: orbit {ref!r} is type: static, a pairing "
                    "with no dynamics, so it cannot move this source.  A "
                    "source that does not move carries no 'orbit:' key."
                )
            if t != "keplerian":
                raise NotImplementedError(
                    f"source.{j}: orbit {ref!r} is type: {t}.  Linear (or "
                    "quadratic) xallarap is deliberately not offered: a "
                    "low-order source drift is exactly degenerate with "
                    "(t_E, t_0, u_0, alpha) in the light curve alone "
                    "(conventions.md C25).  Use a keplerian source orbit."
                )
            body = tuple(self.source_bodies[j])
            p_grp = [tuple(b) for b in prim[o]]
            c_grp = [tuple(b) for b in comp[o]]
            if p_grp == [body]:
                side = 1
            elif c_grp == [body]:
                side = -1
            else:
                raise NotImplementedError(
                    f"source.{j}: orbit {ref!r} has primary group {p_grp} "
                    f"and companion group {c_grp}; a source must be ALONE "
                    f"on one side of its orbit (a source inside a group of "
                    f"bodies is a hierarchical system, review 8.6.33 phase "
                    f"2)."
                )
            self.source_orbit_idx.append(o)
            self.source_orbit_side.append(side)

        by_orbit = {}
        for j, o in enumerate(self.source_orbit_idx):
            if o is not None:
                by_orbit.setdefault(o, []).append(j)
        for o, js in by_orbit.items():
            if len(js) == 2 and (
                self.source_orbit_side[js[0]] == self.source_orbit_side[js[1]]
            ):
                raise ValueError(
                    f"[{self.prefix}] sources {js} name the same orbit from "
                    f"the same side."
                )
            # The other side of a source's orbit may be a dark body or the
            # OTHER source of a binary -- but then that source must name
            # this orbit too, or it would be modeled static while the orbit
            # says it moves.
            for j in js:
                other = comp[o] if self.source_orbit_side[j] == 1 else prim[o]
                for k, sb in enumerate(self.source_bodies):
                    if [tuple(sb)] == [
                        tuple(b) for b in other
                    ] and k not in js:
                        raise ValueError(
                            f"[{self.prefix}] source.{j}'s orbit "
                            f"{refs[j]!r} has source.{k} on its other side, "
                            f"but source.{k} does not name that orbit: put "
                            f"'orbit: {refs[j]}' on source.{k} too (a "
                            f"binary source moves both sources)."
                        )
        self.xal_orbit_idxs = sorted(by_orbit)

        # The lens binary and a source binary are different systems, so a
        # lens companion's `orbit:` and a source's must not name the same
        # orbit.  Resolved from the raw lens block: component construction
        # order is the user's config key order, so the Lens instance may not
        # exist yet.
        if self.xal_orbit_idxs:
            lens_refs = {
                resolve_orbit_ref(config_manager, e["orbit"], f"lens.{i}")
                for i, e in enumerate(sys_cfg.get("lens") or [])
                if isinstance(e, dict) and e.get("orbit") is not None
            }
            shared = lens_refs & set(self.xal_orbit_idxs)
            if shared:
                raise ValueError(
                    f"[{self.prefix}] a lens companion's and a source's "
                    f"`orbit:` name the SAME orbit {sorted(shared)}; the "
                    "lens binary and the source binary are different "
                    "systems."
                )

    def _resolve_t0_par(self, event_config, config_manager):
        """t0_par from the mulensevent block, the source.0.t_0 seed, or the
        historical fallback.  MulensInstrument re-resolves the final value
        at stage 1 (the peak finder's seeds arrive after this snapshot)."""
        if "t0_par" in event_config:
            return float(event_config["t0_par"])
        entry = user_entry(config_manager.user_params, "source.0.t_0")
        val = None if entry is None else entry.get("initval")
        # List-valued initval (P4 multi-seed sampling): t0_par is just a
        # numeric reference epoch, not a per-seed value, so use seed 0.
        if isinstance(val, (list, tuple)):
            val = val[0]
        return float(val) if val is not None else 2450000.0

    @property
    def prefix(self):
        return "mulensevent"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "finite_source",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": "Model finite-source effects. Default false.",
            },
            {
                "key": "t0_par",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Fiducial epoch anchoring the geocentric frame "
                    "(Skowron+2011). Defaults to the source.0.t_0 start, "
                    "the peak finder's t_0, or the median data time."
                ),
            },
            {
                "key": "fitmurel",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Sample the heliocentric lens-source relative proper "
                    "motion (mu_ra_rel/mu_dec_rel) directly -- the "
                    "combination the light curve measures -- and derive "
                    "the lens star's pm as pm_source + mu_rel, instead of "
                    "sampling both stars' pm and deriving mu_rel. A "
                    "coordinate choice like fitvcve: same joint density "
                    "(|J| = 1), better-conditioned axes. Default False."
                ),
            },
            {
                "key": "fitpirel",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Sample the lens-source relative parallax directly (as "
                    "log_pi_rel) -- the combination the light curve "
                    "measures -- and derive the lens star's distance as "
                    "D_l = 1000/(pi_rel + 1000/D_s), automatically inside "
                    "(0, D_s). A coordinate choice like fitvcve; the "
                    "nonlinear map's Jacobian potential is added in "
                    "build_likelihood so the joint density is unchanged. "
                    "Default False."
                ),
            },
            {
                "key": "fitthetae",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Sample the Einstein radius directly (as log_theta_E) "
                    "and derive the HOST star's logmass from "
                    "theta_E^2 = kappa * M_tot * pi_rel (M_host = "
                    "M_tot/(1+q) with a log_q companion). A coordinate "
                    "choice like fitvcve; log-linear map, constant "
                    "Jacobian, no correction potential. At most one "
                    "companion, which must carry a sampled log_q. "
                    "Default False."
                ),
            },
            {
                "key": "peak_find",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Built-in point-lens peak finder for t_0, u_0 and t_E, "
                    "the DEFAULT seeder. Absent (default): runs when one "
                    "of the three has no start from the params file "
                    "(directly or derived); it holds every given value and "
                    "fits only the missing ones. true always runs it on all "
                    "three; false never "
                    "does. It fits a PSPL model to "
                    "the light curves with the source and blend fluxes "
                    "profiled out analytically, and seeds only those three: "
                    "s, alpha, q and rho keep their defaults and the "
                    "sampler finds the anomaly itself. 'auto' is no longer "
                    "accepted (omit the key)."
                ),
            },
            {
                "key": "mag_method",
                "kind": "option",
                "accepts": ["point_source", "auto_vbbl"],
                "required": False,
                "doc": (
                    "Magnification method. Defaults to 'auto_vbbl' for "
                    "finite-source or multi-body lenses, else 'point_source'."
                ),
            },
            {
                "key": "backend",
                "kind": "option",
                "accepts": ["vbm_direct", "mulensmodel"],
                "required": False,
                "doc": (
                    "Magnification engine for the Op path: binary and "
                    "N-lens events and every finite-source single lens. "
                    "Default 'vbm_direct' (VBMicrolensing, linear or "
                    "quadratic limb darkening); 'mulensmodel' is the A/B "
                    "reference, at most 2 lens bodies, and is linear-LD "
                    "only. Its finite-source single lens uses MulensModel's "
                    "Yoo04 where u/rho <= 99 (the end of Yoo04's table) and "
                    "a point source beyond, which is within 0.014 mmag "
                    "there."
                ),
            },
            {
                "key": "use_op",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": (
                    "Force the MulensModel Op even for point-source PSPL "
                    "(default false uses the NUTS-friendly symbolic path)."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _primary_lens(self):
        """(comp_type, star_ndx) of the primary lens body."""
        return self.lens_bodies[0]

    def _primary_source(self):
        """(comp_type, star_ndx) of the primary source body."""
        return self.source_bodies[0]

    def lens_track_star(self, system):
        """The primary lens STAR whose sky position is derived from the
        source's (review 8.10.4), or None.

        The light curve fixes the lens-minus-source separation at every
        instant (C9), so a lens star's ``ra``/``dec`` at the astrometric
        reference epoch are not free: they are the primary source's plus
        ``mulensevent.lens_offset_e/n``.  The flip happens exactly where
        that position is MEASURED -- the lens star is the target of a
        gaia/abs astrometry dataset -- and nowhere else: a lens position
        nothing reads is a pinned bookkeeping value, and deriving it would
        only add a potential to every microlensing fit.

        Stage-safe from stage 1 on: reads the raw astrometry config and
        ``lens_bodies``/``source_bodies`` (``__init__``), never a stage-2
        attribute, because AstrometryInstrument.load_data asks it too.
        """
        l_type, l_idx = self._primary_lens()
        s_type, _ = self._primary_source()
        if l_type != "star" or s_type != "star":
            return None
        astrom = getattr(system, "astrometryinstrument", None)
        if astrom is None:
            return None
        for i, c in enumerate(astrom.config):
            mode = c.get("mode", "gaia")
            # astrometryinstrument.ABSOLUTE_MODES, spelled out (import
            # cycle).  A lens named in a dataset's psf_stars (review 8.10.3)
            # is measured too: its track moves that blended photocenter.
            # psf_stars is resolved to indices at AstrometryInstrument
            # construction, so this stays stage-safe.
            if mode in ("gaia", "abs", "catalog") and (
                int(c.get("star_ndx", 0)) == int(l_idx)
                or int(l_idx) in astrom.psf_stars[i]
            ):
                return int(l_idx)
        return None

    def _validate_lens_track(self, system):
        """Refuse the configurations the derived lens position cannot
        represent, and say what it approximates.  Called at stage 3 when
        ``lens_track_star`` is not None."""
        if self.xal_orbit_idxs:
            raise NotImplementedError(
                f"[{self.prefix}] the lens star carries absolute astrometry, "
                f"so its position is derived from the source's and the "
                f"event geometry (review 8.10.4) -- but this event has a "
                f"source on an orbit (xallarap), and the source's orbital offset "
                f"at the reference epoch is not part of that derivation "
                f"yet.  Drop the lens's astrometry or the source orbit."
            )
        if self.n_companions >= 2:
            raise NotImplementedError(
                f"[{self.prefix}] the lens star carries absolute astrometry, "
                f"so its position is derived from the event geometry "
                f"(review 8.10.4), which places the luminous primary relative "
                f"to the lens center of mass for ONE companion only; this "
                f"lens has {self.n_companions}."
            )
        if self.n_companions == 1 and self.lens_bodies[1][0] == "star":
            logger.warning(
                f"[{self.prefix}] the lens star's position is derived from "
                f"the event geometry (review 8.10.4) and the lens companion "
                f"is a STAR: the derivation places the primary at its offset "
                f"from the center of mass at t0_par, while a luminous "
                f"companion moves the light's photocenter and the binary "
                f"orbit moves both over the years to the imaging epoch (up "
                f"to s theta_E).  Treat the lens astrometry as approximate."
            )

    def check_lens_offset_supported(self, where):
        """Gate for the resolved lens-minus-source astrometry ('lensrel'):
        the luminous primary is placed relative to the lens center of mass
        for at most one companion (``host_offset``)."""
        if self.n_companions >= 2:
            raise NotImplementedError(
                f"[{where}] resolved lens-source astrometry places the "
                f"luminous primary relative to the lens center of mass for "
                f"ONE companion only; this lens has {self.n_companions}."
            )
        if self.n_companions == 1 and self.lens_bodies[1][0] == "star":
            logger.warning(
                f"[{where}] the lens companion is a STAR: the model places "
                f"the primary at its offset from the center of mass at "
                f"t0_par, while a luminous companion moves the light's "
                f"photocenter and the binary orbit moves both over the years "
                f"to the imaging epoch (up to s theta_E)."
            )

    def host_offset(self, system):
        """The luminous primary's offset from the lens center of mass, on
        C9's ``(tau_hat, beta_hat)`` axes in Einstein radii (C12/C15), from
        the companion geometry at t0_par: ``(0, 0)`` for a single lens.
        The tensor twin of the ``binary`` expression of
        ``lens_offset_e/n``; both call ``physics.host_offset_from_com``."""
        if self.n_companions == 0:
            return 0.0, 0.0
        if self.n_companions > 1:
            raise NotImplementedError(
                f"[{self.prefix}] host_offset supports one companion; this "
                f"lens has {self.n_companions} (check_lens_offset_supported "
                f"should have refused it)."
            )
        lens = system.lens
        return host_offset_from_com(
            lens.s.value[1], lens.alpha.value[1], lens.q.value[1]
        )

    def _lens_track_constants(self, system):
        """The context constants of lens_offset_e/n: the astrometric
        reference epoch, t0_par, and the Earth's parallax factors at
        t0_par -- from the same frame anchor (``geocentric_frame``) the
        light curve's deviations use, so the derived offset and the
        trajectory share one frame by construction."""
        _t0, earth_pos, _vel = self.geocentric_frame()
        ra, dec = self._line_of_sight_rad
        p_e, p_n = parallax_factors(np.asarray(earth_pos)[None, :], ra, dec)
        return {
            "track_epoch": float(system.astrometryinstrument.epoch),
            "track_t0_par": float(self.t0_par[0]),
            "track_earth_pe": float(p_e[0]),
            "track_earth_pn": float(p_n[0]),
        }

    def _validate_bodies(self, system):
        """Fail at registration time if a body reference points to a
        component or instance that does not exist, if the PRIMARY lens body
        is not a star, if any SOURCE body is not a star, or if a body is on
        both sides.  Ported from the pre-split Lens._validate_bodies; the
        physics arguments are unchanged."""
        for role, bodies in (
            ("lens", self.lens_bodies),
            ("source", self.source_bodies),
        ):
            for comp_type, ndx in bodies:
                comp = getattr(system, comp_type, None)
                if comp is None:
                    raise ValueError(
                        f"mulensevent: {role} body '{comp_type}.{ndx}' "
                        f"refers to component '{comp_type}', but no "
                        f"'{comp_type}' block exists in the config."
                    )
                if ndx >= comp.n_elements:
                    raise ValueError(
                        f"mulensevent: {role} body '{comp_type}.{ndx}' is "
                        f"out of range: only {comp.n_elements} "
                        f"'{comp_type}' instance(s) are configured."
                    )

        # The primary lens body must be a star.  The lens maps carry only
        # the INDEX, and every primary-side dependency is hard-coded to the
        # star component (star.mass[lens_map], star.distance[lens_map],
        # star.pm_ra/pm_dec[lens_map], build_likelihood's d_l), so a
        # 'planet.0' primary silently models star.0 instead: measured on
        # examples/ob08092, a planet primary built a theta_E bit-identical
        # to the star's, responded to that star's mass, and was completely
        # insensitive to the planet's.  Companions ARE type-aware (their
        # mass deps carry the component type), so only this slot is
        # restricted.
        p_type, p_ndx = self._primary_lens()
        if p_type != "star":
            raise ValueError(
                f"mulensevent: the primary (first) lens body is "
                f"'{p_type}.{p_ndx}', but it must be a star.  The lens "
                f"maps carry only an index, and the lens-side physics "
                f"resolves the primary through star.mass / star.distance "
                f"/ star.pm_ra / star.pm_dec, so a non-star primary would "
                f"silently model star.{p_ndx} instead of "
                f"'{p_type}.{p_ndx}' and report a lens mass that never "
                f"entered the likelihood.  Planet COMPANIONS are "
                f"supported -- put the star first (lens: "
                f"[{{body: star.<name>}}, {{body: {p_type}.{p_ndx}}}]).  "
                f"To model a very low-mass (even planetary-mass) lens, "
                f"declare it as a 'star' block with a low "
                f"star.<name>.logmass instead; logmass reaches -9 dex "
                f"(1e-9 solMass).  For a FREE-FLOATING planet that is "
                f"only half the recipe: give that star block "
                f"'mass_function: ffp' as well, or it draws the stellar "
                f"IMF and is penalized for having the mass you told it "
                f"to have."
            )

        # EVERY source body must be a star -- source_map is index-only
        # exactly like lens_map, and the whole source-side chain resolves
        # through the star component (star.distance/pm_ra/pm_dec/radius/
        # ra/dec).  A planet is not a self-luminous point source at bulge
        # distances, so a non-star source is physically meaningless rather
        # than merely unimplemented.
        for s_type, s_idx in self.source_bodies:
            if s_type != "star":
                raise ValueError(
                    f"mulensevent: source body '{s_type}.{s_idx}' must be "
                    f"a star -- a microlensing source is the background "
                    f"star being monitored for magnification.  Because the "
                    f"map carries only an index, this config would "
                    f"silently model star.{s_idx} instead of "
                    f"'{s_type}.{s_idx}'.  A genuinely faint source (a "
                    f"brown dwarf, say) is a 'star' block with a low "
                    f"star.<name>.logmass."
                )

        # A body cannot lens itself: pi_rel = 1000/d_L - 1000/d_S is then
        # identically 0, theta_E collapses onto its floor and the
        # likelihood is NaN at the first evaluation.
        shared = [b for b in self.lens_bodies if b in self.source_bodies]
        if shared:
            shared_txt = ", ".join(f"'{t}.{n}'" for t, n in shared)
            raise ValueError(
                f"mulensevent: {shared_txt} is listed as BOTH a lens body "
                f"and a source body.  A lens and its source must be "
                f"distinct objects at different distances: with the same "
                f"body on both sides, pi_rel = 1000/d_L - 1000/d_S is "
                f"identically 0, so theta_E is 0 and the likelihood is "
                f"NaN from the first evaluation.  Give the lens and the "
                f"source separate 'lens:'/'source:' entries."
            )

    # ------------------------------------------------------------------
    # Lifecycle stages
    # ------------------------------------------------------------------

    def build_maps(self):
        """Stage 2: index maps for the event-level parameter chain.

        All length 1 (the natural shape of a one-instance component).
        lens_map carries TWO conceptually different roles that happen to
        share one index: the LENSING MASS (star.mass[lens_map] -> theta_E)
        and the KINEMATIC HOST (star.distance/pm_ra/pm_dec[lens_map] ->
        pi_rel and mu_rel).  They coincide only because the primary lens
        body is always a star, which _validate_bodies enforces.  The
        conflation is exactly what let the silent planet-primary bug
        through: a planet has a mass but no distance or proper motion of
        its own, so a planet primary resolved role 1 to the planet (had the
        deps been typed) and role 2 to whatever star sat at the same index.
        Splitting the two would mean inventing a "kinematic host star" for
        a body that by definition has no host -- which is why
        planet-as-lens was abandoned in favor of declaring a low-mass lens
        as a star.  Under the guards the roles can never diverge, so this
        stays one index; the note is here so the next reader does not have
        to rediscover why.  source_map is the PRIMARY source's star: under
        R1 the source SYSTEM's barycentric kinematics are represented by
        body 0 (design 11.3).
        """
        _, p_ndx = self._primary_lens()
        _, s_ndx = self._primary_source()
        self.lens_map = np.array([p_ndx], dtype=int)
        self.source_map = np.array([s_ndx], dtype=int)
        self.primary_lens_map = np.array([p_ndx], dtype=int)
        # One scalar map per companion: companions may live in different
        # component types (star vs planet), so each mass needs its own
        # bracket dep (mlens_total's deps).
        for j, (_, c_ndx) in enumerate(self.lens_bodies[1:]):
            setattr(
                self, f"companion{j}_mass_map", np.array([c_ndx], dtype=int)
            )
        # The derived lens position's deps (lens_offset_e/n): the primary
        # source's element of the `source` component, and the first
        # companion's element of the `lens` component (element 0 is the
        # primary, masked inactive there).
        self.primary_source_element_map = np.array([0], dtype=int)
        if self.n_companions >= 1:
            self.first_companion_map = np.array([1], dtype=int)

    def register_parameters(self, system):
        """Stage 3: Declare the event-level manifest and push hints."""
        self._validate_bodies(system)
        # The geocentric frame's constants for the relaxation engine's t_E
        # and pi_E relations (review 8.6.23); MulensInstrument supplied them
        # at stage 1 already where there are data.
        self.register_frame_constants(system)
        # No seed push here: the one seeder (the peak finder) runs in
        # MulensInstrument.load_data at stage 1, where the seeds must land
        # for its flux bootstrap.  A stage-3 re-push of a seed file once
        # lived here and silently undid `peak_find: true` (reviews 1.6.15,
        # 2.1.12); add_seed_hints now raises on a second registration.

        # fitmurel (coordinate choice, fitvcve family): sample the
        # LC-measured relative proper motion directly; the star component
        # reads this flag and derives the LENS star's pm = pm_source +
        # mu_rel (|J| = 1; see star.py).
        fitmurel = bool(self.config[0].get("fitmurel", False))

        def murel_entry():
            if not fitmurel:
                return {"expr_key": "default"}
            # Sampled mode: start at 0 relative pm through the overrides
            # channel (defaults.yaml deliberately carries no initval --
            # the engine's default-armor would seed mu_rel = 0 into every
            # config and break the t_E derivation chain), with a
            # preliminary 1 mas/yr whitening scale.  The engine's mu_rel
            # relation upgrades the start from any pm/t_E seeds.
            return {"overrides": {"initval": [0.0]}}

        if fitmurel:
            self.config_manager.add_scale_hint("mulensevent.0.mu_ra_rel", 1.0)
            self.config_manager.add_scale_hint("mulensevent.0.mu_dec_rel", 1.0)

        # fitpirel (swap 2): sample log_pi_rel, derive pi_rel from it, and
        # the star component derives the lens distance (see star.py).
        # The pre-split "single-source only" refusal is GONE: pi_rel is
        # event-level by construction now, so per-source overdetermination
        # is unrepresentable (design 1.1 #7).
        fitpirel = bool(self.config[0].get("fitpirel", False))
        self._fitpirel = fitpirel
        if fitpirel:
            self.config_manager.add_scale_hint("mulensevent.0.log_pi_rel", 0.1)

        # fitthetae (swap 3): sample log_theta_E, derive theta_E from it,
        # and the star component derives the HOST logmass (see star.py).
        # The n_sources guard is gone for the same reason as fitpirel's;
        # the companion checks survive (the inverse needs M_host =
        # M_tot/(1+q) with q available as a coordinate).
        fitthetae = bool(self.config[0].get("fitthetae", False))
        if fitthetae:
            reason = None
            if self.n_companions > 1:
                reason = "more than one companion is not supported"
            elif self.n_companions == 1:
                c_type, c_idx = self.lens_bodies[1]
                if c_type != "planet":
                    reason = f"the companion is a '{c_type}', not a planet"
                else:
                    comp = getattr(system, "planet", None)
                    if comp is not None:
                        from ..planet.planet import (
                            lens_companion_samples_log_q,
                        )

                        if not lens_companion_samples_log_q(
                            comp.config[c_idx], f"planet.{comp.names[c_idx]}"
                        ):
                            reason = (
                                "the companion samples a linear mass or "
                                "m sin i, not log_q"
                            )
            if reason is not None:
                logger.warning(
                    f"mulensevent: fitthetae is set but {reason}; ignoring it."
                )
                fitthetae = False
        self._fitthetae = fitthetae
        if fitthetae:
            self.config_manager.add_scale_hint(
                "mulensevent.0.log_theta_E", 0.1
            )

        self.manifest = {
            "pi_rel": {
                "expr_key": "from_log_pi_rel" if fitpirel else "default"
            },
            "theta_E": {
                "expr_key": "from_log_theta_E" if fitthetae else "default"
            },
            "mu_ra_rel": murel_entry(),
            "mu_dec_rel": murel_entry(),
            **({"log_pi_rel": None} if fitpirel else {}),
            **({"log_theta_E": None} if fitthetae else {}),
            "mu_rel_mag": {"expr_key": "default"},
            "mu_ra_rel_geo": {"expr_key": "default"},
            "mu_dec_rel_geo": {"expr_key": "default"},
            "mu_rel_geo_mag": {"expr_key": "default"},
            "t_E": {"expr_key": "default"},
            "pi_E_N": {"expr_key": "default"},
            "pi_E_E": {"expr_key": "default"},
        }

        # The lens star's position, derived from the source's where it is
        # measured (review 8.10.4; Star flips the lens's ra/dec onto these).
        if self.lens_track_star(system) is not None:
            self._validate_lens_track(system)
            key = "binary" if self.n_companions == 1 else "default"
            self.manifest["lens_offset_e"] = {"expr_key": key}
            self.manifest["lens_offset_n"] = {"expr_key": key}

        # The geocentric-frame caveat lives in ONE shared table note (the
        # note_marks dedup collapses identical texts to one letter).
        geo_note = (
            r"Geocentric quantities are evaluated at the fiducial time "
            r"$t_{0,\rm par}$ \citep{Gould:2004, Skowron:2011}."
        )
        for name in (
            "t_E",
            "mu_ra_rel_geo",
            "mu_dec_rel_geo",
            "mu_rel_geo_mag",
        ):
            self.manifest[name]["table_note"] = geo_note

        if self.n_companions >= 1:
            # Multi-lens convention: theta_E (and hence t_E, rho, pi_E) is
            # referenced to the TOTAL lens mass, matching the published
            # parameterization.  Companion component types vary by config,
            # hence one scalar bracket dep per companion.
            companion_mass_deps = [
                f"{c_type}.mass[companion{j}_mass_map]"
                for j, (c_type, _) in enumerate(self.lens_bodies[1:])
            ]
            self.manifest["mlens_total"] = {
                "expr_key": "default",
                "deps": ["star.mass[primary_lens_map]"] + companion_mass_deps,
            }
            if not fitthetae:
                self.manifest["theta_E"] = {
                    **self.manifest["theta_E"],
                    "deps": ["mlens_total", "pi_rel"],
                }

        # Expected proper motions from the galactic model, for the seeds
        # below.  None when the line of sight is not known yet, in which
        # case the pm hints are simply skipped.
        pm_expected = self._galactic_pm_expectations(system)

        # Inject event-level physical hints for the body stars.
        l_type, l_idx = self._primary_lens()

        # PRECEDENCE_MULENS_LENS_DISTANCE overrides the 10 pc defaults.yaml
        # default (PRECEDENCE_DEFAULT) but yields to any value the
        # relaxation engine derives from pi_rel + d_S
        # (PRECEDENCE_MULENS_SOURCE_DISTANCE).  That ordering is what breaks
        # the d_L <-> parallax cycle; see the constants' comment in
        # config.py.
        self.config_manager.add_hint(
            f"star.{l_idx}.distance",
            4000.0,
            rank=PRECEDENCE_MULENS_LENS_DISTANCE,
        )
        self.config_manager.add_scale_hint(f"star.{l_idx}.distance", 5.0)
        self.config_manager.add_hint(f"star.{l_idx}.logmass", -0.5)
        self.config_manager.add_scale_hint(f"star.{l_idx}.logmass", 0.001)
        self.config_manager.add_scale_hint(f"star.{l_idx}.pm_ra", 3.0)
        self.config_manager.add_scale_hint(f"star.{l_idx}.pm_dec", 3.0)
        self.config_manager.add_scale_hint(f"star.{l_idx}.rv", 1e5)
        self._seed_expected_pm(pm_expected, l_idx, "thin_disk", 4000.0, system)

        # Every source body gets the same bulge-source seeding: each source
        # star's distance and pm still get their own start (the secondary's
        # distance feeds ITS rho even though it no longer feeds mu_rel).
        for s_type, s_idx in self.source_bodies:
            self.config_manager.add_hint(
                f"star.{s_idx}.distance",
                8000.0,
                rank=PRECEDENCE_MULENS_SOURCE_DISTANCE,
            )
            self.config_manager.add_scale_hint(f"star.{s_idx}.distance", 5.0)
            self.config_manager.add_hint(f"star.{s_idx}.logmass", -0.5)
            self.config_manager.add_scale_hint(f"star.{s_idx}.logmass", 0.3)
            self.config_manager.add_scale_hint(f"star.{s_idx}.pm_ra", 3.0)
            self.config_manager.add_scale_hint(f"star.{s_idx}.pm_dec", 3.0)
            self.config_manager.add_scale_hint(f"star.{s_idx}.rv", 1e5)
            self._seed_expected_pm(pm_expected, s_idx, "bulge", 8000.0, system)

        # Companion lens bodies (everything beyond the primary)
        for l2_type, l2_idx in self.lens_bodies[1:]:
            if l2_type == "star":
                self.config_manager.add_hint(
                    f"star.{l2_idx}.distance",
                    4000.0,
                    rank=PRECEDENCE_MULENS_LENS_DISTANCE,
                )
                self.config_manager.add_scale_hint(
                    f"star.{l2_idx}.distance", 5.0
                )

        # Tighten lens logmass scale when satellite parallax is available
        if hasattr(system, "mulensinstrument") and hasattr(
            system.mulensinstrument, "inst_ref_pos"
        ):
            ref_pos = system.mulensinstrument.inst_ref_pos
            max_sep = max(
                (
                    float(np.linalg.norm(ref_pos[ii] - ref_pos[jj]))
                    for ii in range(len(ref_pos))
                    for jj in range(ii + 1, len(ref_pos))
                ),
                default=0.0,
            )
            if max_sep > 0.5:
                scale = 0.0005
            elif max_sep > 1e-5:
                scale = 0.00075
            else:
                scale = None
            if scale is not None:
                self.config_manager.add_scale_hint(
                    f"star.{l_idx}.logmass", scale
                )

    def _galactic_pm_expectations(self, system):
        """Line of sight for the galactic-model proper-motion seeds.

        Returns ``(ra_rad, dec_rad)``, or None when the seeding does not
        apply.

        The prior is only allowed to FILL A GAP, never to contradict.  What
        is open here is the physical side: no example pins the lens mass or
        distance, because a published light-curve solution (t_0, u_0, t_E,
        s, q, alpha, rho, sometimes pi_E) does not close the system -- t_E
        and pi_E without theta_E leave mass, distance and proper motion
        free.  That gap is what the engine used to fill by inventing a
        direction (issue #93).

        THE GATE, AND WHY IT IS ONLY pi_E NOW (#93, JDE 2026-10-20).  Until
        #93 any user pi_E_N/pi_E_E OR any user pm_ra/pm_dec (any star)
        skipped ALL seeding, because a prior mean dropped on top of a value
        the config implied fought it.  Measured then, under the OLD engine,
        chi2/N at the seed, ungated vs gated:

            ob140939 (pi_E_N/pi_E_E measured, Yee+2015)  3.04 -> 179.1 | 3.04
            ob161003 (two sources, t_E + rho each)       1.72 ->   3.9 | 1.72
            DC2018_128 (t_0/u_0/t_E/s/q/alpha/rho)       1.42 ->   1.21 (kept)
            ob08092 (t_0/u_0/t_E only, PSPL)             1.50 ->   1.42 (kept)

        Under the precedence ladder (#455) a USER pm (PRECEDENCE_USER)
        outranks this seed (PRECEDENCE_DERIVED_DATA) and is never traded
        against it, so a seed can no longer bend a user's value.  Hence:
        a star whose pm the user gave a VALUE (an initval or a mu; bounds
        alone do not count) is simply not seeded (``_seed_expected_pm``),
        and every OTHER star still is.  The gate
        stays for pi_E alone: two seeded proper motions would fight a user
        pi_E over the DIRECTION of mu_rel, and no distance can absorb a
        direction -- only the magnitude goes into theta_E.

        Re-measured 2026-10-20 under the new engine.  The metric is
        check_user_starts (requested vs produced), with start logp as
        context.  With the narrowed gate EVERY shipped microlensing config
        starts BYTE-IDENTICAL to the old gate (ob08092, ob140939, ob161003,
        ob09020, ob09020_linear, KMT-2019-BLG-1806, ob07224, ob170114,
        OGLE_0383LD, DC2018_128, ob161045, kb180087_obj3,
        KMT-2021-BLG-1122L): each one either sets pi_E, sets the pm of every
        seedable star, or sets none (seeded before and after).  The one
        that changes path is ob161003, which now seeds SourceB; the engine
        derives SourceB's pm from the user Lens pm and the event's one
        mu_rel (DERIVED_USER, which outranks the seed), so its start does
        not move either.  The case the ruling is about, ob08092 with only a user
        star.Lens pm of (-2, -5) mas/yr:

                         seeded   mu_rel_geo PA  |mu_rel_geo|  logp     misses
            old gate     none     172.8 deg     11.55         6240.48  0
            pi_E only    Source    75.5 deg      1.05         6249.94  0

        the source sits at the bulge mean, the lens keeps the user's pm, and
        t_E is absorbed through the lens distance.  Removing the pi_E gate
        too (seeding over a user pi_E) still delivers pi_E and leaves chi2/N
        unchanged -- the old 179.1 blow-up does not return -- but the engine
        then has to throw the proper motions far out to hold the direction:
        ob08092 + pi_E (0.3, -0.1) moves the pms to ~40 mas/yr and logp
        6240.25 -> 5880.32; KMT-2019-BLG-1806 to ~55 mas/yr and logp
        45542.37 -> 44784.14.  That is the gate's measured reason.  A user
        Sgr-A*-relative pm (``pm_ra_sgra``) counts as setting the pm: seeding
        ob140939's source over its user pm_ra_sgra put both pms at ~240
        mas/yr and logp at -320466 against -1233.58.

        WITH NO galacticmodel BLOCK there is no direction information at
        all, so nothing is seeded and the engine picks one (ob08092: PA -86
        deg, almost all in RA); left as is.  If a direction is ever wanted
        there, the honest fix is a mu_rel position-angle parameter (the
        (mu_rel_mag, mu_rel_pa) basis), a sampling-geometry question that
        does not belong here.  tests/test_seed_quality.py pins the gate and
        the per-star skip.

        THE MULTI-SOURCE GATE IS RETIRED (stage 4).  It refused to seed any
        event with more than one source, because pre-split each source
        carried its own mu_rel vector and one prior mean forced them
        together -- the 1.72 -> 3.9 row above.  Under R1 there is exactly
        one mu_rel per event, resolved through source body 0, so the same
        mean is now information rather than a contradiction: source body 1's
        pm feeds only its own distance/rho chain.

        Measured rather than argued, because the reasoning above would have
        justified the retirement even if it were wrong.  All four pinned
        examples come out BYTE-IDENTICAL with the gate gone, which on its
        own proves nothing -- ob161003's params file pins
        star.Source*/star.Lens pm_ra/pm_dec (added with the du-sign work),
        so the then-blanket pm/pi_E gate returned first and this one was
        unreachable for the only multi-source example shipped.  Stripping those four
        pins to reach it, chi2/N at the seed:

            gate present (no pm seeded)   10.91
            gate retired (pm seeded)       6.94

        i.e. the case the gate governed is better without it, and no
        published solution moves at all.  tests/test_seed_quality.py's
        test_a_multi_source_event_is_seeded_from_the_galactic_model pins
        both halves.
        """
        if "galacticmodel" not in getattr(system, "config", {}):
            return None
        # The one remaining gate: a user pi_E fixes the DIRECTION of
        # mu_rel, and two seeded proper motions would fight it over that
        # direction, which no distance can absorb.  A user proper motion
        # is NOT a reason to skip here: it outranks the seed, so it is
        # handled per star in _seed_expected_pm (#93, JDE 2026-10-20).
        up = self.config_manager.user_params
        blockers = [k for k in up if k.endswith((".pi_E_N", ".pi_E_E"))]
        if blockers:
            logger.info(
                f"[{self.prefix}] microlens parallax already given "
                f"({', '.join(sorted(blockers))}); not seeding from the "
                f"galactic model, whose proper motions would contradict "
                f"its direction."
            )
            return None
        # probe_star_radec, not resolve(): a source position LINKED to the
        # lens's read as the defaults.yaml placeholder under resolve(), and
        # an `in user_params` test cannot tell a link from a number (review
        # 1.1.10).  An uninformed position means no line of sight, so no
        # seed -- these seeds are optional.
        source_ndx = int(self.source_map[0])
        pos = probe_star_radec(self.config_manager, [source_ndx])[source_ndx]
        if pos is None:
            logger.debug(
                f"[{self.prefix}] no user-set RA/Dec; skipping the "
                f"galactic-model proper-motion seeds."
            )
            return None
        return pos

    def _user_pm_keys(self, system, star_idx):
        """The params-file keys that set star ``star_idx``'s proper motion.

        All three spellings (broadcast ``star.pm_ra``, index ``star.2.pm_ra``,
        name ``star.Source.pm_ra``), and the Sgr-A*-relative pair as well as
        the absolute one: ``pm_ra_sgra`` is derived from ``pm_ra`` one to
        one, so a user value on it pins the proper motion just as directly
        (ob140939's source).  Sorted, for a stable log line.

        ONLY AN ENTRY THAT GIVES A VALUE COUNTS (JDE 2026-10-20): an
        ``initval`` or a prior mean ``mu``, written as a number or as a link
        (a linked field lives in ``ConfigManager.links``, not in the entry).
        A bounds-only entry (``lower``/``upper``/``sigma`` with no value)
        states no proper motion, so it must not turn the seed off; the seed
        then meets those bounds under the usual rules for any start.
        """
        cm = self.config_manager
        name = system.star.names[star_idx]

        def gives_value(key):
            entry = user_entry(cm.user_params, key)
            if entry is not None and any(
                entry.get(f) is not None for f in ("initval", "mu")
            ):
                return True
            linked = cm.links.get(key, {})
            return "initval" in linked or "mu" in linked

        return sorted(
            key
            for axis in ("pm_ra", "pm_dec", "pm_ra_sgra", "pm_dec_sgra")
            for key in (
                f"star.{axis}",
                f"star.{star_idx}.{axis}",
                f"star.{name}.{axis}",
            )
            if gives_value(key)
        )

    def _seed_expected_pm(
        self, line_of_sight, star_idx, population, dist_pc, system
    ):
        """Seed one star's pm_ra/pm_dec at the galactic model's prior mean.

        Skipped for a star whose proper motion the user set
        (``_user_pm_keys``): a user value (PRECEDENCE_USER) outranks this
        seed, so the seed could never move it and is only noise in the log
        and the ledger.  The OTHER stars are still seeded (#93).

        PRECEDENCE_DERIVED_DATA: this is derived from the galactic model
        the same way an RV offset is derived from the data, so it belongs
        in that tier and must yield to anything in params.yaml.

        It ties with the peak finder's seeds (also PRECEDENCE_DERIVED_DATA),
        which is the point.  Both proper-motion components are now pinned,
        so ``mu_rel`` has a magnitude AND a direction, and the engine no
        longer has to invert
        ``mu_rel_mag**2 = mu_ra_rel**2 + mu_dec_rel**2`` -- one equation in
        two unknowns -- by choosing a point on a circle (issue #93).  Where
        that disagrees with the seeded ``t_E``, the relaxation engine's KS
        rule (review 3.14.22) holds the data tier and lets a LOWER one
        absorb ``t_E = theta_E / |mu_rel_geo|``: theta_E through the lens
        distance (``PRECEDENCE_MULENS_LENS_DISTANCE``) or a defaults.yaml
        lens mass.  So ``t_E`` keeps its measured value, the proper motion
        keeps the prior's, and the lens absorbs the difference -- which is the standard
        microlensing chain (a measured t_E plus an assumed mu_rel implies
        theta_E, hence a mass) and is the quantity a light curve genuinely
        cannot pin down.

        `dist_pc` must match the distance hint seeded for the same star:
        the mean velocity is position-dependent, so a mismatch would seed a
        proper motion for a place the star is not.
        """
        if line_of_sight is None:
            return
        user_keys = self._user_pm_keys(system, star_idx)
        if user_keys:
            logger.info(
                f"[{self.prefix}] star.{star_idx} proper motion given "
                f"({', '.join(user_keys)}); not seeding it from the "
                f"galactic model."
            )
            return
        ra_rad, dec_rad = line_of_sight
        try:
            pm_ra, pm_dec, _rv = expected_proper_motion(
                ra_rad, dec_rad, dist_pc, population
            )
        except Exception as exc:
            # WARNING, not debug: this silently disabled the whole feature
            # once already (degrees were passed where radians were wanted,
            # astropy raised, and the seeds just quietly never happened).
            # A failure here is not fatal -- the old arbitrary start still
            # works -- but it must be visible.
            logger.warning(
                f"[{self.prefix}] could not seed star.{star_idx}'s proper "
                f"motion from the galactic model ({population} at "
                f"{dist_pc:.0f} pc): {exc!r}.  Falling back to the "
                f"defaults.yaml value; the direction of mu_rel will be "
                f"arbitrary (see issue #93)."
            )
            return
        self.config_manager.add_hint(
            f"star.{star_idx}.pm_ra", pm_ra, rank=PRECEDENCE_DERIVED_DATA
        )
        self.config_manager.add_hint(
            f"star.{star_idx}.pm_dec", pm_dec, rank=PRECEDENCE_DERIVED_DATA
        )
        logger.info(
            f"[{self.prefix}] star.{star_idx} proper motion seeded at the "
            f"{population} prior mean for {dist_pc:.0f} pc: "
            f"pm_ra={pm_ra:+.3f}, pm_dec={pm_dec:+.3f} mas/yr."
        )

    def add_parameter(self, model, param_name, system, context_nodes=None):
        """Inject the Earth-velocity context constants for the mu_rel_geo
        chain and the frame constants of the derived lens offset (see
        context_dep_names); everything else is generic."""
        if param_name in ("mu_ra_rel_geo", "mu_dec_rel_geo"):
            context_nodes = dict(context_nodes or {})
            if "earth_vperp_e" not in context_nodes:
                vperp_e, vperp_n = self._earth_vperp_en(system)
                context_nodes["earth_vperp_e"] = pt.as_tensor_variable(vperp_e)
                context_nodes["earth_vperp_n"] = pt.as_tensor_variable(vperp_n)
        if param_name in ("lens_offset_e", "lens_offset_n"):
            context_nodes = dict(context_nodes or {})
            for key, val in self._lens_track_constants(system).items():
                context_nodes.setdefault(key, pt.as_tensor_variable(val))
        return super().add_parameter(model, param_name, system, context_nodes)

    def register_frame_constants(self, system):
        """Hand the relaxation engine Earth's projected velocity at t0_par.

        The engine's t_E and pi_E relations are GEOCENTRIC, as the model's
        are (review 8.6.23): mu_geo = mu_helio - pi_rel * v_perp.  The two
        components of v_perp are exact per-event numbers, not parameters,
        so they reach the relations through ConfigManager.add_constant --
        the same values ``add_parameter`` injects into the model graph.
        Called by MulensInstrument.load_data as soon as t0_par is final
        (before its stage-1 probe of the event geometry) and again at stage
        3, which also covers a fit with no microlensing data (0, 0).
        """
        if self._line_of_sight_rad is None:
            # No microlensing data set it at stage 1 (an astrometry-only
            # or data-free event): the frame is still well defined by the
            # source star's own coordinates, so read them here rather than
            # leave geocentric_frame to raise at stage 7.  probe_star_radec,
            # not resolve(): a linked position read as the placeholder
            # (review 1.1.10).  An UNSTATED position leaves the line of
            # sight unset -- a data-free model may never need the frame,
            # and geocentric_frame raises, naming the coordinates, if
            # anything does -- rather than set it from the placeholder.
            source_ndx = int(self.source_bodies[0][1])
            pos = probe_star_radec(self.config_manager, [source_ndx])
            if pos[source_ndx] is not None:
                self.set_line_of_sight(*pos[source_ndx])
        vperp_e, vperp_n = self._earth_vperp_en(system)
        self.config_manager.add_constant(
            "mulensevent.0.earth_vperp_e", vperp_e
        )
        self.config_manager.add_constant(
            "mulensevent.0.earth_vperp_n", vperp_n
        )

    def _earth_vperp_en(self, system):
        """Earth's velocity at t0_par projected on the sky, (East, North),
        in AU/yr (numerically 1/yr once divided by the 1-AU baseline --
        multiplying by pi_rel in mas gives mas/yr).

        This is the Gould (2004) mu_helio -> mu_geo conversion constant:
        mu_geo = mu_helio - pi_rel * v_perp / AU.  The velocity and the
        (ra, dec) used for the projection come from MulensInstrument -- the
        SAME anchor epoch and sky position its Skowron deltas use, so the
        conversion and the trajectory share one frame by construction.
        Without microlensing data there is no t0_par to anchor the frame;
        the term is dropped (mu_geo == mu_helio) with a warning.
        """
        inst = getattr(system, "mulensinstrument", None)
        vel = getattr(inst, "_earth_vel_ref", None)
        radec = getattr(inst, "_source_radec_rad", None)
        if vel is None or radec is None:
            logger.warning(
                f"[{self.prefix}] No microlensing data to anchor t0_par; "
                "mu_rel_geo falls back to the heliocentric value (Earth-"
                "velocity term dropped)."
            )
            return 0.0, 0.0
        v = np.asarray(vel, dtype=float) * DAYS_PER_YEAR  # AU/day -> AU/yr
        ra, dec = radec
        # The basis itself, not observer_sky_offset: what is projected here
        # is the Earth's VELOCITY, not its position.
        e_hat, n_hat = sky_basis(ra, dec)
        return float(v @ e_hat), float(v @ n_hat)

    def build_likelihood(self, model, system):
        """Stage 7: observational penalties on the EVENT-level quantities.

        The per-companion bound-orbit rope lives on Lens.build_likelihood
        (it is a statement about one companion's geometry); the fitu0te
        Jacobian lives on Source.build_likelihood (u0te is a per-source
        coordinate).  Everything here is a property of the one encounter.
        """
        source = system.source
        lens = system.lens

        # Parameter refs for the post-fit corner plot (plot_corner has no
        # system handle; the refs survive distribute_posterior).
        self._corner_params = {
            "t_0": source.t_0,
            "u_0": source.u_0,
            "t_E": self.t_E,
            # rho only where it shapes the light curve: a point-source event
            # REPORTS rho (Source.register_parameters), but its geometry
            # corner stays the one it always was.
            "rho": getattr(source, "rho", None)
            if self.finite_source
            else None,
            "s": getattr(lens, "s", None),
            "q": getattr(lens, "q", None),
            "alpha": getattr(lens, "alpha", None),
            "lens_names": list(getattr(lens, "names", []) or []),
            "n_lens": int(lens.n_elements),
        }

        # fitpirel's change of variables is NONLINEAR (unlike fitmurel's):
        # D_l = 1000/(pi_rel + 1000/D_s)  =>
        # |dD_l/dlog_pi_rel| = ln10 * pi_rel * D_l^2 / 1000.
        if getattr(self, "_fitpirel", False):
            l_idx = int(self.lens_bodies[0][1])
            d_l_scalar = system.star.distance.value[l_idx]
            pi_rel_v = self.pi_rel.value[0]
            pm.Potential(
                f"{self.prefix}.fitpirel_jacobian",
                pt.log(np.log(10.0) * pi_rel_v * d_l_scalar**2 / 1000.0),
            )

        # GEOCENTRIC mu_rel: the event-rate selection is the sky-sweep rate
        # in the frame the event is observed in, and it is also the divisor
        # of t_E/pi_E, so the singularity guard belongs on it.  These are
        # length-1 event vectors, so the 8.6.18 "one encounter, one weight"
        # indexing ([0]) and the whole-vector sums coincide by construction
        # -- the shape finally matches the physics.
        mu_rel_geo = self.mu_rel_geo_mag.value
        theta_E = self.theta_E.value
        # Deliberately UNNORMALIZED (mulensing.md, "The event-rate prior"):
        # its normalizer E_prior[mu_rel theta_E] is common to 1L1S, 1L1S_fs
        # and a linked 1L2S, and differs for 2L1S by exactly
        # E_q[sqrt(1 + q)] -- the physical rate ratio of a heavier lens,
        # which the evidence is meant to keep.

        pm.Potential(
            f"{self.prefix}.event_rate_prior",
            pt.log(pt.maximum(mu_rel_geo[0], MU_REL_FLOOR))
            + pt.log(pt.maximum(theta_E[0], THETA_E_FLOOR)),
        )
        get_collector(system).add(
            r"We weighted the lens prior by the microlensing event rate, "
            r"$\Gamma \propto \mu_{\rm rel}\,\theta_{\rm E}$ "
            r"\citep[e.g.][]{Batista:2011}.",
            section="priors",
            key=f"{self.prefix}.event_rate",
            rank=30,
        )

        # Shared log-sigmoid barriers (see exozippy.potentials).  The
        # source_behind_lens SUM IS PER SOURCE and must stay a sum: each
        # source must independently sit behind the lens, so N sources
        # really are N constraints (8.6.18's note).  The singularity guards
        # are over the length-1 event vectors.
        d_l = system.star.distance.value[self.lens_map]
        d_s = system.star.distance.value[source.star_map]
        pm.Potential(
            f"{self.prefix}.source_behind_lens",
            pt.sum(soft_lower_bound(d_s - d_l, 10.0, scale=440.0)),
        )

        pm.Potential(
            f"{self.prefix}.mu_rel_singularity",
            pt.sum(soft_lower_bound(mu_rel_geo, 1e-6, scale=1e-5)),
        )

        pm.Potential(
            f"{self.prefix}.theta_E_singularity",
            pt.sum(soft_lower_bound(theta_E, 1e-6, scale=1e-5)),
        )

        if self.xal_orbit_idxs:
            get_collector(system).add(
                r"We modeled the orbital motion of the source (the "
                r"xallarap effect; \citealt{Griest:1992}) by projecting "
                r"the luminous source's Keplerian barycentric orbit onto "
                r"the lens-source trajectory, entering at the same slot "
                r"as the parallax and anchored at the same fiducial epoch "
                r"$t_{0,\rm par}$, with the orbit's physical parameters "
                r"(period, eccentricity, orientation, and the companion "
                r"mass through the barycentric scale) sampled directly "
                r"\citep[cf.][]{Mroz:2026}.",
                section="microlensing",
                key=f"{self.prefix}.xallarap",
                rank=33,
            )

    # ------------------------------------------------------------------
    # Magnification dispatcher (moved from the pre-split Lens; the index
    # reads are retargeted: source trajectories via system.source, the
    # companion geometry via system.lens at ELEMENT j+1 -- element 0 is the
    # masked primary).
    # ------------------------------------------------------------------

    def _alpha_deg(self, system, j=0):
        """Trajectory angle of companion ``j`` in DEGREES -- the unit both
        magnification backends take, while lens.alpha's internal unit is
        radians.  ``j`` is the COMPANION SLOT; the vector element is j+1
        (element 0 is the masked primary)."""
        return system.lens.alpha.value[j + 1] * _RAD_TO_DEG

    def _companion_geometry_series(self, times, system):
        """Per-epoch geometry of EVERY companion, ``(S, ALPHA_DEG)``, each
        of shape ``(n_companions, n_epochs)`` -- row j is companion slot j
        (lens VECTOR ELEMENT j + 1; element 0 is the masked primary) -- or
        ``None`` when no companion moves (review 8.6.33: any number of
        companions may move, each on its own orbit, beside static ones).

        Each row follows its own companion's mode.  The linear mode is
        DEFINITIONAL in these coordinates (C24):

            s_j(t)     = s_j0     + ds_dt[o_j]     * (t - t0_par)/DAYS_PER_YEAR
            alpha_j(t) = alpha_j0 + dalpha_dt[o_j] * (t - t0_par)/DAYS_PER_YEAR

        with o_j companion j's own linear orbit, anchored at t0_par -- the
        same fiducial epoch the parallax uses (5d: one anchor is what makes
        the two effects composable; Skowron Eq. A17).  A keplerian row comes
        from companion j's own orbit through ``lens_geometry_from_orbit``,
        the same physics function the reported s/alpha use; a static row is
        its constant (s_j, alpha_j).  ``alpha`` is returned in DEGREES, the
        unit both magnification backends take; ``dalpha_dt``'s internal
        unit is rad/yr, so the rate converts here alongside alpha itself
        (_alpha_deg).  Skowron's gamma vocabulary maps as
        gamma_par = ds_dt/s_0 and gamma_perp = -dalpha_dt -- the minus is
        C24's rule, and the binary light curve built from these definitions
        is pinned against MulensModel's linear branch (the reference
        implementation) in tests/test_lens_orbital_motion.py.

        ``times`` may be a tensor or a numpy array (the likelihood's
        concatenated epochs, or a plotter's model grid) -- the series is
        built from the argument, never from ``self.time``, so the plotted
        curve is the curve the likelihood fits.
        """
        lens = system.lens
        if not lens.any_motion:
            return None
        dt_yr = (pt.as_tensor_variable(times) - self.t0_par[0]) / DAYS_PER_YEAR
        orbit = system.orbit
        s_rows, alpha_rows = [], []
        for j, (mode, o) in enumerate(
            zip(lens.companion_motion, lens.companion_orbit_idx)
        ):
            elem = j + 1
            if mode == "linear":
                # The rates are the companion's own type: linear ORBIT's
                # (anchored at t0_par, that orbit's epoch --
                # Orbit._taylor_epochs).
                s_rows.append(
                    lens.s.value[elem] + orbit.ds_dt.value[o] * dt_yr
                )
                alpha_rows.append(
                    self._alpha_deg(system, j)
                    + orbit.dalpha_dt.value[o] * _RAD_TO_DEG * dt_yr
                )
            elif mode == "keplerian":
                # No anchor enters -- alpha(t) = phi_pi - PA_axis(t) is
                # absolute (C15/C20/C24), and s(t) is the projected
                # separation in Einstein units.  Adds NO free parameters
                # (8.6.8 5b).
                s_t, alpha_t_rad = lens_geometry_from_orbit(
                    pt.as_tensor_variable(times),
                    orbit.tp_target.value[o],
                    orbit.n.value[o],
                    orbit.ecc.value[o],
                    orbit.sinw.value[o],
                    orbit.cosw.value[o],
                    orbit.cosi.value[o],
                    orbit.bigomega.value[o],
                    orbit.a.value[o],
                    self.theta_E.value[0],
                    system.star.distance.value[int(self.lens_bodies[0][1])],
                    self.pi_E_N.value[0],
                    self.pi_E_E.value[0],
                )
                s_rows.append(s_t)
                alpha_rows.append(alpha_t_rad * _RAD_TO_DEG)
            else:
                # Static beside a moving companion: its constant geometry,
                # broadcast over the epochs.
                zero = pt.zeros_like(dt_yr)
                s_rows.append(lens.s.value[elem] + zero)
                alpha_rows.append(self._alpha_deg(system, j) + zero)
        return pt.stack(s_rows), pt.stack(alpha_rows)

    def _source_offset_series(self, times, system, index=0):
        """Per-epoch xallarap trajectory shift ``(dtau_t, du_t)`` of SOURCE
        slot ``index``, or None for a static source (conventions.md C25;
        review 8.6.9).

        That source's own barycentric sky offset -- on the primary side of
        its orbit the primary track, a1 = a * m_companion / m_total, on the
        companion side the opposite track, a2 = -a * m_primary / m_total
        (a linked binary source puts one source on each side), in Einstein
        units a/(D_S theta_E) -- anchored at t0_par (the shift VANISHES there,
        5d: same anchor as the parallax, so t_0/u_0 keep their meaning),
        and projected on C9's (tau_hat, beta_hat) exactly where the
        parallax terms enter: parallax is the OBSERVER's offset, xallarap
        is the SOURCE's, same slot, same sign discipline (C8/C9/C25).

        Built from the TIMES ARGUMENT, so the plotters' model grids carry
        the same moving source the likelihood fits.
        """
        j = self.source_orbit_idx[index]
        if j is None:
            return None
        orbit = system.orbit
        if self.source_orbit_side[index] == 1:
            a1 = (
                orbit.a.value[j]
                * orbit.m_companion.value[j]
                / orbit.m_total.value[j]
            )
        else:
            a1 = -(
                orbit.a.value[j]
                * (orbit.m_total.value[j] - orbit.m_companion.value[j])
                / orbit.m_total.value[j]
            )
        d_s = system.star.distance.value[int(self.source_bodies[index][1])]
        args = (
            orbit.tp_target.value[j],
            orbit.n.value[j],
            orbit.ecc.value[j],
            orbit.sinw.value[j],
            orbit.cosw.value[j],
            orbit.cosi.value[j],
            orbit.bigomega.value[j],
            a1,
            self.theta_E.value[0],
            d_s,
        )
        sig_N, sig_E = source_offset_from_orbit(
            pt.as_tensor_variable(times), *args
        )
        sig_N0, sig_E0 = source_offset_from_orbit(
            pt.as_tensor_variable(np.array([float(self.t0_par[0])])), *args
        )
        mu_mag = pt.maximum(self.mu_rel_geo_mag.value[0], MU_REL_FLOOR)
        mu_n_hat = self.mu_dec_rel_geo.value[0] / mu_mag
        mu_e_hat = self.mu_ra_rel_geo.value[0] / mu_mag
        return xallarap_trajectory_shift(
            sig_N - sig_N0[0], sig_E - sig_E0[0], mu_n_hat, mu_e_hat
        )

    def _get_safe_mm_params(self, system, index=0):
        """Range-limited single-source trajectory params.  ``index`` is the
        SOURCE slot (an element of the source component); the t_E/theta_E/
        pi_E entries are this component's length-1 vectors.

        Three RANGE decisions survive here -- the t_E floor, the |u_0| floor
        and the no-lensing parallax gate, all defined and justified next to
        their constants in physics.py.  What is deliberately GONE is the NaN
        substitution that used to precede them:

            t_E -> 100 d,  u_0 -> 1,  theta_E -> 0,  pi_E_N -> 0,  pi_E_E -> 0

        i.e. a complete, fabricated PSPL model in place of a failed
        computation.  It is the same defect ``clip_q``'s ``pt.nan_to_num``
        was (review item 4.5), five more times and with a much larger blast
        radius: a fully-NaN parameter vector produced a healthy-looking
        light curve and a finite likelihood.

        Removing it is safe *and* strictly better, for the same two reasons:

        * It is unreachable.  Every one of the five is finite for every
          finite raw vector.  t_0 and u_0 are sampled with two finite hard
          bounds, so the logit transform can only produce a finite number.
          theta_E is
          ``sqrt(max(KAPPA*max(M,1e-12)*max(pi_rel,0), THETA_E_FLOOR**2))``,
          strictly positive and finite for any finite mass and pi_rel, and
          pi_rel is a difference of two 1000/distance terms whose distances
          are logit-bounded away from zero.
          t_E = theta_E/(mu_rel_geo/365.25) and
          pi_E = (pi_rel/theta_E)*(mu_i/mu_rel_geo) are then ratios whose
          denominators are floored at THETA_E_FLOOR and MU_REL_FLOOR --
          those two floors, added in c178305, are exactly what closed the
          0/0 that made this scrub live when it was written (May 2026),
          back when calc_mu_rel_mag was a bare sqrt that could return
          exactly 0.  Measured on examples/ob08092 (PSPL),
          examples/ob140939 (parallax + Spitzer) and examples/DC2018_128
          (binary lens): all five stay finite over the entire raw support
          out to raw = +/-1e12, one variable at a time and all at once,
          plus 2000 random raw points per event.  Three real
          300-tune/300-draw ptde_async fits (28 worker processes each,
          172k / 215k / 223k evaluations) instrumented at the scrub itself
          never once entered the branch.
        * Where it could fire it could only do harm.  These five are NaN
          only when an input is already NaN, i.e. the raw vector itself
          carries a NaN -- and that raw variable's own density term
          (NaN-propagating on every element, flat or not: parameter.py's
          _raw_logp) already makes the total logp NaN, so the proposal is rejected
          whatever this function returns (verified on all three events, for
          every sampled coordinate).  Substituting a "safe" value could
          never rescue a sample; it invented an entire event geometry --
          with a zero gradient, since nan_to_num is a switch -- in place of
          the one quantity that would have named the failure.

        The theta_E substitution was not even that: ``theta_E_scrubbed``
        fed nothing but the ``pt.gt(..., 1e-6)`` comparison, and a
        comparison against NaN is already False, so dropping it is a no-op
        in every case, NaN included.

        A NaN now propagates to logp, which is the sampler's own reject
        signal, so nothing here needs a mid-graph assert (which would kill
        a whole run over a proposal that is already being rejected) or a
        -inf potential (no gradient, and the JAX where-trap).  The two
        SAMPLED start values are checked once, loudly, in
        Source._validate_pspl_start; the numeric Op path names the
        parameter through physics.require_mm_number.
        """
        source = system.source

        tE_raw = self.t_E.value[0]
        u0_raw = source.u_0.value[index]
        theta_E_raw = self.theta_E.value[0]

        tE_safe = pt.maximum(tE_raw, T_E_FLOOR)
        u0_safe = apply_u_0_floor(u0_raw)
        is_physical = pt.gt(theta_E_raw, THETA_E_LENSING_MIN)

        # Keys are the CANONICAL parameter names, matching op.py's
        # _base_mm_params exactly.  They used to be a private dialect
        # (t0/u0/tE/pi_N/pi_E) whose "pi_E" meant pi_E_E, so a grep for
        # pi_E_E missed every consumer of this dict while a grep for pi_E
        # hit the wrong one (review 4.6.1).  Names only -- no sign, no
        # floor and no expression changed; the parallax convention is
        # stated at the one place that applies it, get_magnification below.
        return {
            "t_0": source.t_0.value[index],
            "u_0": u0_safe,
            "t_E": tE_safe,
            "pi_E_N": pt.switch(is_physical, self.pi_E_N.value[0], 0.0),
            "pi_E_E": pt.switch(is_physical, self.pi_E_E.value[0], 0.0),
        }

    def mulensmodel_unsupported(self, system):
        """Why MulensModel cannot represent this event, or None if it can.

        MulensModel carries a point or binary lens, a static or LINEARLY
        moving companion, and a static source.  A 3+ body lens, a companion
        on a keplerian orbit (MulensModel's keplerian branch contradicts its
        own linear one by a sign, conventions.md section 6) and xallarap
        (wired only through VBM, C25) are outside it.  The ``mulensmodel``
        backend raises on each; the plots that draw with MulensModel
        (``MulensInstrument``'s caustic figure and anomaly zooms) skip with
        this reason.
        """
        n_lenses = len(self.lens_bodies)
        if n_lenses > 2:
            return f"a {n_lenses}-body lens (MulensModel carries at most 2)"
        if n_lenses == 2 and system.lens.companion_motion[0] == "keplerian":
            return "a lens companion on a keplerian orbit"
        if self.xal_orbit_idxs:
            return "xallarap (a source orbit)"
        return None

    def mm_param_list(self, system, index=0):
        """The MulensModel parameter vector of source slot ``index``, in
        op.py's positional layout: ``[t_0, u_0, t_E, pi_E_N, pi_E_E]`` +
        ``[rho]`` (finite source) + ``[s, q, alpha_deg]`` (binary lens) +
        ``[ds_dt, dalpha_dt_deg]`` (a companion on a ``type: linear``
        orbit).  ONE builder for the ``mulensmodel`` backend's Ops and the
        MulensModel plots, so the caustics and trajectory drawn are those of
        the parameters the fit reads.  Limb darkening is not part of it:
        the Op caller appends ``u1``.

        The derived chain (theta_E, t_E, rho, pi_E) is already referenced
        to the TOTAL lens mass via mlens_total (C13), so the safe
        single-source params pass straight through -- only the companion
        geometry (s, q, alpha) is added, indexed by COMPANION (binary =
        companion 0 = VECTOR ELEMENT 1 of the lens component; element 0 is
        the masked primary).  q is the q Parameter (physics.calc_q), the
        same ratio every other consumer reads, not a local recomputation
        from the mass nodes.
        """
        reason = self.mulensmodel_unsupported(system)
        if reason is not None:
            raise NotImplementedError(
                f"[{self.prefix}] MulensModel cannot represent {reason}."
            )
        sp = self._get_safe_mm_params(system, index)
        params = [sp["t_0"], sp["u_0"], sp["t_E"], sp["pi_E_N"], sp["pi_E_E"]]
        if self.finite_source:
            params.append(system.source.rho.value[index])
        if len(self.lens_bodies) == 2:
            lens = system.lens
            params.extend(
                [
                    lens.s.value[1],
                    clip_q(lens.q.value[1]),
                    self._alpha_deg(system, 0),
                ]
            )
            if lens.companion_motion[0] == "linear":
                # MulensModel's LINEAR branch is definitional in the same
                # (ds_dt, dalpha_dt) and takes deg/yr; the rates are the
                # companion's type: linear orbit's.
                o = lens.companion_orbit_idx[0]
                params.extend(
                    [
                        system.orbit.ds_dt.value[o],
                        system.orbit.dalpha_dt.value[o] * _RAD_TO_DEG,
                    ]
                )
        return params

    def get_magnification(self, times, obs_pos, system, index=0):
        """Symbolic Paczynski magnification including parallax (PSPL only).

        ``index`` is the SOURCE slot (one trajectory per source body).

        obs_pos : (N, 3) Skowron+2011 geocentric deviations in AU --
        the observer's offset from the linear Earth trajectory anchored at
        t0_par (MulensInstrument._abs_to_delta).  The MulensModel Op path
        consumes the exact same array (fed as satellite_skycoord), so both
        paths carry the same parallax, annual and satellite alike, and are
        interchangeable on this input.  Zero rows mean no parallax.

        The one input they do NOT share is the line of sight: this formula
        reads the live star.ra/star.dec nodes, while the Op takes a
        coordinate STRING frozen at the start value (_frozen_op_coords_deg,
        which warns when they are sampled).  That only separates the two
        paths in a topology that actually samples ra/dec, and by ~1e-5 per
        arcsec of coordinate error -- see that method for why the freeze is
        free.
        """
        tau_p, u_p = self.get_trajectory(times, obs_pos, system, index)
        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        return (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

    def get_trajectory(self, times, obs_pos, system, index=0):
        """The lens-minus-source separation on C9's ``(tau_hat, beta_hat)``
        basis, in Einstein units: ``(tau, beta)``, each ``(N,)``.

        This is the pair ``get_magnification`` always built inline; it is a
        method of its own because the astrometric centroid shift
        (``get_centroid_shift``, conventions.md C30) needs the VECTOR, not
        just ``|u|``.  Same inputs as ``get_magnification`` -- ``obs_pos``
        are the Skowron+2011 geocentric deviations in AU (``skowron_deviations``).
        The op sequence is the one the magnification always had, so the
        refactor is bit-identical on the shipped examples.
        """
        source_ndx = int(self.source_bodies[index][1])
        ra = system.star.ra.value[source_ndx]
        dec = system.star.dec.value[source_ndx]

        delta_e, delta_n = observer_sky_offset(obs_pos, ra, dec, xp=pt)

        p = self._get_safe_mm_params(system, index)
        # MulensModel convention: delta_tau = -delta_N*pi_E_N -
        # delta_E*pi_E_E (negative on both N and E, matching Skowron+2011
        # via MulensModel's sign choice).  Published pi_E values from
        # MulensModel-based fitters are calibrated to this convention.
        tau_p = (
            (times - p["t_0"]) / p["t_E"]
            - delta_n * p["pi_E_N"]
            - delta_e * p["pi_E_E"]
        )
        u_p = p["u_0"] + delta_n * p["pi_E_E"] - delta_e * p["pi_E_N"]

        # Xallarap enters at EXACTLY this slot: parallax is the OBSERVER's
        # offset, this is the SOURCE's own (C25) -- this source's.
        xal = self._source_offset_series(times, system, index)
        if xal is not None:
            tau_p = tau_p + xal[0]
            u_p = u_p + xal[1]
        return tau_p, u_p

    def get_centroid_shift(self, times, obs_pos, system, index=0):
        """Point-lens, point-source astrometric centroid shift (C30):
        ``(delta_N, delta_E)`` in mas, ``(N,)`` each, referenced to the
        source's UNLENSED position.

            delta_theta = - theta_E * dtheta / (|u|^2 + 2)

        with ``dtheta`` the LENS-minus-SOURCE separation of C9 (so the
        shift points away from the lens).  ``dtheta`` is ``tau * tau_hat +
        beta * beta_hat`` in ``(N, E)``: ``tau_hat = mu_hat_rel,geo`` (the
        direction ``pi_E`` is derived from, and defined even when
        ``|pi_E|`` is pinned at 0 -- the light curve fixes ``|u(t)|`` only,
        so this term is what makes the direction observable) and
        ``beta_hat`` is ``tau_hat`` rotated +90 degrees North through East,
        ``(c, d) -> (-d, c)`` -- the same rotation
        ``tests/test_skyframe.py::test_microlensing_trajectory_matches_3d_geometry``
        pins for the trajectory itself.

        ``theta_E`` enters as a LINEAR AMPLITUDE on a shape the light curve
        already fixes, which is what makes this a direct measurement of the
        Einstein radius, and the term decays only as ``theta_E/u``, so it
        belongs INSIDE the full astrometric model
        (``AstrometryInstrument._absolute_model``) rather than in an
        event-window add-on.  Blend dilution and drag are the consumer's
        business (they need the astrometric band's fluxes); see
        ``AstrometryInstrument._apply_lens``.

        The point-source formula is EXACT for a point lens (the lens
        equation is a quadratic) and only safe unresolved: averaged over a
        source disk the shift nearly vanishes at ``rho ~ u`` and reverses
        sign beyond (mulensing.md, "The point-lens shift, exactly").  This
        method is
        the SYMBOLIC path only; ``get_astrometric_terms`` is the consumer's
        entry point and dispatches finite-source / binary / N-lens events
        to VBMicrolensing's disk-integrated centroid (stage 2).
        """
        tau_p, u_p = self.get_trajectory(times, obs_pos, system, index)
        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        scale = -self.theta_E.value[0] / (u2 + 2.0)
        return self._shift_to_sky(tau_p, u_p, scale)

    def check_centroid_shift_supported(self, where):
        """Gate for the astrometric consumer -- now a statement of WHICH
        path serves the event, logged, since stage 2 closed the gaps.

        Point-source single lens: the symbolic closed form
        (``get_centroid_shift``).  Finite source, binary, N-lens, or a
        forced ``use_op``: ``VBMDirectMagOp(astrometry=True)`` through
        ``get_astrometric_terms`` -- disk-integrated and limb-darkened, so
        the stage-1 ``rho << u`` caveat is gone, and gradient-free like the
        photometry it rides with.  Nothing raises here any more; the one
        remaining refusal (a lensed dataset with no light curve) lives
        with its reason in ``AstrometryInstrument._resolve_lens_photometry``.
        """
        if self.uses_op():
            what = (
                f"{self.n_companions + 1}-body lens"
                if self.n_companions
                else "single lens"
            )
            if self.finite_source:
                what += ", finite source"
            logger.info(
                f"[{where}] astrometric centroid shift through "
                f"VBMicrolensing's astrox accumulators ({what}; "
                f"VBMDirectMagOp(astrometry=True), gradient-free)."
            )
        else:
            logger.info(
                f"[{where}] astrometric centroid shift from the symbolic "
                f"point-lens closed form (differentiable)."
            )

    # ------------------------------------------------------------------
    # The Skowron+2011 geocentric frame (conventions.md C5, C6)
    # ------------------------------------------------------------------
    def set_line_of_sight(self, ra_rad, dec_rad):
        """Record the source direction the geocentric frame is built for.
        Called by ``MulensInstrument.load_data`` (both the provisional and
        the final anchoring); a change of direction invalidates the cached
        frame."""
        self._line_of_sight_rad = (float(ra_rad), float(dec_rad))

    def geocentric_frame(self):
        """``(t0_par, earth_pos_ref, earth_vel_ref)`` -- the anchors of the
        geocentric inertial frame: Earth's barycentric position (AU) and
        velocity (AU/day) at ``t0_par``.

        Owned by the event, because more than one consumer now builds
        deviations in this frame (every microlensing light curve AND every
        astrometric dataset of a lensed source), and one frame must serve
        them all.  Read from ``self.t0_par[0]`` at CALL time, not cached
        against a stale value: ``MulensInstrument.load_data`` re-resolves
        ``t0_par`` at stage 1 (the peak finder's seeds arrive after this
        component's ``__init__`` snapshot) and writes it back into that
        length-1 list precisely so later readers see the final value.

        ``t0_par`` is a BJD_TDB tag like every other epoch (C4), so the
        Earth is placed where it was when it RECEIVED that light: at the
        reception instant ``t0_par - Roemer`` toward the source
        (``ephemeris.reception_instant``; review 1.10.13).  The velocity is
        the physical one, d(position)/d(coordinate time), by a central
        difference of +/-0.05 d about that instant ((omega h)^2/6 = 1.2e-7
        relative; it was +/-0.5 d, 1.2e-5, before 1.10.13).  The velocity
        cancels exactly in u(t) -- the rectilinear term and the deviations
        use the same one -- but it sets the REPORTED t_E and the direction
        of pi_E, and a (t_0, u_0) quoted by another code is only
        comparable in a frame spanned by the same velocity (C5).
        """
        from exozippy.ephemeris import get_observer_position, reception_instant

        if self._line_of_sight_rad is None:
            raise RuntimeError(
                f"[{self.prefix}] geocentric_frame needs the source's line "
                f"of sight (set_line_of_sight), which MulensInstrument."
                f"load_data provides at stage 1 and register_frame_constants "
                f"at stage 3 -- from the source star's ra/dec, so give "
                f"star.<source>.ra and .dec (deg) in the params file if "
                f"they are unset; otherwise it was called before either."
            )
        t0 = float(self.t0_par[0])
        ra, dec = self._line_of_sight_rad
        cache = getattr(self, "_geocentric_frame_cache", None)
        if cache is not None and cache[0] == (t0, ra, dec):
            return cache[1]
        jd0 = float(reception_instant(np.array([t0]), ra, dec)[0])
        pos = get_observer_position(
            np.array([jd0]), observer_location="earth"
        )[0]  # (3,) AU
        _dt = FRAME_VELOCITY_STEP_DAYS
        _ep = get_observer_position(
            np.array([jd0 + _dt]), observer_location="earth"
        )[0]
        _em = get_observer_position(
            np.array([jd0 - _dt]), observer_location="earth"
        )[0]
        vel = (_ep - _em) / (2.0 * _dt)  # AU/day
        self._geocentric_frame_cache = ((t0, ra, dec), (t0, pos, vel))
        return self._geocentric_frame_cache[1]

    def skowron_deviations(self, t, xyz_abs):
        """Absolute barycentric observer positions -> Skowron+2011 geocentric
        deviations (AU), the ONE ``obs_pos`` both magnification backends
        and the centroid shift consume (C6):

            delta(t) = xyz_obs(t) - [xyz_earth(t0_par) + v_earth(t0_par) * (t - t0_par)]

        For Earth: the small deviation from straight-line motion (annual
        parallax).  For Spitzer: ~ the Spitzer - Earth vector at t0_par
        (satellite parallax, ~1-2 AU).  Yee+2014 Section 3: "Spitzer's
        offset from the centre of Earth is treated just as any other
        observatory."
        """
        t0, pos, vel = self.geocentric_frame()
        t_delta = (np.asarray(t, dtype=float) - t0)[:, np.newaxis]  # (N, 1)
        return xyz_abs - (pos + vel * t_delta)

    # The event window's half-width is 1 t_E, but never thinner than this
    # (days): a 0.35-day t_E makes "within 1 t_E" a sliver that holds almost
    # no data, and the enrichment ratio is then noise (the DC2026 team's
    # anomsweep.py floor, kept so the numbers agree).
    EVENT_WINDOW_MIN_HALF_WIDTH = 0.5

    def event_windows(self, system, point):
        """The time ranges where this event's signal lives, at ``point``.

        The hook the post-fit robust-likelihood audit
        (``outputs/outliers.py``) duck-types for: a list of ``(lo, hi)``
        epochs (BJD_TDB days, the instruments' time system) -- here
        ``|t - t_0| < max(t_E, EVENT_WINDOW_MIN_HALF_WIDTH)`` around every
        source's ``t_0``.  The audit compares where the hogg mixture's
        outlier-component points fall against this window; it never learns
        what a microlensing event is.  ``point`` is a plotting point
        (internal units, the form ``plot_data`` takes); a parameter it does
        not carry falls back to its initval.
        """
        source = system.source

        def value(param):
            return np.atleast_1d(
                np.asarray(point.get(param.label, param.initval), dtype=float)
            )

        t_E = float(value(self.t_E)[0])
        half = max(t_E, self.EVENT_WINDOW_MIN_HALF_WIDTH)
        if not np.isfinite(half):
            return []
        return [
            (float(t0) - half, float(t0) + half)
            for t0 in value(source.t_0)
            if np.isfinite(t0)
        ]

    def uses_op(self, index=0):
        """True if get_magnification_op will dispatch to a magnification Op
        (VBMDirectMagOp, or a MulensModel Op under ``backend:
        mulensmodel``) rather than the symbolic point-lens formula.

        Event-level property (the lens bodies and finite_source flag are
        shared by all sources), so ``index`` is ignored beyond backward
        compatibility.

        Both paths take the same obs_pos convention (Skowron+2011
        geocentric deviations); callers use this only to pick a
        sampler-compatible path.
        """
        n_lenses = len(self.lens_bodies)
        use_rho = self.finite_source
        forced = self.use_op
        return forced or (n_lenses > 1) or use_rho

    def sampler_requirements(self):
        """Declare sampler constraints for this event configuration.

        Binary/finite-source lenses use a magnification Op (VBMicrolensing,
        or MulensModel under ``backend: mulensmodel``), which is not
        differentiable.  Gradient-based samplers (NUTS, numpyro, blackjax)
        will produce invalid results; PTDE is required.  The SYNCHRONOUS
        loop (ptde) is recommended -- since 2026-10-04 (review 2.4.9, TASK
        8).  ptde_async was recommended while near-caustic evaluations
        concentrated in the hot rungs and stalled every synchronous step
        behind the slowest proposal; the far-field guard and the vbm_direct
        backend removed that tail (DC2018_128, the model that motivated it:
        0 of 115,488 calls over 0.1 s, slowest/median call per step 1.7),
        and head to head on the same model sync gives 5-14x the ESS per
        reserved core-second, every chain in the good-likelihood region,
        and 2-7 temperature round trips where async makes one and discards
        tens of thousands of swaps (samplers/samplers.md, "The recommended
        default is ptde").

        PSPL lenses use a symbolic PyTensor formula and are
        NUTS-compatible, so no constraints are returned.
        """
        if self.uses_op():
            return {
                "incompatible": {"nuts", "numpyro", "blackjax"},
                "recommended": "ptde",
                "reason": (
                    "binary/finite-source microlensing uses a "
                    "VBMicrolensing (or MulensModel) magnification Op, which "
                    "is not differentiable -- gradient-based samplers "
                    "produce invalid results"
                ),
            }
        return {}

    def _frozen_op_coords_deg(self, system, source_ndx):
        """(ra, dec) in degrees baked into the MulensModel / VBM Op, ONCE.

        The Op takes the line of sight as a coordinate STRING, so it cannot
        track a sampled ``star.ra``/``star.dec``: whatever is read here is
        frozen for the whole fit.  That freeze is deliberate and
        numerically free.  Microlensing parallax enters only through the
        PROJECTION of the Earth's orbit onto the event's (N, E) axes, so a
        coordinate error of eps radians perturbs the projection by ~eps
        relative: 1 arcsec is 5e-6, nothing against pi_E uncertainties of
        order 1%.  Making the coordinates dynamic would rebuild the Op
        every likelihood call to buy a correction six orders of magnitude
        below the measurement.

        What is NOT free is doing it silently, so a topology that actually
        samples the source's ra/dec (microlensing + gaia/abs astrometry)
        gets one warning per source naming the frozen values.  Nothing is
        emitted for the overwhelmingly common case where they are pinned --
        a warning on every microlensing fit is a warning nobody reads.

        The value comes from ``initval``, not from ``.eval()`` of the value
        node.  A sampled element's node IS a random variable, so
        ``.eval()`` draws from its prior: the old code did not freeze the
        start value, it froze an arbitrary draw (measured 0.36 deg away on
        a mulens topology with a sampled source position).
        """
        star = system.star
        deg = 180.0 / np.pi
        ra_deg = star.ra.element_start(source_ndx) * deg
        dec_deg = star.dec.element_start(source_ndx) * deg

        warned = getattr(self, "_frozen_coord_warned", None)
        if warned is None:
            warned = self._frozen_coord_warned = set()
        moving = [
            name
            for name, param in (("ra", star.ra), ("dec", star.dec))
            if param.element_is_sampled(source_ndx)
        ]
        if moving and source_ndx not in warned:
            warned.add(source_ndx)
            logger.warning(
                f"[{self.prefix}] star.{'/'.join(moving)} of source body "
                f"{source_ndx} is sampled, but the MulensModel/VBM "
                f"magnification Op takes the line of sight as a fixed "
                f"coordinate string: it is FROZEN at the start value "
                f"(ra={ra_deg:.6f} deg, dec={dec_deg:.6f} deg) for the "
                f"whole fit. This is safe -- the parallax projection is "
                f"perturbed only ~1e-5 per arcsec of coordinate error, far "
                f"below any pi_E uncertainty -- but the sampled ra/dec do "
                f"NOT feed the microlensing model."
            )
        return ra_deg, dec_deg

    def mm_coords(self, system, index=0):
        """The frozen line of sight of source slot ``index`` as the
        coordinate string every MulensModel/VBM Op (and the MulensModel
        plots) take."""
        source_ndx = int(self.source_bodies[index][1])
        ra_deg, dec_deg = self._frozen_op_coords_deg(system, source_ndx)
        return f"{ra_deg}d {dec_deg}d"

    def _resolve_quadratic_ld(self, u2, effective_bandpass):
        """Can the selected backend honor the band's second LD coefficient?

        Returns True to put u2 in the param vector, False to drop it -- and
        when it drops it, says so ONCE, because a silently ignored u2 is
        the exact defect this plumbing was added to fix.  A dropped u2 is
        not merely a wrong profile: on a band whose limb darkening only
        microlensing reads, the magnification is a function of u1 alone, so
        one combination of the sampled Kipping pair (q1, q2) becomes
        likelihood-free -- sampled, reported, and constrained by nothing.

        THE RULE IS THE BACKEND, NOTHING ELSE.  Under ``backend: vbm_direct``
        (the default) every Op-path event -- binary, N-lens, and every
        finite-source single lens -- goes through ``VBMDirectMagOp``, whose
        solvers all carry LDquadratic (BinaryMag2/MultiMag2 and, for the
        point lens, ESPLMag2), so u2 is always honored.  Under ``backend:
        mulensmodel`` (the A/B reference) it never is: ``set_limb_coeff_u``
        takes one coefficient and the Yoo04 B0/B1 factorization MulensModel
        uses for a finite point source is a linear-law formalism, so the
        reference stays linear and says so.

        WHY THE LD LAW NO LONGER PICKS THE SINGLE-LENS BACKEND.  Until
        #488, a finite-source single lens on a LINEAR band kept the
        MulensModel Op even under vbm_direct, so
        that existing answers would not move: the two backends differed by
        up to ~5 mmag, which was attributed to Yoo04's table interpolation.
        Measured against a direct integration of a linear-LD disk (u1 =
        0.55, rho = 0.001-0.05; mulensing.md, "The limb-darkening LAW
        reaches the magnification"), that attribution was wrong: Yoo04's
        tables agree with the integral to <0.01 mmag at rho <= 0.01.  The
        ~5 mmag was the MulensModel backend's own method window, which
        dropped to a POINT source outside t_0 +/- 5 rho t_E, where the finite
        source still raises A by ~rho^2/(8 u^2) ~ 0.5%.  (That window now
        ends at z = u/rho = 99, the end of Yoo04's dense table, where the
        point source is within 0.014 mmag; op.YOO04_Z_MAX.)  VBM's ESPL is
        within 0.1-0.2 mmag of the integral there and almost everywhere
        else; its two narrow weak spots (+/-1.2 mmag within 1% of rho of
        the limb crossing, and an error growing as u/rho -> 0, NaN at u ==
        0) are VBM's own and are tabulated in mulensing.md.  So the
        MulensModel route was the less accurate of the two as configured,
        and no faster; MulensModel is off the default likelihood path
        entirely.  The linear-band fits that moved are stated in
        mulensing.md.
        """
        if u2 is None or effective_bandpass is None:
            return False
        if self.backend == "vbm_direct":
            return True
        if not self._warned_linear_ld_backend:
            self._warned_linear_ld_backend = True
            logger.warning(
                "mulensevent.backend = 'mulensmodel' cannot apply a "
                "quadratic limb-darkening law (MulensModel's "
                "set_limb_coeff_u and its Yoo04 finite-source formalism "
                "are linear-only), so band.u2 is being IGNORED and the "
                "source profile is linear in u1. Note that this leaves one "
                "combination of the band's sampled (q1, q2) constrained by "
                "nothing but its prior. Fixes: use the default 'backend: "
                "vbm_direct' to honor u2, or declare 'ld_law: linear' on "
                "the band to make the linear law deliberate and drop the "
                "unconstrained coordinate."
            )
        return False

    def get_magnification_op(
        self, times, obs_pos, system, index=0, u1=None, u2=None, bandpass=None
    ):
        """Magnification dispatcher.

        ``index`` is the SOURCE slot: each source body has its own
        trajectory (t_0, u_0, rho, ...) but shares the lens bodies.
        Multi-source callers (MulensInstrument) invoke this once per source
        and combine the returned magnifications with per-source fluxes.

        For point-source PSPL (n_lenses==1, finite_source=False,
        use_op=False) falls back to the symbolic PyTensor formula so NUTS
        can differentiate through it without the O(N_params)
        numerical-gradient overhead of _MagGradOp.

        obs_pos: (N, 3) Skowron+2011 geocentric deviations in AU
        (MulensInstrument._abs_to_delta) for BOTH paths -- the symbolic
        formula projects them directly, and the Op path feeds them to
        MulensModel as satellite_skycoord (whose satellite channel then
        carries all parallax, annual + satellite).

        u1/u2/bandpass: when finite_source is True and a Band component is
        wired, u1 (a PyTensor scalar) and bandpass (str) are passed so the
        Op can apply limb darkening.  Passing neither falls back to
        uniform-source finite-source magnification.  u2 is the SECOND
        (quadratic) coefficient and is present only for a band declaring
        ``ld_law: quadratic`` -- see _resolve_quadratic_ld above for which
        backends can honor it and what happens when the selected one
        cannot.

        Set ``use_op: true`` on the mulensevent block to force the Op (e.g.
        for testing or when MulensModel's finite-source parallax is
        needed).

        The param-vector layout and the Op input contract are the
        pre-split ones (op.py unpacks by position) -- only the ELEMENT
        reads shifted: the Op's companion loop index j is its own 0-based
        companion count, and the lens vector element is j+1.
        """
        n_lenses = len(self.lens_bodies)
        if n_lenses > 2 and self.backend != "vbm_direct":
            raise NotImplementedError(
                f"{n_lenses}-lens magnification requires "
                "backend: vbm_direct (VBMicrolensing MultiMag2); the "
                "MulensModel backend supports at most 2 lens bodies."
            )

        if not self.uses_op(index):
            return self.get_magnification(times, obs_pos, system, index)

        lens = system.lens
        coords = self.mm_coords(system, index)

        use_rho = self.finite_source

        # Apply LD only for finite-source and when a band is connected.
        effective_bandpass = bandpass if (use_rho and u1 is not None) else None
        use_u2 = self._resolve_quadratic_ld(u2, effective_bandpass)

        times_tensor = pt.as_tensor_variable(times)
        obs_tensor = pt.as_tensor_variable(obs_pos)

        # Per-epoch xallarap trajectory shift (source orbital motion, C25)
        # -- None for a static source.
        source_series = self._source_offset_series(times, system, index)
        if source_series is not None and self.backend == "mulensmodel":
            raise NotImplementedError(
                "xallarap (a source on an orbit) requires backend: vbm_direct (the "
                "point-source single-lens case takes the symbolic path "
                "and needs neither).  MulensModel's native xi_* xallarap "
                "is not wired as a backend -- unnecessary: C25's machinery "
                "covers it, and the xi_* mapping is verified "
                "(conventions.md C25; a published xi_* solution seeds a "
                "config directly, see examples/ob170114)."
            )

        # Per-epoch companion geometry (lens orbital motion, C24) -- None
        # for a static lens.
        geometry_series = (
            self._companion_geometry_series(times, system)
            if n_lenses >= 2
            else None
        )

        if self.backend == "vbm_direct" and (n_lenses >= 2 or use_rho):
            # EVERY binary/N-lens event and EVERY finite-source single lens
            # (ESPL, n_companions == 0) under the default backend: whatever
            # the band's LD law, with or without xallarap.  The finite-source
            # single lens used to keep MulensModel unless u2 was in play; see
            # _resolve_quadratic_ld for why it no longer does.  The one
            # single-lens case left for the MulensModel Op below is a forced
            # `use_op: true` on a POINT source, which asks for it by name.
            return self._vbm_op_call(
                times,
                obs_pos,
                system,
                index,
                coords=coords,
                use_rho=use_rho,
                effective_bandpass=effective_bandpass,
                use_u2=use_u2,
                u1=u1,
                u2=u2,
                source_series=source_series,
                geometry_series=geometry_series,
                n_companions=self.n_companions,
            )
        elif n_lenses == 2:
            if lens.companion_motion[0] == "keplerian":
                raise NotImplementedError(
                    "a lens companion on a keplerian orbit requires backend: "
                    "vbm_direct."
                    "  MulensModel's own keplerian lens motion contradicts "
                    "its linear mode by a sign and is not a usable reference"
                    " (conventions.md section 6, measured 2026-08-27)."
                )
            param_list = self.mm_param_list(system, index)
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = BinaryLensMagOp(
                coords=coords,
                mag_method=self.mag_method[index],
                use_rho=use_rho,
                bandpass=effective_bandpass,
                orbital_motion=geometry_series is not None,
                t_0_kep=self.t0_par[0],
            )
        else:
            if source_series is not None:
                raise NotImplementedError(
                    "xallarap (a source on an orbit) with a FORCED MulensModel "
                    "single-lens Op (use_op: true) is not wired; drop "
                    "use_op (the symbolic path carries xallarap) or use "
                    "finite_source with backend: vbm_direct."
                )
            param_list = self.mm_param_list(system, index)
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = MulensMagOp(
                coords=coords,
                mag_method=self.mag_method[index],
                use_rho=use_rho,
                bandpass=effective_bandpass,
            )

        return mag_op(pt.stack(param_list), times_tensor, obs_tensor)

    def _vbm_op_call(
        self,
        times,
        obs_pos,
        system,
        index,
        *,
        coords,
        use_rho,
        effective_bandpass,
        use_u2,
        u1,
        u2,
        source_series,
        geometry_series,
        n_companions,
        astrometry=False,
    ):
        """ONE builder of a ``VBMDirectMagOp`` call: the param vector in
        op.py's layout, the Op with its flags, and the per-epoch series
        inputs.  ``get_magnification_op``'s two VBM branches (ESPL,
        binary/N-lens) and ``get_astrometric_terms`` all go through it, so
        the astrometric Op is built from the SAME parameters as the
        photometric one and differs only in ``astrometry=True`` (a separate
        instance -- op.py's class docstring says why).  Returns the Op's
        output: A, or ``(A, dtau, dbeta)`` with ``astrometry``.

        ``n_companions == 0`` is the single-lens ESPL layout (no companion
        block); otherwise every companion slot j contributes
        ``(s, q, alpha_deg)`` from lens vector element j+1.
        """
        source = system.source
        lens = system.lens
        sp = self._get_safe_mm_params(system, index)
        param_list = [
            sp["t_0"],
            sp["u_0"],
            sp["t_E"],
            sp["pi_E_N"],
            sp["pi_E_E"],
        ]
        if use_rho:
            param_list.append(source.rho.value[index])
        for j in range(n_companions):
            # Companion slot j = lens vector element j+1 (masked primary).
            param_list.extend(
                [
                    lens.s.value[j + 1],
                    clip_q(lens.q.value[j + 1]),
                    self._alpha_deg(system, j),
                ]
            )
        if effective_bandpass is not None:
            param_list.append(u1)
            if use_u2:
                param_list.append(u2)
        mag_op = VBMDirectMagOp(
            coords=coords,
            n_companions=n_companions,
            use_rho=use_rho,
            bandpass=effective_bandpass,
            quadratic_ld=use_u2,
            orbital_motion=geometry_series is not None,
            source_motion=source_series is not None,
            astrometry=astrometry,
        )
        op_inputs = [
            pt.stack(param_list),
            pt.as_tensor_variable(times),
            pt.as_tensor_variable(obs_pos),
        ]
        if geometry_series is not None:
            op_inputs += [
                pt.as_tensor_variable(geometry_series[0]),
                pt.as_tensor_variable(geometry_series[1]),
            ]
        if source_series is not None:
            op_inputs += [
                pt.as_tensor_variable(source_series[0]),
                pt.as_tensor_variable(source_series[1]),
            ]
        return mag_op(*op_inputs)

    def get_astrometric_terms(
        self,
        times,
        obs_pos,
        system,
        index=0,
        u1=None,
        u2=None,
        bandpass=None,
    ):
        """``(A, delta_N, delta_E)`` at ``times`` for the astrometric
        consumer (``AstrometryInstrument._apply_lens``): the magnification
        that weights the dilution and the centroid shift in mas (C30), from
        ONE evaluation.

        Point-source single lens (``uses_op`` False): the symbolic pair,
        ``get_magnification`` + ``get_centroid_shift`` -- differentiable.

        Everything else -- finite source, binary, N-lens, or a forced
        ``use_op`` -- goes through ``VBMDirectMagOp(astrometry=True)``
        (review 8.10.1 stage 2), ALWAYS VBM regardless of the photometric
        ``backend:``: MulensModel has no centroid, and VBM's ``astrox`` is
        the same centroid a MulensModel-backed light curve would imply
        (C18: the trajectory conventions are identical; op.py's class
        docstring: the origin is the centre of mass in both).  The A this
        returns is therefore VBM's -- the same A the light curve fits under
        the default backend, and up to ~5 mmag from a ``backend:
        mulensmodel`` light curve's on a finite-source single lens (that
        backend's point-source window; mulensing.md, the LD-law section),
        second order on a dilution WEIGHT.  The finite-source centroid is
        VBM's disk-integrated, limb-darkened one, so the stage-1 "valid
        only for rho << u" caveat does not apply on this path.  The Op has
        no gradient (``sampler_requirements`` already says so for any
        event on it).
        """
        if not self.uses_op(index):
            A = self.get_magnification(times, obs_pos, system, index)
            dN, dE = self.get_centroid_shift(times, obs_pos, system, index)
            return A, dN, dE

        n_lenses = len(self.lens_bodies)
        source_ndx = int(self.source_bodies[index][1])
        ra_deg, dec_deg = self._frozen_op_coords_deg(system, source_ndx)
        coords = f"{ra_deg}d {dec_deg}d"
        use_rho = self.finite_source
        effective_bandpass = bandpass if (use_rho and u1 is not None) else None
        use_u2 = self._resolve_quadratic_ld(u2, effective_bandpass)
        source_series = self._source_offset_series(times, system, index)
        geometry_series = (
            self._companion_geometry_series(times, system)
            if n_lenses >= 2
            else None
        )
        A, dtau, dbeta = self._vbm_op_call(
            times,
            obs_pos,
            system,
            index,
            coords=coords,
            use_rho=use_rho,
            effective_bandpass=effective_bandpass,
            use_u2=use_u2,
            u1=u1,
            u2=u2,
            source_series=source_series,
            geometry_series=geometry_series,
            n_companions=self.n_companions if n_lenses >= 2 else 0,
            astrometry=True,
        )
        dN, dE = self._shift_to_sky(dtau, dbeta, self.theta_E.value[0])
        return A, dN, dE

    def _shift_to_sky(self, dtau, dbeta, scale):
        """``scale * (dtau * tau_hat + dbeta * beta_hat)`` in ``(N, E)``:
        the one rotation from the trajectory frame onto the sky that both
        centroid paths use.  ``tau_hat = mu_hat_rel,geo`` (C9; defined even
        with ``|pi_E|`` pinned at 0) and ``beta_hat`` is ``tau_hat``
        rotated +90 degrees North through East, ``(c, d) -> (-d, c)``."""
        mu_n = self.mu_dec_rel_geo.value[0]
        mu_e = self.mu_ra_rel_geo.value[0]
        mu_mag = self.mu_rel_geo_mag.value[0]  # floored at MU_REL_FLOOR
        tau_n, tau_e = mu_n / mu_mag, mu_e / mu_mag
        beta_n, beta_e = -tau_e, tau_n
        return (
            scale * (dtau * tau_n + dbeta * beta_n),
            scale * (dtau * tau_e + dbeta * beta_e),
        )

    # ------------------------------------------------------------------
    # Auto method brackets
    # ------------------------------------------------------------------

    def resolve_auto_vbbl(self, index=0):
        """Replace 'auto_vbbl' with a concrete method list for multi-body
        lenses.

        Historically this computed hexadecapole-vs-VBM brackets on a time
        grid, but MulensModel implements binary-lens hexadecapole as 13
        python-level VBM.BinaryMag0 calls per epoch while VBM's BinaryMag2
        runs the equivalent quadrupole safety test internally in C++ and
        short-circuits to point-source when safe.  Measured on DC2018_128:
        hexadecapole 32.9 ms vs VBM-everywhere 7.7 ms per 870-point call,
        at equal or better accuracy -- so the bracket machinery optimized
        for the wrong cost model and was removed.

        Single-lens events are left untouched: 'auto_vbbl' is resolved
        inside the PSPL model builder (point_source + finite-source
        window), and the VBM/VBBL methods emitted here are
        binary-lens-only.

        Only the mulensmodel backend consumes the resulting method list;
        the default vbm_direct backend always calls BinaryMag2/MultiMag2.

        The bracket is the WHOLE time axis (review 2.6.9).  It used to be
        the data span +/- 1 d, but the same method list serves every
        epoch the event is ever evaluated at -- the plot grid
        (t_0 +/- 5 t_E, MulensInstrument._model_time_grid) and the GUI's
        grids included -- and outside the bracket MulensModel falls back
        to its default point-source method, so a finite-source plotted
        curve silently changed convention beyond the data.  Padding to
        "the union of the data and plot spans" would still miss any grid
        chosen later; with no bracket left to compute, nothing can
        outrun it.  Nothing is spent on it either: BinaryMag2 runs its own
        quadrupole test and short-circuits to point source where that is
        safe, which is the "VBM everywhere" cost model above.  The data
        epochs are inside both brackets, so the likelihood is unchanged.

        A POINT source gets ``point_source`` (MulensModel's exact binary
        point-source solver), the same choice ``op._build_binary_model``
        makes for the unresolved string.  This emitted ``VBM`` for it too
        until 2026-10, and MulensModel refuses a finite-source method on a
        source with no rho: the Op caught the ValueError and returned NaN,
        so every point-source binary fit on ``backend: mulensmodel`` scored
        logp = NaN at every proposal (DC2018_128: NaN against vbm_direct's
        3373.88).  Tests: ``tests/test_mulens_lens_plane.py``.
        """
        if self.mag_method[index] != "auto_vbbl":
            return
        if len(self.lens_bodies) < 2:
            return
        method = "VBM" if self.finite_source else "point_source"
        self.mag_method[index] = [-np.inf, method, np.inf]

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

    def plot_corner(self, idata, filename_prefix="debug"):
        """Corner plot of the fitted lensing geometry: t_0, u_0, t_E, s, q,
        alpha, rho -- whichever of these the event actually has (rho only
        for finite-source events; s/q/alpha only when there is at least one
        lens companion).  Only meaningful with the full posterior, so this
        is called once, after sampling, via plot_corner (not the
        twice-called plot() hook, which also runs pre-flight on a single
        point).

        t_E (and, for multi-body lenses, q and alpha) are pure physics
        expressions with no sampled elements of their own, so they never
        get a pm.Deterministic node and never appear in idata.posterior
        directly (see Parameter.build_pymc's ``track_node`` logic) -- this
        reads each Parameter's ``.posterior`` instead, which
        System.distribute_posterior (already called earlier in run_fit,
        before this hook) reconstructs for both tracked and pure-expression
        parameters alike.

        The trajectory offsets live on the source component and the
        companion geometry on lens; build_likelihood stashed their
        Parameter refs (this hook has no system handle).  Constant columns
        (the masked primary's s/q/alpha elements) are dropped by
        corner_utils' degenerate-grid filter."""
        stash = getattr(self, "_corner_params", None)
        if not stash:
            return

        source_param = stash.get("t_0")
        n_src = source_param._n_elements() if source_param is not None else 1

        def per_source_labels(param):
            if n_src <= 1:
                return None
            names = getattr(source_param, "names", None) or [
                str(j) for j in range(n_src)
            ]
            return [f"{param}[{name}]" for name in names]

        n_lens = int(stash.get("n_lens", 1))
        lens_names = stash.get("lens_names") or [str(j) for j in range(n_lens)]

        def per_body_labels(param):
            if n_lens <= 2:
                return None
            return [f"{param}[{name}]" for name in lens_names]

        param_specs = [
            (stash["t_0"], per_source_labels("t_0")),
            (stash["u_0"], per_source_labels("u_0")),
            (stash["t_E"], None),
        ]
        if stash.get("rho") is not None:
            param_specs.append((stash["rho"], per_source_labels("rho")))
        if self.n_companions >= 1 and stash.get("s") is not None:
            param_specs.append((stash["s"], per_body_labels("s")))
            param_specs.append((stash["q"], per_body_labels("q")))
            param_specs.append((stash["alpha"], per_body_labels("alpha")))

        samples, labels = collect_parameter_corner_samples(param_specs)
        save_corner_plot(samples, labels, f"{filename_prefix}_lens_corner.png")
