"""The microlensing LENS component (8.6.17 stage 1).

One instance per LENS BODY, primary first.  The per-companion geometry --
`s`, `alpha` (and their coordinates `log_s`, `xalpha`/`yalpha`), the mass
ratio `q`, and the linear orbital-motion rates -- lives here as vectors of
the component's own element count, with element 0 (the primary) INACTIVE
(INACTIVE, the Band/orbit mode idiom): a primary has no separation
from itself, so those are not parameters of that element at all.  "Companion
slot j" is therefore "lens element j+1" everywhere.

The event-level chain (t_E, theta_E, pi_E, mu_rel, pi_rel, mlens_total),
the event config keys, the magnification dispatcher and the event
potentials are on `mulensevent`; the per-source trajectory offsets (t_0,
u_0, rho) are on `source`.
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
from exozippy.components.parameterization import (
    merge_options,
    merge_overrides,
    mode_manifest,
)
from exozippy.config import PRECEDENCE_DEFAULT, user_entry
from exozippy.constants import DAYS_PER_YEAR
from exozippy.outputs.prose import get_collector
from exozippy.potentials import soft_upper_bound

from .bodies import body_entries, derive_body_names, resolve_orbit_ref
from .physics import _Q_NAN_ADVICE, Q_MAX, Q_MIN, S_MERGE

logger = logging.getLogger(__name__)

# alpha is stored in radians (lens.alpha's internal_unit) and consumed in
# degrees by both magnification backends.
_RAD_TO_DEG = 180.0 / np.pi


class Lens(Component):
    """Per-lens-body microlensing geometry.

    YAML shape::

        lens:
          - body: star.Lens          # primary; entry 0 is ALWAYS the primary
          - body: planet.b           # companion (planet or star), may carry
            orbit: LB                #   its own `orbit:` (the orbit's
                                     #   type sets the motion)

    Any number of lens bodies: the default ``backend: vbm_direct`` is
    N-body (VBMicrolensing MultiMag2); only ``backend: mulensmodel`` caps
    the lens at a binary (MulensEvent.get_magnification_op raises).  The
    relaxation engine has one relation set per companion (mulensing.md,
    "Three or more lens bodies").

    In the params file, address a companion's geometry by the body's
    instance name (``lens.b.log_s``) or by element index (``lens.1.log_s``
    -- element 0 is the primary, so the FIRST companion is element 1).  A
    constraint on the primary's element is warned about and dropped: those
    entries are bookkeeping pins, not parameters.
    """

    # The keplerian-mode s/alpha expressions report the geometry AT the
    # anchor epoch; t0_par is injected as a context constant (add_parameter).
    context_dep_names = frozenset({"lens_t0_par", "companion_type_code"})

    # `companion_type_code` carries ONE ENTRY PER LENS ELEMENT, so the
    # per-element machinery may slice it alongside the per-type mass vectors
    # it selects between (8.6.21).  Declaring it here is a promise -- see
    # Component.aligned_context_deps -- and it is true by construction:
    # build_maps writes it full-length, one code per element.  `lens_t0_par`
    # is deliberately NOT here: it is injected as a scalar so the slicing
    # skips it.
    aligned_context_deps = frozenset({"companion_type_code"})

    @classmethod
    def normalize_config_block(cls, block):
        """Name each instance after its body BEFORE ConfigManager exists --
        the timing is the point (see StellarRelation.normalize_config_block
        for the history)."""
        return derive_body_names(block)

    def __init__(self, config, config_manager):
        self.normalize_config_block(config)
        super().__init__(config, config_manager)
        self.label = "Lens Parameters"

        sys_cfg = getattr(config_manager, "system_config", None) or {}
        self.bodies = body_entries(self.config, "lens", sys_cfg)
        self.n_companions = self.n_elements - 1
        # Config-derived, so it lives here rather than in build_maps:
        # register_parameters reads it and must not depend on stage 2 having
        # run.  Sorted for a deterministic per-type dep order.
        self.companion_types = tuple(sorted({t for (t, _) in self.bodies[1:]}))

        # The event options live on the mulensevent block (MulensEvent owns
        # them and the magnification dispatcher).  Checked here as well so a
        # pre-v0.1.0 config -- whose lens block carried them and which has
        # no mulensevent: key to instantiate that component from -- fails
        # with the migration message rather than a missing-attribute error
        # deep in a later stage (R3: hard breaks fail loudly).
        event_block = sys_cfg.get("mulensevent")
        if not isinstance(event_block, list) or not event_block:
            raise ValueError(
                "lens: the config must also declare a 'mulensevent:' block "
                "carrying the event options (finite_source, t0_par, "
                "backend, mag_method, use_op, peak_find, fit* flags).  "
                "Pre-v0.1.0 configs put those on "
                "the lens block; the lens block is now one entry per lens "
                "BODY (body: star.<name>)."
            )

        # The primary SOURCE body's star index: build_maps needs it for
        # primary_source_map (the linear-mode beta expression reads the
        # source distance), and the keplerian single-source refusal below
        # needs the source count.  The source component owns the source
        # parameters; only the indices are read here.
        src_block = sys_cfg.get("source")
        if not isinstance(src_block, list) or not src_block:
            raise ValueError(
                "lens: the config must also declare a 'source:' block, one "
                "entry per source body (body: star.<name>)."
            )
        src_bodies = body_entries(src_block, "source", sys_cfg)
        self._primary_source_ndx = int(src_bodies[0][1])
        n_sources = len(src_bodies)

        # Lens orbital motion (C24) is a property of the ORBIT a companion
        # moves on: its `orbit:` key names an orbit block, and that orbit's
        # `type:` decides the mode -- keplerian derives s(t)/alpha(t) from
        # its elements, linear supplies the rates ds_dt/dalpha_dt.  The
        # per-lens `orbital_motion:` key that used to say this separately
        # (and could disagree with the orbit) is gone.
        # (A leftover `orbital_motion:` key is refused by the shared key
        # check, bodies.REMOVED_KEYS, with the migration message.)
        if isinstance(self.config[0], dict) and self.config[0].get("orbit"):
            raise ValueError(
                "lens.0 (the primary) carries 'orbit:', but orbital motion "
                "is a property of a COMPANION's geometry (s, alpha); put "
                "the key on the companion's entry."
            )
        self._companion_orbit_ref = [
            (c or {}).get("orbit") for c in self.config[1:]
        ]
        types = orbit_types(sys_cfg)
        self._companion_om = []
        self._companion_orbit_idx = []
        for j, ref in enumerate(self._companion_orbit_ref):
            if ref is None:
                self._companion_om.append(None)
                self._companion_orbit_idx.append(None)
                continue
            o = resolve_orbit_ref(config_manager, ref, f"lens.{j + 1}")
            t = types[o] if o < len(types) else "keplerian"
            if t == STATIC_TYPE:
                raise ValueError(
                    f"lens.{j + 1}: orbit {ref!r} is type: static, a pairing "
                    f"with no dynamics, so it cannot move this companion.  A "
                    f"static companion carries no 'orbit:' key (the static "
                    f"orbit block may still pair it with its host)."
                )
            if t not in ("keplerian", "linear"):
                raise NotImplementedError(
                    f"lens.{j + 1}: orbit {ref!r} is type: {t}; a lens "
                    f"companion moves on a keplerian or a linear orbit "
                    f"(the magnification backends take first derivatives "
                    f"only, conventions.md C24)."
                )
            if t == "keplerian":
                # A keplerian orbit gives its COMPANION GROUP's position
                # relative to its PRIMARY GROUP, which is companion j's
                # (s_j, alpha_j) -- defined relative to the lens primary
                # (C15) -- only when the groups are exactly [primary] and
                # [companion j].  Anything else is a hierarchical system
                # (8.6.33 phase 2) and would silently give the wrong
                # geometry.  A linear orbit only holds rates of (s_j,
                # alpha_j) themselves, so its groups do not enter.
                prim, comp = parse_orbit_bodies(sys_cfg.get("orbit"), sys_cfg)
                want = ([tuple(self.bodies[0])], [tuple(self.bodies[j + 1])])
                got = (
                    [tuple(b) for b in prim[o]],
                    [tuple(b) for b in comp[o]],
                )
                if got != want:
                    raise NotImplementedError(
                        f"lens.{j + 1}: keplerian orbit {ref!r} has primary "
                        f"group {got[0]} and companion group {got[1]}; a "
                        f"lens companion's keplerian orbit must pair exactly "
                        f"the lens primary {want[0]} with that companion "
                        f"{want[1]} (hierarchical orbits are review 8.6.33 "
                        f"phase 2)."
                    )
            self._companion_om.append(t)
            self._companion_orbit_idx.append(o)
        # PER COMPANION (review 8.6.33): any number of companions may move,
        # each on its own orbit, linear or keplerian, and a static
        # companion stays static beside a moving one.  `companion_motion[j]`
        # and `companion_orbit_idx[j]` are companion slot j's (lens element
        # j + 1); `motion_orbit_idxs` is every orbit a companion moves on
        # (what Orbit owes the lens); `any_motion` says whether the
        # magnification backends need per-epoch geometry at all.
        self.companion_motion = list(self._companion_om)
        self.companion_orbit_idx = list(self._companion_orbit_idx)
        moving = [o for o in self.companion_orbit_idx if o is not None]
        if len(set(moving)) != len(moving):
            # Phase 1 of the design: each moving companion's orbit is
            # primary <-> that companion.  Two companions on ONE orbit would
            # be a companion GROUP, whose members' positions need the
            # hierarchical composition through the body groups (8.6.33
            # phase 2), not the same s(t)/alpha(t) twice.
            raise NotImplementedError(
                f"lens: companions {self._companion_orbit_ref} share an "
                f"orbit; each moving companion needs its own orbit "
                f"(hierarchical companion groups are not implemented yet, "
                f"review 8.6.33 phase 2)."
            )
        self.motion_orbit_idxs = sorted(moving)
        self.any_motion = bool(moving)
        if "keplerian" in self.companion_motion and n_sources > 1:
            raise NotImplementedError(
                "lens orbital motion on a keplerian orbit currently "
                "supports a single source (theta_E's per-source "
                "normalization would give each source its own "
                "Einstein-unit s(t))."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _user_s_bound(self, elem, field):
        """The user's own ``lens`` s ``field`` for element ``elem``, from
        the most specific of the three spellings, or None."""
        up = self.config_manager.user_params
        for key in (
            f"lens.{elem}.s",
            f"lens.{self.names[elem]}.s",
            "lens.s",
        ):
            entry = user_entry(up, key)
            if entry is not None and entry.get(field) is not None:
                return entry[field]
        return None

    def _s_bound_entries(self):
        """Apply the user's lens.<k>.s lower/upper to the SAMPLED log_s.

        s = 10**log_s is derived, so a bound the user wrote on s can only
        hold as a hard bound on log_s.  log10 is monotonic, so the
        translation is unambiguous and is applied as a convenience (JDE
        2026-10-31), WITHOUT touching ``user_params`` (CLAUDE.md; review
        2.1.26(f) -- this used to pop the user's s bounds and write log_s
        bounds into ``user_params``, so the ledger, ``export_solution`` and
        mkparam reported a log_s bound nobody wrote and lost the s bound the
        user did).  Returns ``{param: (channel, {field: per-element list})}``
        with NaN on every element left alone:

        * log_s takes log10(user bound) through "overrides".  That channel
          combines a bound as the TIGHTER of it and the defaults.yaml one --
          exactly what the same number written by the user on log_s gets,
          since a user bound never widens a defaults.yaml limit (review
          2.1.6) -- so the start is what the old user_params write produced.
          A user who also bounds log_s itself on that element has stated
          one bound twice: that raises, so no override ever clips a user
          log_s bound (whose clip warning would blame a "component-computed
          validity bound").
        * s, the derived element, must not take the user's bound as well --
          a soft barrier on top of the hard one (double counting).  An
          OPTION puts back its defaults.yaml bound on those elements, the
          barrier it had before the user wrote anything.  A user s bound
          WIDER than the default is not applied to log_s either (the
          tighter bound wins), and ``resolve()``'s "WIDENS the defaults.yaml"
          warning for s says so in the user's own spelling; that warning is
          truthful only because s's defaults bounds are 10**log_s's, which
          is checked here.

        Companion slot j is LENS ELEMENT j+1; element 0, the masked
        primary, has no s.  A keplerian-mode companion has no log_s (s is
        derived from the orbit, C24), so a bound there stays a soft barrier
        on s and is not translated.
        """
        n = self.n_elements
        defaults = self.config_manager.base_defaults["lens"]
        out = {"log_s": ("overrides", {}), "s": ("options", {})}
        logged = []
        for field in ("lower", "upper"):
            s_default = float(defaults["s"][field])
            log_default = float(defaults["log_s"][field])
            if not np.isclose(10.0**log_default, s_default, rtol=1e-12):
                raise RuntimeError(
                    f"mulensing/defaults.yaml: lens s {field} ({s_default}) "
                    f"must be 10**(log_s {field}) = {10.0**log_default}; a "
                    "user s bound is applied to log_s, and the two must "
                    "agree for resolve()'s widening warning on s to be true."
                )
            log_vals = np.full(n, np.nan)
            s_vals = np.full(n, np.nan)
            wrote_log_s = self.user_wrote_field("log_s", field)
            for j in range(self.n_companions):
                if self.companion_motion[j] == "keplerian":
                    continue
                elem = j + 1
                val = self._user_s_bound(elem, field)
                if val is None:
                    continue
                val = float(val)
                if val <= 0.0:
                    raise ValueError(
                        f"lens.{self.names[elem]}.s {field} bound must be "
                        f"positive (s > 0); got {val}."
                    )
                if wrote_log_s[elem]:
                    raise ValueError(
                        f"lens.{self.names[elem]}: both 's' and 'log_s' "
                        f"carry a '{field}' bound -- one bound stated "
                        f"twice.  Keep one of them (an s bound is applied "
                        f"to log_s as log10)."
                    )
                log_vals[elem] = np.log10(val)
                s_vals[elem] = s_default
                logged.append(
                    f"lens.{self.names[elem]}.s {field} {val:g} -> "
                    f"log_s {field} {log_vals[elem]:.4f}"
                )
            if np.isfinite(log_vals).any():
                out["log_s"][1][field] = log_vals.tolist()
                out["s"][1][field] = s_vals.tolist()
        if logged:
            logger.info(
                "Applying your s bound(s) to the sampled log_s (s = "
                f"10**log_s): {'; '.join(logged)}.  Your params entries are "
                "unchanged."
            )
        return {k: v for k, v in out.items() if v[1]}

    @property
    def prefix(self):
        return "lens"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "body",
                "kind": "ref",
                "accepts": ["star", "planet"],
                "required": True,
                "doc": (
                    "The lens body, as '<component>.<name-or-index>' (e.g. "
                    "'star.Lens' or 'planet.b').  Entry 0 is the PRIMARY "
                    "and must be a star; companions may be planets or "
                    "stars."
                ),
            },
            {
                "key": "orbit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "The orbit instance (name or index) THIS companion moves "
                    "on, mirroring astrometryinstrument's rel-mode "
                    "vocabulary; absent = a static s/alpha.  The orbit's "
                    "type decides the motion (conventions.md C24): "
                    "'keplerian' drives s(t)/alpha(t) from its elements with "
                    "NO new free parameters; 'linear' expands about t0_par "
                    "with the orbit's rates ds_dt [Einstein radii/yr] and "
                    "dalpha_dt [deg/yr user, rad/yr internal], under a soft "
                    "beta < 1 bound-orbit potential (Skowron+2011 A19).  The "
                    "orbit's bodies should be the lens bodies, so its masses "
                    "and the lens geometry share one mass chain.  Not valid "
                    "on the primary's entry; exactly one companion for now."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Lifecycle stages
    # ------------------------------------------------------------------

    def build_maps(self):
        """Stage 2: index maps.

        All FULL-LENGTH (one entry per LENS ELEMENT: `event_map`,
        `primary_lens_map`, `primary_source_map`, `companion_body_map`,
        `lens_kep_orbit_map`, `lens_motion_orbit_map`), so the per-element expression machinery can
        PROVE alignment and slice them to the active (companion) elements
        -- a length-1 map under a masked vector fails the aligned check by
        design (component.py).  Entry 0 of each is a filler that the
        primary's inactive mask keeps out of every expression.
        """
        _, p_ndx = self.bodies[0]
        n = self.n_elements

        self.event_map = np.zeros(n, dtype=int)
        self.primary_lens_map = np.full(n, p_ndx, dtype=int)
        self.primary_source_map = np.full(
            n, self._primary_source_ndx, dtype=int
        )
        if self.n_companions >= 1:
            # One body index per element; entry 0 (the primary, masked) is
            # filled with the first companion's index so the map indexes
            # only valid rows of the companion component's vectors.
            fill = self.bodies[1][1]
            body_map = np.full(n, fill, dtype=int)
            for j, (_, c_ndx) in enumerate(self.bodies[1:]):
                body_map[j + 1] = c_ndx
            self.companion_body_map = body_map

            # MIXED COMPANION TYPES (8.6.21) need one map PER TYPE, because
            # `companion_body_map` holds each body's index within its OWN
            # component: read through a single typed dep it points a planet's
            # element at a star, and for [star, star, planet] it pointed the
            # planet's element at star.0 -- the PRIMARY -- so q came out 1.0
            # there with the planet's mass never entering.
            #
            # Each per-type map is FULL-LENGTH so the per-element machinery
            # can prove alignment (see this docstring), and elements NOT of
            # that type hold that type's first body index: a filler that
            # keeps the map inside its component's rows.  `calc_q_mixed`
            # scatters rather than sums, so no filler is ever read -- the
            # filler exists only to keep the indexing legal.
            type_code = np.zeros(n, dtype=int)
            for t_pos, c_type in enumerate(self.companion_types):
                own = [
                    (j + 1, c_ndx)
                    for j, (t, c_ndx) in enumerate(self.bodies[1:])
                    if t == c_type
                ]
                per_type = np.full(n, own[0][1], dtype=int)
                for elem, c_ndx in own:
                    per_type[elem] = c_ndx
                    type_code[elem] = t_pos
                # NAME ENDS IN `_map`, which is load-bearing: build_tensor_maps
                # (stage 5) converts exactly the attributes matching that
                # suffix into int32 tensors, so `companion_body_map_star`
                # was silently skipped and the dep resolver found nothing.
                setattr(self, f"companion_body_{c_type}_map", per_type)
            # Element 0 is the masked primary; its code is a filler for the
            # same reason as the maps above.
            self.companion_type_code = type_code
        # Per-element orbit maps (review 8.6.33): element j + 1 holds
        # companion j's own orbit.  An element whose companion is not in
        # that mode holds the mode's first orbit as a FILLER that keeps the
        # index legal; its per-element expression is a different mode's, so
        # no filler is ever read (the same rule as companion_body_map).
        for mode, attr in (
            ("keplerian", "lens_kep_orbit_map"),
            ("linear", "lens_motion_orbit_map"),
        ):
            own = [
                (j + 1, o)
                for j, (m, o) in enumerate(
                    zip(self.companion_motion, self.companion_orbit_idx)
                )
                if m == mode
            ]
            if own:
                orbit_map = np.full(n, own[0][1], dtype=int)
                for elem, o in own:
                    orbit_map[elem] = o
                setattr(self, attr, orbit_map)

    def register_parameters(self, system):
        """Stage 3: the per-body manifest (masked primary)."""
        if self.n_companions >= 1:
            # Seed alpha hints (degrees, user unit) from user xalpha/yalpha
            # entries so inspect_start can display alpha before the
            # expression graph is built.  Companion slot j is ELEMENT j+1.
            up = self.config_manager.user_params
            for j in range(self.n_companions):
                if self.companion_motion[j] == "keplerian":
                    continue  # alpha is derived from the orbit (C24)
                elem = j + 1
                ca_entry = user_entry(up, f"lens.{elem}.xalpha") or {}
                sa_entry = user_entry(up, f"lens.{elem}.yalpha") or {}
                ca = ca_entry.get("initval")
                sa = sa_entry.get("initval")
                # List-valued initval (P4 multi-seed): use seed 0.
                if isinstance(ca, (list, tuple)):
                    ca = ca[0] if ca else None
                if isinstance(sa, (list, tuple)):
                    sa = sa[0] if sa else None
                if ca is not None and sa is not None:
                    alpha_deg = float(
                        np.arctan2(float(sa), float(ca)) * _RAD_TO_DEG
                    )
                    # rank=PRECEDENCE_DEFAULT (20), NOT the add_hint
                    # default of PRECEDENCE_DERIVED_DATA (60): this
                    # value is arithmetic on the USER's own
                    # xalpha/yalpha entries, purely for display, and
                    # must neither be reported as data-derived
                    # provenance (the startup table's source column,
                    # export_solution, the GUI) nor outrank an
                    # engine-solved alpha at rank 40.
                    self.config_manager.add_hint(
                        f"lens.{elem}.alpha",
                        alpha_deg,
                        rank=PRECEDENCE_DEFAULT,
                        unit="deg",
                    )

            # q's mass dependencies are typed by the companion components.
            # All companions of ONE type share a single full-length body
            # map, and calc_q divides the sliced companion-mass vector by
            # the sliced primary-mass vector elementwise.  Mixed types
            # would need one dep per type plus a per-element selector the
            # manifest cannot express; no shipped or tested configuration
            # has a mixed-type multi-companion lens, so it is refused
            # rather than silently mis-paired.
            c_types = self.companion_types
            if len(c_types) > 1:
                # MIXED TYPES (8.6.21): one element-aligned mass vector per
                # type, then the primary's, then the per-element type code
                # naming which vector each element reads.  `calc_q_mixed`
                # SCATTERS into the positions each type owns, so this is
                # order-free -- an interleaved lens (star, planet, star) is
                # right with no "group the elements by type" convention for
                # anyone to forget.  That convention is exactly what the
                # reserved `pt.concatenate` branch in `calc_q` would have
                # needed, and it is the trap recorded in 8.6.21.
                q_spec = {
                    "expr_key": "mixed_types",
                    "deps": [
                        f"{t}.mass[companion_body_{t}_map]" for t in c_types
                    ]
                    + [
                        "star.mass[primary_lens_map]",
                        "companion_type_code",
                    ],
                }
            else:
                (c_type,) = c_types
                q_spec = {
                    "expr_key": "default",
                    "deps": [
                        f"{c_type}.mass[companion_body_map]",
                        "star.mass[primary_lens_map]",
                    ],
                }

        # The parameterization table: element 0 (the primary) is mode
        # "primary", which names NO parameters -- every geometry entry is
        # inactive there, pinned at a bookkeeping value strictly
        # interior to its bounds (the pinned element still goes through the
        # logit transform; design 2.2), given no prior, and reported
        # nowhere.  The inactive values are mutually consistent under the
        # relations that nominally connect them (s = 10**log_s = 1,
        # alpha = arctan2(0, 1) = 0, q = M1/M1 = 1), so no diagnostic that
        # prints the primary row can show a contradictory tuple.
        modes = ["primary"]
        for om in self._companion_om:
            if om == "linear":
                modes.append("companion_linear")
            elif om == "keplerian":
                modes.append("companion_keplerian")
            else:
                modes.append("companion")

        if self.n_companions >= 1:
            static_geom = {
                # log_s is the sampled coordinate; s = 10**log_s is derived
                # (the close/wide degeneracy is then an exact reflection
                # log_s -> -log_s).
                "log_s": None,
                "s": "default",
                "xalpha": None,
                "yalpha": None,
                # alpha derived from xalpha/yalpha via arctan2; internal
                # unit = rad, display = deg.
                "alpha": "default",
                "q": q_spec,
            }
            table = {
                "primary": {},
                "companion": dict(static_geom),
                # Linear lens orbital motion (C24): the companion's orbit is
                # type: linear, whose rates ds_dt/dalpha_dt (orbit
                # parameters, anchored at t0_par -- the parallax anchor, one
                # epoch is what makes the two effects composable) move the
                # static geometry here; the derived beta (Skowron A19) is
                # softly bounded below 1 in build_likelihood.
                "companion_linear": {
                    **static_geom,
                    "beta": "default",
                    # What the linear orbit implies about the period (8.8.14):
                    # a reported lower bound, checked post-fit.
                    "period_min": "default",
                },
                # keplerian mode (C24): NO sampled geometry coordinates at
                # all -- s and alpha are DERIVED from the referenced orbit
                # (evaluated at t0_par for the reported values; the
                # per-epoch series goes to the backends via
                # _companion_geometry_series).  log_s / xalpha / yalpha do
                # not exist in this mode, exactly as a linear-law band has
                # no (q1, q2): a sampled coordinate the likelihood never
                # reads is the 1.6.12 defect.
                "companion_keplerian": {
                    "s": "from_orbit",
                    "alpha": "from_orbit",
                    "q": q_spec,
                },
            }
            options = {
                "s": {"inactive_value": 1.0},
                "log_s": {"inactive_value": 0.0},
                "q": {"inactive_value": 1.0},
                "xalpha": {"inactive_value": 1.0},
                "yalpha": {"inactive_value": 0.0},
                "alpha": {"inactive_value": 0.0},
                "beta": {"inactive_value": 0.0},
                "period_min": {"inactive_value": 0.0},
            }
            self.manifest = mode_manifest(
                modes, table, options=options, where=self.prefix
            )
            # s is derived from the sampled log_s: a user s bound is applied
            # to log_s, never written into user_params (review 2.1.26(f)).
            for param, (channel, fields) in self._s_bound_entries().items():
                if channel == "overrides":
                    self.manifest[param] = merge_overrides(
                        self.manifest[param], fields
                    )
                else:
                    self.manifest[param] = merge_options(
                        self.manifest[param], **fields
                    )
        else:
            # A single point lens: no per-body geometry exists at all.
            self.manifest = {}

    def add_parameter(self, model, param_name, system, context_nodes=None):
        """Inject t0_par for the keplerian-mode s/alpha (their from_orbit
        expressions report the geometry AT the anchor epoch, 5d); everything
        else is generic.  Injected as a SCALAR so the per-element dep
        slicing skips it (a scalar applies to every element)."""
        if (
            param_name in ("s", "alpha")
            and "keplerian" in self.companion_motion
        ):
            context_nodes = dict(context_nodes or {})
            context_nodes.setdefault(
                "lens_t0_par",
                pt.as_tensor_variable(float(system.mulensevent.t0_par[0])),
            )
        if param_name == "q" and len(getattr(self, "companion_types", ())) > 1:
            # Which per-type mass vector each element reads, for
            # `calc_q_mixed`'s scatter (8.6.21).  A constant, so it costs the
            # graph nothing; int32 to match the other index maps.
            context_nodes = dict(context_nodes or {})
            context_nodes.setdefault(
                "companion_type_code",
                pt.as_tensor_variable(
                    self.companion_type_code.astype("int32")
                ),
            )
        return super().add_parameter(model, param_name, system, context_nodes)

    def _validate_q_start(self):
        """Stage 7: check the START value of the mass ratio, loudly and once.

        The magnification path clips q into [Q_MIN, Q_MAX]
        (physics.clip_q) -- a statement about where the backends are
        defined, not a licence to invent a mass ratio.  The clip used to be
        preceded by ``pt.nan_to_num(q, nan=Q_MIN)``, which silently turned
        a failed computation into a healthy-looking likelihood.  That scrub
        is gone; a NaN now reaches logp and the proposal is rejected.  What
        the scrub also hid, though, was the *start*, and a bad start is the
        case that is worth a message rather than a rejection -- so it is
        checked here, once, on the inputs, where a raise costs nothing and
        can say what to do.

        NaN is fatal: the fit cannot start.  Out of range (the infinities
        included -- they at least carry a sign, the same split clip_q_value
        makes) is a warning: the fit will silently begin at the clipped q
        rather than at the seeded one, which is exactly the sort of "the
        number I typed is not the number being fitted" that goes unnoticed
        for months.

        **NaN is fatal on every active element.**  Every companion's q is
        solved by the relaxation engine from the body masses
        (symbolic_physics.get_symbol_map maps one companion each, review
        8.6.13), so a NaN start says the solve failed -- one of the lens
        body masses is already non-finite -- and the advice below is the
        right advice.  Until 2026-10 only lens element 1 was solved, and
        elements 2 and up carried a NaN BOOKKEEPING initval for a derived
        parameter, so they were exempt (review 1.6.5); that exemption went
        with the per-companion relations.  The inactive element 0 (pinned at
        1.0) is excluded from every scan.

        A q that genuinely reaches the magnification backend as NaN is
        still caught at runtime by ``clip_q_value``, which names the
        parameter.
        """
        if self.n_companions < 1 or self.q.initval is None:
            return
        q0 = np.atleast_1d(np.asarray(self.q.initval, dtype=float)).ravel()
        active = np.array(
            [self.q.element_is_active(i) for i in range(q0.size)]
        )
        nan = np.isnan(q0) & active
        if np.any(nan):
            raise ValueError(
                f"{self.prefix}.q starts at {q0.tolist()}, which is not a "
                f"number.  {_Q_NAN_ADVICE}"
            )
        out = active & ~np.isnan(q0) & ((q0 < Q_MIN) | (q0 > Q_MAX))
        if np.any(out):
            logger.warning(
                f"{self.prefix}.q starts at {q0[out].tolist()}, outside the "
                f"[{Q_MIN:g}, {Q_MAX:g}] range the binary-lens "
                "magnification backends are defined on, so the fit will "
                "actually START at the clipped value.  Move the start "
                f"inside the range (set {self.prefix}.q, or the "
                "companion/primary masses it is derived from) rather than "
                "relying on the clip."
            )

    def _seed_starts(self, system, par):
        """``par``'s internal-unit start vector at every seed: seed 0 its
        resolved initval, seed k > 0 the relaxation engine's seed-k
        solution (``config_manager.seed_resolved``), with the elements a
        seed does not touch at seed 0's value -- exactly how
        ``System.get_raw_starts`` builds the seeds the sampler starts from.
        """
        base = np.atleast_1d(np.asarray(par.initval, dtype=float)).ravel()
        out = [base]
        for resolved in (self.config_manager.seed_resolved or [])[1:]:
            vals = system._seed_initvals_for(par, resolved)
            out.append(base if vals is None else vals)
        return out

    def _warn_linear_separation_collapse(self, system, linear):
        """Stage 7: warn (rope, not a gate) when a LINEAR orbit's separation
        s(t) = s_0 + ds_dt (t - t0_par) falls below ``S_MERGE`` -- through
        zero, typically -- inside the light curve's span, at any seed's
        start (review 1.6.22).

        A first-order orbit is an expansion about t0_par; a projected
        separation is not linear through zero, so where the line reaches
        it the approximation has broken down.  The magnification stays
        well defined there (op.py merges the pair into one lens, exact as
        s -> 0), but those epochs see a lens the orbit does not describe.
        OGLE-2013-BLG-0341's close seeds (Gould et al. 2014) carry
        s_0 ~ 0.19 and ds_dt ~ 0.14-0.49 /yr across a 3.6-year light curve,
        so s(t) crosses zero 0.4-1.3 yr before t0_par.  ONE warning per
        build, naming every (companion, seed) it applies to.  Not a
        keplerian orbit's concern: its projected separation is physical
        and may pass near zero, where the merge is simply right.
        """
        inst = getattr(system, "mulensinstrument", None)
        if not linear or inst is None:
            return
        t = np.asarray(inst.time, dtype=float)
        t0_par = float(system.mulensevent.t0_par[0])
        dt_yr = (t - t0_par) / DAYS_PER_YEAR
        s_seeds = self._seed_starts(system, self.s)
        rate_seeds = self._seed_starts(system, system.orbit.ds_dt)
        primary = self._body_label(system, 0)
        lines = []
        for elem in linear:
            o = self.companion_orbit_idx[elem - 1]
            for k, (s_k, r_k) in enumerate(zip(s_seeds, rate_seeds)):
                s0, rate = float(s_k[elem]), float(r_k[o])
                low = s0 + rate * dt_yr < S_MERGE
                if not np.any(low):
                    continue
                t_zero = t0_par - s0 / rate * DAYS_PER_YEAR if rate else np.nan
                cross = (
                    f", through zero at BJD {t_zero:.1f}"
                    if t.min() <= t_zero <= t.max()
                    else ""
                )
                lines.append(
                    f"  {self._body_label(system, elem)} (lens.{self.names[elem]}) "
                    f"on orbit '{system.orbit.names[o]}', seed {k}: s_0 = "
                    f"{s0:.4g} at t0_par = {t0_par:.2f}, ds_dt = {rate:.4g} "
                    f"/yr -> s(t) < {S_MERGE:g} at {int(low.sum())} of "
                    f"{t.size} epochs, BJD {t[low].min():.1f} to "
                    f"{t[low].max():.1f}{cross}"
                )
        if lines:
            logger.warning(
                "lens: a LINEAR (Taylor) orbit drives a companion onto the "
                f"primary {primary} inside the data span:\n"
                + "\n".join(lines)
                + "\nA first-order orbit is an expansion about t0_par and a "
                "projected separation is not linear through zero, so the "
                "approximation has broken down at those epochs.  They are "
                "evaluated with the pair merged into one lens of the summed "
                "mass (exact as s -> 0), which is a lens the orbit does not "
                "describe.  The physical alternative is a keplerian orbit "
                "(type: keplerian); otherwise check that the seed's rate is "
                "meant to apply across the whole light curve."
            )

    def _body_label(self, system, elem):
        """``<type>.<name>`` of lens element ``elem``'s body, for messages."""
        t, idx = self.bodies[elem]
        return f"{t}.{getattr(system, t).names[int(idx)]}"

    def build_likelihood(self, model, system):
        """Stage 7: start-value validation, and the bound-orbit rope for a
        linear-orbital-motion companion.  The EVENT potentials (event rate,
        singularity guards, source-behind-lens, the fitpirel Jacobian) are
        MulensEvent.build_likelihood's; the fitu0te Jacobian is
        Source.build_likelihood's.
        """
        self._validate_q_start()

        linear = [
            j + 1 for j, m in enumerate(self.companion_motion) if m == "linear"
        ]
        self._warn_linear_separation_collapse(system, linear)
        if linear:
            # The bound-orbit rope for the linear rates (C24; Skowron A19):
            # ONE LENS ORBIT, ONE BOUND (review 8.6.18) -- so one bound per
            # linearly moving companion, each on its own element (review
            # 8.6.33), never on element 0, whose pinned bookkeeping value
            # would otherwise contribute a constant logp offset (the
            # design's one whole-vector-reduction hazard, section 2.3).
            # One companion keeps the historical node name.
            for elem in linear:
                name = (
                    f"{self.prefix}.bound_orbit"
                    if len(linear) == 1
                    else f"{self.prefix}.bound_orbit_{elem}"
                )
                pm.Potential(
                    name,
                    soft_upper_bound(self.beta.value[elem], 1.0, scale=0.1),
                )
            get_collector(system).add(
                r"We modeled the orbital motion of the lens binary to "
                r"first order, $s(t) = s_0 + \dot{s}\,(t - t_{0,\rm par})$ "
                r"and $\alpha(t) = \alpha_0 + \dot{\alpha}\,(t - "
                r"t_{0,\rm par})$ \citep{Albrow:2000, Skowron:2011}, and "
                r"applied a soft upper bound at unity on the projected "
                r"kinetic-to-potential energy ratio "
                r"$\beta_{\rm kin}$ \citep{Batista:2011, Skowron:2011}, a "
                r"necessary condition for a bound lens binary.",
                section="microlensing",
                key=f"{self.prefix}.orbital_motion",
                rank=32,
            )
