"""The microlensing LENS component (8.6.17 stage 1).

One instance per LENS BODY, primary first.  The per-companion geometry --
`s`, `alpha` (and their coordinates `log_s`, `xalpha`/`yalpha`), the mass
ratio `q`, and the linear orbital-motion rates -- lives here as vectors of
the component's own element count, with element 0 (the primary) INACTIVE
(manifest role 4, the Band/orbit mode idiom): a primary has no separation
from itself, so those are not parameters of that element at all.  "Companion
slot j" is therefore "lens element j+1" everywhere.

The event-level chain (t_E, theta_E, pi_E, mu_rel, pi_rel, mlens_total) is
on `mulensevent`; the per-source trajectory offsets (t_0, u_0, rho) are on
`source`.

STAGE-1 TRANSITIONAL SHAPE: the magnification dispatcher
(`get_magnification`/`get_magnification_op`/`resolve_auto_vbbl`/`uses_op`),
the event potentials in `build_likelihood`, and the event-facing attributes
MulensInstrument reads (`n_sources`, `n_companions`, `finite_source`,
`t0_par`, `backend`, `use_op`, `source_map`, `lens_bodies`,
`source_bodies`) all remain HERE, reading the new config shape, and move to
MulensEvent in stage 2 with the consumers.  Keeping them here is what lets
stage 1 land without rewriting mulensinstrument, and keeps the stage-0
fixtures' potential names (`lens.event_rate_prior`, ...) stable across this
stage.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.orbit.bodies import component_instance_names
from exozippy.components.parameterization import mode_manifest
from exozippy.constants import DAYS_PER_YEAR
from exozippy.corner_utils import (
    collect_parameter_corner_samples,
    save_corner_plot,
)
from exozippy.outputs.prose import get_collector
from exozippy.potentials import soft_lower_bound, soft_upper_bound
from exozippy.skyframe import observer_sky_offset

from .bodies import body_entries, derive_body_names
from .op import BinaryLensMagOp, MulensMagOp, VBMDirectMagOp
from .physics import (
    _Q_NAN_ADVICE,
    MU_REL_FLOOR,
    Q_MAX,
    Q_MIN,
    T_E_FLOOR,
    THETA_E_FLOOR,
    THETA_E_LENSING_MIN,
    apply_u_0_floor,
    clip_q,
    lens_geometry_from_orbit,
    source_offset_from_orbit,
    xallarap_trajectory_shift,
)

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
            orbital_motion: linear   #   its own orbital-motion keys

    In the params file, address a companion's geometry by the body's
    instance name (``lens.b.log_s``) or by element index (``lens.1.log_s``
    -- element 0 is the primary, so the FIRST companion is element 1).  A
    constraint on the primary's element is warned about and dropped: those
    entries are bookkeeping pins, not parameters.
    """

    # Microlensing is THE topology where a solution the posterior abandons
    # still has to be reported (see mulensevent.py; this class attribute
    # moves there with the dispatcher in stage 2 and is kept here so
    # attribute-generic consumers see it whichever component they scan).
    expects_suppressed_modes = True

    # The keplerian-mode s/alpha expressions report the geometry AT the
    # anchor epoch; t0_par is injected as a context constant (add_parameter).
    context_dep_names = frozenset({"lens_t0_par"})

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

        # --------------------------------------------------------------
        # STAGE-1 EVENT COMPAT: the event options live on the mulensevent
        # block; this component still hosts the dispatcher and the
        # attributes MulensInstrument reads, so it resolves them here.
        # --------------------------------------------------------------
        event_block = sys_cfg.get("mulensevent")
        if not isinstance(event_block, list) or not event_block:
            raise ValueError(
                "lens: the config must also declare a 'mulensevent:' block "
                "carrying the event options (finite_source, t0_par, "
                "backend, mag_method, use_op, mmexofast, fit* flags, "
                "source_orbital_motion).  Pre-v0.1.0 configs put those on "
                "the lens block; the lens block is now one entry per lens "
                "BODY (body: star.<name>)."
            )
        ev = event_block[0] or {}
        self._event_config = ev

        # Source bodies (compat lists for the dispatcher and the
        # instrument's reads; the source component owns the parameters).
        src_block = sys_cfg.get("source")
        if not isinstance(src_block, list) or not src_block:
            raise ValueError(
                "lens: the config must also declare a 'source:' block, one "
                "entry per source body (body: star.<name>)."
            )
        src_bodies = body_entries(src_block, "source", sys_cfg)
        self.lens_bodies = [list(self.bodies)]
        self.source_bodies = [list(src_bodies)]
        self.n_lens_bodies = [self.n_elements]
        self.n_source_bodies = [len(src_bodies)]
        self.n_sources = len(src_bodies)

        self.finite_source = [bool(ev.get("finite_source", False))]
        self.t0_par = [self._resolve_t0_par(ev, config_manager)]

        # One magnification method per source; all sources start from the
        # event-level config value, and resolve_auto_vbbl refines each slot.
        event_method = ev.get(
            "mag_method",
            (
                "auto_vbbl"
                if (self.finite_source[0] or self.n_elements > 1)
                else "point_source"
            ),
        )
        self.mag_method = [event_method] * self.n_sources

        self.use_op = [bool(ev.get("use_op", False))]

        self.backend = ev.get("backend", "vbm_direct")
        self._warned_linear_ld_backend = False
        if self.backend not in ("vbm_direct", "mulensmodel"):
            raise ValueError(
                f"mulensevent.backend must be 'vbm_direct' or "
                f"'mulensmodel', got '{self.backend}'."
            )

        # Lens orbital motion (C24): per-COMPANION keys now.  Declaring
        # orbital_motion on the primary's entry is a config error (a point
        # primary has no s or alpha to move).
        if isinstance(self.config[0], dict) and self.config[0].get(
            "orbital_motion"
        ):
            raise ValueError(
                "lens.0 (the primary) carries 'orbital_motion:', but "
                "orbital motion is a property of a COMPANION's geometry "
                "(s, alpha); put the key on the companion's entry."
            )
        self._companion_om = [
            (c or {}).get("orbital_motion") for c in self.config[1:]
        ]
        self._companion_orbit_ref = [
            (c or {}).get("orbit") for c in self.config[1:]
        ]
        om_set = [(j, om) for j, om in enumerate(self._companion_om) if om]
        om = om_set[0][1] if om_set else None
        # Event-level compat view (the dispatcher and orbit.py read these).
        self.orbital_motion = [om]
        self.orbit_ref = [
            self._companion_orbit_ref[om_set[0][0]] if om_set else None
        ]
        if om is not None:
            if om not in ("linear", "keplerian"):
                raise ValueError(
                    f"lens orbital_motion must be 'linear' or 'keplerian' "
                    f"(or absent for a static geometry), got '{om}'."
                )
            if len(om_set) > 1 or self.n_companions > 1:
                raise NotImplementedError(
                    "lens orbital_motion currently supports exactly one "
                    "companion (the engine's companion relations are "
                    "binary-only; mulensing.md '3+ lens bodies')."
                )
            if om == "keplerian" and self.orbit_ref[0] is None:
                raise ValueError(
                    "lens orbital_motion: keplerian requires `orbit: "
                    "<orbit instance name>` on the companion's entry, "
                    "naming the orbit that moves the lens bodies (the same "
                    "vocabulary astrometryinstrument's rel mode uses)."
                )
            if om == "keplerian" and self.n_sources > 1:
                raise NotImplementedError(
                    "lens orbital_motion: keplerian currently supports a "
                    "single source (theta_E's per-source normalization "
                    "would give each source its own Einstein-unit s(t))."
                )

        self.kep_orbit_idx = None
        if om == "keplerian":
            self.kep_orbit_idx = self._resolve_orbit_ref(
                config_manager, self.orbit_ref[0]
            )

        # Source orbital motion -- xallarap (C25): keys on the mulensevent
        # block; the trajectory shift is applied by the dispatcher here.
        som = ev.get("source_orbital_motion")
        self.source_orbital_motion = [som]
        self.source_orbit_ref = [ev.get("source_orbit")]
        self.xal_orbit_idx = None
        if som is not None:
            if som == "linear":
                raise NotImplementedError(
                    "source_orbital_motion: linear is deliberately not "
                    "offered: a linear source drift is exactly degenerate "
                    "with (t_E, t_0, u_0, alpha) in the light curve alone "
                    "(notes/orbital_motion_and_nbody.txt section 2).  Use "
                    "'keplerian', or wait for the per-star proper-motion "
                    "predicate that would make a linear mode meaningful."
                )
            if som != "keplerian":
                raise ValueError(
                    f"mulensevent.source_orbital_motion must be 'keplerian' "
                    f"(or absent for a static source), got '{som}'."
                )
            if self.source_orbit_ref[0] is None:
                raise ValueError(
                    "mulensevent.source_orbital_motion: keplerian requires "
                    "`source_orbit: <orbit instance name>` on the "
                    "mulensevent block, naming the orbit of the luminous "
                    "source about its (dark or faint) companion."
                )
            if self.n_sources > 1:
                raise NotImplementedError(
                    "source_orbital_motion currently supports a single "
                    "luminous source; the linked binary-source case (both "
                    "sources on one orbit, opposite offsets scaled by the "
                    "mass ratio) is the next step of review 8.6.9."
                )
            self.xal_orbit_idx = self._resolve_orbit_ref(
                config_manager, self.source_orbit_ref[0]
            )
            if self.xal_orbit_idx == self.kep_orbit_idx and (
                self.kep_orbit_idx is not None
            ):
                raise ValueError(
                    f"[{self.prefix}] the lens companion's (`orbit:`) and "
                    f"the event's (`source_orbit:`) references name the "
                    "SAME orbit; the lens binary and the source binary are "
                    "different systems."
                )

    def _resolve_orbit_ref(self, config_manager, ref):
        """Resolve an orbit reference (instance name or index) -- the same
        vocabulary astrometryinstrument's rel mode uses."""
        sys_cfg = getattr(config_manager, "system_config", None) or {}
        orbit_names = component_instance_names(sys_cfg, "orbit")
        if isinstance(ref, int) or str(ref).isdigit():
            idx = int(ref)
            if orbit_names and idx >= len(orbit_names):
                raise ValueError(
                    f"[{self.prefix}] orbit index {idx} out of range; "
                    f"orbits are {orbit_names}."
                )
            return idx
        if ref in orbit_names:
            return orbit_names.index(ref)
        raise ValueError(
            f"[{self.prefix}] unknown orbit '{ref}'; orbits are {orbit_names}."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _translate_s_bounds_to_log_s(self):
        """Rewrite user-supplied lens.<j+1>.s lower/upper bounds onto log_s.

        s is a derived parameter (s = 10**log_s); the sampling bounds live
        on log_s.  Companion slot j is LENS ELEMENT j+1 -- element 0 is the
        masked primary, whose entries would be warn-dropped, so targeting
        it here would silently lose the user's bound.
        Keys are already standardized to lens.<i>.<param> form.
        """
        up = self.config_manager.user_params
        for j in range(self.n_companions):
            elem = j + 1
            entry = up.get(f"lens.{elem}.s")
            if not isinstance(entry, dict):
                continue
            if "lower" not in entry and "upper" not in entry:
                continue
            log_key = f"lens.{elem}.log_s"
            log_entry = up.get(log_key)
            if not isinstance(log_entry, dict):
                log_entry = {}
            for bound in ("lower", "upper"):
                if bound not in entry:
                    continue
                val = float(entry.pop(bound))
                if val <= 0.0:
                    raise ValueError(
                        f"lens.{elem}.s {bound} bound must be positive "
                        f"(s > 0); got {val}."
                    )
                log_entry[bound] = float(np.log10(val))
            up[log_key] = log_entry
            logger.info(
                f"Translated lens.{elem}.s bound(s) to log_s "
                f"(log10): {log_key} = "
                f"{{{', '.join(f'{b}: {log_entry[b]:.4f}' for b in ('lower', 'upper') if b in log_entry)}}}."
            )

    def _resolve_t0_par(self, event_config, config_manager):
        """t0_par from the mulensevent block, the source.0.t_0 seed, or the
        historical fallback.  MulensInstrument re-resolves the final value
        in stage 1 (MMEXOFAST seeds arrive after this snapshot)."""
        if "t0_par" in event_config:
            return float(event_config["t0_par"])
        entry = config_manager.user_params.get("source.0.t_0")
        if isinstance(entry, dict):
            val = entry.get("initval")
        else:
            val = entry
        # List-valued initval (P4 multi-seed sampling): t0_par is just a
        # numeric reference epoch, not a per-seed value, so use seed 0.
        if isinstance(val, (list, tuple)):
            val = val[0]
        return float(val) if val is not None else 2450000.0

    @property
    def prefix(self):
        return "lens"

    @classmethod
    def get_utilities(cls):
        from ...utilities import mmexofast_to_params
        from ...utilities.registry import (
            UtilitySpec,
            argparse_subprocess_runner,
        )

        return [
            UtilitySpec(
                name="mmexofast_to_params",
                label="MMEXOFAST -> params.yaml",
                description=(
                    "Convert an MMEXOFAST fit-results JSON into an EXOZIPPy "
                    "params.yaml seeding the microlensing parameters."
                ),
                component_keys=["lens"],
                available=True,
                build_parser=mmexofast_to_params.build_parser,
                run=argparse_subprocess_runner(
                    "exozippy.utilities.mmexofast_to_params"
                ),
            ),
        ]

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
                "key": "orbital_motion",
                "kind": "option",
                "accepts": ["linear", "keplerian"],
                "required": False,
                "doc": (
                    "Time-dependent geometry of THIS companion "
                    "(conventions.md C24; review 8.6.8). Absent = static "
                    "s/alpha, the default. 'linear' samples rates ds_dt "
                    "[Einstein radii/yr] and dalpha_dt [deg/yr user, "
                    "rad/yr internal], anchored at t0_par, with a soft "
                    "beta < 1 bound-orbit potential (Skowron+2011 A19). "
                    "'keplerian' drives s(t)/alpha(t) from the orbit "
                    "component named by 'orbit:' with NO new free "
                    "parameters. Not valid on the primary's entry; exactly "
                    "one companion for now."
                ),
            },
            {
                "key": "orbit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "orbital_motion: keplerian only -- the orbit instance "
                    "(name or index) whose Keplerian elements move the "
                    "lens bodies, mirroring astrometryinstrument's rel-"
                    "mode vocabulary. The orbit's bodies should be the "
                    "lens bodies, so the RV mass function and the lens "
                    "geometry share one mass chain."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Lifecycle stages
    # ------------------------------------------------------------------

    def build_maps(self):
        """Stage 2: index maps.

        Two families.  The PER-SOURCE maps (`source_map`,
        `primary_source_map`'s value) are stage-1 compat for the dispatcher
        and MulensInstrument.  The FULL-LENGTH maps (one entry per LENS
        ELEMENT: `event_map`, `primary_lens_map`, `primary_source_map`,
        `companion_body_map`, `lens_kep_orbit_map`) exist so the
        per-element expression machinery can PROVE alignment and slice them
        to the active (companion) elements -- a length-1 map under a masked
        vector fails the aligned check by design (component.py).  Entry 0
        of each full-length map is a filler that the primary's inactive
        mask keeps out of every expression.
        """
        _, p_ndx = self.bodies[0]
        n = self.n_elements

        # Per-source compat maps (dispatcher, likelihood, instrument reads).
        self.source_map = np.array(
            [ndx for (_, ndx) in self.source_bodies[0]], dtype=int
        )
        self.lens_map = np.array([p_ndx], dtype=int)

        # Full-length per-element maps for the companion expressions.
        self.event_map = np.zeros(n, dtype=int)
        self.primary_lens_map = np.full(n, p_ndx, dtype=int)
        self.primary_source_map = np.full(
            n, int(self.source_map[0]), dtype=int
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
        if self.kep_orbit_idx is not None:
            self.lens_kep_orbit_map = np.full(n, self.kep_orbit_idx, dtype=int)

    def register_parameters(self, system):
        """Stage 3: the per-body manifest (masked primary)."""
        # s is derived from the sampled log_s; move any user s bounds onto
        # log_s before the manifest/relaxation engine run.  In keplerian
        # orbital-motion mode there IS no log_s.
        if self.orbital_motion[0] != "keplerian":
            self._translate_s_bounds_to_log_s()

        if self.n_companions >= 1:
            # Seed alpha hints (degrees, user unit) from user xalpha/yalpha
            # entries so inspect_start can display alpha before the
            # expression graph is built.  Companion slot j is ELEMENT j+1.
            if self.orbital_motion[0] != "keplerian":
                up = self.config_manager.user_params
                for j in range(self.n_companions):
                    elem = j + 1
                    ca_entry = up.get(f"lens.{elem}.xalpha") or {}
                    sa_entry = up.get(f"lens.{elem}.yalpha") or {}
                    ca = (
                        ca_entry.get("initval")
                        if isinstance(ca_entry, dict)
                        else ca_entry
                    )
                    sa = (
                        sa_entry.get("initval")
                        if isinstance(sa_entry, dict)
                        else sa_entry
                    )
                    # List-valued initval (P4 multi-seed): use seed 0.
                    if isinstance(ca, (list, tuple)):
                        ca = ca[0] if ca else None
                    if isinstance(sa, (list, tuple)):
                        sa = sa[0] if sa else None
                    if ca is not None and sa is not None:
                        alpha_deg = float(
                            np.arctan2(float(sa), float(ca)) * _RAD_TO_DEG
                        )
                        self.config_manager.add_hint(
                            f"lens.{elem}.alpha", alpha_deg
                        )

            # q's mass dependencies are typed by the companion components.
            # All companions of ONE type share a single full-length body
            # map, and calc_q divides the sliced companion-mass vector by
            # the sliced primary-mass vector elementwise.  Mixed types
            # would need one dep per type plus a per-element selector the
            # manifest cannot express; no shipped or tested configuration
            # has a mixed-type multi-companion lens, so it is refused
            # rather than silently mis-paired.
            c_types = {t for (t, _) in self.bodies[1:]}
            if len(c_types) > 1:
                raise NotImplementedError(
                    f"lens: companions of mixed component types "
                    f"({sorted(c_types)}) are not supported by the "
                    f"per-body lens component (q's mass dependency is one "
                    f"typed vector).  Declare the star-type companion as a "
                    f"planet-mass star block, or file an issue."
                )
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
        # inactive there (role 4), pinned at a bookkeeping value strictly
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
                # Linear lens orbital motion (C24): rates anchored at
                # t0_par (the parallax anchor -- one epoch is what makes
                # the two effects composable), plus the derived beta
                # (Skowron A19), softly bounded below 1 in
                # build_likelihood.
                "companion_linear": {
                    **static_geom,
                    "ds_dt": None,
                    "dalpha_dt": None,
                    "beta": "default",
                },
                # keplerian mode (C24): NO sampled geometry coordinates at
                # all -- s and alpha are DERIVED from the referenced orbit
                # (evaluated at t0_par for the reported values; the
                # per-epoch series goes to the backends via
                # _companion_geometry_series).  log_s / xalpha / yalpha do
                # not exist in this mode, exactly as a linear-law band has
                # no (q1, q2).
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
                "ds_dt": {"inactive_value": 0.0},
                "dalpha_dt": {"inactive_value": 0.0},
                "beta": {"inactive_value": 0.0},
            }
            self.manifest = mode_manifest(
                modes, table, options=options, where=self.prefix
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
            and self.orbital_motion[0] == "keplerian"
        ):
            context_nodes = dict(context_nodes or {})
            context_nodes.setdefault(
                "lens_t0_par",
                pt.as_tensor_variable(float(self.t0_par[0])),
            )
        return super().add_parameter(model, param_name, system, context_nodes)

    def _validate_q_start(self):
        """Stage 7: check the START value of the mass ratio, loudly and
        once.  Ported from the pre-split Lens; the fatal/derived split and
        the range warning are unchanged in meaning, re-indexed for the
        masked primary: NaN is fatal only where it MEANS something, which
        is the FIRST COMPANION -- lens element 1 -- because
        symbolic_physics.get_symbol_map maps a single companion, so only
        that element's solve failure indicates a non-finite body mass.
        Elements 2+ are never solved by the engine (their starts come from
        user body-mass hints or stay NaN as bookkeeping while the graph
        recomputes q from the mass nodes).  The inactive element 0 (pinned
        at 1.0) is excluded from every scan."""
        if self.n_companions < 1 or self.q.initval is None:
            return
        q0 = np.atleast_1d(np.asarray(self.q.initval, dtype=float)).ravel()
        active = np.array(
            [self.q.element_is_active(i) for i in range(q0.size)]
        )
        nan = np.isnan(q0) & active
        fatal = [
            i
            for i in np.flatnonzero(nan)
            if i == 1 or not self.q.element_is_derived(int(i))
        ]
        if fatal:
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

    def build_likelihood(self, model, system):
        """Stage 7: Observational penalties on the lensing geometry.

        STAGE-1 NOTE: these potentials stay here (names `lens.*`) and move
        to their owning components in stage 2 (event rate / singularities /
        behind-lens -> mulensevent; the fitu0te Jacobian -> source), so the
        stage-0 fixtures' potential names survive this stage.
        """
        self._validate_q_start()

        event = system.mulensevent
        source = system.source

        # Parameter refs for the post-fit corner plot (plot_corner has no
        # system handle; the refs survive distribute_posterior).
        self._corner_params = {
            "t_0": source.t_0,
            "u_0": source.u_0,
            "t_E": event.t_E,
            "rho": getattr(source, "rho", None),
        }

        # fitu0te's change of variables: u_0 = u0te/t_E, so |du_0/du0te| =
        # 1/t_E and the correction is -log(t_E) per OPTED-IN source track
        # (per-instance flags now; a single opted source reproduces the
        # pre-split single term).
        n_u0te = int(sum(getattr(source, "_fitu0te", [])))
        if n_u0te:
            pm.Potential(
                f"{self.prefix}.fitu0te_jacobian",
                -n_u0te * pt.log(pt.maximum(event.t_E.value[0], 1e-12)),
            )

        # fitpirel's change of variables is NONLINEAR (unlike fitmurel's):
        # D_l = 1000/(pi_rel + 1000/D_s)  =>
        # |dD_l/dlog_pi_rel| = ln10 * pi_rel * D_l^2 / 1000.
        if getattr(event, "_fitpirel", False):
            l_idx = int(self.bodies[0][1])
            d_l_scalar = system.star.distance.value[l_idx]
            pi_rel_v = event.pi_rel.value[0]
            pm.Potential(
                f"{self.prefix}.fitpirel_jacobian",
                pt.log(np.log(10.0) * pi_rel_v * d_l_scalar**2 / 1000.0),
            )

        # GEOCENTRIC mu_rel: the event-rate selection is the sky-sweep rate
        # in the frame the event is observed in, and it is also the divisor
        # of t_E/pi_E, so the singularity guard belongs on it.  These are
        # length-1 event vectors now, so the 8.6.18 "one encounter, one
        # weight" indexing ([0]) and the whole-vector sums coincide by
        # construction -- the shape finally matches the physics.
        mu_rel_geo = event.mu_rel_geo_mag.value
        theta_E = event.theta_E.value

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
        d_s = system.star.distance.value[self.source_map]
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

        if self.xal_orbit_idx is not None:
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

        if self.orbital_motion[0] == "linear":
            # The bound-orbit rope for the linear rates (C24; Skowron A19):
            # ONE LENS ORBIT, ONE BOUND (review 8.6.18).  beta is a
            # per-companion vector with a masked primary now, so the bound
            # reads THE COMPANION'S element -- element 1, never element 0,
            # whose pinned bookkeeping value would otherwise contribute a
            # constant logp offset (the design's one whole-vector-reduction
            # hazard, section 2.3).
            pm.Potential(
                f"{self.prefix}.bound_orbit",
                soft_upper_bound(self.beta.value[1], 1.0, scale=0.1),
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

    # ------------------------------------------------------------------
    # Magnification (stage-1 home; moves to MulensEvent in stage 2)
    # ------------------------------------------------------------------

    def _alpha_deg(self, j=0):
        """Trajectory angle of companion ``j`` in DEGREES -- the unit both
        magnification backends take, while lens.alpha's internal unit is
        radians.  ``j`` is the COMPANION SLOT; the vector element is j+1
        (element 0 is the masked primary)."""
        return self.alpha.value[j + 1] * _RAD_TO_DEG

    def _companion_geometry_series(self, times, system):
        """Per-epoch companion geometry ``(s_t, alpha_t_deg)`` for companion
        0 (vector element 1), or ``None`` when the lens geometry is static.
        See the pre-split docstring for the C24 conventions; only the
        element indexing and the event-parameter homes changed."""
        om = self.orbital_motion[0]
        if om is None:
            return None
        dt_yr = (times - self.t0_par[0]) / DAYS_PER_YEAR
        if om == "linear":
            s_t = self.s.value[1] + self.ds_dt.value[1] * dt_yr
            alpha_t_deg = (
                self._alpha_deg(0)
                + self.dalpha_dt.value[1] * _RAD_TO_DEG * dt_yr
            )
            return s_t, alpha_t_deg
        # keplerian: the same physics function the reported s/alpha use
        # (evaluated there at t0_par), here over the epoch vector.
        j = self.kep_orbit_idx
        orbit = system.orbit
        event = system.mulensevent
        s_t, alpha_t_rad = lens_geometry_from_orbit(
            pt.as_tensor_variable(times),
            orbit.tp.value[j],
            orbit.n.value[j],
            orbit.ecc.value[j],
            orbit.sinw.value[j],
            orbit.cosw.value[j],
            orbit.cosi.value[j],
            orbit.bigomega.value[j],
            orbit.a.value[j],
            event.theta_E.value[0],
            system.star.distance.value[self.lens_map[0]],
            event.pi_E_N.value[0],
            event.pi_E_E.value[0],
        )
        return s_t, alpha_t_rad * _RAD_TO_DEG

    def _source_offset_series(self, times, system):
        """Per-epoch xallarap trajectory shift ``(dtau_t, du_t)``, or None
        for a static source (conventions.md C25; review 8.6.9).  Ported;
        the event chain now reads system.mulensevent."""
        if self.xal_orbit_idx is None:
            return None
        j = self.xal_orbit_idx
        orbit = system.orbit
        event = system.mulensevent
        a1 = (
            orbit.a.value[j]
            * orbit.m_companion.value[j]
            / orbit.m_total.value[j]
        )
        d_s = system.star.distance.value[self.source_map[0]]
        args = (
            orbit.tp.value[j],
            orbit.n.value[j],
            orbit.ecc.value[j],
            orbit.sinw.value[j],
            orbit.cosw.value[j],
            orbit.cosi.value[j],
            orbit.bigomega.value[j],
            a1,
            event.theta_E.value[0],
            d_s,
        )
        sig_N, sig_E = source_offset_from_orbit(
            pt.as_tensor_variable(times), *args
        )
        sig_N0, sig_E0 = source_offset_from_orbit(
            pt.as_tensor_variable(np.array([float(self.t0_par[0])])), *args
        )
        mu_mag = pt.maximum(event.mu_rel_geo_mag.value[0], MU_REL_FLOOR)
        mu_n_hat = event.mu_dec_rel_geo.value[0] / mu_mag
        mu_e_hat = event.mu_ra_rel_geo.value[0] / mu_mag
        return xallarap_trajectory_shift(
            sig_N - sig_N0[0], sig_E - sig_E0[0], mu_n_hat, mu_e_hat
        )

    def _get_safe_mm_params(self, system, index=0):
        """Range-limited single-source trajectory params.  ``index`` is the
        SOURCE slot (an element of the source component).  The t_E/theta_E/
        pi_E entries are the EVENT's (element 0 of mulensevent's length-1
        vectors).  The floors and the deliberately-absent NaN substitution
        are unchanged -- see the pre-split docstring for the full history
        (every value here is finite for every finite raw vector; a NaN
        propagates to logp, the sampler's own reject signal)."""
        event = system.mulensevent
        source = system.source

        tE_raw = event.t_E.value[0]
        u0_raw = source.u_0.value[index]
        theta_E_raw = event.theta_E.value[0]

        tE_safe = pt.maximum(tE_raw, T_E_FLOOR)
        u0_safe = apply_u_0_floor(u0_raw)
        is_physical = pt.gt(theta_E_raw, THETA_E_LENSING_MIN)

        # Keys are the CANONICAL parameter names, matching op.py's
        # _base_mm_params exactly (review 4.6.1).
        return {
            "t_0": source.t_0.value[index],
            "u_0": u0_safe,
            "t_E": tE_safe,
            "pi_E_N": pt.switch(is_physical, event.pi_E_N.value[0], 0.0),
            "pi_E_E": pt.switch(is_physical, event.pi_E_E.value[0], 0.0),
        }

    def _get_binary_mm_params(self, system, index=0):
        """Params for a binary lens.  ``index`` is the SOURCE slot; the lens
        bodies are shared by all sources.  s/q/alpha are indexed by
        COMPANION (binary = companion 0 = VECTOR ELEMENT 1; element 0 is
        the masked primary)."""
        s = self._get_safe_mm_params(system, index)
        return {
            **s,
            "s": self.s.value[1],
            "q": clip_q(self.q.value[1]),
            "alpha": self._alpha_deg(0),
        }

    def get_magnification(self, times, obs_pos, system, index=0):
        """Symbolic Paczynski magnification including parallax (PSPL only).
        Ported; see the pre-split docstring for the obs_pos convention and
        the frozen-coordinates note."""
        source_ndx = self.source_map[index]
        ra = system.star.ra.value[source_ndx]
        dec = system.star.dec.value[source_ndx]

        delta_e, delta_n = observer_sky_offset(obs_pos, ra, dec, xp=pt)

        p = self._get_safe_mm_params(system, index)
        # MulensModel convention: delta_tau = -delta_N*pi_E_N -
        # delta_E*pi_E_E (negative on both N and E, matching Skowron+2011
        # via MulensModel's sign choice).
        tau_p = (
            (times - p["t_0"]) / p["t_E"]
            - delta_n * p["pi_E_N"]
            - delta_e * p["pi_E_E"]
        )
        u_p = p["u_0"] + delta_n * p["pi_E_E"] - delta_e * p["pi_E_N"]

        # Xallarap enters at EXACTLY this slot: parallax is the OBSERVER's
        # offset, this is the SOURCE's own (C25).  getattr, not a bare
        # attribute: test harnesses borrow this method onto minimal fakes
        # (tests/test_trajectory_sanitization.py).
        if getattr(self, "xal_orbit_idx", None) is not None:
            xal = self._source_offset_series(times, system)
            tau_p = tau_p + xal[0]
            u_p = u_p + xal[1]
        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        return (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

    def uses_op(self, index=0):
        """True if get_magnification_op will dispatch to the MulensModel Op.
        Event-level property; ``index`` is ignored beyond backward
        compatibility."""
        n_lenses = self.n_lens_bodies[0]
        use_rho = self.finite_source[0]
        forced = self.use_op[0]
        return forced or (n_lenses > 1) or use_rho

    def sampler_requirements(self):
        """Binary/finite-source lenses use the non-differentiable Op; PSPL
        uses the symbolic path.  Unchanged."""
        if any(self.uses_op(i) for i in range(len(self.n_lens_bodies))):
            return {
                "incompatible": {"nuts", "numpyro", "blackjax"},
                "recommended": "ptde_async",
                "reason": (
                    "binary/finite-source microlensing uses the MulensModel "
                    "Op, which is not differentiable -- gradient-based "
                    "samplers produce invalid results"
                ),
            }
        return {}

    def _frozen_op_coords_deg(self, system, source_ndx):
        """(ra, dec) in degrees baked into the MulensModel / VBM Op, ONCE.
        Ported verbatim; see the pre-split docstring for why the freeze is
        deliberate and numerically free."""
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

    def _resolve_quadratic_ld(self, u2, effective_bandpass):
        """Can the selected backend honour the band's second LD
        coefficient?  Ported verbatim (see the pre-split docstring for the
        backend capabilities and the deliberate non-flip of the single-lens
        default)."""
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
                "vbm_direct' to honour u2, or declare 'ld_law: linear' on "
                "the band to make the linear law deliberate and drop the "
                "unconstrained coordinate."
            )
        return False

    def get_magnification_op(
        self, times, obs_pos, system, index=0, u1=None, u2=None, bandpass=None
    ):
        """Magnification dispatcher.  Ported; the param-vector layout and
        the Op input contract are unchanged (op.py unpacks by position) --
        only the ELEMENT reads shifted: the Op's companion loop index j is
        its own 0-based companion count, and the vector element is j+1."""
        if self.n_lens_bodies[0] > 2 and self.backend != "vbm_direct":
            raise NotImplementedError(
                f"{self.n_lens_bodies[0]}-lens magnification requires "
                "backend: vbm_direct (VBMicrolensing MultiMag2); the "
                "MulensModel backend supports at most 2 lens bodies."
            )

        if not self.uses_op(index):
            return self.get_magnification(times, obs_pos, system, index)

        source = system.source
        source_ndx = self.source_map[index]
        ra_deg, dec_deg = self._frozen_op_coords_deg(system, source_ndx)
        coords = f"{ra_deg}d {dec_deg}d"

        use_rho = self.finite_source[0]
        n_lenses = self.n_lens_bodies[0]

        # Apply LD only for finite-source and when a band is connected.
        effective_bandpass = bandpass if (use_rho and u1 is not None) else None
        use_u2 = self._resolve_quadratic_ld(u2, effective_bandpass)

        times_tensor = pt.as_tensor_variable(times)
        obs_tensor = pt.as_tensor_variable(obs_pos)

        # Per-epoch xallarap trajectory shift (source orbital motion, C25)
        # -- None for a static source.
        source_series = self._source_offset_series(times, system)
        if source_series is not None and self.backend == "mulensmodel":
            raise NotImplementedError(
                "source_orbital_motion requires backend: vbm_direct (the "
                "point-source single-lens case takes the symbolic path "
                "and needs neither).  MulensModel's native xi_* xallarap "
                "is not wired as a backend -- unnecessary: C25's machinery "
                "covers it, and the xi_* mapping is verified "
                "(conventions.md C25; a published xi_* solution seeds a "
                "config directly, see examples/ob170114)."
            )

        single_lens_vbm = (
            n_lenses == 1
            and use_rho
            and (use_u2 or source_series is not None)
            and self.backend == "vbm_direct"
        )

        if single_lens_vbm:
            # ESPL through VBM: see the pre-split comment for when this
            # branch is reached (quadratic LD or xallarap only).
            sp = self._get_safe_mm_params(system, index)
            param_list = [
                sp["t_0"],
                sp["u_0"],
                sp["t_E"],
                sp["pi_E_N"],
                sp["pi_E_E"],
                source.rho.value[index],
            ]
            if effective_bandpass is not None:
                param_list.append(u1)
                if use_u2:
                    param_list.append(u2)
            mag_op = VBMDirectMagOp(
                coords=coords,
                n_companions=0,
                use_rho=True,
                bandpass=effective_bandpass,
                quadratic_ld=use_u2,
                source_motion=source_series is not None,
            )
            op_inputs = [pt.stack(param_list), times_tensor, obs_tensor]
            if source_series is not None:
                op_inputs += [
                    pt.as_tensor_variable(source_series[0]),
                    pt.as_tensor_variable(source_series[1]),
                ]
            return mag_op(*op_inputs)

        # Per-epoch companion geometry (lens orbital motion, C24) -- None
        # for a static lens.
        geometry_series = (
            self._companion_geometry_series(times, system)
            if n_lenses >= 2
            else None
        )

        if n_lenses >= 2 and self.backend == "vbm_direct":
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
            for j in range(self.n_companions):
                # Companion slot j = vector element j+1 (masked primary).
                param_list.extend(
                    [
                        self.s.value[j + 1],
                        clip_q(self.q.value[j + 1]),
                        self._alpha_deg(j),
                    ]
                )
            if effective_bandpass is not None:
                param_list.append(u1)
                if use_u2:
                    param_list.append(u2)
            mag_op = VBMDirectMagOp(
                coords=coords,
                n_companions=self.n_companions,
                use_rho=use_rho,
                bandpass=effective_bandpass,
                quadratic_ld=use_u2,
                orbital_motion=geometry_series is not None,
                source_motion=source_series is not None,
            )
            if geometry_series is not None or source_series is not None:
                op_inputs = [pt.stack(param_list), times_tensor, obs_tensor]
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
        elif n_lenses == 2:
            if self.orbital_motion[0] == "keplerian":
                raise NotImplementedError(
                    "orbital_motion: keplerian requires backend: vbm_direct."
                    "  MulensModel's own keplerian lens motion contradicts "
                    "its linear mode by a sign and is not a usable reference"
                    " (conventions.md section 6, measured 2026-08-27)."
                )
            bp = self._get_binary_mm_params(system, index)
            param_list = [
                bp["t_0"],
                bp["u_0"],
                bp["t_E"],
                bp["pi_E_N"],
                bp["pi_E_E"],
            ]
            if use_rho:
                param_list.append(source.rho.value[index])
            param_list.extend([bp["s"], bp["q"], bp["alpha"]])
            if geometry_series is not None:
                # MulensModel's LINEAR branch is definitional in the same
                # (ds_dt, dalpha_dt) and takes deg/yr; the rates are the
                # companion's vector elements (element 1).
                param_list.extend(
                    [
                        self.ds_dt.value[1],
                        self.dalpha_dt.value[1] * _RAD_TO_DEG,
                    ]
                )
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
                    "source_orbital_motion with a FORCED MulensModel "
                    "single-lens Op (use_op: true) is not wired; drop "
                    "use_op (the symbolic path carries xallarap) or use "
                    "finite_source with backend: vbm_direct."
                )
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
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = MulensMagOp(
                coords=coords,
                mag_method=self.mag_method[index],
                use_rho=use_rho,
                bandpass=effective_bandpass,
            )

        return mag_op(pt.stack(param_list), times_tensor, obs_tensor)

    # ------------------------------------------------------------------
    # Auto method brackets
    # ------------------------------------------------------------------

    def resolve_auto_vbbl(self, times_np, index=0):
        """Replace 'auto_vbbl' with a concrete method list for multi-body
        lenses.  Ported verbatim (see hpc_optimization.txt P1 for why the
        bracket machinery was removed)."""
        if self.mag_method[index] != "auto_vbbl":
            return
        if self.n_lens_bodies[0] < 2:
            return

        t_lo = float(np.min(times_np))
        t_hi = float(np.max(times_np))
        method = "VBM"
        self.mag_method[index] = [t_lo - 1.0, method, t_hi + 1.0]

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

    def plot_corner(self, idata, filename_prefix="debug"):
        """Corner plot of the fitted lensing geometry: t_0, u_0, t_E, s, q,
        alpha, rho -- whichever the event actually has.  The trajectory and
        event parameters live on the source/mulensevent components now;
        build_likelihood stashed their Parameter refs (this hook has no
        system handle).  Constant columns (the masked primary's s/q/alpha
        elements) are dropped by corner_utils' degenerate-grid filter."""
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

        def per_body_labels(param):
            if self.n_elements <= 2:
                return None
            return [f"{param}[{name}]" for name in self.names]

        param_specs = [
            (stash["t_0"], per_source_labels("t_0")),
            (stash["u_0"], per_source_labels("u_0")),
            (stash["t_E"], None),
        ]
        if stash.get("rho") is not None:
            param_specs.append((stash["rho"], per_source_labels("rho")))
        if self.n_companions >= 1:
            param_specs.append((self.s, per_body_labels("s")))
            param_specs.append((self.q, per_body_labels("q")))
            param_specs.append((self.alpha, per_body_labels("alpha")))

        samples, labels = collect_parameter_corner_samples(param_specs)
        save_corner_plot(samples, labels, f"{filename_prefix}_lens_corner.png")
