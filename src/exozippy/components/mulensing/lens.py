"""The microlensing LENS component (8.6.17 stage 1).

One instance per LENS BODY, primary first.  The per-companion geometry --
`s`, `alpha` (and their coordinates `log_s`, `xalpha`/`yalpha`), the mass
ratio `q`, and the linear orbital-motion rates -- lives here as vectors of
the component's own element count, with element 0 (the primary) INACTIVE
(manifest role 4, the Band/orbit mode idiom): a primary has no separation
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
from exozippy.components.parameterization import mode_manifest
from exozippy.config import PRECEDENCE_DEFAULT
from exozippy.outputs.prose import get_collector
from exozippy.potentials import soft_upper_bound

from .bodies import body_entries, derive_body_names, resolve_orbit_ref
from .physics import _Q_NAN_ADVICE, Q_MAX, Q_MIN

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
                "backend, mag_method, use_op, mmexofast, fit* flags, "
                "source_orbital_motion).  Pre-v0.1.0 configs put those on "
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
            if om == "keplerian" and n_sources > 1:
                raise NotImplementedError(
                    "lens orbital_motion: keplerian currently supports a "
                    "single source (theta_E's per-source normalization "
                    "would give each source its own Einstein-unit s(t))."
                )

        self.kep_orbit_idx = None
        if om == "keplerian":
            self.kep_orbit_idx = resolve_orbit_ref(
                config_manager, self.orbit_ref[0], self.prefix
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

        All FULL-LENGTH (one entry per LENS ELEMENT: `event_map`,
        `primary_lens_map`, `primary_source_map`, `companion_body_map`,
        `lens_kep_orbit_map`), so the per-element expression machinery can
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
                    f"typed vector).  Declare every companion with the "
                    f"SAME component type -- e.g. model the planet as a "
                    f"low-mass 'star' block (star.<name>.logmass reaches "
                    f"-9 dex) so all companions are star-type -- or file "
                    f"an issue.  NOTE: design 3.2 promised typed "
                    f"per-companion deps would survive the split; this "
                    f"refusal is a design contradiction awaiting a ruling "
                    f"(stage-1b review, finding 1)."
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
                pt.as_tensor_variable(float(system.mulensevent.t0_par[0])),
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

        **NaN is fatal only where it MEANS something**, which is the FIRST
        COMPANION -- lens element 1 post-split -- (review 1.6.5).  The
        split is not about q being derived -- it always is -- but about
        which elements the relaxation engine can actually solve:
        ``symbolic_physics.get_symbol_map`` maps a SINGLE companion, so for
        element 1 a NaN really does say the solve failed, i.e. one of the
        lens body masses is already non-finite, and the advice below is the
        right advice.  Elements 2 and up are never solved by the engine at
        all: `register_parameters` seeds them from USER body-mass entries
        only, skips the hint when there are none (see 2.6.6), and
        `resolve()` then leaves them NaN because q has no defaults.yaml
        initval.  That NaN is bookkeeping, not a start -- the graph
        recomputes q from the mass nodes, which carry finite defaults --
        and raising on it killed a 3+ body fit that would have run
        perfectly well.  Exactly the false-positive class
        :meth:`Source._validate_pspl_start`'s docstring warns about for the
        derived t_E/theta_E/pi_E (the ob161003 theta_E lesson).  The
        inactive element 0 (pinned at 1.0) is excluded from every scan.

        A q that genuinely reaches the magnification backend as NaN is
        still caught at runtime by ``clip_q_value``, which names the
        parameter.  The derived-ness test is kept for the skipped elements
        so that a future parameterization which SAMPLES one of them gets
        the raise back: for a sampled element the initval IS the start.
        """
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
        """Stage 7: start-value validation, and the bound-orbit rope for a
        linear-orbital-motion companion.  The EVENT potentials (event rate,
        singularity guards, source-behind-lens, the fitpirel Jacobian) are
        MulensEvent.build_likelihood's; the fitu0te Jacobian is
        Source.build_likelihood's.
        """
        self._validate_q_start()

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
