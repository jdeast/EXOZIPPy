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
coordinate flags, mmexofast, source_orbital_motion), the event potentials
(event rate, singularity guards, source-behind-lens, the fitpirel
Jacobian), the seeding hints, and the sampler-compatibility declaration.
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.config import (
    PRECEDENCE_DERIVED_DATA,
    PRECEDENCE_DERIVED_MIXED,
    PRECEDENCE_MULENS_LENS_DISTANCE,
    PRECEDENCE_MULENS_SOURCE_DISTANCE,
)
from exozippy.constants import DAYS_PER_YEAR
from exozippy.corner_utils import (
    collect_parameter_corner_samples,
    save_corner_plot,
)
from exozippy.outputs.prose import get_collector
from exozippy.potentials import soft_lower_bound
from exozippy.skyframe import observer_sky_offset, sky_basis

from ..galacticmodel.physics import expected_proper_motion
from . import mmexofast_support
from .bodies import body_entries, resolve_orbit_ref, validate_event_config
from .op import BinaryLensMagOp, MulensMagOp, VBMDirectMagOp
from .physics import (
    MU_REL_FLOOR,
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
    context_dep_names = frozenset({"earth_vperp_e", "earth_vperp_n"})

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
        # at stage 1 (MMEXOFAST seeds arrive after this snapshot) and writes
        # it back into t0_par[0] -- a length-1 list precisely so that write
        # is visible to every later reader of the same object.
        self.t0_par = [self._resolve_t0_par(ev, config_manager)]

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

        # Source orbital motion -- xallarap (C25): keys on this block; the
        # trajectory shift is applied by the magnification dispatcher below.
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
            self.xal_orbit_idx = resolve_orbit_ref(
                config_manager, self.source_orbit_ref[0], self.prefix
            )
            # The lens binary and the source binary are different systems,
            # so the companion's `orbit:` (keplerian lens motion, on the
            # lens component's entries) and this block's `source_orbit:`
            # must not name the same orbit.  Resolved from the raw lens
            # block: component construction order is the user's config key
            # order, so the Lens instance may not exist yet.
            lens_kep_refs = {
                resolve_orbit_ref(config_manager, e["orbit"], f"lens.{i}")
                for i, e in enumerate(sys_cfg.get("lens") or [])
                if isinstance(e, dict)
                and e.get("orbital_motion") == "keplerian"
                and e.get("orbit") is not None
            }
            if self.xal_orbit_idx in lens_kep_refs:
                raise ValueError(
                    f"[{self.prefix}] the lens companion's (`orbit:`) and "
                    f"the event's (`source_orbit:`) references name the "
                    "SAME orbit; the lens binary and the source binary are "
                    "different systems."
                )

    def _resolve_t0_par(self, event_config, config_manager):
        """t0_par from the mulensevent block, the source.0.t_0 seed, or the
        historical fallback.  MulensInstrument re-resolves the final value
        at stage 1 (MMEXOFAST seeds arrive after this snapshot)."""
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
        return "mulensevent"

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
                component_keys=["mulensevent"],
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
                    "an MMEXOFAST seed, or the median data time."
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
                "key": "mmexofast",
                "kind": "datafile",
                "accepts": "*.json",
                "required": False,
                "doc": (
                    "MMEXOFAST integration: a fit-results JSON path provides "
                    "seed initvals/scales for the microlensing parameters "
                    "plus the bad-data mask and error factors; 'auto' forces "
                    "an MMEXOFAST run on the raw light curves (cached at "
                    "<prefix>_mmexofast.json); false disables the automatic "
                    "run that otherwise happens when the params file lacks "
                    "start values for the microlensing parameters."
                ),
            },
            {
                "key": "mmexofast_options",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Extra MMEXOFASTFitter keyword arguments for the "
                    "automatic run (e.g. {no_parallax: false, "
                    "limb_darkening_coeffs_gamma: {W149: 0.3}}), forwarded "
                    "verbatim."
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
                    "Magnification engine for the multi-lens Op path, and "
                    "for a finite-source single lens whose band declares "
                    "'ld_law: quadratic' (only VBMicrolensing can apply a "
                    "quadratic limb-darkening profile). Default "
                    "'vbm_direct'; 'mulensmodel' is the A/B reference and "
                    "is linear-LD only."
                ),
            },
            {
                "key": "source_orbital_motion",
                "kind": "option",
                "accepts": ["keplerian"],
                "required": False,
                "doc": (
                    "Source orbital motion -- xallarap (conventions.md "
                    "C25; review 8.6.9). The luminous source's own "
                    "barycentric sky offset, driven by the orbit named in "
                    "'source_orbit:', enters the trajectory at exactly "
                    "the parallax slot, anchored at t0_par, with NO new "
                    "sampled parameters. 'linear' is deliberately not "
                    "offered: a linear source drift is exactly degenerate "
                    "with (t_E, t_0, u_0, alpha) in the light curve "
                    "alone. Single luminous source for now."
                ),
            },
            {
                "key": "source_orbit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "source_orbital_motion: keplerian only -- the orbit "
                    "instance (name or index) of the luminous source "
                    "about its (dark or faint) companion. Must differ "
                    "from the lens companion's 'orbit:' (the lens binary)."
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

    def _mass_initval(self, comp_type, ndx):
        """Best-effort mass initval (solMass) for a body at stage 3, from
        user_params mass or logmass entries; None when neither is given."""
        up = self.config_manager.user_params
        entry = up.get(f"{comp_type}.{ndx}.mass")
        val = entry.get("initval") if isinstance(entry, dict) else entry
        if val is not None:
            return float(val)
        entry = up.get(f"{comp_type}.{ndx}.logmass")
        val = entry.get("initval") if isinstance(entry, dict) else entry
        return float(10.0 ** float(val)) if val is not None else None

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
                    f"star being monitored for magnification.  A genuinely "
                    f"faint source (a brown dwarf, say) is a 'star' block "
                    f"with a low star.<name>.logmass."
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

    def _load_mmexofast_seeds(self):
        """Read an optional MMEXOFAST solutions file and push each fit as a
        per-seed hint set for multi-seed sampling (P4).

        MMEXOFAST emits multiple lightly-optimized solutions spanning the
        standard microlensing degeneracies.  Each fit's observable-space
        values (t_0, u_0, t_E, s, q, alpha, rho) are seeded into the
        relaxation engine, which back-solves the physical parameters
        (distances/masses/PMs) exactly as a user typing them into
        params.yaml K times would.  Enabled by a `mmexofast: <file>` key on
        the mulensevent config block (path relative to the run cwd, same as
        the light-curve `file:` key).

        The translation itself (seed sets, scale hints, jd_offset handling,
        alpha/log_s conventions, and the post-split target paths --
        source.0.*, mulensevent.0.t_E, lens.1.*) lives in
        mmexofast_support.push_seed_hints, shared with MulensInstrument's
        stage-1a auto-initialization (which also applies the JSON's
        bad-data mask and error factors -- masks must exist before the
        photometry is read, which is why the instrument owns that half).
        """
        mmx_file = self.config[0].get("mmexofast") if self.config else None
        # Only an explicit file path is handled here. "auto" / absent-key
        # auto-initialization is owned by MulensInstrument (stage 1), which
        # pushes the seed hints itself before this method ever runs; False
        # opts out entirely.
        if not isinstance(mmx_file, str) or mmx_file == "auto":
            return
        # None means the file is ABSENT (warn and run unseeded, as before);
        # a file that exists but cannot be parsed raises out of load_json.
        # exozippy did not write a user-named file and so cannot regenerate
        # it -- only run_or_load's own cache has that recovery.
        data = mmexofast_support.load_json(mmx_file)
        if data is None:
            logger.warning(f"No seeds loaded from '{mmx_file}'.")
            return
        mmexofast_support.push_seed_hints(
            data,
            self.config_manager,
            want_rho=self.finite_source,
            is_binary=self.n_companions >= 1,
            source=mmx_file,
        )

    def register_parameters(self, system):
        """Stage 3: Declare the event-level manifest and push hints."""
        self._validate_bodies(system)
        self._load_mmexofast_seeds()

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
            # relation upgrades the start from any pm/mmexofast seeds.
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
                    mp = "log_q"
                    comp = getattr(system, "planet", None)
                    if comp is not None:
                        mp = comp.config[c_idx].get(
                            "mass_parameterization", "log_q"
                        )
                    if mp != "log_q":
                        reason = (
                            "the companion samples a linear mass, not log_q"
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

        if self.n_companions >= 2:
            # The symbolic relaxation engine only knows the binary mass-sum
            # and q relations (see symbolic_physics.get_symbol_map), so for
            # 3+ lens bodies the mlens_total and per-element q initvals are
            # seeded from the per-body mass initvals instead.  The q and
            # log_s hints target LENS ELEMENTS: companion slot j is lens
            # element j+1 (the off-by-one this refactor is named for).
            body_masses = [
                self._mass_initval(c_type, c_ndx)
                for c_type, c_ndx in self.lens_bodies
            ]
            # Seeded at PRECEDENCE_DERIVED_MIXED: overrides defaults, yields
            # to explicit user values.  Loud, once, at config time, because
            # the alternative -- a start that quietly comes from nowhere --
            # is what review 1.6.5 traced (and what 2.6.6 asks be said out
            # loud until the relations are generalized).  A WARNING and not
            # an INFO: for a 2-body lens the engine derives all of this from
            # ANY of the masses, q or the trajectory, so a user who has
            # never had to supply body masses gets no other signal that a
            # third body changes the rules.
            user_q = [
                f"lens.{j + 1}.q"
                for j in range(1, self.n_companions)
                if self.config_manager.user_params.get(f"lens.{j + 1}.q")
                is not None
            ]
            if user_q:
                logger.warning(
                    f"{self.prefix}: {', '.join(user_q)} sets the START of "
                    "a derived mass ratio but CANNOT set the companion "
                    "mass it is computed from -- the relaxation engine's q "
                    "relation covers the first companion (lens element 1) "
                    "only (see mulensing/symbolic_physics.py).  The fit "
                    "will run at the masses, not at the q you typed.  "
                    "Supply <component>.<body>.mass (or logmass) for every "
                    "lens body instead."
                )
            if any(m is None for m in body_masses):
                missing = [
                    f"{ct}.{cn}"
                    for (ct, cn), m in zip(self.lens_bodies, body_masses)
                    if m is None
                ]
                logger.warning(
                    f"{self.prefix}: no mass initval for lens body/bodies "
                    f"{missing}, so mulensevent.mlens_total and the "
                    "per-companion q starts fall back to defaults.  A lens "
                    "with 3+ bodies REQUIRES explicit body masses: the "
                    "engine's mass-sum and q relations are binary-only, so "
                    "nothing else can supply them (review 2.6.6).  Add "
                    f"mass (or logmass) initvals for {missing} to the "
                    "params file."
                )
            else:
                self.config_manager.add_hint(
                    "mulensevent.0.mlens_total",
                    float(sum(body_masses)),
                    rank=PRECEDENCE_DERIVED_MIXED,
                )
                for j, m_c in enumerate(body_masses[1:]):
                    q_j = m_c / body_masses[0]
                    self.config_manager.add_hint(
                        f"lens.{j + 1}.q", q_j, rank=PRECEDENCE_DERIVED_MIXED
                    )
                    self.config_manager.add_scale_hint(
                        f"lens.{j + 1}.q", 0.1 * q_j
                    )

            # The s <-> log_s relation is mapped only for the first
            # companion (lens element 1), so for 3+ body lenses seed the
            # remaining companions' log_s from any user s initval.
            up = self.config_manager.user_params
            for j in range(1, self.n_companions):
                entry = up.get(f"lens.{j + 1}.s")
                s_val = (
                    entry.get("initval") if isinstance(entry, dict) else entry
                )
                if s_val is None or float(s_val) <= 0.0:
                    continue
                self.config_manager.add_hint(
                    f"lens.{j + 1}.log_s",
                    float(np.log10(float(s_val))),
                    rank=PRECEDENCE_DERIVED_MIXED,
                )

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
        self._seed_expected_pm(pm_expected, l_idx, "thin_disk", 4000.0)

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
            self._seed_expected_pm(pm_expected, s_idx, "bulge", 8000.0)

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

        But where a config DOES imply the proper motion, a prior mean
        dropped on top fights it.  Measured at the seed (raw = 0), chi2/N
        ungated vs gated:

            ob140939 (pi_E_N/pi_E_E measured, Yee+2015)  3.04 -> 179.1 | 3.04
            ob161003 (two sources, t_E + rho each)       1.72 ->   3.9 | 1.72
            DC2018_128 (t_0/u_0/t_E/s/q/alpha/rho)       1.42 ->   1.21 (kept)
            ob08092 (t_0/u_0/t_E only, PSPL)             1.50 ->   1.42 (kept)

        So the gate below is what keeps this from making published
        solutions worse.  Filling only the *direction* and leaving the
        magnitude to the data would serve every case at once, but the
        direction is not a symbol, so provenance cannot express it
        per-symbol; that needs a basis change (mu_rel_mag, mu_rel_pa) which
        is a sampling-geometry question and does not belong here.
        tests/test_seed_quality.py pins all four numbers.

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
        so the earlier blocker returns first and this gate is unreachable
        for the only multi-source example shipped.  Stripping those four
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
        # Skip when something already implies the direction or the
        # magnitude.
        up = self.config_manager.user_params
        blockers = [
            k
            for k in up
            if k.endswith((".pi_E_N", ".pi_E_E"))
            or ".pm_ra" in k
            or ".pm_dec" in k
        ]
        if blockers:
            logger.info(
                f"[{self.prefix}] proper motion or parallax already given "
                f"({', '.join(sorted(blockers))}); not seeding from the "
                f"galactic model, which would contradict it."
            )
            return None
        try:
            n_stars = system.star.n_elements
            source_ndx = int(self.source_map[0])
            ra_all = self.config_manager.resolve(
                "star", "ra", shape=(n_stars,)
            )["initval"]
            dec_all = self.config_manager.resolve(
                "star", "dec", shape=(n_stars,)
            )["initval"]
        except Exception as exc:  # pragma: no cover - seeds are optional
            logger.debug(
                f"[{self.prefix}] could not resolve the line of sight for "
                f"the galactic-model proper-motion seeds: {exc!r}"
            )
            return None

        keys = [f"star.{source_ndx}.ra", "star.ra"]
        names = getattr(system.star, "names", None)
        if names:
            keys.append(f"star.{names[source_ndx]}.ra")
        if not any(k in self.config_manager.user_params for k in keys):
            logger.debug(
                f"[{self.prefix}] no user-set RA/Dec; skipping the "
                f"galactic-model proper-motion seeds."
            )
            return None

        # resolve() hands back the value in the parameter's USER unit,
        # which for ra/dec is degrees; the galactic-model helpers take
        # radians.
        return (
            float(np.radians(np.atleast_1d(ra_all)[source_ndx])),
            float(np.radians(np.atleast_1d(dec_all)[source_ndx])),
        )

    def _seed_expected_pm(self, line_of_sight, star_idx, population, dist_pc):
        """Seed one star's pm_ra/pm_dec at the galactic model's prior mean.

        PRECEDENCE_DERIVED_DATA: this is derived from the galactic model
        the same way an RV offset is derived from the data, so it belongs
        in that tier and must yield to anything in params.yaml.

        It ties with the MMEXOFAST seeds (also PRECEDENCE_DERIVED_DATA),
        which is the point.  Both proper-motion components are now pinned,
        so ``mu_rel`` has a magnitude AND a direction, and the engine no
        longer has to invert
        ``mu_rel_mag**2 = mu_ra_rel**2 + mu_dec_rel**2`` -- one equation in
        two unknowns -- by choosing a point on a circle (issue #93).  Where
        that disagrees with the seeded ``t_E``, Condition B rewrites the
        lowest-rank symbol in ``t_E = theta_E / |mu_rel_geo|``, which is
        ``theta_E`` via the lens mass (defaults.yaml, ``PRECEDENCE_DEFAULT``)
        and distance (``PRECEDENCE_MULENS_LENS_DISTANCE``).  So ``t_E``
        keeps its measured value, the proper motion keeps the prior's, and
        the lens mass absorbs the difference -- which is the standard
        microlensing chain (a measured t_E plus an assumed mu_rel implies
        theta_E, hence a mass) and is the quantity a light curve genuinely
        cannot pin down.

        `dist_pc` must match the distance hint seeded for the same star:
        the mean velocity is position-dependent, so a mismatch would seed a
        proper motion for a place the star is not.
        """
        if line_of_sight is None:
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
        chain (see context_dep_names); everything else is generic."""
        if param_name in ("mu_ra_rel_geo", "mu_dec_rel_geo"):
            context_nodes = dict(context_nodes or {})
            if "earth_vperp_e" not in context_nodes:
                vperp_e, vperp_n = self._earth_vperp_en(system)
                context_nodes["earth_vperp_e"] = pt.as_tensor_variable(vperp_e)
                context_nodes["earth_vperp_n"] = pt.as_tensor_variable(vperp_n)
        return super().add_parameter(model, param_name, system, context_nodes)

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
            "rho": getattr(source, "rho", None),
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
        """Per-epoch companion geometry ``(s_t, alpha_t_deg)`` for companion
        0 (VECTOR ELEMENT 1 of the lens component; element 0 is the masked
        primary), or ``None`` when the lens geometry is static.

        The linear mode is DEFINITIONAL in these coordinates (C24):

            s(t)     = s_0     + ds_dt     * (t - t0_par)/DAYS_PER_YEAR
            alpha(t) = alpha_0 + dalpha_dt * (t - t0_par)/DAYS_PER_YEAR

        anchored at t0_par -- the same fiducial epoch the parallax uses
        (5d: one anchor is what makes the two effects composable; Skowron
        Eq. A17).  ``alpha_t`` is returned in DEGREES, the unit both
        magnification backends take; ``dalpha_dt``'s internal unit is
        rad/yr, so the rate converts here alongside alpha itself
        (_alpha_deg).  Skowron's gamma vocabulary maps as
        gamma_par = ds_dt/s_0 and gamma_perp = -dalpha_dt -- the minus is
        C24's rule, and the light curve built from these definitions is
        pinned against MulensModel's linear branch (the reference
        implementation) in tests/test_lens_orbital_motion.py.

        ``times`` may be a tensor or a numpy array (the likelihood's
        concatenated epochs, or a plotter's model grid) -- the series is
        built from the argument, never from ``self.time``, so the plotted
        curve is the curve the likelihood fits.
        """
        lens = system.lens
        om = lens.orbital_motion[0]
        if om is None:
            return None
        dt_yr = (times - self.t0_par[0]) / DAYS_PER_YEAR
        if om == "linear":
            s_t = lens.s.value[1] + lens.ds_dt.value[1] * dt_yr
            alpha_t_deg = (
                self._alpha_deg(system, 0)
                + lens.dalpha_dt.value[1] * _RAD_TO_DEG * dt_yr
            )
            return s_t, alpha_t_deg
        # keplerian: the same physics function the reported s/alpha use
        # (evaluated there at t0_par), here over the epoch vector.  No
        # anchor enters -- alpha(t) = phi_pi - PA_axis(t) is absolute
        # (C15/C20/C24), and s(t) is the projected separation in Einstein
        # units.  Adds NO free parameters (8.6.8 5b).
        j = lens.kep_orbit_idx
        orbit = system.orbit
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
            self.theta_E.value[0],
            system.star.distance.value[int(self.lens_bodies[0][1])],
            self.pi_E_N.value[0],
            self.pi_E_E.value[0],
        )
        return s_t, alpha_t_rad * _RAD_TO_DEG

    def _source_offset_series(self, times, system):
        """Per-epoch xallarap trajectory shift ``(dtau_t, du_t)``, or None
        for a static source (conventions.md C25; review 8.6.9).

        The luminous source's own barycentric sky offset -- its orbit's
        primary track, a1 = a * m_companion / m_total, in Einstein units
        a1/(D_S theta_E) -- anchored at t0_par (the shift VANISHES there,
        5d: same anchor as the parallax, so t_0/u_0 keep their meaning),
        and projected on C9's (tau_hat, beta_hat) exactly where the
        parallax terms enter: parallax is the OBSERVER's offset, xallarap
        is the SOURCE's, same slot, same sign discipline (C8/C9/C25).

        Built from the TIMES ARGUMENT, so the plotters' model grids carry
        the same moving source the likelihood fits.
        """
        if self.xal_orbit_idx is None:
            return None
        j = self.xal_orbit_idx
        orbit = system.orbit
        a1 = (
            orbit.a.value[j]
            * orbit.m_companion.value[j]
            / orbit.m_total.value[j]
        )
        d_s = system.star.distance.value[int(self.source_bodies[0][1])]
        args = (
            orbit.tp.value[j],
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
          carries a NaN -- and that raw variable's own N(0, 1) prior term
          already makes the total logp NaN, so the proposal is rejected
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

    def _get_binary_mm_params(self, system, index=0):
        """Params for a binary lens.  ``index`` is the SOURCE slot; the lens
        bodies are shared by all sources.

        The derived chain (theta_E, t_E, rho, pi_E) is already referenced
        to the TOTAL lens mass via mlens_total (C13), so the safe
        single-source params pass straight through -- only the companion
        geometry (s, q, alpha) is added here, indexed by COMPANION
        (binary = companion 0 = VECTOR ELEMENT 1 of the lens component;
        element 0 is the masked primary).  q is the q Parameter
        (physics.calc_q), the same ratio every other consumer reads, not a
        local recomputation from the mass nodes."""
        s = self._get_safe_mm_params(system, index)
        lens = system.lens
        return {
            **s,
            "s": lens.s.value[1],
            "q": clip_q(lens.q.value[1]),
            "alpha": self._alpha_deg(system, 0),
        }

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
        source_ndx = int(self.source_bodies[index][1])
        ra = system.star.ra.value[source_ndx]
        dec = system.star.dec.value[source_ndx]

        delta_e, delta_n = observer_sky_offset(obs_pos, ra, dec, xp=pt)

        p = self._get_safe_mm_params(system, index)
        # MulensModel convention: delta_tau = -delta_N*pi_E_N -
        # delta_E*pi_E_E (negative on both N and E, matching Skowron+2011
        # via MulensModel's sign choice).  MMEXOFAST calls MulensModel, so
        # published pi_E values are calibrated to this convention.
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

        Binary/finite-source lenses use the MulensModel Op, which is not
        differentiable.  Gradient-based samplers (NUTS, numpyro, blackjax)
        will produce invalid results; PTDE is required.  The asynchronous
        dispatch loop (ptde_async) is recommended: near-caustic evaluations
        concentrate in the hot rungs and stall the synchronous sampler's
        every step behind the slowest proposal (samplers/ptde_async.py).

        PSPL lenses use a symbolic PyTensor formula and are
        NUTS-compatible, so no constraints are returned.
        """
        if self.uses_op():
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

    def _resolve_quadratic_ld(self, u2, effective_bandpass):
        """Can the selected backend honour the band's second LD coefficient?

        Returns True to put u2 in the param vector, False to drop it -- and
        when it drops it, says so ONCE, because a silently ignored u2 is
        the exact defect this plumbing was added to fix.  A dropped u2 is
        not merely a wrong profile: on a band whose limb darkening only
        microlensing reads, the magnification is a function of u1 alone, so
        one combination of the sampled Kipping pair (q1, q2) becomes
        likelihood-free -- sampled, reported, and constrained by nothing.

        WHO CAN DO WHAT:

        * VBMicrolensing carries LDquadratic for the binary/N-lens solvers
          (BinaryMag2/MultiMag2) and for the point lens (ESPLMag2).  So
          `backend: vbm_direct` -- the default -- can honour u2 everywhere.
        * MulensModel cannot, anywhere: `set_limb_coeff_u` takes one
          coefficient and the Yoo04 B0/B1 factorization it uses for a
          finite point source is a linear-law formalism.  `backend:
          mulensmodel` is the A/B reference, so it keeps being linear and
          says so.

        WHY THE SINGLE-LENS DEFAULT IS NOT FLIPPED WHOLESALE.  A
        finite-source point lens goes to MulensModel today, and VBM's
        ESPLMag2 disagrees with Yoo04 by up to ~5 mmag (1.7 mmag rms) in
        the deep finite-source regime u_0 << rho -- Yoo04's table
        interpolation, measured at rho = 0.001-0.05.  Routing every FSPL
        fit to VBM would therefore move existing answers by more than most
        of these light curves' error bars, silently, as a side effect of an
        unrelated fix.  So the switch is keyed on u2 actually being in
        play: a `ld_law: linear` band keeps MulensModel and is
        bit-identical to before, and only the configuration that was
        already WRONG changes backend.  A user who wants VBM's ESPL for its
        own sake still has no way to ask for it; that is a deliberately
        separate decision (see notes).
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
                "vbm_direct' to honour u2, or declare 'ld_law: linear' on "
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
        backends can honour it and what happens when the selected one
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

        source = system.source
        lens = system.lens
        source_ndx = int(self.source_bodies[index][1])
        ra_deg, dec_deg = self._frozen_op_coords_deg(system, source_ndx)
        coords = f"{ra_deg}d {dec_deg}d"

        use_rho = self.finite_source

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
            # ESPL through VBM: the only backend here that carries a
            # quadratic limb-darkening law for a point lens, and the one a
            # finite-source single lens with XALLARAP routes through (the
            # MulensModel single-lens Op has no slot for a per-epoch
            # trajectory shift).  Without xallarap it is reached ONLY when
            # u2 is genuinely in play (_resolve_quadratic_ld), so a
            # linear-band fit keeps MulensModel and stays bit-identical --
            # see that method.
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
                # Companion slot j = lens vector element j+1 (masked
                # primary).
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
            if lens.orbital_motion[0] == "keplerian":
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
                # companion's lens vector elements (element 1).
                param_list.extend(
                    [
                        lens.ds_dt.value[1],
                        lens.dalpha_dt.value[1] * _RAD_TO_DEG,
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
        lenses.

        Historically this computed hexadecapole-vs-VBM brackets on a time
        grid, but MulensModel implements binary-lens hexadecapole as 13
        python-level VBM.BinaryMag0 calls per epoch while VBM's BinaryMag2
        runs the equivalent quadrupole safety test internally in C++ and
        short-circuits to point-source when safe.  Measured on DC2018_128:
        hexadecapole 32.9 ms vs VBM-everywhere 7.7 ms per 870-point call,
        at equal or better accuracy -- so the bracket machinery optimized
        for the wrong cost model and was removed (see hpc_optimization.txt,
        P1).

        Single-lens events are left untouched: 'auto_vbbl' is resolved
        inside the PSPL model builder (point_source + finite-source
        window), and the VBM/VBBL methods emitted here are
        binary-lens-only.

        Only the mulensmodel backend consumes the resulting method list;
        the default vbm_direct backend always calls BinaryMag2/MultiMag2.
        """
        if self.mag_method[index] != "auto_vbbl":
            return
        if len(self.lens_bodies) < 2:
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
