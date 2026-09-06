"""The microlensing EVENT component (8.6.17 stage 1).

One instance, one event: the encounter between one co-moving lens system and
one co-moving source system (ruling R1).  Everything that is a property of
the ENCOUNTER -- the relative parallax, proper motion, Einstein radius and
crossing time, the parallax vector, the total lens mass -- lives here with
one element each; the per-body geometry lives on the `lens` component (one
entry per lens body, primary first) and the per-source trajectory offsets on
the `source` component (one entry per source body).

Stage-1 note: the magnification dispatcher, the event potentials and the
attributes MulensInstrument reads still live on `Lens` (they move here in
stage 2 with the consumers); this component owns the event PARAMETERS, the
event-level config keys and the seeding hints.
"""

import logging

import numpy as np
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.config import (
    PRECEDENCE_DERIVED_DATA,
    PRECEDENCE_DERIVED_MIXED,
    PRECEDENCE_MULENS_LENS_DISTANCE,
    PRECEDENCE_MULENS_SOURCE_DISTANCE,
)
from exozippy.constants import DAYS_PER_YEAR
from exozippy.skyframe import sky_basis

from ..galacticmodel.physics import expected_proper_motion
from . import mmexofast_support
from .bodies import body_entries, validate_event_config

logger = logging.getLogger(__name__)


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

        self.finite_source = bool(self.config[0].get("finite_source", False))

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
        body is always a star, which _validate_bodies enforces; see the
        pre-split Lens.build_maps note for the history.  source_map is the
        PRIMARY source's star: under R1 the source SYSTEM's barycentric
        kinematics are represented by body 0 (design 11.3).
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
        per-seed hint set for multi-seed sampling (P4).  See the pre-split
        Lens._load_mmexofast_seeds; only the config home moved."""
        mmx_file = self.config[0].get("mmexofast") if self.config else None
        # Only an explicit file path is handled here. "auto" / absent-key
        # auto-initialization is owned by MulensInstrument (stage 1), which
        # pushes the seed hints itself before this method ever runs; False
        # opts out entirely.
        if not isinstance(mmx_file, str) or mmx_file == "auto":
            return
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
        apply.  Ported from the pre-split Lens (its docstring carries the
        measured chi2 gates; tests/test_seed_quality.py pins the numbers).
        The multi-source gate is KEPT at stage 1 -- its retirement is a
        deliberate stage-4 change paired with re-pinning test_seed_quality
        (design section 7).
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
        if self.n_sources > 1:
            logger.info(
                f"[{self.prefix}] {self.n_sources} sources; not seeding "
                f"proper motions from the galactic model (one mean would "
                f"tie their mu_rel together)."
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
        Ported verbatim from the pre-split Lens._seed_expected_pm; see that
        history (issue #93, Condition B, PRECEDENCE_DERIVED_DATA tie)."""
        if line_of_sight is None:
            return
        ra_rad, dec_rad = line_of_sight
        try:
            pm_ra, pm_dec, _rv = expected_proper_motion(
                ra_rad, dec_rad, dist_pc, population
            )
        except Exception as exc:
            # WARNING, not debug: this silently disabled the whole feature
            # once already (degrees were passed where radians were wanted).
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
        in AU/yr.  The Gould (2004) mu_helio -> mu_geo conversion constant;
        ported verbatim from the pre-split Lens._earth_vperp_en."""
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
        """Stage 7: nothing yet.  The event potentials (event rate,
        singularity guards, source-behind-lens) stay on Lens.build_likelihood
        at stage 1 and move here with the consumers in stage 2, so the
        stage-0 fixtures' potential NAMES survive this stage unchanged."""
        pass

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
