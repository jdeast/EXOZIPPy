import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.config import (
    RANK_DEFAULT,
    RANK_DERIVED_DATA,
    RANK_DERIVED_MIXED,
    RANK_MULENS_LENS_DISTANCE,
    RANK_MULENS_SOURCE_DISTANCE,
)
from exozippy.corner_utils import (
    collect_parameter_corner_samples,
    save_corner_plot,
)
from exozippy.outputs.prose import get_collector
from exozippy.potentials import soft_lower_bound
from exozippy.skyframe import observer_sky_offset, sky_basis

from ..galacticmodel.physics import expected_proper_motion
from . import mmexofast_support
from .op import BinaryLensMagOp, MulensMagOp, VBMDirectMagOp
from .physics import (
    _MM_NAN_ADVICE,
    _Q_NAN_ADVICE,
    MU_REL_FLOOR,
    Q_MAX,
    Q_MIN,
    T_E_FLOOR,
    THETA_E_FLOOR,
    THETA_E_LENSING_MIN,
    U_0_FLOOR,
    apply_u_0_floor,
    clip_q,
    floor_u_0_value,
)

logger = logging.getLogger(__name__)

# alpha is stored in radians (lens.alpha's internal_unit) and consumed in
# degrees by both magnification backends.
_RAD_TO_DEG = 180.0 / np.pi


def _parse_body_ref(ref):
    """Parse 'star.0' → ('star', 0), 'planet.1' → ('planet', 1)."""
    parts = str(ref).split(".")
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(
            f"Invalid body reference '{ref}': expected '<component>.<index>', "
            f"e.g. 'star.0' or 'planet.0'."
        )
    return (parts[0], int(parts[1]))


class Lens(Component):
    """Microlensing lens component.

    Supports N sources and up to 2 lens bodies (NSNL; the MulensModel backend
    caps the lens side at binary for now).  Bodies are specified in the YAML
    config as:
        lenses:  ["star.0", "planet.0"]   # 2-body binary
        sources: ["star.1", "star.2"]     # binary source (2S)

    The FIRST entry of ``lenses`` is the primary and must be a ``star``:
    the lens maps carry only an index and the primary-side physics resolves
    through star.mass/star.distance/star.pm_*, so a non-star primary is
    rejected in _validate_bodies rather than silently modeling the star at
    that index.  Companions may be ``planet`` or ``star``.  A
    planetary-mass lens is modeled as a ``star`` block with a low logmass --
    and, if it is a free-floating planet, with ``mass_function: ffp`` so that
    it draws the FFP mass function rather than the stellar IMF.

    Each source follows its own trajectory: t_0, u_0, rho and the derived
    chain (t_E, theta_E, pi_rel, pi_E_*, mu_*) are vectors with one element
    per source, sharing the lens-side parameters (masses, s, alpha).  In the
    params file, address element j either by slot index (lens.1.t_0) or by
    the source star's instance name (lens.SourceB.t_0).

    Backward-compatible shorthand (single-star PSPL):
        lens_ndx:   0
        source_ndx: 1
    """

    # Microlensing is THE topology where a solution the posterior abandons
    # still has to be reported.  Its degeneracies are structural, not
    # accidental: the u_0 sign flip (ob140939's four Yee+2015 basins), the
    # close/wide s <-> 1/s pair, and the ecliptic/jerk-parallax families all
    # give distinct basins that fit the light curve nearly as well while
    # differing by a factor of a few in lens mass and distance.  Published
    # solutions routinely quote two or four of them.  A T=1 posterior keeps
    # only the winner, so without hot-rung draws the alternatives leave no
    # record of having been examined at all -- see
    # samplers._common.resolve_store_hot_chains.
    expects_suppressed_modes = True

    # Deps satisfied by context-node injection in add_parameter (constants,
    # not manifest parameters); graph.py skips them when ordering the build.
    context_dep_names = frozenset({"earth_vperp_e", "earth_vperp_n"})

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Lens Parameters"

        # One event at a time: a single (t_0, u_0, t_E, ...) geometry. Multiple
        # lenses, sources, or instruments all belong to that one event.
        if self.n_elements > 1:
            raise ValueError(
                "Only one lensing event may be modeled at a time. Define a single "
                "lens block and list all bodies in 'lenses'/'sources', primary "
                "first (e.g. lenses: ['star.0', 'planet.0', 'planet.1'] -- the "
                "primary must be a star; companions may be planets or stars)."
            )

        # Parse lens / source body lists per event
        self.lens_bodies = []  # list of lists of (comp_type, ndx) per event
        self.source_bodies = []  # list of lists of (comp_type, ndx) per event

        for c in self.config:
            if "lenses" in c:
                lb = [_parse_body_ref(r) for r in c["lenses"]]
            else:
                lb = [("star", int(c.get("lens_ndx", 0)))]

            if "sources" in c:
                sb = [_parse_body_ref(r) for r in c["sources"]]
            else:
                sb = [("star", int(c.get("source_ndx", 1)))]

            self.lens_bodies.append(lb)
            self.source_bodies.append(sb)

        self.n_lens_bodies = [len(b) for b in self.lens_bodies]
        self.n_source_bodies = [len(b) for b in self.source_bodies]

        # Companions: every lens body beyond the primary. Each carries its own
        # separation s and trajectory angle alpha; mass ratios come from the
        # bodies' masses.
        self.n_companions = self.n_lens_bodies[0] - 1

        # Sources: single event ⇒ one flat list of source bodies; per-source
        # parameters (t_0, u_0, rho, ...) are vectors of this length.
        self.n_sources = self.n_source_bodies[0]

        # Translate lens.<SourceStarName>.<param> user keys to the canonical
        # slot-index form lens.<j>.<param> so resolve() and the relaxation
        # engine see one naming scheme.  Must happen before any stage-1 code
        # (e.g. MulensInstrument.load_data) reads user_params.
        self._rewrite_source_param_keys(config_manager)

        # Convenience maps: primary lens and source (index 0 of each list)
        self.finite_source = [
            c.get("finite_source", False) for c in self.config
        ]
        self.t0_par = [
            self._resolve_t0_par(i, c, config_manager)
            for i, c in enumerate(self.config)
        ]

        # One magnification method per source (each source has its own
        # trajectory and caustic-crossing times); all sources start from the
        # event-level config value, and resolve_auto_vbbl refines each slot.
        event_method = self.config[0].get(
            "mag_method",
            (
                "auto_vbbl"
                if (self.finite_source[0] or self.n_lens_bodies[0] > 1)
                else "point_source"
            ),
        )
        self.mag_method = [event_method] * self.n_sources

        # use_op: force the MulensModel Op even for point-source PSPL.
        # Default False for PSPL (symbolic is NUTS-friendly); True forces the Op
        # (useful for testing or when MulensModel's parallax handling is needed).
        self.use_op = [c.get("use_op", False) for c in self.config]

        # backend: which magnification engine the multi-lens Op path uses.
        #   vbm_direct  — call VBMicrolensing directly (default; ~5x faster,
        #                 supports 2+ lens bodies)
        #   mulensmodel — rebuild an mm.Model per call (A/B reference; binary only)
        self.backend = self.config[0].get("backend", "vbm_direct")
        if self.backend not in ("vbm_direct", "mulensmodel"):
            raise ValueError(
                f"lens.backend must be 'vbm_direct' or 'mulensmodel', "
                f"got '{self.backend}'."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rewrite_source_param_keys(self, config_manager):
        """Rewrite lens.<SourceStarName>.<param> → lens.<j>.<param>.

        The generic standardize_param_names pass only knows the lens event's
        own instance name; addressing a per-source element by the source
        star's name (lens.SourceB.t_0) is lens-specific knowledge, so the
        translation lives here.  Keys already in index or event-name form are
        untouched (event-name form was standardized to lens.0.* and refers to
        source slot 0).
        """
        system_config = getattr(config_manager, "system_config", None) or {}
        slot_by_name = {}
        for j, (comp_type, ndx) in enumerate(self.source_bodies[0]):
            entries = system_config.get(comp_type, [])
            if ndx < len(entries) and isinstance(entries[ndx], dict):
                name = entries[ndx].get("name")
                if name is not None:
                    slot_by_name[str(name)] = j

        up = config_manager.user_params
        for key in list(up.keys()):
            parts = key.split(".")
            if (
                len(parts) == 3
                and parts[0] == self.prefix
                and parts[1] in slot_by_name
            ):
                new_key = f"{self.prefix}.{slot_by_name[parts[1]]}.{parts[2]}"
                if new_key in up:
                    logger.warning(
                        f"Parameter '{key}' duplicates '{new_key}'; keeping '{new_key}'."
                    )
                    del up[key]
                else:
                    up[new_key] = up.pop(key)

    def _translate_s_bounds_to_log_s(self):
        """Rewrite user-supplied lens.<j>.s lower/upper bounds onto log_s.

        s is now a derived parameter (s = 10**log_s); the sampling bounds live
        on log_s.  A user who constrains lens.s expects the sampler to respect
        it, so translate lower/upper -> log10(bound) onto the log_s entry and
        drop them from the s entry (an initval on s is left in place -- the
        relaxation engine translates it through the s <-> log_s relation).
        Keys are already standardized to lens.<j>.<param> form.
        """
        up = self.config_manager.user_params
        for j in range(self.n_companions):
            entry = up.get(f"lens.{j}.s")
            if not isinstance(entry, dict):
                continue
            if "lower" not in entry and "upper" not in entry:
                continue
            log_key = f"lens.{j}.log_s"
            log_entry = up.get(log_key)
            if not isinstance(log_entry, dict):
                log_entry = {}
            for bound in ("lower", "upper"):
                if bound not in entry:
                    continue
                val = float(entry.pop(bound))
                if val <= 0.0:
                    raise ValueError(
                        f"lens.{j}.s {bound} bound must be positive (s > 0); "
                        f"got {val}."
                    )
                log_entry[bound] = float(np.log10(val))
            up[log_key] = log_entry
            logger.info(
                f"Translated lens.{j}.s bound(s) to log_s "
                f"(log10): {log_key} = "
                f"{{{', '.join(f'{b}: {log_entry[b]:.4f}' for b in ('lower', 'upper') if b in log_entry)}}}."
            )

    def _source_instance_names(self):
        """Display names for per-source vector elements (source star names)."""
        system_config = (
            getattr(self.config_manager, "system_config", None) or {}
        )
        names = []
        for comp_type, ndx in self.source_bodies[0]:
            entries = system_config.get(comp_type, [])
            if (
                ndx < len(entries)
                and isinstance(entries[ndx], dict)
                and entries[ndx].get("name")
            ):
                names.append(str(entries[ndx]["name"]))
            else:
                names.append(f"{comp_type}{ndx}")
        return names

    @staticmethod
    def _resolve_t0_par(i, c, config_manager):
        if "t0_par" in c:
            return float(c["t0_par"])
        entry = config_manager.user_params.get(f"lens.{i}.t_0")
        if isinstance(entry, dict):
            val = entry.get("initval")
        else:
            val = entry
        # List-valued initval (P4 multi-seed sampling): t0_par is just a
        # numeric reference epoch, not a per-seed value, so use seed 0 -- the
        # same convention the relaxation engine uses for bounds/scales.
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
                    "params.yaml seeding the lens parameters."
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
                "key": "lens_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": "Index or name of the lens star. Default 0.",
            },
            {
                "key": "source_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": "Index or name of the source star. Default 1.",
            },
            {
                "key": "star_constrains_rho",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Default True: rho is the identity theta_star/theta_E "
                    "-- the stellar model (SED, evolutionary models, "
                    "relations, or priors) constrains rho alongside the "
                    "light curve's finite-source measurement. Set False "
                    "to sever the tie: rho is sampled directly (as "
                    "log_rho) and the stellar prediction is reported as "
                    "rho_pred, so the pull between the two is visible "
                    "instead of silently arbitrated. Requires "
                    "finite_source. Same vocabulary as the planet "
                    "component's beam_constrains_mass and the "
                    "instrument's sed_constrains_blend. A tie is a physics LINK, not a one-way assignment: information flows toward whichever side is less constrained elsewhere (components.md, 'Config flag vocabulary')."
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
                    "Magnification engine for the multi-lens Op path. "
                    "Default 'vbm_direct'."
                ),
            },
            {
                "key": "finite_source",
                "kind": "option",
                "accepts": [True, False],
                "required": False,
                "doc": "Model finite-source effects. Default false.",
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

    def _primary_lens(self, event_idx):
        """Return (comp_type, star_ndx) for the primary lens of event i."""
        return self.lens_bodies[event_idx][0]

    def _primary_source(self, event_idx):
        """Return (comp_type, star_ndx) for the primary source of event i."""
        return self.source_bodies[event_idx][0]

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
        """Fail at registration time if a body reference points to a component
        or instance that does not exist (instead of an AttributeError deep in
        the model build), if the PRIMARY lens body is not a star, or if any
        SOURCE body is not a star."""
        for i in range(self.n_elements):
            for role, bodies in (
                ("lens", self.lens_bodies[i]),
                ("source", self.source_bodies[i]),
            ):
                for comp_type, ndx in bodies:
                    comp = getattr(system, comp_type, None)
                    if comp is None:
                        raise ValueError(
                            f"lens.{i}: {role} body '{comp_type}.{ndx}' refers to "
                            f"component '{comp_type}', but no '{comp_type}' block "
                            f"exists in the config."
                        )
                    if ndx >= comp.n_elements:
                        raise ValueError(
                            f"lens.{i}: {role} body '{comp_type}.{ndx}' is out of "
                            f"range: only {comp.n_elements} '{comp_type}' "
                            f"instance(s) are configured."
                        )

            # The primary lens body must be a star.  build_maps stores only
            # the INDEX (lens_map / primary_lens_map), and every primary-side
            # dependency in defaults.yaml is hard-coded to the star component
            # -- star.mass[lens_map], star.distance[lens_map],
            # star.pm_ra[lens_map], star.pm_dec[lens_map] -- as is
            # build_likelihood's d_l.  A 'planet.0' primary therefore silently
            # models star.0 instead: measured on examples/ob08092, a config
            # with lenses: ["planet.0"] builds a theta_E bit-identical to
            # lenses: ["star.0"], responds to that star's mass, and is
            # completely insensitive to the planet's -- a fit that completes
            # and reports a lens mass which never touched the photometry.
            # Companions ARE type-aware (their mass deps carry the component
            # type), so only this slot is restricted.
            p_type, p_ndx = self._primary_lens(i)
            if p_type != "star":
                raise ValueError(
                    f"lens.{i}: the primary (first) lens body is "
                    f"'{p_type}.{p_ndx}', but it must be a star.  The lens "
                    f"maps carry only an index, and the lens-side physics "
                    f"resolves the primary through star.mass / star.distance "
                    f"/ star.pm_ra / star.pm_dec, so a non-star primary would "
                    f"silently model star.{p_ndx} instead of "
                    f"'{p_type}.{p_ndx}' and report a lens mass that never "
                    f"entered the likelihood.  Planet COMPANIONS are "
                    f"supported -- put the star first, e.g. "
                    f"lenses: ['star.0', '{p_type}.{p_ndx}'].  To model a "
                    f"very low-mass (even planetary-mass) lens, declare it as "
                    f"a 'star' block with a low star.<name>.logmass instead; "
                    f"logmass reaches -9 dex (1e-9 solMass).  For a "
                    f"FREE-FLOATING planet that is only half the recipe: give "
                    f"that star block 'mass_function: ffp' as well, or it "
                    f"draws the stellar IMF and is penalized for having the "
                    f"mass you told it to have."
                )

            # EVERY source body must be a star -- unlike the lens side there
            # is no companion position to spare.  source_map is index-only
            # exactly like lens_map, and the whole source-side chain resolves
            # through the star component: star.distance[source_map],
            # star.pm_ra/pm_dec[source_map], star.radius[source_map], and
            # get_magnification's star.ra/dec[source_ndx].  The multi-source
            # (2S) case does not change this: each source body is an
            # independently monitored luminous star with its own trajectory
            # and flux ratio, so every slot is star-only.
            for s_type, s_idx in self.source_bodies[i]:
                if s_type != "star":
                    raise ValueError(
                        f"lens.{i}: source body '{s_type}.{s_idx}' must be a "
                        f"star -- a microlensing source is the background "
                        f"star being monitored for magnification, and a "
                        f"planet is not a self-luminous point source at "
                        f"bulge distances, so a non-star source is "
                        f"physically meaningless rather than merely "
                        f"unimplemented.  source_map carries only an index "
                        f"and the source-side physics resolves through "
                        f"star.distance / star.pm_ra / star.pm_dec / "
                        f"star.radius / star.ra / star.dec, so this would "
                        f"silently model star.{s_idx} instead of "
                        f"'{s_type}.{s_idx}'.  A genuinely faint source (a "
                        f"brown dwarf, say) is a 'star' block with a low "
                        f"star.<name>.logmass."
                    )

            # A body cannot lens itself.  pi_rel = 1000/d_L - 1000/d_S is
            # then identically 0, so theta_E collapses onto its floor and
            # the likelihood is NaN at the very first evaluation -- which
            # today surfaces as a baffling sampler-initialization failure
            # far from the config line that caused it.  Both spellings are
            # covered for free: the legacy lens_ndx/source_ndx keys are
            # normalized into lens_bodies/source_bodies in __init__, so
            # comparing those two lists catches `lens_ndx: 0, source_ndx: 0`
            # as well as an explicit overlap between the lists (including a
            # body repeated across a multi-source 2S list).
            shared = [
                b for b in self.lens_bodies[i] if b in self.source_bodies[i]
            ]
            if shared:
                shared_txt = ", ".join(f"'{t}.{n}'" for t, n in shared)
                raise ValueError(
                    f"lens.{i}: {shared_txt} is listed as BOTH a lens body "
                    f"and a source body.  A lens and its source must be "
                    f"distinct objects at different distances: with the same "
                    f"body on both sides, pi_rel = 1000/d_L - 1000/d_S is "
                    f"identically 0, so theta_E is 0 and the likelihood is "
                    f"NaN from the first evaluation.  Give the lens and the "
                    f"source separate entries (via 'lenses:'/'sources:', or "
                    f"distinct 'lens_ndx:'/'source_ndx:' values)."
                )

    # ------------------------------------------------------------------
    # Lifecycle stages
    # ------------------------------------------------------------------

    def build_maps(self):
        """Stage 2: Build integer index arrays for lens and source bodies.

        source_map has one entry per SOURCE BODY (not per event): it drives the
        shapes of the per-source parameter chain (pi_rel, t_E, rho, ...) via the
        star.<param>[source_map] dependency slices.

        lens_map carries TWO conceptually different roles that happen to
        share one index:

          1. the LENSING MASS -- star.mass[lens_map] feeds theta_E;
          2. the KINEMATIC HOST -- star.distance[lens_map] and
             star.pm_ra/pm_dec[lens_map] feed pi_rel and mu_rel.

        They coincide only because the primary lens body is always a star,
        which _validate_bodies now enforces.  The conflation is exactly what
        let the silent planet-primary bug through: a planet has a mass but
        no distance or proper motion of its own, so a planet primary
        resolved role 1 to the planet (had the deps been typed) and role 2
        to whatever star sat at the same index.  Splitting the two would
        mean inventing a "kinematic host star" for a body that by
        definition has no host -- which is why planet-as-lens was abandoned
        in favor of declaring a low-mass lens as a star.  Under the guards
        the roles can never diverge, so this stays one index; the note is
        here so the next reader does not have to rediscover why.
        """
        _, l_ndxs = zip(
            *[self._primary_lens(i) for i in range(self.n_elements)]
        )
        self.lens_map = np.array(l_ndxs, dtype=int)
        self.source_map = np.array(
            [ndx for (_, ndx) in self.source_bodies[0]], dtype=int
        )

        if self.n_companions >= 1:
            # Scalar maps (length-1) so the bracket-slice dep yields a scalar
            # mass rather than a full-component mass array.  One map per
            # companion: companions may live in different component types
            # (star vs planet), so each mass needs its own bracket dep.
            _, p_ndx = self.lens_bodies[0][0]
            self.primary_lens_map = np.array([p_ndx], dtype=int)
            for j, (_, c_ndx) in enumerate(self.lens_bodies[0][1:]):
                setattr(
                    self,
                    f"companion{j}_mass_map",
                    np.array([c_ndx], dtype=int),
                )

    def _load_mmexofast_seeds(self):
        """Read an optional MMEXOFAST solutions file and push each fit as a
        per-seed hint set for multi-seed sampling (P4).

        MMEXOFAST emits multiple lightly-optimized solutions spanning the
        standard microlensing degeneracies. Each fit's observable-space values
        (t_0, u_0, t_E, s, q, alpha, rho) are seeded into the relaxation engine,
        which back-solves the physical parameters (distances/masses/PMs) exactly
        as a user typing them into params.yaml K times would. Enabled by a
        `mmexofast: <file>` key on the lens config block (path relative to the
        run cwd, same as the light-curve `file:` key).

        The translation itself (seed sets, scale hints, jd_offset handling,
        alpha/log_s conventions) lives in mmexofast_support.push_seed_hints,
        shared with MulensInstrument's stage-1a auto-initialization (which
        also applies the JSON's bad-data mask and error factors -- masks must
        exist before the photometry is read, which is why the instrument owns
        that half).
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
            want_rho=any(self.finite_source),
            is_binary=self.n_companions >= 1,
            source=mmx_file,
        )

    def register_parameters(self, system):
        """Stage 3: Declare the manifest."""
        self._validate_bodies(system)

        # s is derived from the sampled log_s; move any user s bounds onto log_s
        # before the manifest/relaxation engine run.
        self._translate_s_bounds_to_log_s()

        # Optional multi-seed sampling from a MMEXOFAST solutions file (P4).
        self._load_mmexofast_seeds()

        # Per-source vector parameters: one element per source body.  Elements
        # are displayed and addressed by the source star's instance name
        # (lens.SourceB.t_0) or slot index (lens.1.t_0).
        src_shape = (self.n_sources,)
        src_names = (
            self._source_instance_names() if self.n_sources > 1 else None
        )

        def per_source(expr_key=None):
            entry = {"shape": src_shape}
            if expr_key is not None:
                entry["expr_key"] = expr_key
            if src_names is not None:
                entry["names"] = src_names
            return entry

        self.manifest = {
            "t_0": per_source(),
            "u_0": per_source(),
            "pi_rel": per_source("default"),
            "theta_E": per_source("default"),
            "mu_ra_rel": per_source("default"),
            "mu_dec_rel": per_source("default"),
            "mu_rel_mag": per_source("default"),
            "mu_ra_rel_geo": per_source("default"),
            "mu_dec_rel_geo": per_source("default"),
            "mu_rel_geo_mag": per_source("default"),
            "t_E": per_source("default"),
            "pi_E_N": per_source("default"),
            "pi_E_E": per_source("default"),
        }

        # The geocentric-frame caveat lives in ONE shared table note (the
        # note_marks dedup collapses identical texts to one letter) rather
        # than in every description -- the 25-character description budget
        # is what sets the table's column width.
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

        # Companion geometry: one (s, alpha) pair per lens body beyond the
        # primary. The shape override sizes these by companion count rather
        # than by component element count.
        if self.n_companions >= 1:
            companion_shape = (self.n_companions,)
            # log_s is the sampled coordinate; s = 10**log_s is derived (the
            # close/wide degeneracy is then an exact reflection log_s -> -log_s).
            self.manifest["log_s"] = {"shape": companion_shape}
            self.manifest["s"] = {
                "expr_key": "default",
                "shape": companion_shape,
            }
            self.manifest["xalpha"] = {"shape": companion_shape}
            self.manifest["yalpha"] = {"shape": companion_shape}
            # alpha derived from xalpha/yalpha via arctan2; internal unit = rad, display = deg
            self.manifest["alpha"] = {
                "expr_key": "default",
                "shape": companion_shape,
            }
            # q_j = M_companion_j / M_primary; companion component types vary
            # by config, hence one scalar bracket dep per companion.
            companion_mass_deps = [
                f"{c_type}.mass[companion{j}_mass_map]"
                for j, (c_type, _) in enumerate(self.lens_bodies[0][1:])
            ]
            self.manifest["q"] = {
                "expr_key": "default",
                "shape": companion_shape,
                "deps": companion_mass_deps + ["star.mass[primary_lens_map]"],
            }
            # Multi-lens convention: theta_E (and hence t_E, rho, pi_E) is
            # referenced to the TOTAL lens mass, matching the published
            # parameterization.  mlens_total sums the body masses and replaces
            # the primary mass in the theta_E dependency chain.
            self.manifest["mlens_total"] = {
                "expr_key": "default",
                "shape": (1,),
                "deps": ["star.mass[primary_lens_map]"] + companion_mass_deps,
            }
            theta_entry = dict(self.manifest["theta_E"])
            theta_entry["deps"] = ["mlens_total", "pi_rel"]
            self.manifest["theta_E"] = theta_entry

        if self.n_companions >= 2:
            # The symbolic relaxation engine only knows the binary mass-sum
            # and q relations (see symbolic_physics.get_symbol_map), so for
            # 3+ lens bodies the mlens_total and per-slot q initvals are
            # seeded from the per-body mass initvals instead -- body masses
            # (or logmass) must be supplied in the params file; a user q
            # cannot back-propagate to a companion mass here.
            # RANK_DERIVED_MIXED: overrides defaults, yields to explicit
            # user values.
            body_masses = [
                self._mass_initval(c_type, c_ndx)
                for c_type, c_ndx in self.lens_bodies[0]
            ]
            # Loud, once, at config time, because the alternative -- a start
            # that quietly comes from nowhere -- is what review 1.6.5 traced
            # (and what 2.6.6 asks be said out loud until the relations are
            # generalized).  A WARNING and not an INFO: for a 2-body lens the
            # engine derives all of this from ANY of the masses, q or the
            # trajectory, so a user who has never had to supply body masses
            # gets no other signal that a third body changes the rules.
            user_q = [
                f"lens.{j}.q"
                for j in range(1, self.n_companions)
                if self.config_manager.user_params.get(f"lens.{j}.q")
                is not None
            ]
            if user_q:
                logger.warning(
                    f"{self.prefix}: {', '.join(user_q)} sets the START of a "
                    "derived mass ratio but CANNOT set the companion mass it "
                    "is computed from -- the relaxation engine's q relation "
                    "covers companion slot 0 only (see "
                    "mulensing/symbolic_physics.py).  The fit will run at the "
                    "masses, not at the q you typed.  Supply "
                    "<component>.<body>.mass (or logmass) for every lens body "
                    "instead."
                )
            if any(m is None for m in body_masses):
                missing = [
                    f"{ct}.{cn}"
                    for (ct, cn), m in zip(self.lens_bodies[0], body_masses)
                    if m is None
                ]
                logger.warning(
                    f"{self.prefix}: no mass initval for lens body/bodies "
                    f"{missing}, so lens.0.mlens_total and the per-companion "
                    "q starts fall back to defaults.  A lens with 3+ bodies "
                    "REQUIRES explicit body masses: the engine's mass-sum and "
                    "q relations are binary-only, so nothing else can supply "
                    "them (review 2.6.6).  Add mass (or logmass) initvals for "
                    f"{missing} to the params file."
                )
            else:
                self.config_manager.add_hint(
                    "lens.0.mlens_total",
                    float(sum(body_masses)),
                    rank=RANK_DERIVED_MIXED,
                )
                for j, m_c in enumerate(body_masses[1:]):
                    q_j = m_c / body_masses[0]
                    self.config_manager.add_hint(
                        f"lens.{j}.q", q_j, rank=RANK_DERIVED_MIXED
                    )
                    self.config_manager.add_scale_hint(
                        f"lens.{j}.q", 0.1 * q_j
                    )

            # The s <-> log_s relation is mapped only for companion slot 0, so
            # for 3+ body lenses seed the remaining companions' log_s from any
            # user s initval (same fallback rationale as q/mlens_total above).
            up = self.config_manager.user_params
            for j in range(1, self.n_companions):
                entry = up.get(f"lens.{j}.s")
                s_val = (
                    entry.get("initval") if isinstance(entry, dict) else entry
                )
                if s_val is None or float(s_val) <= 0.0:
                    continue
                self.config_manager.add_hint(
                    f"lens.{j}.log_s",
                    float(np.log10(float(s_val))),
                    rank=RANK_DERIVED_MIXED,
                )

        if any(self.finite_source):
            # `star_constrains_rho: false` severs rho = theta_star/theta_E:
            # rho becomes the light curve's own parameter (sampled as
            # log_rho, mirroring log_s/s) and the stellar chain's
            # prediction is reported separately as rho_pred, so the pull
            # between the two is a published number instead of a silently
            # resolved tension (see rho_pred's defaults.yaml note; the
            # planet component's beam_constrains_mass is the same
            # vocabulary).  The
            # relaxation engine still knows BOTH relations, so a rho seed
            # (user or MMEXOFAST) back-solves to a log_rho start and the
            # stellar chain still seeds consistently.
            if not bool(self.config[0].get("star_constrains_rho", True)):
                self.manifest["log_rho"] = per_source()
                self.manifest["rho"] = per_source("from_log_rho")
                self.manifest["rho_pred"] = per_source("default")
            else:
                self.manifest["rho"] = per_source("default")
        elif not bool(self.config[0].get("star_constrains_rho", True)):
            logger.warning(
                "lens: star_constrains_rho is false but finite_source is "
                "not set -- rho never enters the magnification, so there "
                "is nothing to sever; ignoring it."
            )

        # Seed alpha hint (degrees, user unit) so inspect_start can display it
        # even before the expression graph is built.
        inst = self.names[0] if self.names else "0"
        ca_entry = (
            self.config_manager.user_params.get(f"lens.{inst}.xalpha")
            or self.config_manager.user_params.get(f"lens.0.xalpha")
            or {}
        )
        sa_entry = (
            self.config_manager.user_params.get(f"lens.{inst}.yalpha")
            or self.config_manager.user_params.get(f"lens.0.yalpha")
            or {}
        )
        ca = ca_entry.get("initval")
        sa = sa_entry.get("initval")
        # List-valued initval (P4 multi-seed sampling): use seed 0.
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else None
        if isinstance(sa, (list, tuple)):
            sa = sa[0] if sa else None
        if ca is not None and sa is not None:
            alpha_deg = float(np.arctan2(float(sa), float(ca)) * _RAD_TO_DEG)
            self.config_manager.add_hint(
                f"lens.0.alpha", alpha_deg, rank=RANK_DEFAULT
            )

        # Expected proper motions from the galactic model, for the seeds below.
        # None when the line of sight is not known yet, in which case the pm
        # hints are simply skipped (the old behavior).
        pm_expected = self._galactic_pm_expectations(system)

        # (helpers for the pm seeding live at _galactic_pm_expectations /
        # _seed_expected_pm, below this method.)

        # Inject per-event physical hints
        for i in range(self.n_elements):
            l_type, l_idx = self._primary_lens(i)

            # RANK_MULENS_LENS_DISTANCE overrides the 10 pc defaults.yaml
            # default (RANK_DEFAULT) but yields to any value the relaxation
            # engine derives from pi_rel + d_S
            # (RANK_MULENS_SOURCE_DISTANCE).  That ordering is what breaks
            # the d_L <-> parallax cycle: pi_rel drives d_L to the source
            # rank via Condition B, then the parallax is corrected as the
            # weaker symbol.  See the constants' own comment in config.py.
            self.config_manager.add_hint(
                f"star.{l_idx}.distance",
                4000.0,
                rank=RANK_MULENS_LENS_DISTANCE,
            )
            self.config_manager.add_scale_hint(f"star.{l_idx}.distance", 5.0)
            self.config_manager.add_hint(f"star.{l_idx}.logmass", -0.5)
            self.config_manager.add_scale_hint(f"star.{l_idx}.logmass", 0.001)
            self.config_manager.add_scale_hint(f"star.{l_idx}.pm_ra", 3.0)
            self.config_manager.add_scale_hint(f"star.{l_idx}.pm_dec", 3.0)
            self.config_manager.add_scale_hint(f"star.{l_idx}.rv", 1e5)
            self._seed_expected_pm(pm_expected, l_idx, "thin_disk", 4000.0)

            # Every source body gets the same bulge-source seeding: each source
            # has its own trajectory chain (distance, pm) to initialize.
            for s_type, s_idx in self.source_bodies[i]:
                self.config_manager.add_hint(
                    f"star.{s_idx}.distance",
                    8000.0,
                    rank=RANK_MULENS_SOURCE_DISTANCE,
                )
                self.config_manager.add_scale_hint(
                    f"star.{s_idx}.distance", 5.0
                )
                self.config_manager.add_hint(f"star.{s_idx}.logmass", -0.5)
                self.config_manager.add_scale_hint(
                    f"star.{s_idx}.logmass", 0.3
                )
                self.config_manager.add_scale_hint(f"star.{s_idx}.pm_ra", 3.0)
                self.config_manager.add_scale_hint(f"star.{s_idx}.pm_dec", 3.0)
                self.config_manager.add_scale_hint(f"star.{s_idx}.rv", 1e5)
                self._seed_expected_pm(pm_expected, s_idx, "bulge", 8000.0)

            # Companion lens bodies (everything beyond the primary)
            for l2_type, l2_idx in self.lens_bodies[i][1:]:
                if l2_type == "star":
                    self.config_manager.add_hint(
                        f"star.{l2_idx}.distance",
                        4000.0,
                        rank=RANK_MULENS_LENS_DISTANCE,
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
                for i in range(self.n_elements):
                    _, l_idx = self._primary_lens(i)
                    self.config_manager.add_scale_hint(
                        f"star.{l_idx}.logmass", scale
                    )

    def _galactic_pm_expectations(self, system):
        """Line of sight for the galactic-model proper-motion seeds.

        Returns ``(ra_rad, dec_rad)``, or None when the seeding does not apply.

        The prior is only allowed to FILL A GAP, never to contradict.  What is
        open here is the physical side: no example pins the lens mass or
        distance, because a published light-curve solution (t_0, u_0, t_E, s, q,
        alpha, rho, sometimes pi_E) does not close the system -- t_E and pi_E
        without theta_E leave mass, distance and proper motion free.  That gap
        is what the engine used to fill by inventing a direction (issue #93).

        But where a config DOES imply the proper motion, a prior mean dropped on
        top fights it.  Measured at the seed (raw = 0), chi2/N ungated vs gated:

            ob140939 (pi_E_N/pi_E_E measured, Yee+2015)  3.04 -> 179.1 | 3.04
            ob161003 (two sources, t_E + rho each)       1.72 ->   3.9 | 1.72
            DC2018_128 (t_0/u_0/t_E/s/q/alpha/rho)       1.42 ->   1.21 (kept)
            ob08092 (t_0/u_0/t_E only, PSPL)             1.50 ->   1.42 (kept)

        So the gates below are what keeps this from making published solutions
        worse.  Filling only the *direction* and leaving the magnitude to the
        data would serve every case at once, but the direction is not a symbol,
        so provenance cannot express it per-symbol; that needs a basis change
        (mu_rel_mag, mu_rel_pa) which is a sampling-geometry question and does
        not belong here.  tests/test_seed_quality.py pins all four numbers.
        """
        if "galacticmodel" not in getattr(system, "config", {}):
            # No galactic model in the topology: nothing to take the mean of.
            return None
        # Skip when something already implies the direction or the magnitude.
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
                f"[lens] proper motion or parallax already given "
                f"({', '.join(sorted(blockers))}); not seeding from the "
                f"galactic model, which would contradict it."
            )
            return None
        if self.n_sources > 1:
            # Every source would be seeded at the same bulge mean, forcing one
            # mu_rel for all of them.  A resolved binary source distinguishes
            # them, so do not impose it.
            logger.info(
                f"[lens] {self.n_sources} sources; not seeding proper motions "
                f"from the galactic model (one mean would tie their mu_rel "
                f"together)."
            )
            return None
        try:
            n_stars = system.star.n_elements
            source_ndx = int(system.lens.source_map[0])
            ra_all = self.config_manager.resolve(
                "star", "ra", shape=(n_stars,)
            )["initval"]
            dec_all = self.config_manager.resolve(
                "star", "dec", shape=(n_stars,)
            )["initval"]
        except Exception as exc:  # pragma: no cover - seeds are optional
            logger.debug(
                f"[lens] could not resolve the line of sight for the "
                f"galactic-model proper-motion seeds: {exc!r}"
            )
            return None

        keys = [f"star.{source_ndx}.ra", "star.ra"]
        names = getattr(system.star, "names", None)
        if names:
            keys.append(f"star.{names[source_ndx]}.ra")
        if not any(k in self.config_manager.user_params for k in keys):
            logger.debug(
                "[lens] no user-set RA/Dec; skipping the galactic-model "
                "proper-motion seeds."
            )
            return None

        # resolve() hands back the value in the parameter's USER unit, which for
        # ra/dec is degrees (Parameter.__post_init__ is what converts to the
        # internal radians, and it has not run at stage 3).  The galactic-model
        # helpers take radians.
        return (
            float(np.radians(np.atleast_1d(ra_all)[source_ndx])),
            float(np.radians(np.atleast_1d(dec_all)[source_ndx])),
        )

    def _seed_expected_pm(self, line_of_sight, star_idx, population, dist_pc):
        """Seed one star's pm_ra/pm_dec at the galactic model's prior mean.

        RANK_DERIVED_DATA: this is derived from the galactic model the same way
        an RV offset is derived from the data, so it belongs in that tier and
        must yield to anything in params.yaml.

        It ties with the MMEXOFAST seeds (also RANK_DERIVED_DATA), which is the
        point.  Both proper-motion components are now pinned, so ``mu_rel`` has
        a magnitude AND a direction, and the engine no longer has to invert
        ``mu_rel_mag**2 = mu_ra_rel**2 + mu_dec_rel**2`` -- one equation in two
        unknowns -- by choosing a point on a circle (issue #93).  Where that
        disagrees with the seeded ``t_E``, Condition B rewrites the lowest-rank
        symbol in ``t_E = theta_E / |mu_rel_geo|``, which is ``theta_E`` via the
        lens mass (defaults.yaml, ``RANK_DEFAULT``) and distance
        (``RANK_MULENS_LENS_DISTANCE``).  So ``t_E``
        keeps its measured value, the proper motion keeps the prior's, and the
        lens mass absorbs the difference -- which is the standard microlensing
        chain (a measured t_E plus an assumed mu_rel implies theta_E, hence a
        mass) and is the quantity a light curve genuinely cannot pin down.

        `dist_pc` must match the distance hint seeded for the same star: the
        mean velocity is position-dependent, so a mismatch would seed a proper
        motion for a place the star is not.
        """
        if line_of_sight is None:
            return
        ra_rad, dec_rad = line_of_sight
        try:
            pm_ra, pm_dec, _rv = expected_proper_motion(
                ra_rad, dec_rad, dist_pc, population
            )
        except Exception as exc:
            # WARNING, not debug: this silently disabled the whole feature once
            # already (degrees were passed where radians were wanted, astropy
            # raised, and the seeds just quietly never happened).  A failure
            # here is not fatal -- the old arbitrary start still works -- but it
            # must be visible.
            logger.warning(
                f"[lens] could not seed star.{star_idx}'s proper motion from "
                f"the galactic model ({population} at {dist_pc:.0f} pc): "
                f"{exc!r}.  Falling back to the defaults.yaml value; the "
                f"direction of mu_rel will be arbitrary (see issue #93)."
            )
            return
        self.config_manager.add_hint(
            f"star.{star_idx}.pm_ra", pm_ra, rank=RANK_DERIVED_DATA
        )
        self.config_manager.add_hint(
            f"star.{star_idx}.pm_dec", pm_dec, rank=RANK_DERIVED_DATA
        )
        logger.info(
            f"[lens] star.{star_idx} proper motion seeded at the "
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
        v = np.asarray(vel, dtype=float) * 365.25  # AU/day -> AU/yr
        ra, dec = radec
        # The basis itself, not observer_sky_offset: what is projected here
        # is the Earth's VELOCITY, not its position.
        e_hat, n_hat = sky_basis(ra, dec)
        return float(v @ e_hat), float(v @ n_hat)

    def _validate_q_start(self):
        """Stage 7: check the START value of the mass ratio, loudly and once.

        The magnification path clips q into [Q_MIN, Q_MAX] (physics.clip_q) --
        a statement about where the backends are defined, not a licence to
        invent a mass ratio.  The clip used to be preceded by
        ``pt.nan_to_num(q, nan=Q_MIN)``, which silently turned a failed
        computation into a healthy-looking likelihood.  That scrub is gone; a
        NaN now reaches logp and the proposal is rejected.  What the scrub also
        hid, though, was the *start*, and a bad start is the case that is worth
        a message rather than a rejection -- so it is checked here, once, on
        the inputs, where a raise costs nothing and can say what to do.

        NaN is fatal: the fit cannot start.  Out of range (the infinities
        included -- they at least carry a sign, the same split clip_q_value
        makes) is a warning: the fit will silently begin at the clipped q
        rather than at the seeded one, which is exactly the sort of "the number
        I typed is not the number being fitted" that goes unnoticed for months.

        **NaN is fatal only where it MEANS something**, which is companion
        slot 0 (review 1.6.5).  The split is not about q being derived -- it
        always is -- but about which elements the relaxation engine can
        actually solve: ``symbolic_physics.get_symbol_map`` maps a SINGLE
        companion, so for slot 0 a NaN really does say the solve failed, i.e.
        one of the lens body masses is already non-finite, and the advice
        below is the right advice.  Slots 1 and up are never solved by the
        engine at all: `register_parameters` seeds them from USER body-mass
        entries only, skips the hint when there are none (see 2.6.6), and
        `resolve()` then leaves them NaN because q has no defaults.yaml
        initval.  That NaN is bookkeeping, not a start -- the graph recomputes
        q from the mass nodes, which carry finite defaults -- and raising on it
        killed a 3+ body fit that would have run perfectly well.  Exactly the
        false-positive class :meth:`_validate_pspl_start`'s docstring warns
        about for the derived t_E/theta_E/pi_E (the ob161003 theta_E lesson).

        A q that genuinely reaches the magnification backend as NaN is still
        caught at runtime by ``clip_q_value``, which names the parameter.  The
        derived-ness test is kept for the skipped slots so that a future
        parameterization which SAMPLES one of them gets the raise back: for a
        sampled element the initval IS the start.
        """
        if self.n_companions < 1 or self.q.initval is None:
            return
        q0 = np.atleast_1d(np.asarray(self.q.initval, dtype=float)).ravel()
        nan = np.isnan(q0)
        fatal = [
            i
            for i in np.flatnonzero(nan)
            if i == 0 or not self.q.element_is_derived(int(i))
        ]
        if fatal:
            raise ValueError(
                f"{self.prefix}.q starts at {q0.tolist()}, which is not a "
                f"number.  {_Q_NAN_ADVICE}"
            )
        out = ~nan & ((q0 < Q_MIN) | (q0 > Q_MAX))
        if np.any(out):
            logger.warning(
                f"{self.prefix}.q starts at {q0[out].tolist()}, outside the "
                f"[{Q_MIN:g}, {Q_MAX:g}] range the binary-lens magnification "
                "backends are defined on, so the fit will actually START at "
                "the clipped value.  Move the start inside the range (set "
                f"{self.prefix}.q, or the companion/primary masses it is "
                "derived from) rather than relying on the clip."
            )

    def _start_values(self, name):
        """Resolved start value of a lens parameter as a 1-D float array, or
        None when it has none (unset, or not castable -- a multi-seed entry
        that survived as a ragged object array)."""
        par = getattr(self, name, None)
        if par is None or par.initval is None:
            return None
        try:
            return np.atleast_1d(np.asarray(par.initval, dtype=float)).ravel()
        except (TypeError, ValueError):
            return None

    def _validate_pspl_start(self):
        """Stage 7: check the START values of the SAMPLED trajectory
        parameters, loudly and once.  The sibling of
        :meth:`_validate_q_start`, and it makes the same split for the same
        reason: NaN raises (the fit cannot start), out of range warns (the fit
        begins at the floored value rather than at the seeded one -- the "the
        number I typed is not the number being fitted" case that goes
        unnoticed for months).  This is the half of the old scrub worth
        keeping, moved to where a raise is free: a check on the inputs at
        build time, not a mid-graph assert that would kill a run over a
        proposal the sampler already rejects on its own.

        **Only t_0 and u_0 are checked, and that is deliberate.**  They are
        the two trajectory parameters that are sampled, so their ``initval``
        IS the start: raw = 0 maps to it through the logit transform.  The
        other four quantities `_get_safe_mm_params` handles -- t_E, theta_E,
        pi_E_N and pi_E_E -- are DERIVED, and for a derived parameter
        ``initval``
        is the relaxation engine's own bookkeeping, not the value the model
        starts at; the graph recomputes it from the sampled coordinates.  The
        two genuinely differ, so checking them here would be a false positive
        on working configs.  Measured on `examples/ob161003` (2S2L, two source
        slots): the engine leaves ``lens.theta_E.initval = [nan, 0.8393]`` and
        ``lens.pi_rel.initval = [nan, 0.125]`` -- it only ever needed to solve
        the second slot, both sources sharing one lens -- while the model
        starts at a perfectly good ``theta_E = [0.8393, 0.8393]`` and a finite
        logp.  A NaN there says nothing about the fit.  (That the engine
        writes a NaN into a resolved value at all is a separate, pre-existing
        oddity; it is not this guard's business to report it.)

        The one range check is on ``|u_0| < U_0_FLOOR``: the fit will not start
        where the seed says, it will start at the floored value.  ``u_0: 0`` --
        a plausible seed for a high-magnification event -- is included, and it
        used to be the one case the floor MISSED (``sign(0) = 0`` made the old
        ``sign(u_0) * maximum(|u_0|, U_0_FLOOR)`` return 0 and left the peak
        magnification singular).  ``physics.apply_u_0_floor`` now sends it to
        ``+U_0_FLOOR``; the warning names the value it will actually start at.
        t_0 gets no range check -- it carries two finite hard bounds of its own.
        """
        sampled = {
            "t_0": self._start_values("t_0"),
            "u_0": self._start_values("u_0"),
        }
        nan_named = [
            f"{self.prefix}.{n} = {v.tolist()}"
            for n, v in sampled.items()
            if v is not None and np.any(np.isnan(v))
        ]
        if nan_named:
            raise ValueError(
                "The lensing trajectory starts at a value that is not a "
                f"number: {'; '.join(nan_named)}.  {_MM_NAN_ADVICE}"
            )

        u_0 = sampled["u_0"]
        if u_0 is not None and np.any(np.abs(u_0) < U_0_FLOOR):
            small = u_0[np.abs(u_0) < U_0_FLOOR]
            floored = [floor_u_0_value(v) for v in small]
            logger.warning(
                f"{self.prefix}.u_0 starts at {small.tolist()}, inside the "
                f"{U_0_FLOOR:g} floor on |u_0| (the magnification diverges "
                f"at u = 0), so the fit will actually START at {floored} -- "
                "and an exactly central trajectory has no side, so u_0 = 0 "
                f"is floored to +{U_0_FLOOR:g} by convention.  Seed the "
                "impact parameter you mean, with the sign you mean."
            )

    def build_likelihood(self, model, system):
        """Stage 7: Observational penalties on the lensing geometry."""
        self._validate_q_start()
        self._validate_pspl_start()

        # GEOCENTRIC mu_rel: the event-rate selection is the sky-sweep rate
        # in the frame the event is observed in (rp.py used the geocentric
        # value at t0_par; Batista+2011's rate is in the frame of the
        # measured t_E), and it is also the divisor of t_E/pi_E, so the
        # singularity guard belongs on it.
        mu_rel_geo = self.mu_rel_geo_mag.value
        theta_E = self.theta_E.value

        # Both logs are floored (belt and braces -- calc_theta_E and
        # calc_mu_rel_mag already floor their radicands, see physics.py).  A
        # bare log(0) is a -inf wall with no gradient for NUTS to follow,
        # which is exactly what the soft bounds below exist to avoid; the
        # floors are ~6 decades below their 1e-6 turn-on, so the prior is
        # untouched wherever it was already finite.
        pm.Potential(
            f"{self.prefix}.event_rate_prior",
            pt.sum(
                pt.log(pt.maximum(mu_rel_geo, MU_REL_FLOOR))
                + pt.log(pt.maximum(theta_E, THETA_E_FLOOR))
            ),
        )
        get_collector(system).add(
            r"We weighted the lens prior by the microlensing event rate, "
            r"$\Gamma \propto \mu_{\rm rel}\,\theta_{\rm E}$ "
            r"\citep[e.g.][]{Batista:2011}.",
            section="priors",
            key=f"{self.prefix}.event_rate",
            rank=30,
        )

        # Shared log-sigmoid barriers (see exozippy.potentials): smooth and
        # asymptotically linear, so the sampler feels a restoring gradient
        # instead of a -1e6 cliff. scale=440 pc preserves the previous ~1/pc
        # slope; scale=1e-5 puts the singularity turn-on at ~1e-7 (mas or
        # mas/yr), matching the previous steepness.
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

    # ------------------------------------------------------------------
    # Magnification
    # ------------------------------------------------------------------

    def _alpha_deg(self, j=0):
        """Trajectory angle of companion ``j`` in DEGREES -- the unit both
        magnification backends take, while lens.alpha's internal unit is
        radians.

        Reads the alpha Parameter rather than re-deriving ``arctan2(yalpha,
        xalpha)`` at each call site (it was open-coded twice; review item 4.5).
        alpha's expression IS that arctan2 (physics.calc_alpha), so this is
        bit-identical, and going through the Parameter means the angle handed
        to the backend is by construction the same one the reports, priors and
        plots see.
        """
        return self.alpha.value[j] * _RAD_TO_DEG

    def _get_safe_mm_params(self, index=0):
        """Range-limited single-source trajectory params.  ``index`` is the
        SOURCE slot: the per-source vector parameters (t_0, u_0, t_E, pi_E_*)
        hold one element per source body of the single event.

        Three RANGE decisions survive here -- the t_E floor, the |u_0| floor
        and the no-lensing parallax gate, all defined and justified next to
        their constants in physics.py.  What is deliberately GONE is the NaN
        substitution that used to precede them:

            t_E -> 100 d,  u_0 -> 1,  theta_E -> 0,  pi_E_N -> 0,  pi_E_E -> 0

        i.e. a complete, fabricated PSPL model in place of a failed
        computation.  It is the same defect ``clip_q``'s ``pt.nan_to_num``
        was (review item 4.5), five more times and with a much larger blast
        radius: a fully-NaN parameter vector produced a healthy-looking light
        curve and a finite likelihood.

        Removing it is safe *and* strictly better, for the same two reasons:

        * It is unreachable.  Every one of the five is finite for every finite
          raw vector.  t_0 and u_0 are sampled with two finite hard bounds, so
          the logit transform can only produce a finite number.  theta_E is
          ``sqrt(max(KAPPA*max(M,1e-12)*max(pi_rel,0), THETA_E_FLOOR**2))``,
          strictly positive and finite for any finite mass and pi_rel, and
          pi_rel is a difference of two 1000/distance terms whose distances
          are logit-bounded away from zero.  t_E = theta_E/(mu_rel_geo/365.25)
          and pi_E = (pi_rel/theta_E)*(mu_i/mu_rel_geo) are then ratios whose
          denominators are floored at THETA_E_FLOOR and MU_REL_FLOOR -- those
          two floors, added in c178305, are exactly what closed the 0/0 that
          made this scrub live when it was written (May 2026), back when
          calc_mu_rel_mag was a bare sqrt that could return exactly 0.
          Measured on examples/ob08092 (PSPL), examples/ob140939 (parallax +
          Spitzer) and examples/DC2018_128 (binary lens): all five stay finite
          over the entire raw support out to raw = +/-1e12, one variable at a
          time and all at once, plus 2000 random raw points per event.  Three
          real 300-tune/300-draw ptde_async fits (28 worker processes each,
          172k / 215k / 223k evaluations) instrumented at the scrub itself
          never once entered the branch.
        * Where it could fire it could only do harm.  These five are NaN only
          when an input is already NaN, i.e. the raw vector itself carries a
          NaN -- and that raw variable's own N(0, 1) prior term already makes
          the total logp NaN, so the proposal is rejected whatever this
          function returns (verified on all three events, for every sampled
          coordinate).  Substituting a "safe" value could never rescue a
          sample; it invented an entire event geometry -- with a zero
          gradient, since nan_to_num is a switch -- in place of the one
          quantity that would have named the failure.

        The theta_E substitution was not even that: ``theta_E_scrubbed`` fed
        nothing but the ``pt.gt(..., 1e-6)`` comparison, and a comparison
        against NaN is already False, so dropping it is a no-op in every case,
        NaN included.

        A NaN now propagates to logp, which is the sampler's own reject
        signal, so nothing here needs a mid-graph assert (which would kill a
        whole run over a proposal that is already being rejected) or a -inf
        potential (no gradient, and the JAX where-trap).  The two SAMPLED
        start values are checked once, loudly, in _validate_pspl_start; the
        numeric Op path names the parameter through physics.require_mm_number.
        """
        tE_raw = self.t_E.value[index]
        u0_raw = self.u_0.value[index]
        theta_E_raw = self.theta_E.value[index]

        tE_safe = pt.maximum(tE_raw, T_E_FLOOR)
        u0_safe = apply_u_0_floor(u0_raw)
        is_physical = pt.gt(theta_E_raw, THETA_E_LENSING_MIN)

        # Keys are the CANONICAL parameter names, matching op.py's
        # _base_mm_params exactly.  They used to be a private dialect
        # (t0/u0/tE/pi_N/pi_E) whose "pi_E" meant pi_E_E, so a grep for
        # pi_E_E missed every consumer of this dict while a grep for pi_E hit
        # the wrong one (review 4.6.1).  Names only -- no sign, no floor and
        # no expression changed; the parallax convention is stated at the one
        # place that applies it, get_magnification below.
        return {
            "t_0": self.t_0.value[index],
            "u_0": u0_safe,
            "t_E": tE_safe,
            "pi_E_N": pt.switch(is_physical, self.pi_E_N.value[index], 0.0),
            "pi_E_E": pt.switch(is_physical, self.pi_E_E.value[index], 0.0),
        }

    def _get_binary_mm_params(self, index=0):
        """Params for a binary lens.  ``index`` is the SOURCE slot; the lens
        bodies are shared by all sources (single event ⇒ event index 0).

        The derived chain (theta_E, t_E, rho, pi_E) is already referenced to
        the TOTAL lens mass via mlens_total, so the safe single-source params
        pass straight through — only the companion geometry (s, q, alpha) is
        added here.

        It no longer takes ``system``: q used to be recomputed here from the
        two body mass nodes, which is what needed it.  It now reads the q
        Parameter, which is that same ratio (physics.calc_q) and is what every
        other consumer already uses.
        """
        s = self._get_safe_mm_params(index)

        # s/q/alpha are indexed by companion (binary = companion 0), not by
        # event or source.
        return {
            **s,
            "s": self.s.value[0],
            "q": clip_q(self.q.value[0]),
            "alpha": self._alpha_deg(0),
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
        reads the live star.ra/star.dec nodes, while the Op takes a coordinate
        STRING frozen at the start value (_frozen_op_coords_deg, which warns
        when they are sampled).  That only separates the two paths in a
        topology that actually samples ra/dec, and by ~1e-5 per arcsec of
        coordinate error -- see that method for why the freeze is free.
        """
        source_ndx = self.source_map[index]
        ra = system.star.ra.value[source_ndx]
        dec = system.star.dec.value[source_ndx]

        delta_e, delta_n = observer_sky_offset(obs_pos, ra, dec, xp=pt)

        p = self._get_safe_mm_params(index)
        # MulensModel convention: delta_tau = -delta_N*pi_E_N - delta_E*pi_E_E
        # (negative on both N and E, matching Skowron+2011 via MulensModel's
        # sign choice). MMEXOFAST calls MulensModel, so published pi_E values
        # are calibrated to this convention.
        tau_p = (
            (times - p["t_0"]) / p["t_E"]
            - delta_n * p["pi_E_N"]
            - delta_e * p["pi_E_E"]
        )
        u_p = p["u_0"] + delta_n * p["pi_E_E"] - delta_e * p["pi_E_N"]

        u2 = pt.sqr(tau_p) + pt.sqr(u_p)
        return (u2 + 2.0) / pt.sqrt(u2 * (u2 + 4.0))

    def uses_op(self, index=0):
        """Return True if get_magnification_op will dispatch to the MulensModel Op.

        Event-level property (the lens bodies and finite_source flag are shared
        by all sources), so ``index`` is ignored beyond backward compatibility.

        Both paths take the same obs_pos convention (Skowron+2011 geocentric
        deviations); callers use this only to pick a sampler-compatible path.
        """
        n_lenses = self.n_lens_bodies[0]
        use_rho = self.finite_source[0]
        forced = self.use_op[0]
        return forced or (n_lenses > 1) or use_rho

    def sampler_requirements(self):
        """Declare sampler constraints for this lens configuration.

        Binary/finite-source lenses use the MulensModel Op, which is not
        differentiable.  Gradient-based samplers (NUTS, numpyro, blackjax)
        will produce invalid results; PTDE is required.  The asynchronous
        dispatch loop (ptde_async) is recommended: near-caustic evaluations
        concentrate in the hot rungs and stall the synchronous sampler's
        every step behind the slowest proposal (samplers/ptde_async.py).

        PSPL lenses use a symbolic PyTensor formula and are NUTS-compatible,
        so no constraints are returned.
        """
        if any(self.uses_op(i) for i in range(len(self.n_lens_bodies))):
            return {
                "incompatible": {"nuts", "numpyro", "blackjax"},
                "recommended": "ptde_async",
                "reason": (
                    "binary/finite-source microlensing uses the MulensModel Op, "
                    "which is not differentiable — gradient-based samplers produce "
                    "invalid results"
                ),
            }
        return {}

    def _frozen_op_coords_deg(self, system, source_ndx):
        """(ra, dec) in degrees baked into the MulensModel / VBM Op, ONCE.

        The Op takes the line of sight as a coordinate STRING, so it cannot
        track a sampled ``star.ra``/``star.dec``: whatever is read here is
        frozen for the whole fit.  That freeze is deliberate and numerically
        free.  Microlensing parallax enters only through the PROJECTION of the
        Earth's orbit onto the event's (N, E) axes, so a coordinate error of
        eps radians perturbs the projection by ~eps relative: 1 arcsec is
        5e-6, nothing against pi_E uncertainties of order 1%.  Making the
        coordinates dynamic would rebuild the Op every likelihood call to buy
        a correction six orders of magnitude below the measurement.

        What is NOT free is doing it silently, so a topology that actually
        samples the source's ra/dec (microlensing + gaia/abs astrometry) gets
        one warning per source naming the frozen values.  Nothing is emitted
        for the overwhelmingly common case where they are pinned -- a warning
        on every microlensing fit is a warning nobody reads.

        The value comes from ``initval``, not from ``.eval()`` of the value
        node.  A sampled element's node IS a random variable, so ``.eval()``
        draws from its prior: the old code did not freeze the start value, it
        froze an arbitrary draw (measured 0.36 deg away on a mulens topology
        with a sampled source position).
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
                f"(ra={ra_deg:.6f} deg, dec={dec_deg:.6f} deg) for the whole "
                f"fit. This is safe -- the parallax projection is perturbed "
                f"only ~1e-5 per arcsec of coordinate error, far below any "
                f"pi_E uncertainty -- but the sampled ra/dec do NOT feed the "
                f"microlensing model."
            )
        return ra_deg, dec_deg

    def get_magnification_op(
        self, times, obs_pos, system, index=0, u1=None, bandpass=None
    ):
        """Magnification dispatcher.

        ``index`` is the SOURCE slot: each source body has its own trajectory
        (t_0, u_0, rho, ...) but shares the lens bodies.  Multi-source callers
        (MulensInstrument) invoke this once per source and combine the returned
        magnifications with per-source fluxes.

        For point-source PSPL (n_lenses==1, finite_source=False, use_op=False)
        falls back to the symbolic PyTensor formula so NUTS can differentiate
        through it without the O(N_params) numerical-gradient overhead of
        _MagGradOp.

        obs_pos: (N, 3) Skowron+2011 geocentric deviations in AU
        (MulensInstrument._abs_to_delta) for BOTH paths -- the symbolic
        formula projects them directly, and the Op path feeds them to
        MulensModel as satellite_skycoord (whose satellite channel then
        carries all parallax, annual + satellite).

        u1/bandpass: when finite_source is True and a Band component is wired,
        u1 (a PyTensor scalar) and bandpass (str) are passed so the Op can call
        set_limb_coeff_u and get_magnification(bandpass=...).  Passing neither
        falls back to uniform-source finite-source magnification.

        Set ``use_op: true`` in the lens YAML block to force the Op (e.g. for
        testing or when MulensModel's finite-source parallax is needed).
        """
        if self.n_lens_bodies[0] > 2 and self.backend != "vbm_direct":
            raise NotImplementedError(
                f"{self.n_lens_bodies[0]}-lens magnification requires "
                "backend: vbm_direct (VBMicrolensing MultiMag2); the "
                "MulensModel backend supports at most 2 lens bodies."
            )

        if not self.uses_op(index):
            return self.get_magnification(times, obs_pos, system, index)

        source_ndx = self.source_map[index]
        ra_deg, dec_deg = self._frozen_op_coords_deg(system, source_ndx)
        coords = f"{ra_deg}d {dec_deg}d"

        use_rho = self.finite_source[0]
        n_lenses = self.n_lens_bodies[0]

        # Apply LD only for finite-source and when a band is connected.
        effective_bandpass = bandpass if (use_rho and u1 is not None) else None

        times_tensor = pt.as_tensor_variable(times)
        obs_tensor = pt.as_tensor_variable(obs_pos)

        if n_lenses >= 2 and self.backend == "vbm_direct":
            sp = self._get_safe_mm_params(index)
            param_list = [
                sp["t_0"],
                sp["u_0"],
                sp["t_E"],
                sp["pi_E_N"],
                sp["pi_E_E"],
            ]
            if use_rho:
                param_list.append(self.rho.value[index])
            for j in range(self.n_companions):
                param_list.extend(
                    [
                        self.s.value[j],
                        clip_q(self.q.value[j]),
                        self._alpha_deg(j),
                    ]
                )
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = VBMDirectMagOp(
                coords=coords,
                n_companions=self.n_companions,
                use_rho=use_rho,
                bandpass=effective_bandpass,
            )
        elif n_lenses == 2:
            bp = self._get_binary_mm_params(index)
            param_list = [
                bp["t_0"],
                bp["u_0"],
                bp["t_E"],
                bp["pi_E_N"],
                bp["pi_E_E"],
            ]
            if use_rho:
                param_list.append(self.rho.value[index])
            param_list.extend([bp["s"], bp["q"], bp["alpha"]])
            if effective_bandpass is not None:
                param_list.append(u1)
            mag_op = BinaryLensMagOp(
                coords=coords,
                mag_method=self.mag_method[index],
                use_rho=use_rho,
                bandpass=effective_bandpass,
            )
        else:
            sp = self._get_safe_mm_params(index)
            param_list = [
                sp["t_0"],
                sp["u_0"],
                sp["t_E"],
                sp["pi_E_N"],
                sp["pi_E_E"],
            ]
            if use_rho:
                param_list.append(self.rho.value[index])
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
        """Replace 'auto_vbbl' with a concrete method list for multi-body lenses.
        Historically this computed hexadecapole-vs-VBM brackets on a time
        grid, but MulensModel implements binary-lens hexadecapole as 13
        python-level VBM.BinaryMag0 calls per epoch while VBM's BinaryMag2
        runs the equivalent quadrupole safety test internally in C++ and
        short-circuits to point-source when safe.  Measured on DC2018_128:
        hexadecapole 32.9 ms vs VBM-everywhere 7.7 ms per 870-point call, at
        equal or better accuracy — so the bracket machinery optimized for the
        wrong cost model and was removed (see hpc_optimization.txt, P1).

        Single-lens events are left untouched: 'auto_vbbl' is resolved inside
        the PSPL model builder (point_source + finite-source window), and the
        VBM/VBBL methods emitted here are binary-lens-only.

        Only the mulensmodel backend consumes the resulting method list; the
        default vbm_direct backend always calls BinaryMag2/MultiMag2.
        """
        if self.mag_method[index] != "auto_vbbl":
            return
        if self.n_lens_bodies[0] < 2:
            return

        t_lo = float(np.min(times_np))
        t_hi = float(np.max(times_np))
        method = "VBM"  # if self.finite_source[0] else "VBBL"
        self.mag_method[index] = [t_lo - 1.0, method, t_hi + 1.0]
        # logger.info(f"auto_vbbl: using {method} everywhere for source {index} "
        #            "(hexadecapole bracketing removed — VBM's internal C++ "
        #            "point-source test is faster; see hpc_optimization.txt P1)")

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

    def _companion_instance_names(self):
        """Display names for per-companion vector elements (companion lens bodies)."""
        system_config = (
            getattr(self.config_manager, "system_config", None) or {}
        )
        names = []
        for comp_type, ndx in self.lens_bodies[0][1:]:
            entries = system_config.get(comp_type, [])
            if (
                ndx < len(entries)
                and isinstance(entries[ndx], dict)
                and entries[ndx].get("name")
            ):
                names.append(str(entries[ndx]["name"]))
            else:
                names.append(f"{comp_type}{ndx}")
        return names

    def plot_corner(self, idata, filename_prefix="debug"):
        """Corner plot of the fitted lensing geometry: t_0, u_0, t_E, s, q,
        alpha, rho -- whichever of these the event actually has (rho only for
        finite-source events; s/q/alpha only when there is at least one lens
        companion). Only meaningful with the full posterior, so this is
        called once, after sampling, via plot_corner (not the twice-called
        plot() hook, which also runs pre-flight on a single point).

        t_E (and, for multi-body lenses, q and alpha) are pure physics
        expressions with no sampled elements of their own, so they never get
        a pm.Deterministic node and never appear in idata.posterior directly
        (see Parameter.build_pymc's ``track_node`` logic) -- this reads each
        Parameter's ``.posterior`` instead, which System.distribute_posterior
        (already called earlier in run_fit, before this hook) reconstructs
        for both tracked and pure-expression parameters alike.
        """
        src_names = (
            self._source_instance_names() if self.n_sources > 1 else None
        )
        comp_names = (
            self._companion_instance_names() if self.n_companions > 1 else None
        )

        def per_source_labels(param):
            return (
                [f"{param}[{name}]" for name in src_names]
                if src_names
                else None
            )

        def per_companion_labels(param):
            return (
                [f"{param}[{name}]" for name in comp_names]
                if comp_names
                else None
            )

        param_specs = [
            (self.t_0, per_source_labels("t_0")),
            (self.u_0, per_source_labels("u_0")),
            (self.t_E, per_source_labels("t_E")),
        ]
        if hasattr(self, "rho"):
            param_specs.append((self.rho, per_source_labels("rho")))
        if self.n_companions >= 1:
            param_specs.append((self.s, per_companion_labels("s")))
            param_specs.append((self.q, per_companion_labels("q")))
            param_specs.append((self.alpha, per_companion_labels("alpha")))

        samples, labels = collect_parameter_corner_samples(param_specs)
        save_corner_plot(samples, labels, f"{filename_prefix}_lens_corner.png")
