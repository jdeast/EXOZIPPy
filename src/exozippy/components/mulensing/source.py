"""The microlensing SOURCE component (8.6.17 stage 1).

One instance per SOURCE BODY.  Under ruling R1 every source's path is a
parallel line differing only by offset, so exactly the per-track quantities
live here: `t_0` (when that track crosses closest approach), `u_0` (its
perpendicular offset), `rho` (that source's angular size in Einstein units,
plus its `log_rho`/`rho_pred` coordinates), and the `fitu0te` coordinate
`u0te`.  Everything shared by the tracks -- t_E, theta_E, pi_E, mu_rel --
is on `mulensevent`.

Per-instance flags (the coordinate choice `fitu0te` and the tie toggle
`star_constrains_rho`) live on each source entry: per-instance flags on the
component that owns the coordinate is the established idiom (planet
`mass_parameterization`, orbit `fitvcve`).
"""

import logging

import numpy as np

from exozippy.components.component import Component
from exozippy.components.parameterization import mode_manifest

from .bodies import body_entries, derive_body_names
from .physics import _MM_NAN_ADVICE, U_0_FLOOR, floor_u_0_value

logger = logging.getLogger(__name__)


class Source(Component):
    """Per-source-body microlensing trajectory parameters.

    YAML shape::

        source:
          - body: star.SourceA
          - body: star.SourceB
            fitu0te: true
            star_constrains_rho: false

    Instances are named after their body star (``source.SourceA.t_0``), the
    Mann/Torres pattern, so the name form folds to index form at
    ConfigManager construction and the borrowed-name display machinery of
    the pre-split lens is gone (closing review 2.1.10).
    """

    @classmethod
    def normalize_config_block(cls, block):
        """Name each instance after its body BEFORE ConfigManager exists --
        the timing is the point (see StellarRelation.normalize_config_block
        for the history)."""
        return derive_body_names(block)

    def __init__(self, config, config_manager):
        # Idempotent with the hook above; kept so a direct instantiation
        # (tests, a standalone driver) still gets named instances.
        self.normalize_config_block(config)
        super().__init__(config, config_manager)
        self.label = "Source Parameters"

        sys_cfg = getattr(config_manager, "system_config", None) or {}
        self.bodies = body_entries(self.config, "source", sys_cfg)
        # Per-instance parameterization flags.
        self._fitu0te = [bool(c.get("fitu0te", False)) for c in self.config]
        self._severed = [
            not bool(c.get("star_constrains_rho", True)) for c in self.config
        ]

    @property
    def prefix(self):
        return "source"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "body",
                "kind": "ref",
                "accepts": ["star"],
                "required": True,
                "doc": (
                    "The source star, as '<component>.<name-or-index>' "
                    "(e.g. 'star.SourceA').  A microlensing source is the "
                    "background star being monitored, so only stars are "
                    "accepted."
                ),
            },
            {
                "key": "fitu0te",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Sample the SIGNED effective timescale u0te = u_0*t_E "
                    "(days) for THIS source and derive u_0 = u0te/t_E. "
                    "Named for the product because t_eff collides with the "
                    "stellar effective temperature in the config namespace; "
                    "the LaTeX symbol stays t_eff. A coordinate choice like "
                    "fitvcve; the 1/t_E Jacobian potential is added in "
                    "build_likelihood. Default False."
                ),
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
                    "to sever the tie for THIS source: rho is sampled "
                    "directly (as log_rho) and the stellar prediction is "
                    "reported as rho_pred, so the pull between the two is "
                    "visible instead of silently arbitrated. Requires "
                    "finite_source (on the mulensevent block). Same "
                    "vocabulary as the planet component's "
                    "beam_constrains_mass and the instrument's "
                    "sed_constrains_blend. A tie is a physics LINK, not a "
                    "one-way assignment: information flows toward "
                    "whichever side is less constrained elsewhere "
                    "(components.md, 'Config flag vocabulary')."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Lifecycle stages
    # ------------------------------------------------------------------

    def build_maps(self):
        """Stage 2: star_map (one star index per source body) and a
        length-n event_map of zeros, so the cross-component deps on the
        one-element mulensevent vectors are PROVABLY aligned (the named map
        keeps the aligned=True fast path; a bare `mulensevent.theta_E`
        would broadcast correctly at length 1 but through the fallback path
        component.py warns about)."""
        self.star_map = np.array([ndx for (_, ndx) in self.bodies], dtype=int)
        self.event_map = np.zeros(self.n_elements, dtype=int)

    def register_parameters(self, system):
        """Stage 3: the per-source manifest, via the mode table."""
        event = getattr(system, "mulensevent", None)
        finite_source = bool(getattr(event, "finite_source", False))

        # A severed rho with no finite-source physics has nothing to sever.
        for j, severed in enumerate(self._severed):
            if severed and not finite_source:
                logger.warning(
                    f"{self.prefix}.{self.names[j]}: star_constrains_rho "
                    f"is false but finite_source is not set on the "
                    f"mulensevent block -- rho never enters the "
                    f"magnification, so there is nothing to sever; "
                    f"ignoring it."
                )

        # Per-instance modes: the (fitu0te, severed-rho) combination.  A
        # mode's table row names the parameters that instance uses; a
        # parameter no instance uses is omitted entirely (mode_manifest).
        modes = []
        for j in range(self.n_elements):
            severed = self._severed[j] and finite_source
            mode = "u0te" if self._fitu0te[j] else "plain"
            if severed:
                mode += "_sev"
            modes.append(mode)

        def row(fitu0te, severed):
            out = {"t_0": None}
            out["u_0"] = "from_u0te" if fitu0te else None
            if fitu0te:
                out["u0te"] = None
            if finite_source:
                if severed:
                    out["rho"] = "from_log_rho"
                    out["log_rho"] = None
                    out["rho_pred"] = "default"
                else:
                    out["rho"] = "default"
            return out

        table = {
            "plain": row(False, False),
            "u0te": row(True, False),
            "plain_sev": row(False, True),
            "u0te_sev": row(True, True),
        }
        # Inactive elements of a mixed vector are pinned at a bookkeeping
        # value that must sit strictly inside the hard bounds (the pinned
        # element still goes through the logit transform; design 2.2).
        # u0te = 0 is interior to [-5000, 5000]; log_rho keeps its
        # defaults.yaml initval (-2, interior) so no override is needed.
        options = {"u0te": {"inactive_value": 0.0}}
        self.manifest = mode_manifest(
            modes, table, options=options, where=self.prefix
        )

        for j, fitu0te in enumerate(self._fitu0te):
            if fitu0te:
                self.config_manager.add_scale_hint(f"source.{j}.u0te", 0.05)

    def _start_values(self, name):
        """Resolved start value of a source parameter as a 1-D float array,
        or None when it has none (unset, or a multi-seed entry that
        survived as a ragged object array)."""
        par = getattr(self, name, None)
        if par is None or par.initval is None:
            return None
        try:
            return np.atleast_1d(np.asarray(par.initval, dtype=float)).ravel()
        except (TypeError, ValueError):
            return None

    def _validate_pspl_start(self):
        """Stage 7: check the START values of the SAMPLED trajectory
        parameters, loudly and once.  Ported from the pre-split
        Lens._validate_pspl_start -- t_0 and u_0 are the two trajectory
        parameters that are sampled here, so their ``initval`` IS the start
        (raw = 0 maps to it through the logit transform); the derived
        event-level chain (t_E, theta_E, pi_E) is deliberately NOT checked
        (a derived parameter's initval is engine bookkeeping, not the value
        the model starts at -- the ob161003 theta_E lesson in that
        method's history)."""
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
        """Stage 7: start-value validation only.  The fitu0te Jacobian
        potential stays on Lens.build_likelihood at stage 1 (the
        build_likelihood splits are stage 2)."""
        self._validate_pspl_start()
