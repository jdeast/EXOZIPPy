import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.components.component import Component
from exozippy.components.relations import (
    StellarRelation,
    as_float_vector,
    constrain_schema_entry,
    star_schema_entry,
)

from . import mist_grid, physics
from .mist_grid import OUTPUT_INDEX
from .plot import MISTPlot

logger = logging.getLogger(__name__)

# The stellar quantities this component may be asked to tie to the MIST
# prediction.  Unlike components/relations.py's CONSTRAINABLE (mass, radius),
# none of these is ever made derived -- every one is ALREADY a free star
# Parameter with its own prior (star.feh, star.radius, star.teff, star.age);
# what an entry in `constrain:` changes is only whether a Gaussian potential
# ties it to the MIST-interpolated value.  Declared as the mixin's
# `constrainable` so `_parse_constrain` and `constrain_schema_entry` are
# shared rather than copied.
CONSTRAINABLE = ("feh", "radius", "teff", "age")

# The star parameters this component OWNS in the sense that nothing else
# reads them: they exist only to index a track.  A star that no instance
# names gets all three pinned -- see `_pin_unmodeled_stars`.
TRACK_PARAMS = ("initfeh", "eep", "age")

# Bounds on |dEEP/dage| (per yr) for the EEP -> age Jacobian potential.
#
# The UPPER clip is the one that matters and the reason this is a clip rather
# than a floor.  The potential is -log|dEEP/dage|, so a small derivative is a
# large REWARD: the 1e-30 "floor" this replaced was worth +69 nats, an
# unbounded attractor rather than a safety rail, and it is reachable because
# the tabulated derivative crosses zero between the +/-2**32 entries on the
# non-monotonic-age tracks (see mist_grid's "Sentinels" section).  Compare
# calc_jitter's radicand floor, which caps an infinite GRADIENT -- the
# opposite problem, and why that precedent does not transfer.
#
# The window is set from the shipped grid: real values run from ~1e-14 /yr
# (a slowly evolving low-mass main sequence) to ~1e2 /yr (EEP 1 of a 110
# solMass track, whose age there really is ~4e-3 yr).  Anything outside it is
# a defect of the table, so both clips act only where the table is wrong.
DEEP_DAGE_MIN = 1e-14
DEEP_DAGE_MAX = 1e3


class EvolutionaryModel(StellarRelation, Component):
    """Tie a star's feh/radius/teff/age to the MIST evolutionary tracks.

    One instance per constrained star::

        evolutionarymodel:
          - star: "A"
            constrain: [feh, radius, teff, age]   # default: all four

    Design::

        MCMC steps in:  star.logmass, star.initfeh, star.eep,
                        star.feh, star.radius, star.teff, star.age
        Interpolated at (star.logmass, star.initfeh, star.eep) over the MIST
        grid:           feh_mist, radius_mist, teff_mist, age_mist
        Penalty:        Gaussian potential between each sampled quantity and
                        its MIST-predicted counterpart, floored at a
                        mass-dependent systematic (EXOFASTv2's
                        massradius_mist.pro; see physics.py), so the two
                        agree only up to that floor rather than being locked
                        together.
        Jacobian:       a potential in log|dage/dEEP| that turns the flat
                        prior EEP gets for free (an ordinary bounded free
                        Parameter) into a flat prior in AGE -- EEP steps are
                        not uniform in time.  See build_likelihood.
        Dragons:        `here_be_dragons`, interpolated the same way as the
                        four predicted quantities, feeds a smooth penalty
                        that grows with its value, so the unreliable tail of
                        a track is costly rather than forbidden and the
                        interpolator keeps working across the boundary.

    **This component declares no parameters of its own.**  ``initfeh`` and
    ``eep`` are ``star`` parameters (``star/defaults.yaml``), materialized by
    the ``in_system("evolutionarymodel")`` branch of
    ``Star.register_parameters`` -- which exists precisely so landing this
    component needs no star-side edit -- alongside ``star.age``.  Declaring a
    second ``evolutionarymodel.<name>.initfeh``/``.eep`` pair (what an earlier
    draft did) would leave the star's copies materialized, free, bounded and
    read by nothing: two unconstrained nuisance dimensions per star, plus a
    second spelling for every user entry.

    The star/constrain/masked-penalty wiring comes from the
    ``StellarRelation`` mixin, exactly as in ``components/mann`` and
    ``components/torres``; only :data:`CONSTRAINABLE` differs, and it is
    declared through the mixin's ``constrainable`` hook.
    """

    yaml_key = "evolutionarymodel"

    # See CONSTRAINABLE above: this is the mixin's hook for a component whose
    # constrainable set is not (mass, radius).
    constrainable = CONSTRAINABLE

    @property
    def prefix(self):
        return "evolutionarymodel"

    @classmethod
    def config_schema(cls):
        return [
            star_schema_entry("MIST evolutionary-track"),
            constrain_schema_entry(
                "the MIST tracks", options=list(CONSTRAINABLE)
            ),
            {
                "key": "model",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "MIST grid version directory name. Default 'MISTv2.5'.",
            },
            {
                "key": "alpha",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "[alpha/Fe] grid slice. Only 0.0 ships in this release.",
            },
            {
                "key": "vvcrit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "v/v_crit grid slice. Only 0.0 ships in this release.",
            },
            {
                "key": "model_root",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "Directory holding the models/ tree. Defaults to the packaged root.",
            },
            {
                "key": "feh_floor",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Systematic [Fe/H] error floor (absolute, dex). Default: "
                    "the mass-dependent EXOFASTv2 percenterror (physics.py)."
                ),
            },
            {
                "key": "radius_floor",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Systematic radius error floor (fraction of the MIST "
                    "prediction). Default: the mass-dependent percenterror."
                ),
            },
            {
                "key": "teff_floor",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Systematic Teff error floor (fraction of the MIST "
                    "prediction). Default: the mass-dependent percenterror."
                ),
            },
            {
                "key": "age_floor",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Systematic age error floor (fraction of the MIST "
                    "prediction). Default: the mass-dependent percenterror."
                ),
            },
            {
                "key": "dragon_penalty_weight",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Nats of penalty per unit of the interpolated "
                    "here_be_dragons value. Default 1.0."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Stage 1a
    # ------------------------------------------------------------------
    def load_data(self, system):
        self.star_indices = []
        self.constrain = []
        self.feh_floor = []
        self.radius_floor = []
        self.teff_floor = []
        self.age_floor = []
        self.dragon_penalty_weight = []
        self._grids = []

        for i, (c, nm) in enumerate(zip(self.config, self.names)):
            star_idx = self._resolve_star(system, nm, c.get("star"))
            self.star_indices.append(star_idx)
            self.constrain.append(
                self._parse_constrain(nm, c.get("constrain"))
            )
            self._check_star_opted_in(system, nm, star_idx)

            self.feh_floor.append(c.get("feh_floor"))  # None -> mass-dependent
            self.radius_floor.append(c.get("radius_floor"))
            self.teff_floor.append(c.get("teff_floor"))
            self.age_floor.append(c.get("age_floor"))
            self.dragon_penalty_weight.append(
                float(c.get("dragon_penalty_weight", 1.0))
            )

            # `model_root:` present-but-empty is how YAML spells a key with
            # no value, and it means "use the default" -- `.get(k, default)`
            # would hand the None straight to Path() and die there, several
            # frames down, on a message naming neither the key nor the block.
            model_root = c.get("model_root")
            if model_root is None:
                model_root = mist_grid.DEFAULT_MIST_MODEL_ROOT

            grid = mist_grid.load_mist_grid(
                model=c.get("model", "MISTv2.5"),
                alpha=float(c.get("alpha", 0.0)),
                vvcrit=float(c.get("vvcrit", 0.0)),
                model_root=model_root,
            )
            self._grids.append(grid)

            # for use in prose later
            self.model = c.get("model", "MISTv2.5")
            self.model_root = model_root

            import yaml

            model_yaml_file = (
                f"{self.model_root}/{self.model}/EEPs/{self.model}.grid.yaml"
            )
            with open(model_yaml_file, "r") as f:
                self._model_yaml = yaml.safe_load(f)

            star_name = system.star.names[star_idx]
            self._inject_grid_bounds(star_name, grid)
            self._seed_eep_hint(i, star_name, star_idx, grid)

        if not self.n_elements:
            # An `evolutionarymodel:` block that names no star. The topology
            # key alone is enough for Star.register_parameters to materialize
            # initfeh/eep/age on EVERY star, so this is not a no-op we can
            # ignore: warn, and let _pin_unmodeled_stars below pin all of
            # them, which is the same rule a partially-covered system gets.
            logger.warning(
                f"[{self.prefix}] the block names no star, so no evolutionary "
                f"model is applied. Any star still opted in via 'mist:' "
                f"(which DEFAULTS to True) has its initfeh/eep/age "
                f"materialized by the block's mere presence, and those are "
                f"pinned. Add an entry like '{self.prefix}: [{{star: A}}]', "
                f"or remove the block."
            )

        self._pin_unmodeled_stars(system)

    def _check_star_opted_in(self, system, nm, star_idx):
        """Reject a block naming a star that opted out, or asked for PARSEC.

        ``star.mist:`` (default True) and ``star.parsec:`` (default False) are
        the per-star model switches; ``Star.register_parameters`` gates
        ``initfeh``/``eep``/``age`` on their union.  Only MIST is implemented,
        so a star that asked for PARSEC alone would silently be handed MIST
        tracks, and a star that turned every model off would be constrained by
        one anyway.  Both raise rather than warn: each is a config that says
        one thing and would do another.
        """
        star_name = system.star.names[star_idx]
        mist = bool(system.star.mist[star_idx])
        parsec = bool(system.star.parsec[star_idx])
        if not mist and parsec:
            raise ValueError(
                f"{self.prefix} '{nm}': star '{star_name}' asks for "
                f"'parsec: True, mist: False', but only the MIST tracks are "
                f"implemented -- this block would silently interpolate MIST. "
                f"Set 'mist: True' on the star, or remove this "
                f"{self.prefix} block."
            )
        if not mist and not parsec:
            raise ValueError(
                f"{self.prefix} '{nm}': star '{star_name}' has 'mist: False' "
                f"(and no 'parsec:'), i.e. it opted out of every evolutionary "
                f"model, but this block constrains it with one. Remove the "
                f"block, or drop 'mist: False' from the star."
            )
        if mist and parsec:
            # A warning, not a raise: `mist:` defaults to True, so this is
            # most often a user who added `parsec: True` and did not realize
            # they had to turn MIST off -- and unlike the case above there IS
            # a defensible reading of it (use whichever model is available).
            logger.warning(
                f"[{self.prefix}] '{nm}': star '{star_name}' sets both "
                f"'mist' and 'parsec'; only the MIST tracks are implemented, "
                f"so PARSEC is ignored."
            )

    def _inject_grid_bounds(self, star_name, grid):
        """Tighten this star's logmass/initfeh/eep bounds to the grid extent.

        A validity limit, not a preference (mirrors ``SED._inject_grid_bounds``,
        whose reasoning is written out at length there):
        ``RegularGridInterpolator`` is built with ``fill_value=None`` and so
        linearly EXTRAPOLATES past its axis edges, meaning a star that wanders
        outside the grid gets a meaningless, unboundedly growing prediction
        rather than a clean failure.

        Registered through ``ConfigManager.add_override`` -- the cross-component
        spelling of the manifest ``"overrides"`` channel -- and NOT by writing
        into ``config_manager.user_params``.  A user_params entry is
        indistinguishable from the user's own, so the provenance ledger,
        ``export_solution``, ``initval_source`` and the GUI would all report a
        bound the user never wrote, and ``finalize_user_params`` would register
        the path as a leaf symbol in the relaxation engine.  It is also
        functionally different: ``add_override`` goes through ``apply_value``,
        which combines competing bounds as ``max(lower)``/``min(upper)``
        order-independently and logs when it clips a user's, whereas the
        ``setdefault`` this replaced saw the user's key and never applied the
        grid limit at all.

        Note the initfeh range is the range of the TRIMMED grid (-2.5 dex for
        the shipped one, not defaults.yaml's -4.0) -- see
        ``mist_grid.trim_to_complete_grid``.
        """
        axes = {
            "logmass": grid["logmass_pts"],
            "initfeh": grid["initfeh_pts"],
            "eep": grid["eep_pts"],
        }
        for param, pts in axes.items():
            self.config_manager.add_override(
                f"star.{star_name}.{param}",
                lower=float(np.min(pts)),
                upper=float(np.max(pts)),
            )

    def _pin_unmodeled_stars(self, system):
        """Pin initfeh/eep/age on every star no instance of this component names.

        ``Star.register_parameters`` masks all three by ``mist or parsec``,
        and that mask IS consumed now (it is a real per-element role: an
        opted-out star's track coordinates are inactive).  But ``mist:``
        DEFAULTS TO TRUE, so every star in a config carrying an
        ``evolutionarymodel`` block is opted in by that default -- whether or
        not any instance of this component names it.  The mask answers "did
        this star ask for a track"; this answers "did anything actually give
        it one".  Left alone, a two-star system with one modeled star gives
        the other three free, bounded parameters that no likelihood term
        reads.

        The fix is ``Band._pin_unread_limb_darkening``'s: ``sigma: 0`` through
        the override channel, which layers UNDER the params file, so an
        explicit user entry still frees the element.  Each has a defaults.yaml
        ``initval``, which is what the "a pin must say what it pins to" rule in
        ``build_pymc`` requires.
        """
        modeled = set(self.star_indices)
        for si, star_name in enumerate(system.star.names):
            if si in modeled:
                continue
            for param in TRACK_PARAMS:
                self.config_manager.add_override(
                    f"star.{star_name}.{param}", sigma=0.0
                )
            if system.star.mist[si] or system.star.parsec[si]:
                logger.info(
                    f"[{self.prefix}] star '{star_name}' has an evolutionary "
                    f"model switch set but no {self.prefix} block names it, "
                    f"so nothing reads its initfeh/eep/age; pinning all three. "
                    f"Add an '{self.prefix}: [{{star: {star_name}}}]' entry to "
                    f"put it on a track."
                )

    def _seed_eep_hint(self, i, star_name, star_idx, grid):
        """Data-driven initval guess for this star's eep (RANK_DERIVED_DATA).

        Walks the grid track INTERPOLATED at the star's current (logmass,
        initfeh) initvals and hints the EEP minimizing the same objective
        ``build_likelihood`` adds -- the constrained subset of the four
        systematic-floored chi2 terms, plus the dragon penalty, so the seed
        cannot land in the unreliable tail of a track.  That is strictly
        better than a nearest-age lookup: the fit is being started at whatever
        (teff, radius, feh, age) the user or the defaults supplied, and only
        the age term knows about the age.

        Purely a starting point: ``config_manager.add_hint`` ranks below an
        explicit user initval (RANK_USER), so this never overrides one -- see
        config.py's provenance ranking.

        ``resolve()`` is called with ``shape=()`` (a single coarse scalar, not
        the full per-star vector) but with ``element=`` set so that scalar
        targets THIS star rather than always index 0 -- see
        ``ConfigManager.resolve``'s docstring on ``element``, and
        ``Instrument._register_astrometric_target``'s ``resolve("star", "ra",
        element=star_ndx)`` for the identical pattern.  The values it returns
        are already in INTERNAL units, which for every parameter read here is
        the unit the grid is tabulated in.
        """

        # grab the star's current logmass/initfeh initvals,
        # or None if either is missing or non-finite
        def start(param):
            cfg = self.config_manager.resolve("star", param, element=star_idx)
            val = cfg.get("initval")
            return None if val is None else float(np.atleast_1d(val)[0])

        logmass0 = start("logmass")
        initfeh0 = start("initfeh")
        if (
            logmass0 is None
            or initfeh0 is None
            or not np.isfinite([logmass0, initfeh0]).all()
        ):
            return

        # Bilinear in (logmass, initfeh) rather than snapped to the nearest
        # tabulated track: neighbouring tracks differ everywhere, most of all
        # in age, so a nearest-point search can seed the fit on a track up to
        # half a cell away in both axes. This blends the same four corners
        # the likelihood's own interpolator will.
        track = mist_grid.interpolate_track(grid, logmass0, initfeh0)

        # The same mass-dependent floor build_likelihood uses, in numpy and
        # straight from log10(mass) -- see physics.percent_error_from_logmass.
        percent = float(physics.percent_error_from_logmass(logmass0))
        # keys should correspond to CONSTRAINABLE, but the dicts are keyed on the string
        # literals used in the config, not the CONSTRAINABLE tuple, so that the dicts
        # can be indexed by `which` in the loop below.
        observed = {
            "feh": start("feh"),
            "radius": start("radius"),
            "teff": start("teff"),
            "age": start("age"),
        }
        floors = {
            "feh": self.feh_floor[i],
            "radius": self.radius_floor[i],
            "teff": self.teff_floor[i],
            "age": self.age_floor[i],
        }
        columns = {
            "feh": "feh_mist",
            "radius": "radius_mist",
            "teff": "teff_mist",
            "age": "age_mist_gyr",
        }

        chi2 = np.zeros(track.shape[0], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for which in self.constrain[i]:
                obs = observed[which]
                if obs is None or not np.isfinite(obs):
                    continue
                pred = track[:, OUTPUT_INDEX[columns[which]]]
                frac = (
                    percent if floors[which] is None else float(floors[which])
                )
                # feh's floor is absolute dex; the others scale the prediction (see physics.py)
                sigma = frac if which == "feh" else frac * np.abs(pred)
                chi2 += np.nan_to_num(
                    ((obs - pred) / sigma) ** 2, nan=np.inf, posinf=np.inf
                )
        chi2 += (
            2.0
            * self.dragon_penalty_weight[i]
            * np.maximum(track[:, OUTPUT_INDEX["here_be_dragons"]], 0.0)
        )
        # Penalize pre-main-sequence rows so the seed lands on the main
        # sequence.  The EEP axis is NOT one of the interpolator's output
        # columns (OUTPUT_COLUMNS is the six interpolated quantities) -- it is
        # the grid axis the track's rows are indexed by, so it comes from
        # grid["eep_pts"], which interpolate_track leaves untouched and so is
        # row-aligned with `track` by construction.
        chi2 += np.where(
            np.asarray(grid["eep_pts"], dtype=float)
            < MISTPlot.KIEL_EEP_WINDOW[0],
            30.0,
            0.0,
        )
        if not np.isfinite(chi2).any():
            return
        eep0 = float(grid["eep_pts"][int(np.nanargmin(chi2))])
        # Keyed on the STAR's name, not this instance's: the two coincide by
        # default (StellarRelation names an instance after its star) but an
        # explicit `name:` on the block breaks that, and the hint has to land
        # on the star's parameter either way.
        self.config_manager.add_hint(f"star.{star_name}.eep", eep0)

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def register_parameters(self, system):
        """Stage 2: nothing to declare.

        Every input and every constrained quantity is already a ``star``
        parameter; this component contributes only potentials.  The grid's
        axis extents reach ``star.logmass``/``initfeh``/``eep`` through
        ``ConfigManager.add_override`` in ``load_data`` (see
        ``_inject_grid_bounds``), which is the channel for a bound on a
        parameter another component owns.
        """
        self.manifest = {}

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------
    def _interpolate(self, system):
        """(n_elements, n_outputs) of MIST predictions, one row per instance.

        Instances are grouped by the grid object they query before evaluating:
        ``mist_grid.load_mist_grid`` is cached, so every instance naming the
        same (model, alpha, vvcrit) shares one dict, and the interpolator
        takes a whole ``(ntest, ndim)`` block at once.  A per-instance loop
        would build n_elements separate interpolation subgraphs against the
        same ~86 MB constant.
        """
        star = system.star
        smap = self.star_map_tensor
        coords_by_param = (
            star.logmass.value[smap],  # log10(solMass) -- the grid's mass axis
            star.initfeh.value[smap],
            star.eep.value[smap],
        )

        groups = {}
        for i, grid in enumerate(self._grids):
            groups.setdefault(id(grid), (grid, []))[1].append(i)

        rows = [None] * self.n_elements
        for grid, idxs in groups.values():
            sel = np.asarray(idxs, dtype=int)
            coords = pt.stack([c[sel] for c in coords_by_param], axis=-1)
            block = grid["interpolator"].evaluate(coords)  # (len(sel), n_out)
            for j, i in enumerate(idxs):
                rows[i] = block[j]
        return pt.stack(rows)

    def _floor_vector(self, user_floors, default_tensor):
        """Per-instance systematic floor: the user's, else the default."""
        given = np.array(
            [np.nan if f is None else float(f) for f in user_floors],
            dtype=float,
        )
        return pt.where(
            pt.as_tensor_variable(np.isnan(given)),
            default_tensor,
            as_float_vector(np.nan_to_num(given, nan=0.0)),
        )

    def _warn_outside_grid(self, system):
        """Warn per instance whose star STARTS off the grid it will query.

        The bounds injected in ``load_data`` already keep the sampler inside
        the grid, so this fires only when the START is outside them -- which
        ``build_pymc`` turns into a hard error, and this warning names the
        component responsible for the bound before that happens.  Per-instance
        ranges because two instances may name different grids.
        """
        for param, key, unit in (
            ("logmass", "logmass_pts", " dex(solMass)"),
            ("initfeh", "initfeh_pts", " dex"),
            ("eep", "eep_pts", ""),
        ):
            self._warn_outside_range(
                system,
                getattr(system.star, param),
                [float(np.min(g[key])) for g in self._grids],
                [float(np.max(g[key])) for g in self._grids],
                message=(
                    "star '{star}' starts at {value:.4g}" + unit + ", "
                    f"outside this instance's MIST grid's {param} axis "
                    "([{low:.4g}, {high:.4g}]). That extent is a hard bound "
                    f"on star.{param} (the interpolator extrapolates "
                    "meaninglessly past it), so the start has to move."
                ),
            )

    def build_likelihood(self, model, system):
        if not self.n_elements:
            # Nothing to interpolate; load_data has already warned and pinned.
            # Guarded here rather than left to `pt.stack([])`, whose "No
            # tensor arguments provided." names neither this component nor
            # the empty block that caused it.
            return

        star = system.star
        smap = self.star_map_tensor

        # Deferred to stage 6: the relaxation engine (stage 3) has run by now,
        # so these initvals are the ones the sampler will actually start from.
        self._warn_outside_grid(system)

        pred = self._interpolate(system)

        feh_pred = pred[:, OUTPUT_INDEX["feh_mist"]]
        radius_pred = pred[:, OUTPUT_INDEX["radius_mist"]]
        teff_pred = pred[:, OUTPUT_INDEX["teff_mist"]]
        age_pred = pred[:, OUTPUT_INDEX["age_mist_gyr"]]
        deep_dage = pred[:, OUTPUT_INDEX["dEEP_dage"]]
        dragons = pred[:, OUTPUT_INDEX["here_be_dragons"]]

        pm.Deterministic(f"{self.prefix}.feh_pred", feh_pred)
        pm.Deterministic(f"{self.prefix}.radius_pred", radius_pred)
        pm.Deterministic(f"{self.prefix}.teff_pred", teff_pred)
        pm.Deterministic(f"{self.prefix}.age_pred", age_pred)
        pm.Deterministic(f"{self.prefix}.here_be_dragons", dragons)

        # ---- systematic floors (EXOFASTv2 massradius_mist.pro) -----------
        # Mass-dependent default, per element; a per-instance user override
        # (feh_floor:/radius_floor:/teff_floor:/age_floor:) replaces it for
        # that element only. teff/radius/age floors are FRACTIONS of the
        # MIST prediction; feh's is an ABSOLUTE dex floor (feh can be zero
        # or negative, so "fraction of the prediction" is meaningless) --
        # exactly EXOFASTv2's own convention (see physics.py).
        mstar = star.mass.value[smap]
        percent_floor = physics.calc_mist_percent_error(mstar)

        sigma_feh = self._floor_vector(self.feh_floor, percent_floor)
        sigma_radius = radius_pred * self._floor_vector(
            self.radius_floor, percent_floor
        )
        sigma_teff = teff_pred * self._floor_vector(
            self.teff_floor, percent_floor
        )
        sigma_age = age_pred * self._floor_vector(
            self.age_floor, percent_floor
        )

        # ---- the four MIST penalties --------------------------------------
        # normalize=True throughout: every sigma above is a function of
        # sampled parameters (mstar, or the MIST prediction itself), so the
        # -log(sigma) normalization is not a constant and dropping it would
        # leave the posterior subtly improper (components/mann's reasoning
        # for its own prediction-proportional sigma; components/torres's
        # normalize=False does NOT apply here -- its sigma is a fixed dex
        # constant, ours never is).  EXOFASTv2 accumulates chi2 only, so this
        # is a deliberate departure from massradius_mist.pro, not a port bug.
        penalties = {
            "feh": (star.feh.value[smap], feh_pred, sigma_feh),
            "radius": (star.radius.value[smap], radius_pred, sigma_radius),
            "teff": (star.teff.value[smap], teff_pred, sigma_teff),
            "age": (star.age.value[smap], age_pred, sigma_age),
        }
        for which in CONSTRAINABLE:
            observed, predicted, sigma = penalties[which]
            self._add_penalty(
                which, observed, predicted, sigma, normalize=True
            )

        # ---- EEP -> age Jacobian: turn the free flat-in-EEP prior into a --
        # ---- flat-in-AGE prior --------------------------------------------
        # eep is an ordinary bounded free Parameter, so it already carries a
        # flat prior in EEP for free (Parameter's logit-uniform transform).
        # EEP is not linear in age (see dEEP_dage on the grid), so a flat
        # prior in EEP is NOT a flat prior in age -- the two disagree by the
        # Jacobian |dage/dEEP|. Deriving the needed correction:
        #
        #   p_EEP(e) = p_age(a(e)) * |da/de|          (change of variables)
        #
        # p_EEP(e) is currently flat (const); we want the EFFECTIVE density
        # (after any extra potential) to behave as p_age = const instead.
        # Multiplying the total density by a correction C(e) such that
        #   p_EEP(e) * C(e) = const  =>  C(e) = |da/de|
        # so the ADDITIVE log-potential is log|da/de| = -log|de/da|.
        # dEEP_dage on the grid IS de/da (in 1/yr); the yr->Gyr rescale of
        # 'a' (star.age is sampled in Gyr) only shifts this by an additive
        # CONSTANT (log of the fixed 1e9 yr/Gyr factor), which is dropped
        # here exactly as components/torres drops its own constant -log(sigma)
        # term -- it cannot move the posterior.
        #
        # This is EXOFASTv2's `ageweight`, which is da/dEEP entering its chi2
        # as `chi2 -= 2*alog(ageweight)`, i.e. logp += log|da/de|. Same sign.
        #
        # Sanity check: on the slow-evolving main sequence, a star spends a
        # long TIME per EEP step, so de/da is SMALL there -> -log(small) is
        # a LARGE potential -> EEP draws landing on the main sequence are
        # favored, exactly matching a flat-in-age belief (most of a star's
        # age budget IS spent on the main sequence). Fast phases (de/da
        # large) are correspondingly disfavored.
        #
        # The clip is two-sided and it is the UPPER end that is load-bearing;
        # see DEEP_DAGE_MIN/MAX at the top of this module for why a bare
        # floor here would be an unbounded reward rather than a safety rail.
        jacobian_logp = -pt.sum(
            pt.log(pt.clip(pt.abs(deep_dage), DEEP_DAGE_MIN, DEEP_DAGE_MAX))
        )
        pm.Potential(f"{self.prefix}.eep_age_jacobian", jacobian_logp)

        # ---- here_be_dragons penalty ---------------------------------------
        # A smooth, monotonically-growing cost for wandering into the part
        # of a track past its first unreliable point. here_be_dragons is
        # interpolated exactly like the four predicted quantities above, so
        # the penalty is continuous and differentiable right up to (and
        # past) the boundary rather than a hard cutoff; every dragon-zone
        # grid point stays in the interpolation domain.
        #
        # This weight is NOT what keeps a fit away from the hydrogen-
        # exhausted end of a massive star's track. There feh_mist is 30.0
        # (a real value -- see mist_grid's docstring), so the feh penalty
        # alone is ~5e5 nats against a solar-metallicity star and swamps
        # anything this term contributes. The dragon penalty earns its keep
        # on the OTHER dragon rows, the ~78% whose feh_mist is perfectly
        # ordinary and which nothing else would discourage.
        dragon_weight = as_float_vector(self.dragon_penalty_weight)
        dragon_logp = -pt.sum(dragon_weight * pt.maximum(dragons, 0.0))
        pm.Potential(f"{self.prefix}.dragon_penalty", dragon_logp)

        self._declare_priors(system)
        self._add_prose(system)

    # ------------------------------------------------------------------
    # Reporting: what this component adds to the tables and the draft
    # ------------------------------------------------------------------
    def _declare_priors(self, system):
        """Declare every potential above to the Prior column.

        ``get_prior_str`` can only see a Parameter's own fields, so a
        ``pm.Potential`` a component adds at stage 6 is invisible to it and
        the parameter reports whatever those fields imply -- "Uniform" for a
        bounded element with no sigma, which is exactly the prior these terms
        replace.  Declared per element, because ``constrain:`` is per
        instance and a star no instance names gets none of them.
        """
        star = system.star
        by_quantity = {
            "feh": star.feh,
            "radius": star.radius,
            "teff": star.teff,
            "age": star.age,
        }
        for which, param in by_quantity.items():
            elements = [
                self.star_indices[i]
                for i in range(self.n_elements)
                if which in self.constrain[i]
            ]
            if not elements:
                continue
            param.add_prior_contribution(
                r"$\mathcal{N}(\mathrm{MIST}, \sigma_{\rm sys})$",
                text="Normal(MIST, sigma_sys)",
                elements=elements,
            )

        # The Jacobian REPLACES eep's implicit uniform-over-bounds prior: that
        # is the entire point of the term, so supersedes_bounds=True and the
        # rendered text keeps the interval it is a density over.
        star.eep.add_prior_contribution(
            r"$\propto |{\rm d}\,{\rm Age}/{\rm d}\,{\rm EEP}|$",
            text="propto |dAge/dEEP| (uniform in Age)",
            elements=list(self.star_indices),
            supersedes_bounds=True,
        )

    # ------------------------------------------------------------------
    # Plotting: the Kiel diagram (logg vs Teff)
    #
    # Described ONCE as Charts, per the plotrender contract: matplotlib
    # draws them for the saved PDFs and the GUI draws the same specs with
    # plotly, so there is no second hand-drawn copy to keep in sync.
    # ------------------------------------------------------------------
    def compile_plotters(self, model, system):
        """Compile the per-instance Kiel-diagram quantities.

        One node, ``(n_elements, 9)``, so a posterior draw costs one call:

            logmass, initfeh   the track coordinates -- returned numerically
                               so plot_data can build the whole 807-point
                               curve in numpy (mist_grid.interpolate_track)
                               rather than unrolling it into the graph
            teff_mist, logg_mist    the MIST prediction at (logmass, initfeh, eep)
            teff_fit, logg_fit      what the fit actually settled on
            sigma_teff, sigma_logg  the systematic floor, as error bars
            eep                the fitted EEP, which selects how much of the
                               track to draw (see KIEL_EEP_WINDOW)

        ``logg`` is neither a grid column nor a constrained quantity: MIST
        tabulates radius, and logg follows from it and the mass, which along
        a track is the grid's own mass axis (the "current mass == initial
        mass" assumption every quantity here is computed under).  So both
        loggs come from ``calc_logg_from_logmass`` and differ only in which
        radius they take -- which is exactly the comparison the chart is for.
        """
        import pytensor

        from ..star.physics import calc_logg_from_logmass

        self._compiled_kiel = None
        self._kiel_node = None
        # Invalidated with the graph it is computed from: a rebuild may have
        # moved every parameter, and a stale cache would pin the reported
        # point to the previous model's posterior.
        self._reported_kiel_cache = None
        if not self.n_elements:
            # An `evolutionarymodel:` block naming no star. Same guard as
            # build_likelihood's, and needed separately because System calls
            # compile_plotters too -- `pt.stack([])` raises "No tensor
            # arguments provided.", naming neither this component nor the
            # empty block behind it.
            return

        star = system.star
        smap = self.star_map_tensor

        pred = self._interpolate(system)
        teff_pred = pred[:, OUTPUT_INDEX["teff_mist"]]
        radius_pred = pred[:, OUTPUT_INDEX["radius_mist"]]

        logmass = star.logmass.value[smap]
        percent_floor = physics.calc_mist_percent_error(star.mass.value[smap])
        f_teff = self._floor_vector(self.teff_floor, percent_floor)
        f_radius = self._floor_vector(self.radius_floor, percent_floor)

        # logg = C + logmass - 2*log10(R), so a FRACTIONAL radius floor
        # becomes an ABSOLUTE logg floor of 2*f_R/ln(10). The mass carries no
        # floor -- it is not one of the constrained quantities -- so it
        # contributes nothing to this error bar.
        sigma_logg = (2.0 / np.log(10.0)) * f_radius

        self._kiel_node = pt.stack(
            [
                logmass,
                star.initfeh.value[smap],
                teff_pred,
                calc_logg_from_logmass(logmass, radius_pred),
                star.teff.value[smap],
                calc_logg_from_logmass(logmass, star.radius.value[smap]),
                teff_pred * f_teff,
                sigma_logg,
                star.eep.value[smap],
            ],
            axis=-1,
        )

        try:  # noqa: SIM105 - the except body logs
            self._compiled_kiel = pytensor.function(
                inputs=[p.value for p in system.plot_params],
                outputs=self._kiel_node,
                on_unused_input="ignore",
            )
        except Exception as exc:  # noqa: BLE001 - a plot must not kill a fit
            logger.warning(
                f"[{self.prefix}] Kiel-diagram plotter failed to compile "
                f"({exc}); the evolutionary-model chart will be skipped."
            )
            self._compiled_kiel = None

    def plot_data(self, system, point=None):
        """The Kiel Chart of every star at ``point`` (``[]`` without one).

        This is the ONE description of the chart, and it is what the three
        consumers walk: ``plot`` below (the PDFs), the GUI's Tune tab and
        live evaluator, and ``outputs/modeling.collect_figures``, which
        pairs each Chart's caption with its PDF.  It was deleted in the
        2026-09 MIST rewrite (commit 9b3cb992), which is why the Kiel PDF
        was written every fit yet never appeared in the paper draft and the
        GUI had no Kiel diagram (review 1.8.7): both consumers ask through
        this method and fell back to the base class's empty list.
        """
        return MISTPlot(system, [point]).kiel_specs(point)

    def plot(self, system, points, filename_prefix="debug"):
        """Kiel PDF(s) through the shared renderer, plus the posterior contours.

        Not the ``plot_via_specs`` one-liner, for one reason: the Teff axis
        of a spaghetti figure is the union over draws (``MISTPlot.
        kiel_spec_groups``), which the generic per-point loop cannot know.
        The rendering itself is ``plotrender.render_spec_groups`` -- there
        used to be a private copy here differing only in legend handling,
        which now lives upstream (review 4.11.7).  The contour plot is a
        bespoke posterior-only matplotlib diagnostic, like a corner plot,
        and is not a Chart, so it does not reach the GUI or the draft.
        """
        from exozippy.plotrender import render_spec_groups

        plotter = MISTPlot(system, points)
        render_spec_groups(plotter.kiel_spec_groups(), filename_prefix)
        if plotter._posteriorBool:
            values = plotter._get_posterior_compiled_values()
            for star_idx in range(self.n_elements):
                plotter.plot_contours(
                    values, star_idx, filename_prefix=filename_prefix
                )

    def _add_prose(self, system):
        """Declare the modeling-draft sentences next to the terms they describe.

        Not ``StellarRelation._add_relation_prose``: that sentence is built
        around "the empirical relations of <citation>", and this is a track
        interpolation with a systematic floor and a reparameterized age prior
        -- three facts, not one.  The declare-at-the-implementation-site rule
        is the same (outputs/prose.py).
        """
        from ...outputs.prose import get_collector, join_names
        from ...outputs.texutils import latex_escape

        prose = get_collector(system)
        stars = [system.star.names[si] for si in self.star_indices]
        noun = "star" if len(stars) == 1 else "stars"
        names = join_names(latex_escape(s) for s in stars)
        self.citation = self._model_yaml.get("citation", "")

        prose.add(
            f"We modeled the {noun} {names} with the {self.model} evolutionary "
            rf"tracks \citep{{{self.citation}}}, interpolating the grid "
            r"in initial mass, initial [Fe/H], and equivalent "
            r"evolutionary point (EEP) to predict the current "
            r"[Fe/H], radius, effective temperature, and age. ",
            section="stellar",
            key=f"{self.prefix}.tracks",
            rank=20.0,
        )
        prose.add(
            "By default, each predicted quantity is tied to its sampled counterpart by "
            "a Gaussian penalty whose width is the mass-dependent systematic "
            r"floor described in Section 2.1 of \citet{Eastman:2019}:"
            r"\begin{equation} "
            r"\sigma_{\rm MIST} = 0.03 - 0.025 \log{M_\star} + 0.045(\log{M_\star})^2"
            r"\end{equation} "
            # `\%`, not `%`: a raw percent sign is a LaTeX comment and
            # swallowed the rest of this paragraph; and `\odot` in a raw
            # string takes ONE backslash (`\\odot` is a line break + "odot").
            r"This equation results in fractional errors of about 10\% at 0.1 $M_\odot$, "
            r"3\% at 1 $M_\odot$, and 5\% at 10 $M_\odot$. "
            f"Thus, the fitted values for the {noun} are required to "
            "agree with the tracks only to within the models' own accuracy. ",
            section="stellar",
            key=f"{self.prefix}.floor",
            rank=21.0,
        )
        prose.add(
            "Because EEP is not uniformly distributed in time, we added the "
            r"$\log|{\rm d}\,{\rm Age}/{\rm d}\,{\rm EEP}|$ Jacobian to the "
            "log-likelihood, which makes the uniform prior on EEP a uniform "
            "prior on stellar age. ",
            section="priors",
            key=f"{self.prefix}.eep_jacobian",
            rank=30.0,
        )
