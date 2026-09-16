import logging

import numpy as np
import pymc as pm
import pytensor

logger = logging.getLogger(__name__)
import pytensor.tensor as pt

from exozippy.components.instrument import Instrument
from exozippy.components.limbdark import quad_limb_darkened_flux
from exozippy.outputs.prose import get_collector
from exozippy.outputs.texutils import latex_escape

from .. import ltt
from ..orbit import physics as orbit_physics
from . import physics


class Transit(Instrument):
    prose_noun = "transit photometry"

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Transit Parameters"
        # Filter identity and limb darkening live on the Band component;
        # each instrument references a band block by name.
        self.band_names = [c.get("band") for c in self.config]
        deprecated_filters = [
            c.get("filter") for c in self.config if c.get("filter")
        ]
        if deprecated_filters:
            logger.warning(
                "transit 'filter:' is deprecated and ignored; reference a "
                "band: block instead (bands carry the filter identity and "
                "limb darkening)."
            )
        self.total_detrend_cols = 0
        # SED depth-dilution node, built once by build_likelihood and
        # reused by compile_plotters.
        self._dilution_node = None
        # Light-travel-time (Roemer delay) correction, per file (see
        # components/ltt.py) -- on by default (Jason's decision: transit/rm/
        # astrometry on, rv/mulens off; matches EXOFASTv2). Per-file, not a
        # single component-wide flag, for consistency with every other
        # per-file key (gp:, likelihood:, ninterp:, rm:) -- build_likelihood's
        # group loop and compile_plotters both handle a group/instrument
        # mix of on/off files (see the mask logic there).
        self._light_travel_time_active = np.array(
            [bool(c.get("light_travel_time", True)) for c in self.config]
        )

    def _ltt_active(self, orbits):
        """Per-file light-travel-time flags, forced off when the orbit
        cannot supply the parameters the correction needs.

        `Orbit.register_parameters` declares a/m_primary/m_companion/m_total
        only when its bodies resolve, so a geometry-only orbit has none of
        them. Since `light_travel_time` defaults to ON, reading them
        unguarded would turn any such config -- which built fine before the
        correction existed -- into an AttributeError at build time.
        """
        if ltt.orbit_supports_ltt(orbits):
            return self._light_travel_time_active
        if self._light_travel_time_active.any():
            logger.warning(
                "transit: light-travel-time correction disabled -- the orbit "
                "does not define %s (its bodies did not resolve; see the "
                "orbit component's own warning). Set light_travel_time: "
                "false on the affected transit file(s) to silence this.",
                ", ".join(ltt.REQUIRED_ORBIT_PARAMS),
            )
        return np.zeros_like(self._light_travel_time_active)

    @property
    def prefix(self):
        return "transit"

    @classmethod
    def get_utilities(cls):
        from ...utilities import bls, getdata
        from ...utilities.registry import (
            UtilitySpec,
            argparse_subprocess_runner,
        )

        return [
            UtilitySpec(
                name="getdata",
                label="Download TESS/Kepler light curves",
                description=(
                    "Fetch TESS/Kepler light curves via lightkurve and write "
                    "them in EXOFASTv2/EXOZIPPy format."
                ),
                component_keys=["transit"],
                available=True,
                build_parser=getdata.build_parser,
                run=argparse_subprocess_runner("exozippy.utilities.getdata"),
            ),
            UtilitySpec(
                name="bls",
                label="BLS period search",
                description=(
                    "Box Least Squares transit search: report the period, "
                    "epoch, depth and duration of the strongest signal."
                ),
                component_keys=["transit"],
                available=True,
                build_parser=bls.build_parser,
                run=argparse_subprocess_runner("exozippy.utilities.bls"),
            ),
        ]

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.dat",
                "required": True,
                "doc": (
                    "Whitespace-delimited light curve; columns are time, "
                    "flux, flux_err, then optional detrend columns. Comment "
                    "lines start with '#'."
                ),
            },
            {
                "key": "band",
                "kind": "ref",
                "accepts": ["band"],
                "required": True,
                "doc": (
                    "Name of the band: block carrying this light curve's "
                    "filter identity and limb darkening."
                ),
            },
            {
                "key": "filter",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Deprecated and ignored; reference a band: block "
                    "instead (bands carry the filter identity)."
                ),
            },
            cls._mask_config_schema(),
            cls._columns_config_schema(("time", "flux", "err")),
            *cls._time_config_schema(),
            cls._plot_style_config_schema(),
            cls._gp_config_schema(),
            cls._likelihood_config_schema(),
        ]

    def load_data(self, system):
        """Stage 1: Load CSVs and generate data-driven bounds/inits."""
        self.baseline_init = [1.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements

        # Fixed instrument properties (not fitted parameters), read straight
        # off the config like self.files/self.band_names: exptime is the
        # exposure duration in minutes, ninterp the number of sub-samples
        # used to smear the model over that exposure (EXOFASTv2 parity).
        # Invalid combinations (ninterp < 1, a non-positive exptime, or
        # ninterp > 1 without an exptime to smear over) warn and fall back
        # to the inert defaults (ninterp=1, exptime=1: no smearing) rather
        # than crashing in the grid builder or silently smearing over a
        # made-up exposure time.
        self.exptime_min = [1.0] * self.n_elements
        self.ninterp = [1] * self.n_elements
        for i, c in enumerate(self.config):
            # A non-numeric value is a hard error (unlike the RANGE checks
            # below, which warn and fall back): `exptime: "abc"` is a typo
            # with no sensible fallback, and the bare `float()` used to
            # surface it as "could not convert string to float" naming no
            # instrument and no key (review 2.14.3).
            try:
                exptime = float(c.get("exptime", 1.0))
                ninterp = int(c.get("ninterp", 1))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"[{self.prefix}[{self.names[i]}]] exptime/ninterp must "
                    f"be numeric (exposure duration in minutes and the "
                    f"number of sub-samples); got "
                    f"exptime={c.get('exptime', '<unset>')!r}, "
                    f"ninterp={c.get('ninterp', '<unset>')!r}: {e}"
                ) from e
            if (
                ninterp < 1
                or exptime <= 0
                or (ninterp > 1 and "exptime" not in c)
            ):
                logger.warning(
                    f"transit {self.names[i]}: invalid exposure smearing "
                    f"config (exptime={c.get('exptime', '<unset>')} min, "
                    f"ninterp={c.get('ninterp', '<unset>')}): needs "
                    f"ninterp >= 1 and, when ninterp > 1, an explicit "
                    f"exptime > 0. Ignoring -- defaulting to ninterp=1, "
                    f"exptime=1 (no smearing)."
                )
                exptime, ninterp = 1.0, 1
            self.exptime_min[i] = exptime
            self.ninterp[i] = ninterp

        blocks = self._concat_blocks()
        for i in range(self.n_elements):
            # Shared reader: columns:, mask:, time_* conversion, then one
            # sort per file before anything is derived from it, keeping the
            # flux, errors and detrend columns aligned by construction.
            df = self._read_data(
                i, roles=("time", "flux", "err"), detrend=True
            )
            self.baseline_init[i] = np.median(df.iloc[:, 1].values)
            self.jittervar_lower[i] = self._jitter_floor(df.iloc[:, 2].values)

            blocks.add(
                i,
                time=df.iloc[:, 0].values,
                obs=df.iloc[:, 1].values,
                err=df.iloc[:, 2].values,
                df=df,
            )

        # Shared accumulator: concatenation (time/flux/err), inst_map, the
        # per-file row ranges, the block-diagonal detrend matrix, and the
        # optional GP / robust-likelihood hooks.  No user_factor: the errors
        # are already in the amplitude parameters' unit (relative flux).
        blocks.finalize("flux")

        self._build_oversample_grid()

        # Blind seeding: measure the period and conjunction epoch from the
        # photometry when nothing else supplies them.  Stage 1a, not stage 2
        # -- see components/globalsearch.py for why (Orbit builds tc's hard
        # window at stage 2 from whatever start it can see).
        self.bls_signal = None
        self._seed_from_bls(system)

    def _seed_from_bls(self, system):
        """Seed orbital period, conjunction epoch and radius ratio from BLS.

        Runs only when the relaxation engine cannot already DERIVE the
        period and conjunction time (``globalsearch.starts_satisfied``), and
        pushes a value only for the quantities that were missing -- so a
        params file that gives the period but not the epoch keeps its period
        and gains an epoch, with no precedence question to adjudicate.

        The radius ratio is opportunistic: a missing ``planet.p`` does not
        trigger a search (a fit can start at 1 Jupiter radius), but a search
        that ran for the period reports a depth, and ``sqrt(depth)`` is the
        radius ratio to the accuracy a start value needs.  The engine turns
        it into a ``planet.radius`` through ``Eq(p, radius / star_radius)``.
        """
        from .. import globalsearch

        mode = globalsearch.search_mode(system)
        if mode == "off":
            return
        orbit_ndx = globalsearch.sole_orbit_index(system, self.prefix)
        if orbit_ndx is None:
            return

        cm = self.config_manager
        planet_ndx = self._sole_planet_index(system, orbit_ndx)
        groups = {
            "period": (
                f"orbit.{orbit_ndx}.period",
                f"orbit.{orbit_ndx}.logP",
            ),
            "tc": (f"orbit.{orbit_ndx}.tc",),
        }
        if planet_ndx is not None:
            groups["p"] = (
                f"planet.{planet_ndx}.p",
                f"planet.{planet_ndx}.radius",
            )
        satisfied = globalsearch.starts_satisfied(cm, groups)
        required_missing = not (satisfied["period"] and satisfied["tc"])
        if mode != "force" and not required_missing:
            logger.debug(
                "[%s] BLS not needed: the orbital period and conjunction "
                "time are already derivable.",
                self.prefix,
            )
            return

        logger.info(
            "[%s] no start value for %s -- running a Box Least Squares "
            "search over %d photometric points.",
            self.prefix,
            ", ".join(k for k, v in satisfied.items() if not v) or "(forced)",
            self.time.size,
        )

        # Each file in its own flux system: divide by that file's median so
        # the concatenation is one relative-flux series and the box depth
        # means the same thing in every row.
        baseline = np.asarray(self.baseline_init, dtype=float)
        scale = np.where(
            np.isfinite(baseline) & (baseline != 0.0), baseline, 1.0
        )
        norm = scale[self.inst_map]
        signal = globalsearch.bls_search(
            self.time,
            self.flux / norm,
            self.err / norm,
            context=self.prefix,
        )
        self.bls_signal = signal
        if signal is None:
            return

        q = globalsearch.QUALITY_TRANSIT
        source = f"BLS on {self.n_elements} light curve(s)"
        applied = []
        if mode == "force" or not satisfied["period"]:
            applied.append(
                globalsearch.seed_start(
                    cm, f"orbit.{orbit_ndx}.period", signal.period, q, source
                )
            )
        if mode == "force" or not satisfied["tc"]:
            applied.append(
                globalsearch.seed_start(
                    cm, f"orbit.{orbit_ndx}.tc", signal.epoch, q, source
                )
            )
        if planet_ndx is not None and (
            mode == "force" or not satisfied.get("p", True)
        ):
            applied.append(
                globalsearch.seed_start(
                    cm,
                    f"planet.{planet_ndx}.p",
                    float(np.sqrt(signal.depth)),
                    q,
                    source,
                )
            )

        if not any(applied):
            # Nothing was actually taken (another search of equal or better
            # quality got there first).  Prose describes what the fit did.
            return

        get_collector(system).add(
            "Initial values for the orbital period and time of conjunction "
            "were measured from the photometry with a Box Least Squares "
            r"periodogram \citep{Kovacs:2002}, as implemented in "
            r"\texttt{astropy} \citep{Astropy:2013,Astropy:2018,Astropy:2022}."
            " Starting values do not enter the likelihood and cannot move "
            "the posterior.",
            section="data",
            key=f"{self.prefix}.global_search",
            rank=70,
        )

    def _sole_planet_index(self, system, orbit_ndx):
        """The one planet on this orbit, or None if there is not exactly one.

        Read off the raw config rather than ``planet.orbit_map``: this runs
        at stage 1a, where the maps may not have been built yet.
        """
        planets = getattr(system, "planet", None)
        if planets is None:
            return None
        on_orbit = [
            j
            for j, entry in enumerate(planets.config)
            if int((entry or {}).get("orbit_ndx", 0)) == orbit_ndx
        ]
        return on_orbit[0] if len(on_orbit) == 1 else None

    def _oversample_partition(self, inst_map):
        """Partition rows into per-ninterp groups for exposure smearing.

        ``inst_map`` is the per-row instrument index of whatever time
        vector is being modeled (the data's own ``self.inst_map``, or a
        plot layout's).  Returns a list of ``(rows, offsets, weights)``:
        ``rows`` the row indices in the group, ``offsets`` an
        ``(len(rows), kk)`` array of sub-exposure time offsets in days
        (``None`` for the ninterp=1 group, whose lone sample sits at the
        timestamp itself), ``weights`` the ``(kk,)`` averaging weights.

        EXOFASTv2 parity (exofast_chi2v2.pro): a sub-exposure time grid
        collapsed with a weighted mean per observation.  exptime/ninterp
        are per-instrument, but the time vector is concatenated across
        instruments, so instruments may disagree on both.  Rather than
        padding every observation out to the largest ninterp among the
        active instruments (which would evaluate the transit model at that
        many sub-samples even for observations whose own instrument needs
        only 1), observations are partitioned by their own instrument's
        ninterp value: each group's sub-exposure grid is exactly as wide as
        that group's own ninterp, so an evaluation costs exactly the
        sub-samples each observation's instrument needs.

        The sub-times are midpoint Riemann cells: kk equal-width cells
        across [-exptime/2, +exptime/2], sampled at each cell's center
        (offsets (j + 0.5)/kk - 0.5, never the exposure edges) -- exactly
        EXOFASTv2's grid (dindgen(ninterp)/ninterp - (ninterp-1)/
        (2*ninterp)), chosen there over trapezoid/Simpson: with uniform
        1/kk weights, edge samples would overweight the exposure boundaries
        and systematically over-smear.

        Groups come out in np.unique(ninterp) order, not row order; the
        caller gathers per-group results back into row order with the
        argsort of the concatenated group rows (see ``_lc_model``).
        """
        inst_map = np.asarray(inst_map, dtype=int)
        exptime_days = np.asarray(self.exptime_min, dtype=float) / 1440.0
        ninterp_per_inst = np.asarray(self.ninterp, dtype=int)

        ninterp_obs = ninterp_per_inst[inst_map]  # (N,)
        exptime_obs = exptime_days[inst_map]  # (N,)

        groups = []
        for kk in np.unique(ninterp_obs):
            rows = np.nonzero(ninterp_obs == kk)[0]
            if kk == 1:
                groups.append((rows, None, np.ones(1)))
            else:
                j = np.arange(kk)
                frac = (j + 0.5) / kk - 0.5
                offsets = frac[None, :] * exptime_obs[rows][:, None]
                groups.append((rows, offsets, np.full(kk, 1.0 / kk)))
        return groups

    def _build_oversample_grid(self):
        """
        Record the data's per-ninterp observation groups
        (``self._oversample_groups``, a list of ``(rows, grid, weights)``
        with ``grid`` the concrete ``(n_g, k_g)`` sub-exposure times) --
        the partition ``_oversample_partition`` makes of ``self.inst_map``,
        materialized on ``self.time``.  ``_lc_model`` rebuilds the same
        partition symbolically on whatever time vector it is handed (the
        likelihood's ``pm.Data`` or a plot grid); this concrete copy is the
        inspectable record of what the likelihood smears over (see
        tests/test_transit_exptime_ninterp.py).
        """
        self._oversample_groups = []
        for rows, offsets, weights in self._oversample_partition(
            self.inst_map
        ):
            if offsets is None:
                grid = self.time[rows][:, None]
            else:
                grid = self.time[rows][:, None] + offsets
            self._oversample_groups.append((rows, grid, weights))

    def register_parameters(self, system):
        """Stage 3: Embed data-driven hints into the PyMC manifest."""
        self._hint_baseline()
        self.manifest = {"baseline": None}
        self._register_noise(self.manifest, self.jittervar_lower)
        self._register_gp(self.manifest)
        self._register_robust(self.manifest)

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = {
                "shape": (self.total_detrend_cols,)
            }

        # Limb darkening comes from the Band component; map each
        # instrument (and each observation) to its band. A transit model
        # cannot be computed without LD, so a missing/unknown band
        # reference is an error, not a warning.
        if not hasattr(system, "band"):
            raise ValueError(
                "Transit instruments require band: blocks (bands carry the "
                "filter identity and limb-darkening parameters)."
            )
        name_to_idx = {name: i for i, name in enumerate(system.band.names)}
        missing = [
            (self.names[i], n)
            for i, n in enumerate(self.band_names)
            if n not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"Transit instrument(s) reference unknown band(s): "
                f"{missing}. Available bands: {list(name_to_idx)}."
            )
        self.band_map = np.array(
            [name_to_idx[n] for n in self.band_names], dtype=int
        )
        self.obs_band_map = self.band_map[self.inst_map]

    def _hint_baseline(self):
        """Push each light curve's median flux as a PRECEDENCE_DERIVED_DATA hint.

        The median is measured in ``load_data`` (stage 1), so it is ready
        by the time this runs at stage 3 -- which is what lets it go through
        the provenance pipeline at all.

        It used to be a plain manifest option (``{"baseline": {"initval":
        ...}}``), and options are merged as ``{**cfg, **options}`` AFTER
        ``resolve()``: they beat the user's params file outright and never
        acquire a rank.  For a data-derived START value that is backwards --
        an explicit ``transit.<name>.baseline`` in a params file (a restart
        file, say) was silently discarded.  As a hint it sits at
        PRECEDENCE_DERIVED_DATA (60), the tier this channel exists for: above the
        defaults.yaml 1.0 (20) and below the user (100), exactly like
        ``rvinstrument``'s gamma (median RV) and ``mulensinstrument``'s
        f_source/log_f_total.

        A non-finite median (a file with no usable flux) is skipped rather
        than hinted: the defaults.yaml 1.0 is then what the engine resolves,
        which is also the no-data fallback ``load_data`` seeds
        ``baseline_init`` with, so the two agree by construction.
        """
        baseline_init = getattr(self, "baseline_init", None)
        if baseline_init is None:
            return  # register_parameters without load_data (bare harness)
        arr = np.atleast_1d(baseline_init)
        for i in range(self.n_elements):
            val = float(arr[i])
            if not np.isfinite(val):
                logger.warning(
                    f"transit {self.names[i]}: median flux is not finite; "
                    f"leaving baseline at its defaults.yaml start value."
                )
                continue
            self.config_manager.add_hint(f"{self.prefix}.{i}.baseline", val)

    def _build_dilution(self, system):
        """
        Per-instrument SED-predicted depth dilution factor
        F_host / sum_j F_j in the instrument's band, as a (n_elements,)
        tensor (Deterministic "transit.dilution" for diagnostics), or
        None if no instrument's band filter is in the SED's BC grid.
        Instruments whose band filter is unavailable get dilution 1.

        The cache is per BUILD, not per component: ``build_likelihood``
        clears it with an inline ``self._dilution_node = None`` at the top of
        stage 7 (the general seam is ``Component.per_build_caches`` /
        ``reset_build_caches``, which ``System.build_model`` runs before
        stage 5; this cache does not use it because nothing reads the node
        before stage 7 -- see ``components.md``).  Components persist on the
        System and a second ``system.build_model()`` is supported (the GUI
        does it), so a cache that outlived the model handed the second
        build's likelihood a Deterministic belonging to the FIRST model --
        either a crash at logp compile or, worse, a silently stale dilution.
        Within one build the cache is still wanted: the node is asked for
        twice (the group loop and the beam term) and must be one node.
        """
        if getattr(self, "_dilution_node", None) is not None:
            return self._dilution_node

        sed = system.sed
        band = system.band
        dils = []
        any_diluted = False
        for i in range(self.n_elements):
            band_idx = int(self.band_map[i])
            filter_key = band.filter_mist[band_idx]
            if filter_key and sed.has_filter(filter_key):
                host = int(band.star_indices[band_idx])
                dils.append(
                    sed.predict_flux_fraction(host, filter_key, system)
                )
                any_diluted = True
            else:
                logger.warning(
                    f"transit {self.names[i]}: band filter '{filter_key}' "
                    f"is not in the SED's BC grid; no depth deblending "
                    f"applied for this instrument."
                )
                dils.append(pt.constant(1.0, dtype="float64"))
        if not any_diluted:
            return None
        self._dilution_node = pm.Deterministic(
            f"{self.prefix}.dilution", pt.stack(dils)
        )
        return self._dilution_node

    def _lc_model(self, system, t, blocks):
        """The transit model on times ``t`` -- THE expression, built once.

        ``build_likelihood`` calls this with the data's ``pm.Data`` time
        vector and the files' own row blocks; ``compile_plotters`` calls it
        with a symbolic grid laid out in the same per-file blocks.  So the
        plotted curve is literally the likelihood code run on other times:
        there is no second light-curve graph to keep in step, and every
        per-file choice (band, limb darkening, exptime/ninterp smearing,
        light-travel-time gating, SED dilution, baseline) is resolved from
        the block a row belongs to exactly as it is for the data.  The
        in-repo template is ``AstrometryInstrument._rel_model``.

        ``t``      -- ``(N,)`` time tensor.
        ``blocks`` -- ``[(inst_idx, rows), ...]``: the concrete row indices
                      of ``t`` that belong to each file.  Together they
                      cover ``t``; a file may appear in more than one block
                      and a file with no rows may be omitted.

        Returns ``(lc_model, planet_terms)``:

        ``lc_model`` ``(N,)`` -- baseline + every planet's transit/eclipse/
            thermal/reflection (exposure-smeared on each row's own
            sub-exposure grid) + beaming + ellipsoidal (un-smeared, at the
            flat per-observation time), WITHOUT the detrend term, which is
            per observation and is added by ``build_likelihood`` alone.
        ``planet_terms`` ``(N, N_planets)`` -- the change planet p's step
            makes to the running model, so ``baseline[inst] +
            sum(planet_terms, axis=1)`` reproduces ``lc_model`` for ANY
            planet count.  The ellipsoidal factor multiplies the running
            total (baseline + the earlier planets), exactly as it does in
            the model itself; the old plot path folded only the baseline
            in per planet, which was exact for one planet and an
            approximation for more (reviews 1.5.6, 7.14.1).

        The likelihood's numerics are pinned bit-identical on every shipped
        example, so the op sequence in here is the op sequence the
        likelihood always had -- including associations kept for a 1-ulp
        reason (see the ``r_norm`` comment).  Change it only with the
        start-logp table in hand.
        """
        orbits = system.orbit
        planets = system.planet
        band = system.band

        # The block layout -> per-row instrument index (numpy, known at
        # graph-build time -- the group loop's gates are Python if/else on
        # it) and its int32 tensor twin for indexing per-instrument
        # parameter vectors (the same construction as stage 5's
        # ``*_map_tensor``).
        n_rows = int(sum(len(rows) for _, rows in blocks))
        inst_map = np.empty(n_rows, dtype=int)
        for inst_idx, rows in blocks:
            inst_map[np.asarray(rows, dtype=int)] = int(inst_idx)
        inst_map_t = pt.as_tensor_variable(inst_map).astype("int32")
        obs_band_map = self.band_map[inst_map]
        obs_band_map_t = pt.as_tensor_variable(obs_band_map).astype("int32")

        # 1. Start with the photometric baseline
        lc_model = self.baseline.value[inst_map_t]

        # 1b. Per-planet orbital vectors, gathered onto the planet index.
        ecc_p = orbits.ecc.value[planets.orbit_map]  # (N_planets,)
        inc_p = orbits.inc.value[planets.orbit_map]
        period_p = orbits.period.value[planets.orbit_map]
        tc_p = orbits.tc.value[planets.orbit_map]
        ar_p = planets.ar.value

        # The per-planet transit/occultation geometry (impact
        # parameters and durations) used to be built here as bare
        # Deterministics.  It moved to the planet component in review 8.8.7,
        # where it is a set of ordinary derived Parameters -- table rows,
        # LaTeX macros, units, and a user-settable Gaussian, which is what
        # lets a published duration or eclipse time constrain e and omega.
        # It is geometry, not photometry: no light curve enters it, and an
        # RV-only fit has the same durations.  The phased-plot x-range
        # reads `planet.t14` for that reason.

        # 2. Orbital elements per planet. These don't depend on the
        # observation/sub-exposure grid, so they're computed once and
        # reused by every ninterp group below (see 4.).
        tp = orbits.tp.value[planets.orbit_map][
            None, None, :
        ]  # (1, 1, N_planets)
        n = orbits.n.value[planets.orbit_map][None, None, :]
        ecc = ecc_p[None, None, :]
        cosw = orbits.cosw.value[planets.orbit_map][None, None, :]
        sinw = orbits.sinw.value[planets.orbit_map][None, None, :]
        inc = inc_p[None, None, :]
        a_rstar = ar_p[None, None, :]
        sin_i = pt.sin(inc)
        cos_i = pt.cos(inc)

        # 2b. Light-travel-time (Roemer delay) inputs -- per-file gating via
        # ltt_active (self._light_travel_time_active, forced off when the
        # orbit cannot supply these parameters), resolved per group below.
        # Three roles, three factors -- see ltt.py's `factor` docs. The
        # occultation seam takes the mass DIFFERENCE; light EMITTED by the
        # planet (reflection) takes the planet's own barycentric fraction;
        # light emitted by the STAR (beaming, ellipsoidal) takes the
        # star's. One corrected time array cannot serve all three, and
        # using the geometry's for everything (or leaving the stellar
        # terms uncorrected, as here until 2026-08-15) mixes time
        # references differing by ~a/c within a single phase curve.
        ltt_active = self._ltt_active(orbits)
        # Structural, not a runtime test: True only where every orbit
        # these planets sit on has its sqrt(e) pair PINNED at zero, in
        # which case the Kepler solve is a sine and a cosine
        # (orbit.physics.solve_kepler, review 6.8.2).
        circular_kepler = orbits._all_circular(planets.orbit_map)
        a_rel = ltt_factor = ltt_reflect_factor = ltt_star_factor = None
        if ltt_active.any():
            a_rel = orbits.a.value[planets.orbit_map][
                None, None, :
            ]  # (1, 1, N_planets), physical semi-major axis [R_sun]
            # Barycentric scaling for an OCCULTATION seam: the mass
            # DIFFERENCE, not the planet's own barycentric fraction. A
            # transit is not an emission event -- the planet blocks light
            # the STAR emitted -- so both bodies enter at their own
            # retarded times and the star's delay partially cancels the
            # planet's. See ltt.py's `factor` docs for the derivation, and
            # for why m_primary/m_total (used here until 2026-08-15, and by
            # EXOFASTv2's target2bjd.pro) agrees to O(q) for a planet but
            # predicts a spurious a/c offset for a comparable-mass pair
            # whose true offset is exactly zero.
            m_primary = orbits.m_primary.value[planets.orbit_map]
            m_companion = orbits.m_companion.value[planets.orbit_map]
            m_total = orbits.m_total.value[planets.orbit_map]
            ltt_factor = ((m_primary - m_companion) / m_total)[None, None, :]
            # Reflected light comes off the planet's disk, so it rides the
            # planet's own delay about the barycenter.
            ltt_reflect_factor = (m_primary / m_total)[None, None, :]
            # Doppler beaming and ellipsoidal variation are the STAR's own
            # flux (its radial motion and its tidal shape), so they ride
            # the star's delay -- a factor ~q, not ~1. Flat (N_planets,):
            # these terms are evaluated per observation, not on the
            # sub-exposure grid.
            ltt_star_factor = m_companion / m_total

        # 3. Limb Darkening Setup (per observation, mapped from each
        # instrument's Band). When every band uses the linear law, Band's
        # manifest has no u2; the quadratic term is then zero.

        # Secondary-eclipse thermal emission (fitthermal): gate the whole
        # branch on the resolved parameter state, not the fitthermal
        # flags -- the manifest "overrides" pin is layered UNDER user
        # params, so params.yaml can free or set thermal without
        # fitthermal. When every element is pinned at exactly 0 the
        # thermal math (a second quad_solution_vector per planet, as
        # expensive as the transit itself) is skipped entirely, matching
        # exofast_tran.pro's `if thermal ne 0d0` runtime gate.
        thermal_active = band.thermal_may_be_nonzero()
        # BEER (PR 1.b): reflection is smeared on the same sub-exposure
        # grid as thermal (see the group loop below) and shares the same
        # resolved-state gate. Ellipsoidal is smooth on-orbit and is not
        # smeared -- it stays a flat per-observation array, applied once
        # after the group loop.
        reflect_active = band.reflect_may_be_nonzero()
        # Manifest-gated (see Band/Planet.register_parameters): with no
        # fitellip / beam flag anywhere the parameters do not exist, and
        # the terms are skipped entirely.
        ellip_active = band.ellipsoidal_may_be_nonzero()
        ellip_mapped = (
            band.ellipsoidal.value[obs_band_map_t] if ellip_active else None
        )
        beam_active = "beam" in planets.manifest

        # 3b. SED deblending (EXOFASTv2 parity): with more than one
        # modeled star, only the host contributes the transit, so the
        # observed depth is diluted by dil = F_host / sum_j F_j in the
        # instrument's band (host = the band's star_ndx).
        dil_inst = None
        if hasattr(system, "sed") and system.star.n_elements > 1:
            dil_inst = self._build_dilution(system)
        # Flat per-observation dilution for the post-loop beam term below
        # (EXOFASTv2 parity -- see the beam comment past the group loop).
        dil_obs_flat = None
        if dil_inst is not None:
            dil_obs_flat = dil_inst[inst_map_t]  # (N,)

        # 4. Exoplanet-core Transit Model, evaluated once per distinct
        # ninterp group (see _oversample_partition) instead of once for
        # the whole component at the largest ninterp: each group's
        # sub-exposure axis is exactly that group's own ninterp wide, so a
        # short-cadence (ninterp=1) observation is never evaluated at
        # another instrument's larger ninterp. With ninterp==1 everywhere
        # there is exactly one group of width 1, identical to the
        # original (pre-oversampling) computation.
        groups = self._oversample_partition(inst_map)
        inverse_order = np.argsort(np.concatenate([g[0] for g in groups]))
        planet_group_decrement = [[] for _ in range(planets.n_elements)]
        for rows, offsets_np, weights_np in groups:
            # This group's sub-exposure time grid, built ON the time tensor
            # so the same code serves the data and a plot grid.  ninterp=1:
            # the lone sample sits at the timestamp itself and exptime is
            # irrelevant (the original instantaneous model).
            t_rows = t[rows][:, None]  # (n_g, 1)
            if offsets_np is None:
                t_grid = t_rows[:, :, None]  # (n_g, 1, 1)
            else:
                t_grid = (t_rows + pt.constant(offsets_np))[
                    :, :, None
                ]  # (n_g, k_g, 1)
            w_g = pt.constant(weights_np)  # (k_g,)

            # Light-travel-time correction. The factor depends on WHICH
            # observable is being retarded, not on the timestamps, so one
            # corrected time array cannot serve the whole model (see
            # ltt.py's `factor` docs). This group needs two of them:
            #
            #   geometry (transit/eclipse shape, and via planetvisible the
            #     thermal gating) -- the occultation seam, ltt_factor;
            #   reflected light -- emitted by the PLANET, so its own
            #     barycentric fraction m_primary/m_total.
            #
            # Beaming and ellipsoidal are stellar and un-smeared; they are
            # corrected after this loop with the primary's factor.
            #
            # Per-file gate (this group's own rows may mix files that want
            # it on and off, since groups are formed by ninterp value, not
            # by file). Costs nothing extra when every row in the group is
            # off (no ltt.retarded_time call, no pt.where); costs one extra
            # Kepler solve per role in use, no pt.where, when every row is
            # on (the default); costs one extra Kepler solve PLUS one
            # pt.where only for a genuinely mixed group.
            lt_active_rows = ltt_active[
                inst_map[rows]
            ]  # (n_g,) bool, numpy -- known at graph-build time

            def _retard_grid(role_factor):
                """t_grid retarded with `role_factor`, honoring the
                per-row gate. Returns t_grid untouched when no row in this
                group wants the correction."""
                if not lt_active_rows.any():
                    return t_grid
                corrected, _ = ltt.retarded_time(
                    t_grid,
                    tp,
                    n,
                    ecc,
                    sinw,
                    cosw,
                    sin_i,
                    a_rel,
                    factor=role_factor,
                    z0=0.0,
                    circular=circular_kepler,
                )
                if lt_active_rows.all():
                    return corrected
                # Both branches are ordinary, everywhere-finite time
                # values (no singularity like solve_delay's az=0
                # branch), so this pt.where carries none of the
                # where-trap risk that formula guarded against --
                # verified directly (not just asserted) by
                # tests/test_transit_ltt.py's
                # test_mixed_group_ltt_gradient_is_finite.
                lt_mask = pt.constant(
                    lt_active_rows[:, None, None].astype("float64")
                )
                return pt.where(lt_mask > 0.5, corrected, t_grid)

            t_grid_final = _retard_grid(ltt_factor)
            # Broadcast to (n_g, k_g, N_planets) unconditionally (a no-op
            # when t_grid_final already has that shape) so the per-planet
            # slice below is safe regardless of N_planets: with LTT off (or
            # a face-value t_grid pass-through), the last dim is the
            # unbroadcast size 1 from t_grid, and t_grid_final[:, :, p]
            # would index-error for any p > 0 without this.
            time_g_corrected = t_grid_final + pt.zeros(
                (1, 1, planets.n_elements)
            )

            # Reflected light is emitted by the PLANET, so its phase runs
            # on the planet's own retarded time, not the occultation
            # seam's. Only built when reflection is actually on.
            time_g_reflect = None
            if reflect_active and ltt_reflect_factor is not None:
                time_g_reflect = _retard_grid(ltt_reflect_factor) + pt.zeros(
                    (1, 1, planets.n_elements)
                )

            # The shared Kepler-to-state kernel (skips the Newton iteration
            # outright when every orbit these planets sit on is pinned
            # circular, review 6.8.2).
            terms = orbit_physics.state_vector_terms(
                t_grid_final,
                tp,
                n,
                ecc,
                sinw=sinw,
                cosw=cosw,
                circular=orbits._all_circular(planets.orbit_map),
            )
            # NOT a_rstar * terms.r_over_a: this association is the
            # pre-refactor one, kept because the reassociated product
            # rounded differently (1 ulp on hat3's start logp) and the
            # 4.8.2 refactor is pinned bit-identical.
            r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * terms.cosf)
            sin_wf = terms.sinwf
            cos_wf = terms.coswf

            # (n_g, k_g, N_planets)
            b = pt.sqrt(
                pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i)
            )
            Z = r_norm * sin_wf * sin_i

            u1_mapped = band.u1.value[obs_band_map[rows]]  # (n_g,)
            if "u2" in band.manifest:
                u2_mapped = band.u2.value[obs_band_map[rows]]  # (n_g,)
            else:
                u2_mapped = pt.zeros_like(u1_mapped)

            thermal_g = None
            if thermal_active:
                # (n_g,) ppm; 0 for any band pinned off (see
                # Band.register_parameters).
                thermal_g = band.thermal.value[obs_band_map[rows]]

            reflect_g = None
            if reflect_active:
                # (n_g,) ppm; 0 for any band pinned off (see
                # Band.register_parameters).
                reflect_g = band.reflect.value[obs_band_map[rows]]

            dil_obs = None
            if dil_inst is not None:
                dil_obs = dil_inst[inst_map[rows]]  # (n_g,)

            for p_idx in range(planets.n_elements):
                b_p = b[
                    :, :, p_idx
                ]  # (n_g, k_g) sky-plane separation in units of R_*
                Z_p = Z[
                    :, :, p_idx
                ]  # (n_g, k_g) line-of-sight coord (+ = planet in front of star)
                r_p = planets.p.value[p_idx]  # scalar R_p/R_*

                # Limb-darkened flux fraction: 1.0 off-disk, <1.0 during transit.
                # u1/u2_mapped are (n_g,); add the sub-exposure axis so they
                # broadcast against b_p.
                flux_frac = quad_limb_darkened_flux(
                    b_p, r_p, u1_mapped[:, None], u2_mapped[:, None]
                )  # (n_g, k_g)

                # Fraction of stellar flux blocked (0 off-disk, ~ r^2 at disk centre)
                blocked = 1.0 - flux_frac

                # Primary transit only; secondary eclipse (planet behind star) has Z < 0
                blocked = pt.where(Z_p > 0.0, blocked, 0.0)

                # Secondary eclipse / constant thermal emission (fitthermal)
                # and BEER (PR 1.b) reflection. Both come off the planet's
                # disk and share planetvisible; both live inside this group
                # loop so they get exposure-smeared on the same sub-exposure
                # grid as the transit (EXOFASTv2 averages the full model,
                # thermal included). Beam/ellipsoidal are smooth on-orbit
                # and are handled separately, un-smeared, after this loop.
                net = blocked
                if thermal_g is not None or reflect_g is not None:
                    visible = physics.calc_planet_visible(
                        b_p, Z_p, r_p
                    )  # (n_g, k_g)
                    if thermal_g is not None:
                        net = net - 1e-6 * thermal_g[:, None] * visible
                    if reflect_g is not None:
                        t_ref = (
                            time_g_reflect
                            if time_g_reflect is not None
                            else time_g_corrected
                        )
                        reflect_term_g = physics.calc_reflect_term(
                            t_ref[:, :, p_idx],
                            tc_p[p_idx],
                            period_p[p_idx],
                            reflect_g[:, None],
                            visible,
                        )
                        net = net - reflect_term_g

                if dil_obs is not None:
                    # One blended-aperture dilution for the whole
                    # perturbation from unity: EXOFASTv2's
                    # f0*(model*(1-dilute)+dilute) scales the transit dip
                    # and the planet's extra flux by the same factor.
                    net = net * dil_obs[:, None]

                # Weighted mean over this group's own ninterp sub-samples
                # (weights sum to 1). With ninterp==1 this is a no-op
                # identity (single column, weight 1).
                net_avg_g = pt.sum(net * w_g[None, :], axis=1)  # (n_g,)
                planet_group_decrement[p_idx].append(net_avg_g)

        planet_terms = []
        for p_idx in range(planets.n_elements):
            # Groups were visited in np.unique(ninterp) order, not row
            # order; inverse_order restores the original per-observation
            # order after concatenation. This decrement already includes
            # the transit, thermal, and (BEER, PR 1.b) reflection terms,
            # all computed and exposure-smeared in the group loop above.
            net_avg = pt.concatenate(planet_group_decrement[p_idx])[
                inverse_order
            ]
            lc_before = lc_model
            lc_model = lc_model - net_avg
            # This planet's own change to the running model, accumulated
            # alongside lc_model term by term (see the docstring).
            step = -net_avg

            tc_this = tc_p[p_idx]  # scalar, this planet's time of conjunction
            period_this = period_p[p_idx]

            # Beaming and ellipsoidal are the STAR's own flux, so they are
            # evaluated at the star's retarded time -- a different time
            # base from the occultation geometry above (see ltt.py's
            # `factor` docs). Until 2026-08-15 they used the uncorrected
            # time while reflection used the geometry's, so a phase curve
            # mixed three time references differing by ~a/c. Un-smeared,
            # so this is a flat (N,) correction, and it costs one Kepler
            # solve per planet only when a stellar term is on.
            time_star = t
            need_stellar = beam_active or ellip_mapped is not None
            if (
                need_stellar
                and ltt_star_factor is not None
                and ltt_active.any()
            ):
                # The orbital elements above are shaped (1, 1, N_planets)
                # for the sub-exposure grid; this term is un-smeared and
                # per planet, so it needs the FLAT scalars -- indexing the
                # 3-D versions with [p_idx] would slice axis 0 (size 1).
                star_corrected, _ = ltt.retarded_time(
                    t,
                    tp[0, 0, p_idx],
                    n[0, 0, p_idx],
                    ecc[0, 0, p_idx],
                    sinw[0, 0, p_idx],
                    cosw[0, 0, p_idx],
                    sin_i[0, 0, p_idx],
                    orbits.a.value[planets.orbit_map][p_idx],
                    factor=ltt_star_factor[p_idx],
                    z0=0.0,
                    circular=circular_kepler,
                )
                if ltt_active.all():
                    time_star = star_corrected
                else:
                    star_mask = pt.constant(
                        ltt_active[inst_map].astype("float64")
                    )
                    time_star = pt.where(star_mask > 0.5, star_corrected, t)

            # Beaming is diluted the same way thermal/reflect are above --
            # EXOFASTv2 parity: exofast_chi2v2.pro:1517/1556 pass both beam
            # and dilute into exofast_tran, which adds beam at
            # exofast_tran.pro:146 and applies the dilution scaling at
            # exofast_tran.pro:157 (after beam, so beam is diluted too).
            # Ellipsoidal is NOT diluted -- it's a multiplicative factor on
            # the running lc_model (baseline + this planet's transit/
            # eclipse/thermal/reflect/beam so far), not an additive flux
            # term, so the dilution scaling doesn't apply to it the same
            # way. Neither term is gated by planetvisible: both are stellar
            # effects (the star's own RV motion / tidal shape), present
            # regardless of the planet's occultation state. Both are smooth
            # on-orbit and are deliberately NOT exposure-smeared -- they are
            # evaluated at the flat per-observation time, unlike thermal/
            # reflect above.
            if beam_active:
                beam_p = planets.beam.value[p_idx]  # scalar, ppm
                beam_term = physics.calc_beam_term(
                    time_star, tc_this, period_this, beam_p
                )
                if dil_obs_flat is not None:
                    beam_term = beam_term * dil_obs_flat
                lc_model = lc_model + beam_term
                step = step + beam_term

            # Ellipsoidal is multiplicative (exofast_tran.pro), applied to
            # the running lc_model (baseline + this planet's transit/
            # eclipse/thermal/reflect/beam so far). With >1 planet sharing
            # a band, each planet's factor multiplies in turn -- order-
            # dependent for N>1, exact for the single-planet case this
            # targets.  Its DEVIATION is diluted like every other term:
            # exofast_tran.pro applies the dilution to (modelflux - 1)
            # after the ellipsoidal factor multiplies in, so ellipsoidal
            # IS diluted there (review of PR #53; the difference is the
            # second-order (1-d)^2 vs (1-d) cross term, ~ppm x depth).
            if ellip_mapped is not None:
                ellip_dev = (
                    physics.calc_ellipsoidal_factor(
                        time_star, tc_this, period_this, ellip_mapped
                    )
                    - 1.0
                )
                if dil_obs_flat is not None:
                    ellip_dev = ellip_dev * dil_obs_flat
                lc_model = lc_model * (1.0 + ellip_dev)
                # (before + step) * (1 + dev) - before == step + (before +
                # step) * dev: the factor acts on the whole running total.
                step = step + (lc_before + step) * ellip_dev
            planet_terms.append(step)

        return lc_model, pt.stack(planet_terms, axis=1)

    def _data_blocks(self):
        """The data's own row blocks, ``[(i, rows_i), ...]`` in file order.

        ``Instrument.rows`` publishes each file's contiguous range;
        materialized as an index array rather than left as the slice for
        the reason ``rows`` documents (a ``pm.Data``'s length is symbolic,
        so a slice's shape is too, and the JAX backend cannot trace it).
        """
        blocks = []
        for i in range(self.n_elements):
            sl = self.rows(i)
            blocks.append((i, np.arange(sl.start, sl.stop)))
        return blocks

    def build_likelihood(self, model, system):
        # Stage 7 is once per BUILD, and a second system.build_model() on one
        # System is supported (the GUI does it), so every cached NODE has to
        # be dropped here.  A dilution node that outlived its model was handed
        # to the second build's likelihood -- a crash at logp compile, or a
        # silently stale dilution.
        self._dilution_node = None

        time = pm.Data("transit_time", self.time)
        flux = pm.Data("transit_data", self.flux)
        err = pm.Data("transit_err", self.err)

        planets = system.planet
        band = system.band

        # The one light-curve expression (see _lc_model), on the data's own
        # times and row blocks.  Retained as plain attributes, not
        # Deterministics: at (N_obs,) either would add N_obs * draws *
        # chains floats to every trace.  compile_plotters compiles them as
        # the plotted model AT the observations (the phased panels' "other
        # planets" cleaning), and tests compile a one-off function from
        # _model_flux_node.
        lc_model, planet_terms = self._lc_model(
            system, time, self._data_blocks()
        )
        self._lc_physical_node = lc_model
        self._lc_planet_terms_node = planet_terms

        if self.total_detrend_cols > 0:
            detrend = pm.Data("transit_detrend", self.detrend_matrix)
            lc_model += pt.dot(detrend, self.detrend_coeffs.value)

        # Full per-observation model prediction (baseline + detrend +
        # exposure-averaged transit decrement).
        self._model_flux_node = lc_model

        # 5. Likelihood (shared base helper: sqrt(err^2 + jitter_variance)).
        # add_observation_likelihood is the plain Normal unless a light curve
        # asked for a GP, in which case that curve gets a celerite2 marginal
        # likelihood around this same transit model.
        sigma = self.total_sigma(err)
        self.add_observation_likelihood(
            "transit_likelihood",
            mu=lc_model,
            sigma=sigma,
            observed=flux,
            system=system,
        )

        # Modeling-draft prose for the transit model itself (the shared
        # data/noise sentences came from the dispatcher above).  The same
        # resolved-state gates _lc_model builds the terms from.
        terms = []
        if band.thermal_may_be_nonzero():
            terms.append("constant thermal emission")
        if band.reflect_may_be_nonzero():
            terms.append("reflected light")
        if band.ellipsoidal_may_be_nonzero():
            terms.append("ellipsoidal variation")
        if "beam" in planets.manifest:
            terms.append(r"Doppler beaming \citep{Faigler:2011}")
        if terms:
            from exozippy.outputs.prose import join_names

            get_collector(system).add(
                "The transit model includes phase-curve terms for "
                + join_names(terms)
                + ", following EXOFASTv2's parameterization "
                + r"\citep{Eastman:2019}.",
                section="planetary",
                key=f"{self.prefix}.phase_curve",
                rank=22,
            )
        get_collector(system).add(
            r"We modeled each transit with the analytic quadratic "
            r"limb-darkening light curve of \citet{Agol:2020}, as "
            r"implemented in exoplanet-core \citep{ForemanMackey:2021}.",
            section="planetary",
            key=f"{self.prefix}.lc_model",
            rank=20,
        )
        get_collector(system).add_software("exoplanet-core")
        if getattr(system, "band", None) is not None and (
            system.band.ld_laws and system.band.ld_laws[0] == "quadratic"
        ):
            get_collector(system).add(
                r"Limb-darkening coefficients were sampled in the "
                r"$(q_1, q_2)$ parameterization of \citet{Kipping:2013}.",
                section="planetary",
                key=f"{self.prefix}.ld_param",
                rank=21,
            )

    # Points per instrument on a plotted model grid.  One size for the
    # unphased span and the phased period window, so the two panels share
    # a single compiled layout (see compile_plotters).
    _PLOT_GRID_N = 2000

    def compile_plotters(self, model, system):
        """Compile the plotted model -- the likelihood's expression, twice.

        Two compiled functions, both from ``_lc_model`` (there is no
        separate plotting graph any more; reviews 1.5.6, 7.14.1):

        ``_lc_data_fn(*params)`` -> ``(lc_physical, planet_terms)`` AT the
            observations: literally the nodes ``build_likelihood`` scored
            (minus the detrend term, which comes off the plotted data
            instead -- see Instrument.detrend_at_data), compiled against
            the plot parameters.  The phased panels' "other planets"
            cleaning reads its ``planet_terms``.
        ``_lc_grid_fn(t, *params)`` -> ``(lc_full, lc_matrix)`` on a plot
            GRID laid out in the same per-file blocks as the data:
            ``_PLOT_GRID_N`` rows per instrument, in file order, so a row's
            band, smearing, LTT gate and dilution resolve exactly as a data
            row's would.  Every model curve -- unphased span, phased period
            window -- is this one function fed a different ``t``.

        Exposure smearing therefore comes from the builder's own ninterp/
        exptime path; the NumPy re-implementation that used to average an
        instantaneous compiled function over sub-exposure offsets is gone,
        and with it the drift it caused: it smeared beaming and ellipsoidal
        too, which the likelihood deliberately does not (review 1.5.6).
        """
        t_input = pt.vector("lc_t_input")
        param_symbols = [p.value for p in system.plot_params]
        planets = getattr(system, "planet", None)
        orbits = getattr(system, "orbit", None)

        if planets is not None and orbits is not None:
            n_grid = self._PLOT_GRID_N
            blocks = [
                (i, np.arange(i * n_grid, (i + 1) * n_grid))
                for i in range(self.n_elements)
            ]
            lc_full_node, lc_matrix = self._lc_model(system, t_input, blocks)

            # Retain the symbolic nodes and their time input so plot_data
            # can derive param_deps (graph walk) and hand the GUI the
            # symbolic tensors behind the model traces.  Unused by plot().
            self._lc_t_input = t_input
            self._lc_matrix_node = lc_matrix
            self._lc_full_node = lc_full_node

            self._lc_grid_fn = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=[lc_full_node, lc_matrix],
                on_unused_input="ignore",
            )
            phys = getattr(self, "_lc_physical_node", None)
            terms = getattr(self, "_lc_planet_terms_node", None)
            self._lc_data_fn = (
                pytensor.function(
                    inputs=param_symbols,
                    outputs=[phys, terms],
                    on_unused_input="ignore",
                )
                if phys is not None and terms is not None
                else None
            )

        # Per-file GP conditional-mean evaluators (no-op without a gp: key).
        self._compile_gp_plotters(system)

    def _eval_lc_grid(self, t_blocks, param_values):
        """Evaluate the plotted model on one grid per instrument.

        ``t_blocks[i]`` is instrument ``i``'s ``(_PLOT_GRID_N,)`` time
        grid.  Returns ``(full, matrix)`` with ``full[i]`` the model light
        curve (baseline included) and ``matrix[i]`` the ``(_PLOT_GRID_N,
        N_planets)`` per-planet terms on that grid -- the concatenated
        layout ``compile_plotters`` compiled, split back per file.
        """
        n_grid = self._PLOT_GRID_N
        t_all = np.concatenate(
            [np.asarray(t, dtype=np.float64) for t in t_blocks]
        )
        if t_all.shape[0] != n_grid * self.n_elements:
            raise ValueError(
                f"[{self.prefix}] plot grid layout needs {n_grid} points "
                f"per instrument, got {[len(t) for t in t_blocks]}."
            )
        full, matrix = self._lc_grid_fn(t_all, *param_values)
        full = np.asarray(full).reshape(self.n_elements, n_grid)
        matrix = np.asarray(matrix).reshape(self.n_elements, n_grid, -1)
        return full, matrix

    def _eval_lc_data(self, param_values):
        """The likelihood's own model at the observations: ``(lc_physical,
        planet_terms)`` as ``(N_obs,)`` and ``(N_obs, N_planets)``."""
        phys, terms = self._lc_data_fn(*param_values)
        return np.asarray(phys), np.asarray(terms)

    def _lc_at_times(self, param_values, i, t):
        """Instrument ``i``'s plotted model at arbitrary times ``t``.

        The compiled grid layout is fixed at ``_PLOT_GRID_N`` rows per
        instrument, so ``t`` is fed through it in chunks of that size --
        block ``i`` carrying the chunk (padded with its last time), every
        other block a dummy -- and the results stitched back.  A
        convenience for a caller that wants the likelihood's model for ONE
        file at times of its own choosing (tests, a GUI probe); the panels
        feed the layout directly.  Returns ``(full, matrix)``: the model
        light curve (baseline included, GP excluded) ``(len(t),)`` and the
        per-planet terms ``(len(t), N_planets)``, smeared over instrument
        ``i``'s own exposure exactly as its data rows are.
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        n_grid = self._PLOT_GRID_N
        fulls, mats = [], []
        for start in range(0, max(t.size, 1), n_grid):
            chunk = t[start : start + n_grid]
            fill = chunk[-1] if chunk.size else float(self.time[0])
            block = np.full(n_grid, fill)
            block[: chunk.size] = chunk
            t_blocks = [np.full(n_grid, fill) for _ in range(self.n_elements)]
            t_blocks[i] = block
            full, matrix = self._eval_lc_grid(t_blocks, param_values)
            fulls.append(full[i][: chunk.size])
            mats.append(matrix[i][: chunk.size])
        return np.concatenate(fulls), np.concatenate(mats)

    # ------------------------------------------------------------------
    # Shared data preparation. The matplotlib plot() path and the GUI
    # plot_data() path both go through these helpers, so the two paths
    # always draw the exact same arrays (see chart.Chart).
    # ------------------------------------------------------------------
    def _unphased_lc_curves(self, system, point):
        """Full model light curve per instrument: baseline + transit + GP.

        Returns ``[(t_pretty_i, y_i), ...]`` in file order, from ONE
        evaluation of the compiled grid layout.  The unphased panel shows
        the model the likelihood actually fits, so any GP this light curve
        requested is included; the phased panels take it back out of the
        data instead (see _phased_lc_arrays).  The GP term is zero for a
        light curve without a gp: key.
        """
        param_values = self._point_to_plot_params(point, system)
        t_blocks = []
        for i in range(self.n_elements):
            t_data = self.time[self.rows(i)]
            t_blocks.append(
                np.linspace(
                    t_data.min(), t_data.max(), self._PLOT_GRID_N
                ).astype(np.float64)
            )
        full, _ = self._eval_lc_grid(t_blocks, param_values)
        curves = []
        for i, t_pretty in enumerate(t_blocks):
            y_gp = self.gp_mean_on_grid(system, point, i, t_pretty)
            curves.append((t_pretty, full[i] + y_gp))
        return curves

    def _eval_unphased_lc(self, system, point, i):
        """Instrument ``i``'s unphased curve, ``(t_pretty, y)``; see
        ``_unphased_lc_curves``, which plot_data calls once for all."""
        return self._unphased_lc_curves(system, point)[i]

    def _phased_lc_shared(self, system, point):
        """The parts of a phased panel that do NOT depend on which planet.

        ``_phased_lc_arrays`` is called once per (planet, instrument) and
        recomputed the same things every time: the marshalled parameter
        values, the per-observation GP and detrend corrections (both
        point-only), and the model at the observations -- which the
        likelihood's own compiled node now supplies for every instrument
        at once (review 6.5.1 hoisted the per-instrument copies; the
        builder made the pass a single call).

        Returned as one dict per (component, point).  The per-planet phased
        model grids fill in lazily as planets are reached
        (``_phased_lc_grid``), one layout evaluation per planet shared by
        every instrument's panel.
        """
        param_values = self._point_to_plot_params(point, system)
        _, data_terms = self._eval_lc_data(param_values)
        return {
            "param_values": param_values,
            # Removed from the phased data along with the other planets':
            # the correlated component would smear the fold, and the fitted
            # trend is a per-observation term no pretty-grid curve carries.
            # Both are zeros when the feature is off.
            "extra_signals": self.gp_mean_at_data(system, point)
            + self.detrend_at_data(point),
            # (N_obs, N_planets): each planet's term at every observation,
            # from the likelihood's own node.
            "data_terms": data_terms,
            "grid": {},
        }

    def _phased_lc_grid(self, shared, p_idx, t_model):
        """Every instrument's per-planet terms on planet ``p_idx``'s phase
        grid, ``(n_inst, _PLOT_GRID_N, N_planets)``, cached per planet."""
        cache = shared["grid"]
        if p_idx not in cache:
            _, matrix = self._eval_lc_grid(
                [t_model] * self.n_elements, shared["param_values"]
            )
            cache[p_idx] = matrix
        return cache[p_idx]

    def _phased_lc_arrays(self, system, point, p_idx, i, shared=None):
        """
        One-period phase grid, isolated model term for planet p_idx on
        instrument i, and the baseline-subtracted, other-planet-cleaned
        flux at the observed times -- used by plot_data() (and via it
        plot()).  Both the grid curve and the cleaning come from
        ``_lc_model``, so the panel shows the exposure-smeared model the
        likelihood fit, with beaming/ellipsoidal un-smeared as there.

        ``shared`` is this point's ``_phased_lc_shared`` dict; omit it and
        one is built, which is what a standalone caller wants and what the
        per-planet loop must NOT do.
        """
        if shared is None:
            shared = self._phased_lc_shared(system, point)
        planets = system.planet
        P_ref = self._point_value(point, system.orbit.period, p_idx)
        tc_ref = self._point_value(point, system.orbit.tc, p_idx)

        t_model = np.linspace(
            tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, self._PLOT_GRID_N
        ).astype(np.float64)
        phase_model = ((t_model - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
        time_from_center_model = phase_model * P_ref
        sort_m = np.argsort(phase_model)

        y_planet = self._phased_lc_grid(shared, p_idx, t_model)[i][:, p_idx]

        rows = self.rows(i)
        data_terms = shared["data_terms"][rows]
        other_mask = np.ones(planets.n_elements, dtype=bool)
        other_mask[p_idx] = False
        other_terms = np.sum(data_terms[:, other_mask], axis=1)

        baseline = self._point_value(point, self.baseline, i)
        cleaned_flux = (
            self.flux[rows]
            - baseline
            - other_terms
            - shared["extra_signals"][rows]
        )
        data_phases = ((self.time[rows] - tc_ref) / P_ref + 0.5) % 1.0 - 0.5

        return {
            "P_ref": P_ref,
            "tc_ref": tc_ref,
            "x_model": time_from_center_model[sort_m],
            "y_model": y_planet[sort_m],
            "x_data": data_phases * P_ref,
            "y_data": cleaned_flux,
        }

    def plot(self, system, points, filename_prefix="debug"):
        """Render the unphased + phased LC PDFs from plot_data specs.

        The specs are the single description of these plots -- the GUI draws
        the same ones via plotly (see plotrender.py's module docstring).
        """
        from exozippy.plotrender import plot_via_specs

        plot_via_specs(self, system, points, filename_prefix=filename_prefix)

    def plot_data(self, system, point=None):
        """
        GUI charts for the transit photometry: per instrument an
        unphased flux-vs-time chart, and (with a point) one phased chart
        per planet/instrument. point=None returns only the raw data
        traces. See Component.plot_data and chart.Chart.
        """
        from exozippy.chart import Chart, Trace

        specs = []
        full_deps = self._model_trace_param_deps(
            getattr(self, "_lc_full_node", None), system
        )
        matrix_deps = self._model_trace_param_deps(
            getattr(self, "_lc_matrix_node", None), system
        )

        # The baseline, the fitted detrend model and the GP conditional
        # mean enter the panels in numpy (_point_value / detrend_at_data /
        # gp_mean_at_data), not through the symbolic nodes, so the graph
        # walk cannot see them -- without these deps a baseline, detrend
        # coefficient or GP hyperparameter slider would never refresh these
        # charts in the GUI (review 1.12.9 for the GP labels).
        baseline_label = getattr(
            getattr(self, "baseline", None), "label", None
        )
        numpy_deps = (
            ([baseline_label] if baseline_label else [])
            + self.detrend_dep_labels()
            + self.gp_dep_labels()
        )
        full_deps = full_deps + [
            lbl for lbl in numpy_deps if lbl not in full_deps
        ]
        matrix_deps = matrix_deps + [
            lbl for lbl in numpy_deps if lbl not in matrix_deps
        ]

        # ---- Unphased: flux vs time, per instrument -------------------
        # The fitted trend is per observation, so it comes off the DATA
        # rather than going onto the model curve; zeros without detrend
        # columns (Instrument.detrend_at_data).
        detrend = self.detrend_at_data(point)
        # Every instrument's curve comes from ONE evaluation of the compiled
        # grid layout.  A failed model eval keeps the data-only panels
        # (matching the per-point tolerance of the old hand-drawn loop).
        curves = None
        if point is not None:
            try:
                curves = self._unphased_lc_curves(system, point)
            except Exception as e:  # noqa: BLE001 - bad point/draw
                logger.warning(f"LC model eval failed: {e}")
        for i in range(self.n_elements):
            mask = self.inst_map == i
            traces = []
            deps = []
            if curves is not None:
                t_pretty, y_full = curves[i]
                deps = full_deps
                traces.append(
                    Trace(
                        name="model",
                        role="model",
                        kind="line",
                        x=t_pretty,
                        y=y_full,
                        node=getattr(self, "_lc_full_node", None),
                    )
                )
            traces.append(
                Trace(
                    name=self.names[i],
                    role="data",
                    kind="scatter",
                    x=self.time[mask],
                    y=self.flux[mask] - detrend[mask],
                    yerr=self.err[mask],
                    # Black-dot default (the historical PDF look); a user
                    # plot: color/marker still wins via _data_trace_style.
                    style={
                        "color": "k",
                        "marker": ".",
                        **self._data_trace_style(i),
                    },
                )
            )
            specs.append(
                Chart(
                    id=f"{self.prefix}.unphased.{self.names[i]}",
                    component={
                        "yaml_key": self.prefix,
                        "instance": self.names[i],
                    },
                    title=f"Transit Photometry: {self.names[i]}",
                    xlabel="Time [BJD]",
                    ylabel="Relative Flux",
                    traces=traces,
                    param_deps=deps,
                    meta={
                        "phase_folded": False,
                        "instrument": self.names[i],
                        "file_tag": f"LC_unphased_{self.names[i]}",
                        "figsize": (12, 5),
                        # The unphased DATA are detrend-subtracted, so they
                        # move with the point whenever this instrument has
                        # detrend columns (and only then).
                        "dynamic_data": self.total_detrend_cols > 0,
                        "caption": (
                            "Transit photometry from "
                            + latex_escape(self.names[i])
                            + " with the best-fit model (red)."
                            + self.detrend_caption()
                        ),
                    },
                )
            )

        # ---- Phased: one chart per planet/instrument (needs a model) --
        if point is not None:
            planets = system.planet
            # Once per (component, point), not once per planet x instrument
            # (6.5.1); the per-instrument matrices fill in lazily.
            shared = self._phased_lc_shared(system, point)
            for p_idx in range(planets.n_elements):
                for i in range(self.n_elements):
                    # A failed prep skips this panel, exactly as the old
                    # hand-drawn loop skipped its figure.
                    try:
                        prep = self._phased_lc_arrays(
                            system, point, p_idx, i, shared=shared
                        )
                    except Exception as e:  # noqa: BLE001 - bad point/draw
                        logger.warning(f"LC phased model eval failed: {e}")
                        continue
                    mask = self.inst_map == i
                    traces = [
                        Trace(
                            name="model",
                            role="model",
                            kind="line",
                            x=prep["x_model"],
                            y=prep["y_model"],
                            node=getattr(self, "_lc_matrix_node", None),
                            style={"lw": 1.0},
                        ),
                        Trace(
                            name=self.names[i],
                            role="data",
                            kind="scatter",
                            x=prep["x_data"],
                            y=prep["y_data"],
                            yerr=self.err[mask],
                            style={
                                "color": "k",
                                "marker": ".",
                                **self._data_trace_style(i),
                            },
                        ),
                    ]
                    pname = planets.names[p_idx]
                    meta = {
                        "phase_folded": True,
                        "planet": pname,
                        "instrument": self.names[i],
                        "period": prep["P_ref"],
                        "tc": prep["tc_ref"],
                        "file_tag": (f"LC_phased_{self.names[i]}_{pname}"),
                        "figsize": (10, 6),
                        "hline_y": 0.0,
                        "caption": (
                            "Phase-folded transit of planet "
                            + latex_escape(pname)
                            + " in "
                            + latex_escape(self.names[i])
                            + ", baseline and other planets removed."
                            + self.detrend_caption()
                        ),
                        # The phased DATA re-folds with tc/P and its cleaning
                        # subtracts the baseline, other planets and any GP --
                        # all point-dependent, so live evals must re-ship it.
                        "dynamic_data": True,
                    }
                    # Zoom to +/- t14 around mid-transit when a transit
                    # duration is known for this planet.  `planet.t14`, not
                    # `transit.t14`: the durations are planet geometry now
                    # (review 8.8.7).  Read through _point_value like the
                    # period and tc above, NOT `point["planet.t14"][p_idx]`:
                    # this read sits outside the panel's try, so a scalar or
                    # short t14 in the point IndexError'ed on p_idx > 0 and
                    # killed plot_data for the whole component (review
                    # 2.5.4); the helper falls back to element 0 for a
                    # broadcast scalar and to the Parameter's own start when
                    # the point lacks it.  A non-finite duration (no start
                    # to fall back on) simply means no zoom.
                    x_range = None
                    t14_param = getattr(planets, "t14", None)
                    if t14_param is not None:
                        t14_ref = self._point_value(point, t14_param, p_idx)
                        if np.isfinite(t14_ref):
                            x_range = [-t14_ref, t14_ref]
                    specs.append(
                        Chart(
                            id=f"{self.prefix}.phased.{self.names[i]}.{pname}",
                            component={
                                "yaml_key": self.prefix,
                                "instance": self.names[i],
                            },
                            title=f"Phased LC: {pname} -- {self.names[i]}",
                            xlabel=f"Time from Mid-Transit [d] (P = {prep['P_ref']:.5f} d)",
                            ylabel="Flux - Baseline",
                            traces=traces,
                            param_deps=matrix_deps,
                            x_range=x_range,
                            meta=meta,
                        )
                    )

        return specs
