import logging

import numpy as np

logger = logging.getLogger(__name__)

import astropy.units as u
import pymc as pm
import pytensor
import pytensor.tensor as pt

from exozippy.components.instrument import Instrument
from exozippy.outputs.prose import get_collector
from exozippy.outputs.texutils import latex_escape


class RVInstrument(Instrument):
    prose_noun = "radial velocity"

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Instrument Parameters"
        self.units = [
            self._parse_rv_unit(i, c) for i, c in enumerate(self.config)
        ]
        # Which star the RVs are of; its Doppler signal is the sum over
        # every orbit that star is a body of (planetary reflex and stellar
        # companions alike).
        # Index or name (as the schema below advertises): resolved through
        # the one shared translator, which reads the star instance names off
        # the raw system config -- available here, where system.star is not.
        self.star_ndx = [
            self.resolve_star_ndx(
                c.get("star_ndx"),
                f"[{self.prefix}] {c.get('name', i)} star_ndx",
            )
            for i, c in enumerate(self.config)
        ]
        # Rossiter-McLaughlin: a file may set `rm: <orbit_name>` to add the
        # in-transit RM distortion of that orbit to this instrument's RV
        # model (off by default -> the RV likelihood is unchanged). Optional
        # `rm_band: <band_name>` selects the limb darkening; else defaults.
        self.rm_orbit = [c.get("rm") for c in self.config]
        self.rm_band = [c.get("rm_band") for c in self.config]
        # `rm_model: hirano2010 | hirano2011` (default hirano2011). hirano2010 is
        # the fast closed-form series; hirano2011 the disk integral.
        self.rm_model = [c.get("rm_model", "hirano2011") for c in self.config]
        _valid_rm = {"hirano2010", "hirano2011"}
        for m in self.rm_model:
            if m not in _valid_rm:
                raise ValueError(
                    f"[{self.prefix}] unknown rm_model {m!r}; expected one of "
                    f"{sorted(_valid_rm)}."
                )
        # Light-travel-time (Roemer delay) correction on the RM occultation
        # geometry (see components/rm.py, components/ltt.py) -- on by
        # default (Jason's decision: transit/rm/astrometry on, rv/mulens
        # off; matches EXOFASTv2). Only meaningful on a file that also sets
        # `rm:`; harmless (unread) otherwise, same as rm_band/rm_model.
        self.light_travel_time = [
            bool(c.get("light_travel_time", True)) for c in self.config
        ]
        self.total_detrend_cols = 0

    @property
    def prefix(self):
        return "rvinstrument"

    def _parse_rv_unit(self, i, entry):
        """Resolve a file's ``unit:`` key to an astropy Unit.

        The YAML value is a plain string (``unit: km/s``), so it has to go
        through ``u.Unit`` before ``load_data`` can call ``.to()`` on it --
        exactly what ``astrometryinstrument`` does for its ``sep_unit``.
        Anything astropy accepts as a velocity works; the default is m/s.
        """
        raw = entry.get("unit", "m/s")
        name = entry.get("name", i)
        try:
            unit = u.Unit(raw)
            unit.to(u.m / u.s)
        except Exception as exc:
            raise ValueError(
                f"[{self.prefix}] {name}: unit: {raw!r} is not a velocity "
                f"astropy can parse (e.g. 'm/s', 'km/s', 'km s-1')."
            ) from exc
        return unit

    @classmethod
    def get_utilities(cls):
        from ...utilities import lomb_scargle
        from ...utilities.registry import (
            UtilitySpec,
            argparse_subprocess_runner,
        )

        return [
            UtilitySpec(
                name="lomb_scargle",
                label="Lomb-Scargle periodogram",
                description=(
                    "Lomb-Scargle radial-velocity periodogram: report the "
                    "period, epoch and semi-amplitude of the strongest "
                    "signal."
                ),
                component_keys=["rvinstrument"],
                available=True,
                build_parser=lomb_scargle.build_parser,
                run=argparse_subprocess_runner(
                    "exozippy.utilities.lomb_scargle"
                ),
            ),
        ]

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "file",
                "kind": "datafile",
                "accepts": "*.rv",
                "required": True,
                "doc": (
                    "Whitespace-delimited radial-velocity data; columns are "
                    "time, RV, RV_err, then optional detrend columns. Comment "
                    "lines start with '#'."
                ),
            },
            {
                "key": "star_ndx",
                "kind": "ref",
                "accepts": ["star"],
                "required": False,
                "doc": (
                    "Index or name of the observed star (default 0); the "
                    "'star.<name>' path spelling works too. The RV model "
                    "sums orbit.K over every orbit containing this star."
                ),
            },
            {
                "key": "unit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Astropy unit string for the RV/error columns. Default "
                    "'m/s'."
                ),
            },
            cls._mask_config_schema(),
            cls._columns_config_schema(("time", "rv", "err")),
            *cls._time_config_schema(),
            cls._plot_style_config_schema(),
            cls._gp_config_schema(),
            cls._likelihood_config_schema(),
        ]

    def load_data(self, system):
        """Stage 1: Load CSVs and generate data-driven bounds/inits."""
        self.gamma_init = [0.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements

        blocks = self._concat_blocks()
        for i in range(self.n_elements):
            # Shared reader: columns:, mask:, time_* conversion, then one
            # sort per file before anything is derived from it, keeping the
            # RVs, errors and detrend columns aligned by construction.
            df = self._read_data(i, roles=("time", "rv", "err"), detrend=True)
            factor = self.units[i].to(u.solRad / u.d)

            m_s_factor = self.units[i].to(u.m / u.s)
            self.gamma_init[i] = np.mean(df.iloc[:, 1].values) * m_s_factor
            self.jittervar_lower[i] = self._jitter_floor(
                df.iloc[:, 2].values, factor=m_s_factor
            )

            blocks.add(
                i,
                time=df.iloc[:, 0].values,
                obs=df.iloc[:, 1].values * factor,
                err=df.iloc[:, 2].values * factor,
                df=df,
            )

        # Shared accumulator: concatenation (time/rv/err), inst_map, the
        # per-file row ranges, the block-diagonal detrend matrix, and the
        # optional GP / robust-likelihood hooks.  self.err ends up in
        # solRad/d while the GP amplitude and out_scale are declared in m/s,
        # hence user_factor.
        blocks.finalize("rv", user_factor=(u.solRad / u.d).to(u.m / u.s))

        self.k_init = self._estimate_k_init()

        # Blind seeding: measure the period and conjunction epoch from the
        # velocities when nothing else supplies them.  Stage 1a, not stage 2
        # -- see components/globalsearch.py for why (Orbit builds tc's hard
        # window at stage 2 from whatever start it can see).
        self.ls_signal = None
        self._seed_from_lombscargle(system)

    def _seed_from_lombscargle(self, system):
        """Seed orbital period and conjunction epoch from a Lomb-Scargle peak.

        Runs only when the relaxation engine cannot already DERIVE the period
        and conjunction time, and seeds only the quantities that were
        missing.  A transit search on the same orbit outranks this one
        (``globalsearch.QUALITY_TRANSIT``), so on a system with both, the
        photometric period and epoch stand and the RVs contribute the
        semi-amplitude.

        The semi-amplitude is not pushed as a hint: it REPLACES
        ``self.k_init``, which ``Planet.register_parameters`` already turns
        into the ``planet.K`` hint at stage 2.  One channel, one number --
        and the sinusoid fit is the better estimator of the two, since
        ``sqrt(2) * std`` counts the noise variance as signal.
        """
        from .. import globalsearch

        mode = globalsearch.search_mode(system)
        if mode == "off":
            return
        orbit_ndx = globalsearch.sole_orbit_index(system, self.prefix)
        if orbit_ndx is None:
            return

        cm = self.config_manager
        groups = {
            "period": (
                f"orbit.{orbit_ndx}.period",
                f"orbit.{orbit_ndx}.logP",
            ),
            "tc": (f"orbit.{orbit_ndx}.tc",),
        }
        satisfied = globalsearch.starts_satisfied(cm, groups)
        if mode != "force" and all(satisfied.values()):
            logger.debug(
                "[%s] Lomb-Scargle not needed: the orbital period and "
                "conjunction time are already derivable.",
                self.prefix,
            )
            return

        logger.info(
            "[%s] no start value for %s -- running a Lomb-Scargle search "
            "over %d velocities.",
            self.prefix,
            ", ".join(k for k, v in satisfied.items() if not v) or "(forced)",
            self.time.size,
        )

        # Work in m/s with each instrument's own offset removed: a
        # periodogram of the raw concatenation measures the offsets, not the
        # planet.
        to_ms = (u.solRad / u.d).to(u.m / u.s)
        gammas = np.asarray(self.gamma_init, dtype=float)
        residual = self.rv * to_ms - gammas[self.inst_map]
        signal = globalsearch.lombscargle_search(
            self.time,
            residual,
            self.err * to_ms,
            inst_map=self.inst_map,
            context=self.prefix,
        )
        self.ls_signal = signal
        if signal is None:
            return

        q = globalsearch.QUALITY_RV
        source = f"Lomb-Scargle on {self.n_elements} RV data set(s)"
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
        if np.isfinite(signal.amplitude) and signal.amplitude > 0.0:
            logger.info(
                "[%s] planet.K start moved from %.4g to %.4g m/s (the "
                "Lomb-Scargle sinusoid's semi-amplitude replaces "
                "sqrt(2) x scatter).",
                self.prefix,
                self.k_init,
                signal.amplitude,
            )
            self.k_init = float(signal.amplitude)

        if not any(applied):
            # Every orbital seed was declined -- a transit search got there
            # first (QUALITY_TRANSIT).  Saying the period came from the RVs
            # would be false, and the transit component has already said
            # where it really came from.
            return

        get_collector(system).add(
            "Initial values for the orbital period and time of conjunction "
            "were measured from the radial velocities with a Lomb-Scargle "
            r"periodogram \citep{Lomb:1976,Scargle:1982}, as implemented in "
            r"\texttt{astropy} \citep{VanderPlas:2018,Astropy:2022}. "
            "Starting values do not enter the likelihood and cannot move the "
            "posterior.",
            section="data",
            key=f"{self.prefix}.global_search",
            rank=70,
        )

    def _estimate_k_init(self):
        """Seed for the planetary RV semi-amplitude, in m/s.

        ``sqrt(2) * std`` is the semi-amplitude of a sinusoid, but only when
        the scatter it is measured on is the SIGNAL's.  Measured on the raw
        concatenation of every instrument it is dominated instead by the
        constant offsets BETWEEN instruments: one absolute-RV instrument
        sitting at a ~30 km/s systemic velocity next to a relative one seeds
        ``planet.K`` at ~20 km/s for an m/s-level planet.  So each file's own
        ``gamma_init`` (its mean, already computed above) is removed first.

        With a single instrument this is identical to the old expression --
        subtracting a constant does not change a standard deviation.

        Degenerate inputs (one point per file, a file whose RVs are all
        identical, zero-variance data) leave no scatter to measure at all and
        would seed K = 0, which the relaxation engine happily turns into a
        ~1e-20 Mjup planet mass.  There the median error bar -- the white
        noise level, i.e. the amplitude at the detection limit -- is the
        honest answer, and 1 m/s the last resort if even that vanishes.
        """
        to_ms = (u.solRad / u.d).to(u.m / u.s)
        gammas = np.asarray(self.gamma_init, dtype=float)
        residual = self.rv * to_ms - gammas[self.inst_map]
        k = np.sqrt(2.0) * np.std(residual)
        if np.isfinite(k) and k > 0.0:
            return float(k)

        median_err = float(np.median(self.err)) * to_ms
        if np.isfinite(median_err) and median_err > 0.0:
            logger.warning(
                "[%s] the RVs carry no scatter about their per-instrument "
                "means; seeding K from the median error bar (%.3g m/s) "
                "instead.",
                self.prefix,
                median_err,
            )
            return median_err

        logger.warning(
            "[%s] the RVs carry neither scatter nor usable error bars; "
            "seeding K at 1 m/s.",
            self.prefix,
        )
        return 1.0

    def register_parameters(self, system):
        """Stage 3: Embed data-driven hints into the PyMC manifest."""
        gamma_arr = np.atleast_1d(self.gamma_init)
        for i in range(self.n_elements):
            val = (
                gamma_arr[i].item()
                if hasattr(gamma_arr[i], "item")
                else float(gamma_arr[i])
            )
            self.config_manager.add_hint(f"{self.prefix}.{i}.gamma", val)

        # gamma is FREE: it is the sampled per-instrument RV offset, and
        # rvinstrument/defaults.yaml has carried no expressions: block for it
        # since cc26d77.  It said "default" until 2026-08 -- harmless only
        # because there was no block to find, and only until someone added
        # one, at which point every RV fit would have quietly derived its
        # offset instead of sampling it.  manifest.expression_config now
        # raises on that mismatch, so this has to say what it means.
        self.manifest = {"gamma": None}
        self._register_noise(self.manifest, self.jittervar_lower)
        self._register_gp(self.manifest)
        self._register_robust(self.manifest)

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = {
                "shape": (self.total_detrend_cols,)
            }

    def _orbit_rv_terms(self, system, star_idx):
        """
        Per-orbit RV semi-amplitudes for one star: (K_vec, orbit_map) over
        the orbits that star is a body of.  A primary-group member carries
        the primary reflex K directly; a companion-group member moves with
        the opposite phase (omega_* + 180 deg) and an amplitude scaled by
        the group mass ratio, expressed here as a negated, rescaled K with
        the same omega_* phase formula.
        """
        orbits = system.orbit
        members = orbits.star_membership(star_idx)
        if not members:
            raise ValueError(
                f"[{self.prefix}] star {star_idx} is not a body of any "
                f"orbit; no RV model can be built. Add it to an orbit's "
                f"primary/companion group."
            )
        if not hasattr(orbits, "K"):
            raise ValueError(
                f"[{self.prefix}] the orbit component has no K parameter "
                f"(its body groups did not resolve against the active "
                f"system); RVs require orbits with resolvable bodies."
            )
        k_nodes, omap = [], []
        for o, role in members:
            if role == "primary":
                k_nodes.append(orbits.K.value[o])
            else:
                k_nodes.append(
                    -orbits.K.value[o]
                    * orbits.m_primary.value[o]
                    / orbits.m_companion.value[o]
                )
            omap.append(o)
        return pt.stack(k_nodes), np.asarray(omap, dtype=int)

    def build_likelihood(self, model, system):
        time = pm.Data("rv_time", self.time)
        rv = pm.Data("rv_data", self.rv)
        err = pm.Data("rv_err", self.err)

        orbits = system.orbit
        if len(set(self.star_ndx)) > 1:
            raise NotImplementedError(
                f"[{self.prefix}] all RV instruments must observe the same "
                f"star for now (got star_ndx={self.star_ndx})."
            )

        # 1. Construct the RV Model: start with the gamma constant offset
        rv_model = self.gamma.value[self.inst_map_tensor]

        # sum the contribution from every orbit containing the observed star
        K_vec, omap = self._orbit_rv_terms(system, self.star_ndx[0])
        rv_model += pt.sum(
            orbits.get_radial_velocity(time, K_vec, omap), axis=1
        )

        # 1b. Rossiter-McLaughlin in-transit distortion. No-op unless a file
        # set `rm: <orbit_name>` -> the RV model above is unchanged byte for
        # byte (mirrors the GP opt-in). compute_rm_rv returns m/s; convert to
        # the internal RV unit (solRad/d) and add only to that file's rows.
        #
        # INDEX, do not pt.switch. A switch over the branch VALUES would
        # evaluate the Hirano kernel at every instrument's timestamps and then
        # throw away the rows it does not apply to -- the JAX where-trap
        # (CLAUDE.md): a `where` whose unselected branch can be invalid poisons
        # the gradient of the selected one too. Slicing the RM instrument's own
        # rows makes the unselected rows unreachable by construction instead of
        # merely masked, and is cheaper by exactly the fraction of the data
        # that is not the RM file (the H2011 kernel is a 201 x 64 quadrature
        # PER ROW; on a 40-of-73-row example it was 83% wasted work).
        if any(self.rm_orbit):
            from ..rm import compute_rm_rv, resolve_rm_indices

            rv_ms_per_internal = float((u.solRad / u.d).to(u.m / u.s))
            for i, oname in enumerate(self.rm_orbit):
                if not oname:
                    continue
                # The published contiguous range (Instrument.rows), not a
                # scan: inc_subtensor over a slice stays a plain subtensor.
                rows = self.rows(i)
                if rows.stop == rows.start:
                    continue
                oidx, pidx, bidx = resolve_rm_indices(
                    system, oname, self.rm_band[i]
                )
                rm_ms = compute_rm_rv(
                    system,
                    time[rows],
                    oidx,
                    pidx,
                    bidx,
                    model=self.rm_model[i],
                    light_travel_time_active=self.light_travel_time[i],
                )  # (rows.stop - rows.start,) m/s
                rv_model = pt.inc_subtensor(
                    rv_model[rows], rm_ms / rv_ms_per_internal
                )

        # detrending
        if self.total_detrend_cols > 0:
            detrend = pm.Data("rv_detrend", self.detrend_matrix)
            rv_model += pt.dot(detrend, self.detrend_coeffs.value)

        # 2. Define the Likelihood.  Total variance = data_error^2 + jitter^2
        # (shared base helper).  The shared dispatcher writes the plain Normal
        # unless a file asked for a GP, in which case that file's residuals get
        # a celerite2 marginal likelihood with this same mu and sigma.
        sigma = self.total_sigma(err)

        self.add_observation_likelihood(
            f"{self.prefix}.model",
            mu=rv_model,
            sigma=sigma,
            observed=rv,
            system=system,
        )

        # Modeling-draft prose for the RV model itself (the shared
        # data/noise sentences came from the dispatcher above).
        get_collector(system).add(
            "Radial velocities were modeled as a sum of Keplerian orbits "
            "(every orbit containing the observed star), plus a "
            "per-instrument velocity offset, using the Kepler solver in "
            r"exoplanet-core \citep{ForemanMackey:2021}.",
            section="orbits",
            key=f"{self.prefix}.rv_model",
            rank=20,
        )
        get_collector(system).add_software("exoplanet-core")

    def compile_plotters(self, model, system):
        """Compiles the fast PyTensor functions used by the plot_data specs."""
        # 1. We need a time grid input
        t_input = pt.vector("t_input")

        # 2. Get the global symbols to match the MCMC trace signature
        param_symbols = [p.value for p in system.plot_params]

        # 3. Pull the physics from the system
        orbits = getattr(system, "orbit", None)

        if orbits is not None:
            K_vec, omap = self._orbit_rv_terms(system, self.star_ndx[0])
            self._plot_orbit_map = omap

            # The matrix of shape (N_times, N_member_orbits)
            rv_matrix_node = orbits.get_radial_velocity(t_input, K_vec, omap)

            # Rossiter-McLaughlin: fold the in-transit distortion into the
            # plotted model (it is part of the model the likelihood fits), so
            # the RM anomaly shows in BOTH the unphased (full) and phased
            # (per-orbit) RV panels. Added to the RM orbit's own matrix column.
            # No-op unless a file set `rm:`.
            if any(self.rm_orbit):
                from ..rm import compute_rm_rv, resolve_rm_indices

                rv_ms_per_internal = float((u.solRad / u.d).to(u.m / u.s))
                omap_list = list(omap)
                seen = set()
                for i, oname in enumerate(self.rm_orbit):
                    if not oname or oname in seen:
                        continue
                    seen.add(oname)
                    oidx, pidx, bidx = resolve_rm_indices(
                        system, oname, self.rm_band[i]
                    )
                    if oidx not in omap_list:
                        continue
                    col = omap_list.index(oidx)
                    rm_col = (
                        compute_rm_rv(
                            system,
                            t_input,
                            oidx,
                            pidx,
                            bidx,
                            model=self.rm_model[i],
                            light_travel_time_active=self.light_travel_time[i],
                        )
                        / rv_ms_per_internal
                    )
                    rv_matrix_node = pt.set_subtensor(
                        rv_matrix_node[:, col], rv_matrix_node[:, col] + rm_col
                    )

            rv_full_node = pt.sum(rv_matrix_node, axis=1)

            # Retain the symbolic nodes and their time input so plot_data
            # can (a) derive param_deps by walking the graph and (b) hand
            # G5 the symbolic tensors behind the model traces for its own
            # compiled re-evaluation. Not needed by the CLI plot() path.
            self._rv_t_input = t_input
            self._rv_matrix_node = rv_matrix_node
            self._rv_full_node = rv_full_node

            # Save them to SELF, not the system!
            self._compiled_full_rv = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=rv_full_node,
                on_unused_input="ignore",
            )

            self._compiled_rv_matrix = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=rv_matrix_node,
                on_unused_input="ignore",
            )

        # Per-file GP conditional-mean evaluators (no-op without a gp: key).
        self._compile_gp_plotters(system)

    # ------------------------------------------------------------------
    # Shared data preparation. Both the matplotlib plot() path and the
    # GUI plot_data() path go through these helpers, so the two paths
    # always draw the exact same arrays (see plotspec.PlotSpec).
    # ------------------------------------------------------------------
    def _rv_factor(self):
        """Internal-units -> user-units (m/s) conversion for RV values.

        Uses the gamma Parameter's factor once the model is built; falls
        back to the raw solRad/d -> m/s conversion so plot_data works in
        data-only mode (point=None), before any Parameter exists.
        """
        gamma = getattr(self, "gamma", None)
        if gamma is not None and hasattr(gamma, "element_factor"):
            return gamma.element_factor(0)
        return (u.solRad / u.d).to(u.m / u.s)

    def _unphased_grid(self):
        """Smooth 64-bit time grid spanning the data (for model curves)."""
        return np.linspace(self.time.min(), self.time.max(), 2000).astype(
            np.float64
        )

    def _eval_unphased_model(self, system, point):
        """Summed RV model on the pretty grid, returned in m/s.

        Physical (orbit + gamma-free) signal only; any GP is per-instrument
        and is added by _eval_unphased_gp_models.
        """
        t_pretty = self._unphased_grid()
        param_values = self._point_to_plot_params(point, system)
        y_model = self._compiled_full_rv(t_pretty, *param_values)
        if y_model.ndim > 1:
            y_model = np.squeeze(y_model)
        return t_pretty, y_model * self._rv_factor()

    def _eval_unphased_gp_models(self, system, point):
        """Full (physical + GP) unphased curves, one per GP instrument.

        The GP is a per-instrument noise model, so there is no single "full
        model" curve: each instrument that requested a GP gets its own,
        evaluated only over the span where that instrument actually has data
        (the conditional mean reverts to zero outside it, which would draw a
        misleading flat line across the whole plot). Returns a list of
        (instrument index, t, y in m/s); empty without any GP.
        """
        if not self.has_gp_plotters():
            return []
        factor = self._rv_factor()
        param_values = self._point_to_plot_params(point, system)
        out = []
        for i in sorted(self._gp_pred_on_grid):
            mask = self.inst_map == i
            t_i = np.linspace(
                self.time[mask].min(), self.time[mask].max(), 2000
            ).astype(np.float64)
            y_phys = self._compiled_full_rv(t_i, *param_values)
            if y_phys.ndim > 1:
                y_phys = np.squeeze(y_phys)
            y_gp = self.gp_mean_on_grid(system, point, i, t_i)
            out.append((i, t_i, (y_phys + y_gp) * factor))
        return out

    def _phased_shared(self, system, point):
        """The parts of a phased panel that do NOT depend on which orbit.

        ``_phased_arrays`` is called once per member orbit, and three of the
        arrays it built were the same every time: the marshalled parameter
        values, the RV matrix at the OBSERVED times (the model grid's matrix
        does vary -- its time grid is that orbit's own period window), and
        the per-observation GP + detrend corrections.  Recomputing them per
        orbit meant N_orbits evaluations of a compiled function over the full
        data set per posterior draw, and the spaghetti re-runs the whole
        thing per draw (review 6.5.1).  Hoisted to once per (instrument,
        point) and passed down.

        Kept as a separate method rather than a cache keyed on the point:
        ``point`` is a plain dict, so identity is the only key available and
        it is not a safe one.
        """
        param_values = self._point_to_plot_params(point, system)
        return {
            "param_values": param_values,
            "data_rv_matrix": self._compiled_rv_matrix(
                self.time, *param_values
            ),
            # Phasing data that still contains the correlated (e.g. rotation)
            # signal just smears the panel, so the GP conditional mean comes
            # out of the data along with the other orbits' signal -- as does
            # the fitted detrend model, which the likelihood adds per
            # observation (build_likelihood's pt.dot) but no model curve on a
            # pretty grid can carry.  Both are zeros when the feature is off,
            # so this is a no-op then.
            "extra_signals": self.gp_mean_at_data(system, point)
            + self.detrend_at_data(point),
        }

    def _phased_arrays(self, system, point, col, o_idx, shared=None):
        """
        Phase grid, isolated model curve, and the per-observation
        background (all other member orbits' signal) for one member
        orbit -- used by plot_data() (and via it plot()).

        ``shared`` is this (instrument, point)'s ``_phased_shared`` dict;
        omit it and one is built, which is what a standalone caller wants
        and what the per-orbit loop must NOT do.
        """
        if shared is None:
            shared = self._phased_shared(system, point)
        factor = self._rv_factor()
        P_ref = self._point_value(point, system.orbit.period, o_idx)
        tc_ref = self._point_value(point, system.orbit.tc, o_idx)

        t_model = np.linspace(
            tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000
        ).astype(np.float64)
        phase_model = np.mod((t_model - tc_ref) / P_ref + 0.25, 1.0)
        sort_m = np.argsort(phase_model)

        rv_matrix = self._compiled_rv_matrix(t_model, *shared["param_values"])
        y_orbit = rv_matrix[:, col]

        other_mask = np.ones(len(self._plot_orbit_map), dtype=bool)
        other_mask[col] = False
        other_signals = np.sum(shared["data_rv_matrix"][:, other_mask], axis=1)

        return {
            "P_ref": P_ref,
            "tc_ref": tc_ref,
            "factor": factor,
            "phase_model": phase_model[sort_m],
            "y_model": y_orbit[sort_m] * factor,
            "other_signals": other_signals + shared["extra_signals"],
        }

    def plot(self, system, points, filename_prefix="debug"):
        """Render the unphased + per-orbit phased PDFs from plot_data specs.

        The specs are the single description of these plots -- the GUI draws
        the same ones via plotly (see plotrender.py's module docstring).
        """
        from exozippy.plotrender import plot_via_specs

        plot_via_specs(self, system, points, filename_prefix=filename_prefix)

    def plot_data(self, system, point=None):
        """
        GUI plot specs for the RV instrument: one unphased RV-vs-time
        chart plus one phased chart per member orbit. With point=None only
        the observed data traces are returned (raw preview, no model);
        with a point, model curves are added via the shared prep helpers.
        See Component.plot_data and plotspec.PlotSpec.
        """
        from exozippy.plotspec import PlotSpec, Trace

        factor = self._rv_factor()
        specs = []

        # ---- Unphased: RV vs time -------------------------------------
        traces = []
        model_deps = []
        if point is not None:
            t_pretty, y_model = self._eval_unphased_model(system, point)
            deps = self._model_trace_param_deps(
                getattr(self, "_rv_full_node", None), system
            )
            model_deps = deps
            traces.append(
                Trace(
                    name="model",
                    role="model",
                    kind="line",
                    x=t_pretty,
                    y=y_model,
                    node=getattr(self, "_rv_full_node", None),
                )
            )
            # One physical+GP curve per GP instrument (see
            # _eval_unphased_gp_models). No symbolic node: the GP conditional
            # mean is not part of the model graph, so the GUI cannot re-render
            # these on a slider move -- it must ask for a fresh point.
            for i, t_gp, y_gp in self._eval_unphased_gp_models(system, point):
                traces.append(
                    Trace(
                        name=f"{self.names[i]} model+GP",
                        role="model",
                        kind="line",
                        x=t_gp,
                        y=y_gp,
                        style={"series_index": int(i), "lw": 1.0},
                    )
                )
        # The fitted trend is per observation, so it comes off the DATA
        # rather than going onto the model curve (Instrument.detrend_at_data);
        # zeros without detrend columns.
        detrend = self.detrend_at_data(point)
        for i in range(self.n_elements):
            mask = self.inst_map == i
            # gamma offset only when a point supplies it; raw data otherwise
            g = (
                self._point_value(point, self.gamma, i)
                if point is not None
                else 0.0
            )
            traces.append(
                Trace(
                    name=self.names[i],
                    role="data",
                    kind="scatter",
                    x=self.time[mask],
                    y=(self.rv[mask] - g - detrend[mask]) * factor,
                    yerr=self.err[mask] * factor,
                    style=self._data_trace_style(i),
                )
            )
        # The data traces are gamma- and detrend-subtracted, so they move with
        # the point too (dynamic_data) and those sliders must reach this
        # component through param_deps -- both are applied in numpy, not
        # through the symbolic model node, so the graph walk alone would miss
        # them.
        gamma_label = getattr(getattr(self, "gamma", None), "label", None)
        numpy_deps = ([gamma_label] if gamma_label else []) + (
            self.detrend_dep_labels()
        )
        if point is not None:
            model_deps = model_deps + [
                lbl for lbl in numpy_deps if lbl not in model_deps
            ]
        specs.append(
            PlotSpec(
                id=f"{self.prefix}.unphased",
                component={"yaml_key": self.prefix, "instance": None},
                title=f"Unphased RV Model: {getattr(system, 'name', '')}",
                xlabel="Time [BJD]",
                ylabel="Relative RV [m/s]",
                traces=traces,
                param_deps=model_deps,
                meta={
                    "phase_folded": False,
                    "file_tag": "RV_unphased",
                    "figsize": (12, 6),
                    "dynamic_data": True,
                    "caption": (
                        "Radial velocities with the best-fit model "
                        "(red); posterior draws are overplotted with "
                        "low opacity." + self.detrend_caption()
                    ),
                },
            )
        )

        # ---- Phased: one chart per member orbit (needs a model) -------
        omap = getattr(self, "_plot_orbit_map", None)
        if point is not None and omap is not None:
            deps = self._model_trace_param_deps(
                getattr(self, "_rv_matrix_node", None), system
            )
            # The phased DATA moves with the point too: the fold uses tc/P,
            # and the cleaning subtracts gamma + the fitted detrend model +
            # the other orbits' signal (all applied in numpy) -- hence
            # dynamic_data below and the explicit deps the graph walk cannot
            # see.
            deps = deps + [lbl for lbl in numpy_deps if lbl not in deps]
            # Once per (instrument, point), not once per orbit (6.5.1).
            shared = self._phased_shared(system, point)
            for col, o_idx in enumerate(omap):
                prep = self._phased_arrays(
                    system, point, col, o_idx, shared=shared
                )
                P_ref, tc_ref = prep["P_ref"], prep["tc_ref"]
                otraces = [
                    Trace(
                        name="model",
                        role="model",
                        kind="line",
                        x=prep["phase_model"],
                        y=prep["y_model"],
                        node=getattr(self, "_rv_matrix_node", None),
                    )
                ]
                for i in range(self.n_elements):
                    mask = self.inst_map == i
                    g = self._point_value(point, self.gamma, i)
                    cleaned = (
                        self.rv[mask] - g - prep["other_signals"][mask]
                    ) * factor
                    data_phases = np.mod(
                        (self.time[mask] - tc_ref) / P_ref + 0.25, 1.0
                    )
                    otraces.append(
                        Trace(
                            name=self.names[i],
                            role="data",
                            kind="scatter",
                            x=data_phases,
                            y=cleaned,
                            yerr=self.err[mask] * factor,
                            style=self._data_trace_style(i),
                        )
                    )
                oname = system.orbit.names[o_idx]
                specs.append(
                    PlotSpec(
                        id=f"{self.prefix}.phased.{oname}",
                        component={"yaml_key": self.prefix, "instance": None},
                        title=(
                            f"Phased RV: {oname} "
                            f"({getattr(system, 'name', '')})"
                        ),
                        xlabel=f"Phase (P = {P_ref:.5f} d, Tc at 0.25)",
                        ylabel="Isolated RV [m/s]",
                        traces=otraces,
                        param_deps=deps,
                        meta={
                            "phase_folded": True,
                            "orbit": oname,
                            "period": P_ref,
                            "tc": tc_ref,
                            "file_tag": f"RV_phased_{oname}",
                            "figsize": (10, 6),
                            "caption": (
                                "Radial velocities phase-folded on "
                                "orbit "
                                + latex_escape(oname)
                                + ", with the other orbits' "
                                "contributions removed."
                                + self.detrend_caption()
                            ),
                            "hline_y": 0.0,
                            "dynamic_data": True,
                        },
                    )
                )

        return specs
