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

    def _rv_model(self, system, t, blocks, offset=None):
        """The RV model on times ``t`` -- THE expression, built once.

        ``build_likelihood`` calls this with the data's ``pm.Data`` time
        vector, the files' own row blocks and the per-row ``gamma`` as
        ``offset``; ``compile_plotters`` calls it with a symbolic grid laid
        out in per-file blocks and no offset (the plotted curves are
        gamma-free, the data are drawn gamma-subtracted).  So the plotted
        curve is literally the likelihood code run on other times, and the
        Rossiter-McLaughlin anomaly lands only on the rows of a file that
        asked for it -- by INDEX, exactly as in the likelihood -- rather
        than on a whole matrix column for every instrument (review 1.5.5).
        The in-repo template is ``AstrometryInstrument._rel_model``.

        ``t``      -- ``(N,)`` time tensor.
        ``blocks`` -- ``[(inst_idx, rows), ...]``: the concrete row indices
                      of ``t`` that belong to each file (a file may appear
                      more than once; a file with no rows may be omitted).
        ``offset`` -- optional ``(N,)`` tensor added first (the likelihood's
                      ``gamma[inst_map]``), or None.

        Returns ``(rv_model, rv_matrix)``, both in the internal RV unit:

        ``rv_model`` ``(N,)`` -- ``offset`` + the sum over every orbit
            containing the observed star of ``K * phase`` + each RM file's
            anomaly on that file's own rows.  WITHOUT the detrend term,
            which is per observation and is added by ``build_likelihood``
            alone.
        ``rv_matrix`` ``(N, n_member_orbits)`` -- the per-orbit columns of
            the same model (``self._plot_orbit_map`` names the orbit of
            each column), an RM file's anomaly sitting in ITS orbit's
            column on ITS rows only, so ``sum(rv_matrix, axis=1)`` is
            ``rv_model - offset`` and the phased panels' "other orbits"
            cleaning subtracts the RM from a non-RM instrument's points
            never, and from the RM instrument's points exactly when the
            panel is for another orbit.

        The likelihood's numerics are pinned bit-identical on every shipped
        example, so the op sequence in here -- offset first, then the
        Keplerian sum, then the RM rows in file order -- is the op sequence
        the likelihood always had.
        """
        orbits = system.orbit
        K_vec, omap = self._orbit_rv_terms(system, self.star_ndx[0])
        self._plot_orbit_map = omap

        # sum the contribution from every orbit containing the observed star
        kep = orbits.get_radial_velocity(t, K_vec, omap)  # (N, n_member)
        rv_model = pt.sum(kep, axis=1)
        if offset is not None:
            rv_model = offset + rv_model
        rv_matrix = kep

        # Rossiter-McLaughlin in-transit distortion. No-op unless a file set
        # `rm: <orbit_name>` -> the RV model above is unchanged byte for
        # byte (mirrors the GP opt-in). compute_rm_rv returns m/s; convert to
        # the internal RV unit (solRad/d) and add only to that file's rows.
        #
        # INDEX, do not pt.switch. A switch over the branch VALUES would
        # evaluate the Hirano kernel at every instrument's timestamps and then
        # throw away the rows it does not apply to -- the JAX where-trap
        # (the CLAUDE.md invariant): a `where` whose unselected branch can be
        # invalid poisons the gradient of the selected one too. Slicing the
        # RM instrument's own rows makes the unselected rows unreachable by
        # construction instead of merely masked, and is cheaper by exactly
        # the fraction of the data that is not the RM file (the H2011 kernel
        # is a 201 x 64 quadrature PER ROW; on a 40-of-73-row example it was
        # 83% wasted work).
        if any(self.rm_orbit):
            from ..rm import compute_rm_rv, resolve_rm_indices

            rv_ms_per_internal = float((u.solRad / u.d).to(u.m / u.s))
            omap_list = list(omap)
            for inst_idx, rows in blocks:
                oname = self.rm_orbit[inst_idx]
                if not oname:
                    continue
                # A concrete index array, not a slice: `t` may be a pm.Data
                # whose length pytensor treats as symbolic, so a slice's
                # shape is symbolic too and the JAX backend cannot trace the
                # RM subtensor ("Shapes must be 1D sequences of concrete
                # values of integer type").  An advanced index of constants
                # has a statically known length and traces fine.
                rows = np.asarray(rows, dtype=int)
                if rows.size == 0:
                    continue
                oidx, pidx, bidx = resolve_rm_indices(
                    system, oname, self.rm_band[inst_idx]
                )
                if oidx not in omap_list:
                    # The anomaly is the transited star's own line
                    # distortion; an RM orbit this instrument's star is not
                    # a body of has no column to live in and no physics to
                    # justify.  The old plot path dropped it silently.
                    raise ValueError(
                        f"[{self.prefix}.{self.names[inst_idx]}] rm: "
                        f"{oname!r} names an orbit that does not contain "
                        f"the observed star (star_ndx={self.star_ndx[inst_idx]}"
                        f"); the RM anomaly can only be modeled on an orbit "
                        f"the star is a body of."
                    )
                rm_ms = compute_rm_rv(
                    system,
                    t[rows],
                    oidx,
                    pidx,
                    bidx,
                    model=self.rm_model[inst_idx],
                    light_travel_time_active=self.light_travel_time[inst_idx],
                )  # (len(rows),) m/s
                rm_internal = rm_ms / rv_ms_per_internal
                rv_model = pt.inc_subtensor(rv_model[rows], rm_internal)
                col = omap_list.index(oidx)
                rv_matrix = pt.inc_subtensor(rv_matrix[rows, col], rm_internal)

        return rv_model, rv_matrix

    def _data_blocks(self):
        """The data's own row blocks, ``[(i, rows_i), ...]`` in file order
        (``Instrument.rows``, materialized -- see ``_rv_model``)."""
        blocks = []
        for i in range(self.n_elements):
            sl = self.rows(i)
            blocks.append((i, np.arange(sl.start, sl.stop)))
        return blocks

    def build_likelihood(self, model, system):
        time = pm.Data("rv_time", self.time)
        rv = pm.Data("rv_data", self.rv)
        err = pm.Data("rv_err", self.err)

        if len(set(self.star_ndx)) > 1:
            raise NotImplementedError(
                f"[{self.prefix}] all RV instruments must observe the same "
                f"star for now (got star_ndx={self.star_ndx})."
            )

        # 1. The one RV expression (see _rv_model): the gamma constant
        # offset first, then the Keplerian sum, then each RM file's rows.
        # Retained as plain attributes (not Deterministics -- (N_obs,) per
        # draw per chain); compile_plotters compiles the matrix as the
        # plotted model AT the observations.
        rv_model, rv_matrix = self._rv_model(
            system,
            time,
            self._data_blocks(),
            offset=self.gamma.value[self.inst_map_tensor],
        )
        self._rv_model_data_node = rv_model
        self._rv_matrix_data_node = rv_matrix

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

    # Points per block on a plotted model grid; one size for the unphased
    # span and the phased period window so both share one compiled layout.
    _PLOT_GRID_N = 2000

    def _rm_signature(self, i):
        """What makes file ``i``'s RV model differ from another file's:
        its Rossiter-McLaughlin settings (None without ``rm:``)."""
        if not self.rm_orbit[i]:
            return None
        return (
            self.rm_orbit[i],
            self.rm_band[i],
            self.rm_model[i],
            bool(self.light_travel_time[i]),
        )

    def _plot_layout(self):
        """The per-file blocks the plotted model is compiled for.

        Block 0 is the REFERENCE instrument: the first file without
        ``rm:`` -- the plain Keplerian every non-RM file's likelihood
        scores -- or file 0 when every file has RM.  Its curve is the
        chart's "model" trace.  Then one block per instrument whose model
        differs from the reference's (a different RM signature) and per GP
        instrument: those get their own curves, mirroring the GP treatment
        (a per-instrument "model+GP" curve over that instrument's own
        span).  Returns ``(ref, extra)`` with ``extra`` the sorted list of
        instrument indices after block 0.
        """
        ref = next(
            (i for i in range(self.n_elements) if not self.rm_orbit[i]), 0
        )
        ref_sig = self._rm_signature(ref)
        extra = {
            i
            for i in range(self.n_elements)
            if self._rm_signature(i) != ref_sig
        }
        extra.update(int(i) for i in getattr(self, "_gp_pred_on_grid", {}))
        return ref, sorted(extra)

    def compile_plotters(self, model, system):
        """Compile the plotted model -- the likelihood's expression, twice.

        ``_rv_data_fn(*params)`` -> the per-orbit matrix AT the
            observations: literally the node ``build_likelihood`` built
            (``_rv_model`` on the data blocks), compiled against the plot
            parameters.  The phased panels' "other orbits" cleaning reads
            it, so a non-RM instrument's in-transit points no longer have a
            spurious RM bump subtracted (review 1.5.5).
        ``_rv_grid_fn(t, *params)`` -> ``(rv_full, rv_matrix)`` on a plot
            GRID laid out in ``_plot_layout``'s blocks, ``_PLOT_GRID_N``
            rows each.  Every model curve -- unphased span, phased period
            window, per-instrument RM and GP curves -- is this one function
            fed a different ``t``.
        """
        t_input = pt.vector("t_input")
        param_symbols = [p.value for p in system.plot_params]
        orbits = getattr(system, "orbit", None)

        # Per-file GP conditional-mean evaluators (no-op without a gp: key).
        # First, because the plot layout gives every GP instrument a block.
        self._compile_gp_plotters(system)

        if orbits is not None:
            n_grid = self._PLOT_GRID_N
            ref, extra = self._plot_layout()
            self._rv_layout = [ref] + extra
            blocks = [
                (inst_idx, np.arange(k * n_grid, (k + 1) * n_grid))
                for k, inst_idx in enumerate(self._rv_layout)
            ]
            rv_full_node, rv_matrix_node = self._rv_model(
                system, t_input, blocks
            )

            # Retain the symbolic nodes and their time input so plot_data
            # can (a) derive param_deps by walking the graph and (b) hand
            # the GUI the symbolic tensors behind the model traces.  Not
            # needed by the CLI plot() path.
            self._rv_t_input = t_input
            self._rv_matrix_node = rv_matrix_node
            self._rv_full_node = rv_full_node

            # Save them to SELF, not the system!
            self._rv_grid_fn = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=[rv_full_node, rv_matrix_node],
                on_unused_input="ignore",
            )
            data_node = getattr(self, "_rv_matrix_data_node", None)
            self._rv_data_fn = (
                pytensor.function(
                    inputs=param_symbols,
                    outputs=data_node,
                    on_unused_input="ignore",
                )
                if data_node is not None
                else None
            )

    # ------------------------------------------------------------------
    # Shared data preparation. Both the matplotlib plot() path and the
    # GUI plot_data() path go through these helpers, so the two paths
    # always draw the exact same arrays (see chart.Chart).
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
        return np.linspace(
            self.time.min(), self.time.max(), self._PLOT_GRID_N
        ).astype(np.float64)

    def _instrument_grid(self, i):
        """Smooth time grid over instrument ``i``'s own data span."""
        t_i = self.time[self.rows(i)]
        return np.linspace(t_i.min(), t_i.max(), self._PLOT_GRID_N).astype(
            np.float64
        )

    def _eval_rv_grid(self, t_blocks, param_values):
        """Evaluate the plotted model on one grid per layout block.

        ``t_blocks[k]`` is the ``(_PLOT_GRID_N,)`` grid for block ``k`` of
        ``self._rv_layout``.  Returns ``(full, matrix)`` split back per
        block: ``full[k]`` the gamma-free model on that grid, ``matrix[k]``
        its ``(_PLOT_GRID_N, n_member_orbits)`` per-orbit columns.
        Internal units.
        """
        n_grid = self._PLOT_GRID_N
        t_all = np.concatenate(
            [np.asarray(t, dtype=np.float64) for t in t_blocks]
        )
        if t_all.shape[0] != n_grid * len(self._rv_layout):
            raise ValueError(
                f"[{self.prefix}] plot grid layout needs {n_grid} points "
                f"per block, got {[len(t) for t in t_blocks]}."
            )
        full, matrix = self._rv_grid_fn(t_all, *param_values)
        full = np.asarray(full).reshape(len(self._rv_layout), n_grid)
        matrix = np.asarray(matrix).reshape(len(self._rv_layout), n_grid, -1)
        return full, matrix

    def _rv_at_times(self, param_values, i, t):
        """Instrument ``i``'s plotted (gamma-free) model at arbitrary times.

        The compiled grid layout is fixed at ``_PLOT_GRID_N`` rows per
        block, so ``t`` is fed through it in chunks of that size --
        instrument ``i``'s block carrying the chunk (padded with its last
        time), every other block a dummy -- and the results stitched back.
        An instrument that has no block of its own shares the reference's
        model by construction (see ``_plot_layout``) and reads block 0.
        A convenience for a caller that wants the likelihood's model for
        ONE file at times of its own choosing (tests, a GUI probe); the
        panels feed the layout directly.  Returns ``(full, matrix)`` in
        internal units: ``(len(t),)`` and ``(len(t), n_member_orbits)``.
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        n_grid = self._PLOT_GRID_N
        k = self._rv_layout.index(i) if i in self._rv_layout else 0
        fulls, mats = [], []
        for start in range(0, max(t.size, 1), n_grid):
            chunk = t[start : start + n_grid]
            fill = chunk[-1] if chunk.size else float(self.time[0])
            block = np.full(n_grid, fill)
            block[: chunk.size] = chunk
            t_blocks = [np.full(n_grid, fill) for _ in self._rv_layout]
            t_blocks[k] = block
            full, matrix = self._eval_rv_grid(t_blocks, param_values)
            fulls.append(full[k][: chunk.size])
            mats.append(matrix[k][: chunk.size])
        return np.concatenate(fulls), np.concatenate(mats)

    def _unphased_shared(self, system, point):
        """One evaluation of the plot layout for the unphased panel: block
        0 on the full data span, every other block on its own instrument's
        span.  ``_eval_unphased_model``, ``_eval_unphased_rm_models`` and
        ``_eval_unphased_gp_models`` all read from it."""
        param_values = self._point_to_plot_params(point, system)
        t_blocks = [self._unphased_grid()] + [
            self._instrument_grid(i) for i in self._rv_layout[1:]
        ]
        full, _ = self._eval_rv_grid(t_blocks, param_values)
        return {"param_values": param_values, "t": t_blocks, "full": full}

    def _eval_unphased_model(self, system, point, shared=None):
        """Summed RV model on the pretty grid, returned in m/s.

        The reference instrument's physical (gamma-free) model over the
        whole data span (see ``_plot_layout``); the per-instrument RM and
        GP curves are added by the two helpers below.
        """
        if shared is None:
            shared = self._unphased_shared(system, point)
        return shared["t"][0], shared["full"][0] * self._rv_factor()

    def _eval_unphased_rm_models(self, system, point, shared=None):
        """Physical curves for the instruments whose model differs from the
        reference's (their own RM settings), each over that instrument's
        own span.  Returns a list of (instrument index, t, y in m/s); empty
        when every file shares the reference's model -- the single-RM-file
        case then draws exactly what it always did.
        """
        if shared is None:
            shared = self._unphased_shared(system, point)
        ref_sig = self._rm_signature(self._rv_layout[0])
        factor = self._rv_factor()
        out = []
        for k, i in enumerate(self._rv_layout):
            if k == 0 or self._rm_signature(i) == ref_sig:
                continue
            out.append((i, shared["t"][k], shared["full"][k] * factor))
        return out

    def _eval_unphased_gp_models(self, system, point, shared=None):
        """Full (physical + GP) unphased curves, one per GP instrument.

        The GP is a per-instrument noise model, so there is no single "full
        model" curve: each instrument that requested a GP gets its own,
        evaluated only over the span where that instrument actually has data
        (the conditional mean reverts to zero outside it, which would draw a
        misleading flat line across the whole plot).  Its physical part is
        that instrument's OWN block of the layout, so an RM file's GP curve
        carries its RM too.  Returns a list of (instrument index, t, y in
        m/s); empty without any GP.
        """
        if not self.has_gp_plotters():
            return []
        if shared is None:
            shared = self._unphased_shared(system, point)
        factor = self._rv_factor()
        out = []
        for k, i in enumerate(self._rv_layout):
            if k == 0 or i not in self._gp_pred_on_grid:
                continue
            t_i = shared["t"][k]
            y_gp = self.gp_mean_on_grid(system, point, i, t_i)
            out.append((i, t_i, (shared["full"][k] + y_gp) * factor))
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
            # The likelihood's own per-orbit matrix at the observations
            # (see compile_plotters): an RM file's anomaly in its orbit's
            # column on its rows only.
            "data_rv_matrix": np.asarray(self._rv_data_fn(*param_values)),
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

        ``y_model`` is the reference instrument's curve; ``extra_models``
        lists ``(instrument index, y)`` on the same phase grid for every
        instrument whose model differs on THIS orbit (its RM lives in this
        column, or the reference's does), so the RM anomaly is drawn on its
        own instrument's curve and nowhere else.

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
            tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, self._PLOT_GRID_N
        ).astype(np.float64)
        phase_model = np.mod((t_model - tc_ref) / P_ref + 0.25, 1.0)
        sort_m = np.argsort(phase_model)

        _, matrix = self._eval_rv_grid(
            [t_model] * len(self._rv_layout), shared["param_values"]
        )
        y_orbit = matrix[0][:, col]

        oname = system.orbit.names[o_idx]
        ref = self._rv_layout[0]
        ref_sig = self._rm_signature(ref)
        extra_models = []
        for k, i in enumerate(self._rv_layout):
            if k == 0 or self._rm_signature(i) == ref_sig:
                continue
            if oname not in (self.rm_orbit[i], self.rm_orbit[ref]):
                continue  # identical to the reference on this orbit
            extra_models.append((i, matrix[k][:, col][sort_m] * factor))

        other_mask = np.ones(len(self._plot_orbit_map), dtype=bool)
        other_mask[col] = False
        other_signals = np.sum(shared["data_rv_matrix"][:, other_mask], axis=1)

        return {
            "P_ref": P_ref,
            "tc_ref": tc_ref,
            "factor": factor,
            "phase_model": phase_model[sort_m],
            "y_model": y_orbit[sort_m] * factor,
            "extra_models": extra_models,
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
        GUI charts for the RV instrument: one unphased RV-vs-time
        chart plus one phased chart per member orbit. With point=None only
        the observed data traces are returned (raw preview, no model);
        with a point, model curves are added via the shared prep helpers.
        See Component.plot_data and chart.Chart.
        """
        from exozippy.chart import Chart, Trace

        factor = self._rv_factor()
        specs = []

        # ---- Unphased: RV vs time -------------------------------------
        traces = []
        model_deps = []
        if point is not None:
            # One evaluation of the compiled layout serves the reference
            # curve and every per-instrument curve below.
            shared = self._unphased_shared(system, point)
            t_pretty, y_model = self._eval_unphased_model(
                system, point, shared=shared
            )
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
            # One physical curve per instrument whose model differs from the
            # reference's -- its own Rossiter-McLaughlin settings -- over
            # that instrument's own span, so the RM anomaly is drawn on the
            # curve of the file whose likelihood scored it and on no other
            # (review 1.5.5).  Empty when every file shares one model.
            for i, t_rm, y_rm in self._eval_unphased_rm_models(
                system, point, shared=shared
            ):
                traces.append(
                    Trace(
                        name=f"{self.names[i]} model",
                        role="model",
                        kind="line",
                        x=t_rm,
                        y=y_rm,
                        node=getattr(self, "_rv_full_node", None),
                        style={"series_index": int(i), "lw": 1.0},
                    )
                )
            # One physical+GP curve per GP instrument (see
            # _eval_unphased_gp_models). No symbolic node: the GP conditional
            # mean is not part of the model graph, so the GUI re-renders
            # these by asking for a fresh point on a slider move -- which its
            # eval path does, PROVIDED the GP hyperparameters are declared in
            # param_deps (gp_dep_labels, below).  A previous version of this
            # comment said the GUI "cannot re-render these on a slider move";
            # it could all along, and the missing deps were the only blocker
            # (review 1.12.9).
            for i, t_gp, y_gp in self._eval_unphased_gp_models(
                system, point, shared=shared
            ):
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
        # them.  The GP hyperparameters are the same case (the model+GP
        # curves above and the phased cleaning both use the compiled
        # celerite2 conditional mean), see Instrument.gp_dep_labels.
        gamma_label = getattr(getattr(self, "gamma", None), "label", None)
        numpy_deps = (
            ([gamma_label] if gamma_label else [])
            + self.detrend_dep_labels()
            + self.gp_dep_labels()
        )
        if point is not None:
            model_deps = model_deps + [
                lbl for lbl in numpy_deps if lbl not in model_deps
            ]
        specs.append(
            Chart(
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
                # An instrument whose RM sits on this orbit gets its own
                # isolated curve, the anomaly included (see _phased_arrays).
                for i, y_extra in prep["extra_models"]:
                    otraces.append(
                        Trace(
                            name=f"{self.names[i]} model",
                            role="model",
                            kind="line",
                            x=prep["phase_model"],
                            y=y_extra,
                            node=getattr(self, "_rv_matrix_node", None),
                            style={"series_index": int(i), "lw": 1.0},
                        )
                    )
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
                    Chart(
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
