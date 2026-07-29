import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import astropy.units as u
import matplotlib.pyplot as plt
import pymc as pm
import pytensor
import pytensor.tensor as pt

from exozippy.components.instrument import Instrument

from . import physics


class RVInstrument(Instrument):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Instrument Parameters"
        self.units = [c.get("unit", u.m / u.s) for c in self.config]
        # Which star the RVs are of; its Doppler signal is the sum over
        # every orbit that star is a body of (planetary reflex and stellar
        # companions alike).
        self.star_ndx = [int(c.get("star_ndx", 0)) for c in self.config]
        self.total_detrend_cols = 0

    @property
    def prefix(self):
        return "rvinstrument"

    @classmethod
    def get_utilities(cls):
        from ...utilities.registry import UtilitySpec

        return [
            UtilitySpec(
                name="lomb_scargle",
                label="Lomb-Scargle periodogram",
                description=(
                    "Lomb-Scargle radial-velocity periodogram (not yet "
                    "implemented)."
                ),
                component_keys=["rvinstrument"],
                available=False,
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
                    "Index or name of the observed star (default 0). The RV "
                    "model sums orbit.K over every orbit containing this star."
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
            cls._plot_style_config_schema(),
            cls._gp_config_schema(),
        ]

    def load_data(self, system):
        """Stage 1a: Load CSVs and generate data-driven bounds/inits."""
        all_times, all_rvs, all_errs, inst_indices, all_detrend = (
            [],
            [],
            [],
            [],
            [],
        )
        self.gamma_init = [0.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements

        for i, file in enumerate(self.files):
            df = pd.read_csv(
                file, sep=r"\s+", engine="c", header=None, comment="#"
            )
            # One sort per file, before anything is derived from it: keeps the
            # RVs, errors and detrend columns aligned by construction.
            df = self._sort_by_time(df)
            n_obs = len(df)
            factor = self.units[i].to(u.solRad / u.d)
            all_times.append(df.iloc[:, 0].values)
            all_rvs.append(df.iloc[:, 1].values * factor)
            all_errs.append(df.iloc[:, 2].values * factor)
            inst_indices.append(np.full(n_obs, i))

            m_s_factor = self.units[i].to(u.m / u.s)
            self.gamma_init[i] = np.mean(df.iloc[:, 1].values) * m_s_factor
            self.jittervar_lower[i] = self._jitter_floor(
                df.iloc[:, 2].values, factor=m_s_factor
            )

            if df.shape[1] > 3:
                all_detrend.append(df.iloc[:, 3:].values.astype(float))
            else:
                all_detrend.append(np.empty((n_obs, 0)))

        self.time = np.concatenate(all_times).astype(float)
        self.rv = np.concatenate(all_rvs).astype(float)
        self.err = np.concatenate(all_errs).astype(float)

        # By naming this `inst_map`, the base class auto-generates `inst_map_tensor`
        self.inst_map = np.concatenate(inst_indices).astype(int)

        self.n_total_obs = len(self.time)
        self.k_init = (
            ((u.solRad / u.d).to(u.m / u.s)) * np.sqrt(2.0) * np.std(self.rv)
        )

        # Block Diagonal Matrix (shared builder keeps coeffs per-instrument)
        (
            self.detrend_matrix,
            self.n_detrend_per_inst,
            self.total_detrend_cols,
        ) = self._build_block_detrend(all_detrend, self.n_total_obs)

        # Optional per-file Gaussian process (no-op unless a file sets `gp:`).
        # self.err is in solRad/d; the amplitude parameter is declared in m/s.
        self._prepare_gp(
            self.time,
            self.err,
            self.inst_map,
            user_factor=(u.solRad / u.d).to(u.m / u.s),
        )

    def register_parameters(self, system):
        """Stage 2: Embed data-driven hints into the PyMC manifest."""
        gamma_arr = np.atleast_1d(self.gamma_init)
        for i in range(self.n_elements):
            val = (
                gamma_arr[i].item()
                if hasattr(gamma_arr[i], "item")
                else float(gamma_arr[i])
            )
            self.config_manager.add_hint(f"{self.prefix}.{i}.gamma", val)

        self.manifest = {"gamma": "default"}
        self._register_noise(self.manifest, self.jittervar_lower)
        self._register_gp(self.manifest)

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
            f"{self.prefix}.model", mu=rv_model, sigma=sigma, observed=rv
        )

    def compile_plotters(self, model, system):
        """Compiles the fast PyTensor functions used by plot_unphased and plot_phased."""
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
        if gamma is not None and hasattr(gamma, "_get_conversion_factors"):
            return gamma._get_conversion_factors()[0]
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

    def _instrument_gamma(self, point, i):
        """The reference-point gamma for instrument i, in internal units."""
        gamma_vals = np.atleast_1d(point.get(self.gamma.label, 0.0))
        return gamma_vals[i] if i < len(gamma_vals) else gamma_vals[0]

    def _phased_arrays(self, system, point, col, o_idx):
        """
        Phase grid, isolated model curve, and the per-observation
        background (all other member orbits' signal) for one member
        orbit -- shared by plot_phased() and plot_data().
        """
        factor = self._rv_factor()
        P_ref = float(
            np.atleast_1d(point.get(system.orbit.period.label))[o_idx]
        )
        tc_ref = float(np.atleast_1d(point.get(system.orbit.tc.label))[o_idx])

        t_model = np.linspace(
            tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000
        ).astype(np.float64)
        phase_model = np.mod((t_model - tc_ref) / P_ref + 0.25, 1.0)
        sort_m = np.argsort(phase_model)

        param_values = self._point_to_plot_params(point, system)
        rv_matrix = self._compiled_rv_matrix(t_model, *param_values)
        y_orbit = rv_matrix[:, col]

        data_rv_matrix = self._compiled_rv_matrix(self.time, *param_values)
        other_mask = np.ones(len(self._plot_orbit_map), dtype=bool)
        other_mask[col] = False
        other_signals = np.sum(data_rv_matrix[:, other_mask], axis=1)

        # Phasing data that still contains the correlated (e.g. rotation)
        # signal just smears the panel, so the GP conditional mean is removed
        # from the data here along with the other orbits' signal. Zeros for
        # instruments without a GP, so this is a no-op then.
        gp_signals = self.gp_mean_at_data(system, point)

        return {
            "P_ref": P_ref,
            "tc_ref": tc_ref,
            "factor": factor,
            "phase_model": phase_model[sort_m],
            "y_model": y_orbit[sort_m] * factor,
            "other_signals": other_signals + gp_signals,
        }

    def plot(self, system, points, filename_prefix="debug"):
        self.plot_unphased(system, points, filename_prefix=filename_prefix)
        self.plot_phased(system, points, filename_prefix=filename_prefix)

    def plot_unphased(self, system, points, filename_prefix="debug"):
        """
        Generates a non-phased RV plot (spaghetti or single model).
        Saves to {filename_prefix}_RV_unphased.pdf
        """

        if isinstance(points, dict):
            points = [points]
        if len(points) == 0:
            logger.warning("No points provided for plotting.")
            return

        plt.figure(figsize=(12, 6))

        factor = self._rv_factor()

        # 1. Plot the Model Ensemble (The Spaghetti)
        for idx, point in enumerate(points):
            try:
                # Shared prep: summed RV across all orbits, already in m/s
                t_pretty, y_model = self._eval_unphased_model(system, point)

                # Transparency: Solid for one point, faint for spaghetti
                alpha = 0.8 if len(points) == 1 else 0.1
                plt.plot(
                    t_pretty, y_model, "r-", lw=1.5, alpha=alpha, zorder=2
                )

                # One "physical + GP" curve per GP instrument, drawn per draw
                # so the spaghetti shows the GP's own uncertainty too.
                for i, t_gp, y_gp in self._eval_unphased_gp_models(
                    system, point
                ):
                    plt.plot(
                        t_gp,
                        y_gp,
                        "-",
                        lw=1.0,
                        alpha=alpha,
                        zorder=3,
                        color=f"C{i}",
                    )
            except Exception as e:
                logger.warning(f"Failed to evaluate model for draw {idx}: {e}")
                continue

        # 2. Plot the Actual Data per Instrument
        # We center the plot by the reference instrument's gamma from the first point
        ref_point = points[0]
        for i in range(self.n_elements):
            mask = self.inst_map == i

            # Extract Gamma for this specific instrument
            g = self._instrument_gamma(ref_point, i)

            plt.errorbar(
                self.time[mask],
                (self.rv[mask] - g) * factor,
                yerr=self.err[mask] * factor,
                fmt="o",
                label=self.names[i],
                alpha=0.6,
                zorder=1,
            )

        plt.xlabel("Time [BJD]")
        plt.ylabel("Relative RV [m/s]")
        plt.title(f"Unphased RV Model: {system.name}")
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()

        # 3. Save to PDF
        pdf_path = f"{filename_prefix}_RV_unphased.pdf"
        plt.savefig(pdf_path)
        plt.close()

    def plot_phased(self, system, points, filename_prefix="debug"):
        """
        Generates a phased RV plot for each orbit the observed star is a
        body of, isolating that orbit's signal from the rest.
        """
        if isinstance(points, dict):
            points = [points]

        omap = self._plot_orbit_map

        # Iterate over the member orbits (columns of the compiled matrix)
        for col, o_idx in enumerate(omap):
            plt.figure(figsize=(10, 6))

            # 1. Setup Phase Grid using the reference point (first draw)
            ref_point = points[0]

            # Shared prep (phase grid, isolated model, per-obs background)
            prep = self._phased_arrays(system, ref_point, col, o_idx)
            P_ref, tc_ref = prep["P_ref"], prep["tc_ref"]
            factor = prep["factor"]

            # 2. Plot Model Spaghetti
            for idx, point in enumerate(points):
                p_prep = (
                    prep
                    if point is ref_point
                    else self._phased_arrays(system, point, col, o_idx)
                )
                alpha = 0.8 if len(points) == 1 else 0.1
                plt.plot(
                    p_prep["phase_model"],
                    p_prep["y_model"],
                    "r-",
                    alpha=alpha,
                    lw=1,
                    zorder=2,
                )

            # 3. Plot Phased Data (Isolating the planet)
            other_signals = prep["other_signals"]

            for i in range(self.n_elements):
                mask = self.inst_map == i

                # Subtract Gamma and other planet signals
                g = self._instrument_gamma(ref_point, i)
                cleaned_rv = self.rv[mask] - g - other_signals[mask]

                # Phase the actual data points
                data_phases = np.mod(
                    (self.time[mask] - tc_ref) / P_ref + 0.25, 1.0
                )

                plt.errorbar(
                    data_phases,
                    cleaned_rv * factor,
                    yerr=self.err[mask] * factor,
                    fmt="o",
                    label=self.names[i],
                    alpha=0.6,
                    zorder=1,
                )

            plt.axhline(0, color="black", linestyle=":", alpha=0.5)
            plt.xlabel(f"Phase (P = {P_ref:.5f} d, $T_c$ at 0.25)")
            plt.ylabel("Isolated RV [m/s]")
            plt.title(
                f"Phased RV: {system.orbit.names[o_idx]} ({system.name})"
            )
            plt.legend(loc="best", fontsize="small")
            plt.tight_layout()

            pdf_path = (
                f"{filename_prefix}_RV_phased_{system.orbit.names[o_idx]}.pdf"
            )
            plt.savefig(pdf_path)
            plt.close()

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
                        style={"series_index": int(i)},
                    )
                )
        for i in range(self.n_elements):
            mask = self.inst_map == i
            # gamma offset only when a point supplies it; raw data otherwise
            g = self._instrument_gamma(point, i) if point is not None else 0.0
            traces.append(
                Trace(
                    name=self.names[i],
                    role="data",
                    kind="scatter",
                    x=self.time[mask],
                    y=(self.rv[mask] - g) * factor,
                    yerr=self.err[mask] * factor,
                    style=self._data_trace_style(i),
                )
            )
        specs.append(
            PlotSpec(
                id=f"{self.prefix}.unphased",
                component={"yaml_key": self.prefix, "instance": None},
                title=f"Unphased RV: {getattr(system, 'name', '')}",
                xlabel="Time [BJD]",
                ylabel="Relative RV [m/s]",
                traces=traces,
                param_deps=model_deps,
                meta={"phase_folded": False},
            )
        )

        # ---- Phased: one chart per member orbit (needs a model) -------
        omap = getattr(self, "_plot_orbit_map", None)
        if point is not None and omap is not None:
            deps = self._model_trace_param_deps(
                getattr(self, "_rv_matrix_node", None), system
            )
            for col, o_idx in enumerate(omap):
                prep = self._phased_arrays(system, point, col, o_idx)
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
                    g = self._instrument_gamma(point, i)
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
                        title=f"Phased RV: {oname}",
                        xlabel=f"Phase (P = {P_ref:.5f} d, Tc at 0.25)",
                        ylabel="Isolated RV [m/s]",
                        traces=otraces,
                        param_deps=deps,
                        meta={
                            "phase_folded": True,
                            "orbit": oname,
                            "period": P_ref,
                            "tc": tc_ref,
                        },
                    )
                )

        return specs
