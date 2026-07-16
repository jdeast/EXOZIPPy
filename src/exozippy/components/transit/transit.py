
import logging
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
import pytensor.tensor as pt
from exoplanet_core.pymc import ops as ops
from exozippy.components.component import Component
from . import physics


class Transit(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Transit Parameters"
        self.files = [c.get("file") for c in self.config]
        # Filter identity and limb darkening live on the Band component;
        # each instrument references a band block by name.
        self.band_names = [c.get("band") for c in self.config]
        deprecated_filters = [c.get("filter") for c in self.config if c.get("filter")]
        if deprecated_filters:
            logger.warning(
                "transit 'filter:' is deprecated and ignored; reference a "
                "band: block instead (bands carry the filter identity and "
                "limb darkening)."
            )
        self.total_detrend_cols = 0
        self.n_total_obs = 0
        # SED depth-dilution node, built once by build_likelihood and
        # reused by compile_plotters.
        self._dilution_node = None

    @property
    def prefix(self):
        return "transit"

    def sampler_requirements(self):
        """Declare sampler constraints for limb-darkened transit models.

        The quadratic limb-darkening solution vector (exoplanet_core's
        ``quad_solution_vector`` Op) is only differentiable through
        PyTensor's own gradient machinery (used by the C/numba-backed
        'nuts' and 'nutpie' samplers). The installed exoplanet_core's
        jax_support wires the PyTensor Op straight to the raw,
        non-custom_jvp JAX FFI call, so any sampler that funcifies the
        whole logp graph to JAX ('numpyro', 'blackjax') fails at HMC
        init with "cannot be differentiated".
        """
        return {
            'incompatible': {'numpyro', 'blackjax'},
            'recommended': 'nuts',
            'reason': (
                "the transit component's limb-darkening op "
                "(exoplanet_core quad_solution_vector) is not "
                "differentiable through JAX with the installed "
                "exoplanet_core build — use a PyTensor-backed sampler"
            ),
        }

    def load_data(self, system):
        """Stage 1a: Load CSVs and generate data-driven bounds/inits."""
        all_times, all_fluxes, all_errs, inst_indices, all_detrend = [], [], [], [], []
        self.baseline_init = [1.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')
            n_obs = len(df)
            all_times.append(df.iloc[:, 0].values)
            all_fluxes.append(df.iloc[:, 1].values)
            all_errs.append(df.iloc[:, 2].values)
            inst_indices.append(np.full(n_obs, i))

            self.baseline_init[i] = np.median(df.iloc[:, 1].values)
            self.jittervar_lower[i] = -0.95 * (np.min(df.iloc[:, 2].values) ** 2)

            if df.shape[1] > 3:
                all_detrend.append(df.iloc[:, 3:].values.astype(float))
            else:
                all_detrend.append(np.empty((n_obs, 0)))

        self.time = np.concatenate(all_times).astype(float)
        self.flux = np.concatenate(all_fluxes).astype(float)
        self.err = np.concatenate(all_errs).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)
        self.n_total_obs = len(self.time)

        # Block Diagonal Matrix
        self.n_detrend_per_inst = [d.shape[1] for d in all_detrend]
        self.total_detrend_cols = sum(self.n_detrend_per_inst)
        self.detrend_matrix = np.zeros((self.n_total_obs, self.total_detrend_cols))

        r, c = 0, 0
        for d_block in all_detrend:
            n_r, n_c = d_block.shape
            if n_c > 0: self.detrend_matrix[r:r + n_r, c:c + n_c] = d_block
            r, c = r + n_r, c + n_c

    def register_parameters(self, system):
        """Stage 2: Embed data-driven hints into the PyMC manifest."""
        self.manifest = {
            "baseline": {"initval": self.baseline_init},
            "jitter_variance": {"lower": self.jittervar_lower},
            "jitter": "default",
        }

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = {"shape": (self.total_detrend_cols,)}

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
            (self.names[i], n) for i, n in enumerate(self.band_names)
            if n not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"Transit instrument(s) reference unknown band(s): "
                f"{missing}. Available bands: {list(name_to_idx)}."
            )
        self.band_map = np.array(
            [name_to_idx[n] for n in self.band_names], dtype=int)
        self.obs_band_map = self.band_map[self.inst_map]

    def _build_dilution(self, system):
        """
        Per-instrument SED-predicted depth dilution factor
        F_host / sum_j F_j in the instrument's band, as a (n_elements,)
        tensor (Deterministic "transit.dilution" for diagnostics), or
        None if no instrument's band filter is in the SED's BC grid.
        Instruments whose band filter is unavailable get dilution 1.
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
                dils.append(sed.predict_flux_fraction(host, filter_key, system))
                any_diluted = True
            else:
                logger.warning(
                    f"transit {self.names[i]}: band filter '{filter_key}' "
                    f"is not in the SED's BC grid; no depth deblending "
                    f"applied for this instrument."
                )
                dils.append(pt.constant(1.0))
        if not any_diluted:
            return None
        self._dilution_node = pm.Deterministic(
            f"{self.prefix}.dilution", pt.stack(dils))
        return self._dilution_node

    def build_likelihood(self, model, system):
        time = pm.Data("transit_time", self.time)
        flux = pm.Data("transit_data", self.flux)
        err = pm.Data("transit_err", self.err)

        orbits = system.orbit
        planets = system.planet

        # 1. Start with the photometric baseline
        lc_model = self.baseline.value[self.inst_map_tensor]

        # 1b. Per-planet transit/occultation geometry (impact parameter & durations),
        # exposed as Deterministics for diagnostics and plotting (e.g. phased-plot xlim).
        ecc_p = orbits.ecc.value[planets.orbit_map]  # (N_planets,)
        esinw_p = orbits.esinw.value[planets.orbit_map]
        inc_p = orbits.inc.value[planets.orbit_map]
        period_p = orbits.period.value[planets.orbit_map]
        ar_p = planets.ar.value
        p_p = planets.p.value

        # Numerical-stability floor for the geometry below. Keeps arcsin arguments
        # strictly inside (-1, 1) (where its derivative is finite) and denominators
        # away from 0, so a transient excursion during NUTS leapfrog steps (e.g.
        # inc away from 90 deg, or ecc/esinw near 1) can't produce a NaN/inf
        # gradient. Values at the actual posterior mode are far from these floors,
        # so the reported b/t14/tau are unaffected.
        _GEOM_EPS = 1e-6

        sini_p = pt.sin(inc_p)
        cosi_p = pt.cos(inc_p)
        ecc_factor = pt.sqrt(pt.clip(1.0 - pt.sqr(ecc_p), _GEOM_EPS, 1.0))

        denom_minus = pt.clip(1.0 - esinw_p, _GEOM_EPS, np.inf)
        denom_plus = pt.clip(1.0 + esinw_p, _GEOM_EPS, np.inf)
        sini_ar = pt.clip(pt.abs(sini_p * ar_p), _GEOM_EPS, np.inf)

        dur_b = ar_p * cosi_p * (1.0 - pt.sqr(ecc_p)) / denom_minus
        dur_bs = ar_p * cosi_p * (1.0 - pt.sqr(ecc_p)) / denom_plus

        def _arcsin_term(p_offset_sq, dur_bx):
            radicand = pt.clip(p_offset_sq - pt.sqr(dur_bx), 0.0, np.inf)
            arg = pt.clip(pt.sqrt(radicand) / sini_ar, -1.0 + _GEOM_EPS, 1.0 - _GEOM_EPS)
            return pt.arcsin(arg)

        dur_t14 = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 + p_p), dur_b) * ecc_factor / denom_minus
        dur_t14s = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 + p_p), dur_bs) * ecc_factor / denom_plus

        dur_tfwhm = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 - p_p), dur_b) * ecc_factor / denom_minus
        dur_tfwhms = (period_p / np.pi) * _arcsin_term(pt.sqr(1.0 - p_p), dur_bs) * ecc_factor / denom_plus

        dur_tau = (dur_t14 - dur_tfwhm) / 2.0
        dur_taus = (dur_t14s - dur_tfwhms) / 2.0

        pm.Deterministic(f"{self.prefix}.b", dur_b)
        pm.Deterministic(f"{self.prefix}.bs", dur_bs)
        pm.Deterministic(f"{self.prefix}.t14", dur_t14)
        pm.Deterministic(f"{self.prefix}.t14s", dur_t14s)
        pm.Deterministic(f"{self.prefix}.tfwhm", dur_tfwhm)
        pm.Deterministic(f"{self.prefix}.tfwhms", dur_tfwhms)
        pm.Deterministic(f"{self.prefix}.tau", dur_tau)
        pm.Deterministic(f"{self.prefix}.taus", dur_taus)

        # 2. Orbital Geometry Broadcast
        t_grid = time[:, None]  # (N_obs, 1)
        tp = orbits.tp.value[planets.orbit_map][None, :]  # (1, N_planets)
        n = orbits.n.value[planets.orbit_map][None, :]
        ecc = ecc_p[None, :]
        cosw = orbits.cosw.value[planets.orbit_map][None, :]
        sinw = orbits.sinw.value[planets.orbit_map][None, :]
        inc = inc_p[None, :]

        M = (t_grid - tp) * n
        sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

        a_rstar = planets.ar.value[None, :]
        p_ratio = planets.p.value[None, :]

        r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)

        sin_wf = sinw * cosf + cosw * sinf
        cos_wf = cosw * cosf - sinw * sinf
        sin_i = pt.sin(inc)
        cos_i = pt.cos(inc)

        b = pt.sqrt(pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i))
        Z = r_norm * sin_wf * sin_i

        # 3. Limb Darkening Setup (per observation, mapped from each
        # instrument's Band). When every band uses the linear law, Band's
        # manifest has no u2; the quadratic term is then zero.
        band = system.band
        u1_mapped = band.u1.value[self.obs_band_map_tensor]  # (N_obs,)
        if "u2" in band.manifest:
            u2_mapped = band.u2.value[self.obs_band_map_tensor]  # (N_obs,)
        else:
            u2_mapped = pt.zeros_like(u1_mapped)

        # 3b. SED deblending (EXOFASTv2 parity): with more than one
        # modeled star, only the host contributes the transit, so the
        # observed depth is diluted by dil = F_host / sum_j F_j in the
        # instrument's band (host = the band's star_ndx).
        dil_obs = None
        if hasattr(system, "sed") and system.star.n_elements > 1:
            dil_inst = self._build_dilution(system)
            if dil_inst is not None:
                dil_obs = dil_inst[self.inst_map_tensor]  # (N_obs,)
        # exoplanet_core's quad_solution_vector returns s in starry's Green's basis,
        # not powers of mu. Converting the quadratic law (u1, u2) into that basis
        # requires the change-of-basis in Agol, Luger & Foreman-Mackey (2020),
        # matching exoplanet.light_curves.limb_dark.get_cl():
        #   c0 = 1 - u1 - 1.5*u2, c1 = u1 + 2*u2, c2 = -0.25*u2
        #   norm = dot(s_off, c) = pi*(c0 + c1/1.5), s_off = [pi, 2pi/3, 0]
        c0 = 1.0 - u1_mapped - 1.5 * u2_mapped
        c1 = u1_mapped + 2.0 * u2_mapped
        c2 = -0.25 * u2_mapped
        ld_norm = np.pi * (c0 + c1 / 1.5)

        # 4. Exoplanet-core Transit Model
        for p_idx in range(planets.n_elements):
            b_p = b[:, p_idx]  # (N_obs,) sky-plane separation in units of R_*
            Z_p = Z[:, p_idx]  # (N_obs,) line-of-sight coord (+ = planet in front of star)
            r_p = planets.p.value[p_idx]  # scalar R_p/R_*

            # quad_solution_vector(b, r) -> (N_obs, 3) solution vector s.
            # Broadcast scalar r_p to (N_obs,) following the ops.kepler() pattern.
            sol = ops.quad_solution_vector(b_p, r_p + pt.zeros_like(b_p))

            # Limb-darkened flux fraction: 1.0 off-disk, <1.0 during transit.
            # Verified against brute-force disk integration: lc = dot(s, c) / dot(s_off, c)
            flux_frac = (
                sol[:, 0] * c0 +
                sol[:, 1] * c1 +
                sol[:, 2] * c2
            ) / ld_norm

            # Fraction of stellar flux blocked (0 off-disk, ≈ r² at disk centre)
            blocked = 1.0 - flux_frac

            # Primary transit only; secondary eclipse (planet behind star) has Z < 0
            blocked = pt.where(Z_p > 0.0, blocked, 0.0)
            if dil_obs is not None:
                blocked = blocked * dil_obs
            lc_model = lc_model - blocked

        if self.total_detrend_cols > 0:
            detrend = pm.Data("transit_detrend", self.detrend_matrix)
            lc_model += pt.dot(detrend, self.detrend_coeffs.value)

        # 5. Likelihood
        sigma = pt.sqrt(pt.sqr(err) + self.jitter_variance.value[self.inst_map_tensor])
        pm.Normal("transit_likelihood", mu=lc_model, sigma=sigma, observed=flux)

    def compile_plotters(self, model, system):
        """Compiles the fast PyTensor functions for generating plotting lightcurves."""
        t_input = pt.vector("lc_t_input")
        inst_idx = pt.iscalar("lc_inst_idx")

        param_symbols = [p.value for p in system.plot_params]
        planets = getattr(system, 'planet', None)
        orbits = getattr(system, 'orbit', None)

        if planets is not None and orbits is not None:
            t_grid = t_input[:, None]
            tp = orbits.tp.value[planets.orbit_map][None, :]
            n = orbits.n.value[planets.orbit_map][None, :]
            ecc = orbits.ecc.value[planets.orbit_map][None, :]
            cosw = orbits.cosw.value[planets.orbit_map][None, :]
            sinw = orbits.sinw.value[planets.orbit_map][None, :]
            inc = orbits.inc.value[planets.orbit_map][None, :]

            M = (t_grid - tp) * n
            sinf, cosf = ops.kepler(M, ecc + pt.zeros_like(M))

            a_rstar = planets.ar.value[None, :]
            p_ratio = planets.p.value[None, :]
            r_norm = a_rstar * (1.0 - pt.sqr(ecc)) / (1.0 + ecc * cosf)

            sin_wf = sinw * cosf + cosw * sinf
            cos_wf = cosw * cosf - sinw * sinf
            sin_i = pt.sin(inc)
            cos_i = pt.cos(inc)

            b = pt.sqrt(pt.sqr(r_norm * cos_wf) + pt.sqr(r_norm * sin_wf * cos_i))
            Z = r_norm * sin_wf * sin_i

            band = system.band
            band_idx = self.band_map_tensor[inst_idx]
            u1_inst = band.u1.value[band_idx]  # scalar for this instrument
            if "u2" in band.manifest:
                u2_inst = band.u2.value[band_idx]
            else:
                u2_inst = pt.zeros_like(u1_inst)
            # See build_likelihood for the Green's-basis change-of-basis derivation.
            c0_inst = 1.0 - u1_inst - 1.5 * u2_inst
            c1_inst = u1_inst + 2.0 * u2_inst
            c2_inst = -0.25 * u2_inst
            ld_norm_inst = np.pi * (c0_inst + c1_inst / 1.5)

            decrement_matrix_list = []
            for p_idx in range(planets.n_elements):
                b_p = b[:, p_idx]   # (N_times,)
                Z_p = Z[:, p_idx]
                r_p = planets.p.value[p_idx]

                sol = ops.quad_solution_vector(b_p, r_p + pt.zeros_like(b_p))
                flux_frac = (
                    sol[:, 0] * c0_inst +
                    sol[:, 1] * c1_inst +
                    sol[:, 2] * c2_inst
                ) / ld_norm_inst
                # Negative so that _compiled_full_lc output + baseline gives a transit dip
                blocked = pt.where(Z_p > 0.0, 1.0 - flux_frac, 0.0)
                # match the likelihood's SED depth dilution (built there first)
                dil_node = getattr(self, "_dilution_node", None)
                if dil_node is not None:
                    blocked = blocked * dil_node[inst_idx]
                decrement_matrix_list.append(-blocked)

            lc_matrix = pt.stack(decrement_matrix_list, axis=1)  # (N_times, N_planets)
            lc_full_node = pt.sum(lc_matrix, axis=1)

            # Retain the symbolic nodes and their non-param inputs so
            # plot_data can derive param_deps (graph walk) and hand G5 the
            # symbolic tensors behind the model traces. Unused by plot().
            self._lc_t_input = t_input
            self._lc_inst_idx = inst_idx
            self._lc_matrix_node = lc_matrix
            self._lc_full_node = lc_full_node

            self._compiled_full_lc = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=lc_full_node,
                on_unused_input='ignore'
            )
            self._compiled_lc_matrix = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=lc_matrix,
                on_unused_input='ignore'
            )

    # ------------------------------------------------------------------
    # Shared data preparation. The matplotlib plot() path and the GUI
    # plot_data() path both go through these helpers, so the two paths
    # always draw the exact same arrays (see plotspec.PlotSpec).
    # ------------------------------------------------------------------
    def _baseline_for(self, point, i):
        """Baseline flux for instrument i from a point (default 1.0)."""
        base_vals = np.atleast_1d(point.get(self.baseline.label, 1.0))
        return float(base_vals[i])

    def _eval_unphased_lc(self, system, point, i):
        """Full model light curve (baseline + decrement) for instrument i."""
        mask = (self.inst_map == i)
        t_data = self.time[mask]
        t_pretty = np.linspace(t_data.min(), t_data.max(), 2000).astype(np.float64)
        param_values = self._point_to_plot_params(point, system)
        y_decrement = self._compiled_full_lc(t_pretty, i, *param_values)
        return t_pretty, self._baseline_for(point, i) + y_decrement

    def _phased_lc_arrays(self, system, point, p_idx, i):
        """
        One-period phase grid, isolated model decrement for planet p_idx,
        and the baseline-subtracted, other-planet-cleaned flux at the
        observed times -- shared by plot_phased() and plot_data().
        """
        planets = system.planet
        P_ref = float(np.atleast_1d(point.get(system.orbit.period.label))[p_idx])
        tc_ref = float(np.atleast_1d(point.get(system.orbit.tc.label))[p_idx])

        t_model = np.linspace(tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000).astype(np.float64)
        phase_model = ((t_model - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
        time_from_center_model = phase_model * P_ref
        sort_m = np.argsort(phase_model)

        param_values = self._point_to_plot_params(point, system)
        lc_matrix = self._compiled_lc_matrix(t_model, i, *param_values)
        y_planet = lc_matrix[:, p_idx]

        mask = (self.inst_map == i)
        data_lc_matrix = self._compiled_lc_matrix(self.time[mask], i, *param_values)
        other_mask = np.ones(planets.n_elements, dtype=bool)
        other_mask[p_idx] = False
        other_decrements = np.sum(data_lc_matrix[:, other_mask], axis=1)

        baseline = self._baseline_for(point, i)
        cleaned_flux = self.flux[mask] - baseline - other_decrements
        data_phases = ((self.time[mask] - tc_ref) / P_ref + 0.5) % 1.0 - 0.5

        return {
            "P_ref": P_ref, "tc_ref": tc_ref,
            "x_model": time_from_center_model[sort_m], "y_model": y_planet[sort_m],
            "x_data": data_phases * P_ref, "y_data": cleaned_flux,
        }

    def plot(self, system, points, filename_prefix="debug"):
        self.plot_unphased(system, points, filename_prefix=filename_prefix)
        self.plot_phased(system, points, filename_prefix=filename_prefix)

    def plot_unphased(self, system, points, filename_prefix="debug"):
        if isinstance(points, dict): points = [points]
        if len(points) == 0: return

        for i in range(self.n_elements):
            plt.figure(figsize=(12, 5))
            mask = (self.inst_map == i)
            t_data, f_data, e_data = self.time[mask], self.flux[mask], self.err[mask]

            # Plot spaghetti models
            for point in points:
                try:
                    # Shared prep: full model light curve (baseline + decrement)
                    t_pretty, y_full = self._eval_unphased_lc(system, point, i)

                    alpha = 0.8 if len(points) == 1 else 0.1
                    plt.plot(t_pretty, y_full, 'r-', lw=1.5, alpha=alpha, zorder=2)
                except Exception as e:
                    logger.warning(f"LC model eval failed: {e}")
                    continue

            # Plot raw data
            plt.errorbar(t_data, f_data, yerr=e_data, fmt='k.', alpha=0.5, zorder=1, label=self.names[i])

            plt.xlabel("Time [BJD]")
            plt.ylabel("Relative Flux")
            plt.title(f"Transit Photometry: {self.names[i]}")
            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_LC_unphased_{self.names[i]}.pdf")
            plt.close()

    def plot_phased(self, system, points, filename_prefix="debug"):
        planets = system.planet
        if isinstance(points, dict): points = [points]

        for p_idx in range(planets.n_elements):

            for i in range(self.n_elements):
                plt.figure(figsize=(10, 6))
                ref_point = points[0]

                # 1. Spaghetti models
                prep = None
                for point in points:
                    try:
                        p_prep = self._phased_lc_arrays(system, point, p_idx, i)
                    except Exception as e:
                        logger.warning(f"LC phased model eval failed: {e}")
                        continue
                    if point is ref_point:
                        prep = p_prep
                    alpha = 0.8 if len(points) == 1 else 0.1
                    plt.plot(p_prep["x_model"], p_prep["y_model"], 'r-', alpha=alpha, lw=1, zorder=2)

                # 2. Clean and phase the data (reference point)
                if prep is None:
                    plt.close()
                    continue
                P_ref, tc_ref = prep["P_ref"], prep["tc_ref"]

                plt.errorbar(prep["x_data"], prep["y_data"], yerr=self.err[self.inst_map == i],
                             fmt='k.', alpha=0.5, zorder=1, label=self.names[i])

                t14_raw = ref_point.get(f"{self.prefix}.t14")
                if t14_raw is not None:
                    t14_ref = float(np.atleast_1d(t14_raw)[p_idx])
                    plt.xlim(-t14_ref, t14_ref)

                plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
                plt.xlabel(f"Time from Mid-Transit [d] (P = {P_ref:.5f} d)")
                plt.ylabel("Flux − Baseline")
                plt.title(f"Phased LC: {planets.names[p_idx]} — {self.names[i]}")
                plt.legend(loc='best', fontsize='small')
                plt.tight_layout()
                plt.savefig(f"{filename_prefix}_LC_phased_{self.names[i]}_{planets.names[p_idx]}.pdf")
                plt.close()

    def plot_data(self, system, point=None):
        """
        GUI plot specs for the transit photometry: per instrument an
        unphased flux-vs-time chart, and (with a point) one phased chart
        per planet/instrument. point=None returns only the raw data
        traces. See Component.plot_data and plotspec.PlotSpec.
        """
        from exozippy.plotspec import PlotSpec, Trace

        specs = []
        full_deps = self._model_trace_param_deps(getattr(self, "_lc_full_node", None), system)
        matrix_deps = self._model_trace_param_deps(getattr(self, "_lc_matrix_node", None), system)

        # ---- Unphased: flux vs time, per instrument -------------------
        for i in range(self.n_elements):
            mask = (self.inst_map == i)
            traces = []
            deps = []
            if point is not None:
                t_pretty, y_full = self._eval_unphased_lc(system, point, i)
                deps = full_deps
                traces.append(Trace(name="model", role="model", kind="line",
                                    x=t_pretty, y=y_full,
                                    node=getattr(self, "_lc_full_node", None)))
            traces.append(Trace(
                name=self.names[i], role="data", kind="scatter",
                x=self.time[mask], y=self.flux[mask], yerr=self.err[mask]))
            specs.append(PlotSpec(
                id=f"{self.prefix}.unphased.{self.names[i]}",
                component={"yaml_key": self.prefix, "instance": self.names[i]},
                title=f"Transit Photometry: {self.names[i]}",
                xlabel="Time [BJD]", ylabel="Relative Flux",
                traces=traces, param_deps=deps,
                meta={"phase_folded": False, "instrument": self.names[i]}))

        # ---- Phased: one chart per planet/instrument (needs a model) --
        if point is not None:
            planets = system.planet
            for p_idx in range(planets.n_elements):
                for i in range(self.n_elements):
                    prep = self._phased_lc_arrays(system, point, p_idx, i)
                    mask = (self.inst_map == i)
                    traces = [
                        Trace(name="model", role="model", kind="line",
                              x=prep["x_model"], y=prep["y_model"],
                              node=getattr(self, "_lc_matrix_node", None)),
                        Trace(name=self.names[i], role="data", kind="scatter",
                              x=prep["x_data"], y=prep["y_data"], yerr=self.err[mask]),
                    ]
                    pname = planets.names[p_idx]
                    specs.append(PlotSpec(
                        id=f"{self.prefix}.phased.{self.names[i]}.{pname}",
                        component={"yaml_key": self.prefix, "instance": self.names[i]},
                        title=f"Phased LC: {pname} -- {self.names[i]}",
                        xlabel=f"Time from Mid-Transit [d] (P = {prep['P_ref']:.5f} d)",
                        ylabel="Flux - Baseline",
                        traces=traces, param_deps=matrix_deps,
                        meta={"phase_folded": True, "planet": pname,
                              "instrument": self.names[i],
                              "period": prep["P_ref"], "tc": prep["tc_ref"]}))

        return specs