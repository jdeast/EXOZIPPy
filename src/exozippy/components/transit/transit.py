
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
        self.filters = [c.get("filter", "Kepler") for c in self.config]
        self.total_detrend_cols = 0
        self.n_total_obs = 0

    @property
    def prefix(self):
        return "transit"

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
            "q1": "default", "q2": "default", "u1": "default", "u2": "default"
        }

        if self.total_detrend_cols > 0:
            self.manifest["detrend_coeffs"] = {"shape": (self.total_detrend_cols,)}

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

        # 3. Limb Darkening Setup (per observation, mapped from per-instrument values)
        u1_mapped = self.u1.value[self.inst_map_tensor]  # (N_obs,)
        u2_mapped = self.u2.value[self.inst_map_tensor]  # (N_obs,)
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

            u1_inst = self.u1.value[inst_idx]  # scalar for this instrument
            u2_inst = self.u2.value[inst_idx]
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
                decrement_matrix_list.append(-blocked)

            lc_matrix = pt.stack(decrement_matrix_list, axis=1)  # (N_times, N_planets)

            self._compiled_full_lc = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=pt.sum(lc_matrix, axis=1),
                on_unused_input='ignore'
            )
            self._compiled_lc_matrix = pytensor.function(
                inputs=[t_input, inst_idx] + param_symbols,
                outputs=lc_matrix,
                on_unused_input='ignore'
            )

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

            t_pretty = np.linspace(t_data.min(), t_data.max(), 2000).astype(np.float64)

            # Plot spaghetti models
            for point in points:
                param_values = [
                    float(np.squeeze(np.asarray(point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                    else np.atleast_1d(point.get(p.label, p.initval)) for p in system.plot_params]

                try:
                    # _compiled_full_lc returns the flux decrement (<= 0)
                    y_decrement = self._compiled_full_lc(t_pretty, i, *param_values)

                    # Get the baseline for this instrument
                    base_vals = np.atleast_1d(point.get(self.baseline.label, 1.0))
                    baseline = float(base_vals[i])

                    alpha = 0.8 if len(points) == 1 else 0.1
                    plt.plot(t_pretty, baseline + y_decrement, 'r-', lw=1.5, alpha=alpha, zorder=2)
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

                # 1. Spaghetti models
                for point in points:
                    ref_point = point
                    P_ref = float(np.atleast_1d(ref_point.get(system.orbit.period.label))[p_idx])
                    tc_ref = float(np.atleast_1d(ref_point.get(system.orbit.tc.label))[p_idx])

                    # One-period model grid; phase in [-0.5, 0.5] with transit centred at 0
                    t_model = np.linspace(tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000).astype(np.float64)
                    phase_model = ((t_model - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
                    time_from_center_model = phase_model * P_ref
                    sort_m = np.argsort(phase_model)

                    param_values = [
                        float(np.squeeze(np.asarray(point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                        else np.atleast_1d(point.get(p.label, p.initval)) for p in system.plot_params
                    ]
                    try:
                        lc_matrix = self._compiled_lc_matrix(t_model, i, *param_values)
                        y_planet = lc_matrix[:, p_idx]
                        alpha = 0.8 if len(points) == 1 else 0.1
                        plt.plot(time_from_center_model[sort_m], y_planet[sort_m], 'r-', alpha=alpha, lw=1, zorder=2)
                    except Exception as e:
                        logger.warning(f"LC phased model eval failed: {e}")
                        continue

                # 2. Clean and phase the data
                mask = (self.inst_map == i)
                clean_params = [
                    float(np.squeeze(np.asarray(ref_point.get(p.label, p.initval)))) if getattr(p.value, "ndim", 0) == 0
                    else np.atleast_1d(ref_point.get(p.label, p.initval)) for p in system.plot_params
                ]
                try:
                    data_lc_matrix = self._compiled_lc_matrix(self.time[mask], i, *clean_params)
                except Exception as e:
                    logger.warning(f"LC phased data eval failed: {e}")
                    plt.close()
                    continue

                base_vals = np.atleast_1d(ref_point.get(self.baseline.label, 1.0))
                baseline = float(base_vals[i])

                # Subtract baseline and other planets' contributions to isolate p_idx
                other_mask = np.ones(planets.n_elements, dtype=bool)
                other_mask[p_idx] = False
                other_decrements = np.sum(data_lc_matrix[:, other_mask], axis=1)

                cleaned_flux = self.flux[mask] - baseline - other_decrements
                data_phases = ((self.time[mask] - tc_ref) / P_ref + 0.5) % 1.0 - 0.5
                time_from_center_data = data_phases * P_ref

                plt.errorbar(time_from_center_data, cleaned_flux, yerr=self.err[mask],
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