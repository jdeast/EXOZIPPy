# note, this is an untested, AI generated placeholder

import logging
import numpy as np
import pandas as pd
import pymc as pm

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

        # 2. Orbital Geometry Broadcast
        t_grid = time[:, None]  # (N_obs, 1)
        tp = orbits.tp.value[system.orbit_map][None, :]  # (1, N_planets)
        n = orbits.n.value[system.orbit_map][None, :]
        ecc = orbits.ecc.value[system.orbit_map][None, :]
        cosw = orbits.cosw.value[system.orbit_map][None, :]
        sinw = orbits.sinw.value[system.orbit_map][None, :]
        inc = orbits.inc.value[system.orbit_map][None, :]

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

        # 3. Limb Darkening Setup
        u1_mapped = self.u1.value[self.inst_map_tensor]
        u2_mapped = self.u2.value[self.inst_map_tensor]
        u_stack = pt.stack([u1_mapped, u2_mapped], axis=0)  # Shape (2, N_obs)

        # 4. Exoplanet-core Transit Model
        for p_idx in range(planets.n_elements):
            b_p = b[:, p_idx]
            p_ratio_p = p_ratio[:, p_idx]
            Z_p = Z[:, p_idx]

            decrement = ops.quad_limb_dark(u_stack, b_p, p_ratio_p)
            decrement = pt.where(Z_p > 0, decrement, 0.0)  # Hide secondary eclipses
            lc_model += decrement

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
            tp = orbits.tp.value[system.orbit_map][None, :]
            n = orbits.n.value[system.orbit_map][None, :]
            ecc = orbits.ecc.value[system.orbit_map][None, :]
            cosw = orbits.cosw.value[system.orbit_map][None, :]
            sinw = orbits.sinw.value[system.orbit_map][None, :]
            inc = orbits.inc.value[system.orbit_map][None, :]

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

            # Broadcast limb darkening to (2, 1) to match (N_times,) impact parameters
            u1_inst = self.u1.value[inst_idx]
            u2_inst = self.u2.value[inst_idx]
            u_stack = pt.stack([u1_inst, u2_inst], axis=0)[:, None]

            decrement_matrix_list = []
            for p_idx in range(planets.n_elements):
                b_p = b[:, p_idx]
                p_ratio_p = p_ratio[:, p_idx]
                Z_p = Z[:, p_idx]
                decrement = ops.quad_limb_dark(u_stack, b_p, p_ratio_p)
                decrement = pt.where(Z_p > 0, decrement, 0.0)
                decrement_matrix_list.append(decrement)

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
        # Add a call to self.plot_phased(...) here when you are ready

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