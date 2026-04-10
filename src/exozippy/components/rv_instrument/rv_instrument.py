# generic imports
import numpy as np
import pandas as pd
from pathlib import Path

# plotting imports
import matplotlib.pyplot as plt

# astronomy imports
import astropy.units as u

# pymc imports
import pymc as pm
import pytensor.tensor as pt
import pytensor
from exoplanet_core.pymc import ops as ops

# local imports
from exozippy.components.parameter import Parameter
from exozippy.components.component import Component
# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics

# debugging imports
import ipdb

class RVInstrument(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Instrument Parameters"

        # Metadata only
        self.files = [c.get("file") for c in self.config]
        self.units = [c.get("unit", u.m / u.s) for c in self.config]

        # Initialize "Data Buckets" to empty/zero defaults
        self.total_detrend_cols = 0
        self.n_total_obs = 0
        self.time = np.array([])
        self.rv = np.array([])
        self.err = np.array([])
        self.gamma_init = [0.0] * self.n_elements
        self.jittervar_lower = [0.0] * self.n_elements

    def build_parameters(self, model):
        # all our parameters will be initialized from the data
        pass

    def load_data(self):
        """
        Vectorized loader that handles concatenated data and
        padded block-diagonal detrending matrices.
        """
        all_times, all_rvs, all_errs, inst_indices = [], [], [], []
        all_detrend = []

        for i, file in enumerate(self.files):
            df = pd.read_csv(file, sep=r'\s+', engine='c', header=None, comment='#')

            n_obs = len(df)

            # internally, we use rsun/day to reduce unit conversion overheads
            factor = self.units[i].to(u.solRad / u.d)
            all_times.append(df.iloc[:, 0].values)
            all_rvs.append(df.iloc[:, 1].values*factor)
            all_errs.append(df.iloc[:, 2].values*factor)
            inst_indices.append(np.full(n_obs, i))

            factor = self.units[i].to(u.m / u.s) # make sure it's m/s
            self.gamma_init[i] = np.mean(df.iloc[:, 1].values)*factor
            self.jittervar_lower[i] = -0.95*(np.min(df.iloc[:, 2].values*factor)**2)

            # Capture extra columns (detrending vectors)
            if df.shape[1] > 3:
                all_detrend.append(df.iloc[:, 3:].values.astype(float))
            else:
                # Still need an empty array with the right number of rows
                all_detrend.append(np.empty((n_obs, 0)))

        # 1. Flatten standard vectors
        self.time = np.concatenate(all_times).astype(float)
        self.rv = np.concatenate(all_rvs).astype(float)
        self.err = np.concatenate(all_errs).astype(float)
        self.inst_map = np.concatenate(inst_indices).astype(int)
        self.n_total_obs = len(self.time)

        factor = (u.solRad/u.d).to(u.m/u.s)
        #ipdb.set_trace()
        self.k_init = factor * np.sqrt(2.0) * np.std(self.rv) # m/s

        # 2. Build the Padded Block Diagonal Matrix
        # We need to know the TOTAL number of detrending coefficients
        self.n_detrend_per_inst = [d.shape[1] for d in all_detrend]
        self.total_detrend_cols = sum(self.n_detrend_per_inst)

        # Create the large zero-padded matrix: (N_total_obs x Total_Detrend_Cols)
        self.detrend_matrix = np.zeros((self.n_total_obs, self.total_detrend_cols))

        current_row = 0
        current_col = 0
        for i, d_block in enumerate(all_detrend):
            n_r, n_c = d_block.shape
            if n_c > 0:
                # Place the block in its "slot"
                self.detrend_matrix[current_row:current_row + n_r, current_col:current_col + n_c] = d_block

            current_row += n_r
            current_col += n_c

    def build_map(self, system):
        # Map the observation rows to the instrument parameters (gamma/jitter)
        self.inst_map_tensor = pt.as_tensor_variable(self.inst_map).astype("int32")


    def build_dependent_parameters(self, model, system):
        prefix = "inst"

        # We pass the calculated data-driven values as overrides to the YAML defaults
        parameters = {
            "gamma": {
                "initval": self.gamma_init,
            },
            "jitter_variance": {
                "lower": self.jittervar_lower,
            },
            "jitter": "default"
        }

        # If we have detrending vectors, add those coefficients too
        if self.total_detrend_cols > 0:
            parameters["detrend_coeffs"] = {
                "shape": (self.total_detrend_cols,),
            }

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=prefix)


    def build_likelihood(self, model, system):
        time = pm.ConstantData("rv_time",self.time)
        rv = pm.ConstantData("rv_data",self.rv)
        err = pm.ConstantData("rv_err",self.err)

        orbits = system.orbit
        planets = system.planet

        # 1. Construct the RV Model: start with the gamma constant offset
        rv_model = self.gamma.value[self.inst_map_tensor]

        # sum the contribution from all planets
        rv_model += pt.sum(orbits.get_radial_velocity(time, planets.K.value[planets.orbit_map], planets.orbit_map),axis=1)

        # detrending
        if self.total_detrend_cols > 0:
            detrend = pm.ConstantData("rv_detrend",self.detrend_matrix)
            rv_model += pt.dot(detrend, self.detrend_coeffs.value)

        # 2. Define the Likelihood (The Normal Distribution)
        # Total variance = data_error^2 + jitter^2
        sigma = pt.sqrt(pt.sqr(err) + self.jitter_variance.value[self.inst_map_tensor])

        pm.Normal(
            f"rv_model",
            mu=rv_model,
            sigma=sigma,
            observed=rv
        )

        """
        # GP implementation (this replaces pm.normal above)
        log_sigma_rv_gp = pm.Normal("log_sigma_rv_gp", mu=np.log(2.0), sigma=1.0)
        log_rho_rv_gp = pm.Normal("log_rho_rv_gp", mu=np.log(10.0), sigma=1.0)
        kernel_rv = terms.SHOTerm(sigma=pt.exp(log_sigma_rv_gp), rho=pt.exp(log_rho_rv_gp), Q=1.0/3.0)
        gp_rv = GaussianProcess(
            kernel_rv,
            t=pm.time,
            yerr=sigma,
            mean=rv_model)
        gp_rv.marginal("obs_rv", observed=rv)
        """

    def compile_plotters(self, model, system):
        """Compiles the fast PyTensor functions used by plot_unphased and plot_phased."""
        # 1. We need a time grid input
        t_input = pt.vector("t_input")

        # 2. Get the global symbols to match the MCMC trace signature
        param_symbols = [p.value for p in system.plot_params]

        # 3. Pull the physics from the system
        planets = getattr(system, 'planet', None)
        orbits = getattr(system, 'orbit', None)

        if planets is not None and orbits is not None:
            K_mapped = planets.K.value[planets.orbit_map]

            # The matrix of shape (N_times, N_planets)
            rv_matrix_node = orbits.get_radial_velocity(t_input, K_mapped, planets.orbit_map)

            # Save them to SELF, not the system!
            self._compiled_full_rv = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=pt.sum(rv_matrix_node, axis=1),
                on_unused_input='ignore'
            )

            self._compiled_rv_matrix = pytensor.function(
                inputs=[t_input] + param_symbols,
                outputs=rv_matrix_node,
                on_unused_input='ignore'
            )

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
            print("No points provided for plotting.")
            return

        t_min, t_max = self.time.min(), self.time.max()
        # Create a smooth time grid (64-bit for the C-compiled function)
        t_pretty = np.linspace(t_min, t_max, 2000).astype(np.float64)

        plt.figure(figsize=(12, 6))

        factor = self.gamma._get_conversion_factors()[0]

        # 1. Plot the Model Ensemble (The Spaghetti)
        for idx, point in enumerate(points):
            param_values = []
            for p in system.plot_params:
                # Grab the physical value from the point
                val = np.asarray(point.get(p.label, p.initval), dtype=np.float64)

                # Match the exact dimensions PyTensor expects
                if getattr(p.value, "ndim", 0) == 0:
                    param_values.append(float(np.squeeze(val)))
                else:
                    param_values.append(np.atleast_1d(val))

            try:
                # Evaluate the compiled graph (Summed RV across all planets)
                y_model = self._compiled_full_rv(t_pretty, *param_values)

                # Squeeze to ensure it's (2000,) for matplotlib
                if y_model.ndim > 1:
                    y_model = np.squeeze(y_model)

                # Transparency: Solid for one point, faint for spaghetti
                alpha = 0.8 if len(points) == 1 else 0.1
                plt.plot(t_pretty, y_model*factor, 'r-', lw=1.5, alpha=alpha, zorder=2)
            except Exception as e:
                print(f"Warning: Failed to evaluate model for draw {idx}: {e}")
                continue

        # 2. Plot the Actual Data per Instrument
        # We center the plot by the reference instrument's gamma from the first point
        ref_point = points[0]
        for i in range(self.n_elements):
            mask = (self.inst_map == i)

            # Extract Gamma for this specific instrument
            gamma_vals = np.atleast_1d(ref_point.get(self.gamma.label, 0.0))
            g = gamma_vals[i] if i < len(gamma_vals) else gamma_vals[0]

            plt.errorbar(self.time[mask], (self.rv[mask] - g)*factor,
                         yerr=self.err[mask]*factor, fmt='o', label=self.names[i],
                         alpha=0.6, zorder=1)

        plt.xlabel("Time [BJD]")
        plt.ylabel("Relative RV [m/s]")
        plt.title(f"Unphased RV Model: {system.name}")
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()

        # 3. Save to PDF
        pdf_path = f"{filename_prefix}_RV_unphased.pdf"
        plt.savefig(pdf_path)
        plt.close()

    def plot_phased(self, system, points, filename_prefix="debug"):
        """
        Generates a phased RV plot for each planet in the system.
        Accesses planet parameters via vectorized indices.
        """
        planets = system.planet

        if isinstance(points, dict): points = [points]

        # _compiled_rv_matrix returns (N_times, N_planets)
        compiled_matrix = self._compiled_rv_matrix

        # Iterate through the number of planets defined in the component
        for p_idx in range(planets.n_elements):
            plt.figure(figsize=(10, 6))

            # 1. Setup Phase Grid using the reference point (first draw)
            ref_point = points[0]

            # Accessing the vectorized period and tc for this specific planet index
            # These are stored in the Orbit object owned by the system
            P_vals = np.atleast_1d(ref_point.get(system.orbit.period.label))
            tc_vals = np.atleast_1d(ref_point.get(system.orbit.tc.label))

            P_ref = float(P_vals[p_idx])
            tc_ref = float(tc_vals[p_idx])

            #P_ref = float(np.atleast_1d(ref_point[system.orbits.period.label])[p_idx])
            #tc_ref = float(np.atleast_1d(ref_point[system.orbits.tc.label])[p_idx])

            t_model = np.linspace(tc_ref - 0.5 * P_ref, tc_ref + 0.5 * P_ref, 1000).astype(np.float64)
            # Map time to [0, 1] phase, centering conjunction at 0.25
            phase_model = np.mod((t_model - tc_ref) / P_ref + 0.25, 1.0)
            sort_m = np.argsort(phase_model)

            factor = self.gamma._get_conversion_factors()[0]

            # 2. Plot Model Spaghetti
            for idx, point in enumerate(points):
                param_values = []
                for p in system.plot_params:
                    val = np.asarray(point.get(p.label, p.initval), dtype=np.float64)
                    if getattr(p.value, "ndim", 0) == 0:
                        param_values.append(float(np.squeeze(val)))
                    else:
                        param_values.append(np.atleast_1d(val))

                # Get the (N_times, N_planets) matrix
                rv_matrix = compiled_matrix(t_model, *param_values)
                y_planet = rv_matrix[:, p_idx]

                alpha = 0.8 if len(points) == 1 else 0.1
                plt.plot(phase_model[sort_m], y_planet[sort_m]*factor, 'r-', alpha=alpha, lw=1, zorder=2)

            # 3. Plot Phased Data (Isolating the planet)
            clean_params = []
            for p in system.plot_params:
                val = np.asarray(ref_point.get(p.label, p.initval), dtype=np.float64)
                if getattr(p.value, "ndim", 0) == 0:
                    clean_params.append(float(np.squeeze(val)))
                else:
                    clean_params.append(np.atleast_1d(val))

            # Get signals of ALL planets at observation times
            data_rv_matrix = compiled_matrix(self.time, *clean_params)

            # Mask out the target planet to get the "background" RV
            other_mask = np.ones(planets.n_elements, dtype=bool)
            other_mask[p_idx] = False
            other_signals = np.sum(data_rv_matrix[:, other_mask], axis=1)

            for i in range(self.n_elements):
                mask = (self.inst_map == i)

                # Subtract Gamma and other planet signals
                gamma_vals = np.atleast_1d(ref_point.get(self.gamma.label, 0.0))
                g = gamma_vals[i] if i < len(gamma_vals) else gamma_vals[0]

                cleaned_rv = self.rv[mask] - g - other_signals[mask]

                # Phase the actual data points
                data_phases = np.mod((self.time[mask] - tc_ref) / P_ref + 0.25, 1.0)

                plt.errorbar(data_phases, cleaned_rv*factor, yerr=self.err[mask]*factor,
                             fmt='o', label=self.names[i], alpha=0.6, zorder=1)

            plt.axhline(0, color='black', linestyle=':', alpha=0.5)
            plt.xlabel(f"Phase (P = {P_ref:.5f} d, $T_c$ at 0.25)")
            plt.ylabel("Isolated RV [m/s]")
            plt.title(f"Phased RV: {planets.names[p_idx]} ({system.name})")
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()

            pdf_path = f"{filename_prefix}_RV_phased_{planets.names[p_idx]}.pdf"
            plt.savefig(pdf_path)
            plt.close()